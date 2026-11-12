#!/usr/bin/env bash
#
# KNOWDIT — one-shot setup (Solidity + Rust / Solana learning & auditing).
#
# Installs the toolchain and builds everything Knowdit needs:
#   - system build deps (cc, clang, cmake, protoc, pkg-config, libssl, libudev, bzip2, curl, git, python3)
#   - Rust (stable, edition 2024 => rustc >= 1.85, plus llvm-tools-preview) + the `knowdit` CLI
#   - forge: the customized Knowdit fork (abortfuzz/foundry release), else Foundry
#   - solc via svm (optional; foundry-compilers also auto-downloads it)
#   - Solana / Rust audit toolchain:
#       * Agave SBF compiler (`cargo-build-sbf`, `solana`) + pre-warmed SBF platform-tools
#       * `cargo-llvm-cov` (Gate 2 Tier-1 Rust line coverage)
#       * Python 3 + git check for `scripts/build_*_dataset.py --languages solidity,rust` & `autobuild_solana.py`
#   - Docker sandboxes (Linux):
#       * pulls `lazymio/knowdit:foundry` (Solidity sandbox)
#       * builds `knowdit:solana` locally from `docker/Dockerfile.solana` (offline LiteSVM + SBF sandbox)
#   - optional: python venv with fastembed (local link-router embeddings)
#   - .env template (existing .env is preserved)
#
# Idempotent: safe to re-run. Nothing is deleted; an existing .env is preserved.
#
# Usage:
#   ./setup.sh [flags]
#
# Flags:
#   --strict        Treat optional-component failures as fatal (default: warn only)
#   --no-verify     Skip the `knowdit --help` smoke checks / cargo test
#   --with-tests    Also run `cargo test --workspace --lib` (off by default)
#   --debug         Build a debug binary instead of release
#   --skip-system   Skip OS package installation
#   --skip-rust     Skip Rust toolchain + knowdit build
#   --skip-forge    Skip forge installation (and foundry docker pull)
#   --skip-solc     Skip solc / svm installation
#   --skip-solana   Skip Solana / Agave SBF toolchain, cargo-llvm-cov, and `knowdit:solana` image build
#   --skip-solana-docker  Skip building the local `knowdit:solana` Docker image (use host cargo + cargo-build-sbf)
#   --skip-docker   Skip pulling `lazymio/knowdit:foundry` and building `knowdit:solana`
#   --skip-python   Skip the python/fastembed venv
#   -h, --help      Show this help
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
  C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'
else
  C_RESET=''; C_BOLD=''; C_DIM=''; C_RED=''; C_GREEN=''; C_YELLOW=''; C_BLUE=''
fi

STEP=0
step()  { STEP=$((STEP + 1)); printf '\n%s[%d] %s%s\n' "$C_BOLD$C_BLUE" "$STEP" "$*" "$C_RESET"; }
info()  { printf '    %s%s%s\n' "$C_DIM" "$*" "$C_RESET"; }
ok()    { printf '    %s✓%s %s\n' "$C_GREEN" "$C_RESET" "$*"; }
warn()  { printf '    %s!%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
err()   { printf '    %s✗%s %s\n' "$C_RED" "$C_RESET" "$*" >&2; }
have()  { command -v "$1" >/dev/null 2>&1; }

# Optional-component failure handler: fatal under --strict, a warning otherwise.
soft_fail() {
  if [ "$STRICT" -eq 1 ]; then err "$*"; exit 1; fi
  warn "$* (continuing; use --strict to make this fatal)"
}

# Append `export PATH="<dir>:$PATH"` to the user's shell profile, once.
profile_add_path() {
  local dir="$1" line rc
  line="export PATH=\"$dir:\$PATH\""
  case "${SHELL:-}" in
    */zsh)  rc="$HOME/.zshrc" ;;
    */bash) rc="$HOME/.bashrc" ;;
    *)      rc="$HOME/.profile" ;;
  esac
  [ -e "$rc" ] || rc="$HOME/.profile"
  if [ -f "$rc" ] && grep -qF "$line" "$rc" 2>/dev/null; then
    return 0
  fi
  printf '\n# Added by Knowdit setup.sh\n%s\n' "$line" >> "$rc" 2>/dev/null || return 1
  return 0
}

# Resolve a GitHub release asset download URL for <repo>, matching <regex> on
# the asset name. Prefers jq, falls back to a plain grep/sed parse.
gh_asset_url() {
  local repo="$1" pat="$2" json url
  json="$(curl -fsSL -m 30 "https://api.github.com/repos/$repo/releases/latest" 2>/dev/null)" || return 1
  if have jq; then
    url="$(printf '%s' "$json" | jq -r --arg p "$pat" \
      '.assets[] | select(.name | test($p)) | .browser_download_url' 2>/dev/null | head -1)"
  else
    url="$(printf '%s' "$json" \
      | grep -o '"browser_download_url":[[:space:]]*"[^"]*"' \
      | sed 's/.*"\(https[^"]*\)"/\1/' | grep -E "$pat" | head -1)"
  fi
  [ -n "$url" ] || return 1
  printf '%s\n' "$url"
}

# Download a release tarball, extract <bin>, install it to <dest_dir>/<out_name>.
install_release_tarball() {
  local repo="$1" pat="$2" bin="$3" dest_dir="$4" out_name="$5" url tmp found
  url="$(gh_asset_url "$repo" "$pat")" || return 1
  tmp="$(mktemp -d)"
  info "downloading $url"
  if ! curl -fsSL -m 300 "$url" -o "$tmp/pkg.tgz" 2>/dev/null; then rm -rf "$tmp"; return 1; fi
  if ! tar -xzf "$tmp/pkg.tgz" -C "$tmp" 2>/dev/null; then rm -rf "$tmp"; return 1; fi
  found="$(find "$tmp" -type f -name "$bin" | head -1)"
  if [ -z "$found" ]; then rm -rf "$tmp"; return 1; fi
  mkdir -p "$dest_dir"
  install -m 0755 "$found" "$dest_dir/$out_name" || { rm -rf "$tmp"; return 1; }
  rm -rf "$tmp"
  return 0
}

# True when `forge` (a path or name on PATH) is really Foundry.
is_foundry_forge() {
  local bin="$1" out
  [ -x "$bin" ] || have "$bin" || return 1
  out="$("$bin" --version 2>&1)" || return 1
  printf '%s' "$out" | grep -qiE 'forge.*[0-9]+\.[0-9]+\.[0-9]+'
}

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
STRICT=0
DO_VERIFY=1
DO_TESTS=0
DO_RELEASE=1
DO_SYSTEM=1
DO_RUST=1
DO_FORGE=1
DO_SOLC=1
DO_SOLANA=1
DO_SOLANA_DOCKER=1
DO_DOCKER=1
DO_PYTHON=1
AGAVE_VERSION="${AGAVE_VERSION:-2.1.15}"

usage() { sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    --strict)              STRICT=1 ;;
    --no-verify)           DO_VERIFY=0 ;;
    --with-tests)          DO_TESTS=1 ;;
    --debug)               DO_RELEASE=0 ;;
    --skip-system)         DO_SYSTEM=0 ;;
    --skip-rust)           DO_RUST=0 ;;
    --skip-forge)          DO_FORGE=0 ;;
    --skip-solc)           DO_SOLC=0 ;;
    --skip-solana)         DO_SOLANA=0; DO_SOLANA_DOCKER=0 ;;
    --skip-solana-docker)  DO_SOLANA_DOCKER=0 ;;
    --skip-docker)         DO_DOCKER=0 ;;
    --skip-python)         DO_PYTHON=0 ;;
    -h|--help)             usage ;;
    *) err "unknown flag: $1"; usage ;;
  esac
  shift
done

# ---------------------------------------------------------------------------
# Repo root + sanity
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

step "Checking repository layout"
for f in Cargo.toml crates/knowdit/Cargo.toml crates/knowdit-rust/Cargo.toml docker/Dockerfile.solana scripts/autobuild_solana.py scripts/requirements-router-embeddings.txt; do
  [ -e "$f" ] || { err "not a complete Knowdit checkout: missing $f"; exit 1; }
done
ok "Knowdit repo at $REPO_ROOT"

# Cargo.toml patches llmy-tokenizer to ./vendor/llmy-tokenizer; a missing path
# makes every cargo command fail with an opaque manifest error.
if grep -q '^llmy-tokenizer *= *{ *path' Cargo.toml; then
  VENDOR_PATH="$(sed -n 's/^llmy-tokenizer *= *{ *path *= *"\([^"]*\)".*/\1/p' Cargo.toml | head -1)"
  if [ -n "$VENDOR_PATH" ] && [ ! -d "$VENDOR_PATH" ]; then
    err "Cargo.toml patches llmy-tokenizer to $VENDOR_PATH, which does not exist"
    err "restore it (git checkout / submodule) or remove the [patch.crates-io] entry"
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# OS detection
# ---------------------------------------------------------------------------
OS="unknown"
case "$(uname -s)" in
  Linux*)  OS="linux" ;;
  Darwin*) OS="macos" ;;
esac

ARCH="$(uname -m)"
LOCAL_BIN="$HOME/.local/bin"
CARGO_BIN_DIR="${CARGO_HOME:-$HOME/.cargo}/bin"
SOLANA_ACTIVE_BIN="$HOME/.local/share/solana/install/active_release/bin"
KNOWDIT_HOME="${KNOWDIT_HOME:-$HOME/.knowdit}"
KNOWDIT_FORGE_DIR="$KNOWDIT_HOME/bin"
mkdir -p "$LOCAL_BIN"
for pdir in "$LOCAL_BIN" "$CARGO_BIN_DIR" "$SOLANA_ACTIVE_BIN"; do
  if [ -d "$pdir" ]; then
    case ":$PATH:" in *":$pdir:"*) ;; *) export PATH="$pdir:$PATH" ;; esac
  fi
done

PKG="none"
if have apt-get; then PKG="apt"
elif have dnf;     then PKG="dnf"
elif have pacman;  then PKG="pacman"
elif have brew;    then PKG="brew"
fi

SUDO=""
if [ "$OS" = "linux" ] && [ "$(id -u)" -ne 0 ] && have sudo; then SUDO="sudo"; fi

pkg_install() {
  [ $# -eq 0 ] && return 0
  case "$PKG" in
    apt)    $SUDO apt-get update -qq && $SUDO apt-get install -y "$@" ;;
    dnf)    $SUDO dnf install -y "$@" ;;
    pacman) $SUDO pacman -Sy --noconfirm --needed "$@" ;;
    brew)   brew install "$@" ;;
    *)      return 1 ;;
  esac
}

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
if [ "$DO_SYSTEM" -eq 1 ]; then
  step "Installing system build dependencies ($OS / $PKG)"
  if have cc && have curl && have git && have pkg-config && have bzip2 && have python3; then
    ok "build tools, curl, git, pkg-config, bzip2, python3 already present"
    if [ "$DO_SOLANA" -eq 1 ] && { ! have clang || ! have cmake || ! have protoc; }; then
      info "optional host C/Solana headers (clang, cmake, protobuf-compiler, libudev-dev) not all on PATH; Docker sandbox (knowdit:solana) bundles them, or install via $PKG for host-only LiteSVM builds"
    fi
  elif [ "$PKG" = "none" ]; then
    warn "no supported package manager found; install a C toolchain, clang, cmake, protoc, bzip2, python3, curl, git and pkg-config manually"
  else
    case "$PKG" in
      apt)    pkg_install build-essential bzip2 clang cmake curl git ca-certificates libclang-dev libssl-dev libudev-dev llvm pkg-config protobuf-compiler python3 python3-venv || soft_fail "system package install failed" ;;
      dnf)    pkg_install gcc gcc-c++ bzip2 clang clang-devel cmake make curl git ca-certificates llvm openssl-devel pkgconf-pkg-config protobuf-compiler python3 systemd-devel || soft_fail "system package install failed" ;;
      pacman) pkg_install base-devel bzip2 clang cmake curl git ca-certificates llvm openssl pkgconf protobuf python systemd-libs || soft_fail "system package install failed" ;;
      brew)   pkg_install bzip2 cmake curl git llvm pkg-config protobuf python3 || soft_fail "system package install failed" ;;
    esac
    ok "system dependencies present"
  fi
else
  step "Skipping system dependencies (--skip-system)"
fi

# ---------------------------------------------------------------------------
# 2. Rust + knowdit
# ---------------------------------------------------------------------------
PROFILE_DIR="release"
CARGO_FLAGS="--release"
if [ "$DO_RELEASE" -eq 0 ]; then PROFILE_DIR="debug"; CARGO_FLAGS=""; fi
KNOWDIT_BIN="$REPO_ROOT/target/$PROFILE_DIR/knowdit"

if [ "$DO_RUST" -eq 1 ]; then
  step "Setting up Rust (stable)"
  if ! have rustup && ! have cargo; then
    info "installing rustup…"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
      || soft_fail "rustup install failed"
  fi
  if [ -f "$HOME/.cargo/env" ]; then . "$HOME/.cargo/env"; fi
  if have rustup; then
    rustup toolchain install stable --profile minimal >/dev/null 2>&1 || true
    if [ "$DO_SOLANA" -eq 1 ]; then
      rustup component add llvm-tools-preview rustfmt >/dev/null 2>&1 || true
    fi
  fi
  have cargo || { err "cargo not on PATH"; exit 1; }
  ok "cargo $(cargo --version | awk '{print $2}')"

  # The workspace uses edition 2024, which needs rustc >= 1.85.
  RUSTC_VER="$(rustc --version | awk '{print $2}')"
  RUSTC_MINOR="$(printf '%s' "$RUSTC_VER" | cut -d. -f2)"
  if [ "${RUSTC_MINOR:-0}" -lt 85 ]; then
    err "rustc $RUSTC_VER is too old for edition 2024 (need >= 1.85); run: rustup update stable"
    exit 1
  fi
  ok "rustc $RUSTC_VER (edition 2024 ok)"

  step "Building knowdit ($PROFILE_DIR)"
  info "cargo build -p knowdit $CARGO_FLAGS"
  # shellcheck disable=SC2086
  cargo build -p knowdit $CARGO_FLAGS || { err "knowdit build failed"; exit 1; }
  ok "binary: $KNOWDIT_BIN"

  if [ "$DO_VERIFY" -eq 1 ] && [ "$DO_TESTS" -eq 1 ]; then
    step "Running cargo test --workspace --lib"
    cargo test --workspace --lib || { err "cargo test failed"; exit 1; }
    ok "tests passed"
  fi
else
  step "Skipping Rust/knowdit (--skip-rust)"
fi

# ---------------------------------------------------------------------------
# 3. forge (Knowdit currently relies on a customized fork)
# ---------------------------------------------------------------------------
KNOWDIT_FORGE=""
if [ "$DO_FORGE" -eq 1 ]; then
  step "Setting up forge (customized Knowdit fork)"
  if [ -x "$KNOWDIT_FORGE_DIR/forge" ] && is_foundry_forge "$KNOWDIT_FORGE_DIR/forge"; then
    KNOWDIT_FORGE="$KNOWDIT_FORGE_DIR/forge"
    ok "knowdit forge present: $KNOWDIT_FORGE"
  else
    case "$OS:$ARCH" in
      linux:x86_64)                 fpat='linux_amd64\.tar\.gz$' ;;
      linux:aarch64|linux:arm64)    fpat='linux_arm64\.tar\.gz$' ;;
      macos:x86_64)                 fpat='darwin_amd64\.tar\.gz$' ;;
      macos:arm64|macos:aarch64)    fpat='darwin_arm64\.tar\.gz$' ;;
      *)                            fpat='' ;;
    esac
    if [ -n "$fpat" ] && have curl \
       && install_release_tarball "abortfuzz/foundry" "$fpat" forge "$KNOWDIT_FORGE_DIR" forge; then
      KNOWDIT_FORGE="$KNOWDIT_FORGE_DIR/forge"
      ok "installed knowdit forge: $KNOWDIT_FORGE"
    else
      warn "could not fetch abortfuzz/foundry release; falling back to stock Foundry"
    fi
  fi

  if [ -z "$KNOWDIT_FORGE" ]; then
    if have forge && is_foundry_forge "$(command -v forge)"; then
      KNOWDIT_FORGE="$(command -v forge)"
      ok "using stock forge: $KNOWDIT_FORGE (customized fork recommended; not guaranteed)"
    elif have curl; then
      info "installing Foundry via foundryup…"
      curl -L https://foundry.paradigm.xyz | bash >/dev/null 2>&1 || true
      if [ -x "$HOME/.foundry/bin/foundryup" ]; then
        "$HOME/.foundry/bin/foundryup" >/dev/null 2>&1 || true
        if [ -x "$HOME/.foundry/bin/forge" ]; then
          KNOWDIT_FORGE="$HOME/.foundry/bin/forge"
          ok "installed stock forge: $KNOWDIT_FORGE"
        fi
      fi
    fi
    [ -n "$KNOWDIT_FORGE" ] || soft_fail "no forge available; download one from github.com/abortfuzz/foundry/releases"
  fi
else
  step "Skipping forge (--skip-forge)"
fi

# ---------------------------------------------------------------------------
# 4. solc (optional)
# ---------------------------------------------------------------------------
if [ "$DO_SOLC" -eq 1 ]; then
  step "Ensuring solc (via svm)"
  if have solc; then
    ok "solc $(solc --version 2>/dev/null | sed -n 's/.*Version: \([0-9.]*\).*/\1/p' | head -1) at $(command -v solc)"
  else
    if ! have svm && have cargo; then
      info "installing svm (cargo install svm-rs)…"
      cargo install svm-rs --locked >/dev/null 2>&1 || true
      hash -r 2>/dev/null || true
    fi
    if have svm && svm install 0.8.28 >/dev/null 2>&1 && svm use 0.8.28 >/dev/null 2>&1; then
      hash -r 2>/dev/null || true
      ok "solc 0.8.28 installed via svm"
    else
      info "solc not installed; knowdit (foundry-compilers svm-solc) and forge download compilers on demand"
    fi
  fi
else
  step "Skipping solc (--skip-solc)"
fi

# ---------------------------------------------------------------------------
# 5. Rust / Solana Learning & Auditing Toolchain (Agave SBF + cargo-llvm-cov)
# ---------------------------------------------------------------------------
KNOWDIT_SBF=""
KNOWDIT_LLVM_COV=""
if [ "$DO_SOLANA" -eq 1 ]; then
  step "Setting up Rust / Solana learning & LiteSVM audit toolchain"

  # 5a. Verify Python 3 + git for `scripts/build_*_dataset.py --languages solidity,rust` & `autobuild_solana.py`
  if have python3 && have git; then
    ok "python3 + git ready for Rust/Solana KG dataset builders & scripts/autobuild_solana.py"
  else
    soft_fail "python3 and/or git missing (needed by dataset builders and scripts/autobuild_solana.py)"
  fi

  # 5b. Agave / Solana SBF compiler (`cargo-build-sbf`)
  AGAVE_CACHED_TAR=""
  if have cargo-build-sbf; then
    KNOWDIT_SBF="$(command -v cargo-build-sbf)"
    ok "cargo-build-sbf present: $KNOWDIT_SBF"
  elif [ -x "$SOLANA_ACTIVE_BIN/cargo-build-sbf" ]; then
    export PATH="$SOLANA_ACTIVE_BIN:$PATH"
    profile_add_path "$SOLANA_ACTIVE_BIN" || true
    KNOWDIT_SBF="$SOLANA_ACTIVE_BIN/cargo-build-sbf"
    ok "cargo-build-sbf present: $KNOWDIT_SBF"
  elif have curl; then
    case "$OS:$ARCH" in
      linux:x86_64)                 agave_target="x86_64-unknown-linux-gnu" ;;
      linux:aarch64|linux:arm64)    agave_target="aarch64-unknown-linux-gnu" ;;
      macos:x86_64)                 agave_target="x86_64-apple-darwin" ;;
      macos:arm64|macos:aarch64)    agave_target="aarch64-apple-darwin" ;;
      *)                            agave_target="" ;;
    esac
    SOLANA_INSTALL_ROOT="$HOME/.local/share/solana/install"
    SOLANA_REL_DIR="$SOLANA_INSTALL_ROOT/releases/v${AGAVE_VERSION}"
    AGAVE_CACHE_DIR="$KNOWDIT_HOME/cache"
    AGAVE_ARCHIVE="$AGAVE_CACHE_DIR/solana-release-v${AGAVE_VERSION}-${agave_target}.tar.bz2"
    AGAVE_URL="https://github.com/anza-xyz/agave/releases/download/v${AGAVE_VERSION}/solana-release-${agave_target}.tar.bz2"
    if [ -n "$agave_target" ]; then
      info "downloading Agave v${AGAVE_VERSION} SBF release tarball ($agave_target, resumable)…"
      mkdir -p "$SOLANA_REL_DIR" "$AGAVE_CACHE_DIR"
      if curl -fL --retry 5 --connect-timeout 30 -C - "$AGAVE_URL" -o "$AGAVE_ARCHIVE" \
         && tar -xjf "$AGAVE_ARCHIVE" -C "$SOLANA_REL_DIR" \
         && ln -sfn "$SOLANA_REL_DIR/solana-release" "$SOLANA_INSTALL_ROOT/active_release" \
         && [ -x "$SOLANA_ACTIVE_BIN/cargo-build-sbf" ]; then
        AGAVE_CACHED_TAR="$AGAVE_ARCHIVE"
        export PATH="$SOLANA_ACTIVE_BIN:$PATH"
        profile_add_path "$SOLANA_ACTIVE_BIN" || true
        KNOWDIT_SBF="$SOLANA_ACTIVE_BIN/cargo-build-sbf"
        ok "installed Agave SBF toolchain: $KNOWDIT_SBF"
      else
        rm -f "$AGAVE_ARCHIVE"
        soft_fail "Agave SBF download failed (knowdit:solana Docker image can still compile SBF programs)"
      fi
    else
      soft_fail "unsupported OS/arch ($OS:$ARCH) for prebuilt Agave SBF release"
    fi
  fi

  # Pre-warm Agave SBF platform-tools once so offline / cached SBF builds don't stall later.
  if [ -n "$KNOWDIT_SBF" ]; then
    SBF_WARM_TMP="$(mktemp -d)"
    mkdir -p "$SBF_WARM_TMP/src"
    printf '[package]\nname = "knowdit_sbf_warm"\nversion = "0.1.0"\nedition = "2021"\n\n[lib]\ncrate-type = ["cdylib"]\n\n[dependencies]\n' > "$SBF_WARM_TMP/Cargo.toml"
    printf '#![no_std]\n#[panic_handler]\nfn panic(_info: &core::panic::PanicInfo) -> ! { loop {} }\n#[no_mangle]\npub extern "C" fn entrypoint(_input: *mut u8) -> u64 { 0 }\n' > "$SBF_WARM_TMP/src/lib.rs"
    info "verifying SBF platform-tools SDK via cargo-build-sbf…"
    if "$KNOWDIT_SBF" --manifest-path "$SBF_WARM_TMP/Cargo.toml" >/dev/null 2>&1; then
      ok "SBF platform-tools SDK cached and verified"
    else
      warn "SBF platform-tools warm-up did not complete (will retry on first Solana preflight)"
    fi
    rm -rf "$SBF_WARM_TMP"
  fi

  # 5c. `cargo-llvm-cov` (Gate 2 Tier-1 Rust source coverage; Tier-2 SVM trace works even without it)
  if have cargo && cargo llvm-cov --version >/dev/null 2>&1; then
    KNOWDIT_LLVM_COV="cargo llvm-cov"
    ok "cargo-llvm-cov already installed"
  else
    case "$OS:$ARCH" in
      linux:x86_64)                 cov_pat='x86_64-unknown-linux-gnu\.tar\.gz$' ;;
      linux:aarch64|linux:arm64)    cov_pat='aarch64-unknown-linux-gnu\.tar\.gz$' ;;
      macos:x86_64)                 cov_pat='x86_64-apple-darwin\.tar\.gz$' ;;
      macos:arm64|macos:aarch64)    cov_pat='aarch64-apple-darwin\.tar\.gz$' ;;
      *)                            cov_pat='' ;;
    esac
    COV_DEST="$CARGO_BIN_DIR"
    [ -d "$COV_DEST" ] || COV_DEST="$LOCAL_BIN"
    if [ -n "$cov_pat" ] && have curl \
       && install_release_tarball "taiki-e/cargo-llvm-cov" "$cov_pat" cargo-llvm-cov "$COV_DEST" cargo-llvm-cov; then
      hash -r 2>/dev/null || true
      KNOWDIT_LLVM_COV="$COV_DEST/cargo-llvm-cov"
      ok "installed prebuilt cargo-llvm-cov: $KNOWDIT_LLVM_COV"
    elif have cargo; then
      info "installing cargo-llvm-cov via cargo install…"
      if cargo install --locked cargo-llvm-cov >/dev/null 2>&1; then
        KNOWDIT_LLVM_COV="cargo llvm-cov"
        ok "installed cargo-llvm-cov from crates.io"
      else
        info "cargo-llvm-cov skipped (SolanaHarness Tier-2 SVM log coverage remains active)"
      fi
    fi
  fi
else
  step "Skipping Solana / Rust toolchain (--skip-solana)"
fi

# ---------------------------------------------------------------------------
# 6. Docker images for sandboxed runs (Linux: Foundry pull + local Solana build)
# ---------------------------------------------------------------------------
FOUNDRY_DOCKER_IMAGE="lazymio/knowdit:foundry"
SOLANA_DOCKER_IMAGE="knowdit:solana"
if [ "$DO_DOCKER" -eq 1 ]; then
  step "Checking Docker execution sandboxes"
  if [ "$OS" != "linux" ]; then
    info "Docker sandboxes are Linux-only; host binaries (--forge-bin / --solana-cargo-bin) are used on $OS"
  elif ! have docker; then
    info "docker not installed; using host forge / cargo / cargo-build-sbf binaries"
  elif ! docker info >/dev/null 2>&1; then
    warn "docker daemon not reachable (is it running / are you in the docker group?)"
  else
    # 6a. Solidity Foundry image (pulled from Docker Hub)
    if [ "$DO_FORGE" -eq 1 ]; then
      if docker image inspect "$FOUNDRY_DOCKER_IMAGE" >/dev/null 2>&1; then
        ok "Solidity image $FOUNDRY_DOCKER_IMAGE already present"
      else
        info "docker pull $FOUNDRY_DOCKER_IMAGE"
        if docker pull "$FOUNDRY_DOCKER_IMAGE" >/dev/null 2>&1; then
          ok "pulled $FOUNDRY_DOCKER_IMAGE"
        else
          soft_fail "docker pull $FOUNDRY_DOCKER_IMAGE failed"
        fi
      fi
    fi

    # 6b. Solana LiteSVM + SBF image (`knowdit:solana` is built locally from
    #     `docker/Dockerfile.solana`, NOT pulled from Docker Hub)
    if [ "$DO_SOLANA" -eq 1 ] && [ "$DO_SOLANA_DOCKER" -eq 1 ]; then
      if docker image inspect "$SOLANA_DOCKER_IMAGE" >/dev/null 2>&1; then
        ok "Solana image $SOLANA_DOCKER_IMAGE already built locally"
      else
        info "building local Solana sandbox image $SOLANA_DOCKER_IMAGE from docker/Dockerfile.solana…"
        DOCKER_CTX_TMP="$(mktemp -d)"
        cp "$REPO_ROOT/docker/Dockerfile.solana" "$DOCKER_CTX_TMP/Dockerfile.solana"
        if [ -n "${AGAVE_CACHED_TAR:-}" ] && [ -f "$AGAVE_CACHED_TAR" ]; then
          ln "$AGAVE_CACHED_TAR" "$DOCKER_CTX_TMP/solana-release.tar.bz2" 2>/dev/null \
            || cp "$AGAVE_CACHED_TAR" "$DOCKER_CTX_TMP/solana-release.tar.bz2"
        fi
        if docker build -f "$DOCKER_CTX_TMP/Dockerfile.solana" -t "$SOLANA_DOCKER_IMAGE" "$DOCKER_CTX_TMP"; then
          ok "built local Solana sandbox image $SOLANA_DOCKER_IMAGE"
        else
          soft_fail "docker build -t $SOLANA_DOCKER_IMAGE failed (host cargo + cargo-build-sbf fallback remains available)"
        fi
        rm -rf "$DOCKER_CTX_TMP"
      fi
    fi
  fi
else
  step "Skipping Docker images (--skip-docker)"
fi

# ---------------------------------------------------------------------------
# 7. Python venv with fastembed (local link-router embeddings, optional)
# ---------------------------------------------------------------------------
VENV_DIR="$REPO_ROOT/.venv"
if [ "$DO_PYTHON" -eq 1 ]; then
  step "Setting up python venv for local embeddings (fastembed)"
  if ! have python3; then
    info "python3 not found; skipping (only needed for the local embedding backend)"
  elif [ -x "$VENV_DIR/bin/python" ] && "$VENV_DIR/bin/python" -c 'import fastembed' >/dev/null 2>&1; then
    ok "fastembed already installed in $VENV_DIR"
  else
    if python3 -m venv "$VENV_DIR" >/dev/null 2>&1 \
       && "$VENV_DIR/bin/pip" install --quiet -r scripts/requirements-router-embeddings.txt; then
      ok "fastembed installed in $VENV_DIR"
    else
      soft_fail "python venv / fastembed install failed (python3-venv missing?)"
    fi
  fi
else
  step "Skipping python venv (--skip-python)"
fi

# ---------------------------------------------------------------------------
# 8. .env template
# ---------------------------------------------------------------------------
step "Writing environment file (.env)"
if [ ! -f .env ]; then
  {
    echo "# Knowdit configuration — loaded via direnv (.envrc) or: set -a; source .env; set +a"
    echo "#OPENAI_API_KEY="
    echo "#OPENAI_API_URL=https://api.openai.com/v1"
    echo "#OPENAI_API_MODEL=gpt-5.4"
    echo "# Cap total spend in USD"
    echo "#OPENAI_BILLING_CAP=50"
    echo "# Save raw LLM conversations to a sqlite file"
    echo "#LLM_DEBUG=debug-conversation.sqlite3"
    echo "# Historical database (sqlite/mysql, anything sea-orm supports)"
    echo "#DATABASE_URL=sqlite://knowdit.db?mode=rwc"
  } > .env
  ok "created .env (edit it and set OPENAI_API_KEY)"
else
  ok ".env already exists; left untouched"
fi

# ---------------------------------------------------------------------------
# 9. Verification
# ---------------------------------------------------------------------------
if [ "$DO_VERIFY" -eq 1 ] && [ "$DO_RUST" -eq 1 ]; then
  step "Verifying knowdit CLI (Solidity + Rust/Solana subcommands)"
  if "$KNOWDIT_BIN" --help >/dev/null 2>&1 \
     && "$KNOWDIT_BIN" static rust --help >/dev/null 2>&1 \
     && "$KNOWDIT_BIN" workflow learn --help >/dev/null 2>&1 \
     && "$KNOWDIT_BIN" workflow streamloop --help >/dev/null 2>&1; then
    ok "$("$KNOWDIT_BIN" --version 2>/dev/null || echo knowdit) verified (Solidity + Rust/Solana learn & audit ready)"
  else
    err "$KNOWDIT_BIN CLI verification failed"; exit 1
  fi
else
  step "Skipping verification"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
printf '\n%s%sKnowdit setup complete.%s\n\n' "$C_BOLD" "$C_GREEN" "$C_RESET"
FORGE_ARG=""
[ -n "$KNOWDIT_FORGE" ] && FORGE_ARG=" -b $KNOWDIT_FORGE"
SOLANA_IMG_STATE="<not built>"
if have docker && docker image inspect "$SOLANA_DOCKER_IMAGE" >/dev/null 2>&1; then
  SOLANA_IMG_STATE="$SOLANA_DOCKER_IMAGE (ready)"
fi
cat <<EOF
Next steps:
  1. Configure LLM:        edit .env  (OPENAI_API_KEY / OPENAI_API_URL / OPENAI_API_MODEL)
  2. Load config:          set -a; source .env; set +a      (or: direnv allow)
  3. Learn contests (KG):  $KNOWDIT_BIN learn c4 --database-url sqlite://knowdit.db?mode=rwc
                           $KNOWDIT_BIN workflow learn -p name:/path/to/solana-repo --language rust --report report.md --database-url sqlite://knowdit.db?mode=rwc
                           python3 scripts/build_c4_dataset.py --languages solidity,rust ...
  4. Audit Solidity repo:  $KNOWDIT_BIN workflow streamloop -p name:/path/to/repo --database-url sqlite://knowdit.db?mode=rwc -o ./runs/name -d 8 --gen-specs-granularity pair${FORGE_ARG}
  5. Audit Solana repo:    $KNOWDIT_BIN workflow streamloop -p name:/path/to/solana-repo --database-url sqlite://knowdit.db?mode=rwc -o ./runs/name -d 8 --gen-specs-granularity pair

Binaries & Sandboxes:
  knowdit         : $KNOWDIT_BIN
  forge           : ${KNOWDIT_FORGE:-$(have forge && command -v forge || echo '<not installed; docker backend or --forge-bin needed>')}
  cargo-build-sbf : ${KNOWDIT_SBF:-$(have cargo-build-sbf && command -v cargo-build-sbf || echo '<not installed; knowdit:solana docker backend used>')}
  cargo-llvm-cov  : ${KNOWDIT_LLVM_COV:-$(have cargo-llvm-cov && command -v cargo-llvm-cov || echo '<optional; Tier-2 SVM trace active>')}
  solana docker   : $SOLANA_IMG_STATE
  docker          : $(have docker && command -v docker || echo '<not installed>')
  venv            : $([ -d "$VENV_DIR" ] && echo "$VENV_DIR" || echo '<not created>')
EOF
