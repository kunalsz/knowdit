#!/usr/bin/env bash
#
# KNOWDIT — one-shot setup.
#
# Installs the toolchain and builds everything Knowdit needs:
#   - system build deps (cc, pkg-config, curl, git)
#   - Rust (stable, edition 2024 => rustc >= 1.85) + the `knowdit` CLI (release build)
#   - forge: the customized Knowdit fork (abortfuzz/foundry release), else Foundry
#   - solc via svm (optional; foundry-compilers also auto-downloads it)
#   - optional: docker image used for sandboxed forge runs (Linux)
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
#   --no-verify     Skip the `knowdit --help` smoke check / cargo test
#   --with-tests    Also run `cargo test` (slow; off by default)
#   --debug         Build a debug binary instead of release
#   --skip-system   Skip OS package installation
#   --skip-rust     Skip Rust toolchain + knowdit build
#   --skip-forge    Skip forge installation
#   --skip-solc     Skip solc / svm installation
#   --skip-docker   Skip pulling the forge docker image
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
  PROFILE_UPDATED="$rc"
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
DO_DOCKER=1
DO_PYTHON=1

usage() { sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    --strict)       STRICT=1 ;;
    --no-verify)    DO_VERIFY=0 ;;
    --with-tests)   DO_TESTS=1 ;;
    --debug)        DO_RELEASE=0 ;;
    --skip-system)  DO_SYSTEM=0 ;;
    --skip-rust)    DO_RUST=0 ;;
    --skip-forge)   DO_FORGE=0 ;;
    --skip-solc)    DO_SOLC=0 ;;
    --skip-docker)  DO_DOCKER=0 ;;
    --skip-python)  DO_PYTHON=0 ;;
    -h|--help)      usage ;;
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
for f in Cargo.toml crates/knowdit/Cargo.toml scripts/requirements-router-embeddings.txt; do
  [ -e "$f" ] || { err "not a Knowdit checkout: missing $f"; exit 1; }
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
KNOWDIT_HOME="${KNOWDIT_HOME:-$HOME/.knowdit}"
KNOWDIT_FORGE_DIR="$KNOWDIT_HOME/bin"
mkdir -p "$LOCAL_BIN"
case ":$PATH:" in *":$LOCAL_BIN:"*) ;; *) export PATH="$LOCAL_BIN:$PATH" ;; esac

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
  if have cc && have curl && have git && have pkg-config; then
    ok "build tools, curl, git, pkg-config already present"
  elif [ "$PKG" = "none" ]; then
    warn "no supported package manager found; install a C toolchain, curl, git and pkg-config manually"
  else
    case "$PKG" in
      apt)    pkg_install build-essential curl git ca-certificates pkg-config libssl-dev || soft_fail "system package install failed" ;;
      dnf)    pkg_install gcc gcc-c++ make curl git ca-certificates pkgconf-pkg-config openssl-devel || soft_fail "system package install failed" ;;
      pacman) pkg_install base-devel curl git ca-certificates pkgconf openssl || soft_fail "system package install failed" ;;
      brew)   pkg_install curl git pkg-config || soft_fail "system package install failed" ;;
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
  if have rustup; then rustup toolchain install stable --profile minimal >/dev/null 2>&1 || true; fi
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
    step "Running cargo test"
    cargo test || { err "cargo test failed"; exit 1; }
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
# 5. Docker image for sandboxed forge runs (Linux, default forge backend)
# ---------------------------------------------------------------------------
DOCKER_IMAGE="lazymio/knowdit:foundry"
if [ "$DO_DOCKER" -eq 1 ]; then
  step "Checking docker forge backend ($DOCKER_IMAGE)"
  if [ "$OS" != "linux" ]; then
    info "docker backend is Linux-only; use --forge-bin on $OS"
  elif ! have docker; then
    info "docker not installed; pass --forge-bin <path> to use a local forge instead"
  elif ! docker info >/dev/null 2>&1; then
    warn "docker daemon not reachable (is it running / are you in the docker group?)"
  elif docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    ok "image $DOCKER_IMAGE already pulled"
  else
    info "docker pull $DOCKER_IMAGE"
    if docker pull "$DOCKER_IMAGE" >/dev/null 2>&1; then
      ok "pulled $DOCKER_IMAGE"
    else
      soft_fail "docker pull $DOCKER_IMAGE failed"
    fi
  fi
else
  step "Skipping docker image (--skip-docker)"
fi

# ---------------------------------------------------------------------------
# 6. Python venv with fastembed (local link-router embeddings, optional)
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
# 7. .env template
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
# 8. Verification
# ---------------------------------------------------------------------------
if [ "$DO_VERIFY" -eq 1 ] && [ "$DO_RUST" -eq 1 ]; then
  step "Verifying knowdit binary"
  if "$KNOWDIT_BIN" --help >/dev/null 2>&1; then
    ok "$("$KNOWDIT_BIN" --version 2>/dev/null || echo knowdit) runs"
  else
    err "$KNOWDIT_BIN --help failed"; exit 1
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
cat <<EOF
Next steps:
  1. Configure LLM:    edit .env  (OPENAI_API_KEY / OPENAI_API_URL / OPENAI_API_MODEL)
  2. Load config:      set -a; source .env; set +a      (or: direnv allow)
  3. Train database:   $KNOWDIT_BIN learn c4 --database-url sqlite://knowdit.db?mode=rwc
                       $KNOWDIT_BIN learn projects --database-url ...
  4. Audit a project:  DATABASE_URL=... $KNOWDIT_BIN workflow streamloop -p <project>${FORGE_ARG}

Binaries:
  knowdit : $KNOWDIT_BIN
  forge   : ${KNOWDIT_FORGE:-$(have forge && command -v forge || echo '<not installed; docker backend or --forge-bin needed>')}
  docker  : $(have docker && command -v docker || echo '<not installed>')
  venv    : $([ -d "$VENV_DIR" ] && echo "$VENV_DIR" || echo '<not created>')
EOF
