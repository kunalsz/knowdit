//! Sandboxed Cargo / SBF / `LiteSVM` subprocess runner (`SolanaBackend` + `SolanaRunner`).
//!
//! Mirrors [`crate::harness::forge`] across three isolation modes:
//!   1. [`SolanaBackendKind::LocalDirect`] — host `cargo` with `kill_on_drop(true)`.
//!   2. [`SolanaBackendKind::LocalCgroup`] — host `cargo` inside a per-invocation
//!      cgroup v2 slice (`ScopedCgroup`: `memory.max`, `memory.swap.max = 0`,
//!      atomic `cgroup.kill` on `Drop`).
//!   3. [`SolanaBackendKind::Docker`] — `docker run --rm --network none --user uid:gid`
//!      against `lazymio/knowdit:solana` (offline vendored crate cache) with explicit
//!      `docker kill` on `Drop`.
//!
//! Concurrent link pipelines share a single `<harness_dir>/.knowdit-cargo-target`
//! serialized through [`SOLANA_CARGO_LOCK`] so `litesvm` + `solana-sdk` + `proptest`
//! compile once during `preflight` and each per-spec test compiles incrementally.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use color_eyre::eyre::{Result, WrapErr, eyre};
use knowdit_rust::RustWorkspace;
use tokio::process::Command;
use tokio::sync::Mutex;

#[cfg(target_os = "linux")]
use crate::harness::cgroup::ScopedCgroup;

static PREFLIGHT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Process-global serialization of `cargo` / `cargo-build-sbf` invocations sharing
/// `.knowdit-cargo-target` and `<repo>/target/deploy`.
static SOLANA_CARGO_LOCK: Mutex<()> = Mutex::const_new(());

/// Extra time a caller will wait on [`SOLANA_CARGO_LOCK`] beyond its own timeout
/// before declaring a wedged holder and failing open for remaining links.
const SOLANA_LOCK_GRACE: Duration = Duration::from_secs(60);

/// Source of the helper library written into `<per_spec_dir>/src/lib.rs`
/// (`[lib] name = "knowdit_harness_rt"`). Gives every generated integration test
/// deterministic `KNOWDIT_SOLANA_SUMMARY_JSON:` emission, Anchor 8-byte sighash
/// helpers, and `SBF_OUT_DIR` resolution out-of-the-box.
pub const HARNESS_RT_LIB_RS: &str = r#"//! Auto-generated Knowdit Solana / LiteSVM harness runtime helpers.
use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SOLANA_SUMMARY_PREFIX: &str = "KNOWDIT_SOLANA_SUMMARY_JSON:";
pub const POC_VIOLATION_MARKER: &str = "KNOWDIT_POST_ATTACK_REACHED:";
pub const SETUP_FAILED_MARKER: &str = "KNOWDIT_SETUP_FAILED:";

static GLOBAL_CALLS: AtomicI64 = AtomicI64::new(0);
static GLOBAL_REVERTS: AtomicI64 = AtomicI64::new(0);

#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ExecutedInstruction {
    pub program: String,
    pub instruction: String,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SolanaCounterCall {
    pub contract: Option<String>,
    pub function: Option<String>,
    pub args: Option<String>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SolanaWireSummary {
    pub setup_failed: bool,
    pub failure_reason: Option<String>,
    pub total_calls: i64,
    pub total_reverts: i64,
    pub violated: bool,
    pub deterministic_test_failure: bool,
    pub executed_instructions: Vec<ExecutedInstruction>,
    pub counterexample_calls: Vec<SolanaCounterCall>,
}

/// Resolve `<sbf_so_name>` (e.g. `"solana_vault.so"`) inside `SBF_OUT_DIR`.
pub fn sbf_so_path(so_name: &str) -> PathBuf {
    let base = std::env::var_os("SBF_OUT_DIR")
        .or_else(|| std::env::var_os("BPF_OUT_DIR"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/deploy"));
    base.join(so_name)
}

/// Compute Anchor's 8-byte instruction sighash `sha256("global:<ix_name>")[..8]`.
pub fn anchor_ix_discriminator(ix_name: &str) -> [u8; 8] {
    let mut hasher = Sha256::new();
    hasher.update(format!("global:{ix_name}").as_bytes());
    let digest = hasher.finalize();
    let mut out = [0u8; 8];
    out.copy_from_slice(&digest[..8]);
    out
}

/// Compute Anchor's 8-byte account discriminator `sha256("account:<TypeName>")[..8]`.
pub fn anchor_account_discriminator(account_type_name: &str) -> [u8; 8] {
    let mut hasher = Sha256::new();
    hasher.update(format!("account:{account_type_name}").as_bytes());
    let digest = hasher.finalize();
    let mut out = [0u8; 8];
    out.copy_from_slice(&digest[..8]);
    out
}

/// Per-sequence / per-test tracker that records SVM calls and emits
/// `KNOWDIT_SOLANA_SUMMARY_JSON:{...}`.
#[derive(Debug, Default, Clone)]
pub struct SummaryTracker {
    pub executed: Vec<ExecutedInstruction>,
    pub sequence_calls: Vec<SolanaCounterCall>,
}

impl SummaryTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset_counters() {
        GLOBAL_CALLS.store(0, Ordering::Relaxed);
        GLOBAL_REVERTS.store(0, Ordering::Relaxed);
    }

    pub fn record_call(
        &mut self,
        program: &str,
        instruction: &str,
        args: impl Into<String>,
        succeeded: bool,
    ) {
        GLOBAL_CALLS.fetch_add(1, Ordering::Relaxed);
        if !succeeded {
            GLOBAL_REVERTS.fetch_add(1, Ordering::Relaxed);
        } else {
            self.executed.push(ExecutedInstruction {
                program: program.to_string(),
                instruction: instruction.to_string(),
            });
        }
        self.sequence_calls.push(SolanaCounterCall {
            contract: Some(program.to_string()),
            function: Some(instruction.to_string()),
            args: Some(args.into()),
        });
    }

    pub fn emit_ok(&self) {
        let payload = SolanaWireSummary {
            setup_failed: false,
            failure_reason: None,
            total_calls: GLOBAL_CALLS.load(Ordering::Relaxed),
            total_reverts: GLOBAL_REVERTS.load(Ordering::Relaxed),
            violated: false,
            deterministic_test_failure: false,
            executed_instructions: self.executed.clone(),
            counterexample_calls: Vec::new(),
        };
        if let Ok(json) = serde_json::to_string(&payload) {
            println!("{SOLANA_SUMMARY_PREFIX}{json}");
        }
    }

    pub fn emit_violation(&self, reason: impl Into<String>, deterministic: bool) {
        let reason_str = reason.into();
        let full_reason = if reason_str.contains(POC_VIOLATION_MARKER) {
            reason_str
        } else {
            format!("{POC_VIOLATION_MARKER} {reason_str}")
        };
        let payload = SolanaWireSummary {
            setup_failed: false,
            failure_reason: Some(full_reason),
            total_calls: GLOBAL_CALLS.load(Ordering::Relaxed).max(1),
            total_reverts: GLOBAL_REVERTS.load(Ordering::Relaxed),
            violated: true,
            deterministic_test_failure: deterministic,
            executed_instructions: self.executed.clone(),
            counterexample_calls: if deterministic {
                Vec::new()
            } else {
                self.sequence_calls.clone()
            },
        };
        if let Ok(json) = serde_json::to_string(&payload) {
            println!("{SOLANA_SUMMARY_PREFIX}{json}");
        }
    }

    pub fn fail_setup(reason: impl Into<String>) -> ! {
        let msg = format!("{SETUP_FAILED_MARKER} {}", reason.into());
        let payload = SolanaWireSummary {
            setup_failed: true,
            failure_reason: Some(msg.clone()),
            total_calls: GLOBAL_CALLS.load(Ordering::Relaxed),
            total_reverts: GLOBAL_REVERTS.load(Ordering::Relaxed),
            violated: false,
            deterministic_test_failure: false,
            executed_instructions: Vec::new(),
            counterexample_calls: Vec::new(),
        };
        if let Ok(json) = serde_json::to_string(&payload) {
            println!("{SOLANA_SUMMARY_PREFIX}{json}");
        }
        panic!("{msg}");
    }
}
"#;

#[derive(Debug, Clone)]
pub enum SolanaBackendKind {
    LocalDirect {
        cargo_bin: PathBuf,
        build_sbf_bin: Option<PathBuf>,
    },
    #[cfg(target_os = "linux")]
    LocalCgroup {
        cargo_bin: PathBuf,
        build_sbf_bin: Option<PathBuf>,
        cgroup_parent: String,
        mem_max_bytes: u64,
    },
    #[cfg(target_os = "linux")]
    Docker {
        image: String,
        mem_max_bytes: u64,
    },
}

/// Capabilities probed once at [`SolanaBackend::new`] time.
#[derive(Debug, Clone, Default)]
pub struct SolanaFeatures {
    /// True when `cargo llvm-cov --version` succeeded (enables Gate 2 Tier 1
    /// host source-line coverage when target crates link on host).
    pub llvm_cov_available: bool,
    /// True when `cargo-build-sbf` is available in the chosen backend.
    pub build_sbf_available: bool,
}

/// Resolved Solana execution runtime + probed feature flags.
#[derive(Debug, Clone)]
pub struct SolanaBackend {
    pub kind: SolanaBackendKind,
    pub features: SolanaFeatures,
}

impl SolanaBackend {
    /// Resolve the Solana execution runtime (`Docker` -> `$PATH` `cargo` fallback,
    /// or explicit `--solana-cargo-bin` local mode) and probe `llvm-cov` + `build-sbf`.
    pub fn new(
        cargo_bin: Option<PathBuf>,
        build_sbf_bin: Option<PathBuf>,
        docker_image: String,
        mem_cap_bytes: Option<u64>,
    ) -> Result<Self> {
        let kind = Self::resolve_kind(cargo_bin, build_sbf_bin, docker_image, mem_cap_bytes)?;
        let features = Self::probe_features(&kind);
        Ok(Self { kind, features })
    }

    fn resolve_kind(
        cargo_bin: Option<PathBuf>,
        build_sbf_bin: Option<PathBuf>,
        docker_image: String,
        mem_cap_bytes: Option<u64>,
    ) -> Result<SolanaBackendKind> {
        if let Some(cargo) = cargo_bin {
            return Self::local_kind(cargo, build_sbf_bin, mem_cap_bytes);
        }
        #[cfg(target_os = "linux")]
        {
            if Self::docker_image_available(&docker_image) {
                let cap = mem_cap_bytes.unwrap_or(0);
                tracing::info!(
                    "SolanaBackend: Docker image={docker_image} mem_max={} MB",
                    cap / (1024 * 1024),
                );
                return Ok(SolanaBackendKind::Docker {
                    image: docker_image,
                    mem_max_bytes: cap,
                });
            }
            if Self::docker_runtime_present() && Self::find_on_path("cargo").is_none() {
                let cap = mem_cap_bytes.unwrap_or(0);
                tracing::info!(
                    "SolanaBackend: Docker image={docker_image} (not yet pulled locally) mem_max={} MB",
                    cap / (1024 * 1024),
                );
                return Ok(SolanaBackendKind::Docker {
                    image: docker_image,
                    mem_max_bytes: cap,
                });
            }
            tracing::info!(
                "SolanaBackend: Docker image `{docker_image}` not present locally; resolving `cargo` from $PATH"
            );
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = docker_image;
        }
        match Self::find_on_path("cargo") {
            Some(cargo) => Self::local_kind(cargo, build_sbf_bin, mem_cap_bytes),
            None => Err(eyre!(
                "no usable Solana / Rust runtime: `--solana-cargo-bin` unset, Docker image \
                 not available, and `cargo` not found in $PATH. Build the image via \
                 `docker build -f docker/Dockerfile.solana -t lazymio/knowdit:solana .`, \
                 install Rust + Agave (`agave-install`), or pass `--solana-cargo-bin <path>`."
            )),
        }
    }

    fn local_kind(
        cargo_bin: PathBuf,
        build_sbf_bin: Option<PathBuf>,
        mem_cap_bytes: Option<u64>,
    ) -> Result<SolanaBackendKind> {
        let cargo_bin = std::fs::canonicalize(&cargo_bin).map_err(|err| {
            eyre!(
                "--solana-cargo-bin {} cannot be canonicalized ({err:#})",
                cargo_bin.display()
            )
        })?;
        let resolved_sbf = match build_sbf_bin {
            Some(explicit) => Some(std::fs::canonicalize(&explicit).map_err(|err| {
                eyre!(
                    "--solana-build-sbf-bin {} cannot be canonicalized ({err:#})",
                    explicit.display()
                )
            })?),
            None => Self::find_on_path("cargo-build-sbf"),
        };

        #[cfg(target_os = "linux")]
        {
            if let Some(cap) = mem_cap_bytes {
                if let Some(parent) = ScopedCgroup::probe_parent() {
                    tracing::info!(
                        "SolanaBackend: LocalCgroup cargo={} parent={parent} mem_max={} MB",
                        cargo_bin.display(),
                        cap / (1024 * 1024),
                    );
                    return Ok(SolanaBackendKind::LocalCgroup {
                        cargo_bin,
                        build_sbf_bin: resolved_sbf,
                        cgroup_parent: parent,
                        mem_max_bytes: cap,
                    });
                }
                tracing::warn!(
                    "SolanaBackend: --solana-mem-cap-gb requested but no writable cgroup v2 \
                     memory parent found; falling back to LocalDirect"
                );
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            if mem_cap_bytes.is_some() {
                tracing::warn!("SolanaBackend: --solana-mem-cap-gb is Linux-only; ignoring");
            }
        }
        tracing::info!("SolanaBackend: LocalDirect cargo={}", cargo_bin.display());
        Ok(SolanaBackendKind::LocalDirect {
            cargo_bin,
            build_sbf_bin: resolved_sbf,
        })
    }

    fn probe_features(kind: &SolanaBackendKind) -> SolanaFeatures {
        use std::process::{Command as StdCommand, Stdio};
        match kind {
            SolanaBackendKind::LocalDirect {
                cargo_bin,
                build_sbf_bin,
            } => {
                let llvm_cov = StdCommand::new(cargo_bin)
                    .args(["llvm-cov", "--version"])
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(false);
                SolanaFeatures {
                    llvm_cov_available: llvm_cov,
                    build_sbf_available: build_sbf_bin.is_some(),
                }
            }
            #[cfg(target_os = "linux")]
            SolanaBackendKind::LocalCgroup {
                cargo_bin,
                build_sbf_bin,
                ..
            } => {
                let llvm_cov = StdCommand::new(cargo_bin)
                    .args(["llvm-cov", "--version"])
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(false);
                SolanaFeatures {
                    llvm_cov_available: llvm_cov,
                    build_sbf_available: build_sbf_bin.is_some(),
                }
            }
            #[cfg(target_os = "linux")]
            SolanaBackendKind::Docker { image, .. } => {
                let llvm_cov = StdCommand::new("docker")
                    .args([
                        "run",
                        "--rm",
                        "--network",
                        "none",
                        "--entrypoint",
                        "/usr/local/cargo/bin/cargo",
                        image,
                        "llvm-cov",
                        "--version",
                    ])
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(false);
                let build_sbf = StdCommand::new("docker")
                    .args([
                        "run",
                        "--rm",
                        "--network",
                        "none",
                        "--entrypoint",
                        "/opt/solana/active_release/bin/cargo-build-sbf",
                        image,
                        "--version",
                    ])
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(true);
                SolanaFeatures {
                    llvm_cov_available: llvm_cov,
                    build_sbf_available: build_sbf,
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn docker_runtime_present() -> bool {
        std::process::Command::new("docker")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    #[cfg(target_os = "linux")]
    fn docker_image_available(image: &str) -> bool {
        std::process::Command::new("docker")
            .args(["image", "inspect", image])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn find_on_path(bin_name: &str) -> Option<PathBuf> {
        let path = std::env::var_os("PATH")?;
        for dir in std::env::split_paths(&path) {
            let candidate = dir.join(bin_name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        None
    }
}

/// Raw subprocess result from [`SolanaRunner`].
#[derive(Debug, Clone)]
pub struct SolanaOutput {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: i64,
    pub timed_out: bool,
}

/// Sandboxed runner bound to one audited repository and shared `.knowdit-cargo-target`.
#[derive(Debug, Clone)]
pub struct SolanaRunner {
    backend: SolanaBackend,
    repo_root: PathBuf,
    harness_dir: PathBuf,
    shared_target_dir: PathBuf,
    sbf_out_dir: PathBuf,
    timeout: Duration,
    lock_grace: Option<Duration>,
}

impl SolanaRunner {
    pub fn new(
        backend: SolanaBackend,
        repo_root: PathBuf,
        harness_dir: PathBuf,
        timeout: Duration,
    ) -> Self {
        let repo_root = Self::make_absolute(repo_root);
        let harness_dir = Self::make_absolute(harness_dir);
        let shared_target_dir = harness_dir.join(".knowdit-cargo-target");
        let sbf_out_dir = repo_root.join("target").join("deploy");
        Self {
            backend,
            repo_root,
            harness_dir,
            shared_target_dir,
            sbf_out_dir,
            timeout,
            lock_grace: None,
        }
    }

    pub fn features(&self) -> &SolanaFeatures {
        &self.backend.features
    }

    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }

    pub fn harness_dir(&self) -> &Path {
        &self.harness_dir
    }

    pub fn shared_target_dir(&self) -> &Path {
        &self.shared_target_dir
    }

    pub fn sbf_out_dir(&self) -> &Path {
        &self.sbf_out_dir
    }

    /// Generate `Cargo.toml` + `src/lib.rs` (`knowdit_harness_rt`) + `tests/`
    /// inside `per_spec_dir`, copying `<harness_dir>/.knowdit-seed.lock` when
    /// available so Cargo skips version resolution.
    pub async fn write_per_spec_crate_scaffold(
        &self,
        per_spec_dir: &Path,
        spec_id: i32,
    ) -> Result<()> {
        let src_dir = per_spec_dir.join("src");
        let tests_dir = per_spec_dir.join("tests");
        tokio::fs::create_dir_all(&src_dir)
            .await
            .wrap_err_with(|| format!("failed to create {}", src_dir.display()))?;
        tokio::fs::create_dir_all(&tests_dir)
            .await
            .wrap_err_with(|| format!("failed to create {}", tests_dir.display()))?;

        let cargo_toml = format!(
            r#"[package]
name = "knowdit-solana-spec-{spec_id}"
version = "0.1.0"
edition = "2021"
publish = false

[workspace]

[lib]
name = "knowdit_harness_rt"
path = "src/lib.rs"

[dependencies]
borsh = {{ version = "1", features = ["derive"] }}
bs58 = "0.5"
bytemuck = {{ version = "1", features = ["derive"] }}
litesvm = "0.6"
proptest = "1"
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"
sha2 = "0.10"
solana-program = "2.2"
solana-sdk = "2.2"
spl-associated-token-account = "6"
spl-token = "7"
"#
        );

        tokio::fs::write(per_spec_dir.join("Cargo.toml"), cargo_toml)
            .await
            .wrap_err("failed to write per-spec Cargo.toml")?;
        tokio::fs::write(src_dir.join("lib.rs"), HARNESS_RT_LIB_RS)
            .await
            .wrap_err("failed to write per-spec src/lib.rs")?;

        let seed_lock = self.harness_dir.join(".knowdit-seed.lock");
        if seed_lock.is_file() {
            let _ = tokio::fs::copy(&seed_lock, per_spec_dir.join("Cargo.lock")).await;
        }
        Ok(())
    }

    /// Run `cargo <args...>` under [`SOLANA_CARGO_LOCK`] with `CARGO_TARGET_DIR`,
    /// `SBF_OUT_DIR`, `BPF_OUT_DIR`, and `extra_env` configured.
    pub async fn run_cargo(
        &self,
        args: &[String],
        work_dir: &Path,
        extra_env: &BTreeMap<String, String>,
    ) -> Result<SolanaOutput> {
        self.run_locked(SpawnBinary::Cargo, args, work_dir, extra_env)
            .await
    }

    /// Run `cargo-build-sbf` under [`SOLANA_CARGO_LOCK`] to compile target SBF `.so`
    /// binaries into `<repo_root>/target/deploy/`.
    pub async fn run_build_sbf(&self, manifest_path: Option<&Path>) -> Result<SolanaOutput> {
        let mut args = vec![
            "--sbf-out-dir".to_string(),
            self.sbf_out_dir.display().to_string(),
        ];
        if let Some(mp) = manifest_path {
            args.push("--manifest-path".to_string());
            args.push(mp.display().to_string());
        }
        self.run_locked(
            SpawnBinary::BuildSbf,
            &args,
            &self.repo_root,
            &BTreeMap::new(),
        )
        .await
    }

    /// Preflight the Solana / LiteSVM environment before any LLM stage runs:
    /// 1. Ensures SBF `.so` artifacts exist in `<repo>/target/deploy/` (building
    ///    them via `cargo-build-sbf` if missing and the SBF toolchain is present).
    /// 2. Compiles and runs a minimal `LiteSVM` + `knowdit_harness_rt` smoke test
    ///    against the shared `.knowdit-cargo-target`, caching `Cargo.lock` for
    ///    all subsequent per-spec harnesses.
    pub async fn preflight(&self) -> Result<()> {
        tokio::fs::create_dir_all(&self.harness_dir)
            .await
            .wrap_err_with(|| {
                format!(
                    "preflight: failed to create harness_dir {}",
                    self.harness_dir.display()
                )
            })?;
        tokio::fs::create_dir_all(&self.shared_target_dir)
            .await
            .wrap_err_with(|| {
                format!(
                    "preflight: failed to create shared Cargo target dir {}",
                    self.shared_target_dir.display()
                )
            })?;
        tokio::fs::create_dir_all(&self.sbf_out_dir)
            .await
            .wrap_err_with(|| {
                format!(
                    "preflight: failed to create SBF deploy dir {}",
                    self.sbf_out_dir.display()
                )
            })?;

        let workspace = RustWorkspace::discover(&self.repo_root)?;
        self.ensure_sbf_programs_built(&workspace).await?;

        let preflight_id = PREFLIGHT_COUNTER.fetch_add(1, Ordering::Relaxed);
        let preflight_dir = self.harness_dir.join(format!(
            "_knowdit_solana_preflight_{}_{}_{}",
            std::process::id(),
            Self::nanos_now(),
            preflight_id
        ));
        self.write_per_spec_crate_scaffold(&preflight_dir, 0)
            .await?;

        let smoke_rs = r#"use knowdit_harness_rt::SummaryTracker;
use litesvm::LiteSVM;

#[test]
fn test_knowdit_solana_preflight() {
    let mut svm = LiteSVM::new();
    let pk = solana_sdk::pubkey::Pubkey::new_unique();
    svm.airdrop(&pk, 1_000_000_000).expect("preflight airdrop");
    assert_eq!(svm.get_balance(&pk), Some(1_000_000_000));
    let mut tracker = SummaryTracker::new();
    tracker.record_call("preflight", "airdrop", "1_sol", true);
    tracker.emit_ok();
}
"#;
        let smoke_path = preflight_dir.join("tests").join("preflight_smoke.rs");
        tokio::fs::write(&smoke_path, smoke_rs)
            .await
            .wrap_err_with(|| format!("preflight: failed to write {}", smoke_path.display()))?;

        let argv = vec![
            "test".to_string(),
            "--manifest-path".to_string(),
            preflight_dir.join("Cargo.toml").display().to_string(),
            "--test".to_string(),
            "preflight_smoke".to_string(),
            "--".to_string(),
            "--nocapture".to_string(),
        ];
        let out_res = self
            .run_cargo(&argv, &preflight_dir, &BTreeMap::new())
            .await;

        let generated_lock = preflight_dir.join("Cargo.lock");
        if generated_lock.is_file() {
            let _ = tokio::fs::copy(&generated_lock, self.harness_dir.join(".knowdit-seed.lock"))
                .await;
        }
        if let Err(err) = tokio::fs::remove_dir_all(&preflight_dir).await {
            tracing::warn!(
                preflight_dir = %preflight_dir.display(),
                "preflight cleanup failed (non-fatal): {err}"
            );
        }

        let out = out_res.wrap_err("preflight: Solana cargo test spawn failed")?;
        let combined_tail = format!(
            "{}\n{}",
            Self::tail_utf8(&out.stdout, 4_000),
            Self::tail_utf8(&out.stderr, 4_000),
        );
        if out.exit_code == 0
            && (combined_tail.contains("KNOWDIT_SOLANA_SUMMARY_JSON:")
                || combined_tail.contains("test result: ok"))
        {
            tracing::info!(
                duration_ms = out.duration_ms,
                "preflight: Solana LiteSVM environment OK"
            );
            return Ok(());
        }

        if out.timed_out {
            return Err(eyre!(
                "preflight: Solana LiteSVM smoke compile timed out after {}s. \
                 Cold compiles of `litesvm` + `solana-sdk` can take 1-3 minutes on first run; \
                 raise `--solana-timeout-secs` and re-run.\n\
                 --- cargo output tail ---\n{}",
                self.timeout.as_secs(),
                combined_tail,
            ));
        }

        Err(eyre!(
            "preflight: Solana LiteSVM smoke test failed (exit_code={}). \
             Ensure `lazymio/knowdit:solana` is built (`docker build -f docker/Dockerfile.solana -t lazymio/knowdit:solana .`) \
             or host Cargo has access to `litesvm` / `solana-sdk` crates.\n\
             --- cargo output tail ---\n{}",
            out.exit_code,
            combined_tail,
        ))
    }

    async fn ensure_sbf_programs_built(&self, workspace: &RustWorkspace) -> Result<()> {
        let programs = workspace.programs();
        if programs.is_empty() {
            return Ok(());
        }

        // Copy any `.so` already present in `target/sbf-solana-solana/release` into `target/deploy`.
        let alt_sbf_dir = self
            .repo_root
            .join("target")
            .join("sbf-solana-solana")
            .join("release");
        for prog in &programs {
            let deploy_so = self.sbf_out_dir.join(&prog.sbf_so_name);
            let alt_so = alt_sbf_dir.join(&prog.sbf_so_name);
            if !deploy_so.is_file() && alt_so.is_file() {
                let _ = tokio::fs::copy(&alt_so, &deploy_so).await;
            }
        }

        let missing: Vec<String> = programs
            .iter()
            .filter(|p| !self.sbf_out_dir.join(&p.sbf_so_name).is_file())
            .map(|p| p.sbf_so_name.clone())
            .collect();
        if missing.is_empty() {
            return Ok(());
        }

        if !self.backend.features.build_sbf_available {
            return Err(eyre!(
                "preflight: missing compiled Solana SBF shared objects {:?} under {}, \
                 and `cargo-build-sbf` is not available in the active backend. \
                 Run `python3 scripts/autobuild_solana.py {}`, build `lazymio/knowdit:solana` \
                 (`docker build -f docker/Dockerfile.solana -t lazymio/knowdit:solana .`), \
                 or install Agave (`agave-install`) and pass `--solana-build-sbf-bin`.",
                missing,
                self.sbf_out_dir.display(),
                self.repo_root.display(),
            ));
        }

        tracing::info!(
            missing = ?missing,
            "preflight: building Solana SBF programs via cargo-build-sbf"
        );
        let out = self.run_build_sbf(None).await?;
        if out.exit_code != 0 {
            return Err(eyre!(
                "preflight: `cargo-build-sbf` failed (exit_code={}):\n{}\n{}",
                out.exit_code,
                Self::tail_utf8(&out.stdout, 4_000),
                Self::tail_utf8(&out.stderr, 4_000),
            ));
        }
        Ok(())
    }

    async fn run_locked(
        &self,
        which: SpawnBinary,
        args: &[String],
        work_dir: &Path,
        extra_env: &BTreeMap<String, String>,
    ) -> Result<SolanaOutput> {
        let grace = self.lock_grace.unwrap_or(SOLANA_LOCK_GRACE);
        let lock_timeout = self.timeout.saturating_add(grace);
        let _guard = match tokio::time::timeout(lock_timeout, SOLANA_CARGO_LOCK.lock()).await {
            Ok(guard) => guard,
            Err(_) => {
                return Err(eyre!(
                    "timed out after {}s waiting for the shared Solana cargo lock; another \
                     invocation is wedged. Failing this invocation so remaining links proceed.",
                    lock_timeout.as_secs()
                ));
            }
        };

        let start = Instant::now();
        let handle = self.spawn(which, args, work_dir, extra_env)?;
        let timeout = self.timeout;
        let outcome = tokio::time::timeout(timeout, handle.wait_with_output()).await;
        let duration_ms = start.elapsed().as_millis() as i64;

        match outcome {
            Ok(Ok(out)) => Ok(SolanaOutput {
                exit_code: out.status.code().unwrap_or(-2),
                stdout: Self::bytes_into_string(out.stdout),
                stderr: Self::bytes_into_string(out.stderr),
                duration_ms,
                timed_out: false,
            }),
            Ok(Err(io_err)) => Err(io_err).wrap_err("Solana subprocess wait_with_output failed"),
            Err(_elapsed) => Ok(SolanaOutput {
                exit_code: -1,
                stdout: String::new(),
                stderr: format!(
                    "knowdit: Solana cargo subprocess timed out after {} seconds and was killed. \
                     Consider raising `--solana-timeout-secs` or lowering `--solana-test-runs`.",
                    timeout.as_secs()
                ),
                duration_ms,
                timed_out: true,
            }),
        }
    }

    fn spawn(
        &self,
        which: SpawnBinary,
        args: &[String],
        work_dir: &Path,
        extra_env: &BTreeMap<String, String>,
    ) -> Result<SolanaHandle> {
        let mut envs: BTreeMap<String, String> = extra_env.clone();
        envs.entry("CARGO_TARGET_DIR".to_string())
            .or_insert_with(|| self.shared_target_dir.display().to_string());
        envs.entry("SBF_OUT_DIR".to_string())
            .or_insert_with(|| self.sbf_out_dir.display().to_string());
        envs.entry("BPF_OUT_DIR".to_string())
            .or_insert_with(|| self.sbf_out_dir.display().to_string());

        match &self.backend.kind {
            SolanaBackendKind::LocalDirect {
                cargo_bin,
                build_sbf_bin,
            } => {
                let binary = Self::select_local_bin(which, cargo_bin, build_sbf_bin.as_ref())?;
                let mut cmd = Command::new(&binary);
                cmd.args(args)
                    .current_dir(work_dir)
                    .envs(&envs)
                    .kill_on_drop(true);
                let child = cmd
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .wrap_err_with(|| format!("failed to spawn `{} {:?}`", binary.display(), args))?;
                Ok(SolanaHandle::LocalDirect { child: Some(child) })
            }
            #[cfg(target_os = "linux")]
            SolanaBackendKind::LocalCgroup {
                cargo_bin,
                build_sbf_bin,
                cgroup_parent,
                mem_max_bytes,
            } => {
                let binary = Self::select_local_bin(which, cargo_bin, build_sbf_bin.as_ref())?;
                let cg_name = format!(
                    "knowdit-solana-{}-{}",
                    std::process::id(),
                    Self::nanos_now()
                );
                let scoped = ScopedCgroup::create(cgroup_parent, &cg_name, *mem_max_bytes)
                    .map_err(|e| eyre!("failed to create cgroup {cgroup_parent}/{cg_name}: {e}"))?;

                let mut cmd = Command::new(&binary);
                cmd.args(args)
                    .current_dir(work_dir)
                    .envs(&envs)
                    .kill_on_drop(true);
                let child = cmd
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .wrap_err_with(|| format!("failed to spawn `{} {:?}`", binary.display(), args))?;
                if let Some(pid) = child.id()
                    && let Err(err) = scoped.add_pid(pid)
                {
                    tracing::warn!(
                        "SolanaRunner: failed to attach pid {pid} to cgroup: {err}"
                    );
                }
                Ok(SolanaHandle::LocalCgroup {
                    child: Some(child),
                    _cgroup: scoped,
                })
            }
            #[cfg(target_os = "linux")]
            SolanaBackendKind::Docker {
                image,
                mem_max_bytes,
            } => {
                let container_name = format!(
                    "knowdit-solana-{}-{}",
                    std::process::id(),
                    Self::nanos_now()
                );
                let (uid, gid) = Self::read_proc_uid_gid();
                let repo_str = self.repo_root.display().to_string();
                let work_str = work_dir.display().to_string();
                let entrypoint = match which {
                    SpawnBinary::Cargo => "/usr/local/cargo/bin/cargo",
                    SpawnBinary::BuildSbf => "/opt/solana/active_release/bin/cargo-build-sbf",
                };

                let mut cmd = Command::new("docker");
                cmd.arg("run")
                    .arg("--rm")
                    .arg("--name")
                    .arg(&container_name)
                    .arg("--user")
                    .arg(format!("{uid}:{gid}"))
                    .arg("--network")
                    .arg("none")
                    .arg("-v")
                    .arg(format!("{repo_str}:{repo_str}"));
                if !self.harness_dir.starts_with(&self.repo_root) {
                    let h_str = self.harness_dir.display().to_string();
                    cmd.arg("-v").arg(format!("{h_str}:{h_str}"));
                }
                cmd.arg("-w").arg(&work_str);
                if *mem_max_bytes > 0 {
                    cmd.arg(format!("--memory={mem_max_bytes}b"))
                        .arg(format!("--memory-swap={mem_max_bytes}b"));
                }
                cmd.arg("-e").arg("CARGO_NET_OFFLINE=true");
                for (k, v) in &envs {
                    cmd.arg("-e").arg(format!("{k}={v}"));
                }
                cmd.arg("--entrypoint")
                    .arg(entrypoint)
                    .arg(image)
                    .args(args)
                    .kill_on_drop(true);

                let child = cmd
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .wrap_err_with(|| format!("failed to spawn `docker run ... {image}`"))?;
                Ok(SolanaHandle::Docker {
                    child: Some(child),
                    container_name,
                })
            }
        }
    }

    fn select_local_bin(
        which: SpawnBinary,
        cargo_bin: &Path,
        build_sbf_bin: Option<&PathBuf>,
    ) -> Result<PathBuf> {
        match which {
            SpawnBinary::Cargo => Ok(cargo_bin.to_path_buf()),
            SpawnBinary::BuildSbf => build_sbf_bin.cloned().ok_or_else(|| {
                eyre!(
                    "`cargo-build-sbf` is not available in Local mode; install Agave \
                     (`agave-install`) or pass `--solana-build-sbf-bin <path>`"
                )
            }),
        }
    }

    fn make_absolute(path: PathBuf) -> PathBuf {
        std::fs::canonicalize(&path)
            .or_else(|_| std::path::absolute(&path))
            .unwrap_or(path)
    }

    fn nanos_now() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    }

    fn bytes_into_string(bytes: Vec<u8>) -> String {
        String::from_utf8(bytes)
            .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
    }

    fn tail_utf8(s: &str, max_bytes: usize) -> &str {
        if s.len() <= max_bytes {
            return s;
        }
        let mut start = s.len() - max_bytes;
        while start < s.len() && !s.is_char_boundary(start) {
            start += 1;
        }
        &s[start..]
    }

    #[cfg(target_os = "linux")]
    fn read_proc_uid_gid() -> (u32, u32) {
        let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let mut uid: u32 = 1000;
        let mut gid: u32 = 1000;
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("Uid:") {
                if let Some(real) = rest.split_whitespace().next()
                    && let Ok(v) = real.parse()
                {
                    uid = v;
                }
            } else if let Some(rest) = line.strip_prefix("Gid:")
                && let Some(real) = rest.split_whitespace().next()
                && let Ok(v) = real.parse()
            {
                gid = v;
            }
        }
        (uid, gid)
    }
}

#[derive(Debug, Clone, Copy)]
enum SpawnBinary {
    Cargo,
    BuildSbf,
}

#[derive(Debug)]
enum SolanaHandle {
    LocalDirect {
        child: Option<tokio::process::Child>,
    },
    #[cfg(target_os = "linux")]
    LocalCgroup {
        child: Option<tokio::process::Child>,
        _cgroup: ScopedCgroup,
    },
    #[cfg(target_os = "linux")]
    Docker {
        child: Option<tokio::process::Child>,
        container_name: String,
    },
}

impl SolanaHandle {
    fn take_child(&mut self) -> Option<tokio::process::Child> {
        match self {
            Self::LocalDirect { child } => child.take(),
            #[cfg(target_os = "linux")]
            Self::LocalCgroup { child, .. } => child.take(),
            #[cfg(target_os = "linux")]
            Self::Docker { child, .. } => child.take(),
        }
    }

    async fn wait_with_output(mut self) -> std::io::Result<std::process::Output> {
        let Some(child) = self.take_child() else {
            return Err(std::io::Error::other(
                "SolanaHandle::wait_with_output: child already consumed",
            ));
        };
        child.wait_with_output().await
    }
}

impl Drop for SolanaHandle {
    fn drop(&mut self) {
        match self {
            Self::LocalDirect { .. } => {}
            #[cfg(target_os = "linux")]
            Self::LocalCgroup { .. } => {}
            #[cfg(target_os = "linux")]
            Self::Docker { container_name, .. } => {
                let _ = std::process::Command::new("docker")
                    .args(["kill", container_name.as_str()])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn waiting_for_wedged_solana_cargo_lock_times_out() {
        let dir = std::env::temp_dir();
        let mut runner = SolanaRunner::new(
            SolanaBackend {
                kind: SolanaBackendKind::LocalDirect {
                    cargo_bin: PathBuf::from("/bin/true"),
                    build_sbf_bin: None,
                },
                features: SolanaFeatures::default(),
            },
            dir.clone(),
            dir.clone(),
            Duration::from_millis(50),
        );
        runner.lock_grace = Some(Duration::from_millis(100));

        let _held = SOLANA_CARGO_LOCK.lock().await;
        let err = runner
            .run_cargo(&["--version".to_string()], &dir, &BTreeMap::new())
            .await
            .expect_err("wedged lock must time out");
        assert!(
            err.to_string()
                .contains("waiting for the shared Solana cargo lock"),
            "unexpected error: {err}"
        );
    }
}
