//! `forge` invocation backends.
//!
//! Three flavors, picked once at startup and held by every per-spec
//! `RunForgeTool`:
//!
//!   1. [`ForgeBackend::LocalDirect`] — spawn the host's `forge` binary,
//!      no memory cap.
//!   2. [`ForgeBackend::LocalCgroup`] — spawn the host's `forge` inside
//!      a fresh cgroup v2 child of our delegated user.slice ancestor,
//!      with `memory.max` enforcing a per-invocation ceiling. Linux only.
//!   3. [`ForgeBackend::Docker`] — spawn `docker run --rm
//!      --memory=<N> --memory-swap=<N> -v <repo>:<repo> -w <repo>
//!      --entrypoint forge <image> <args...>`. Linux only.
//!
//! Per-invocation cleanup goes through [`ForgeHandle`] whose `Drop` does
//! the right thing for the variant — most importantly an explicit
//! `docker kill --signal=KILL <name>` for the Docker case, since
//! SIGKILL'ing the docker CLI client doesn't reliably propagate to the
//! daemon-managed container.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Monotonically increasing process-local counter, used to uniquify
/// concurrent preflight directories (`_knowdit_preflight_{pid}_{nanos}_{ctr}`)
/// so two preflights in the same nanosecond bucket still get distinct
/// dirs. `Ordering::Relaxed` is fine — we only need uniqueness, not
/// ordering between writes.
static PREFLIGHT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Last `max_bytes` of `s`, snapped to a UTF-8 boundary. Used for
/// embedding forge output tails in error messages without risking
/// a `str` slice panic on multi-byte boundaries.
/// Lossy UTF-8 conversion that reuses the buffer when it is already valid
/// (the common case) instead of allocating and copying a second copy.
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

use color_eyre::eyre::{Result, WrapErr, eyre};
use tokio::process::Command;
use tokio::sync::{Mutex, OnceCell};

#[cfg(target_os = "linux")]
use super::cgroup::ScopedCgroup;

/// Extra time a caller will wait for a slot in [`ForgeSlotPool`] beyond its own
/// forge timeout. A well-behaved holder releases within its timeout, so this
/// only trips when all slot holders are genuinely wedged.
const FORGE_LOCK_GRACE: std::time::Duration = std::time::Duration::from_secs(60);

/// Subdirectory under `<work_dir>` holding `template/{out,cache}` and
/// `slot_{i}/{out,cache}` so sharded build directories work identically on the
/// host and inside Docker's `-v <work_dir>:<work_dir>` mount.
const FORGE_SLOTS_DIR: &str = ".knowdit-forge-slots";

/// Relative `--out` and `--cache-path` paths for one initialized build slot.
#[derive(Debug, Clone)]
pub struct SlotPaths {
    pub rel_out: String,
    pub rel_cache: String,
}

/// RAII lease on one build slot (`0..K`) from [`ForgeSlotPool`]. Always returns
/// its slot index to the bounded channel on [`Drop`] — whether the caller
/// finishes normally, times out, aborts, or unwinds from a panic.
#[derive(Debug)]
pub struct ForgeSlotGuard {
    slot_idx: usize,
    tx: async_channel::Sender<usize>,
}

impl ForgeSlotGuard {
    pub fn slot_idx(&self) -> usize {
        self.slot_idx
    }
}

impl Drop for ForgeSlotGuard {
    fn drop(&mut self) {
        let _ = self.tx.try_send(self.slot_idx);
    }
}

/// Bounded pool of `K` isolated Foundry build/cache slots (`slot_0..slot_{K-1}`)
/// seeded lazily from `preflight`'s `template/{out,cache}` directory.
#[derive(Debug)]
pub struct ForgeSlotPool {
    capacity: usize,
    tx: async_channel::Sender<usize>,
    rx: async_channel::Receiver<usize>,
    initialized: Mutex<BTreeMap<(PathBuf, usize), Arc<OnceCell<SlotPaths>>>>,
}

impl ForgeSlotPool {
    pub fn new(concurrency: usize) -> Self {
        let capacity = concurrency.max(1);
        let (tx, rx) = async_channel::bounded(capacity);
        for slot_idx in 0..capacity {
            let _ = tx.try_send(slot_idx);
        }
        Self {
            capacity,
            tx,
            rx,
            initialized: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Acquire one slot lease, failing after `wait_timeout` if every slot in
    /// the pool remains wedged.
    pub async fn acquire(&self, wait_timeout: Duration) -> Result<ForgeSlotGuard> {
        match tokio::time::timeout(wait_timeout, self.rx.recv()).await {
            Ok(Ok(slot_idx)) => Ok(ForgeSlotGuard {
                slot_idx,
                tx: self.tx.clone(),
            }),
            Ok(Err(_)) => Err(eyre!("forge slot pool channel closed unexpectedly")),
            Err(_) => Err(eyre!(
                "timed out after {}s waiting for the shared forge slot pool (capacity={}); \
                 all forge slot holders are wedged (killed child / hung holder). Failing this \
                 invocation so remaining links can proceed.",
                wait_timeout.as_secs(),
                self.capacity,
            )),
        }
    }

    /// Drop cached slot-init markers for `work_dir` when a fresh `preflight`
    /// rebuilds `template/{out,cache}`.
    pub async fn reset_work_dir(&self, work_dir: &Path) {
        let mut guard = self.initialized.lock().await;
        guard.retain(|(dir, _), _| dir != work_dir);
    }

    /// Lazily clone `<work_dir>/.knowdit-forge-slots/template` into
    /// `<work_dir>/.knowdit-forge-slots/slot_{slot_idx}` on first use, returning
    /// the relative `--out` and `--cache-path` strings for `forge` CLI argv.
    pub async fn ensure_slot_paths(&self, work_dir: &Path, slot_idx: usize) -> Result<SlotPaths> {
        let cell = {
            let mut guard = self.initialized.lock().await;
            guard
                .entry((work_dir.to_path_buf(), slot_idx))
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };
        let work_dir_owned = work_dir.to_path_buf();
        let paths = cell
            .get_or_try_init(|| async move {
                Self::init_slot_on_disk(&work_dir_owned, slot_idx).await
            })
            .await?;
        Ok(paths.clone())
    }

    async fn init_slot_on_disk(work_dir: &Path, slot_idx: usize) -> Result<SlotPaths> {
        let slots_root = work_dir.join(FORGE_SLOTS_DIR);
        tokio::fs::create_dir_all(&slots_root)
            .await
            .wrap_err_with(|| format!("failed to create {}", slots_root.display()))?;

        let template_dir = slots_root.join("template");
        let slot_dir = slots_root.join(format!("slot_{slot_idx}"));
        if slot_dir.exists() {
            let _ = tokio::fs::remove_dir_all(&slot_dir).await;
        }

        if template_dir.is_dir() {
            let cloned_via_cp = Command::new("cp")
                .arg("--reflink=auto")
                .arg("-a")
                .arg(&template_dir)
                .arg(&slot_dir)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .await
                .map(|s| s.success())
                .unwrap_or(false);
            if !cloned_via_cp {
                Self::copy_dir_recursive(&template_dir, &slot_dir).await?;
            }
        }

        let slot_out = slot_dir.join("out");
        let slot_cache = slot_dir.join("cache");
        tokio::fs::create_dir_all(&slot_out)
            .await
            .wrap_err_with(|| format!("failed to create {}", slot_out.display()))?;
        tokio::fs::create_dir_all(&slot_cache)
            .await
            .wrap_err_with(|| format!("failed to create {}", slot_cache.display()))?;

        Ok(SlotPaths {
            rel_out: format!("{FORGE_SLOTS_DIR}/slot_{slot_idx}/out"),
            rel_cache: format!("{FORGE_SLOTS_DIR}/slot_{slot_idx}/cache"),
        })
    }

    async fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
        let mut stack: Vec<(PathBuf, PathBuf)> = vec![(src.to_path_buf(), dst.to_path_buf())];
        while let Some((cur_src, cur_dst)) = stack.pop() {
            tokio::fs::create_dir_all(&cur_dst)
                .await
                .wrap_err_with(|| format!("failed to create {}", cur_dst.display()))?;
            let mut entries = tokio::fs::read_dir(&cur_src)
                .await
                .wrap_err_with(|| format!("failed to read {}", cur_src.display()))?;
            while let Some(entry) = entries.next_entry().await? {
                let file_type = entry.file_type().await?;
                let next_src = entry.path();
                let next_dst = cur_dst.join(entry.file_name());
                if file_type.is_dir() {
                    stack.push((next_src, next_dst));
                } else if file_type.is_file() {
                    tokio::fs::copy(&next_src, &next_dst)
                        .await
                        .wrap_err_with(|| {
                            format!(
                                "failed to copy {} -> {}",
                                next_src.display(),
                                next_dst.display()
                            )
                        })?;
                }
            }
        }
        Ok(())
    }
}

/// One forge invocation backend = how to spawn `forge <args>`. The
/// outer [`ForgeBackend`] owns this plus a probed [`ForgeFeatures`]
/// table so callers can branch on optional flags (currently:
/// `forge coverage --force-via-ir`) without re-running probes.
#[derive(Debug, Clone)]
pub enum ForgeBackendKind {
    LocalDirect {
        binary: PathBuf,
    },
    #[cfg(target_os = "linux")]
    LocalCgroup {
        binary: PathBuf,
        /// Path relative to `/sys/fs/cgroup` (the value returned by
        /// [`ScopedCgroup::probe_parent`]).
        cgroup_parent: String,
        mem_max_bytes: u64,
    },
    #[cfg(target_os = "linux")]
    Docker {
        image: String,
        mem_max_bytes: u64,
    },
}

/// Capability set probed at backend construction. Lets callers
/// (e.g. the per-spec `RunForgeTool`) opt into newer forge flags
/// without re-asking each invocation.
#[derive(Debug, Clone, Default)]
pub struct ForgeFeatures {
    /// Whether the probed `forge coverage` accepts the
    /// `--force-via-ir` flag. Required to compile via-IR-only
    /// projects under coverage (forge coverage historically refused
    /// viaIR for source-map fidelity reasons; the flag opts back in
    /// at the cost of some line-mapping accuracy).
    pub coverage_force_via_ir: bool,
}

/// Forge invocation runtime. Combines the chosen spawn strategy
/// (`kind`), the capability set we probed at construction time
/// (`features`), and the sharded [`ForgeSlotPool`] isolating concurrent
/// `out/` and `cache/` directories. Constructed once per fuzz pass;
/// cloned into each per-spec [`ForgeRunner`].
#[derive(Debug, Clone)]
pub struct ForgeBackend {
    pub kind: ForgeBackendKind,
    pub features: ForgeFeatures,
    pub slot_pool: Arc<ForgeSlotPool>,
}

impl ForgeBackend {
    /// Resolve a usable `forge` runtime with a 1-slot pool (use
    /// [`Self::with_concurrency`] to size a multi-slot pool).
    pub fn new(
        forge_bin: Option<PathBuf>,
        docker_image: String,
        mem_cap_bytes: Option<u64>,
    ) -> Result<Self> {
        Self::with_concurrency(forge_bin, docker_image, mem_cap_bytes, 1)
    }

    /// Resolve a usable `forge` runtime, probe its feature flags, and allocate
    /// a [`ForgeSlotPool`] with `forge_concurrency.max(1)` isolated build slots.
    pub fn with_concurrency(
        forge_bin: Option<PathBuf>,
        docker_image: String,
        mem_cap_bytes: Option<u64>,
        forge_concurrency: usize,
    ) -> Result<Self> {
        let kind = resolve_kind(forge_bin, docker_image, mem_cap_bytes)?;
        let features = probe_features(&kind);
        let slot_pool = Arc::new(ForgeSlotPool::new(forge_concurrency.max(1)));
        tracing::info!(
            "ForgeBackend: initialized ForgeSlotPool with {} slot(s)",
            slot_pool.capacity()
        );
        Ok(Self {
            kind,
            features,
            slot_pool,
        })
    }
}

fn resolve_kind(
    forge_bin: Option<PathBuf>,
    docker_image: String,
    mem_cap_bytes: Option<u64>,
) -> Result<ForgeBackendKind> {
    if let Some(binary) = forge_bin {
        return local_kind(binary, mem_cap_bytes);
    }
    #[cfg(target_os = "linux")]
    {
        if docker_runtime_present() {
            let cap = mem_cap_bytes.unwrap_or(0);
            tracing::info!(
                "ForgeBackend: Docker image={docker_image} mem_max={} MB",
                cap / (1024 * 1024),
            );
            return Ok(ForgeBackendKind::Docker {
                image: docker_image,
                mem_max_bytes: cap,
            });
        }
        tracing::warn!("ForgeBackend: docker not available; falling back to `forge` in $PATH");
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = docker_image;
    }
    match find_forge_on_path() {
        Some(binary) => {
            tracing::info!(
                "ForgeBackend: resolved `forge` from $PATH at {}",
                binary.display()
            );
            local_kind(binary, mem_cap_bytes)
        }
        None => Err(eyre!(
            "no usable forge: --forge-bin unset, docker runtime not available, \
             and no `forge` executable in $PATH. Install foundryup, run `docker pull \
             lazymio/knowdit:foundry`, or pass `--forge-bin <path>`."
        )),
    }
}

fn local_kind(binary: PathBuf, mem_cap_bytes: Option<u64>) -> Result<ForgeBackendKind> {
    // `Command::new(relative).current_dir(work_dir)` resolves the
    // binary relative to `work_dir`, NOT the caller's cwd — so a
    // user-supplied `--forge-bin ../foundry/target/release/forge`
    // ends up looked up under the project root (e.g.
    // `out_git/foo/../foundry/...`) and silently fails to spawn.
    // Canonicalize up front so the spawn path is unambiguous, mirroring
    // the same trick `ForgeRunner::new` already does for `work_dir`.
    let binary = match std::fs::canonicalize(&binary) {
        Ok(p) => p,
        Err(err) => {
            return Err(eyre!(
                "--forge-bin {} cannot be canonicalized ({err:#}). \
                 Pass an existing path; relative paths are interpreted \
                 against your shell's cwd at the moment knowdit was launched.",
                binary.display()
            ));
        }
    };
    #[cfg(target_os = "linux")]
    {
        if let Some(cap) = mem_cap_bytes {
            if let Some(parent) = ScopedCgroup::probe_parent() {
                tracing::info!(
                    "ForgeBackend: LocalCgroup binary={} parent={parent} mem_max={} MB",
                    binary.display(),
                    cap / (1024 * 1024)
                );
                return Ok(ForgeBackendKind::LocalCgroup {
                    binary,
                    cgroup_parent: parent,
                    mem_max_bytes: cap,
                });
            }
            tracing::warn!(
                "ForgeBackend: --forge-mem-cap-gb requested but no usable cgroup v2 \
                 parent found; falling back to direct exec without a memory ceiling"
            );
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        if mem_cap_bytes.is_some() {
            tracing::warn!("ForgeBackend: --forge-mem-cap-gb is Linux-only; ignoring on this OS");
        }
    }
    tracing::info!("ForgeBackend: LocalDirect binary={}", binary.display());
    Ok(ForgeBackendKind::LocalDirect { binary })
}

/// Probe the resolved backend for optional `forge coverage` flags.
/// Currently we only care about `--force-via-ir`; if missing, callers
/// that want via-IR coverage must either upgrade their forge or
/// accept that coverage will not compile their project.
///
/// Probing runs `forge coverage --help` through the same backend the
/// fuzz pass would use (host binary direct, or
/// `docker run <image> forge coverage --help`), so a docker image
/// that lacks the flag is detected the same way as a stale host
/// binary.
fn probe_features(kind: &ForgeBackendKind) -> ForgeFeatures {
    let help = coverage_help_output(kind).unwrap_or_default();
    let coverage_force_via_ir = help.contains("--force-via-ir");
    if !coverage_force_via_ir {
        tracing::warn!(
            "ForgeBackend: `forge coverage` does not support `--force-via-ir` on this binary. \
             Coverage runs cannot use viaIR; via-IR-only projects may fail to compile under \
             `forge coverage`. Upgrade foundry or pull a newer `--forge-docker-image`."
        );
    } else {
        tracing::info!("ForgeBackend: `forge coverage --force-via-ir` supported");
    }
    ForgeFeatures {
        coverage_force_via_ir,
    }
}

/// Best-effort capture of `forge coverage --help`. Returns `None`
/// on any failure (probe is non-fatal — features default to off).
fn coverage_help_output(kind: &ForgeBackendKind) -> Option<String> {
    use std::process::{Command, Stdio};
    let mut cmd = match kind {
        ForgeBackendKind::LocalDirect { binary } => {
            let mut c = Command::new(binary);
            c.args(["coverage", "--help"]);
            c
        }
        #[cfg(target_os = "linux")]
        ForgeBackendKind::LocalCgroup { binary, .. } => {
            let mut c = Command::new(binary);
            c.args(["coverage", "--help"]);
            c
        }
        #[cfg(target_os = "linux")]
        ForgeBackendKind::Docker { image, .. } => {
            let mut c = Command::new("docker");
            c.args([
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "/usr/local/bin/forge",
                image,
                "coverage",
                "--help",
            ]);
            c
        }
    };
    let out = cmd.stdin(Stdio::null()).output().ok()?;
    let mut combined = String::from_utf8_lossy(&out.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&out.stderr));
    Some(combined)
}

/// True iff `docker --version` exits successfully. Cheap; used by
/// [`ForgeBackend::resolve`] to decide between docker mode and the
/// `$PATH` forge fallback.
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

/// First executable named `forge` found by walking `$PATH`. Mirrors
/// what `Command::new("forge")` would resolve at exec time but
/// returns the concrete path so we can log it + (on Linux) take the
/// cgroup-wrapping branch.
fn find_forge_on_path() -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join("forge");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Owns the chosen backend + per-invocation defaults. Splits forge
/// invocations into `Test` (yields counter-example + trace) and `Coverage`
/// (yields lcov rows).
#[derive(Debug, Clone)]
pub struct ForgeRunner {
    backend: ForgeBackend,
    /// Working directory — the directory that contains `foundry.toml`.
    /// Forge resolves remappings + sources from here. In docker mode it's
    /// also the bind-mount source AND target (we mount the same path on
    /// both sides so foundry.toml + remappings + error-message paths all
    /// match the host).
    work_dir: PathBuf,
    timeout: Duration,
    /// Optional override for the project's `test = '...'` foundry directive,
    /// passed through `FOUNDRY_TEST=<path>` so forge only walks one
    /// subdirectory and its dependency closure. Lets us isolate concurrent
    /// per-spec harnesses from each other's compile errors.
    test_dir_override: Option<PathBuf>,
    /// Override for the extra time a caller waits on [`FORGE_LOCK`].
    /// `None` uses [`FORGE_LOCK_GRACE`]; tests shrink it so the bounded-wait
    /// behaviour is assertable without sleeping a minute.
    lock_grace: Option<Duration>,
}

impl ForgeRunner {
    pub fn new(backend: ForgeBackend, work_dir: PathBuf, timeout: Duration) -> Self {
        // Docker rejects relative `-v`/`-w` paths ("the working directory
        // '...' is invalid, it needs to be an absolute path"), and a
        // relative `work_dir` is brittle for the local backends too once
        // anything in the runtime changes its CWD. Resolve to an absolute
        // path up front: prefer canonicalize (drops `..`, resolves
        // symlinks) and fall back to `std::path::absolute` if the path
        // doesn't exist yet.
        let work_dir = match std::fs::canonicalize(&work_dir) {
            Ok(p) => p,
            Err(_) => match std::path::absolute(&work_dir) {
                Ok(p) => p,
                Err(err) => {
                    tracing::warn!(
                        "ForgeRunner: failed to make work_dir {} absolute ({err:#}); \
                         the docker backend will likely reject this",
                        work_dir.display(),
                    );
                    work_dir
                }
            },
        };
        Self {
            backend,
            work_dir,
            timeout,
            test_dir_override: None,
            lock_grace: None,
        }
    }

    /// Return a copy of `self` with `FOUNDRY_TEST` set to the given path
    /// for every forge invocation. Path may be absolute or relative to
    /// `work_dir`.
    pub fn with_test_dir(mut self, test_dir: PathBuf) -> Self {
        self.test_dir_override = Some(test_dir);
        self
    }

    /// The working directory forge runs in. Callers (notably the
    /// coverage-result reader) need this to find lcov.info on disk.
    pub fn work_dir(&self) -> &std::path::Path {
        &self.work_dir
    }

    /// Expose the probed feature set so per-spec tools can decide
    /// whether to pass optional flags like `--force-via-ir`.
    pub fn features(&self) -> &ForgeFeatures {
        &self.backend.features
    }

    /// Monotonic-ish nanosecond stamp for unique cgroup / container
    /// names. Wallclock-based; collisions only happen if two invocations
    /// land in the same nanosecond on the same `pid`, which is fine
    /// because the names also include `std::process::id()`.
    fn nanos_now() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    }

    /// Read effective uid/gid by parsing `/proc/self/status`. Only
    /// called from the docker spawn path so this is Linux-only. Falls
    /// back to `(1000, 1000)` if the file can't be read or parsed —
    /// matches the default uid baked into `lazymio/knowdit:foundry`.
    #[cfg(target_os = "linux")]
    fn read_proc_uid_gid() -> (u32, u32) {
        let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let mut uid: u32 = 1000;
        let mut gid: u32 = 1000;
        for line in s.lines() {
            // Format: `Uid:\t<real>\t<effective>\t<saved>\t<filesystem>`
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

    /// Spawn `forge args...` inside an isolated build slot from
    /// [`ForgeSlotPool`] and return [`ForgeOutput`]. Hard-bounded by
    /// `self.timeout` — if exceeded the handle is dropped and the
    /// per-backend cleanup fires (kill child / docker kill / cgroup.kill)
    /// while [`ForgeSlotGuard`] returns the slot index on drop.
    pub async fn run(&self, args: &[String]) -> Result<ForgeOutput> {
        let grace = self.lock_grace.unwrap_or(FORGE_LOCK_GRACE);
        let lock_timeout = self.timeout.saturating_add(grace);
        let slot_guard = self.backend.slot_pool.acquire(lock_timeout).await?;

        let mut effective_args: Vec<String> = args.to_vec();
        let uses_build_artifacts = matches!(
            args.first().map(String::as_str),
            Some("test" | "coverage" | "build" | "inspect" | "script")
        );
        let has_explicit_out = args.iter().any(|a| a == "--out" || a == "--cache-path");
        if uses_build_artifacts && !has_explicit_out {
            let slot_paths = self
                .backend
                .slot_pool
                .ensure_slot_paths(&self.work_dir, slot_guard.slot_idx())
                .await?;
            effective_args.push("--out".to_string());
            effective_args.push(slot_paths.rel_out);
            effective_args.push("--cache-path".to_string());
            effective_args.push(slot_paths.rel_cache);
        }

        let start = Instant::now();
        let handle = self.spawn(&effective_args)?;
        let timeout = self.timeout;
        let outcome = tokio::time::timeout(timeout, handle.wait_with_output()).await;
        let duration_ms = start.elapsed().as_millis() as i64;
        drop(slot_guard);
        match outcome {
            Ok(Ok(out)) => Ok(ForgeOutput {
                exit_code: out.status.code().unwrap_or(-2),
                // Take ownership of the bytes instead of copying them: with
                // `-vvvvvv` a single test run's stdout can be hundreds of MB,
                // and a lossy copy would briefly hold two of them.
                stdout: bytes_into_string(out.stdout),
                stderr: bytes_into_string(out.stderr),
                duration_ms,
                timed_out: false,
            }),
            Ok(Err(io_err)) => Err(io_err).wrap_err("forge wait_with_output failed"),
            Err(_elapsed) => {
                // The child is killed by `kill_on_drop` when `handle` drops at
                // the end of this arm. Say so in the output, because a timed-out
                // forge is often the machine's OOM killer taking the child (and
                // its solc) rather than forge genuinely running long — the
                // caller reading this line cannot tell otherwise.
                Ok(ForgeOutput {
                    exit_code: -1,
                    stdout: String::new(),
                    stderr: format!(
                        "knowdit: forge timed out after {} seconds and was killed. This is \
                         usually a very slow `via_ir` compile, or the OOM killer taking forge / \
                         solc. Check `dmesg` for 'Killed process' and consider a smaller \
                         --forge-test-runs or dropping `via_ir` in foundry.toml.",
                        timeout.as_secs()
                    ),
                    duration_ms,
                    timed_out: true,
                })
            }
        }
    }

    /// Smoke-test the forge environment with a trivial harness that
    /// just `emit log_string("ok")` after `import "forge-std/Test.sol"`.
    /// On success, the project's Foundry environment (foundry.toml,
    /// remappings, forge-std availability, solc version, docker
    /// bind-mounts) is wired up well enough for the LLM-driven fuzz
    /// agent to make progress.
    ///
    /// On failure, returns `Err` with forge's own stdout + stderr
    /// embedded so the user can read the actual compile error rather
    /// than burn LLM tokens chasing a phantom "No tests found".
    /// Cleans up the temp file regardless of outcome.
    ///
    /// `harness_dir` is where the smoke test lives; it must be inside
    /// (or equal to) [`Self::work_dir`] so docker bind-mounts and
    /// `--match-path` agree. Same `forge_test_runs=1` flag — this
    /// isn't a fuzz, just a discoverability + compile check.
    pub async fn preflight(&self, harness_dir: &Path) -> Result<()> {
        const PREFLIGHT_DIR_PREFIX: &str = "_knowdit_preflight";
        const PREFLIGHT_HARNESS: &str = r#"// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

import "forge-std/Test.sol";

contract KnowditPreflight is Test {
    function test_knowdit_preflight() public {
        emit log_string("knowdit preflight: ok");
    }
}
"#;

        let preflight_id = PREFLIGHT_COUNTER.fetch_add(1, Ordering::Relaxed);
        let preflight_dir = harness_dir.join(format!(
            "{PREFLIGHT_DIR_PREFIX}_{}_{}_{}",
            std::process::id(),
            Self::nanos_now(),
            preflight_id
        ));
        tokio::fs::create_dir_all(&preflight_dir)
            .await
            .wrap_err_with(|| {
                format!(
                    "preflight: failed to create {}; harness_dir may not be writable",
                    preflight_dir.display()
                )
            })?;
        let test_path = preflight_dir.join("KnowditPreflight.t.sol");
        tokio::fs::write(&test_path, PREFLIGHT_HARNESS)
            .await
            .wrap_err_with(|| format!("preflight: failed to write {}", test_path.display()))?;

        // Seed `<work_dir>/.knowdit-forge-slots/template/{out,cache}` during
        // preflight so every lazily-initialized worker slot (`slot_0..slot_K`)
        // starts with a warm solc artifact + cache snapshot instead of paying
        // cold compilation per slot.
        let slots_root = self.work_dir.join(FORGE_SLOTS_DIR);
        let template_out = slots_root.join("template").join("out");
        let template_cache = slots_root.join("template").join("cache");
        tokio::fs::create_dir_all(&template_out)
            .await
            .wrap_err_with(|| {
                format!(
                    "preflight: failed to create template out dir {}",
                    template_out.display()
                )
            })?;
        tokio::fs::create_dir_all(&template_cache)
            .await
            .wrap_err_with(|| {
                format!(
                    "preflight: failed to create template cache dir {}",
                    template_cache.display()
                )
            })?;
        self.backend.slot_pool.reset_work_dir(&self.work_dir).await;

        // Reuse the same FOUNDRY_TEST + --match-path scoping the per-spec
        // runner uses, so this smoke test doesn't drag any existing
        // harness file into its compile graph.
        let canonical_pre =
            std::fs::canonicalize(&preflight_dir).unwrap_or_else(|_| preflight_dir.clone());
        let pre_rel = canonical_pre
            .strip_prefix(self.work_dir())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|_| canonical_pre.clone());
        let scoped = self.clone().with_test_dir(preflight_dir.clone());
        let argv: Vec<String> = vec![
            "test".to_string(),
            "--match-contract".to_string(),
            "KnowditPreflight".to_string(),
            "--match-path".to_string(),
            format!("{}/*", pre_rel.to_string_lossy().trim_end_matches('/')),
            "--fuzz-runs".to_string(),
            "1".to_string(),
            "--out".to_string(),
            format!("{FORGE_SLOTS_DIR}/template/out"),
            "--cache-path".to_string(),
            format!("{FORGE_SLOTS_DIR}/template/cache"),
        ];

        let out_result = scoped.run(&argv).await;

        // Clean up regardless — leaving the file around lets stale
        // state poison the next preflight (different solc / shim).
        if let Err(e) = tokio::fs::remove_dir_all(&preflight_dir).await {
            tracing::warn!(
                preflight_dir = %preflight_dir.display(),
                "preflight cleanup failed (non-fatal): {e}"
            );
        }

        let out = out_result
            .wrap_err("preflight: forge spawn failed (binary path / docker / cgroup issue?)")?;
        let combined_tail = format!(
            "{}{}",
            tail_utf8(&out.stdout, 4_000),
            tail_utf8(&out.stderr, 4_000),
        );
        let looks_passed = combined_tail.contains("[PASS] test_knowdit_preflight")
            || combined_tail.contains("1 passed; 0 failed")
            || combined_tail.contains("1 tests passed");
        if out.exit_code == 0 && looks_passed {
            tracing::info!(
                duration_ms = out.duration_ms,
                "preflight: forge environment OK"
            );
            return Ok(());
        }

        // A timeout is not the same failure as a broken environment: the
        // smoke test never got to report either way, and the usual cause is
        // a cold or via_ir compile outrunning the budget rather than a
        // misconfigured project. Saying "environment is broken" here sends
        // people to reinstall forge-std for a problem that raising
        // --forge-timeout-secs fixes.
        if out.timed_out {
            return Err(eyre!(
                "preflight: forge did not finish within {}s — the smoke test was killed before it \
                 could report. Its output is truncated or empty because of the kill. Most often the \
                 compile itself is the cost, not a broken project: `via_ir = true` in foundry.toml, \
                 a large dependency tree, or a cold solc/docker cache can take minutes inside solc \
                 before the test runs. Raise `--forge-timeout-secs` (the preflight uses it as its \
                 budget, 60s floor) and re-run, or warm the cache once with `forge build` at the \
                 work_dir.\n\
                 forge exit_code={} timed_out={} duration_ms={}\n\
                 --- forge output captured before the kill (last ~4K of stdout + stderr) ---\n{}",
                self.timeout.as_secs(),
                out.exit_code,
                out.timed_out,
                out.duration_ms,
                combined_tail,
            ));
        }

        Err(eyre!(
            "preflight: forge environment is broken — the trivial smoke test \
             `import \"forge-std/Test.sol\"; emit log_string(\"ok\")` did NOT pass. \
             Common causes: forge-std missing under lib/, foundry.toml absent or \
             pinning a solc version no installed solc satisfies, --match-path \
             resolving outside the project, or docker bind-mount misconfigured. \
             Fix the environment (autobuild.py handles most Hardhat→Foundry shim \
             setup) and re-run.\n\
             forge exit_code={} timed_out={} duration_ms={}\n\
             --- forge output (last ~4K of stdout + stderr) ---\n{}",
            out.exit_code,
            out.timed_out,
            out.duration_ms,
            combined_tail,
        ))
    }

    /// Spawn the forge invocation and return a [`ForgeHandle`] whose
    /// `Drop` performs backend-specific cleanup.
    fn spawn(&self, args: &[String]) -> Result<ForgeHandle> {
        let test_env = self
            .test_dir_override
            .as_ref()
            .map(|p| ("FOUNDRY_TEST", p.clone()));

        match &self.backend.kind {
            ForgeBackendKind::LocalDirect { binary } => {
                let mut cmd = Command::new(binary);
                cmd.args(args)
                    .current_dir(&self.work_dir)
                    .kill_on_drop(true);
                if let Some((k, v)) = &test_env {
                    cmd.env(k, v);
                }
                let child = cmd
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .wrap_err_with(|| {
                        format!("failed to spawn `{} {:?}`", binary.display(), args)
                    })?;
                Ok(ForgeHandle::LocalDirect { child: Some(child) })
            }

            #[cfg(target_os = "linux")]
            ForgeBackendKind::LocalCgroup {
                binary,
                cgroup_parent,
                mem_max_bytes,
            } => {
                let cg_name = format!("knowdit-forge-{}-{}", std::process::id(), Self::nanos_now());
                let scoped = ScopedCgroup::create(cgroup_parent, &cg_name, *mem_max_bytes)
                    .map_err(|e| eyre!("failed to create cgroup {cgroup_parent}/{cg_name}: {e}"))?;

                let mut cmd = Command::new(binary);
                cmd.args(args)
                    .current_dir(&self.work_dir)
                    .kill_on_drop(true);
                if let Some((k, v)) = &test_env {
                    cmd.env(k, v);
                }
                let child = cmd
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .wrap_err_with(|| {
                        format!("failed to spawn `{} {:?}`", binary.display(), args)
                    })?;
                if let Some(pid) = child.id()
                    && let Err(e) = scoped.add_pid(pid)
                {
                    tracing::warn!(
                        "ForgeRunner: failed to attach pid {pid} to cgroup; \
                             memory cap won't apply: {e}"
                    );
                }
                Ok(ForgeHandle::LocalCgroup {
                    child: Some(child),
                    _cgroup: scoped,
                })
            }

            #[cfg(target_os = "linux")]
            ForgeBackendKind::Docker {
                image,
                mem_max_bytes,
            } => {
                let container_name =
                    format!("knowdit-forge-{}-{}", std::process::id(), Self::nanos_now());
                let (uid, gid) = Self::read_proc_uid_gid();
                let work_str = self.work_dir.display().to_string();
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
                    .arg(format!("{work_str}:{work_str}"))
                    .arg("-w")
                    .arg(&work_str);
                if *mem_max_bytes > 0 {
                    cmd.arg(format!("--memory={mem_max_bytes}b"))
                        .arg(format!("--memory-swap={mem_max_bytes}b"));
                }
                if let Some((k, v)) = &test_env {
                    cmd.arg("-e").arg(format!("{k}={}", v.display()));
                }
                cmd.arg("--entrypoint")
                    .arg("/usr/local/bin/forge")
                    .arg(image)
                    .args(args)
                    .kill_on_drop(true);
                let child = cmd
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .wrap_err_with(|| format!("failed to spawn `docker run ... {image}`"))?;
                Ok(ForgeHandle::Docker {
                    child: Some(child),
                    container_name,
                })
            }
        }
    }
}

/// Per-invocation handle. Holds the child + any backend-specific cleanup
/// state. `Drop` kills the right things in the right order.
///
/// `child` is wrapped in `Option` so [`Self::wait_with_output`] can
/// `take()` it without moving out of `self` (which `Drop`-implementing
/// types forbid). After a successful wait the child slot is `None`, so
/// the per-variant Drop logic still runs but has nothing left to kill —
/// `docker kill` against an already-exited container is a harmless no-op,
/// and the cgroup is already empty so `cgroup.kill` + rmdir succeed
/// immediately.
#[derive(Debug)]
enum ForgeHandle {
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

impl ForgeHandle {
    fn take_child(&mut self) -> Option<tokio::process::Child> {
        match self {
            ForgeHandle::LocalDirect { child } => child.take(),
            #[cfg(target_os = "linux")]
            ForgeHandle::LocalCgroup { child, .. } => child.take(),
            #[cfg(target_os = "linux")]
            ForgeHandle::Docker { child, .. } => child.take(),
        }
    }

    /// Block until the child exits and collect stdout/stderr. `Drop` runs
    /// when this future resolves (or is cancelled), so backend cleanup
    /// fires on every path.
    async fn wait_with_output(mut self) -> std::io::Result<std::process::Output> {
        let child = self
            .take_child()
            .expect("ForgeHandle::wait_with_output: child already consumed");
        child.wait_with_output().await
    }
}

impl Drop for ForgeHandle {
    fn drop(&mut self) {
        match self {
            ForgeHandle::LocalDirect { .. } => {
                // Child (if still here) gets SIGKILL'd via tokio
                // `kill_on_drop`. Nothing else to do.
            }
            #[cfg(target_os = "linux")]
            ForgeHandle::LocalCgroup { .. } => {
                // Field-order drop: `child` (kill_on_drop SIGKILL), then
                // `_cgroup` (cgroup.kill + retry rmdir). Already in
                // declaration order; no manual sequencing needed.
            }
            #[cfg(target_os = "linux")]
            ForgeHandle::Docker { container_name, .. } => {
                // Killing the docker CLI client (tokio kill_on_drop) does
                // NOT reliably propagate to the daemon-managed container,
                // so explicitly tell dockerd to kill it. SIGKILL is
                // `docker kill`'s default. Best-effort, fire-and-forget.
                let _ = std::process::Command::new("docker")
                    .args(["kill", container_name.as_str()])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn();
            }
        }
    }
}

/// Raw subprocess outcome before parsing.
#[derive(Debug, Clone)]
pub struct ForgeOutput {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: i64,
    pub timed_out: bool,
}

#[cfg(test)]
mod lock_timeout_tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::time::{Duration, Instant};

    fn local_runner(
        work_dir: std::path::PathBuf,
        binary: std::path::PathBuf,
        concurrency: usize,
        timeout: Duration,
    ) -> ForgeRunner {
        let mut runner = ForgeRunner::new(
            ForgeBackend {
                kind: ForgeBackendKind::LocalDirect { binary },
                features: ForgeFeatures::default(),
                slot_pool: Arc::new(ForgeSlotPool::new(concurrency)),
            },
            work_dir,
            timeout,
        );
        runner.lock_grace = Some(Duration::from_millis(100));
        runner
    }

    #[tokio::test]
    async fn waiting_for_a_wedged_lock_times_out() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let runner = local_runner(
            dir.path().to_path_buf(),
            std::path::PathBuf::from("/bin/true"),
            1,
            Duration::from_millis(50),
        );

        // Hold the sole slot so `runner.run` sees a wedged pool.
        let _held = runner
            .backend
            .slot_pool
            .acquire(Duration::from_secs(1))
            .await?;

        let started = Instant::now();
        let err = runner
            .run(&["--version".to_string()])
            .await
            .expect_err("must not hang, and must not succeed without a slot");
        let waited = started.elapsed();

        assert!(
            err.to_string().contains("waiting for the shared forge slot pool"),
            "unexpected error: {err}"
        );
        assert!(
            waited < Duration::from_secs(5),
            "slot pool wait was not bounded: {waited:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn slot_pool_returns_slots_on_task_abort_and_panic() -> Result<()> {
        let pool = Arc::new(ForgeSlotPool::new(2));

        // 1. Task aborted while holding slot 0.
        let pool_abort = pool.clone();
        let aborted = tokio::spawn(async move {
            let _guard = pool_abort.acquire(Duration::from_secs(1)).await;
            tokio::time::sleep(Duration::from_secs(60)).await;
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        aborted.abort();
        let _ = aborted.await;

        // 2. Task panicking while holding a slot.
        let pool_panic = pool.clone();
        let panicked = tokio::spawn(async move {
            let _guard = pool_panic.acquire(Duration::from_secs(1)).await;
            panic!("simulated worker panic");
        });
        let _ = panicked.await;

        // Both slots must be back in the pool and immediately acquirable!
        let g1 = pool.acquire(Duration::from_millis(100)).await?;
        let g2 = pool.acquire(Duration::from_millis(100)).await?;
        assert_ne!(g1.slot_idx(), g2.slot_idx());
        Ok(())
    }

    #[tokio::test]
    async fn slot_pool_clones_template_and_issues_distinct_cli_dirs_concurrently() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let work_dir = dir.path().to_path_buf();

        // Seed `.knowdit-forge-slots/template/{out,cache}` to verify lazy slot cloning.
        let tpl_out = work_dir.join(FORGE_SLOTS_DIR).join("template").join("out");
        let tpl_cache = work_dir.join(FORGE_SLOTS_DIR).join("template").join("cache");
        tokio::fs::create_dir_all(&tpl_out).await?;
        tokio::fs::create_dir_all(&tpl_cache).await?;
        tokio::fs::write(tpl_out.join("Seeded.json"), b"{\"abi\":[]}").await?;
        tokio::fs::write(tpl_cache.join("solidity-files-cache.json"), b"{}").await?;

        // Fake `forge` script that sleeps briefly (so 2 invocations overlap) and echoes argv.
        let fake_forge = work_dir.join("fake_forge.sh");
        tokio::fs::write(
            &fake_forge,
            "#!/usr/bin/env bash\nsleep 0.05\nprintf '%s\\n' \"$@\"\n",
        )
        .await?;
        let mut perms = tokio::fs::metadata(&fake_forge).await?.permissions();
        perms.set_mode(0o755);
        tokio::fs::set_permissions(&fake_forge, perms).await?;

        let runner = local_runner(work_dir.clone(), fake_forge, 2, Duration::from_secs(5));
        let r1 = runner.clone();
        let r2 = runner.clone();
        let argv_a = vec![
            "test".to_string(),
            "--match-contract".to_string(),
            "A".to_string(),
        ];
        let argv_b = vec![
            "test".to_string(),
            "--match-contract".to_string(),
            "B".to_string(),
        ];
        let (o1, o2) = tokio::try_join!(r1.run(&argv_a), r2.run(&argv_b))?;
        assert_eq!(o1.exit_code, 0);
        assert_eq!(o2.exit_code, 0);

        let combined = format!("{}\n{}", o1.stdout, o2.stdout);
        assert!(
            combined.contains(".knowdit-forge-slots/slot_0/out")
                && combined.contains(".knowdit-forge-slots/slot_0/cache"),
            "slot_0 flags missing:\n{combined}"
        );
        assert!(
            combined.contains(".knowdit-forge-slots/slot_1/out")
                && combined.contains(".knowdit-forge-slots/slot_1/cache"),
            "slot_1 flags missing:\n{combined}"
        );
        // Both slots must carry the cloned template artifacts at slot_{i}/out/Seeded.json!
        assert!(
            work_dir
                .join(FORGE_SLOTS_DIR)
                .join("slot_0")
                .join("out")
                .join("Seeded.json")
                .is_file()
        );
        assert!(
            work_dir
                .join(FORGE_SLOTS_DIR)
                .join("slot_1")
                .join("out")
                .join("Seeded.json")
                .is_file()
        );
        Ok(())
    }
}
