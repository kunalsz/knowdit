//! Foundry / fuzz-agent runtime knobs shared by `agentic fuzz`,
//! `agentic regen`, and `workflow autoloop` (which drives both).
//! The same `SolidityFuzzGenerator` underpins all three, so the
//! forge-backend config (binary / docker image / memory cap), the
//! per-spec LLM-agent budget, and the Gate 2 inline-fidelity
//! threshold are a single struct flattened wherever needed.
//!
//! Long-flag names carry a phase-disambiguating prefix
//! (`--harness-*` / `--forge-*` / `--gate2-*`) so flattening
//! [`HarnessSharedArgs`] into the autoloop alongside other phase
//! `SharedArgs` doesn't trip clap's duplicate-arg check. The CLI
//! standalone commands inherit the same names — there is no
//! pre-rename compatibility layer.

use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Args;
use color_eyre::eyre::{Result, eyre};
use knowdit_audit::harness::forge::{ForgeBackend, ForgeRunner};
use knowdit_audit::harness::solana::{SolanaBackend, SolanaFuzzOptions, SolanaHarness};
use knowdit_audit::harness::solidity::{FuzzOptions, HarnessKind, SolidityHarness};
use knowdit_repo_model::SourceLanguage;

/// CLI mirror of [`HarnessKind`] (kebab-cased on the command line).
/// Two-variant `clap::ValueEnum` that we map to the audit-crate enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum HarnessModeCli {
    /// Handler+invariant: Foundry handler contract + invariant test;
    /// fuzzer drives input sequences (existing default).
    Handler,
    /// Deterministic PoC: agent writes a `test_…()` that reproduces
    /// the attack deterministically and asserts post-attack with
    /// `KNOWDIT_POST_ATTACK_REACHED:`.
    Poc,
}

impl From<HarnessModeCli> for HarnessKind {
    fn from(v: HarnessModeCli) -> Self {
        match v {
            HarnessModeCli::Handler => Self::Handler,
            HarnessModeCli::Poc => Self::Poc,
        }
    }
}

/// Forge backend + harness-codegen-agent knobs. Flattened by
/// `agentic fuzz`, `agentic regen`, and `workflow autoloop`. All
/// fields use phase-disambiguated long names so multiple consumers
/// can co-exist in one `clap::Args` parse tree.
#[derive(Args, Clone, Debug)]
pub struct HarnessSharedArgs {
    /// `llmy` cache key prefix. When unset, each subcommand falls
    /// back to a per-subcommand default
    /// (`{project}-knowdit-fuzz` / `{project}-knowdit-regen`).
    #[arg(long = "harness-cache-key")]
    pub harness_cache_key: Option<String>,

    /// Debug-prefix forwarded to llmy for LLM-call dumps.
    #[arg(long = "harness-debug-prefix")]
    pub harness_debug_prefix: Option<String>,

    /// Path to a local `forge` binary. When set, knowdit runs forge
    /// directly on the host (optionally wrapped in a cgroup v2 memory
    /// cap — see `--forge-mem-cap-gb`). When unset (default), knowdit
    /// uses `docker run --rm <docker-image> forge ...` so a
    /// memory-runaway forge can't OOM the host.
    #[arg(short = 'b', long = "forge-bin")]
    pub forge_bin: Option<PathBuf>,

    /// Docker image used when `--forge-bin` is not provided. The image
    /// must have `forge` installed at `/usr/local/bin/forge` and must
    /// run as a uid that matches the host (the default
    /// `lazymio/knowdit:foundry` runs uid 1000).
    #[arg(long = "forge-docker-image", default_value = "lazymio/knowdit:foundry")]
    pub docker_image: String,

    /// Per-invocation memory ceiling, in GiB. In docker mode this is
    /// plumbed through `--memory` / `--memory-swap`. In local mode
    /// this is enforced via cgroup v2 (when a writable delegated
    /// parent is available); if no usable cgroup is found we log a
    /// warning and fall back to direct exec. Set to 0 to disable the
    /// cap entirely.
    #[arg(long = "forge-mem-cap-gb", default_value_t = 12)]
    pub forge_mem_cap_gb: u64,

    /// Where to write `.sol` harness files. If omitted: auto-detect —
    /// if the repo has `foundry.toml` + `test/`, write to
    /// `<repo>/test/knowdit_harness/`, otherwise `<repo>/knowdit_harness/`.
    #[arg(long = "harness-dir")]
    pub harness_dir: Option<PathBuf>,

    /// Hard timeout per forge subprocess (seconds).
    #[arg(long = "forge-timeout-secs", default_value_t = 600)]
    pub forge_timeout_secs: u64,

    /// `--fuzz-runs` for the `forge test` invocation.
    #[arg(long = "forge-test-runs", default_value_t = 100_000)]
    pub forge_test_runs: u64,

    /// `--fuzz-runs` for the `forge coverage` invocation. Coverage is
    /// much slower per sample, so 256 runs suffices for line-coverage
    /// fidelity checks while avoiding redundant 5,000-run interpreter spend.
    #[arg(long = "forge-coverage-runs", default_value_t = 256)]
    pub forge_coverage_runs: u64,

    /// Number of isolated Foundry build slots (`--out` / `--cache-path`)
    /// allowed to run `forge` concurrently. In `workflow streamloop`,
    /// `--default-concurrency` populates this (capped at `8` when inferred);
    /// standalone `fuzz`/`regen` fall back to `--fuzz-concurrency` /
    /// `--regen-concurrency`.
    #[arg(long = "forge-concurrency")]
    pub forge_concurrency: Option<usize>,

    /// Maximum agent steps per spec across all restarts. Cumulative
    /// budget — restarts reuse the same step counter so a model that
    /// thrashes can't reset its budget by re-opening.
    #[arg(long = "harness-max-agent-steps", default_value_t = 120)]
    pub harness_max_agent_steps: usize,

    /// Approximate token count at which the orchestrator restarts the
    /// agent. Defaults to None (no restart; rely on step budget). Set
    /// this to ~80% of the model's input window to avoid hitting
    /// hard context limits during long iterations.
    #[arg(long = "harness-window-restart-threshold-tokens")]
    pub harness_window_restart_threshold_tokens: Option<usize>,

    /// Maximum number of agent restarts per spec before the spec is
    /// abandoned. Each restart drops conversation history but keeps
    /// the harness file on disk plus a short "previous attempt" note
    /// in the new system prompt so the new agent doesn't restart from
    /// scratch.
    #[arg(long = "harness-max-restarts", default_value_t = 3)]
    pub harness_max_restarts: usize,

    /// Gate 2 fidelity threshold for the inline coverage gate. The
    /// agent gets `[GATE 2 FAILED — coverage]` in its `run_forge`
    /// tool result when `hit_fns / expected_fns < this`. Same default
    /// as `agentic reflect`.
    #[arg(long = "gate2-fidelity-threshold", default_value_t = 0.5)]
    pub gate2_fidelity_threshold: f64,

    /// Tell the harness pipeline that the project's solc build uses
    /// viaIR. Plumbed into the per-spec prompt (so the agent knows
    /// coverage might fail) and into `forge coverage --force-via-ir`
    /// when the resolved forge backend supports the flag. The
    /// autoloop orchestrator overrides this from its top-level
    /// `--via-ir` so the user doesn't have to pass it twice.
    #[arg(long = "harness-via-ir", default_value_t = false)]
    pub harness_via_ir: bool,

    /// Pick which test-authoring mode the harness agent runs.
    /// `handler` (default) is the Foundry handler + invariant pipeline;
    /// `poc` writes a deterministic `test_…()` that asserts with
    /// `KNOWDIT_POST_ATTACK_REACHED:`. The two modes are not mixed —
    /// each invocation is one or the other.
    #[arg(short = 'm', long = "harness-mode", value_enum, default_value_t = HarnessModeCli::Handler)]
    pub harness_mode: HarnessModeCli,

    /// Recursion budget for the `list_test_files` agent tool when it
    /// scans the project's `test/` / `script/` / etc. directories.
    /// 0 disables the scan entirely.
    #[arg(long = "harness-list-test-files-max-depth", default_value_t = 6)]
    pub harness_list_test_files_max_depth: usize,
}

/// Per-subcommand fields [`HarnessSharedArgs`] doesn't know about
/// (project root, fuzz-only caps, default cache key, project viaIR
/// state). The caller fills this in;
/// [`HarnessSharedArgs::to_fuzz_options`] consumes it.
#[derive(Debug, Clone)]
pub struct FuzzOptionsBuild {
    pub repo_root: PathBuf,
    /// Used when `HarnessSharedArgs::cache_key` is `None`. Subcommands
    /// pass `{project}-knowdit-fuzz` or `{project}-knowdit-regen`.
    pub default_cache_key: String,
    pub max_specs: usize,
    pub concurrency: usize,
    pub regenerate: bool,
    /// Whether the project's solc settings use viaIR. Plumbs into
    /// `forge coverage --force-via-ir` when the backend supports it
    /// and into the per-spec prompt note when it doesn't. Standalone
    /// `agentic fuzz` / `agentic regen` get this from the project
    /// foundry profile via `--harness-via-ir`; the autoloop
    /// orchestrator overrides it from `--via-ir` on the foundry
    /// flatten group.
    pub via_ir: bool,
}

impl HarnessSharedArgs {
    /// Resolve a [`ForgeBackend`] (preflight + feature probe) from
    /// these args. Callers do this exactly once per pipeline
    /// invocation and thread the result through every phase that
    /// needs a forge runtime — the per-spec [`ForgeRunner`]
    /// constructor takes `ForgeBackend` by value, so the
    /// `forge coverage --help` probe runs only once even when fuzz
    /// + regen both run inside the same autoloop cycle.
    pub fn to_forge_backend(&self) -> Result<ForgeBackend> {
        self.to_forge_backend_with_concurrency(1)
    }

    pub fn to_forge_backend_with_concurrency(&self, fallback_concurrency: usize) -> Result<ForgeBackend> {
        let mem_cap_bytes =
            (self.forge_mem_cap_gb > 0).then(|| self.forge_mem_cap_gb * 1024 * 1024 * 1024);
        let slots = self
            .forge_concurrency
            .unwrap_or(fallback_concurrency)
            .max(1);
        ForgeBackend::with_concurrency(
            self.forge_bin.clone(),
            self.docker_image.clone(),
            mem_cap_bytes,
            slots,
        )
    }

    /// Smoke-test the forge environment before any LLM-driven phase
    /// touches the project. Build a one-off [`ForgeRunner`] off
    /// `backend` (so the `forge coverage --help` probe doesn't
    /// re-fire) with a short 60s timeout, point it at the same
    /// harness dir the fuzz phase would use, and run
    /// [`ForgeRunner::preflight`] — a trivial compile + execute of
    /// `import "forge-std/Test.sol"`. Returns `Err` with forge's
    /// verbatim stdout/stderr on failure so the user sees the actual
    /// compile error rather than burning extract/map/spec-gen tokens
    /// on a project that was always going to fail.
    ///
    /// Cheap: one short forge subprocess. Designed to be called once
    /// per pipeline invocation at the top of streamloop / autoloop /
    /// standalone fuzz / standalone regen, before any DB-heavy or
    /// LLM-heavy stage runs.
    pub async fn preflight(&self, backend: &ForgeBackend, repo_root: &Path) -> Result<()> {
        // Reuse `to_fuzz_options` purely for its harness_dir resolution
        // logic (foundry.toml → test/knowdit_harness; else
        // knowdit_harness). Other field values are placeholders since
        // `resolve_harness_dir` only reads `harness_dir` + `repo_root`.
        let opts = self.to_fuzz_options(FuzzOptionsBuild {
            repo_root: repo_root.to_path_buf(),
            default_cache_key: String::new(),
            max_specs: 0,
            concurrency: 1,
            regenerate: false,
            via_ir: false,
        });
        let harness_dir = opts.resolve_harness_dir();
        tokio::fs::create_dir_all(&harness_dir).await.map_err(|e| {
            color_eyre::eyre::eyre!(
                "preflight: failed to create harness_dir {}: {e}",
                harness_dir.display()
            )
        })?;
        let harness_dir = std::fs::canonicalize(&harness_dir).unwrap_or(harness_dir);
        let forge_work_dir = if repo_root.join("foundry.toml").exists() {
            repo_root.to_path_buf()
        } else {
            harness_dir.clone()
        };
        // Preflight is a single trivial test, but the compile it triggers is
        // not trivial: projects with `via_ir = true`, large dependency trees
        // or a cold solc cache can spend minutes inside solc before the test
        // ever runs. Budget against the same per-invocation timeout the agent
        // gets, with 60s as the floor so the default profile stays fast.
        let preflight_timeout = Duration::from_secs(self.forge_timeout_secs.max(60));
        let runner = ForgeRunner::new(backend.clone(), forge_work_dir, preflight_timeout);
        runner.preflight(&harness_dir).await
    }

    /// Compose a [`SolidityHarness`] in one step: resolve the
    /// [`ForgeBackend`] (one-time `forge coverage --help` probe),
    /// fold these args into [`FuzzOptions`], bundle both with the
    /// [`HarnessKind`] mode. Single seam for "CLI args → backend
    /// handle" so orchestrators don't re-thread the underlying
    /// `(backend, options, kind)` tuple manually.
    pub fn to_solidity_harness(&self, build: FuzzOptionsBuild) -> Result<SolidityHarness> {
        let forge_backend = self.to_forge_backend_with_concurrency(build.concurrency)?;
        let options = self.to_fuzz_options(build);
        Ok(SolidityHarness::new(
            self.harness_mode.into(),
            options,
            forge_backend,
        ))
    }

    /// Compose [`FuzzOptions`] from `self` plus the per-subcommand
    /// pieces in `build`. Single source of truth for forge-runtime
    /// config — no caller spells `FuzzOptions { ... }` directly.
    pub fn to_fuzz_options(&self, build: FuzzOptionsBuild) -> FuzzOptions {
        let cache_key = self
            .harness_cache_key
            .clone()
            .unwrap_or(build.default_cache_key);
        let forge_mem_cap_bytes =
            (self.forge_mem_cap_gb > 0).then(|| self.forge_mem_cap_gb * 1024 * 1024 * 1024);
        FuzzOptions {
            repo_root: build.repo_root,
            harness_dir: self.harness_dir.clone(),
            forge_bin: self.forge_bin.clone(),
            docker_image: self.docker_image.clone(),
            forge_mem_cap_bytes,
            forge_timeout_secs: self.forge_timeout_secs,
            forge_test_runs: self.forge_test_runs,
            forge_coverage_runs: self.forge_coverage_runs,
            max_agent_steps: self.harness_max_agent_steps,
            max_specs: build.max_specs,
            concurrency: build.concurrency.max(1),
            regenerate: build.regenerate,
            cache_key,
            debug_prefix: self.harness_debug_prefix.clone(),
            llm_settings: None,
            window_restart_threshold_tokens: self.harness_window_restart_threshold_tokens,
            max_restarts: self.harness_max_restarts,
            gate2_fidelity_threshold: self.gate2_fidelity_threshold,
            via_ir: build.via_ir,
            list_test_files_max_depth: self.harness_list_test_files_max_depth,
        }
    }
}

/// Resolve the audit-phase language prompt prefix for standalone CLI commands
/// (`map-semantics`, `gen-specs`, `reflect`, `review-findings`).
pub fn language_prompt_prefix_for(language: SourceLanguage) -> Result<&'static str> {
    match language {
        SourceLanguage::Solidity => Ok(SolidityHarness::PROMPT_PREFIX),
        SourceLanguage::Rust => Ok(SolanaHarness::PROMPT_PREFIX),
        SourceLanguage::Move => Err(eyre!(
            "Move target auditing is not supported in this build"
        )),
    }
}

/// Solana `LiteSVM` + SBF runtime knobs (`--solana-*`), flattened alongside
/// [`HarnessSharedArgs`] in `agentic fuzz`, `agentic regen`, and
/// `workflow streamloop` without flag collisions.
#[derive(Args, Clone, Debug)]
pub struct SolanaHarnessSharedArgs {
    /// Path to a local `cargo` binary. When set, Knowdit runs `cargo test`
    /// directly on the host (with optional cgroup v2 memory cap). When unset,
    /// Knowdit uses the locally built `knowdit:solana` Docker image if present,
    /// otherwise resolves `cargo` from `$PATH`.
    #[arg(long = "solana-cargo-bin")]
    pub solana_cargo_bin: Option<PathBuf>,

    /// Path to a local `cargo-build-sbf` binary used to compile missing SBF
    /// `.so` artifacts in local mode.
    #[arg(long = "solana-build-sbf-bin")]
    pub solana_build_sbf_bin: Option<PathBuf>,

    /// Local Docker image used for sandboxed offline `LiteSVM` + SBF builds
    /// (built from `docker/Dockerfile.solana` via `./setup.sh` or `docker build`).
    #[arg(long = "solana-docker-image", default_value = "knowdit:solana")]
    pub solana_docker_image: String,

    /// Per-invocation memory ceiling in GiB (`0` disables the ceiling).
    #[arg(long = "solana-mem-cap-gb", default_value_t = 12)]
    pub solana_mem_cap_gb: u64,

    /// Directory where per-spec `LiteSVM` harness crates and the shared
    /// `.knowdit-cargo-target` live (defaults to `<repo>/knowdit_solana_harness`).
    #[arg(long = "solana-harness-dir")]
    pub solana_harness_dir: Option<PathBuf>,

    /// Hard timeout per `cargo test` / `cargo-build-sbf` subprocess (seconds).
    #[arg(long = "solana-timeout-secs", default_value_t = 600)]
    pub solana_timeout_secs: u64,

    /// `PROPTEST_CASES` count for stateful `Handler` sequences.
    #[arg(long = "solana-test-runs", default_value_t = 256)]
    pub solana_test_runs: u64,

    /// `PROPTEST_CASES` count for `cargo llvm-cov` coverage runs.
    #[arg(long = "solana-coverage-runs", default_value_t = 64)]
    pub solana_coverage_runs: u64,
}

#[derive(Debug, Clone)]
pub struct SolanaFuzzOptionsBuild {
    pub repo_root: PathBuf,
    pub default_cache_key: String,
    pub max_specs: usize,
    pub concurrency: usize,
    pub regenerate: bool,
}

impl SolanaHarnessSharedArgs {
    pub fn to_solana_backend(&self) -> Result<SolanaBackend> {
        let mem_cap_bytes =
            (self.solana_mem_cap_gb > 0).then(|| self.solana_mem_cap_gb * 1024 * 1024 * 1024);
        SolanaBackend::new(
            self.solana_cargo_bin.clone(),
            self.solana_build_sbf_bin.clone(),
            self.solana_docker_image.clone(),
            mem_cap_bytes,
        )
    }

    pub fn to_solana_fuzz_options(
        &self,
        shared_harness: &HarnessSharedArgs,
        build: SolanaFuzzOptionsBuild,
    ) -> SolanaFuzzOptions {
        let cache_key = shared_harness
            .harness_cache_key
            .clone()
            .unwrap_or(build.default_cache_key);
        let mem_cap_bytes =
            (self.solana_mem_cap_gb > 0).then(|| self.solana_mem_cap_gb * 1024 * 1024 * 1024);
        SolanaFuzzOptions {
            repo_root: build.repo_root,
            harness_dir: self
                .solana_harness_dir
                .clone()
                .or_else(|| shared_harness.harness_dir.clone()),
            cargo_bin: self.solana_cargo_bin.clone(),
            build_sbf_bin: self.solana_build_sbf_bin.clone(),
            docker_image: self.solana_docker_image.clone(),
            mem_cap_bytes,
            timeout_secs: self.solana_timeout_secs,
            test_runs: self.solana_test_runs,
            coverage_runs: self.solana_coverage_runs,
            max_agent_steps: shared_harness.harness_max_agent_steps,
            max_specs: build.max_specs,
            concurrency: build.concurrency.max(1),
            regenerate: build.regenerate,
            cache_key,
            debug_prefix: shared_harness.harness_debug_prefix.clone(),
            llm_settings: None,
            window_restart_threshold_tokens: shared_harness.harness_window_restart_threshold_tokens,
            max_restarts: shared_harness.harness_max_restarts,
            gate2_fidelity_threshold: shared_harness.gate2_fidelity_threshold,
            list_test_files_max_depth: shared_harness.harness_list_test_files_max_depth,
        }
    }

    pub fn to_solana_harness(
        &self,
        shared_harness: &HarnessSharedArgs,
        build: SolanaFuzzOptionsBuild,
    ) -> Result<SolanaHarness> {
        let backend = self.to_solana_backend()?;
        let options = self.to_solana_fuzz_options(shared_harness, build);
        Ok(SolanaHarness::new(
            shared_harness.harness_mode.into(),
            options,
            backend,
        ))
    }
}
