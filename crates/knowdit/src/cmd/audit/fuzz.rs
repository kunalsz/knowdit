//! `agentic fuzz` subcommand: synthesize Foundry fuzzing harnesses for the
//! generated `AuditSpecification`s and run forge against them.
//!
//! Reads `specification` rows from the per-project SQLite database, runs an
//! LLM agent per spec to produce a `.sol` harness, invokes the configured
//! `forge` binary (test + coverage), and persists outcomes to `code_gen` /
//! `harness_run` / `line_coverage`. Resume-safe: a re-run skips specs whose
//! `code_gen` row is already `Completed` unless `--fuzz-regenerate` is set.

use clap::Args;
use color_eyre::eyre::{Result, WrapErr, eyre};
use knowdit_audit::harness::HarnessBackend;
use knowdit_audit::harness::solidity::FuzzOutcome;
use knowdit_repo_model::{RepoDatabase, SourceLanguage};
use llmy::client::client::LLM;

use super::fuzz_common::{
    FuzzOptionsBuild, HarnessSharedArgs, SolanaFuzzOptionsBuild, SolanaHarnessSharedArgs,
};
use crate::cli::{DatabaseArgs, LoadedRepoDatabase, ProjectArgs};

/// Fuzz-only knobs (the harness backend itself lives in
/// [`HarnessSharedArgs`], shared with `agentic regen`).
#[derive(Args, Clone, Debug)]
pub struct FuzzSharedArgs {
    /// Cap on number of specs processed in this run. `0` means no cap.
    #[arg(long = "fuzz-max-specs", default_value_t = 0)]
    pub fuzz_max_specs: usize,

    /// Number of specs processed in parallel. Defaults to `1`
    /// (strict serial) when omitted. Typed `Option` so streamloop's
    /// `--default-concurrency` can tell "user passed it" from
    /// "default kicked in" — an explicit value wins over `-d`.
    #[arg(long = "fuzz-concurrency")]
    pub fuzz_concurrency: Option<usize>,

    /// Regenerate every harness from scratch — clears `code_gen`,
    /// `harness_run`, and `line_coverage` at the start. Default is
    /// to skip any spec whose `code_gen` row is already `Completed`.
    #[arg(long = "fuzz-regenerate", default_value_t = false)]
    pub fuzz_regenerate: bool,
}

#[derive(Args)]
pub struct FuzzArgs {
    #[command(flatten)]
    pub project: ProjectArgs,

    #[command(flatten)]
    pub db: DatabaseArgs,

    /// Forge backend + per-spec agent budget + Gate 2 threshold.
    /// Shared with `agentic regen` and `workflow autoloop`.
    #[command(flatten)]
    pub harness: HarnessSharedArgs,

    /// Solana `LiteSVM` + SBF backend knobs (`--solana-*`) used when
    /// `project_metadata.profile` declares `Rust`.
    #[command(flatten)]
    pub solana_harness: SolanaHarnessSharedArgs,

    #[command(flatten)]
    pub shared: FuzzSharedArgs,
}

impl FuzzArgs {
    /// CLI entry: open the project DB, resolve the project's language backend
    /// (`SolidityHarness` or `SolanaHarness`), preflight, run [`Self::fuzz`],
    /// and print one-line summary.
    pub async fn run(self, llm: &LLM) -> Result<()> {
        let LoadedRepoDatabase { spec, repo, .. } = self
            .project
            .to_repo_database(self.db.database_path.clone(), self.db.variant_render_cap)
            .await?;
        let language = match repo.get_project_profile().await? {
            Some(p) => p.language,
            None => self.project.to_project_data().await?.language,
        };
        let outcome = match language {
            SourceLanguage::Solidity => {
                let harness_backend = self.harness.to_solidity_harness(FuzzOptionsBuild {
                    repo_root: spec.root.clone(),
                    default_cache_key: format!("{}-knowdit-fuzz", spec.name),
                    max_specs: self.shared.fuzz_max_specs,
                    concurrency: self.shared.fuzz_concurrency.unwrap_or(1).max(1),
                    regenerate: self.shared.fuzz_regenerate,
                    via_ir: self.harness.harness_via_ir,
                })?;
                harness_backend
                    .preflight(&spec.root)
                    .await
                    .wrap_err("forge environment preflight failed")?;
                Self::fuzz(&harness_backend, &repo, llm).await?
            }
            SourceLanguage::Rust => {
                let harness_backend = self.solana_harness.to_solana_harness(
                    &self.harness,
                    SolanaFuzzOptionsBuild {
                        repo_root: spec.root.clone(),
                        default_cache_key: format!("{}-knowdit-fuzz", spec.name),
                        max_specs: self.shared.fuzz_max_specs,
                        concurrency: self.shared.fuzz_concurrency.unwrap_or(1).max(1),
                        regenerate: self.shared.fuzz_regenerate,
                    },
                )?;
                harness_backend
                    .preflight(&spec.root)
                    .await
                    .wrap_err("Solana LiteSVM environment preflight failed")?;
                Self::fuzz(&harness_backend, &repo, llm).await?
            }
            SourceLanguage::Move => {
                return Err(eyre!("Move target fuzzing is not supported in this build"));
            }
        };
        println!(
            "Fuzz finished: processed={} completed={} abandoned={} error={} violated={} skipped_resumed={}",
            outcome.processed_count,
            outcome.completed_count,
            outcome.abandoned_count,
            outcome.error_count,
            outcome.violated_count,
            outcome.skipped_resumed,
        );
        Ok(())
    }

    /// In-process entry: drive the configured harness backend through
    /// a full sweep of pending specs. Generic over the backend so
    /// `workflow autoloop` and a future Move dispatch can call the
    /// same path with their own [`HarnessBackend`] impl. The
    /// already-built `harness_backend` carries the resolved forge
    /// runtime so `forge coverage --help` only probes once per
    /// pipeline invocation.
    pub async fn fuzz<B: HarnessBackend>(
        harness_backend: &B,
        repo: &RepoDatabase,
        llm: &LLM,
    ) -> Result<FuzzOutcome> {
        harness_backend.fuzz_sweep(repo, llm).await
    }
}
