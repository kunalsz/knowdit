//! `knowdit eval run` — execute a fixed-context or growth-replay
//! learning run against the suite baseline.

use crate::cmd::learn::merge_args::MergeCliArgs;
use clap::{Args, ValueEnum};
use color_eyre::eyre::{Result, bail};
use knowdit_eval::manifest::{RunConfig, RunMode};
use knowdit_eval::replay::{ReplayOptions, run as replay_run};
use knowdit_eval::sandbox::Baseline;
use llmy::clap::OpenAISetup;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct EvalRunArgs {
    #[command(flatten)]
    pub llm: OpenAISetup,

    /// Path to the suite root (contains `suite.json`).
    #[arg(long)]
    pub suite: PathBuf,

    /// Replay mode.
    #[arg(long, value_enum, default_value_t = RunModeArg::FixedContext)]
    pub mode: RunModeArg,

    /// Number of full repetitions (only meaningful for live models).
    #[arg(long, default_value_t = 1)]
    pub repetitions: usize,

    /// Run the cross-project finding link pass after admission.
    #[arg(long, default_value_t = false)]
    pub link: bool,

    /// Extraction concurrency.
    #[arg(long, default_value_t = 1)]
    pub concurrency: usize,

    /// Run output directory (manifest + artifacts land here).
    #[arg(long, default_value = "runs")]
    pub output: PathBuf,

    #[command(flatten)]
    pub merge: MergeCliArgs,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum RunModeArg {
    FixedContext,
    GrowthReplay,
}

impl EvalRunArgs {
    pub async fn run(self) -> Result<()> {
        self.merge.validate()?;
        if self.repetitions == 0 {
            bail!("repetitions must be at least 1");
        }

        let suite = knowdit_eval::corpus::LoadedSuite::load(&self.suite)?;
        let baseline = Baseline::load(&suite.root, suite.require_baseline_manifest()?)?;
        let llm = self.llm.to_llm().await;

        let config = RunConfig {
            mode: match self.mode {
                RunModeArg::FixedContext => RunMode::FixedContext,
                RunModeArg::GrowthReplay => RunMode::GrowthReplay,
            },
            repetitions: self.repetitions,
            run_link_pass: self.link,
            concurrency: self.concurrency,
            model_id: llm.model.model_id_str().to_string(),
            prompt_digests: std::collections::BTreeMap::new(),
            git_commit: "unknown".to_string(),
            dirty_patch_digest: String::new(),
        };

        let options = ReplayOptions {
            config,
            agent_options: self.merge.to_agent_options(),
            merge_chunking: self.merge.to_chunking_options(),
            link_options: self
                .link
                .then(|| knowdit_kg::link::FindingLinkOptions {
                    concurrency: self.concurrency,
                    input_token_budget: None,
                    finding_token_ratio: 0.35,
                    max_semantics_per_batch: usize::MAX,
                    max_findings_per_batch: 135,
                    max_response_attempts: 3,
                    max_agent_steps: 160,
                    include_unlinked: false,
                    candidate_max_semantic_id: None,
                    context_window_utilization: 0.2,
                    variant_render_cap: 5,
                    render_raw_children: true,
                    raw_child_char_cap: 0,
                    evidence_min_high_medium: 40,
                    evidence_min_low: 15,
                    high_quote_min_chars: 48,
                }),
        };

        let outcome = replay_run(&suite, &baseline, &llm, options, &self.output).await?;
        let failed = outcome
            .manifest
            .document_status
            .values()
            .filter(|status| status.starts_with("failed"))
            .count();
        println!(
            "run {} complete: {} documents, {} failed; artifacts in {}",
            outcome.run_id,
            outcome.documents.len(),
            failed,
            outcome.run_dir.display()
        );
        Ok(())
    }
}
