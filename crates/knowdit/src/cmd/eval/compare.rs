//! `knowdit eval compare` — paired bootstrap comparison between a
//! baseline and a candidate run, with promotion-gate evaluation.

use clap::Args;
use color_eyre::eyre::Result;
use knowdit_eval::compare::{RunScores, compare_runs, gate_error};
use knowdit_eval::manifest::Policy;
use knowdit_eval::report::Report;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct EvalCompareArgs {
    /// Baseline run directory (contains `run-scores.json`).
    #[arg(long)]
    pub baseline: PathBuf,

    /// Candidate run directory (contains `run-scores.json`).
    #[arg(long)]
    pub candidate: PathBuf,

    /// Suite root that supplied both runs (contains `policy.json`).
    #[arg(long)]
    pub suite: PathBuf,

    /// Fail with a nonzero exit code when the promotion gate fails.
    #[arg(long, default_value_t = false)]
    pub gate: bool,

    /// Bootstrap resamples (default 10_000).
    #[arg(long, default_value_t = 10_000)]
    pub resamples: usize,

    /// Fixed bootstrap seed for reproducible comparisons.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

impl EvalCompareArgs {
    pub async fn run(self) -> Result<()> {
        let baseline_scores: RunScores = serde_json::from_str(&std::fs::read_to_string(
            self.baseline.join("run-scores.json"),
        )?)?;
        let candidate_scores: RunScores = serde_json::from_str(&std::fs::read_to_string(
            self.candidate.join("run-scores.json"),
        )?)?;

        let policy_path = self.suite.join("policy.json");
        let policy = if policy_path.is_file() {
            serde_json::from_str::<Policy>(&std::fs::read_to_string(&policy_path)?)?
        } else {
            Policy::default()
        };

        let comparison =
            compare_runs(&baseline_scores, &candidate_scores, &policy, self.resamples, self.seed)?;

        let report = Report {
            comparison: Some(comparison),
            graph_delta: None,
            graph_metrics_before: None,
            graph_metrics_after: None,
        };
        println!("{}", serde_json::to_string_pretty(&report)?);

        if self.gate {
            let comparison = report.comparison.as_ref().ok_or_else(|| {
                color_eyre::eyre::eyre!("comparison report missing after successful compare")
            })?;
            gate_error(comparison)?;
            println!("gate: PASS");
        }
        Ok(())
    }
}
