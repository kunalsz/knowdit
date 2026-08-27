//! `knowdit eval corpus` — load and verify a benchmark suite.

use clap::Args;
use color_eyre::eyre::{Result, bail};
use knowdit_eval::corpus::LoadedSuite;
use serde::Serialize;
use std::path::PathBuf;

#[derive(Serialize)]
struct CorpusJson {
    version: String,
    digest: String,
    order: Vec<String>,
    document_count: usize,
}

#[derive(Args, Debug)]
pub struct CorpusArgs {
    /// Path to the suite root (contains `suite.json`).
    #[arg(long)]
    pub suite: PathBuf,

    /// Print the ordered replay document list as JSON and exit.
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

impl CorpusArgs {
    pub async fn run(self) -> Result<()> {
        let suite = LoadedSuite::load(&self.suite)?;
        let digest = suite.suite.digest()?;

        if self.json {
            println!(
                "{}",
                serde_json::to_string_pretty(&CorpusJson {
                    version: suite.suite.version.clone(),
                    digest,
                    order: suite.suite.order.clone(),
                    document_count: suite.suite.documents.len(),
                })?
            );
            return Ok(());
        }

        let problems = suite.verify().await?;
        if problems.is_empty() {
            println!(
                "suite {} verified: {} documents, digest {}",
                suite.suite.version,
                suite.suite.documents.len(),
                digest
            );
            Ok(())
        } else {
            for problem in &problems {
                eprintln!("  - {problem}");
            }
            bail!("suite verification failed with {} problem(s)", problems.len())
        }
    }
}
