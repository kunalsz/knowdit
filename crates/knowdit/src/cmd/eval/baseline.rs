//! `knowdit eval baseline` — freeze a database into an immutable
//! suite baseline snapshot + manifest.

use crate::cli::HistoricalDatabaseArgs;
use clap::Args;
use color_eyre::eyre::Result;
use knowdit_eval::baseline;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct BaselineArgs {
    #[command(subcommand)]
    pub command: BaselineCommands,
}

#[derive(clap::Subcommand, Debug)]
pub enum BaselineCommands {
    /// Export the historical KG at `--database-url` as an immutable
    /// suite baseline (SQL snapshot + manifest + validation).
    Create(BaselineCreateArgs),
}

#[derive(Args, Debug)]
pub struct BaselineCreateArgs {
    #[command(flatten)]
    pub database: HistoricalDatabaseArgs,

    /// Path to the suite root (must already contain `suite.json`).
    #[arg(long)]
    pub suite: PathBuf,

    /// Git commit that produced the source DB (recorded for
    /// attribution; the harness does not shell out to git).
    #[arg(long, default_value = "unknown")]
    pub git_commit: String,
}

impl BaselineArgs {
    pub async fn run(self) -> Result<()> {
        match self.command {
            BaselineCommands::Create(args) => args.run().await,
        }
    }
}

impl BaselineCreateArgs {
    pub async fn run(self) -> Result<()> {
        let suite = knowdit_eval::corpus::LoadedSuite::load(&self.suite)?;
        let db = self.database.connect_init().await?;

        let (manifest, sql) = baseline::create_baseline(
            &db,
            &suite.suite.version,
            &self.git_commit,
        )
        .await?;
        baseline::write_baseline(&self.suite, &manifest, &sql)?;
        println!(
            "baseline {} written: {} rows across {} tables, {} validation issues, snapshot {}",
            manifest.suite_version,
            manifest.table_rows.values().sum::<usize>(),
            manifest.table_rows.len(),
            manifest.validation_issues,
            &manifest.snapshot_sha256[..16]
        );
        Ok(())
    }
}
