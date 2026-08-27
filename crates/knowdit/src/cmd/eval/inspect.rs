//! `knowdit eval inspect` — inspect a comparison, a run's stage
//! events, or a single document's artifacts.

use clap::Args;
use color_eyre::eyre::{Result, bail};
use knowdit_eval::replay::DocumentArtifacts;
use serde::Serialize;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct EvalInspectArgs {
    /// Run directory to inspect.
    #[arg(long)]
    pub run: PathBuf,

    /// Optional document ID: print that document's artifacts as JSON.
    #[arg(long)]
    pub document: Option<String>,

    /// Print only the stage-event timeline for this run.
    #[arg(long, default_value_t = false)]
    pub events: bool,
}

#[derive(Serialize)]
struct TimelineRow {
    document: String,
    stage: String,
    ok: bool,
    detail: String,
}

impl EvalInspectArgs {
    pub async fn run(self) -> Result<()> {
        if self.events {
            let events_path = self.run.join("stage-events.jsonl");
            let text = std::fs::read_to_string(&events_path)?;
            let rows: Vec<TimelineRow> = text
                .lines()
                .filter(|line| !line.trim().is_empty())
                .filter_map(|line| {
                    let event: knowdit_eval::replay::StageEvent =
                        serde_json::from_str(line).ok()?;
                    Some(TimelineRow {
                        document: event.document_id,
                        stage: event.stage,
                        ok: event.ok,
                        detail: event.detail,
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&rows)?);
            return Ok(());
        }

        let Some(document_id) = &self.document else {
            let manifest = std::fs::read_to_string(self.run.join("manifest.json"))?;
            println!("{manifest}");
            return Ok(());
        };

        let artifacts_path = self.run.join("document-artifacts.jsonl");
        let text = std::fs::read_to_string(&artifacts_path)?;
        let artifacts: Vec<DocumentArtifacts> = text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                serde_json::from_str(line).map_err(|e| color_eyre::eyre::eyre!("{e}"))
            })
            .collect::<Result<_, _>>()?;
        let Some(artifact) = artifacts
            .iter()
            .find(|a| &a.document_id == document_id)
        else {
            bail!("document {document_id} not found in run artifacts");
        };
        println!("{}", serde_json::to_string_pretty(artifact)?);
        Ok(())
    }
}
