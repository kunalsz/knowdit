//! `knowdit eval score` — score a run's captured artifacts against
//! its suite's gold labels and write `scores.json` (aggregated stage
//! scores) plus `run-scores.json` (per-document vectors used by
//! `eval compare`).

use clap::Args;
use color_eyre::eyre::Result;
use knowdit_eval::compare::RunScores;
use knowdit_eval::labels::GoldBundle;
use knowdit_eval::replay::DocumentArtifacts;
use knowdit_eval::scorer::{
    CapturedFinding, CapturedLink, CapturedMergeDecision, CapturedSemantic, score_extraction,
    score_links, score_merges,
};
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct EvalScoreArgs {
    /// Run directory produced by `knowdit eval run`.
    #[arg(long)]
    pub run: PathBuf,

    /// Suite root that supplied the run's documents.
    #[arg(long)]
    pub suite: PathBuf,
}

impl EvalScoreArgs {
    pub async fn run(self) -> Result<()> {
        let artifacts = load_artifacts(&self.run)?;
        let gold = load_gold(&self.suite)?;

        let mut run_scores = RunScores::default();
        let mut extraction_reports = Vec::new();
        let mut merge_semantics_reports = Vec::new();
        let mut merge_findings_reports = Vec::new();
        let mut link_reports = Vec::new();

        for artifact in &artifacts {
            let extraction_report = match gold.extraction.get(&artifact.document_id) {
                Some(extraction_gold) => Some(score_extraction(
                    extraction_gold,
                    &artifact
                        .semantics
                        .iter()
                        .map(|s| CapturedSemantic {
                            name: s.name.clone(),
                            category: s.category.clone(),
                        })
                        .collect::<Vec<_>>(),
                    &artifact
                        .findings
                        .iter()
                        .map(|f| CapturedFinding {
                            title: f.title.clone(),
                            severity: f.severity.clone(),
                            category: f.category.clone(),
                            subcategory: f.subcategory.clone(),
                        })
                        .collect::<Vec<_>>(),
                )),
                None => {
                    tracing::warn!("no extraction gold for {}", artifact.document_id);
                    None
                }
            };

            let (sem_merge_report, find_merge_report) =
                match gold.merges.get(&artifact.document_id) {
                    Some(merge_gold) => {
                        let sem_decisions: Vec<CapturedMergeDecision> = artifact
                            .semantic_merges
                            .iter()
                            .map(|m| CapturedMergeDecision {
                                raw_name: m.raw.clone(),
                                action: m.action.clone(),
                                target_fingerprints: m
                                    .target_ids
                                    .iter()
                                    .map(|id| format!("db-id-{id}"))
                                    .collect(),
                            })
                            .collect();
                        let find_decisions: Vec<CapturedMergeDecision> = artifact
                            .finding_merges
                            .iter()
                            .map(|m| CapturedMergeDecision {
                                raw_name: m.raw.clone(),
                                action: m.action.clone(),
                                target_fingerprints: m
                                    .target_ids
                                    .iter()
                                    .map(|id| format!("db-id-{id}"))
                                    .collect(),
                            })
                            .collect();
                        (
                            Some(score_merges(
                                &knowdit_eval::labels::MergeGold {
                                    document_id: merge_gold.document_id.clone(),
                                    semantics: merge_gold.semantics.clone(),
                                    findings: Vec::new(),
                                },
                                &sem_decisions,
                            )),
                            Some(score_merges(
                                &knowdit_eval::labels::MergeGold {
                                    document_id: merge_gold.document_id.clone(),
                                    semantics: Vec::new(),
                                    findings: merge_gold.findings.clone(),
                                },
                                &find_decisions,
                            )),
                        )
                    }
                    None => (None, None),
                };

            let link_report = match gold.links.get(&artifact.document_id) {
                Some(link_gold) => {
                    let captured: Vec<CapturedLink> = artifact
                        .in_project_links
                        .iter()
                        .filter_map(|(finding_idx, semantic_idx)| {
                            let finding = artifact.findings.get(*finding_idx)?;
                            let semantic = artifact.semantics.get(*semantic_idx)?;
                            Some(CapturedLink {
                                finding_name: finding.title.clone(),
                                semantic_names: vec![semantic.name.clone()],
                            })
                        })
                        .collect();
                    Some(score_links(link_gold, &captured))
                }
                None => None,
            };

            if let Some(report) = &extraction_report {
                run_scores
                    .extraction_f1
                    .push((artifact.document_id.clone(), report.f1));
                extraction_reports.push(report.clone());
            }
            if let Some(report) = sem_merge_report {
                run_scores
                    .merge_f1
                    .push((artifact.document_id.clone(), report.f1));
                merge_semantics_reports.push(report);
            }
            if let Some(report) = find_merge_report {
                // One merge score per document: when both sides were
                // scored, use their average; otherwise use the side
                // that had gold.
                if let Some((id, existing)) = run_scores
                    .merge_f1
                    .iter_mut()
                    .find(|(id, _)| id == &artifact.document_id)
                {
                    let _ = id;
                    *existing = (*existing + report.f1) / 2.0;
                } else {
                    run_scores
                        .merge_f1
                        .push((artifact.document_id.clone(), report.f1));
                }
                merge_findings_reports.push(report);
            }
            if let Some(report) = &link_report {
                run_scores
                    .link_f1
                    .push((artifact.document_id.clone(), report.f1));
                link_reports.push(report.clone());
            }
        }

        let scores = knowdit_eval::scorer::aggregate(
            &extraction_reports,
            &merge_semantics_reports,
            &merge_findings_reports,
            &link_reports,
        );

        std::fs::write(
            self.run.join("scores.json"),
            serde_json::to_string_pretty(&scores)?,
        )?;
        std::fs::write(
            self.run.join("run-scores.json"),
            serde_json::to_string_pretty(&run_scores)?,
        )?;
        println!(
            "scores written to {}/scores.json and run-scores.json",
            self.run.display()
        );
        Ok(())
    }
}

fn load_artifacts(run_dir: &PathBuf) -> Result<Vec<DocumentArtifacts>> {
    let artifacts_path = run_dir.join("document-artifacts.jsonl");
    let artifacts_text = std::fs::read_to_string(&artifacts_path)?;
    artifacts_text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).map_err(|e| color_eyre::eyre::eyre!("{e}")))
        .collect()
}

/// Load the suite's `gold/bundle.json` (optional — a suite without
/// gold yields empty stage scores rather than an error).
fn load_gold(suite_root: &PathBuf) -> Result<GoldBundle> {
    let path = suite_root.join("gold").join("bundle.json");
    if !path.is_file() {
        tracing::warn!(
            "no gold bundle at {} — stage scores will be empty",
            path.display()
        );
        return Ok(GoldBundle::default());
    }
    let bundle: GoldBundle = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    bundle.validate()?;
    Ok(bundle)
}
