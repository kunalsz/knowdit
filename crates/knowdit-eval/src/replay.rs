//! Run orchestration: fixed-context and growth-replay learning with
//! stage-artifact capture.
//!
//! The driver uses the production pipeline's public stages
//! (`categorize_and_extract`, `merge_decisions`,
//! `write_project_completed_txn`, link pass) and records every stage
//! boundary into a JSONL event file. Documents are admitted through
//! the same loaders the CLI uses (`knowdit_project::ProjectData` /
//! `C4PairedProjectData`), so the benchmark measures the real path.

use crate::error::{EvalError, Result};
use crate::manifest::{DocumentKind, DocumentSpec, RunConfig, RunManifest, RunMode};
use crate::sandbox::{Baseline, Sandbox};
use knowdit_kg::agent_runner::AgentRunOptions;
use knowdit_kg::agents::MergeChunkingOptions;
use knowdit_kg::learn::{ExtractResult, FindingMergeResult, MergeResult};
use knowdit_kg::link::FindingLinkOptions;
use knowdit_kg::project_loader::ProjectData;
use llmy::client::client::LLM;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// One stage event recorded to the run's JSONL artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageEvent {
    pub run_id: String,
    pub document_id: String,
    pub stage: String,
    pub ok: bool,
    pub detail: String,
    /// Unix millis when the event was recorded.
    pub at_unix_ms: u64,
    /// Extra structured payload (stage-specific).
    pub payload: Option<StageEventPayload>,
}

/// Typed payload union for stage events. `None` stages carry no
/// structured payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StageEventPayload {
    ExtractSummary {
        categories: Vec<String>,
        semantic_count: usize,
        finding_count: usize,
        link_count: usize,
    },
}

/// Options controlling one replay run.
#[derive(Debug, Clone)]
pub struct ReplayOptions {
    pub config: RunConfig,
    pub agent_options: AgentRunOptions,
    pub merge_chunking: MergeChunkingOptions,
    /// Optional cross-project link pass options; when `None`, the link
    /// pass is skipped even if `config.run_link_pass` is set.
    pub link_options: Option<FindingLinkOptions>,
}

/// Captured stage artifacts for one document. Written to disk as JSON
/// inside the run directory.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocumentArtifacts {
    pub document_id: String,
    /// `categories` from the categorize stage.
    pub categories: Vec<String>,
    /// Extracted (post-dedup) semantics.
    pub semantics: Vec<CapturedExtractedSemantic>,
    /// Extracted (post-dedup) findings.
    pub findings: Vec<CapturedExtractedFinding>,
    /// In-project links as `(finding_index, semantic_index)` pairs.
    pub in_project_links: Vec<(usize, usize)>,
    /// Semantic merge decisions.
    pub semantic_merges: Vec<CapturedMerge>,
    /// Finding merge decisions.
    pub finding_merges: Vec<CapturedMerge>,
    /// Whether the admission transaction committed.
    pub committed: bool,
    /// Validation issue count after this document (growth mode).
    pub post_validation_issues: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedExtractedSemantic {
    pub name: String,
    pub category: String,
    pub definition: String,
    pub description: String,
    pub functions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedExtractedFinding {
    pub title: String,
    pub severity: String,
    pub category: String,
    pub subcategory: String,
    pub root_cause: String,
    pub description: String,
    pub patterns: String,
    pub exploits: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedMerge {
    /// Normalized raw identity (name or title).
    pub raw: String,
    /// `new` or `merge`.
    pub action: String,
    /// Target IDs the decision merged into (DB-integer IDs — only
    /// meaningful within one run; fingerprints are added at score
    /// time by the graph module).
    pub target_ids: Vec<i32>,
    pub updated_description: Option<String>,
    pub updated_patterns: Option<String>,
    pub updated_exploits: Option<String>,
    pub appended_description: Option<String>,
    pub appended_patterns: Option<String>,
    pub appended_exploits: Option<String>,
}

/// The result of a full replay run.
#[derive(Debug)]
pub struct ReplayOutcome {
    pub run_id: String,
    pub manifest: RunManifest,
    /// Artifacts per document, in replay order.
    pub documents: Vec<DocumentArtifacts>,
    /// The final sandbox (growth mode) or the seed-only sandbox
    /// (fixed-context mode keeps one admission sandbox at a time, so
    /// the returned sandbox is the last one used).
    pub final_sandbox: Sandbox,
    /// Absolute directory the run artifacts were written to.
    pub run_dir: PathBuf,
}

/// Load one document through the production loaders.
pub async fn load_document(spec: &DocumentSpec) -> Result<ProjectData> {
    match &spec.kind {
        DocumentKind::SourceDir { spec: dir_spec } => {
            ProjectData::from_source_dir_spec(dir_spec).await.map_err(|e| {
                EvalError::corpus(format!("document {} failed to load: {e}", spec.id))
            })
        }
        DocumentKind::C4 {
            dataset_dir,
            contest_id,
        } => ProjectData::from_c4(dataset_dir, *contest_id).await.map_err(|e| {
            EvalError::corpus(format!("document {} failed to load C4: {e}", spec.id))
        }),
        DocumentKind::Sherlock {
            out_dir,
            contest_id,
        } => match ProjectData::from_sherlock(out_dir, *contest_id).await.map_err(|e| {
            EvalError::corpus(format!("document {} failed to load sherlock: {e}", spec.id))
        })? {
            Some(project) => Ok(project),
            None => Err(EvalError::corpus(format!(
                "document {} is not ingestible (sherlock contest {contest_id})",
                spec.id
            ))),
        },
    }
}

/// Capture the extract-stage artifacts into `DocumentArtifacts`.
fn capture_extract(document_id: &str, extract: &ExtractResult) -> DocumentArtifacts {
    DocumentArtifacts {
        document_id: document_id.to_string(),
        categories: extract.categories.iter().map(|c| c.as_str().to_string()).collect(),
        semantics: extract
            .semantics
            .iter()
            .map(|s| CapturedExtractedSemantic {
                name: s.name.clone(),
                category: s.category.as_str().to_string(),
                definition: s.definition.clone(),
                description: s.description.clone(),
                functions: s.functions.iter().map(|f| f.name.clone()).collect(),
            })
            .collect(),
        findings: extract
            .findings
            .iter()
            .map(|f| CapturedExtractedFinding {
                title: f.title.clone(),
                severity: f.severity.to_string(),
                category: f.category.to_string(),
                subcategory: f.subcategory.clone(),
                root_cause: f.root_cause.clone(),
                description: f.description.clone(),
                patterns: f.patterns.clone(),
                exploits: f.exploits.clone(),
            })
            .collect(),
        in_project_links: extract.in_project_links.edges.clone(),
        ..DocumentArtifacts::default()
    }
}

fn capture_semantic_merges(decisions: &[MergeResult]) -> Vec<CapturedMerge> {
    decisions
        .iter()
        .map(|d| {
            let (action, target_ids, updated_description, appended_description) = match &d.action {
                knowdit_kg::learn::MergeAction::New => ("new".to_string(), vec![], None, None),
                knowdit_kg::learn::MergeAction::Merge {
                    target_ids,
                    updated_description,
                    appended_description,
                } => (
                    "merge".to_string(),
                    target_ids.clone(),
                    updated_description.clone(),
                    appended_description.clone(),
                ),
            };
            CapturedMerge {
                raw: d.semantic.name.clone(),
                action,
                target_ids,
                updated_description,
                updated_patterns: None,
                updated_exploits: None,
                appended_description,
                appended_patterns: None,
                appended_exploits: None,
            }
        })
        .collect()
}

fn capture_finding_merges(decisions: &[FindingMergeResult]) -> Vec<CapturedMerge> {
    decisions
        .iter()
        .map(|d| {
            let (action, target_ids, updated_description, updated_patterns, updated_exploits, appended_description, appended_patterns, appended_exploits) =
                match &d.action {
                    knowdit_kg::learn::FindingMergeAction::New => {
                        ("new".to_string(), vec![], None, None, None, None, None, None)
                    }
                    knowdit_kg::learn::FindingMergeAction::Merge {
                        target_ids,
                        updated_description,
                        updated_patterns,
                        updated_exploits,
                        appended_description,
                        appended_patterns,
                        appended_exploits,
                    } => (
                        "merge".to_string(),
                        target_ids.clone(),
                        updated_description.clone(),
                        updated_patterns.clone(),
                        updated_exploits.clone(),
                        appended_description.clone(),
                        appended_patterns.clone(),
                        appended_exploits.clone(),
                    ),
                };
            CapturedMerge {
                raw: d.finding.title.clone(),
                action,
                target_ids,
                updated_description,
                updated_patterns,
                updated_exploits,
                appended_description,
                appended_patterns,
                appended_exploits,
            }
        })
        .collect()
}

/// Run one document against one sandbox: extract → merge decisions →
/// transactional write. Returns captured artifacts and updates the
/// sandbox in place.
async fn run_document(
    run_id: &str,
    document_id: &str,
    project: &ProjectData,
    sandbox: &Sandbox,
    llm: &LLM,
    options: &ReplayOptions,
    events: &mut Vec<StageEvent>,
) -> Result<DocumentArtifacts> {
    // Stage 1: categorize + extract + in-project link.
    let extract = match project
        .categorize_and_extract(llm, &options.agent_options, None)
        .await
    {
        Ok(extract) => {
            events.push(StageEvent {
                run_id: run_id.to_string(),
                document_id: document_id.to_string(),
                stage: "categorize_and_extract".to_string(),
                ok: true,
                detail: format!(
                    "categories={} semantics={} findings={} links={}",
                    extract.categories.len(),
                    extract.semantics.len(),
                    extract.findings.len(),
                    extract.in_project_links.edges.len()
                ),
                at_unix_ms: now_ms(),
                payload: Some(StageEventPayload::ExtractSummary {
                    categories: extract.categories.iter().map(|c| c.as_str().to_string()).collect(),
                    semantic_count: extract.semantics.len(),
                    finding_count: extract.findings.len(),
                    link_count: extract.in_project_links.edges.len(),
                }),
            });
            extract
        }
        Err(e) => {
            events.push(StageEvent {
                run_id: run_id.to_string(),
                document_id: document_id.to_string(),
                stage: "categorize_and_extract".to_string(),
                ok: false,
                detail: format!("{e}"),
                at_unix_ms: now_ms(),
                payload: None,
            });
            return Err(e.into());
        }
    };

    let mut artifacts = capture_extract(document_id, &extract);

    // Stage 2: merge decisions (read-only against the live sandbox).
    let (semantic_decisions, finding_decisions) = match project
        .merge_decisions(
            &sandbox.db,
            llm,
            &extract,
            &options.agent_options,
            options.merge_chunking,
        )
        .await
    {
        Ok(pair) => {
            events.push(StageEvent {
                run_id: run_id.to_string(),
                document_id: document_id.to_string(),
                stage: "merge_decisions".to_string(),
                ok: true,
                detail: format!(
                    "semantic_decisions={} finding_decisions={}",
                    pair.0.len(),
                    pair.1.len()
                ),
                at_unix_ms: now_ms(),
                payload: None,
            });
            pair
        }
        Err(e) => {
            events.push(StageEvent {
                run_id: run_id.to_string(),
                document_id: document_id.to_string(),
                stage: "merge_decisions".to_string(),
                ok: false,
                detail: format!("{e}"),
                at_unix_ms: now_ms(),
                payload: None,
            });
            return Err(e.into());
        }
    };

    artifacts.semantic_merges = capture_semantic_merges(&semantic_decisions);
    artifacts.finding_merges = capture_finding_merges(&finding_decisions);

    // Stage 3: transactional admission using the SAME decisions that
    // were captured above (the production `merge_and_write` would
    // re-run the merge agents and diverge from the captured
    // artifacts, so the driver writes the decisions directly).
    let committed = match sandbox
        .db
        .write_project_completed(
            project.name(),
            project.platform_id(),
            &extract.categories,
            &semantic_decisions,
            &finding_decisions,
            &extract.in_project_links,
        )
        .await
    {
        Ok(()) => {
            events.push(StageEvent {
                run_id: run_id.to_string(),
                document_id: document_id.to_string(),
                stage: "admission".to_string(),
                ok: true,
                detail: "transaction committed".to_string(),
                at_unix_ms: now_ms(),
                payload: None,
            });
            true
        }
        Err(e) => {
            events.push(StageEvent {
                run_id: run_id.to_string(),
                document_id: document_id.to_string(),
                stage: "admission".to_string(),
                ok: false,
                detail: format!("{e}"),
                at_unix_ms: now_ms(),
                payload: None,
            });
            return Err(e.into());
        }
    };
    artifacts.committed = committed;

    // Stage 4: post-document invariant check.
    let post = sandbox.validation_issue_count().await?;
    artifacts.post_validation_issues = Some(post);
    if post > 0 {
        events.push(StageEvent {
            run_id: run_id.to_string(),
            document_id: document_id.to_string(),
            stage: "validation".to_string(),
            ok: false,
            detail: format!("{post} remaining validation issues"),
            at_unix_ms: now_ms(),
            payload: None,
        });
    }

    Ok(artifacts)
}

/// Execute a full replay run against a loaded suite.
pub async fn run(
    suite: &crate::corpus::LoadedSuite,
    baseline: &Baseline,
    llm: &LLM,
    options: ReplayOptions,
    run_dir: &Path,
) -> Result<ReplayOutcome> {
    let run_id = format!(
        "{}-{}",
        suite.suite.version,
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    );
    std::fs::create_dir_all(run_dir)?;

    let mut events: Vec<StageEvent> = Vec::new();
    let mut document_status = std::collections::BTreeMap::new();
    let mut artifacts_out = Vec::new();

    let mut last_sandbox: Option<Sandbox> = None;

    match options.config.mode {
        RunMode::FixedContext => {
            for doc in suite.suite.replay_documents() {
                // Fresh seed per document isolates stage quality.
                let sandbox = baseline.materialize(run_dir).await?;
                let project = load_document(doc).await?;
                match run_document(
                    &run_id,
                    &doc.id,
                    &project,
                    &sandbox,
                    llm,
                    &options,
                    &mut events,
                )
                .await
                {
                    Ok(artifacts) => {
                        document_status.insert(doc.id.clone(), "ok".to_string());
                        artifacts_out.push(artifacts);
                    }
                    Err(e) => {
                        document_status
                            .insert(doc.id.clone(), format!("failed: {e}"));
                        artifacts_out.push(DocumentArtifacts {
                            document_id: doc.id.clone(),
                            ..DocumentArtifacts::default()
                        });
                    }
                }
                last_sandbox = Some(sandbox);
            }
        }
        RunMode::GrowthReplay => {
            // One evolving sandbox; documents admitted in declared
            // order measure accumulation and canonical drift.
            let sandbox = baseline.materialize(run_dir).await?;
            for doc in suite.suite.replay_documents() {
                let project = match load_document(doc).await {
                    Ok(p) => p,
                    Err(e) => {
                        document_status
                            .insert(doc.id.clone(), format!("failed: {e}"));
                        continue;
                    }
                };
                match run_document(&run_id, &doc.id, &project, &sandbox, llm, &options, &mut events)
                    .await
                {
                    Ok(artifacts) => {
                        document_status.insert(doc.id.clone(), "ok".to_string());
                        artifacts_out.push(artifacts);
                    }
                    Err(e) => {
                        document_status.insert(doc.id.clone(), format!("failed: {e}"));
                        artifacts_out.push(DocumentArtifacts {
                            document_id: doc.id.clone(),
                            ..DocumentArtifacts::default()
                        });
                    }
                }
            }
            last_sandbox = Some(sandbox);
        }
    }

    // A suite with no replay documents still materializes one
    // sandbox so the manifest and graph-after artifacts reflect the
    // seed state.
    let sandbox = match last_sandbox {
        Some(sandbox) => sandbox,
        None => baseline.materialize(run_dir).await?,
    };

    // Cross-project link pass when requested (growth mode keeps the
    // sandbox; fixed-context runs it on the last seed state).
    if options.config.run_link_pass {
        if let Some(link_options) = &options.link_options {
            match link_options
                .link_pending_findings(&sandbox.db, llm)
                .await
            {
                Ok(()) => events.push(StageEvent {
                    run_id: run_id.clone(),
                    document_id: "*".to_string(),
                    stage: "link_pass".to_string(),
                    ok: true,
                    detail: "cross-project link pass completed".to_string(),
                    at_unix_ms: now_ms(),
                    payload: None,
                }),
                Err(e) => events.push(StageEvent {
                    run_id: run_id.clone(),
                    document_id: "*".to_string(),
                    stage: "link_pass".to_string(),
                    ok: false,
                    detail: format!("{e}"),
                    at_unix_ms: now_ms(),
                    payload: None,
                }),
            }
        }
    }

    // ── manifest ──
    let suite_digest = suite.suite.digest()?;
    let manifest = RunManifest {
        run_id: run_id.clone(),
        suite_version: suite.suite.version.clone(),
        suite_digest,
        baseline_digest: baseline.manifest.snapshot_sha256.clone(),
        config: options.config.clone(),
        environment: std::collections::BTreeMap::new(),
        started_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        document_status,
    };

    // ── write artifacts ──
    let events_path = run_dir.join("stage-events.jsonl");
    let mut events_file = std::fs::File::create(&events_path)?;
    use std::io::Write;
    for event in &events {
        writeln!(events_file, "{}", serde_json::to_string(event)?)?;
    }
    let artifacts_path = run_dir.join("document-artifacts.jsonl");
    let mut artifacts_file = std::fs::File::create(&artifacts_path)?;
    for artifact in &artifacts_out {
        writeln!(artifacts_file, "{}", serde_json::to_string(artifact)?)?;
    }
    std::fs::write(
        run_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    std::fs::write(
        run_dir.join("graph-after.jsonl"),
        {
            let graph = sandbox.knowledge_graph().await?;
            let normalized = crate::graph::normalize(&graph);
            let mut out = String::new();
            for sem in &normalized.semantics {
                out.push_str(&format!("semantic\t{}\n", serde_json::to_string(sem)?));
            }
            for f in &normalized.findings {
                out.push_str(&format!("finding\t{}\n", serde_json::to_string(f)?));
            }
            for l in &normalized.links {
                out.push_str(&format!("link\t{}\n", serde_json::to_string(l)?));
            }
            for m in &normalized.semantic_merges {
                out.push_str(&format!("semantic_merge\t{}\n", serde_json::to_string(m)?));
            }
            for m in &normalized.finding_merges {
                out.push_str(&format!("finding_merge\t{}\n", serde_json::to_string(m)?));
            }
            out
        },
    )?;

    Ok(ReplayOutcome {
        run_id,
        manifest,
        documents: artifacts_out,
        final_sandbox: sandbox,
        run_dir: run_dir.to_path_buf(),
    })
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
