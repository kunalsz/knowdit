use crate::agent_runner::AgentRunOptions;
use crate::agents::{
    AggregatedFindingMergeDecision, AggregatedSemanticMergeDecision, CategorizeRunner,
    FindingChunkExtractor, FindingMerger, MergeChunkingOptions, NarrativeChunkExtractor,
    NarrativeCombinedItem, NarrativeFindingRecord, NarrativeLinkRecord, NarrativeSemanticRecord,
    SemanticChunkExtractor, SemanticMerger,
};
use crate::category::DeFiCategory;
use crate::db::HistoricalDatabase;
use crate::error::{KgError, Result};
pub use crate::link::{FindingLinkOptions, PendingFindingForLinking, PersistedFindingLinkResult};
use crate::project_loader::ProjectData;
use crate::prompts;
use crate::vulnerability::{VulnerabilityCategory, resolve_taxonomy_entry};
use async_trait::async_trait;
use itertools::Itertools;
pub use knowdit_kg_model::{ExtractedFinding, ExtractedFunction, ExtractedSemantic};
use llmy::client::client::LLM;
use llmy::client::context::TokenCursor;
use llmy::client::model::OpenAIModel;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::time::Instant;

// ── LLM response types ──────────────────────────────────────────────
//
// The categorize / extract semantic / extract finding / semantic merge /
// finding merge phases all use llmy `Agent` + tool-calls now; their tool
// arguments + tool implementations live in [`crate::agents`]. Only the
// in-project linking phase still uses the JSON-mode prompt+parse pattern,
// because it's a single LLM call producing a small, simply-shaped index map.

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct InProjectLinkEntry {
    finding_index: usize,
    semantic_indices: Vec<usize>,
    #[serde(default)]
    reasoning: String,
}

#[derive(Debug, Deserialize)]
struct InProjectLinkResponse {
    links: Vec<InProjectLinkEntry>,
}

fn render_semantics_for_in_project_link(semantics: &[ExtractedSemantic]) -> String {
    semantics
        .iter()
        .enumerate()
        .map(|(idx, s)| {
            format!(
                "S{idx}\nname: {name}\ncategory: {category}\ndefinition: {definition}\ndescription: {description}\n\n",
                idx = idx,
                name = s.name,
                category = s.category.as_str(),
                definition = s.definition.trim(),
                description = s.description.trim()
            )
        })
        .collect::<String>()
}

fn render_findings_for_in_project_link(findings: &[ExtractedFinding]) -> String {
    findings
        .iter()
        .enumerate()
        .map(|(idx, f)| {
            format!(
                "F{idx}\ntitle: {title}\nseverity: {severity}\ncategory: {category}\nsubcategory: {subcategory}\nroot_cause: {root_cause}\ndescription: {description}\npatterns: {patterns}\n\n",
                idx = idx,
                title = f.title,
                severity = f.severity,
                category = f.category,
                subcategory = f.subcategory,
                root_cause = f.root_cause.trim(),
                description = f.description.trim(),
                patterns = f.patterns.trim(),
            )
        })
        .collect::<String>()
}

// ── Public merge types (used by HistoricalDatabase) ────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergeAction {
    /// Admit the new raw as a fresh canonical (no existing matches).
    New,
    /// Fold the new raw into one *or more* existing canonicals. Each
    /// `target_id` becomes one row in `semantic_merge`. The canonical's
    /// `name` and `definition` are stable identity and are NEVER touched
    /// by a merge. `updated_description`, when present, REPLACES each
    /// merged-into canonical's description with a generalization; `None`
    /// keeps the existing description.
    Merge {
        target_ids: Vec<i32>,
        updated_description: Option<String>,
        /// One-or-two-sentence note (how this raw extends the canonical)
        /// written to every `semantic_merge` edge this fold creates.
        appended_description: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeResult {
    pub semantic: ExtractedSemantic,
    pub action: MergeAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FindingMergeAction {
    /// Admit the new raw finding as a fresh canonical.
    New,
    /// Fold the new raw into one *or more* existing canonical findings.
    /// Canonical's `title`, `severity`, and `root_cause` are stable
    /// identity and are NEVER touched. The `updated_*` fields, when present,
    /// REPLACE the canonical's description / patterns / exploits with
    /// generalizations at write time; `None` keeps the existing value.
    Merge {
        target_ids: Vec<i32>,
        updated_description: Option<String>,
        updated_patterns: Option<String>,
        updated_exploits: Option<String>,
        /// One-or-two-sentence notes (how this raw extends the canonical's
        /// description / patterns / exploits) written to every `finding_merge`
        /// edge this fold creates.
        appended_description: Option<String>,
        appended_patterns: Option<String>,
        appended_exploits: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingMergeResult {
    pub finding: ExtractedFinding,
    pub action: FindingMergeAction,
}

#[derive(Debug, Serialize)]
struct SemanticMergeCheckpointManifest {
    version: &'static str,
    stage: &'static str,
    model: String,
    max_agent_steps: usize,
    context_window_utilization: f64,
    new_item_token_ratio: f64,
    merge_concurrency: usize,
    new_item_batch_size: usize,
    raw_child_variant_cap: usize,
    raw_child_char_cap: usize,
    full_candidate_context: bool,
    candidate_routing: bool,
    extracted: Vec<ExtractedSemantic>,
    candidates:
        Vec<crate::agents::CanonicalWithChildren<knowdit_kg_model::db::semantic_node::Model>>,
}

#[derive(Debug, Serialize)]
struct FindingMergeCheckpointManifest {
    version: &'static str,
    stage: &'static str,
    model: String,
    max_agent_steps: usize,
    context_window_utilization: f64,
    new_item_token_ratio: f64,
    merge_concurrency: usize,
    new_item_batch_size: usize,
    raw_child_variant_cap: usize,
    raw_child_char_cap: usize,
    full_candidate_context: bool,
    candidate_routing: bool,
    extracted: Vec<ExtractedFinding>,
    candidates: Vec<crate::agents::FindingCanonicalWithTaxonomy>,
}

impl ProjectData {
    fn serialized_hash<T: Serialize>(value: &T) -> Result<String> {
        let mut h: u64 = 0xcbf29ce484222325;
        for &byte in serde_json::to_vec(value)?.iter() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        Ok(format!("{h:016x}"))
    }

    async fn load_merge_checkpoint<T: DeserializeOwned>(
        &self,
        db: &HistoricalDatabase,
        stage: &str,
        model: &str,
        content_hash: &str,
    ) -> Result<Option<Vec<T>>> {
        let rows = db.load_extraction_chunks(&self.display_id(), stage).await?;
        if rows.is_empty() {
            return Ok(None);
        }
        if !db
            .extraction_chunks_match(&self.display_id(), stage, model, content_hash)
            .await?
            || rows.len() != 1
            || rows[0].chunk_idx != 0
        {
            db.clear_extraction_chunks(&self.display_id(), stage)
                .await?;
            return Ok(None);
        }
        Ok(Some(serde_json::from_str(&rows[0].chunk_json)?))
    }
}

// ── ProjectData learning pipeline ───────────────────────────────────

/// Index-based links produced by the in-project linking step. Each
/// `(finding_index, semantic_index)` pair refers to positions in the
/// parent `ExtractResult.findings` and `ExtractResult.semantics` arrays.
/// The atomic admission step translates these positional indices into
/// concrete row ids for `semantic_finding_link` after all raw inserts.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InProjectLinks {
    pub edges: Vec<(usize, usize)>,
}

impl InProjectLinks {
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

/// Intermediate result from the categorize + extract + in-project link phase.
/// Can be computed concurrently across projects (no DB writes happen here).
pub struct ExtractResult {
    pub categories: Vec<DeFiCategory>,
    pub semantics: Vec<ExtractedSemantic>,
    pub findings: Vec<ExtractedFinding>,
    /// In-project semantic↔finding links: every finding has ≥1 entry
    /// (LLM-enforced). Indices are positional in `semantics` / `findings`.
    pub in_project_links: InProjectLinks,
}

#[derive(Debug, Clone)]
pub struct KnownExtractedChunk<T> {
    pub chunk_idx: usize,
    pub results: Vec<T>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NarrativeCombinedChunk {
    items: Vec<NarrativeCombinedItem>,
}

#[derive(Debug, Default)]
struct NarrativeRawExtraction {
    semantics: Vec<NarrativeSemanticRecord>,
    findings: Vec<NarrativeFindingRecord>,
    links: Vec<NarrativeLinkRecord>,
}

impl NarrativeRawExtraction {
    fn append_chunk(&mut self, chunk_idx: usize, chunk: NarrativeCombinedChunk) -> Result<()> {
        let mut semantic_ids = BTreeMap::new();
        let mut finding_ids = BTreeMap::new();

        for item in &chunk.items {
            match item {
                NarrativeCombinedItem::Semantic(record) => {
                    let global_id = format!("sem-{chunk_idx}-{}", record.id);
                    if semantic_ids
                        .insert(record.id.clone(), global_id.clone())
                        .is_some()
                    {
                        return Err(KgError::other(format!(
                            "narrative chunk {chunk_idx} emitted duplicate semantic id `{}`",
                            record.id
                        )));
                    }
                    self.semantics.push(NarrativeSemanticRecord {
                        id: global_id,
                        semantic: record.semantic.clone(),
                    });
                }
                NarrativeCombinedItem::Finding(record) => {
                    let global_id = format!("finding-{chunk_idx}-{}", record.id);
                    if finding_ids
                        .insert(record.id.clone(), global_id.clone())
                        .is_some()
                    {
                        return Err(KgError::other(format!(
                            "narrative chunk {chunk_idx} emitted duplicate finding id `{}`",
                            record.id
                        )));
                    }
                    self.findings.push(NarrativeFindingRecord {
                        id: global_id,
                        finding: ProjectData::canonicalize_finding(record.finding.clone())?,
                    });
                }
                NarrativeCombinedItem::Link(_) => {}
            }
        }

        for item in chunk.items {
            let NarrativeCombinedItem::Link(link) = item else {
                continue;
            };
            let Some(finding_id) = finding_ids.get(&link.finding_id) else {
                return Err(KgError::other(format!(
                    "narrative chunk {chunk_idx} linked unknown finding id `{}`",
                    link.finding_id
                )));
            };
            let mut semantic_global_ids = Vec::new();
            for semantic_id in link.semantic_ids {
                let Some(global_id) = semantic_ids.get(&semantic_id) else {
                    return Err(KgError::other(format!(
                        "narrative chunk {chunk_idx} linked unknown semantic id `{semantic_id}`"
                    )));
                };
                semantic_global_ids.push(global_id.clone());
            }
            self.links.push(NarrativeLinkRecord {
                finding_id: finding_id.clone(),
                semantic_ids: semantic_global_ids,
            });
        }
        Ok(())
    }

    fn into_extract_parts(
        self,
    ) -> Result<(
        Vec<ExtractedSemantic>,
        Vec<ExtractedFinding>,
        InProjectLinks,
        bool,
    )> {
        let mut semantics: Vec<ExtractedSemantic> = Vec::new();
        let mut semantic_indices: BTreeMap<String, usize> = BTreeMap::new();
        let mut semantic_ids: BTreeMap<String, usize> = BTreeMap::new();
        for record in self.semantics {
            let key = record
                .semantic
                .name
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase();
            let index = if let Some(index) = semantic_indices.get(&key).copied() {
                let existing = &mut semantics[index];
                for function in record.semantic.functions {
                    if !existing.functions.iter().any(|candidate| {
                        candidate.name == function.name && candidate.contract == function.contract
                    }) {
                        existing.functions.push(function);
                    }
                }
                if record.semantic.description.len() > existing.description.len() {
                    existing.description = record.semantic.description;
                    existing.definition = record.semantic.definition;
                }
                index
            } else {
                let index = semantics.len();
                semantic_indices.insert(key, index);
                semantics.push(record.semantic);
                index
            };
            semantic_ids.insert(record.id, index);
        }

        let mut findings: Vec<ExtractedFinding> = Vec::new();
        let mut finding_indices: BTreeMap<String, usize> = BTreeMap::new();
        let mut finding_ids: BTreeMap<String, usize> = BTreeMap::new();
        for record in self.findings {
            let key = format!(
                "{} {}",
                record
                    .finding
                    .title
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" "),
                record
                    .finding
                    .root_cause
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            )
            .to_lowercase();
            let index = if let Some(index) = finding_indices.get(&key).copied() {
                let existing = &mut findings[index];
                existing.severity = existing.severity.max(record.finding.severity);
                if record.finding.description.len() > existing.description.len() {
                    existing.category = record.finding.category;
                    existing.subcategory = record.finding.subcategory.clone();
                    existing.description = record.finding.description.clone();
                }
                if record.finding.root_cause.len() > existing.root_cause.len() {
                    existing.root_cause = record.finding.root_cause.clone();
                }
                if record.finding.patterns.len() > existing.patterns.len() {
                    existing.patterns = record.finding.patterns.clone();
                }
                if record.finding.exploits.len() > existing.exploits.len() {
                    existing.exploits = record.finding.exploits.clone();
                }
                index
            } else {
                let index = findings.len();
                finding_indices.insert(key, index);
                findings.push(record.finding);
                index
            };
            finding_ids.insert(record.id, index);
        }

        let mut edges = BTreeSet::new();
        for link in self.links {
            let Some(&finding_index) = finding_ids.get(&link.finding_id) else {
                return Err(KgError::other(format!(
                    "narrative link references missing finding `{}`",
                    link.finding_id
                )));
            };
            for semantic_id in link.semantic_ids {
                let Some(&semantic_index) = semantic_ids.get(&semantic_id) else {
                    return Err(KgError::other(format!(
                        "narrative link references missing semantic `{semantic_id}`"
                    )));
                };
                edges.insert((finding_index, semantic_index));
            }
        }
        let covered: BTreeSet<usize> = edges.iter().map(|(finding, _)| *finding).collect();
        let needs_fallback = covered.len() != findings.len();
        Ok((
            semantics,
            findings,
            InProjectLinks {
                edges: edges.into_iter().collect(),
            },
            needs_fallback,
        ))
    }
}

#[async_trait]
pub trait ExtractionCheckpointSink: Send + Sync {
    async fn save_extraction_chunk(
        &self,
        project_key: &str,
        stage: &str,
        chunk_idx: usize,
        model: &str,
        content_hash: &str,
        chunk_json: String,
    ) -> Result<()>;
}

#[async_trait]
impl ExtractionCheckpointSink for HistoricalDatabase {
    async fn save_extraction_chunk(
        &self,
        project_key: &str,
        stage: &str,
        chunk_idx: usize,
        model: &str,
        content_hash: &str,
        chunk_json: String,
    ) -> Result<()> {
        HistoricalDatabase::save_extraction_chunk(
            self,
            project_key,
            stage,
            i32::try_from(chunk_idx)
                .map_err(|_| KgError::other("extraction chunk index exceeds i32"))?,
            model,
            content_hash,
            &chunk_json,
        )
        .await
    }
}

impl ProjectData {
    // ── Content hashing for extraction chunk invalidation ──
    //
    // FNV-1a — not cryptographic; only needs to detect source/prompt changes.

    fn content_hash(&self) -> String {
        let mut h: u64 = 0xcbf29ce484222325;
        for &byte in self.build_project_prompt_body().as_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        format!("{h:016x}")
    }

    fn findings_content_hash(&self) -> String {
        let body = self.build_report_prompt_body().unwrap_or_default();
        let mut h: u64 = 0xcbf29ce484222325;
        for &byte in body.as_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        format!("{h:016x}")
    }

    fn narrative_combined_content_hash(&self, suffix: &str, content: &str) -> String {
        let mut h: u64 = 0xcbf29ce484222325;
        for &byte in b"narrative-combined-v1"
            .iter()
            .chain(suffix.as_bytes())
            .chain(content.as_bytes())
        {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        format!("{h:016x}")
    }

    /// Phase 1: Categorize the project and extract semantics.
    /// Safe to run concurrently across multiple projects.
    ///
    /// When `db` is `Some`, per-chunk extraction progress is checkpointed
    /// to the `extraction_chunk` table. On resume, completed chunks are
    /// skipped. Pass `None` for stateless use (e.g. agentic pipelines).
    pub async fn categorize_and_extract(
        &self,
        llm: &LLM,
        agent_options: &AgentRunOptions,
        chunk_input_budget: Option<usize>,
        db: Option<&HistoricalDatabase>,
        force_remove_pending_chunks: bool,
    ) -> Result<ExtractResult> {
        if self.is_narrative {
            return self
                .categorize_and_extract_narrative(
                    llm,
                    agent_options,
                    chunk_input_budget,
                    db,
                    force_remove_pending_chunks,
                )
                .await;
        }

        let pid = self.display_id();

        tracing::info!(
            "Processing project {}: {} ({} source files)",
            pid,
            self.name(),
            self.source_files().len()
        );

        if self.source_files().is_empty() {
            tracing::warn!("No source files found for project {}", pid);
            return Ok(ExtractResult {
                categories: vec![],
                semantics: vec![],
                findings: vec![],
                in_project_links: InProjectLinks::default(),
            });
        }

        let categories = self
            .categorize(llm, agent_options, db, force_remove_pending_chunks)
            .await?;
        tracing::info!("Project {} categorized as: {:?}", pid, categories);

        let known_semantics = self
            .load_known_semantic_chunks(llm, db, force_remove_pending_chunks)
            .await?;
        let known_findings = self
            .load_known_finding_chunks(llm, db, force_remove_pending_chunks)
            .await?;
        let checkpoint_sink = db.map(|db| db as &dyn ExtractionCheckpointSink);
        let (all_semantics, all_findings) = tokio::try_join!(
            self.extract_semantics(
                llm,
                &categories,
                agent_options,
                chunk_input_budget,
                &known_semantics,
                checkpoint_sink,
            ),
            self.extract_findings(
                llm,
                &categories,
                agent_options,
                chunk_input_budget,
                &known_findings,
                checkpoint_sink,
            )
        )?;

        tracing::info!(
            "Extracted {} raw semantics from project {}",
            all_semantics.len(),
            pid
        );

        tracing::info!(
            "Extracted {} raw findings from project {}",
            all_findings.len(),
            pid
        );

        let deduped = Self::dedup_semantics(all_semantics);
        let deduped_findings = Self::dedup_findings(all_findings);
        tracing::info!(
            "After intra-project dedup: {} semantics for project {}",
            deduped.len(),
            pid
        );

        tracing::info!(
            "After intra-project dedup: {} findings for project {}",
            deduped_findings.len(),
            pid
        );

        // Run the in-project linking step before any cross-project merge.
        // Every finding must claim ≥1 same-project semantic; this is what
        // grounds raw findings to raw semantics inside the atomic admission
        // transaction even when both later get merged into canonicals.
        let in_project_links = if deduped_findings.is_empty() || deduped.is_empty() {
            InProjectLinks::default()
        } else {
            self.link_findings_in_project(llm, &categories, &deduped, &deduped_findings)
                .await?
        };
        tracing::info!(
            "In-project linking for project {}: {} edge(s) over {} finding(s)",
            pid,
            in_project_links.edges.len(),
            deduped_findings.len()
        );

        Ok(ExtractResult {
            categories,
            semantics: deduped,
            findings: deduped_findings,
            in_project_links,
        })
    }

    async fn categorize_and_extract_narrative(
        &self,
        llm: &LLM,
        agent_options: &AgentRunOptions,
        chunk_input_budget: Option<usize>,
        db: Option<&HistoricalDatabase>,
        force_remove_pending_chunks: bool,
    ) -> Result<ExtractResult> {
        let pid = self.display_id();
        let categories = self
            .categorize(llm, agent_options, db, force_remove_pending_chunks)
            .await?;
        let system_prompt = prompts::NARRATIVE_ROLE_SYSTEM;
        let user_suffix = prompts::narrative_combined_extract_user_suffix(&categories);
        let model = &llm.model;
        let system_tokens = model.config.count_tokens_lossy(system_prompt);
        let suffix_tokens = model.config.count_tokens_lossy(&user_suffix);
        let total_budget = get_context_budget(model, agent_options.context_window_utilization);
        let chunk_budget = match chunk_input_budget {
            Some(cap) => cap.min(total_budget.saturating_sub(system_tokens + suffix_tokens)),
            None => total_budget.saturating_sub(system_tokens + suffix_tokens),
        };
        if chunk_budget == 0 {
            return Err(KgError::other(
                "narrative extraction has no remaining input budget after prompt overhead",
            ));
        }

        let content = self.build_project_prompt_body();
        let content_hash = self.narrative_combined_content_hash(&user_suffix, &content);
        let stage = "narrative_combined";
        if let Some(db) = db {
            let legacy_semantics = db.load_extraction_chunks(&pid, "semantics").await?;
            let legacy_findings = db.load_extraction_chunks(&pid, "findings").await?;
            if !legacy_semantics.is_empty() || !legacy_findings.is_empty() {
                if !force_remove_pending_chunks {
                    return Err(KgError::other(format!(
                        "legacy narrative extraction checkpoints exist for {pid}; rerun with --force-remove-pending-chunks to discard them"
                    )));
                }
                db.clear_extraction_chunks(&pid, "semantics").await?;
                db.clear_extraction_chunks(&pid, "findings").await?;
            }
        }
        let checkpoint_rows = if let Some(db) = db {
            let rows = db.load_extraction_chunks(&pid, stage).await?;
            if !rows.is_empty()
                && !db
                    .extraction_chunks_match(&pid, stage, model.model_id_str(), &content_hash)
                    .await?
            {
                if !force_remove_pending_chunks {
                    return Err(KgError::other(format!(
                        "narrative extraction checkpoints for {pid} do not match this run; rerun with --force-remove-pending-chunks to discard them"
                    )));
                }
                db.clear_extraction_chunks(&pid, stage).await?;
                Vec::new()
            } else {
                rows
            }
        } else {
            Vec::new()
        };

        let Some(mut cursor) = TokenCursor::new(content, model.clone()) else {
            return Err(KgError::other(
                "Failed to initialize TokenCursor for narrative extraction",
            ));
        };
        let mut raw = NarrativeRawExtraction::default();
        for (expected_idx, row) in checkpoint_rows.iter().enumerate() {
            let row_idx = usize::try_from(row.chunk_idx)
                .map_err(|_| KgError::other("negative narrative extraction chunk index"))?;
            if row_idx != expected_idx {
                return Err(KgError::other(format!(
                    "narrative extraction checkpoints for {pid} are not contiguous"
                )));
            }
            let chunk: NarrativeCombinedChunk = serde_json::from_str(&row.chunk_json)?;
            raw.append_chunk(row_idx, chunk)?;
            if cursor.next_chunk(chunk_budget).is_none() {
                return Err(KgError::other(format!(
                    "narrative extraction checkpoint {row_idx} exceeds the current report content"
                )));
            }
        }

        let checkpoint_sink = db.map(|db| db as &dyn ExtractionCheckpointSink);
        let mut chunk_idx = checkpoint_rows.len();
        while let Some(chunk) = cursor.next_chunk(chunk_budget) {
            let user_prompt = format!("{}{}", chunk, user_suffix);
            tracing::info!(
                "Extracting combined narrative chunk {} (~{} tokens, done={})",
                chunk_idx,
                system_tokens + model.config.count_tokens_lossy(&user_prompt),
                cursor.is_done(),
            );
            let extractor = NarrativeChunkExtractor {
                llm: llm.clone(),
                options: agent_options.scoped(&format!("narrative-combined-chunk{chunk_idx}")),
                system_prompt: system_prompt.to_string(),
                user_prompt,
                cache_key: format!(
                    "{}-narrative-combined-chunk{chunk_idx}",
                    self.prompt_cache_key()
                ),
                label: format!("narrative-extract-{}-chunk{chunk_idx}", self.display_id()),
            };
            let items = extractor.run().await?;
            let checkpoint = NarrativeCombinedChunk { items };
            raw.append_chunk(chunk_idx, checkpoint.clone())?;
            if let Some(checkpoint_sink) = checkpoint_sink {
                checkpoint_sink
                    .save_extraction_chunk(
                        &pid,
                        stage,
                        chunk_idx,
                        model.model_id_str(),
                        &content_hash,
                        serde_json::to_string(&checkpoint)?,
                    )
                    .await?;
            }
            chunk_idx += 1;
        }

        let (semantics, findings, mut in_project_links, needs_fallback) =
            raw.into_extract_parts()?;
        if needs_fallback && !findings.is_empty() && !semantics.is_empty() {
            let link_manifest = (&categories, &semantics, &findings);
            let link_hash = Self::serialized_hash(&link_manifest)?;
            let fallback = if let Some(db) = db {
                if let Some(mut links) = self
                    .load_merge_checkpoint::<InProjectLinks>(
                        db,
                        "narrative_link",
                        llm.model.model_id_str(),
                        &link_hash,
                    )
                    .await?
                {
                    match links.pop() {
                        Some(links) => links,
                        None => {
                            return Err(KgError::other(
                                "narrative link checkpoint contained no link result",
                            ));
                        }
                    }
                } else {
                    let links = self
                        .link_findings_in_project(llm, &categories, &semantics, &findings)
                        .await?;
                    db.save_extraction_chunk(
                        &pid,
                        "narrative_link",
                        0,
                        llm.model.model_id_str(),
                        &link_hash,
                        &serde_json::to_string(&vec![links.clone()])?,
                    )
                    .await?;
                    links
                }
            } else {
                self.link_findings_in_project(llm, &categories, &semantics, &findings)
                    .await?
            };
            let mut edges = in_project_links.edges.into_iter().collect::<BTreeSet<_>>();
            edges.extend(fallback.edges);
            in_project_links.edges = edges.into_iter().collect();
        }
        if !findings.is_empty() {
            let linked: BTreeSet<usize> = in_project_links
                .edges
                .iter()
                .map(|(finding_idx, _)| *finding_idx)
                .collect();
            if linked.len() != findings.len() {
                return Err(KgError::other(format!(
                    "narrative extraction left {} finding(s) without a semantic link",
                    findings.len().saturating_sub(linked.len())
                )));
            }
        }
        tracing::info!(
            "Combined narrative extraction for {} produced {} semantic(s), {} finding(s), and {} link(s)",
            pid,
            semantics.len(),
            findings.len(),
            in_project_links.edges.len(),
        );
        Ok(ExtractResult {
            categories,
            semantics,
            findings,
            in_project_links,
        })
    }

    /// Categorize the project and extract only project semantics.
    /// This is used by consumers that need semantic context but do not need audit findings.
    pub async fn categorize_and_extract_semantics(
        &self,
        llm: &LLM,
        agent_options: &AgentRunOptions,
        chunk_input_budget: Option<usize>,
        db: Option<&HistoricalDatabase>,
        force_remove_pending_chunks: bool,
    ) -> Result<ExtractResult> {
        let pid = self.display_id();

        tracing::info!(
            "Processing project {} for semantics only: {} ({} source files)",
            pid,
            self.name(),
            self.source_files().len()
        );

        if self.source_files().is_empty() {
            tracing::warn!("No source files found for project {}", pid);
            return Ok(ExtractResult {
                categories: vec![],
                semantics: vec![],
                findings: vec![],
                in_project_links: InProjectLinks::default(),
            });
        }

        let categories = self
            .categorize(llm, agent_options, db, force_remove_pending_chunks)
            .await?;
        tracing::info!("Project {} categorized as: {:?}", pid, categories);

        let known_semantics = self
            .load_known_semantic_chunks(llm, db, force_remove_pending_chunks)
            .await?;
        let checkpoint_sink = db.map(|db| db as &dyn ExtractionCheckpointSink);
        let all_semantics = self
            .extract_semantics(
                llm,
                &categories,
                agent_options,
                chunk_input_budget,
                &known_semantics,
                checkpoint_sink,
            )
            .await?;
        tracing::info!(
            "Extracted {} raw semantics from project {}",
            all_semantics.len(),
            pid
        );

        let deduped = Self::dedup_semantics(all_semantics);
        tracing::info!(
            "After intra-project dedup: {} semantics for project {}",
            deduped.len(),
            pid
        );

        Ok(ExtractResult {
            categories,
            semantics: deduped,
            findings: Vec::new(),
            in_project_links: InProjectLinks::default(),
        })
    }

    /// Phase 2: Merge extracted semantics with existing KB and write to DB.
    /// MUST be run serially (one project at a time) to avoid merge conflicts.
    /// Commit one project's full merge output to the historical KG.
    ///
    /// Begins its own transaction via
    /// [`HistoricalDatabase::write_project_completed`] under the
    /// hood and discards the new-canonical id list. Use this when
    /// the caller doesn't need to compose additional writes
    /// (`pending_semantic` enqueue etc.) inside the same
    /// transaction. Bulk learn paths (`learn moves` / `learn c4`
    /// / `learn projects`) go through here because they never run
    /// retro-link.
    pub async fn merge_and_write(
        &self,
        db: &HistoricalDatabase,
        llm: &LLM,
        extract: &ExtractResult,
        agent_options: &AgentRunOptions,
        merge_chunking: MergeChunkingOptions,
    ) -> Result<()> {
        let txn = db.begin().await?;
        self.merge_and_write_txn(&txn, db, llm, extract, agent_options, merge_chunking)
            .await?;
        txn.commit().await?;
        if let Err(error) = db
            .clear_extraction_chunks_for_project(&self.display_id())
            .await
        {
            tracing::warn!(
                "Project {} committed, but checkpoint cleanup failed: {}",
                self.display_id(),
                error
            );
        }
        Ok(())
    }

    /// Transaction-scoped variant of [`Self::merge_and_write`].
    /// Performs the merge-LLM passes (against the LIVE DB — these
    /// are reads, not writes, so don't depend on the txn) and then
    /// writes the resulting rows through
    /// [`HistoricalDatabase::write_project_completed_txn`] using
    /// the supplied transaction. Returns the canonical semantic
    /// ids this project newly introduced, so the caller can chain
    /// [`HistoricalDatabase::enqueue_pending_canonical_semantics_txn`]
    /// in the same transaction when needed (incremental
    /// `workflow learn` flow).
    pub async fn merge_and_write_txn(
        &self,
        conn: &sea_orm::DatabaseTransaction,
        db: &HistoricalDatabase,
        llm: &LLM,
        extract: &ExtractResult,
        agent_options: &AgentRunOptions,
        merge_chunking: MergeChunkingOptions,
    ) -> Result<Vec<i32>> {
        let pid = self.display_id();

        if extract.semantics.is_empty() && extract.findings.is_empty() {
            let new_canonicals = db
                .write_project_completed_txn_with_source(
                    conn,
                    self.name(),
                    self.platform_id(),
                    &extract.categories,
                    &[],
                    &[],
                    &InProjectLinks::default(),
                    self.feed_source.as_ref(),
                )
                .await?;
            tracing::info!("Project {} written (no semantics or findings)", pid);
            return Ok(new_canonicals);
        }

        let semantic_merge_results = self
            .merge_with_existing(db, llm, extract, agent_options, merge_chunking)
            .await?;
        let finding_merge_results = self
            .merge_findings_with_existing(db, llm, extract, agent_options, merge_chunking)
            .await?;

        let new_canonicals = db
            .write_project_completed_txn_with_source(
                conn,
                self.name(),
                self.platform_id(),
                &extract.categories,
                &semantic_merge_results,
                &finding_merge_results,
                &extract.in_project_links,
                self.feed_source.as_ref(),
            )
            .await?;

        tracing::info!("Project {} fully processed and saved", pid);
        Ok(new_canonicals)
    }

    /// Check if this project is already completed in the DB.
    pub async fn is_completed(&self, db: &HistoricalDatabase) -> Result<bool> {
        if let Some(source) = &self.feed_source {
            return db.is_feed_report_current(source).await;
        }
        if let Some(pid) = self.platform_id() {
            db.is_project_completed(pid).await
        } else {
            Ok(db
                .get_project_by_name(self.name())
                .await?
                .map(|p| p.status == "completed")
                .unwrap_or(false))
        }
    }

    fn build_project_prompt_body(&self) -> String {
        if self.is_narrative {
            let mut content = prompts::narrative_project_user_prefix();
            for file in self.source_files() {
                content.push_str(&format!(
                    "### {}\n\n{}\n\n",
                    file.relative_path.display(),
                    file.content
                ));
            }
            content
        } else {
            let mut content = prompts::project_user_prefix();
            content.push_str("## Source Files\n\n");

            for file in self.source_files() {
                content.push_str(&format!(
                    "### {}\n```{}\n{}\n```\n\n",
                    file.relative_path.display(),
                    self.source_language().code_fence(),
                    file.content
                ));
            }

            if let Some(readme) = self.load_readme() {
                content.push_str("## README\n\n");
                content.push_str(&readme);
                content.push_str("\n\n");
            }

            content
        }
    }

    fn build_report_prompt_body(&self) -> Option<String> {
        let report = self.audit_report()?.render();
        let prefix = if self.is_narrative {
            prompts::narrative_report_user_prefix()
        } else {
            prompts::report_user_prefix()
        };
        let mut content = prefix;
        content.push_str(&report);
        content.push_str("\n\n");
        Some(content)
    }

    fn load_readme(&self) -> Option<String> {
        for name in &["README.md", "readme.md", "Readme.md"] {
            let readme_path = self.root_dir().join(name);
            if readme_path.exists()
                && let Ok(readme) = std::fs::read_to_string(&readme_path)
            {
                return Some(readme);
            }
        }

        None
    }

    fn prompt_cache_key(&self) -> String {
        sanitize_prompt_prefix(&self.display_id())
    }

    fn debug_key(&self, stage: &str) -> String {
        format!(
            "{}-{}",
            sanitize_prompt_prefix(stage),
            self.prompt_cache_key()
        )
    }

    fn merge_cache_key(&self) -> String {
        format!("{}-merge", self.prompt_cache_key())
    }

    fn finding_cache_key(&self) -> String {
        format!("{}-finding", self.prompt_cache_key())
    }

    fn finding_merge_cache_key(&self) -> String {
        format!("{}-finding-merge", self.prompt_cache_key())
    }

    // ── Private pipeline steps ──────────────────────────────────────

    /// Categorize the project via an `Agent` with two tools
    /// (`set_project_categories` + `finalize_categorization`). Fills the
    /// context window with the README and as many source files as fit.
    async fn categorize(
        &self,
        llm: &LLM,
        agent_options: &AgentRunOptions,
        db: Option<&HistoricalDatabase>,
        force_remove_pending_chunks: bool,
    ) -> Result<Vec<DeFiCategory>> {
        let started_at = Instant::now();
        let model = &llm.model;
        let (system_prompt, user_suffix): (&str, &str) = if self.is_narrative {
            (
                prompts::NARRATIVE_ROLE_SYSTEM,
                prompts::NARRATIVE_CATEGORIZE_USER_SUFFIX,
            )
        } else {
            (
                prompts::GENERAL_ROLE_SYSTEM,
                prompts::CATEGORIZE_USER_SUFFIX,
            )
        };
        let cache_key = sanitize_prompt_prefix(&self.display_id());
        let sys_tokens = model.config.count_tokens_lossy(system_prompt);
        let suffix_tokens = model.config.count_tokens_lossy(user_suffix);
        let budget = get_context_budget(model, agent_options.context_window_utilization)
            .saturating_sub(sys_tokens + suffix_tokens);
        let content = self.build_project_prompt_body();
        let content_hash = self.content_hash();

        // ── DB checkpoint: skip categorize if already saved ──
        if let Some(db) = db {
            let chunks = db
                .load_extraction_chunks(&self.display_id(), "categorize")
                .await?;
            if !chunks.is_empty()
                && db
                    .extraction_chunks_match(
                        &self.display_id(),
                        "categorize",
                        model.model_id_str(),
                        &content_hash,
                    )
                    .await?
            {
                let cats: Vec<DeFiCategory> = serde_json::from_str(&chunks[0].chunk_json)?;
                tracing::info!(
                    "categorize checkpoint hit for {} — skipped LLM call ({} categories)",
                    self.display_id(),
                    cats.len(),
                );
                return Ok(cats);
            }
            if !chunks.is_empty() {
                if !force_remove_pending_chunks {
                    return Err(KgError::other(format!(
                        "categorize checkpoint for {} does not match this run; rerun with --force-remove-pending-chunks to discard it",
                        self.display_id()
                    )));
                }
                tracing::info!(
                    "removing stale categorize checkpoint for {}",
                    self.display_id()
                );
                db.clear_extraction_chunks(&self.display_id(), "categorize")
                    .await?;
            }
        }

        tracing::info!(
            "categorize preparing {}: source_files={}, body_chars={}, budget={}",
            self.display_id(),
            self.source_files().len(),
            content.len(),
            budget,
        );

        let Some(mut cursor) = TokenCursor::new(content, model.clone()) else {
            return Err(KgError::other(
                "Failed to initialize TokenCursor for categorization",
            ));
        };
        let user_prompt = format!("{}{}", cursor.next_chunk(budget).unwrap_or(""), user_suffix,);
        tracing::info!(
            "Categorization prompt: ~{} tokens (budget: {})",
            sys_tokens + model.config.count_tokens_lossy(&user_prompt),
            sys_tokens + budget,
        );

        let label = format!("categorize-{}", self.display_id());
        let local_options = agent_options.scoped(&self.debug_key("categorize"));
        let runner = CategorizeRunner {
            llm: llm.clone(),
            options: local_options,
            system_prompt: system_prompt.to_string(),
            user_prompt,
            cache_key,
            label,
        };
        let record = runner.run().await?;

        // ── Save categorize checkpoint ──
        if let Some(db) = db {
            let json = serde_json::to_string(&record.categories)?;
            db.save_extraction_chunk(
                &self.display_id(),
                "categorize",
                0,
                model.model_id_str(),
                &content_hash,
                &json,
            )
            .await?;
        }

        tracing::info!(
            "categorize finished for {} in {:?}: categories={:?} ({})",
            self.display_id(),
            started_at.elapsed(),
            record.categories,
            record.reasoning,
        );
        Ok(record.categories)
    }

    /// Extract semantics from the project's source files. Splits the source
    /// text into chunks that fit the context window and runs one Agent per
    /// chunk; each chunk's semantics are emitted via `emit_semantic` tool
    /// calls and the chunk is closed with `finalize_semantic_extraction`.
    async fn load_known_semantic_chunks(
        &self,
        llm: &LLM,
        db: Option<&HistoricalDatabase>,
        force_remove_pending_chunks: bool,
    ) -> Result<Vec<KnownExtractedChunk<ExtractedSemantic>>> {
        let Some(db) = db else {
            return Ok(Vec::new());
        };
        let hash = self.content_hash();
        let model = &llm.model;
        if !db
            .extraction_chunks_match(&self.display_id(), "semantics", model.model_id_str(), &hash)
            .await?
        {
            if !force_remove_pending_chunks {
                return Err(KgError::other(format!(
                    "semantic extraction checkpoints for {} do not match this run; rerun with --force-remove-pending-chunks to discard them",
                    self.display_id()
                )));
            }
            db.clear_extraction_chunks(&self.display_id(), "semantics")
                .await?;
            return Ok(Vec::new());
        }
        db.load_extraction_chunks(&self.display_id(), "semantics")
            .await?
            .into_iter()
            .map(|row| {
                Ok(KnownExtractedChunk {
                    chunk_idx: usize::try_from(row.chunk_idx)
                        .map_err(|_| KgError::other("negative extraction chunk index"))?,
                    results: serde_json::from_str(&row.chunk_json)?,
                })
            })
            .collect()
    }

    pub async fn extract_semantics(
        &self,
        llm: &LLM,
        categories: &[DeFiCategory],
        agent_options: &AgentRunOptions,
        chunk_input_budget: Option<usize>,
        known_chunks: &[KnownExtractedChunk<ExtractedSemantic>],
        checkpoint_sink: Option<&dyn ExtractionCheckpointSink>,
    ) -> Result<Vec<ExtractedSemantic>> {
        let system_prompt = if self.is_narrative {
            prompts::NARRATIVE_ROLE_SYSTEM
        } else {
            prompts::GENERAL_ROLE_SYSTEM
        };
        let model = &llm.model;
        let debug_key = self.debug_key("extract");
        let cache_key_root = self.prompt_cache_key();
        let sys_tokens = model.config.count_tokens_lossy(system_prompt);
        let total_budget = get_context_budget(model, agent_options.context_window_utilization);
        let user_suffix = if self.is_narrative {
            prompts::narrative_extract_semantics_user_suffix(categories)
        } else {
            prompts::extract_semantics_user_suffix(categories)
        };
        let suffix_tokens = model.config.count_tokens_lossy(&user_suffix);

        // Caller-overridable per-chunk input budget; default = ~80% of
        // model max input minus the fixed system + suffix overhead.
        let chunk_budget = match chunk_input_budget {
            Some(cap) => cap.min(total_budget.saturating_sub(sys_tokens + suffix_tokens)),
            None => total_budget.saturating_sub(sys_tokens + suffix_tokens),
        };

        let all_files = self.build_project_prompt_body();
        let content_hash = self.content_hash();
        let Some(mut cursor) = TokenCursor::new(all_files.clone(), model.clone()) else {
            return Err(KgError::other(
                "Failed to initialize TokenCursor for extraction",
            ));
        };

        let mut all_semantics = known_chunks
            .iter()
            .flat_map(|chunk| chunk.results.clone())
            .collect::<Vec<_>>();
        let mut chunk_idx = known_chunks.len();
        for _ in known_chunks {
            let _ = cursor.next_chunk(chunk_budget);
        }

        while let Some(chunk) = cursor.next_chunk(chunk_budget) {
            let user_prompt = format!("{}{}", chunk, user_suffix);
            tracing::info!(
                "Extracting semantics from chunk {} (~{} tokens, done={})",
                chunk_idx,
                sys_tokens + model.config.count_tokens_lossy(&user_prompt),
                cursor.is_done(),
            );
            let chunk_label = format!("semantic-extract-{}-chunk{}", self.display_id(), chunk_idx);
            let chunk_debug = format!("{}-chunk{}", debug_key, chunk_idx);
            let local_options = agent_options.scoped(&chunk_debug);
            let extractor = SemanticChunkExtractor {
                llm: llm.clone(),
                options: local_options,
                system_prompt: system_prompt.to_string(),
                user_prompt,
                cache_key: format!("{}-chunk{}", cache_key_root, chunk_idx),
                label: chunk_label,
            };
            let chunk_semantics = extractor.run().await?;

            // ── Save chunk checkpoint ──
            if let Some(checkpoint_sink) = checkpoint_sink {
                let json = serde_json::to_string(&chunk_semantics)?;
                checkpoint_sink
                    .save_extraction_chunk(
                        &self.display_id(),
                        "semantics",
                        chunk_idx,
                        model.model_id_str(),
                        &content_hash,
                        json,
                    )
                    .await?;
            }

            tracing::info!(
                "Chunk {} produced {} semantic(s)",
                chunk_idx,
                chunk_semantics.len(),
            );
            all_semantics.extend(chunk_semantics);
            chunk_idx += 1;
        }
        Ok(all_semantics)
    }

    /// LLM-driven in-project linking. Returns positional `(finding_idx,
    /// semantic_idx)` edges. Validates that every finding got at least one
    /// link (the LLM is instructed to enforce this; we re-check and bail
    /// on violations rather than silently produce an unlinked finding).
    async fn link_findings_in_project(
        &self,
        llm: &LLM,
        categories: &[DeFiCategory],
        semantics: &[ExtractedSemantic],
        findings: &[ExtractedFinding],
    ) -> Result<InProjectLinks> {
        debug_assert!(!findings.is_empty() && !semantics.is_empty());

        let semantics_block = render_semantics_for_in_project_link(semantics);
        let findings_block = render_findings_for_in_project_link(findings);
        let user_msg =
            prompts::in_project_link_user_message(categories, &semantics_block, &findings_block);

        let debug_key = self.debug_key("in-project-link");
        let cache_key = format!("{}-in-project-link", self.prompt_cache_key());

        let parsed: InProjectLinkResponse = llm
            .prompt_json_with_retry(
                prompts::GENERAL_ROLE_SYSTEM,
                &user_msg,
                Some(&debug_key),
                Some(&cache_key),
                None,
            )
            .await?;

        let mut edges: Vec<(usize, usize)> = Vec::new();
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        let mut covered: HashSet<usize> = HashSet::new();
        for entry in &parsed.links {
            let f_idx = entry.finding_index;
            if f_idx >= findings.len() {
                return Err(KgError::other(format!(
                    "in-project link response references unknown finding_index {}",
                    f_idx
                )));
            }
            if entry.semantic_indices.is_empty() {
                return Err(KgError::other(format!(
                    "in-project link response left finding_index {} unlinked (rule violation)",
                    f_idx
                )));
            }
            for s_idx in &entry.semantic_indices {
                if *s_idx >= semantics.len() {
                    return Err(KgError::other(format!(
                        "in-project link response references unknown semantic_index {} for finding {}",
                        s_idx, f_idx
                    )));
                }
                if seen.insert((f_idx, *s_idx)) {
                    edges.push((f_idx, *s_idx));
                }
            }
            covered.insert(f_idx);
        }
        if covered.len() != findings.len() {
            let missing: Vec<usize> = (0..findings.len())
                .filter(|i| !covered.contains(i))
                .collect();
            return Err(KgError::other(format!(
                "in-project link response missing {} finding(s): {:?}",
                missing.len(),
                missing
            )));
        }

        Ok(InProjectLinks { edges })
    }

    /// Extract audit findings from the project's report. Same chunked
    /// agent pattern as [`Self::extract_semantics`]: one Agent per chunk,
    /// `emit_finding` tool per finding, terminated by
    /// `finalize_finding_extraction`.
    async fn load_known_finding_chunks(
        &self,
        llm: &LLM,
        db: Option<&HistoricalDatabase>,
        force_remove_pending_chunks: bool,
    ) -> Result<Vec<KnownExtractedChunk<ExtractedFinding>>> {
        let (Some(db), Some(_)) = (db, self.build_report_prompt_body()) else {
            return Ok(Vec::new());
        };
        let hash = self.findings_content_hash();
        if !db
            .extraction_chunks_match(
                &self.display_id(),
                "findings",
                llm.model.model_id_str(),
                &hash,
            )
            .await?
        {
            if !force_remove_pending_chunks {
                return Err(KgError::other(format!(
                    "finding extraction checkpoints for {} do not match this run; rerun with --force-remove-pending-chunks to discard them",
                    self.display_id()
                )));
            }
            db.clear_extraction_chunks(&self.display_id(), "findings")
                .await?;
            return Ok(Vec::new());
        }
        db.load_extraction_chunks(&self.display_id(), "findings")
            .await?
            .into_iter()
            .map(|row| {
                let results = serde_json::from_str::<Vec<ExtractedFinding>>(&row.chunk_json)?;
                Ok(KnownExtractedChunk {
                    chunk_idx: usize::try_from(row.chunk_idx)
                        .map_err(|_| KgError::other("negative extraction chunk index"))?,
                    results,
                })
            })
            .collect()
    }

    async fn extract_findings(
        &self,
        llm: &LLM,
        categories: &[DeFiCategory],
        agent_options: &AgentRunOptions,
        chunk_input_budget: Option<usize>,
        known_chunks: &[KnownExtractedChunk<ExtractedFinding>],
        checkpoint_sink: Option<&dyn ExtractionCheckpointSink>,
    ) -> Result<Vec<ExtractedFinding>> {
        let Some(report_body) = self.build_report_prompt_body() else {
            tracing::warn!("No audit report found for project {}", self.display_id());
            return Ok(Vec::new());
        };

        let system_prompt = if self.is_narrative {
            prompts::NARRATIVE_ROLE_SYSTEM
        } else {
            prompts::GENERAL_ROLE_SYSTEM
        };
        let model = &llm.model;
        let debug_key = self.debug_key("finding-extract");
        let cache_key_root = self.finding_cache_key();
        let sys_tokens = model.config.count_tokens_lossy(system_prompt);
        let total_budget = get_context_budget(model, agent_options.context_window_utilization);
        let user_suffix = if self.is_narrative {
            prompts::narrative_extract_findings_user_suffix(categories)
        } else {
            prompts::extract_findings_user_suffix(categories)
        };
        let suffix_tokens = model.config.count_tokens_lossy(&user_suffix);
        let chunk_budget = match chunk_input_budget {
            Some(cap) => cap.min(total_budget.saturating_sub(sys_tokens + suffix_tokens)),
            None => total_budget.saturating_sub(sys_tokens + suffix_tokens),
        };

        let Some(mut cursor) = TokenCursor::new(report_body.clone(), model.clone()) else {
            return Err(KgError::other(
                "Failed to initialize TokenCursor for finding extraction",
            ));
        };

        let content_hash = self.findings_content_hash();
        let mut all_findings = Vec::new();
        for chunk in known_chunks {
            for finding in &chunk.results {
                all_findings.push(Self::canonicalize_finding(finding.clone())?);
            }
            let _ = cursor.next_chunk(chunk_budget);
        }
        let mut chunk_idx = known_chunks.len();

        while let Some(chunk) = cursor.next_chunk(chunk_budget) {
            let user_prompt = format!("{}{}", chunk, user_suffix);
            tracing::info!(
                "Extracting findings from chunk {} (~{} tokens, done={})",
                chunk_idx,
                sys_tokens + model.config.count_tokens_lossy(&user_prompt),
                cursor.is_done(),
            );
            let chunk_label = format!("finding-extract-{}-chunk{}", self.display_id(), chunk_idx);
            let chunk_debug = format!("{}-chunk{}", debug_key, chunk_idx);
            let local_options = agent_options.scoped(&chunk_debug);
            let extractor = FindingChunkExtractor {
                llm: llm.clone(),
                options: local_options,
                system_prompt: system_prompt.to_string(),
                user_prompt,
                cache_key: format!("{}-chunk{}", cache_key_root, chunk_idx),
                label: chunk_label,
            };
            let raw_findings = extractor.run().await?;

            // ── Save chunk checkpoint ──
            if let Some(checkpoint_sink) = checkpoint_sink {
                let json = serde_json::to_string(&raw_findings)?;
                checkpoint_sink
                    .save_extraction_chunk(
                        &self.display_id(),
                        "findings",
                        chunk_idx,
                        model.model_id_str(),
                        &content_hash,
                        json,
                    )
                    .await?;
            }

            tracing::info!(
                "Chunk {} produced {} finding(s)",
                chunk_idx,
                raw_findings.len(),
            );
            for finding in raw_findings {
                all_findings.push(Self::canonicalize_finding(finding)?);
            }
            chunk_idx += 1;
        }

        Ok(all_findings)
    }

    /// Deduplicate semantics by name (case-insensitive). Keeps the longer
    /// description and merges function lists.
    fn dedup_semantics(semantics: Vec<ExtractedSemantic>) -> Vec<ExtractedSemantic> {
        let mut by_name: HashMap<String, ExtractedSemantic> = HashMap::new();

        for sem in semantics {
            let key = sem.name.to_lowercase().trim().to_string();
            if let Some(existing) = by_name.get_mut(&key) {
                for func in sem.functions {
                    let already_has = existing
                        .functions
                        .iter()
                        .any(|f| f.name == func.name && f.contract == func.contract);
                    if !already_has {
                        existing.functions.push(func);
                    }
                }
                if sem.description.len() > existing.description.len() {
                    existing.description = sem.description;
                    existing.definition = sem.definition;
                }
            } else {
                by_name.insert(key, sem);
            }
        }

        by_name.into_values().collect()
    }

    fn dedup_findings(findings: Vec<ExtractedFinding>) -> Vec<ExtractedFinding> {
        let mut by_title: HashMap<String, ExtractedFinding> = HashMap::new();

        for finding in findings {
            let key = finding.title.to_lowercase().trim().to_string();
            if let Some(existing) = by_title.get_mut(&key) {
                existing.severity = existing.severity.max(finding.severity);

                if finding.description.len() > existing.description.len() {
                    existing.category = finding.category;
                    existing.subcategory = finding.subcategory.clone();
                    existing.description = finding.description.clone();
                }

                if finding.root_cause.len() > existing.root_cause.len() {
                    existing.root_cause = finding.root_cause.clone();
                }

                if finding.patterns.len() > existing.patterns.len() {
                    existing.patterns = finding.patterns.clone();
                }

                if finding.exploits.len() > existing.exploits.len() {
                    existing.exploits = finding.exploits.clone();
                }
            } else {
                by_title.insert(key, finding);
            }
        }

        by_title.into_values().collect()
    }

    fn canonicalize_finding(mut finding: ExtractedFinding) -> Result<ExtractedFinding> {
        finding.title = finding.title.trim().to_string();
        finding.root_cause = finding.root_cause.trim().to_string();
        finding.description = finding.description.trim().to_string();
        finding.patterns = finding.patterns.trim().to_string();
        finding.exploits = finding.exploits.trim().to_string();

        let Some(entry) = resolve_taxonomy_entry(finding.category, &finding.subcategory) else {
            return Err(KgError::other(format!(
                "Unknown vulnerability subcategory '{}' for category '{}'",
                finding.subcategory, finding.category
            )));
        };

        finding.subcategory = entry.subcategory.to_string();
        Ok(finding)
    }
}

#[cfg(test)]
mod narrative_tests {
    use super::*;
    use crate::vulnerability::{FindingSeverity, VulnerabilityCategory};
    use knowdit_kg_model::category::DeFiCategory;

    fn semantic(name: &str) -> ExtractedSemantic {
        ExtractedSemantic {
            name: name.to_string(),
            category: DeFiCategory::Others,
            definition: "A reusable exploit mechanism".to_string(),
            description: "Concrete mechanism details".to_string(),
            functions: vec![ExtractedFunction {
                name: "_narrative".to_string(),
                contract: "report.md".to_string(),
                signature: None,
            }],
        }
    }

    fn finding(title: &str, root_cause: &str) -> ExtractedFinding {
        ExtractedFinding {
            title: title.to_string(),
            severity: FindingSeverity::High,
            category: VulnerabilityCategory::AccessControl,
            subcategory: "Missing Function-Level Access Control".to_string(),
            root_cause: root_cause.to_string(),
            description: "Concrete finding details".to_string(),
            patterns: "Concrete pattern".to_string(),
            exploits: "Concrete exploit".to_string(),
        }
    }

    #[test]
    fn narrative_extraction_keeps_distinct_same_title_mechanisms() {
        let raw = NarrativeRawExtraction {
            semantics: vec![NarrativeSemanticRecord {
                id: "sem-0".to_string(),
                semantic: semantic("Governance Capture"),
            }],
            findings: vec![
                NarrativeFindingRecord {
                    id: "finding-0".to_string(),
                    finding: finding("TOP Governance Takeover", "Cheap quorum capture"),
                },
                NarrativeFindingRecord {
                    id: "finding-1".to_string(),
                    finding: finding("TOP Governance Takeover", "Missing timelock"),
                },
            ],
            links: vec![
                NarrativeLinkRecord {
                    finding_id: "finding-0".to_string(),
                    semantic_ids: vec!["sem-0".to_string()],
                },
                NarrativeLinkRecord {
                    finding_id: "finding-1".to_string(),
                    semantic_ids: vec!["sem-0".to_string()],
                },
            ],
        };

        let (semantics, findings, links, needs_fallback) = match raw.into_extract_parts() {
            Ok(parts) => parts,
            Err(error) => panic!("unexpected narrative extraction error: {error}"),
        };
        assert_eq!(semantics.len(), 1);
        assert_eq!(findings.len(), 2);
        assert_eq!(links.edges, vec![(0, 0), (1, 0)]);
        assert!(!needs_fallback);
    }

    #[test]
    fn narrative_extraction_requests_fallback_for_cross_chunk_link_gap() {
        let raw = NarrativeRawExtraction {
            semantics: vec![NarrativeSemanticRecord {
                id: "sem-1".to_string(),
                semantic: semantic("Governance Capture"),
            }],
            findings: vec![NarrativeFindingRecord {
                id: "finding-0".to_string(),
                finding: finding("Governance Takeover", "Cheap quorum capture"),
            }],
            links: Vec::new(),
        };

        let (_, findings, links, needs_fallback) = match raw.into_extract_parts() {
            Ok(parts) => parts,
            Err(error) => panic!("unexpected narrative extraction error: {error}"),
        };
        assert_eq!(findings.len(), 1);
        assert!(links.is_empty());
        assert!(needs_fallback);
    }

    #[test]
    fn merge_results_round_trip_for_retry_checkpoint() {
        let result = MergeResult {
            semantic: semantic("Arithmetic Overflow"),
            action: MergeAction::Merge {
                target_ids: vec![7, 9],
                updated_description: Some("Concrete updated description".to_string()),
                appended_description: Some("Concrete merge delta".to_string()),
            },
        };
        let encoded = match serde_json::to_string(&vec![result.clone()]) {
            Ok(encoded) => encoded,
            Err(error) => panic!("unexpected merge serialization error: {error}"),
        };
        let decoded: Vec<MergeResult> = match serde_json::from_str(&encoded) {
            Ok(decoded) => decoded,
            Err(error) => panic!("unexpected merge deserialization error: {error}"),
        };
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].semantic.name, result.semantic.name);
        assert!(matches!(
            &decoded[0].action,
            MergeAction::Merge { target_ids, .. } if target_ids == &vec![7, 9]
        ));
    }

    #[test]
    fn finding_merge_results_round_trip_for_retry_checkpoint() {
        let result = FindingMergeResult {
            finding: finding("Overflow", "Unchecked arithmetic"),
            action: FindingMergeAction::New,
        };
        let encoded = match serde_json::to_string(&vec![result]) {
            Ok(encoded) => encoded,
            Err(error) => panic!("unexpected finding merge serialization error: {error}"),
        };
        let decoded: Vec<FindingMergeResult> = match serde_json::from_str(&encoded) {
            Ok(decoded) => decoded,
            Err(error) => panic!("unexpected finding merge deserialization error: {error}"),
        };
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].finding.title, "Overflow");
        assert!(matches!(decoded[0].action, FindingMergeAction::New));
    }

    #[test]
    fn narrative_chunk_serialization_preserves_all_item_kinds() {
        let chunk = NarrativeCombinedChunk {
            items: vec![
                NarrativeCombinedItem::Semantic(NarrativeSemanticRecord {
                    id: "sem-0".to_string(),
                    semantic: semantic("Governance Capture"),
                }),
                NarrativeCombinedItem::Finding(NarrativeFindingRecord {
                    id: "finding-0".to_string(),
                    finding: finding("Governance Takeover", "Missing timelock"),
                }),
                NarrativeCombinedItem::Link(NarrativeLinkRecord {
                    finding_id: "finding-0".to_string(),
                    semantic_ids: vec!["sem-0".to_string()],
                }),
            ],
        };
        let encoded = match serde_json::to_string(&chunk) {
            Ok(encoded) => encoded,
            Err(error) => panic!("unexpected narrative serialization error: {error}"),
        };
        let decoded: NarrativeCombinedChunk = match serde_json::from_str(&encoded) {
            Ok(decoded) => decoded,
            Err(error) => panic!("unexpected narrative deserialization error: {error}"),
        };
        assert_eq!(decoded.items.len(), 3);
    }
}

impl ProjectData {
    /// Merge newly-extracted semantics against the historical KB. Chunks
    /// the existing canonicals (with their merged-away raw children
    /// rendered alongside, so the LLM's `updated_*` generalizations can
    /// take prior merges into account) by token budget, runs one merge
    /// agent per chunk in parallel via [`SemanticMerger`], and unions the
    /// per-chunk decisions into the final per-raw [`MergeAction`].
    ///
    /// A new raw can merge into multiple canonicals; that's encoded in
    /// `MergeAction::Merge { target_ids, .. }`.
    async fn merge_with_existing(
        &self,
        db: &HistoricalDatabase,
        llm: &LLM,
        extract: &ExtractResult,
        agent_options: &AgentRunOptions,
        merge_chunking: MergeChunkingOptions,
    ) -> Result<Vec<MergeResult>> {
        if extract.semantics.is_empty() {
            return Ok(Vec::new());
        }
        let semantic_categories: Vec<DeFiCategory> = extract
            .semantics
            .iter()
            .map(|sem| sem.category)
            .unique()
            .collect();
        let candidates = db
            .canonical_semantics_with_children_for_categories(&semantic_categories)
            .await?;
        let manifest = SemanticMergeCheckpointManifest {
            version: "semantic-merge-v1",
            stage: "semantic_merge",
            model: llm.model.model_id_str().to_string(),
            max_agent_steps: agent_options.max_agent_steps,
            context_window_utilization: agent_options.context_window_utilization,
            new_item_token_ratio: merge_chunking.new_item_token_ratio,
            merge_concurrency: merge_chunking.concurrency,
            new_item_batch_size: merge_chunking.new_item_batch_size,
            raw_child_variant_cap: merge_chunking.raw_child_variant_cap,
            raw_child_char_cap: merge_chunking.raw_child_char_cap,
            full_candidate_context: merge_chunking.full_candidate_context,
            candidate_routing: merge_chunking.candidate_routing,
            extracted: extract.semantics.clone(),
            candidates: candidates.clone(),
        };
        let content_hash = Self::serialized_hash(&manifest)?;
        if let Some(results) = self
            .load_merge_checkpoint::<MergeResult>(
                db,
                "semantic_merge",
                llm.model.model_id_str(),
                &content_hash,
            )
            .await?
        {
            return Ok(results);
        }
        let merger = SemanticMerger {
            new_semantics: extract.semantics.clone(),
            candidates,
            llm: llm.clone(),
            agent_options: agent_options.clone(),
            chunking: merge_chunking,
            cache_key_root: self.merge_cache_key(),
            debug_key_root: self.debug_key("merge"),
            label_root: format!("semantic-merge-{}", self.display_id()),
        };
        let aggregated = merger.run().await?;
        let results = Self::semantic_merge_results_from_aggregated(&extract.semantics, aggregated);
        db.save_extraction_chunk(
            &self.display_id(),
            "semantic_merge",
            0,
            llm.model.model_id_str(),
            &content_hash,
            &serde_json::to_string(&results)?,
        )
        .await?;
        Ok(results)
    }

    /// Convert the orchestrator's `(name → AggregatedSemanticMergeDecision)`
    /// list into the persistence layer's `MergeResult` (one per raw, in the
    /// same order as `extracted`).
    fn semantic_merge_results_from_aggregated(
        extracted: &[ExtractedSemantic],
        aggregated: Vec<AggregatedSemanticMergeDecision>,
    ) -> Vec<MergeResult> {
        let by_name: HashMap<String, AggregatedSemanticMergeDecision> = aggregated
            .into_iter()
            .map(|d| (d.new_semantic_name.to_lowercase(), d))
            .collect();
        extracted
            .iter()
            .map(|sem| {
                let action = match by_name.get(&sem.name.to_lowercase()) {
                    Some(d) if !d.merge_target_ids.is_empty() => MergeAction::Merge {
                        target_ids: d.merge_target_ids.clone(),
                        updated_description: d.updated_description.clone(),
                        appended_description: d.appended_description.clone(),
                    },
                    _ => MergeAction::New,
                };
                MergeResult {
                    semantic: sem.clone(),
                    action,
                }
            })
            .collect()
    }

    /// Merge newly-extracted findings against the historical KB. Same
    /// chunked + parallel design as [`Self::merge_with_existing`]; uses
    /// [`FindingMerger`] under the hood. Multi-target merges supported.
    async fn merge_findings_with_existing(
        &self,
        db: &HistoricalDatabase,
        llm: &LLM,
        extract: &ExtractResult,
        agent_options: &AgentRunOptions,
        merge_chunking: MergeChunkingOptions,
    ) -> Result<Vec<FindingMergeResult>> {
        if extract.findings.is_empty() {
            return Ok(Vec::new());
        }
        let finding_categories: Vec<VulnerabilityCategory> = extract
            .findings
            .iter()
            .map(|finding| finding.category)
            .unique()
            .collect();
        let candidates = db
            .canonical_findings_with_children_for_categories(&finding_categories)
            .await?;
        let manifest = FindingMergeCheckpointManifest {
            version: "finding-merge-v1",
            stage: "finding_merge",
            model: llm.model.model_id_str().to_string(),
            max_agent_steps: agent_options.max_agent_steps,
            context_window_utilization: agent_options.context_window_utilization,
            new_item_token_ratio: merge_chunking.new_item_token_ratio,
            merge_concurrency: merge_chunking.concurrency,
            new_item_batch_size: merge_chunking.new_item_batch_size,
            raw_child_variant_cap: merge_chunking.raw_child_variant_cap,
            raw_child_char_cap: merge_chunking.raw_child_char_cap,
            full_candidate_context: merge_chunking.full_candidate_context,
            candidate_routing: merge_chunking.candidate_routing,
            extracted: extract.findings.clone(),
            candidates: candidates.clone(),
        };
        let content_hash = Self::serialized_hash(&manifest)?;
        if let Some(results) = self
            .load_merge_checkpoint::<FindingMergeResult>(
                db,
                "finding_merge",
                llm.model.model_id_str(),
                &content_hash,
            )
            .await?
        {
            return Ok(results);
        }
        let merger = FindingMerger {
            new_findings: extract.findings.clone(),
            candidates,
            llm: llm.clone(),
            agent_options: agent_options.clone(),
            chunking: merge_chunking,
            cache_key_root: self.finding_merge_cache_key(),
            debug_key_root: self.debug_key("finding-merge"),
            label_root: format!("finding-merge-{}", self.display_id()),
        };
        let aggregated = merger.run().await?;
        let results = Self::finding_merge_results_from_aggregated(&extract.findings, aggregated);
        db.save_extraction_chunk(
            &self.display_id(),
            "finding_merge",
            0,
            llm.model.model_id_str(),
            &content_hash,
            &serde_json::to_string(&results)?,
        )
        .await?;
        Ok(results)
    }

    fn finding_merge_results_from_aggregated(
        extracted: &[ExtractedFinding],
        aggregated: Vec<AggregatedFindingMergeDecision>,
    ) -> Vec<FindingMergeResult> {
        let by_title: HashMap<String, AggregatedFindingMergeDecision> = aggregated
            .into_iter()
            .map(|d| (d.new_finding_title.to_lowercase(), d))
            .collect();
        extracted
            .iter()
            .map(|finding| {
                let action = match by_title.get(&finding.title.to_lowercase()) {
                    Some(d) if !d.merge_target_ids.is_empty() => FindingMergeAction::Merge {
                        target_ids: d.merge_target_ids.clone(),
                        updated_description: d.updated_description.clone(),
                        updated_patterns: d.updated_patterns.clone(),
                        updated_exploits: d.updated_exploits.clone(),
                        appended_description: d.appended_description.clone(),
                        appended_patterns: d.appended_patterns.clone(),
                        appended_exploits: d.appended_exploits.clone(),
                    },
                    _ => FindingMergeAction::New,
                };
                FindingMergeResult {
                    finding: finding.clone(),
                    action,
                }
            })
            .collect()
    }
}

/// Default fraction of a model's max input a single prompt may fill, used
/// where no CLI override is threaded in (categorize / extract). Kept well
/// below 1.0 on purpose: over-long prompts dilute the model's attention, so
/// we trade packing efficiency (more, smaller chunks/batches) for sharper
/// per-prompt focus. Overridable per-command — e.g. `merge-kg`'s
/// `--context-window-utilization` threads a value into the merge + link
/// budgets.
pub const DEFAULT_CONTEXT_WINDOW_UTILIZATION: f64 = 0.4;

/// Token budget for a single prompt: `utilization` × the model's max input.
///
/// Falls back to [`knowdit_kg_model::FALLBACK_CONTEXT_WINDOW_TOKENS`] when the
/// model is absent from llmy's registry (`max_input_tokens == 0`), which would
/// otherwise collapse the budget to 0 and produce degenerate one-item chunks.
/// `utilization` is clamped to a sane `(0, 1]` band so a mis-typed CLI value
/// can't zero out (or overflow) the budget.
pub(crate) fn get_context_budget(model: &OpenAIModel, utilization: f64) -> usize {
    knowdit_kg_model::context_budget(model.config.max_input(), utilization)
}

pub(crate) fn sanitize_prompt_prefix(value: &str) -> String {
    let mut out = String::new();
    let mut last_was_dash = false;

    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_was_dash = false;
        } else if !last_was_dash {
            out.push('-');
            last_was_dash = true;
        }

        if out.len() >= 48 {
            break;
        }
    }

    let trimmed = out.trim_matches('-');
    if trimmed.is_empty() {
        "project".to_string()
    } else {
        trimmed.to_string()
    }
}

// Merge-response validation lived here in the JSON-mode era: the prompt
// returned a flat array of decisions which we cross-checked against the
// existing canonical id set. With the agent-tool form, validation is
// inlined into `project_*_merge_results`: decisions referencing an
// unknown id are downgraded to `New` (and logged) rather than rejected.

// ── `Others`-bucket re-classification ────────────────────────────────

/// One LLM decision for a single stranded canonical node.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReclassifyNodeDecision {
    pub semantic_id: i32,
    pub category: String,
}

#[derive(Debug, Deserialize)]
struct ReclassifyBatchResponse {
    nodes: Vec<ReclassifyNodeDecision>,
}

/// Re-classify every canonical semantic parked in `Others` into a real
/// category. Renders each node with up to three linked finding titles as
/// coverage evidence, batches them, and runs one JSON-mode LLM call per
/// batch. `Others` answers are dropped with a warning — the pass's whole
/// point is emptying that bucket.
///
/// Returns the validated `(semantic_id, category)` decisions; the caller
/// decides whether to apply them (dry-run vs `--apply`).
pub async fn reclassify_others(
    db: &HistoricalDatabase,
    llm: &LLM,
    batch_size: usize,
    context_window_utilization: f64,
) -> Result<Vec<ReclassifyNodeDecision>> {
    use crate::prompts;

    let nodes = db
        .canonical_semantics_in_category(DeFiCategory::Others)
        .await?;
    if nodes.is_empty() {
        tracing::info!("No canonical semantics in `Others`; nothing to reclassify.");
        return Ok(Vec::new());
    }

    // Coverage evidence: top linked findings per node (title + root cause).
    let linked = db
        .findings_for_semantic_ids(&nodes.iter().map(|n| n.id).collect::<Vec<_>>())
        .await?;
    let mut findings_by_semantic: HashMap<i32, Vec<(String, String)>> = HashMap::new();
    for lf in linked {
        findings_by_semantic
            .entry(lf.canonical_semantic_id)
            .or_default()
            .push((lf.finding.title.clone(), lf.finding.root_cause.clone()));
    }
    for entries in findings_by_semantic.values_mut() {
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries.dedup_by(|a, b| a.0 == b.0);
        entries.truncate(3);
    }

    let model = &llm.model;
    let system_prompt = prompts::RECLASSIFY_SEMANTIC_SYSTEM;
    let budget = crate::learn::get_context_budget(model, context_window_utilization)
        .saturating_sub(model.config.count_tokens_lossy(system_prompt))
        .saturating_sub(
            model
                .config
                .count_tokens_lossy(prompts::RECLASSIFY_SEMANTIC_USER_HEADER),
        )
        .saturating_sub(
            model
                .config
                .count_tokens_lossy(prompts::RECLASSIFY_SEMANTIC_USER_FOOTER),
        );

    let render_node = |node: &knowdit_kg_model::db::semantic_node::Model| -> String {
        let evidence = findings_by_semantic
            .get(&node.id)
            .map(|entries| {
                entries
                    .iter()
                    .map(|(title, root_cause)| {
                        format!(
                            "- finding \"{title}\" — root cause: {}",
                            root_cause.trim().chars().take(200).collect::<String>()
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_else(|| "- (no linked findings)".to_string());
        format!(
            "### semantic_id={}\nname: {}\ndefinition: {}\ndescription: {}\ncoverage evidence:\n{}\n",
            node.id,
            node.name,
            node.definition.trim(),
            node.description.trim(),
            evidence,
        )
    };

    // ── DB checkpoints: reuse decisions from a previous run (e.g. a
    //    dry-run that was never applied) so `--apply` doesn't re-bill
    //    the LLM. Keyed per node id under project_key
    //    "reclassify-others", stage "reclassify".
    let checkpoint_key = "reclassify-others";
    let checkpoint_stage = "reclassify";
    let mut decisions: Vec<ReclassifyNodeDecision> = Vec::new();
    let mut pending: Vec<knowdit_kg_model::db::semantic_node::Model> = Vec::new();
    let checkpoint_chunks = db
        .load_extraction_chunks(checkpoint_key, checkpoint_stage)
        .await?;
    for node in &nodes {
        let hit = checkpoint_chunks.iter().find(|chunk| {
            chunk
                .chunk_json
                .contains(&format!("\"semantic_id\":{}", node.id))
        });
        if let Some(chunk) = hit {
            if let Ok(entry) = serde_json::from_str::<ReclassifyNodeDecision>(&chunk.chunk_json) {
                tracing::info!(
                    "reclassify checkpoint hit for sem-{} — skipped LLM call ({})",
                    node.id,
                    entry.category
                );
                decisions.push(entry);
                continue;
            }
        }
        pending.push(node.clone());
    }

    let mut batch: Vec<String> = Vec::new();
    let mut batch_nodes: Vec<knowdit_kg_model::db::semantic_node::Model> = Vec::new();
    let mut batch_tokens = 0usize;
    for node in pending {
        let record = render_node(&node);
        let record_tokens = model.config.count_tokens_lossy(&record);
        if !batch.is_empty()
            && (batch.len() >= batch_size.max(1) || batch_tokens + record_tokens > budget)
        {
            let batch_decisions = run_reclassify_batch(llm, &batch_nodes, &batch).await?;
            persist_reclassify_checkpoints(
                db,
                checkpoint_key,
                checkpoint_stage,
                model.model_id_str(),
                &batch_decisions,
            )
            .await?;
            decisions.extend(batch_decisions);
            batch.clear();
            batch_nodes.clear();
            batch_tokens = 0;
        }
        batch_tokens += record_tokens;
        batch.push(record);
        batch_nodes.push(node);
    }
    if !batch.is_empty() {
        let batch_decisions = run_reclassify_batch(llm, &batch_nodes, &batch).await?;
        persist_reclassify_checkpoints(
            db,
            checkpoint_key,
            checkpoint_stage,
            model.model_id_str(),
            &batch_decisions,
        )
        .await?;
        decisions.extend(batch_decisions);
    }

    Ok(decisions)
}

/// Persist one reclassify decision per `extraction_chunk` row, so a
/// follow-up run (e.g. `--apply` after a dry-run) reuses the LLM verdict
/// instead of re-calling the model.
async fn persist_reclassify_checkpoints(
    db: &HistoricalDatabase,
    project_key: &str,
    stage: &str,
    model: &str,
    decisions: &[ReclassifyNodeDecision],
) -> Result<()> {
    for decision in decisions {
        let json = serde_json::to_string(decision)?;
        db.save_extraction_chunk(
            project_key,
            stage,
            decision.semantic_id,
            model,
            "reclassify",
            &json,
        )
        .await?;
    }
    Ok(())
}

/// One JSON-mode LLM call for a batch of stranded nodes. Validates ids,
/// parses categories, and drops `Others` answers.
async fn run_reclassify_batch(
    llm: &LLM,
    batch: &[knowdit_kg_model::db::semantic_node::Model],
    records: &[String],
) -> Result<Vec<ReclassifyNodeDecision>> {
    use crate::prompts;

    let batch_ids: HashSet<i32> = batch.iter().map(|n| n.id).collect();
    let records = records.join("\n");
    let user_prompt = format!(
        "{}\n\n## Semantic nodes to re-classify\n\n{}{}",
        prompts::RECLASSIFY_SEMANTIC_USER_HEADER,
        records,
        prompts::RECLASSIFY_SEMANTIC_USER_FOOTER,
    );

    let cache_key = format!(
        "reclassify-others-{}",
        batch.iter().map(|n| n.id).min().unwrap_or(0)
    );
    let parsed: ReclassifyBatchResponse = llm
        .prompt_json_with_retry(
            prompts::RECLASSIFY_SEMANTIC_SYSTEM,
            &user_prompt,
            None,
            Some(&cache_key),
            None,
        )
        .await?;

    let mut out: Vec<ReclassifyNodeDecision> = Vec::new();
    let mut seen: HashSet<i32> = HashSet::new();
    for decision in parsed.nodes {
        if !batch_ids.contains(&decision.semantic_id) {
            tracing::warn!(
                "reclassify batch: dropping decision for unknown semantic_id {}",
                decision.semantic_id
            );
            continue;
        }
        if !seen.insert(decision.semantic_id) {
            tracing::warn!(
                "reclassify batch: dropping duplicate decision for semantic_id {}",
                decision.semantic_id
            );
            continue;
        }
        match DeFiCategory::parse(&decision.category) {
            Some(DeFiCategory::Others) | None => {
                tracing::warn!(
                    "reclassify batch: dropping decision for semantic_id {} with unusable category '{}'",
                    decision.semantic_id,
                    decision.category
                );
            }
            Some(category) => out.push(ReclassifyNodeDecision {
                semantic_id: decision.semantic_id,
                category: category.as_str().to_string(),
            }),
        }
    }
    let covered: HashSet<i32> = seen.into_iter().collect();
    for node in batch {
        if !covered.contains(&node.id) {
            tracing::warn!(
                "reclassify batch: semantic_id {} left unjudged by the model",
                node.id
            );
        }
    }
    Ok(out)
}
