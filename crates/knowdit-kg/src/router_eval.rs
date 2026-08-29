//! Offline replay evaluation for finding-to-semantic candidate routers.
//!
//! The evaluator treats completed, globally-produced High/Medium links as a
//! silver-label set. It never uses link evidence while retrieving candidates:
//! only finding fields available before linking are allowed into the query.

use crate::category::DeFiCategory;
use crate::db::{HistoricalDatabase, IN_PROJECT_LINK_EVIDENCE_PREFIX};
use crate::error::{KgError, Result};
use knowdit_kg_model::db::{
    audit_finding, audit_finding_category, category, finding_category, finding_link_status,
    finding_merge, project_category, project_finding, semantic_finding_link, semantic_merge,
    semantic_node, semantic_node_category,
};
use knowdit_kg_model::link_strength::LinkStrength;
use sea_orm::EntityTrait;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

pub const ROUTER_EMBEDDING_CACHE_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterKind {
    Bm25,
    Hybrid,
}

#[derive(Debug, Clone)]
pub struct RouterReplayOptions {
    pub router: RouterKind,
    pub max_candidates: usize,
    pub category_boost: f64,
    pub variant_render_cap: usize,
    pub raw_child_char_cap: usize,
    pub mechanism_candidates_per_shard: usize,
    pub embedding_cache: Option<RouterEmbeddingCache>,
    pub min_high_recall: f64,
    pub min_medium_recall: f64,
    pub require_cross_category_high_perfect: bool,
    pub max_misses_in_report: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouterEmbeddingDocumentKind {
    Finding,
    Semantic,
}

#[derive(Debug, Clone, Serialize)]
pub struct RouterEmbeddingDocument {
    pub kind: RouterEmbeddingDocumentKind,
    pub id: i32,
    pub fingerprint: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterEmbeddingRecord {
    pub kind: RouterEmbeddingDocumentKind,
    pub id: i32,
    pub fingerprint: String,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterEmbeddingCache {
    pub schema_version: u32,
    pub model: String,
    pub dimensions: usize,
    pub records: Vec<RouterEmbeddingRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RouterReplayReport {
    pub router: String,
    pub corpus: ReplayCorpusStats,
    pub high: RecallMetric,
    pub medium: RecallMetric,
    pub cross_category_high: RecallMetric,
    pub all_high_recovered_findings: RecallMetric,
    pub candidates: CandidateSetStats,
    pub gate: ReplayGate,
    pub misses: Vec<ReplayMiss>,
    pub omitted_misses: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReplayCorpusStats {
    pub completed_findings: usize,
    pub evaluated_findings: usize,
    pub active_canonical_semantics: usize,
    pub high_edges: usize,
    pub medium_edges: usize,
    pub cross_category_high_edges: usize,
    pub edges_with_unknown_finding_category: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecallMetric {
    pub hits: usize,
    pub total: usize,
    pub recall: Option<f64>,
    pub wilson_95_low: Option<f64>,
    pub wilson_95_high: Option<f64>,
    pub required: f64,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct CandidateSetStats {
    pub configured_max: usize,
    pub mean: f64,
    pub p50: usize,
    pub p95: usize,
    pub max: usize,
    /// Reduction in semantic-document characters presented to the linker.
    /// This does not include the finding or stable rubric text.
    pub estimated_semantic_char_reduction: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReplayGate {
    pub passed: bool,
    pub min_high_recall: f64,
    pub min_medium_recall: f64,
    pub require_cross_category_high_perfect: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReplayMiss {
    pub finding_id: i32,
    pub finding_title: String,
    pub target_semantic_id: i32,
    pub target_semantic_name: String,
    pub strength: LinkStrength,
    pub cross_category: Option<bool>,
    pub finding_project_categories: Vec<DeFiCategory>,
    pub semantic_categories: Vec<DeFiCategory>,
    pub target_rank: usize,
    pub target_score: f64,
}

#[derive(Debug, Clone)]
struct ReplayFinding {
    id: i32,
    title: String,
    project_categories: BTreeSet<DeFiCategory>,
    query_fields: Vec<(String, f64)>,
    embedding_text: String,
    mechanisms: BTreeSet<&'static str>,
}

#[derive(Debug, Clone)]
struct ReplaySemantic {
    id: i32,
    name: String,
    categories: BTreeSet<DeFiCategory>,
    weighted_terms: HashMap<String, f64>,
    weighted_len: f64,
    rendered_chars: usize,
    embedding_text: String,
    mechanisms: BTreeSet<&'static str>,
}

#[derive(Debug, Clone)]
struct ReplayEdge {
    finding_id: i32,
    semantic_id: i32,
    strength: LinkStrength,
    cross_category: Option<bool>,
}

#[derive(Debug, Clone)]
struct RankedCandidate {
    semantic_id: i32,
    score: f64,
}

#[derive(Debug)]
struct Bm25Router {
    documents: Vec<ReplaySemantic>,
    document_frequency: HashMap<String, usize>,
    average_len: f64,
    category_boost: f64,
}

impl Bm25Router {
    fn new(documents: Vec<ReplaySemantic>, category_boost: f64) -> Self {
        let mut document_frequency = HashMap::new();
        let mut total_len = 0.0;
        for document in &documents {
            total_len += document.weighted_len;
            for term in document.weighted_terms.keys() {
                *document_frequency.entry(term.clone()).or_insert(0) += 1;
            }
        }
        let average_len = if documents.is_empty() {
            1.0
        } else {
            (total_len / documents.len() as f64).max(1.0)
        };
        Self {
            documents,
            document_frequency,
            average_len,
            category_boost,
        }
    }

    fn rank(&self, finding: &ReplayFinding) -> Vec<RankedCandidate> {
        let query_terms = weighted_terms(&finding.query_fields);
        let document_count = self.documents.len() as f64;
        let k1 = 1.2;
        let b = 0.75;
        let mut ranked = self
            .documents
            .iter()
            .map(|document| {
                let mut score = 0.0;
                for (term, query_weight) in &query_terms {
                    let Some(term_frequency) = document.weighted_terms.get(term) else {
                        continue;
                    };
                    let df = *self.document_frequency.get(term).unwrap_or(&0) as f64;
                    let idf = (1.0 + (document_count - df + 0.5) / (df + 0.5)).ln();
                    let normalization =
                        k1 * (1.0 - b + b * document.weighted_len / self.average_len);
                    score += query_weight * idf * (term_frequency * (k1 + 1.0))
                        / (term_frequency + normalization);
                }
                if !finding.project_categories.is_disjoint(&document.categories) {
                    score += self.category_boost;
                }
                RankedCandidate {
                    semantic_id: document.id,
                    score,
                }
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|lhs, rhs| {
            rhs.score
                .partial_cmp(&lhs.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| lhs.semantic_id.cmp(&rhs.semantic_id))
        });
        ranked
    }
}

#[derive(Debug)]
struct HybridRouter {
    lexical: Bm25Router,
    finding_vectors: HashMap<i32, Vec<f32>>,
    semantic_vectors: HashMap<i32, Vec<f32>>,
    max_candidates: usize,
    mechanism_candidates_per_shard: usize,
}

impl HybridRouter {
    fn new(
        documents: Vec<ReplaySemantic>,
        findings: &[ReplayFinding],
        category_boost: f64,
        max_candidates: usize,
        mechanism_candidates_per_shard: usize,
        cache: &RouterEmbeddingCache,
    ) -> Result<Self> {
        if cache.schema_version != ROUTER_EMBEDDING_CACHE_VERSION {
            return Err(KgError::other(format!(
                "unsupported router embedding cache version {}; expected {}",
                cache.schema_version, ROUTER_EMBEDDING_CACHE_VERSION
            )));
        }
        if cache.dimensions == 0 {
            return Err(KgError::other("router embedding cache has zero dimensions"));
        }

        let mut records = HashMap::new();
        for record in &cache.records {
            if record.vector.len() != cache.dimensions
                || record.vector.iter().any(|value| !value.is_finite())
            {
                return Err(KgError::other(format!(
                    "invalid embedding vector for {:?} {}",
                    record.kind, record.id
                )));
            }
            if records.insert((record.kind, record.id), record).is_some() {
                return Err(KgError::other(format!(
                    "duplicate embedding record for {:?} {}",
                    record.kind, record.id
                )));
            }
        }

        let mut finding_vectors = HashMap::new();
        for finding in findings {
            let record = require_embedding_record(
                &records,
                RouterEmbeddingDocumentKind::Finding,
                finding.id,
                &finding.embedding_text,
            )?;
            finding_vectors.insert(finding.id, normalized(&record.vector)?);
        }
        let mut semantic_vectors = HashMap::new();
        for semantic in &documents {
            let record = require_embedding_record(
                &records,
                RouterEmbeddingDocumentKind::Semantic,
                semantic.id,
                &semantic.embedding_text,
            )?;
            semantic_vectors.insert(semantic.id, normalized(&record.vector)?);
        }

        Ok(Self {
            lexical: Bm25Router::new(documents, category_boost),
            finding_vectors,
            semantic_vectors,
            max_candidates: max_candidates.max(1),
            mechanism_candidates_per_shard,
        })
    }

    fn rank(&self, finding: &ReplayFinding) -> Vec<RankedCandidate> {
        let lexical = self.lexical.rank(finding);
        let lexical_rank = lexical
            .iter()
            .enumerate()
            .map(|(index, candidate)| (candidate.semantic_id, index + 1))
            .collect::<HashMap<_, _>>();
        let finding_vector = &self.finding_vectors[&finding.id];
        let mut embedding = self
            .lexical
            .documents
            .iter()
            .map(|semantic| RankedCandidate {
                semantic_id: semantic.id,
                score: dot(finding_vector, &self.semantic_vectors[&semantic.id]),
            })
            .collect::<Vec<_>>();
        embedding.sort_by(|lhs, rhs| {
            rhs.score
                .partial_cmp(&lhs.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| lhs.semantic_id.cmp(&rhs.semantic_id))
        });
        // Keep BM25 as the primary ordering: it is substantially stronger on
        // this corpus than an unconstrained dense/RRF reorder.  Dense and
        // mechanism signals are used below only to rescue candidates that
        // would otherwise fall outside the configured lexical cutoff.
        let rescue_budget = if self.mechanism_candidates_per_shard == 0 {
            0
        } else {
            self.mechanism_candidates_per_shard
                .saturating_div(4)
                .max(4)
                .min(self.max_candidates / 64 + 1)
        };
        let lexical_cutoff = self
            .max_candidates
            .saturating_sub(rescue_budget)
            .min(lexical.len());
        let mut selected = lexical
            .iter()
            .take(lexical_cutoff)
            .cloned()
            .collect::<Vec<_>>();
        let mut selected_ids = selected
            .iter()
            .map(|candidate| candidate.semantic_id)
            .collect::<HashSet<_>>();

        let semantics_by_id = self
            .lexical
            .documents
            .iter()
            .map(|semantic| (semantic.id, semantic))
            .collect::<HashMap<_, _>>();
        let mut rescue = Vec::new();
        let mut rescue_seen = HashSet::new();
        let shard_candidates = finding
            .mechanisms
            .iter()
            .map(|mechanism| {
                lexical
                    .iter()
                    .filter(|candidate| {
                        semantics_by_id[&candidate.semantic_id]
                            .mechanisms
                            .contains(mechanism)
                    })
                    .take(self.mechanism_candidates_per_shard)
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for offset in 0..self.mechanism_candidates_per_shard {
            for shard in &shard_candidates {
                if let Some(candidate) = shard.get(offset)
                    && !selected_ids.contains(&candidate.semantic_id)
                    && rescue_seen.insert(candidate.semantic_id)
                {
                    rescue.push(candidate.clone());
                }
            }
        }
        // Add a small dense tail as a category-agnostic safety net.  The
        // mechanism pool remains first because it is deterministic and
        // explainable; dense candidates cover vocabulary/wording drift.
        for (index, candidate) in embedding
            .iter()
            .take(self.max_candidates.saturating_mul(2).max(64))
            .enumerate()
        {
            let dense_rank = index + 1;
            let lexical_rank_for_candidate = lexical_rank[&candidate.semantic_id];
            let materially_better_dense = dense_rank + 32 < lexical_rank_for_candidate;
            if materially_better_dense
                && !selected_ids.contains(&candidate.semantic_id)
                && rescue_seen.insert(candidate.semantic_id)
            {
                rescue.push(candidate.clone());
            }
        }
        rescue.truncate(rescue_budget);
        for candidate in rescue {
            if selected.len() >= self.max_candidates {
                break;
            }
            if selected_ids.insert(candidate.semantic_id) {
                selected.push(candidate);
            }
        }
        // Fill any unused capacity with the original lexical tail.
        selected.extend(
            lexical
                .into_iter()
                .filter(|candidate| selected_ids.insert(candidate.semantic_id)),
        );
        selected
    }
}

fn require_embedding_record<'a>(
    records: &'a HashMap<(RouterEmbeddingDocumentKind, i32), &'a RouterEmbeddingRecord>,
    kind: RouterEmbeddingDocumentKind,
    id: i32,
    text: &str,
) -> Result<&'a RouterEmbeddingRecord> {
    let record = records
        .get(&(kind, id))
        .copied()
        .ok_or_else(|| KgError::other(format!("embedding cache is missing {:?} {}", kind, id)))?;
    let expected = embedding_fingerprint(text);
    if record.fingerprint != expected {
        return Err(KgError::other(format!(
            "stale embedding cache entry for {:?} {}; rebuild the cache",
            kind, id
        )));
    }
    Ok(record)
}

fn normalized(vector: &[f32]) -> Result<Vec<f32>> {
    let norm = vector
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    if norm <= f64::EPSILON {
        return Err(KgError::other("embedding cache contains a zero vector"));
    }
    Ok(vector
        .iter()
        .map(|value| (*value as f64 / norm) as f32)
        .collect())
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .map(|(lhs, rhs)| f64::from(*lhs) * f64::from(*rhs))
        .sum()
}

impl HistoricalDatabase {
    pub async fn evaluate_link_router(
        &self,
        options: RouterReplayOptions,
    ) -> Result<RouterReplayReport> {
        let (findings, semantics, edges, completed_findings, unknown_category_edges) =
            self.load_router_replay_corpus(&options).await?;
        let semantics_by_id = semantics
            .iter()
            .map(|semantic| (semantic.id, semantic.clone()))
            .collect::<HashMap<_, _>>();
        let edges_by_finding = edges.iter().fold(
            BTreeMap::<i32, Vec<&ReplayEdge>>::new(),
            |mut grouped, edge| {
                grouped.entry(edge.finding_id).or_default().push(edge);
                grouped
            },
        );
        let finding_by_id = findings
            .iter()
            .map(|finding| (finding.id, finding))
            .collect::<HashMap<_, _>>();
        let lexical_router = Bm25Router::new(semantics.clone(), options.category_boost);
        let hybrid_router = match options.router {
            RouterKind::Bm25 => None,
            RouterKind::Hybrid => Some(HybridRouter::new(
                semantics.clone(),
                &findings,
                options.category_boost,
                options.max_candidates,
                options.mechanism_candidates_per_shard,
                options
                    .embedding_cache
                    .as_ref()
                    .ok_or_else(|| KgError::other("hybrid router requires an embedding cache"))?,
            )?),
        };
        let full_semantic_chars = semantics
            .iter()
            .map(|semantic| semantic.rendered_chars)
            .sum::<usize>();

        let mut high = (0usize, 0usize);
        let mut medium = (0usize, 0usize);
        let mut cross_high = (0usize, 0usize);
        let mut all_high_findings = (0usize, 0usize);
        let mut candidate_counts = Vec::new();
        let mut candidate_char_ratios = Vec::new();
        let mut misses = Vec::new();

        for (finding_id, finding_edges) in &edges_by_finding {
            let Some(finding) = finding_by_id.get(finding_id) else {
                continue;
            };
            let ranked = hybrid_router.as_ref().map_or_else(
                || lexical_router.rank(finding),
                |router| router.rank(finding),
            );
            let candidate_count = options.max_candidates.max(1).min(ranked.len());
            let selected = ranked
                .iter()
                .take(candidate_count)
                .map(|candidate| candidate.semantic_id)
                .collect::<HashSet<_>>();
            let rank_by_id = ranked
                .iter()
                .enumerate()
                .map(|(index, candidate)| (candidate.semantic_id, (index + 1, candidate.score)))
                .collect::<HashMap<_, _>>();
            candidate_counts.push(candidate_count);
            let selected_chars = selected
                .iter()
                .filter_map(|id| semantics_by_id.get(id))
                .map(|semantic| semantic.rendered_chars)
                .sum::<usize>();
            if full_semantic_chars > 0 {
                candidate_char_ratios.push(selected_chars as f64 / full_semantic_chars as f64);
            }

            let high_edges = finding_edges
                .iter()
                .filter(|edge| edge.strength == LinkStrength::High)
                .collect::<Vec<_>>();
            if !high_edges.is_empty() {
                all_high_findings.1 += 1;
                if high_edges
                    .iter()
                    .all(|edge| selected.contains(&edge.semantic_id))
                {
                    all_high_findings.0 += 1;
                }
            }

            for edge in finding_edges {
                let hit = selected.contains(&edge.semantic_id);
                match edge.strength {
                    LinkStrength::High => {
                        high.1 += 1;
                        high.0 += usize::from(hit);
                        if edge.cross_category == Some(true) {
                            cross_high.1 += 1;
                            cross_high.0 += usize::from(hit);
                        }
                    }
                    LinkStrength::Medium => {
                        medium.1 += 1;
                        medium.0 += usize::from(hit);
                    }
                    LinkStrength::Low => {}
                }
                if !hit {
                    let Some(semantic) = semantics_by_id.get(&edge.semantic_id) else {
                        continue;
                    };
                    let (target_rank, target_score) = rank_by_id
                        .get(&edge.semantic_id)
                        .copied()
                        .unwrap_or((ranked.len() + 1, 0.0));
                    misses.push(ReplayMiss {
                        finding_id: finding.id,
                        finding_title: finding.title.clone(),
                        target_semantic_id: semantic.id,
                        target_semantic_name: semantic.name.clone(),
                        strength: edge.strength,
                        cross_category: edge.cross_category,
                        finding_project_categories: finding
                            .project_categories
                            .iter()
                            .copied()
                            .collect(),
                        semantic_categories: semantic.categories.iter().copied().collect(),
                        target_rank,
                        target_score,
                    });
                }
            }
        }

        misses.sort_by(|lhs, rhs| {
            rhs.strength
                .rank()
                .cmp(&lhs.strength.rank())
                .then_with(|| rhs.cross_category.cmp(&lhs.cross_category))
                .then_with(|| lhs.target_rank.cmp(&rhs.target_rank))
                .then_with(|| lhs.finding_id.cmp(&rhs.finding_id))
        });
        let omitted_misses = misses.len().saturating_sub(options.max_misses_in_report);
        misses.truncate(options.max_misses_in_report);

        let high_metric = recall_metric(high.0, high.1, options.min_high_recall);
        let medium_metric = recall_metric(medium.0, medium.1, options.min_medium_recall);
        let cross_requirement = if options.require_cross_category_high_perfect {
            1.0
        } else {
            options.min_high_recall
        };
        let cross_metric = recall_metric(cross_high.0, cross_high.1, cross_requirement);
        let all_high_metric = recall_metric(all_high_findings.0, all_high_findings.1, 0.0);
        let gate_passed = high_metric.passed && medium_metric.passed && cross_metric.passed;

        Ok(RouterReplayReport {
            router: format!(
                "{}(max_candidates={},category_boost={},variants={},child_chars={},mechanism_per_shard={})",
                match options.router {
                    RouterKind::Bm25 => "bm25".to_string(),
                    RouterKind::Hybrid => format!(
                        "hybrid-lexical-rescue[{}]",
                        options
                            .embedding_cache
                            .as_ref()
                            .map(|cache| cache.model.as_str())
                            .unwrap_or("missing")
                    ),
                },
                options.max_candidates.max(1),
                options.category_boost,
                options.variant_render_cap,
                options.raw_child_char_cap,
                options.mechanism_candidates_per_shard,
            ),
            corpus: ReplayCorpusStats {
                completed_findings,
                evaluated_findings: edges_by_finding.len(),
                active_canonical_semantics: semantics.len(),
                high_edges: high.1,
                medium_edges: medium.1,
                cross_category_high_edges: cross_high.1,
                edges_with_unknown_finding_category: unknown_category_edges,
            },
            high: high_metric,
            medium: medium_metric,
            cross_category_high: cross_metric,
            all_high_recovered_findings: all_high_metric,
            candidates: candidate_stats(
                options.max_candidates.max(1),
                candidate_counts,
                candidate_char_ratios,
            ),
            gate: ReplayGate {
                passed: gate_passed,
                min_high_recall: options.min_high_recall,
                min_medium_recall: options.min_medium_recall,
                require_cross_category_high_perfect: options.require_cross_category_high_perfect,
            },
            misses,
            omitted_misses,
        })
    }

    async fn load_router_replay_corpus(
        &self,
        options: &RouterReplayOptions,
    ) -> Result<(
        Vec<ReplayFinding>,
        Vec<ReplaySemantic>,
        Vec<ReplayEdge>,
        usize,
        usize,
    )> {
        let category_names = category::Entity::find()
            .all(self.conn())
            .await?
            .into_iter()
            .map(|row| (row.id, row.name))
            .collect::<HashMap<_, _>>();
        let mut project_categories = HashMap::<i32, BTreeSet<DeFiCategory>>::new();
        for row in project_category::Entity::find().all(self.conn()).await? {
            if let Some(category) = category_names.get(&row.category_id) {
                project_categories
                    .entry(row.project_id)
                    .or_default()
                    .insert(*category);
            }
        }

        let mut finding_projects = HashMap::<i32, BTreeSet<i32>>::new();
        for row in project_finding::Entity::find().all(self.conn()).await? {
            finding_projects
                .entry(row.audit_finding_id)
                .or_default()
                .insert(row.project_id);
        }
        let mut categories_by_finding = HashMap::<i32, BTreeSet<DeFiCategory>>::new();
        for (finding_id, project_ids) in &finding_projects {
            let categories = categories_by_finding.entry(*finding_id).or_default();
            for project_id in project_ids {
                if let Some(project_categories) = project_categories.get(project_id) {
                    categories.extend(project_categories.iter().copied());
                }
            }
        }

        let finding_categories = finding_category::Entity::find()
            .all(self.conn())
            .await?
            .into_iter()
            .map(|row| (row.id, row))
            .collect::<HashMap<_, _>>();
        let mut taxonomy_by_finding = HashMap::new();
        for row in audit_finding_category::Entity::find()
            .all(self.conn())
            .await?
        {
            if let Some(taxonomy) = finding_categories.get(&row.finding_category_id) {
                taxonomy_by_finding.insert(row.audit_finding_id, taxonomy.clone());
            }
        }

        let completed_ids = finding_link_status::Entity::find()
            .all(self.conn())
            .await?
            .into_iter()
            .map(|row| row.audit_finding_id)
            .collect::<HashSet<_>>();
        let all_findings = audit_finding::Entity::find().all(self.conn()).await?;
        let finding_by_id = all_findings
            .iter()
            .map(|finding| (finding.id, finding))
            .collect::<HashMap<_, _>>();
        let mut finding_variants = HashMap::<i32, Vec<String>>::new();
        for merge in finding_merge::Entity::find().all(self.conn()).await? {
            let Some(child) = finding_by_id.get(&merge.from_finding_id) else {
                continue;
            };
            finding_variants
                .entry(merge.to_finding_id)
                .or_default()
                .push(format!(
                    "{} {} {} {} {}",
                    child.title,
                    bounded(&child.root_cause, options.raw_child_char_cap),
                    bounded(&child.description, options.raw_child_char_cap),
                    bounded(&child.patterns, options.raw_child_char_cap),
                    bounded(&child.exploits, options.raw_child_char_cap),
                ));
        }
        drop(finding_by_id);
        let findings = all_findings
            .into_iter()
            .filter(|finding| completed_ids.contains(&finding.id))
            .map(|finding| {
                let taxonomy = taxonomy_by_finding.get(&finding.id);
                let taxonomy_text = taxonomy
                    .map(|row| format!("{} {}", row.category, row.name))
                    .unwrap_or_default();
                let variants = finding_variants
                    .remove(&finding.id)
                    .unwrap_or_default()
                    .into_iter()
                    .take(options.variant_render_cap)
                    .collect::<Vec<_>>()
                    .join(" ");
                let embedding_text = format!(
                    "{}\n{}\n{}\n{}\n{}\n{}\n{}",
                    finding.title,
                    finding.root_cause,
                    finding.description,
                    finding.patterns,
                    finding.exploits,
                    taxonomy_text,
                    variants,
                );
                let mechanisms = mechanism_shards(&embedding_text);
                let query_fields = vec![
                    (finding.title.clone(), 3.0),
                    (finding.root_cause, 3.0),
                    (finding.description, 1.0),
                    (finding.patterns, 1.0),
                    (finding.exploits, 0.5),
                    (taxonomy_text, 2.0),
                    (variants, 1.0),
                ];
                ReplayFinding {
                    id: finding.id,
                    title: finding.title.clone(),
                    project_categories: categories_by_finding
                        .remove(&finding.id)
                        .unwrap_or_default(),
                    query_fields,
                    embedding_text,
                    mechanisms,
                }
            })
            .collect::<Vec<_>>();

        let all_semantics = semantic_node::Entity::find().all(self.conn()).await?;
        let semantic_by_id = all_semantics
            .iter()
            .map(|semantic| (semantic.id, semantic))
            .collect::<HashMap<_, _>>();
        let mut folded_semantics = HashSet::new();
        let mut semantic_variants = HashMap::<i32, Vec<String>>::new();
        for merge in semantic_merge::Entity::find().all(self.conn()).await? {
            folded_semantics.insert(merge.from_semantic_id);
            let Some(child) = semantic_by_id.get(&merge.from_semantic_id) else {
                continue;
            };
            semantic_variants
                .entry(merge.to_semantic_id)
                .or_default()
                .push(format!(
                    "{} {}",
                    child.name,
                    bounded(&child.description, options.raw_child_char_cap)
                ));
        }
        drop(semantic_by_id);
        let mut secondary_categories = HashMap::<i32, BTreeSet<DeFiCategory>>::new();
        for row in semantic_node_category::Entity::find()
            .all(self.conn())
            .await?
        {
            if let Some(category) = category_names.get(&row.category_id) {
                secondary_categories
                    .entry(row.semantic_node_id)
                    .or_default()
                    .insert(*category);
            }
        }
        let semantics = all_semantics
            .into_iter()
            .filter(|semantic| !folded_semantics.contains(&semantic.id))
            .map(|semantic| {
                let mut categories = secondary_categories
                    .remove(&semantic.id)
                    .unwrap_or_default();
                categories.insert(semantic.category);
                let variants = semantic_variants
                    .remove(&semantic.id)
                    .unwrap_or_default()
                    .into_iter()
                    .take(options.variant_render_cap)
                    .collect::<Vec<_>>();
                let variant_chars = variants.iter().map(String::len).sum::<usize>();
                let fields = vec![
                    (semantic.name.clone(), 4.0),
                    (semantic.definition.clone(), 2.0),
                    (semantic.description.clone(), 1.0),
                    (variants.join(" "), 1.0),
                ];
                let embedding_text = fields
                    .iter()
                    .map(|(text, _)| text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                let mechanisms = mechanism_shards(&embedding_text);
                let terms = weighted_terms(&fields);
                let rendered_chars = semantic.name.len()
                    + semantic.definition.len()
                    + semantic.description.len()
                    + variant_chars;
                ReplaySemantic {
                    id: semantic.id,
                    name: semantic.name,
                    categories,
                    weighted_len: terms.values().sum::<f64>().max(1.0),
                    weighted_terms: terms,
                    rendered_chars,
                    embedding_text,
                    mechanisms,
                }
            })
            .collect::<Vec<_>>();
        let semantic_categories = semantics
            .iter()
            .map(|semantic| (semantic.id, semantic.categories.clone()))
            .collect::<HashMap<_, _>>();
        let finding_category_sets = findings
            .iter()
            .map(|finding| (finding.id, finding.project_categories.clone()))
            .collect::<HashMap<_, _>>();

        let mut unknown_category_edges = 0usize;
        let edges = semantic_finding_link::Entity::find()
            .all(self.conn())
            .await?
            .into_iter()
            .filter(|edge| completed_ids.contains(&edge.audit_finding_id))
            .filter(|edge| !edge.evidence.starts_with(IN_PROJECT_LINK_EVIDENCE_PREFIX))
            .filter(|edge| edge.strength != LinkStrength::Low)
            .filter_map(|edge| {
                let semantic_categories = semantic_categories.get(&edge.semantic_node_id)?;
                let finding_categories = finding_category_sets.get(&edge.audit_finding_id)?;
                let cross_category = if finding_categories.is_empty() {
                    unknown_category_edges += 1;
                    None
                } else {
                    Some(finding_categories.is_disjoint(semantic_categories))
                };
                Some(ReplayEdge {
                    finding_id: edge.audit_finding_id,
                    semantic_id: edge.semantic_node_id,
                    strength: edge.strength,
                    cross_category,
                })
            })
            .collect::<Vec<_>>();

        Ok((
            findings,
            semantics,
            edges,
            completed_ids.len(),
            unknown_category_edges,
        ))
    }

    pub async fn link_router_embedding_documents(
        &self,
        variant_render_cap: usize,
        raw_child_char_cap: usize,
    ) -> Result<Vec<RouterEmbeddingDocument>> {
        let options = RouterReplayOptions {
            router: RouterKind::Bm25,
            max_candidates: 1,
            category_boost: 0.0,
            variant_render_cap,
            raw_child_char_cap,
            mechanism_candidates_per_shard: 0,
            embedding_cache: None,
            min_high_recall: 0.0,
            min_medium_recall: 0.0,
            require_cross_category_high_perfect: false,
            max_misses_in_report: 0,
        };
        let (findings, semantics, _, _, _) = self.load_router_replay_corpus(&options).await?;
        let mut documents = findings
            .into_iter()
            .map(|finding| RouterEmbeddingDocument {
                kind: RouterEmbeddingDocumentKind::Finding,
                id: finding.id,
                fingerprint: embedding_fingerprint(&finding.embedding_text),
                text: finding.embedding_text,
            })
            .chain(
                semantics
                    .into_iter()
                    .map(|semantic| RouterEmbeddingDocument {
                        kind: RouterEmbeddingDocumentKind::Semantic,
                        id: semantic.id,
                        fingerprint: embedding_fingerprint(&semantic.embedding_text),
                        text: semantic.embedding_text,
                    }),
            )
            .collect::<Vec<_>>();
        documents.sort_by_key(|document| (document.kind, document.id));
        Ok(documents)
    }
}

pub fn embedding_fingerprint(text: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(text.as_bytes());
    format!("{:x}", digest.finalize())
}

fn mechanism_shards(text: &str) -> BTreeSet<&'static str> {
    const SHARDS: &[(&str, &[&str])] = &[
        (
            "oracle_price",
            &[
                "oracle",
                "twap",
                "spot price",
                "price feed",
                "reserve manipulation",
            ],
        ),
        (
            "access_control",
            &[
                "access control",
                "unauthor",
                "privilege",
                "admin",
                "owner",
                "role",
                "permission",
            ],
        ),
        (
            "reentrancy_callback",
            &["reentran", "callback", "external call", "hook", "fallback"],
        ),
        (
            "accounting_precision",
            &[
                "accounting",
                "rounding",
                "precision",
                "share price",
                "exchange rate",
                "balance drift",
            ],
        ),
        (
            "initialization_upgrade",
            &[
                "initializ",
                "upgrade",
                "proxy",
                "storage layout",
                "deployment",
                "implementation",
            ],
        ),
        (
            "signature_replay",
            &[
                "signature",
                "nonce",
                "replay",
                "permit",
                "eip-712",
                "erc2771",
                "meta-transaction",
            ],
        ),
        (
            "liquidation_collateral",
            &[
                "liquidat",
                "collateral",
                "health factor",
                "solvency",
                "bad debt",
            ],
        ),
        (
            "token_authority",
            &[
                "allowance",
                "approval",
                "transferfrom",
                "mint",
                "burn",
                "token transfer",
            ],
        ),
        (
            "cross_chain",
            &[
                "cross-chain",
                "cross chain",
                "bridge",
                "message proof",
                "chain id",
                "relayer",
            ],
        ),
        (
            "governance",
            &["governance", "proposal", "voting", "quorum", "delegate"],
        ),
        (
            "amm_liquidity",
            &[
                "amm",
                "liquidity",
                "swap",
                "pool reserve",
                "constant product",
                "slippage",
            ],
        ),
        (
            "availability_gas",
            &[
                "denial of service",
                " dos ",
                "out of gas",
                "unbounded loop",
                "gas grief",
                "revert",
            ],
        ),
        (
            "flash_loan_atomic",
            &["flash loan", "flashloan", "single transaction", "atomic"],
        ),
        (
            "time_ordering",
            &[
                "timestamp",
                "deadline",
                "front-run",
                "frontrun",
                "sandwich",
                "ordering",
            ],
        ),
    ];
    let normalized = format!(" {} ", text.to_ascii_lowercase().replace('_', " "));
    SHARDS
        .iter()
        .filter_map(|(name, needles)| {
            needles
                .iter()
                .any(|needle| normalized.contains(needle))
                .then_some(*name)
        })
        .collect()
}

fn weighted_terms(fields: &[(String, f64)]) -> HashMap<String, f64> {
    let mut terms = HashMap::new();
    for (text, weight) in fields {
        for token in tokenize(text) {
            *terms.entry(token).or_insert(0.0) += *weight;
        }
    }
    terms
}

fn bounded(text: &str, char_cap: usize) -> String {
    if char_cap == 0 {
        text.to_string()
    } else {
        text.chars().take(char_cap).collect()
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter_map(|raw| {
            let token = raw.to_ascii_lowercase();
            if token.len() < 3 || STOP_WORDS.contains(&token.as_str()) {
                None
            } else {
                Some(normalize_suffix(token))
            }
        })
        .collect()
}

fn normalize_suffix(mut token: String) -> String {
    for suffix in [
        "ization", "ation", "ments", "ment", "ingly", "ing", "ed", "es", "s",
    ] {
        if token.len() > suffix.len() + 3 && token.ends_with(suffix) {
            token.truncate(token.len() - suffix.len());
            break;
        }
    }
    token
}

const STOP_WORDS: &[&str] = &[
    "and", "are", "but", "can", "does", "for", "from", "has", "have", "into", "its", "not", "that",
    "the", "their", "then", "this", "through", "use", "uses", "using", "was", "when", "where",
    "which", "while", "with", "without",
];

fn recall_metric(hits: usize, total: usize, required: f64) -> RecallMetric {
    let recall = (total > 0).then_some(hits as f64 / total as f64);
    let (wilson_95_low, wilson_95_high) = wilson_interval(hits, total);
    RecallMetric {
        hits,
        total,
        recall,
        wilson_95_low,
        wilson_95_high,
        required,
        passed: recall.is_some_and(|value| value + f64::EPSILON >= required),
    }
}

fn wilson_interval(hits: usize, total: usize) -> (Option<f64>, Option<f64>) {
    if total == 0 {
        return (None, None);
    }
    let n = total as f64;
    let p = hits as f64 / n;
    let z = 1.959_963_984_540_054;
    let denominator = 1.0 + z * z / n;
    let center = (p + z * z / (2.0 * n)) / denominator;
    let margin = z * ((p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt()) / denominator;
    (
        Some((center - margin).max(0.0)),
        Some((center + margin).min(1.0)),
    )
}

fn candidate_stats(
    configured_max: usize,
    mut counts: Vec<usize>,
    selected_char_ratios: Vec<f64>,
) -> CandidateSetStats {
    counts.sort_unstable();
    let percentile = |fraction: f64| -> usize {
        if counts.is_empty() {
            return 0;
        }
        let index = ((counts.len() - 1) as f64 * fraction).ceil() as usize;
        counts[index]
    };
    let mean = if counts.is_empty() {
        0.0
    } else {
        counts.iter().sum::<usize>() as f64 / counts.len() as f64
    };
    let mean_char_ratio = if selected_char_ratios.is_empty() {
        1.0
    } else {
        selected_char_ratios.iter().sum::<f64>() / selected_char_ratios.len() as f64
    };
    CandidateSetStats {
        configured_max,
        mean,
        p50: percentile(0.50),
        p95: percentile(0.95),
        max: counts.last().copied().unwrap_or(0),
        estimated_semantic_char_reduction: (1.0 - mean_char_ratio).clamp(0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn semantic(id: i32, text: &str, category: DeFiCategory) -> ReplaySemantic {
        let fields = vec![(text.to_string(), 1.0)];
        let terms = weighted_terms(&fields);
        ReplaySemantic {
            id,
            name: text.to_string(),
            categories: BTreeSet::from([category]),
            weighted_len: terms.values().sum(),
            weighted_terms: terms,
            rendered_chars: text.len(),
            embedding_text: text.to_string(),
            mechanisms: mechanism_shards(text),
        }
    }

    #[test]
    fn bm25_keeps_cross_category_mechanism_match() {
        let router = Bm25Router::new(
            vec![
                semantic(1, "oracle stale price manipulation", DeFiCategory::Dexes),
                semantic(2, "loan repayment accounting", DeFiCategory::Lending),
            ],
            0.25,
        );
        let finding = ReplayFinding {
            id: 1,
            title: "stale oracle".to_string(),
            project_categories: BTreeSet::from([DeFiCategory::Lending]),
            query_fields: vec![("stale oracle price".to_string(), 1.0)],
            embedding_text: "stale oracle price".to_string(),
            mechanisms: mechanism_shards("stale oracle price"),
        };
        assert_eq!(router.rank(&finding)[0].semantic_id, 1);
    }

    #[test]
    fn empty_metric_does_not_pass_a_gate() {
        assert!(!recall_metric(0, 0, 1.0).passed);
    }

    #[test]
    fn recall_gate_uses_point_estimate() {
        assert!(recall_metric(199, 200, 0.995).passed);
        assert!(!recall_metric(198, 200, 0.995).passed);
    }

    #[test]
    fn hybrid_reserves_cross_category_mechanism_candidates() {
        let documents = vec![
            semantic(1, "stale oracle price consumption", DeFiCategory::Dexes),
            semantic(2, "loan repayment accounting", DeFiCategory::Lending),
        ];
        let finding = ReplayFinding {
            id: 7,
            title: "stale oracle".to_string(),
            project_categories: BTreeSet::from([DeFiCategory::Lending]),
            query_fields: vec![("loan repayment stale oracle".to_string(), 1.0)],
            embedding_text: "stale oracle price".to_string(),
            mechanisms: mechanism_shards("stale oracle price"),
        };
        let cache = RouterEmbeddingCache {
            schema_version: ROUTER_EMBEDDING_CACHE_VERSION,
            model: "test".to_string(),
            dimensions: 2,
            records: vec![
                embedding_record(
                    RouterEmbeddingDocumentKind::Finding,
                    finding.id,
                    &finding.embedding_text,
                    vec![1.0, 0.0],
                ),
                embedding_record(
                    RouterEmbeddingDocumentKind::Semantic,
                    documents[0].id,
                    &documents[0].embedding_text,
                    vec![1.0, 0.0],
                ),
                embedding_record(
                    RouterEmbeddingDocumentKind::Semantic,
                    documents[1].id,
                    &documents[1].embedding_text,
                    vec![0.0, 1.0],
                ),
            ],
        };
        let router = HybridRouter::new(documents, &[finding.clone()], 10.0, 1, 1, &cache)
            .expect("hybrid router");
        assert_eq!(router.rank(&finding)[0].semantic_id, 1);
    }

    #[test]
    fn hybrid_rejects_stale_embedding_cache() {
        let documents = vec![semantic(1, "oracle price", DeFiCategory::Dexes)];
        let finding = ReplayFinding {
            id: 7,
            title: "oracle".to_string(),
            project_categories: BTreeSet::new(),
            query_fields: vec![("oracle".to_string(), 1.0)],
            embedding_text: "oracle".to_string(),
            mechanisms: mechanism_shards("oracle"),
        };
        let cache = RouterEmbeddingCache {
            schema_version: ROUTER_EMBEDDING_CACHE_VERSION,
            model: "test".to_string(),
            dimensions: 1,
            records: vec![
                RouterEmbeddingRecord {
                    kind: RouterEmbeddingDocumentKind::Finding,
                    id: finding.id,
                    fingerprint: "stale".to_string(),
                    vector: vec![1.0],
                },
                embedding_record(
                    RouterEmbeddingDocumentKind::Semantic,
                    documents[0].id,
                    &documents[0].embedding_text,
                    vec![1.0],
                ),
            ],
        };
        assert!(HybridRouter::new(documents, &[finding], 0.0, 1, 1, &cache).is_err());
    }

    fn embedding_record(
        kind: RouterEmbeddingDocumentKind,
        id: i32,
        text: &str,
        vector: Vec<f32>,
    ) -> RouterEmbeddingRecord {
        RouterEmbeddingRecord {
            kind,
            id,
            fingerprint: embedding_fingerprint(text),
            vector,
        }
    }
}
