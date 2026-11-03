//! Knowledge Mapper agent (v2 — profile-aware, strength-scoring).
//!
//! Step 2 of the agentic auditing workflow. Given a project that
//! already has its extracted DeFi semantics + a generated
//! [`ProjectProfile`] stored in [`RepoDatabase`], this agent
//! fuzzy-matches each extracted semantic against the historical
//! knowledge graph and attaches a per-pair **strength** label
//! (High / Medium / Low) plus a free-form **evidence** rationale.
//! Downstream filters / sorts / budgets by strength to keep the
//! pipeline anchored to project-relevant historicals only.
//!
//! Pipeline:
//! 1. Read extracted semantics + the [`ProjectProfile`] from the repo
//!    database. The profile MUST exist (run `agentic profile` first);
//!    if absent we abort instead of guessing.
//! 2. Collect categories + `Others`, pull canonical historical
//!    semantics in those categories from the KG.
//! 3. Per batch of historicals, ask the LLM to emit one entry per
//!    matched `(extract, historical)` pair with strength + evidence,
//!    grounded in the project profile we inlined into the system
//!    prompt.
//! 4. Aggregate, run pass 2 (widened categories) on any extracts
//!    still unmatched, persist semantics + findings + match edges
//!    via [`RepoDatabase::write_semantic_match_results`].

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use color_eyre::eyre::{Result, WrapErr};
use itertools::Itertools;
use knowdit_kg::agents::CanonicalWithChildren;
use knowdit_kg::category::DeFiCategory;
use knowdit_kg::db::HistoricalDatabase;
use knowdit_kg::learn::ExtractedSemantic;
use knowdit_kg_model::db::semantic_node;
use knowdit_repo_model::{
    HistoricalFindingChild, HistoricalLinkedFinding, HistoricalSemanticChild,
    HistoricalSemanticRecord, MatchStrength, ProjectProfile, RepoDatabase, SemanticMatch,
    SemanticMatchSet,
};
use llmy::client::client::LLM;
use serde::Deserialize;
use tokio::task::JoinSet;

const DEFAULT_BATCH_SIZE: usize = 16;
/// Default cap on raw children rendered per canonical in the matcher
/// prompt. Overridable per-run via [`MapperOptions::max_rendered_raw_children`]
/// (CLI `--map-max-raw-children`).
const DEFAULT_MAX_RENDERED_RAW_CHILDREN: usize = 5;

/// Minimum length of the LLM-emitted `evidence` string per strength
/// tier. Plan §6.4 forced-per-candidate-emit + §6.1 evidence rules:
/// High/Medium must be substantive (40 chars); Low can be short
/// (15 chars) because the agent is required to emit per-candidate
/// and most candidates will be Low — over-tight evidence floor on Low
/// just inflates output token cost without adding signal.
const MIN_EVIDENCE_LEN_HIGH_MEDIUM: usize = 40;
const MIN_EVIDENCE_LEN_LOW: usize = 15;

/// Extra attempts around one mapper LLM call when the failure is a
/// transport problem (connection refused, DNS, reset) rather than a
/// response the server actually produced.
const NETWORK_RETRY_ATTEMPTS: u32 = 6;
/// First backoff delay; doubles each attempt, capped by the `min(4)`
/// shift — roughly 2s, 4s, 8s, 16s, 32s.
const NETWORK_RETRY_BASE_DELAY: std::time::Duration = std::time::Duration::from_secs(2);
/// Backoff for gateway timeouts (524), which already consumed ~100s of
/// proxy time before failing — 10s, 20s, 40s, 80s, 160s.
const GATEWAY_TIMEOUT_BASE_DELAY: std::time::Duration = std::time::Duration::from_secs(10);
/// Responses longer than this that yield no entries are dumped to disk —
/// a 14-char `{"matches":[]}` needs no investigation, a 180K-char one does.
const UNUSABLE_DUMP_MAX_CHARS: usize = 10_000;

/// Substrings that mark a failure as worth replaying: the request never
/// produced a verdict (refused/reset/DNS) or an edge proxy gave up on an
/// over-long upstream call (524 and friends).
const TRANSIENT_ERROR_NEEDLES: [&str; 14] = [
    "connection refused",
    "tcp connect error",
    "error sending request",
    "connection reset",
    "timed out",
    "dns error",
    "broken pipe",
    "error code: 524",
    "error code 524",
    "gateway time-out",
    "502 bad gateway",
    "503 service",
    "504 gateway",
    "upstream request timeout",
];

/// True when `err` looks like the request never reached the server, so
/// replaying it after a delay is safe and likely to succeed.
fn is_transient_transport_error(err: &llmy::agent::LLMYError) -> bool {
    use llmy::agent::LLMYError;
    let text = err.to_string().to_lowercase();
    match err {
        LLMYError::IO(_) => true,
        // A billing cap, filtered response or bad tool call is
        // deterministic — retrying only wastes budget.
        LLMYError::Billing { .. }
        | LLMYError::Filtered(_)
        | LLMYError::IncorrectToolCall(..)
        | LLMYError::OutputLength => false,
        _ => {
            // Gateways front these endpoints with a ~100s proxy timeout and
            // return the failure as a bare `error code 524` (Cloudflare).
            // That is the edge giving up, not a verdict on the request: the
            // upstream completion may well be fine on a retry, and it is the
            // dominant failure for prompts that ask for 1,200+ JSON entries.
            TRANSIENT_ERROR_NEEDLES.iter().any(|n| text.contains(n))
        }
    }
}

/// Tunables for one Knowledge Mapper pass.
#[derive(Debug, Clone)]
pub struct MapperOptions {
    /// Historical semantics presented per matching prompt.
    pub batch_size: usize,
    /// Max historical batches matched concurrently. Each batch is one
    /// LLM call, so this fans the matching prompts out over this many
    /// workers (async-channel + `JoinSet`, the same idiom as the fuzz
    /// runtime). `1` keeps the old strictly-sequential behaviour.
    pub concurrency: usize,
    /// Optional `llmy` cache key prefix.
    pub cache_key: Option<String>,
    /// Pre-rendered Markdown block describing the project's
    /// source language; verbatim-prepended to the mapper's
    /// system prompt. See [`crate::profile::ProfileOptions::language_prompt_prefix`]
    /// for the same dispatch convention.
    pub language_prompt_prefix: String,
    /// Extra historical categories to consider on top of the project's own
    /// extracted-semantic categories (which always include `Others`). Lets a
    /// caller widen the candidate pool when the project auto-classified too
    /// narrowly — e.g. a P2P-escrow project tagged only `Others` can add
    /// `Services` to surface signature/authorization historical knowledge.
    pub extra_categories: Vec<DeFiCategory>,
    /// Cap on merged raw children rendered per canonical in the matcher
    /// prompt. `0` suppresses the `merged_raw_examples` block entirely.
    /// Larger values give the LLM more concrete examples per cluster at
    /// the cost of prompt tokens. See [`DEFAULT_MAX_RENDERED_RAW_CHILDREN`].
    pub max_rendered_raw_children: usize,
}

impl Default for MapperOptions {
    fn default() -> Self {
        Self {
            batch_size: DEFAULT_BATCH_SIZE,
            concurrency: 1,
            cache_key: Some("knowdit-mapper".to_string()),
            language_prompt_prefix: String::new(),
            extra_categories: Vec::new(),
            max_rendered_raw_children: DEFAULT_MAX_RENDERED_RAW_CHILDREN,
        }
    }
}

/// Knowledge Mapper agent.
///
/// Owns the project profile + language prompt prefix; `repo` /
/// `kg` / `llm` are method arguments to [`Self::run`] so this
/// struct has no lifetime params (plan §1.6).
#[derive(Debug, Clone)]
pub struct KnowledgeMapper {
    profile: ProjectProfile,
    language_prompt_prefix: String,
}

impl KnowledgeMapper {
    pub fn new(profile: ProjectProfile, language_prompt_prefix: String) -> Self {
        Self {
            profile,
            language_prompt_prefix,
        }
    }

    /// Run one mapping pass and write the result into `repo`.
    pub async fn run(
        &self,
        repo: &RepoDatabase,
        kg: &HistoricalDatabase,
        llm: &LLM,
        options: &MapperOptions,
    ) -> Result<MapperOutcome> {
        let extracted = repo
            .load_project_semantics()
            .await
            .wrap_err("failed to load extracted project semantics")?;

        if extracted.is_empty() {
            tracing::warn!("Knowledge Mapper invoked with no extracted semantics; nothing to do");
            return Ok(MapperOutcome::default());
        }

        // Stable indices — must match `project_semantic.id` insertion order.
        let extracted_with_id: Vec<(i32, &ExtractedSemantic)> = extracted
            .iter()
            .enumerate()
            .map(|(i, sem)| ((i as i32) + 1, sem))
            .collect();

        let categories = collect_categories(&extracted, &options.extra_categories);
        if !options.extra_categories.is_empty() {
            tracing::info!(
                "Knowledge Mapper: --mapper-extra-categories added {}",
                options
                    .extra_categories
                    .iter()
                    .map(|c| c.as_str())
                    .join(", "),
            );
        }
        tracing::info!(
            "Knowledge Mapper considering {} historical categories: {}",
            categories.len(),
            categories.iter().map(|c| c.as_str()).join(", ")
        );

        let historical_semantics = kg
            .canonical_semantics_with_children_for_categories(&categories)
            .await
            .wrap_err("failed to load historical semantics from knowledge graph")?;

        if historical_semantics.is_empty() {
            tracing::warn!(
                "Knowledge Mapper found no historical semantics for the project's categories"
            );
            repo.write_semantic_match_results(&SemanticMatchSet::default())
                .await?;
            return Ok(MapperOutcome::default());
        }

        // Secondary categories per canonical (category-recall): rendered in
        // the historical block so the LLM sees the full coverage a canonical
        // absorbed through cross-category folds.
        let secondary_categories: HashMap<i32, Vec<DeFiCategory>> = {
            let canonical_ids: Vec<i32> = historical_semantics
                .iter()
                .map(|c| c.canonical.id)
                .collect();
            kg.semantic_secondary_categories(&canonical_ids)
                .await
                .wrap_err("failed to load secondary categories from knowledge graph")?
        };

        tracing::info!(
            "Mapping {} extracted semantic(s) against {} historical semantic(s) (batch_size={})",
            extracted_with_id.len(),
            historical_semantics.len(),
            options.batch_size
        );

        let extract_ids: BTreeSet<i32> = extracted_with_id.iter().map(|(id, _)| *id).collect();
        let system_prompt = build_system_prompt(&self.profile, &self.language_prompt_prefix);

        let pass1_matches = batch_match(
            llm,
            options,
            &system_prompt,
            &extracted_with_id,
            &extract_ids,
            &historical_semantics,
            &secondary_categories,
            "pass1",
        )
        .await?;

        let mut matches = pass1_matches;
        dedup_keep_strongest(&mut matches);

        // Pass 2: widen to remaining categories for any extracts still
        // unmatched after pass 1.
        let pass1_matched_ids: BTreeSet<i32> = matches.iter().map(|m| m.extract_id).collect();
        let unmatched_after_pass1: Vec<(i32, &ExtractedSemantic)> = extracted_with_id
            .iter()
            .filter(|(id, _)| !pass1_matched_ids.contains(id))
            .copied()
            .collect();

        let pass1_categories: BTreeSet<DeFiCategory> = categories.iter().copied().collect();
        let fallback_categories: Vec<DeFiCategory> = DeFiCategory::ALL
            .iter()
            .copied()
            .filter(|c| !pass1_categories.contains(c))
            .collect();

        let pass2_historicals = if unmatched_after_pass1.is_empty()
            || fallback_categories.is_empty()
        {
            Vec::new()
        } else {
            tracing::info!(
                "Widening match for {} unmatched extract(s) across {} fallback categor(ies): {}",
                unmatched_after_pass1.len(),
                fallback_categories.len(),
                fallback_categories.iter().map(|c| c.as_str()).join(", ")
            );
            kg.canonical_semantics_with_children_for_categories(&fallback_categories)
                .await
                .wrap_err("failed to load fallback historical semantics from knowledge graph")?
        };

        let pass2_matches = if pass2_historicals.is_empty() {
            Vec::new()
        } else {
            tracing::info!(
                "Pass 2: matching {} unmatched extract(s) against {} fallback historical(s)",
                unmatched_after_pass1.len(),
                pass2_historicals.len(),
            );
            let pass2_extract_ids: BTreeSet<i32> =
                unmatched_after_pass1.iter().map(|(id, _)| *id).collect();
            batch_match(
                llm,
                options,
                &system_prompt,
                &unmatched_after_pass1,
                &pass2_extract_ids,
                &pass2_historicals,
                &secondary_categories,
                "pass2",
            )
            .await?
        };

        let pass2_match_count = pass2_matches.len();
        matches.extend(pass2_matches);
        dedup_keep_strongest(&mut matches);

        let matched_extract_ids: BTreeSet<i32> = matches.iter().map(|m| m.extract_id).collect();
        let total_historical_pool = historical_semantics.len() + pass2_historicals.len();
        for (extract_id, sem) in &extracted_with_id {
            if !matched_extract_ids.contains(extract_id) {
                tracing::warn!(
                    "Extracted semantic '{}' (id={}) had no historical match across {} candidate(s) even after fallback widening",
                    sem.name,
                    extract_id,
                    total_historical_pool
                );
            }
        }

        let matched_historical_ids: Vec<i32> =
            matches.iter().map(|m| m.historical_id).unique().collect();

        let mut semantic_by_id: HashMap<i32, semantic_node::Model> = historical_semantics
            .into_iter()
            .map(|c| (c.canonical.id, c.canonical))
            .collect();
        for c in pass2_historicals {
            semantic_by_id.entry(c.canonical.id).or_insert(c.canonical);
        }

        let finding_pairs = if matched_historical_ids.is_empty() {
            Vec::new()
        } else {
            kg.findings_for_semantic_ids(&matched_historical_ids)
                .await
                .wrap_err("failed to load historical findings for matched semantics")?
        };

        // Folded children + their merge-edge notes, so the mirror carries each
        // canonical's merged variants (concrete deltas) for downstream rendering.
        let mut sem_children = kg
            .semantic_children_with_notes(&matched_historical_ids)
            .await
            .wrap_err("failed to load semantic children for matched canonicals")?;
        let finding_ids: Vec<i32> = finding_pairs
            .iter()
            .map(|l| l.finding.id)
            .unique()
            .collect();
        let mut finding_children = kg
            .finding_children_with_notes(&finding_ids)
            .await
            .wrap_err("failed to load finding children for matched canonicals")?;

        let mut findings_by_semantic: BTreeMap<i32, Vec<HistoricalLinkedFinding>> = BTreeMap::new();
        for linked in finding_pairs {
            // Attach children to the first link of each finding only — the write
            // path dedups child rows, and one edge insert per (child, canonical)
            // avoids a duplicate-key collision.
            let raw_children = finding_children
                .remove(&linked.finding.id)
                .unwrap_or_default()
                .into_iter()
                .map(|(finding, d, p, x)| HistoricalFindingChild {
                    finding,
                    appended_description: d,
                    appended_patterns: p,
                    appended_exploits: x,
                })
                .collect();
            findings_by_semantic
                .entry(linked.canonical_semantic_id)
                .or_default()
                .push(HistoricalLinkedFinding {
                    finding: linked.finding,
                    strength: linked.strength,
                    evidence: linked.evidence,
                    raw_children,
                    rendered_description: String::new(),
                    rendered_patterns: String::new(),
                    rendered_exploits: String::new(),
                });
        }

        let historicals: Vec<HistoricalSemanticRecord> = matched_historical_ids
            .iter()
            .filter_map(|id| {
                semantic_by_id.get(id).cloned().map(|semantic| {
                    let findings = findings_by_semantic.remove(id).unwrap_or_default();
                    let raw_children = sem_children
                        .remove(id)
                        .unwrap_or_default()
                        .into_iter()
                        .map(|(node, appended_description)| HistoricalSemanticChild {
                            node,
                            appended_description,
                        })
                        .collect();
                    HistoricalSemanticRecord {
                        semantic,
                        findings,
                        raw_children,
                        rendered_description: String::new(),
                    }
                })
            })
            .collect();

        let mut strength_counts = StrengthCounts::default();
        for m in &matches {
            strength_counts.bump(m.strength);
        }

        let outcome = MapperOutcome {
            extracted_count: extracted_with_id.len(),
            historical_considered: total_historical_pool,
            matched_pair_count: matches.len(),
            widened_match_pair_count: pass2_match_count,
            unmatched_extract_count: extracted_with_id.len() - matched_extract_ids.len(),
            historical_finding_count: historicals.iter().map(|r| r.findings.len()).sum(),
            strength_counts,
        };

        let set = SemanticMatchSet {
            historicals,
            matches,
        };
        repo.write_semantic_match_results(&set)
            .await
            .wrap_err("failed to persist mapper output to project database")?;

        tracing::info!(
            "Knowledge Mapper wrote {} match edge(s) (High={}/Medium={}/Low={}; {} from widened fallback) covering {} historical semantic(s) and {} finding(s); {} extracted semantic(s) had no match",
            outcome.matched_pair_count,
            outcome.strength_counts.high,
            outcome.strength_counts.medium,
            outcome.strength_counts.low,
            outcome.widened_match_pair_count,
            set.historicals.len(),
            outcome.historical_finding_count,
            outcome.unmatched_extract_count,
        );

        Ok(outcome)
    }
}

/// Drive one batched matching pass.
async fn batch_match(
    llm: &LLM,
    options: &MapperOptions,
    system_prompt: &str,
    extracts: &[(i32, &ExtractedSemantic)],
    valid_extract_ids: &BTreeSet<i32>,
    historicals: &[CanonicalWithChildren<semantic_node::Model>],
    secondary_categories: &HashMap<i32, Vec<DeFiCategory>>,
    label_prefix: &str,
) -> Result<Vec<SemanticMatch>> {
    if extracts.is_empty() || historicals.is_empty() {
        return Ok(Vec::new());
    }

    let batch_size = options.batch_size.max(1);
    let concurrency = options.concurrency.max(1);
    let max_rendered_raw_children = options.max_rendered_raw_children;
    let total_batches = historicals.len().div_ceil(batch_size);
    // Shared, read-only inputs every worker needs. Wrapped in `Arc`
    // once so each spawned worker holds a cheap clone — `JoinSet`
    // tasks must be `'static`, so nothing can be borrowed across the
    // spawn boundary (same async-channel + `JoinSet` idiom the fuzz
    // runtime uses). Workers only need the extract *ids* (for the
    // forced-per-candidate-emit check) plus the pre-rendered extract
    // block, so the `ExtractedSemantic` values themselves never have
    // to cross the boundary.
    let extracted_block = Arc::new(render_extracted_block(extracts));
    let system_prompt = Arc::new(system_prompt.to_string());
    let valid_extract_ids = Arc::new(valid_extract_ids.clone());
    let cache_key = options.cache_key.clone();
    let label_prefix = Arc::new(label_prefix.to_string());
    let secondary_categories = Arc::new(secondary_categories.clone());

    type Batch = Vec<CanonicalWithChildren<semantic_node::Model>>;
    // Job queue sized to hold every batch, so feeding it never blocks
    // on a slow worker; results come back on an mpsc the collector
    // drains.
    let (tx, rx) = async_channel::bounded::<(usize, Batch)>(total_batches + 1);
    let (out_tx, mut out_rx) =
        tokio::sync::mpsc::channel::<Result<Vec<SemanticMatch>>>(concurrency + 1);

    let mut workers = JoinSet::new();
    for _ in 0..concurrency {
        let rx = rx.clone();
        let out = out_tx.clone();
        let llm = llm.clone();
        let extracted_block = extracted_block.clone();
        let system_prompt = system_prompt.clone();
        let valid_extract_ids = valid_extract_ids.clone();
        let cache_key = cache_key.clone();
        let label_prefix = label_prefix.clone();
        let secondary_categories = secondary_categories.clone();
        workers.spawn(async move {
            while let Ok((batch_idx, batch)) = rx.recv().await {
                let label = format!(
                    "knowledge-mapper {} batch {}/{}",
                    label_prefix,
                    batch_idx + 1,
                    total_batches
                );
                let res = run_one_batch(
                    &llm,
                    cache_key.as_deref(),
                    &system_prompt,
                    &extracted_block,
                    &valid_extract_ids,
                    &batch,
                    max_rendered_raw_children,
                    &secondary_categories,
                    &label,
                )
                .await;
                // A closed receiver means the collector hit an error
                // and bailed; stop pulling work.
                if out.send(res).await.is_err() {
                    break;
                }
            }
        });
    }
    drop(out_tx);
    drop(rx);

    for (batch_idx, batch) in historicals.chunks(batch_size).enumerate() {
        tx.send((batch_idx, batch.to_vec()))
            .await
            .expect("mapper workers kept alive");
    }
    drop(tx);

    let mut matches: Vec<SemanticMatch> = Vec::new();
    let mut first_err: Option<color_eyre::eyre::Error> = None;
    while let Some(res) = out_rx.recv().await {
        match res {
            Ok(batch_matches) => matches.extend(batch_matches),
            Err(err) => {
                // Abort the remaining batches the moment one fails —
                // matches the old `?`-on-first-error behaviour without
                // waiting on in-flight LLM calls.
                first_err = Some(err);
                break;
            }
        }
    }
    if first_err.is_some() {
        workers.abort_all();
    }
    drop(out_rx);
    while let Some(joined) = workers.join_next().await {
        match joined {
            Ok(()) => {}
            Err(err) if err.is_cancelled() => {}
            Err(err) => tracing::error!("knowledge-mapper worker task panicked: {err:#}"),
        }
    }
    if let Some(err) = first_err {
        return Err(err);
    }

    Ok(matches)
}

/// Match one batch of historical semantics against the (pre-rendered)
/// extract block: one LLM call + the per-record validation /
/// forced-per-candidate-emit accounting. Returns the surviving
/// [`SemanticMatch`] rows; an `Err` aborts the whole mapping pass.
async fn run_one_batch(
    llm: &LLM,
    cache_key: Option<&str>,
    system_prompt: &str,
    extracted_block: &str,
    valid_extract_ids: &BTreeSet<i32>,
    batch: &[CanonicalWithChildren<semantic_node::Model>],
    max_rendered_raw_children: usize,
    secondary_categories: &HashMap<i32, Vec<DeFiCategory>>,
    label: &str,
) -> Result<Vec<SemanticMatch>> {
    let user_prompt = build_user_prompt(
        extracted_block,
        batch,
        max_rendered_raw_children,
        secondary_categories,
    );

    let response: MatchResponse = prompt_match_response(llm, system_prompt, &user_prompt, cache_key, label).await?;

    let valid_historical_ids: BTreeSet<i32> = batch.iter().map(|c| c.canonical.id).collect();

    // Forced per-candidate emit (plan_link.md §6.4): every
    // (extract_id × historical_id) pair shown in this batch should
    // appear in the response. We don't bounce the response on missing
    // coverage (no agent re-loop on the JSON path right now), but we
    // log a warning so calibration runs can see when the model is
    // sneakily omitting Low pairs.
    let expected_pairs: BTreeSet<(i32, i32)> = valid_extract_ids
        .iter()
        .flat_map(|eid| valid_historical_ids.iter().map(move |hid| (*eid, *hid)))
        .collect();
    let mut emitted_pairs: BTreeSet<(i32, i32)> = BTreeSet::new();
    let mut matches: Vec<SemanticMatch> = Vec::new();

    for record in response.matches {
        if !valid_historical_ids.contains(&record.historical_id) {
            tracing::warn!(
                "{label}: dropping match for unknown historical id {}",
                record.historical_id
            );
            continue;
        }
        if !valid_extract_ids.contains(&record.extract_id) {
            tracing::warn!(
                "{label}: dropping match for unknown extract id {}",
                record.extract_id
            );
            continue;
        }
        // `strength` / `evidence` are `Value` so one malformed field cannot
        // fail the batch; require an actual string here.
        let Some(strength_raw) = record.strength.as_str() else {
            tracing::warn!(
                "{label}: dropping (extract={}, historical={}): strength is not a string: {}",
                record.extract_id,
                record.historical_id,
                record.strength
            );
            continue;
        };
        let Some(strength) = MatchStrength::parse(strength_raw) else {
            tracing::warn!(
                "{label}: dropping (extract={}, historical={}): unparseable strength '{}'",
                record.extract_id,
                record.historical_id,
                strength_raw
            );
            continue;
        };
        let evidence = record
            .evidence
            .as_ref()
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        let floor = match strength {
            MatchStrength::High | MatchStrength::Medium => MIN_EVIDENCE_LEN_HIGH_MEDIUM,
            MatchStrength::Low => MIN_EVIDENCE_LEN_LOW,
        };
        if evidence.len() < floor {
            tracing::warn!(
                "{label}: dropping (extract={}, historical={}, strength={}): evidence shorter than {} chars: {:?}",
                record.extract_id,
                record.historical_id,
                strength,
                floor,
                evidence
            );
            continue;
        }
        emitted_pairs.insert((record.extract_id, record.historical_id));
        matches.push(SemanticMatch {
            extract_id: record.extract_id,
            historical_id: record.historical_id,
            strength,
            evidence,
        });
    }

    let missing: Vec<(i32, i32)> = expected_pairs.difference(&emitted_pairs).copied().collect();
    if !missing.is_empty() {
        tracing::warn!(
            "{label}: response omitted {} of {} expected (extract, historical) pair(s); \
             plan §6.4 contract requires per-candidate emit. First few missing: {:?}",
            missing.len(),
            expected_pairs.len(),
            &missing[..missing.len().min(5)],
        );
    }

    Ok(matches)
}

/// Ask the mapper LLM for a [`MatchResponse`], repairing the two
/// single-character JSON corruptions the model reliably produces on
/// long responses.
///
/// Mapper replies run to ~160K characters (16 historicals × up to 87
/// extracts forces an entry per pair), and at that length the model
/// typos a key roughly one call in three — `"historical_id=45` for
/// `"historical_id":45`, or `"historical_id:15` with the closing quote
/// dropped. `llmy`'s typed retry only re-issues the whole call, so a
/// single dropped character costs a full 30s+ regeneration and still
/// fails if the next sample typos too. Since the corruption is
/// mechanical and unambiguous, repair it locally first and only fall
/// back to a retry when the response is genuinely unusable.
async fn prompt_match_response(
    llm: &LLM,
    system_prompt: &str,
    user_prompt: &str,
    cache_key: Option<&str>,
    label: &str,
) -> Result<MatchResponse> {
    // Fetch the reply as text rather than through `prompt_json_with_retry`:
    // that path deserializes inside llmy, so a key typo is rejected before
    // this code ever sees the characters it needs to repair, and each
    // rejection costs a full regenerations of a ~160K-char response.
    //
    // One retry loop covers both transport failures (connection refused,
    // reset, DNS) and degenerate responses (empty array, no content):
    // llmy retries transport errors without backoff, so its attempts are
    // all spent before a brief network blip clears, and a stressed
    // provider occasionally returns a valid-but-empty completion that
    // yields no entries at all. Both are worth one more cheap try — the
    // prompt is ~97% prompt-cache hits — before giving up on the batch.
    let mut last_err: Option<String> = None;
    for attempt in 0..NETWORK_RETRY_ATTEMPTS {
        if attempt > 0 {
            // A gateway timeout already burned its full proxy window, so
            // back off further than for an instantly-refused connection.
            let base = if last_err
                .as_deref()
                .is_some_and(|e| e.to_lowercase().contains("524"))
            {
                GATEWAY_TIMEOUT_BASE_DELAY
            } else {
                NETWORK_RETRY_BASE_DELAY
            };
            let delay = base * (1 << (attempt - 1).min(4));
            tracing::warn!(
                "{label}: mapper attempt {attempt}/{NETWORK_RETRY_ATTEMPTS} unusable \
                 ({}); retrying in {delay:?}",
                last_err.as_deref().unwrap_or("unknown")
            );
            tokio::time::sleep(delay).await;
        }

        let resp = match llm
            .prompt_once_with_retry(system_prompt, user_prompt, None, cache_key, None)
            .await
        {
            Ok(r) => r,
            Err(err) => {
                if !is_transient_transport_error(&err) {
                    return Err(err.into());
                }
                last_err = Some(err.to_string());
                continue;
            }
        };

        let Some(body) = resp
            .choices
            .first()
            .and_then(|c| c.inner.message.content.as_deref())
            .map(|s| s.to_string())
        else {
            last_err = Some("response had no content".to_string());
            continue;
        };

        // The model sometimes wraps the object in a ```json fence even
        // though the prompt asks for bare JSON.
        let body = llmy::client::filters::strip_markdown_fence(&body)
            .unwrap_or_else(|| body.clone());
        let entries = salvage_match_entries(&repair_match_json(&body));

        if entries.is_empty() {
            // `{"matches":[]}` is a legitimate reply shape (a stressed
            // provider can shorten a completion to nothing), so it is a
            // candidate for one retry — but if the model insists on
            // returning nothing, the right outcome is an empty match set
            // plus a loud warning, not aborting a multi-hour pass.
            last_err = Some(format!("response yielded no match entries ({} chars)", body.len()));
            if body.len() > UNUSABLE_DUMP_MAX_CHARS {
                let dump = std::env::temp_dir().join(format!(
                    "knowdit-mapper-unparseable-{}.txt",
                    std::process::id()
                ));
                let _ = std::fs::write(&dump, &body);
                tracing::warn!(
                    "{label}: large response yielded no entries; body written to {}",
                    dump.display()
                );
            }
            continue;
        }

        let mut matches = Vec::with_capacity(entries.len());
        let mut skipped = 0usize;
        for entry in entries {
            match serde_json::from_value::<MatchEntry>(entry) {
                Ok(entry) => matches.push(entry),
                Err(_) => skipped += 1,
            }
        }
        if skipped > 0 {
            tracing::warn!(
                "{label}: skipped {skipped} malformed match entr{} of {} (missing or ill-typed field)",
                if skipped == 1 { "y" } else { "ies" },
                matches.len() + skipped
            );
        }
        return Ok(MatchResponse { matches });
    }

    // Exhausted. An empty-but-valid response is not a failure — it is a
    // batch with nothing to say, which downstream "omitted pair" logging
    // already surfaces. Only a persistent transport error aborts.
    if let Some(err) = &last_err
        && err.contains("no match entries")
    {
        tracing::error!(
            "{label}: model returned no match entries on {NETWORK_RETRY_ATTEMPTS} \
             consecutive attempts; continuing with an empty match set"
        );
        return Ok(MatchResponse { matches: Vec::new() });
    }
    Err(color_eyre::eyre::eyre!(
        "mapper request failed after {NETWORK_RETRY_ATTEMPTS} attempts: {}",
        last_err.unwrap_or_else(|| "unknown error".to_string())
    ))
}

/// Pull every well-formed object out of the `matches` array, ignoring
/// anything that is not a JSON object.
///
/// The model reliably produces *some* valid entries even when it mangles
/// one, so parsing entry-by-entry means a single bad record costs one
/// pair rather than the whole batch. Falls back to parsing the whole
/// document when the array itself cannot be scanned.
fn salvage_match_entries(body: &str) -> Vec<serde_json::Value> {
    let array = find_matches_array(body);
    let Some(array) = array else {
        return Vec::new();
    };

    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut start = None;

    for (idx, ch) in array.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => {
                if depth == 0 {
                    start = Some(idx);
                }
                depth += 1;
            }
            '}' => {
                if depth == 0 {
                    continue;
                }
                depth -= 1;
                if depth == 0
                    && let Some(s) = start.take()
                    && let Ok(value) = serde_json::from_str::<serde_json::Value>(&array[s..=idx])
                {
                    out.push(value);
                }
            }
            _ => {}
        }
    }
    out
}

/// The contents between the `[` that opens the `matches` array and the
/// last `]` in the body.
///
/// A response cut off by the completion cap never closes the array, so
/// the upper bound falls back to the end of the body — the entries that
/// did arrive are still worth keeping.
fn find_matches_array(body: &str) -> Option<&str> {
    let key = body.find("\"matches\"")?;
    let after_key = &body[key..];
    let open = key + after_key.find('[')? + 1;
    let close = body.rfind(']').unwrap_or(body.len());
    (close > open).then(|| &body[open..close])
}

/// Repair the key corruptions the mapper model emits in long JSON
/// arrays, without touching text inside string values:
///
/// * `"historical_id=45` — `=` where `:` belongs, with the key's
///   closing quote dropped.
/// * `"historical_id:15` — closing quote dropped before the `:`.
/// * `{extract_id=3` — opening quote dropped as well.
/// * `{"historical_id"=56` — key quoted correctly but `=` used.
/// * a garbled fragment such as `{"extract_id":2":0}` — an entry the
///   model aborted mid-write. Such a fragment carries no usable fields,
///   so it is dropped rather than repaired; the surrounding entries are
///   intact and the pair simply shows up as "missing" downstream.
///
/// Only the four fields of a match entry are considered, and only in
/// key position — a value like `"evidence":"id=3"` is left alone
/// because the scan never treats text after an opening quote as a key.
fn repair_match_json(body: &str) -> String {
    const KEYS: [&str; 4] = ["extract_id", "historical_id", "strength", "evidence"];

    let bytes = body.as_bytes();
    let mut out = String::with_capacity(body.len());
    let mut i = 0;
    let mut in_string = false;

    while i < bytes.len() {
        let c = bytes[i];

        if in_string {
            // Inside a string: copy verbatim until the closing quote.
            if c == b'\\' && i + 1 < bytes.len() {
                out.push_str(&body[i..i + 2]);
                i += 2;
                continue;
            }
            if c == b'"' {
                in_string = false;
            }
            out.push(c as char);
            i += 1;
            continue;
        }

        // Outside a string. A key starts at `"` or, when the model has
        // dropped the opening quote, at a bare `{`/`,` followed by a key.
        let starts_bare = c == b'{' || c == b',';
        let key_start =
            (c == b'"' || (starts_bare && i + 1 < bytes.len())).then_some(i + 1);

        if let Some(start) = key_start
            && let Some(kw) = KEYS
                .iter()
                .find(|kw| body[start..].starts_with(*kw))
        {
            let after = start + kw.len();
            let rest = &body[after..];
            // Expect an optional `"` then `:` or `=`.
            let (consumed, had_quote, sep_ok) = match rest.strip_prefix('"') {
                Some(r) => (
                    1,
                    true,
                    r.trim_start().starts_with(':') || r.trim_start().starts_with('='),
                ),
                None => (
                    0,
                    false,
                    rest.trim_start().starts_with(':') || rest.trim_start().starts_with('='),
                ),
            };
            if sep_ok {
                // When the opening quote was dropped the scan started just
                // past a bare `{`/`,`; that delimiter still has to be
                // written out or the object never opens.
                if !had_quote && starts_bare {
                    out.push(c as char);
                }
                out.push_str(&format!("\"{kw}\":"));
                let mut j = after + consumed;
                // Skip the original separator (and its whitespace).
                while j < bytes.len() && (bytes[j] as char).is_whitespace() {
                    j += 1;
                }
                if j < bytes.len() && (bytes[j] == b':' || bytes[j] == b'=') {
                    j += 1;
                }
                while j < bytes.len() && (bytes[j] as char).is_whitespace() {
                    j += 1;
                }
                // `"extract_id="36"` — the model put the dropped quote on
                // the *value* instead of the key, turning an integer into a
                // string. Strip it so the numeric fields still deserialize.
                if j < bytes.len()
                    && bytes[j] == b'"'
                    && matches!(*kw, "extract_id" | "historical_id")
                {
                    let mut k = j + 1;
                    while k < bytes.len() && (bytes[k] as char).is_ascii_digit() {
                        k += 1;
                    }
                    if k > j + 1 && k < bytes.len() && bytes[k] == b'"' {
                        out.push_str(&body[j + 1..k]);
                        i = k + 1;
                        continue;
                    }
                }
                i = j;
                continue;
            }
        }

        if c == b'"' {
            in_string = true;
        }
        out.push(c as char);
        i += 1;
    }

    // Neutralise stray quotes so the bracket scan in
    // `salvage_match_entries` sees the real structure, and so an entry
    // like `{"extract_id":2":0}` is rejected on its own instead of
    // swallowing the rest of the array as one long string.
    neutralise_stray_quotes(&out)
}

/// Replace stray quotes that sit between a key's `:` and a numeric
/// value — `{"extract_id":2":0}` becomes `{"extract_id":2:0}` — so the
/// bracket scanner in [`salvage_match_entries`] is not fooled into
/// treating the rest of the array as one long string. The entry is still
/// rejected afterwards; this only keeps the scan honest.
fn neutralise_stray_quotes(body: &str) -> String {
    const KEYS: [&str; 4] = ["extract_id", "historical_id", "strength", "evidence"];
    let mut out = String::with_capacity(body.len());
    let mut rest = body;

    while let Some(pos) = rest.find('"') {
        out.push_str(&rest[..pos]);
        let after = &rest[pos + 1..];
        let Some(kw) = KEYS.iter().find(|k| after.starts_with(*k)) else {
            out.push('"');
            rest = after;
            continue;
        };
        let after_key = &after[kw.len()..];
        let Some(after_colon) = after_key.strip_prefix("\":") else {
            out.push('"');
            rest = after;
            continue;
        };
        let digits: String = after_colon.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            out.push('"');
            rest = after;
            continue;
        }
        let tail = &after_colon[digits.len()..];
        if let Some(tail) = tail.strip_prefix('"') {
            // `"key":"<digits>"` is a legitimate string value for
            // `strength`/`evidence`, but never for the id fields.
            if !matches!(*kw, "extract_id" | "historical_id") {
                out.push('"');
                rest = after;
                continue;
            }
            out.push_str(&format!("\"{kw}\":{digits}"));
            rest = tail;
        } else {
            out.push('"');
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// Drop duplicate `(extract_id, historical_id)` rows; on collision
/// keep the higher-strength entry (so a Pass-2 retry that re-emits
/// the same pair at a different strength doesn't downgrade Pass 1).
fn dedup_keep_strongest(matches: &mut Vec<SemanticMatch>) {
    let mut by_key: BTreeMap<(i32, i32), SemanticMatch> = BTreeMap::new();
    for m in matches.drain(..) {
        let key = (m.extract_id, m.historical_id);
        by_key
            .entry(key)
            .and_modify(|existing| {
                if m.strength.rank() > existing.strength.rank() {
                    *existing = m.clone();
                }
            })
            .or_insert(m);
    }
    matches.extend(by_key.into_values());
    // Stable sort by (extract_id, historical_id) so downstream snapshots
    // are deterministic.
    matches.sort_by(|a, b| (a.extract_id, a.historical_id).cmp(&(b.extract_id, b.historical_id)));
}

/// Summary statistics for one mapper pass.
#[derive(Debug, Clone, Default)]
pub struct MapperOutcome {
    pub extracted_count: usize,
    pub historical_considered: usize,
    pub matched_pair_count: usize,
    /// Subset of `matched_pair_count` rescued by the widened fallback.
    pub widened_match_pair_count: usize,
    pub unmatched_extract_count: usize,
    pub historical_finding_count: usize,
    pub strength_counts: StrengthCounts,
}

#[derive(Debug, Clone, Default)]
pub struct StrengthCounts {
    pub high: usize,
    pub medium: usize,
    pub low: usize,
}

impl StrengthCounts {
    fn bump(&mut self, s: MatchStrength) {
        match s {
            MatchStrength::High => self.high += 1,
            MatchStrength::Medium => self.medium += 1,
            MatchStrength::Low => self.low += 1,
        }
    }
}

fn collect_categories(
    extracted: &[ExtractedSemantic],
    extra: &[DeFiCategory],
) -> Vec<DeFiCategory> {
    let mut cats: BTreeSet<DeFiCategory> = extracted.iter().map(|s| s.category).collect();
    cats.insert(DeFiCategory::Others);
    cats.extend(extra.iter().copied());
    cats.into_iter().collect()
}

fn render_extracted_block(extracted: &[(i32, &ExtractedSemantic)]) -> String {
    let mut out = String::new();
    for (id, sem) in extracted {
        out.push_str(&format!(
            "- extract_id={} | category={} | name=\"{}\"\n  definition: {}\n  description: {}\n",
            id,
            sem.category.as_str(),
            sem.name,
            sem.definition.trim(),
            sem.description.trim()
        ));
    }
    out
}

fn render_historical_block(
    batch: &[CanonicalWithChildren<semantic_node::Model>],
    max_rendered_raw_children: usize,
    secondary_categories: &HashMap<i32, Vec<DeFiCategory>>,
) -> String {
    let mut out = String::new();
    for entry in batch {
        let canonical = &entry.canonical;
        let secondary = secondary_categories
            .get(&canonical.id)
            .map(|cats| format!(" (+{})", cats.iter().map(|c| c.as_str()).join(", ")))
            .unwrap_or_default();
        out.push_str(&format!(
            "- historical_id={} | category={}{} | name=\"{}\"\n  definition: {}\n  description: {}\n",
            canonical.id,
            canonical.category.as_str(),
            secondary,
            canonical.name,
            canonical.definition.trim(),
            canonical.description.trim()
        ));
        if !entry.raw_children.is_empty() && max_rendered_raw_children > 0 {
            let total = entry.raw_children.len();
            let shown = total.min(max_rendered_raw_children);
            out.push_str(&format!(
                "  merged_raw_examples ({} of {}; canonical may match if any align):\n",
                shown, total
            ));
            for raw in entry.raw_children.iter().take(shown) {
                out.push_str(&format!(
                    "    * raw \"{}\" — {}\n",
                    raw.name,
                    raw.description.trim().replace('\n', " ")
                ));
            }
            if total > shown {
                out.push_str(&format!(
                    "    * (+{} more merged raw children not shown)\n",
                    total - shown
                ));
            }
        }
    }
    out
}

fn build_user_prompt(
    extracted_block: &str,
    batch: &[CanonicalWithChildren<semantic_node::Model>],
    max_rendered_raw_children: usize,
    secondary_categories: &HashMap<i32, Vec<DeFiCategory>>,
) -> String {
    format!(
        "## Extracted project semantics\n{extracted}\n\
         \n## Historical knowledge-graph semantics (this batch)\n\
         Each historical entry is a *canonical semantic cluster*; `merged_raw_examples` (when \
         present) shows specific raw semantics merged into the canonical and indicates the kinds \
         of behaviours the cluster covers. A `(+X)` after `category` lists secondary categories \
         the canonical also covers via folded variants.\n{historical}\n\
         \n## Task\nEmit ONE entry in the `matches` array for EVERY `(extract_id, historical_id)` \
         pair shown above — no silent omission. Each entry must carry an explicit `strength` from \
         {{`High`, `Medium`, `Low`}} plus an `evidence` rationale. Apply the strength rubric from \
         the system prompt; default to `Low` when uncertain. Evidence length floors: `High` and \
         `Medium` >= 40 chars (cite the specific shared mechanism), `Low` >= 15 chars (a brief \
         reason is enough). Respond with strict JSON in the exact shape:\n\
         ```json\n\
         {{\"matches\":[{{\"extract_id\":1,\"historical_id\":123,\"strength\":\"High\",\"evidence\":\"...\"}},{{\"extract_id\":1,\"historical_id\":124,\"strength\":\"Low\",\"evidence\":\"different mechanism\"}}]}}\n\
         ```\nNever invent ids that were not listed in the inputs. Coverage is mandatory: missing \
         pairs are treated as the agent shirking the contract and a warning is logged.",
        extracted = extracted_block,
        historical =
            render_historical_block(batch, max_rendered_raw_children, secondary_categories),
    )
}

fn build_system_prompt(profile: &ProjectProfile, language_prompt_prefix: &str) -> String {
    let mut profile_block = String::new();
    if !language_prompt_prefix.trim().is_empty() {
        profile_block.push_str(language_prompt_prefix.trim_end());
        profile_block.push_str("\n\n");
    }
    profile_block.push_str("## Project profile\n");
    profile_block.push_str(&format!(
        "Domain summary: {}\n\n",
        profile.domain_summary.trim()
    ));
    profile_block.push_str("Subsystems (mapper compares historical descriptions against these `summary` lines, not against `name`):\n");
    for s in &profile.subsystems {
        profile_block.push_str(&format!("- {} — {}\n", s.name, s.summary.trim()));
    }
    profile_block.push('\n');
    profile_block.push_str(
        "Core components (project source the mapper may cite when justifying a High match):\n",
    );
    for c in &profile.core_components {
        profile_block.push_str(&format!("- {} ({}) — {}\n", c.name, c.path, c.role.trim()));
    }
    profile_block.push('\n');
    profile_block.push_str("Out-of-scope notes (use these to confidently emit Low when the historical's mechanism is excluded):\n");
    profile_block.push_str(profile.out_of_scope_notes.trim());
    profile_block.push_str("\n\n");

    format!(
        "{PREAMBLE}\n\n{profile_block}{RUBRIC}",
        PREAMBLE = MAPPER_PREAMBLE,
        profile_block = profile_block,
        RUBRIC = MAPPER_RUBRIC,
    )
}

const MAPPER_PREAMBLE: &str = "You are the Knowledge Mapper agent for an agentic smart-contract auditing pipeline. For each candidate (project_semantic ↔ historical_semantic) pair you emit, you MUST attach a `strength` label drawn from {High, Medium, Low} together with an `evidence` rationale (>= 40 chars). The label gates downstream cost: High pairs drive expensive spec generation; Low pairs may be abandoned. Wrong-tier labels waste budget — over-labeling as High burns tokens on dead leads; under-labeling as Low drops real bugs. Calibrate deliberately. Reply with strict JSON only, no commentary.";

const MAPPER_RUBRIC: &str = r#"## Strength rubric (single axis: MECHANISM OVERLAP)

Strength is determined by ONE check: does the historical semantic describe a mechanism that this project actually implements?

Both project and historical semantics are LLM-extracted records with the same shape: `category` (one of the DeFi enum values), `name` (free-form short label), `definition` (one-sentence formal statement), `description` (abstract description of user interaction / value flow). Names are NOT drawn from a shared vocabulary — different extractions of the same mechanism may have different names. Judge on definition + description content, not surface name match.

Map your judgment directly to strength. The bar between Medium and Low matters most — most cross-domain pairs should be Low, not Medium.

  High    Direct mechanism match. The historical's definition/description describes the same mechanism that appears in either:
            (a) one of this project's extracted semantics (match on described mechanism, not surface name), OR
            (b) a subsystem listed in the project profile above.
          The historical's described behaviour would apply to project code with at most cosmetic substitution (rename a variable, point at a different function in the same role).

  Medium  Partial mechanism overlap. The historical describes a process that SHARES CONCRETE SUB-STEPS with something the project does but differs in detail. To label Medium you must point to a specific shared sub-pattern that BOTH sides exhibit, such as:
            - same lifecycle stage with different accounting formula (e.g. both compute a per-user pending balance, but with different aggregation math)
            - same access-control style with different role granularity (e.g. both gate on roles, but project uses two roles and historical uses N)
            - same external-call pattern with different re-entrancy surface (e.g. both push tokens then update state, but in different orderings)
          The historical's behaviour cannot be applied without re-interpretation — but the re-interpretation has a concrete anchor on the project side.

          NOT acceptable as Medium evidence:
            - "same DeFiCategory" alone (Services / Others / Lending are far too broad; many unrelated mechanisms share them)
            - "both involve smart contracts" / "both are DeFi"
            - "both have an admin role" without naming what the role gates
          If you cannot name a concrete shared sub-pattern, the answer is Low, not Medium.

  Low     No meaningful mechanism overlap. Includes:
            - historical's mechanism has no analog in the project's extracted semantics or profile subsystems
            - project profile out_of_scope_notes explicitly excludes the historical's subdomain
            - same DeFiCategory but the mechanisms are entirely different (e.g. both Services, but one is timelock and the other is meta-transaction relayer)
          Most cross-domain pairs land here. This is the expected default — do not apologise for emitting Low. You MUST emit a `Low` entry for every candidate that isn't a clear High/Medium — there is no silent omission. The user prompt restates this as the per-batch contract; the response handler logs a warning when pairs are missing.

## Evidence requirements

The `evidence` field is REQUIRED. Length floors are tier-dependent because the new contract forces emitting per-candidate and most candidates land Low:
  - For High (>= 40 chars): describe the shared mechanism in your own words, citing the project-side (extract_id from the user prompt or a subsystem name from the profile) and the historical side (paraphrase its definition/description).
  - For Medium (>= 40 chars): name the concrete shared sub-pattern (NOT a category, NOT "both DeFi"); state what is shared AND what differs (the detail that prevents High).
  - For Low (>= 15 chars): brief reason is fine. State why the mechanism doesn't apply. If you can only write "same category" or "both X-protocols", that is not Medium — that is Low. Don't pad Low evidence; short reasons are expected.

## Anti-patterns

  • DEFAULT TO LOW when uncertain. No penalty for emitting Low. Real penalty for emitting Medium-or-above without concrete shared-sub-pattern evidence.
  • CATEGORY MATCH IS NOT MEDIUM. Two semantics sharing DeFiCategory is not, by itself, partial overlap.
  • DO NOT label every match Medium as a hedge. If you cannot name a concrete sub-pattern, the correct label is Low.
  • DO NOT promote to High solely because category fields agree, or because surface names overlap. High requires same mechanism, not same label.
  • DO NOT downgrade to Low just because the historical is from a different project. That is expected — every historical is from a different project. Downgrade only when the project's extracted semantics / profile subsystems / out_of_scope_notes show the mechanism truly isn't present."#;

#[derive(Debug, Deserialize)]
struct MatchResponse {
    matches: Vec<MatchEntry>,
}

#[derive(Debug, Deserialize)]
struct MatchEntry {
    extract_id: i32,
    historical_id: i32,
    /// The model occasionally emits `null`, a number, or an object here
    /// instead of the requested string. Deserializing into `Value` keeps a
    /// single odd field from failing the whole batch; the entry is then
    /// rejected by the strength/evidence checks further down, which is
    /// exactly the handling a bad value should get.
    strength: serde_json::Value,
    #[serde(default)]
    evidence: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every case below is a verbatim excerpt of a real mapper response
    /// captured with `--llm-debug` on the plether-perps project: the model
    /// corrupts one key per response at ~160K chars.
    #[test]
    fn repairs_key_typos_model_emits_in_long_responses() {
        let cases = [
            // `=` separator with the closing quote dropped.
            (r#"{"matches":[{"extract_id":3,"historical_id=24,"strength":"Low","evidence":"x"}]}"#,
             r#"{"matches":[{"extract_id":3,"historical_id":24,"strength":"Low","evidence":"x"}]}"#),
            // `=` separator with both quotes dropped.
            (r#"{"matches":[{"extract_id=3,"historical_id":24,"strength":"Low","evidence":"x"}]}"#,
             r#"{"matches":[{"extract_id":3,"historical_id":24,"strength":"Low","evidence":"x"}]}"#),
            // Closing quote dropped before the colon.
            (r#"{"matches":[{"extract_id":27,"historical_id:15,"strength":"Low","evidence":"x"}]}"#,
             r#"{"matches":[{"extract_id":27,"historical_id":15,"strength":"Low","evidence":"x"}]}"#),
            // Opening quote dropped on the first key of an object.
            (r#"{"matches":[{extract_id=3,"historical_id":24,"strength":"Low","evidence":"x"}]}"#,
             r#"{"matches":[{"extract_id":3,"historical_id":24,"strength":"Low","evidence":"x"}]}"#),
        ];
        for (input, expected) in cases {
            assert_eq!(repair_match_json(input), expected, "input: {input}");
        }
    }

    /// A `=` inside a value must survive: only key positions are rewritten.
    #[test]
    fn repair_leaves_string_values_alone() {
        let input = r#"{"matches":[{"extract_id":1,"historical_id":2,"strength":"Low","evidence":"mint id=3 and extract_id=4 differ"}]}"#;
        assert_eq!(repair_match_json(input), input);
        let parsed: MatchResponse = serde_json::from_str(&repair_match_json(input)).unwrap();
        assert_eq!(
            parsed.matches[0].evidence.as_ref().and_then(|v| v.as_str()),
            Some("mint id=3 and extract_id=4 differ")
        );
    }

    #[test]
    fn repaired_response_deserializes_into_match_entries() {
        let input = r#"{"matches":[{"extract_id":1,"historical_id=2,"strength":"Low","evidence":"a"},{"extract_id":1,"historical_id:3,"strength":"High","evidence":"b"}]}"#;
        let parsed: MatchResponse = serde_json::from_str(&repair_match_json(input)).unwrap();
        assert_eq!(parsed.matches.len(), 2);
        assert_eq!(parsed.matches[1].historical_id, 3);
        assert_eq!(parsed.matches[1].strength.as_str(), Some("High"));
    }

    /// Valid JSON must pass through byte-identical.
    #[test]
    fn repair_is_identity_on_valid_json() {
        let input = r#"{"matches":[{"extract_id":1,"historical_id":2,"strength":"Low","evidence":"e"}]}"#;
        assert_eq!(repair_match_json(input), input);
    }
}

#[cfg(test)]
mod real_capture_tests {
    use super::*;

    /// (label, response body as the model emitted it, entries expected back)
    const CAPTURES: [(&str, &str, usize); 10] = [
        ("equals_with_value_quote", r##"{"matches":[{"extract_id="36","historical_id":23,"strength":"Low","evidence":"vault price manipulation; unrelated"}]}"##, 1),
        ("equals_no_quotes", r##"{"matches":[{extract_id=3,"historical_id":24,"strength":"Medium","evidence":"both maintain segregated collateral accounting"}]}"##, 1),
        ("equals_closing_dropped", r##"{"matches":[{"extract_id":4,"historical_id=15,"strength":"Low","evidence":"unbounded mint; unrelated"}]}"##, 1),
        ("colon_closing_dropped", r##"{"matches":[{"extract_id":27,"historical_id:15,"strength":"Low","evidence":"unbounded mint; project seed floor prevents drawdown"}]}"##, 1),
        ("garbled_fragment_between", r##"{"matches":[{"extract_id":2,"historical_id":3,"strength":"Low","evidence":"a"},{"extract_id=2":0}, {"extract_id":2,"historical_id":6,"strength":"Low","evidence":"b"}]}"##, 2),
        ("fenced", "```json\n{\"matches\":[{\"extract_id\":1,\"historical_id=45,\"strength\":\"Low\",\"evidence\":\"reward hack\"}]}\n```", 1),
        ("clean", r#"{"matches":[{"extract_id":1,"historical_id":45,"strength":"Low","evidence":"no overlap"}]}"#, 1),
        // `"evidence":null` is valid JSON but was not a String. This aborted
        // the whole mapping pass with "invalid type: null, expected a string"
        // until the entry fields became `Value`.
        ("null_evidence", r#"{"matches":[{"extract_id":1,"historical_id":2,"strength":"Low","evidence":null}]}"#, 1),
        // A non-string `strength` must not fail the batch either.
        ("numeric_strength", r#"{"matches":[{"extract_id":1,"historical_id":2,"strength":3,"evidence":"some reason here"}]}"#, 1),
        // `{"matches":[]}` must be a warning-and-continue, not an abort —
        // observed live batch 60/457 when the provider was degraded.
        ("empty_array", r#"{"matches":[]}"#, 0),
    ];

    #[test]
    fn every_captured_corruption_parses_with_all_entries_intact() {
        for (label, raw, expected) in CAPTURES {
            let body = llmy::client::filters::strip_markdown_fence(raw)
                .unwrap_or_else(|| raw.to_string());
            let repaired = repair_match_json(&body);
            let entries = salvage_match_entries(&repaired);
            assert_eq!(entries.len(), expected, "{label}: wrong entry count salvaged");
            let parsed: MatchResponse = MatchResponse {
                matches: entries
                    .into_iter()
                    .filter_map(|e| serde_json::from_value::<MatchEntry>(e).ok())
                    .collect(),
            };
            assert_eq!(parsed.matches.len(), expected, "{label}");
            if label == "null_evidence" {
                assert!(parsed.matches[0].evidence.is_none(), "null must stay null");
                continue;
            }
            if label == "numeric_strength" {
                assert_eq!(parsed.matches[0].strength.as_str(), None);
                continue;
            }
            if label == "empty_array" {
                assert!(parsed.matches.is_empty(), "empty array must stay empty");
                continue;
            }
            for m in &parsed.matches {
                assert!(m.extract_id > 0, "{label}: lost extract_id");
                assert!(m.historical_id > 0, "{label}: lost historical_id");
                assert!(
                    m.evidence.as_ref().and_then(|v| v.as_str()).is_some_and(|e| !e.is_empty()),
                    "{label}: lost evidence"
                );
            }
        }
    }
}

#[cfg(test)]
mod truncation_tests {
    use super::*;

    /// A response cut off by the completion cap never closes its array.
    /// Before this was handled, one truncated batch produced zero entries
    /// and aborted the run; the complete entries that did arrive should
    /// survive.
    #[test]
    fn unterminated_array_still_yields_complete_entries() {
        let body = r#"{"matches":[
            {"extract_id":1,"historical_id":2,"strength":"Low","evidence":"first"},
            {"extract_id":1,"historical_id":3,"strength":"Low","evidence":"second"},
            {"extract_id":1,"hist"#;
        let repaired = repair_match_json(body);
        let entries = salvage_match_entries(&repaired);
        assert_eq!(entries.len(), 2, "complete entries before the cut must survive");
        assert_eq!(entries[1]["historical_id"], 3);
    }

    /// The same body with its closing bracket must parse identically.
    #[test]
    fn terminated_array_is_unaffected() {
        let body = r#"{"matches":[{"extract_id":1,"historical_id":2,"strength":"Low","evidence":"a"},{"extract_id":1,"historical_id":3,"strength":"Low","evidence":"b"}]}"#;
        let entries = salvage_match_entries(&repair_match_json(body));
        assert_eq!(entries.len(), 2);
    }
}

#[cfg(test)]
mod transient_error_tests {
    use super::*;
    use llmy::agent::LLMYError;

    #[test]
    fn connection_refused_is_retried() {
        // The exact failure this guards: `tcp connect error` / `os error 111`.
        let err = LLMYError::IO(std::io::Error::new(
            std::io::ErrorKind::ConnectionRefused,
            "Connection refused (os error 111)",
        ));
        assert!(is_transient_transport_error(&err));
    }

    #[test]
    fn deterministic_errors_are_not_retried() {
        assert!(!is_transient_transport_error(&LLMYError::OutputLength));
        assert!(!is_transient_transport_error(&LLMYError::Filtered(
            "content policy".into()
        )));
    }
}

#[cfg(test)]
mod gateway_timeout_tests {
    use super::*;
    use llmy::agent::LLMYError;

    /// Cloudflare's proxy timeout surfaces as a bare `error code: 524`.
    /// Before this was classified, it fell through as a hard failure and
    /// killed the pass on the first large batch.
    /// The substring match, exercised on the exact text llmy renders for
    /// a gateway timeout: `openai error: error code: 524`.
    #[test]
    fn gateway_timeout_text_is_matched() {
        for text in [
            "openai error: error code: 524",
            "error code 524",
            "504 gateway time-out",
            "502 bad gateway",
        ] {
            let lower = text.to_lowercase();
            assert!(
                TRANSIENT_ERROR_NEEDLES.iter().any(|n| lower.contains(n)),
                "{text} must classify as transient"
            );
        }
    }

    /// Model-not-found and billing are verdicts, never worth retrying.
    #[test]
    fn non_retryable_text_is_rejected() {
        for text in [
            "openai error: not_found_error: Model 'x' is not supported",
            "billing error: scope Some(\"mapper\") reached cap",
            "response filtered",
        ] {
            let lower = text.to_lowercase();
            assert!(
                !TRANSIENT_ERROR_NEEDLES.iter().any(|n| lower.contains(n)),
                "{text} must not be retried"
            );
        }
    }

    #[test]
    fn output_length_limit_is_not_retried() {
        // Must not be confused with a proxy timeout.
        assert!(!is_transient_transport_error(&LLMYError::OutputLength));
    }
}
