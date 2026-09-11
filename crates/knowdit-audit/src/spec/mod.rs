//! Specification Generator agent.
//!
//! Step 2 of the agentic auditing workflow described in
//! `paper/samples/sections/3-methodology.tex`. Given the Knowledge Mapper
//! output (extract semantic ↔ historical semantic ↔ historical finding
//! "links") stored in a project's [`RepoDatabase`], this agent decides for
//! each link whether the historical vulnerability pattern can be reproduced
//! on the current project, and if so emits one or more
//! [`AuditSpecification`]s describing the setup / pre-attack / post-attack
//! state invariants and the core call sequence to exercise.
//!
//! The agent is memory-equipped (long-term: the link details, persisted in
//! the system prompt; short-term: one entry per project contract preloaded
//! from the static-analysis tables). It builds each spec incrementally
//! through tool calls so failures are easier to attribute.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::num::NonZeroUsize;
use std::sync::Arc;

use color_eyre::eyre::{Result, WrapErr};
use itertools::Itertools;
use knowdit_kg_model::ExtractedSemantic;
use knowdit_repo_model::{
    HistoricalSemanticRecord, LinkCandidate, LinkResumeState, MatchStrength, RepoDatabase,
    SemanticMatchSet, repo::SpecificationRecord,
};
use llmy::agent::{LLMYError, StepResult};
use llmy::client::client::LLM;
use llmy::client::rust_decimal::Decimal;
use llmy::client::settings::LLMSettings;
use llmy::harness::memory::AgentMemorySystemPromptCriteria;
use llmy::harness::{Agent, AgentConfig};
use serde::Serialize;
use tokio::sync::Mutex;

use crate::types::AuditSpecification;

mod backend;
mod index;
mod prompt;
mod tools;
pub use backend::SpecBackend;
pub use index::ProjectIndex;
use prompt::{build_regen_prompt_extension, build_system_prompt, build_user_prompt};
pub(crate) use tools::{
    LookupCallGraphTool, LookupStateVariableXrefsTool, ReadContractSourceTool,
    ReadFunctionSourceTool,
};

// `LinkInput` / `LinkKey` are the language-agnostic per-link work units; they
// live in `knowdit-repo-model` so both the Solidity (here) and Move
// (`knowdit-move`) spec pipelines share them. Re-exported so existing
// `crate::spec::LinkInput` paths keep resolving.
pub use knowdit_repo_model::{LinkInput, LinkKey};

// ---------------------------------------------------------------------------
// Billing-cap exhaustion
// ---------------------------------------------------------------------------

/// Details of a billing-cap exhaustion ([`LLMYError::Billing`]) lifted out of
/// an eyre error chain. A billing exhaustion is **run-fatal**, not a per-link
/// failure: `llmy` fails fast on the pre-flight `check_cap` before every
/// request, so once the cap is hit *every* later LLM call fails the same way.
/// Orchestrators surface this (which cap, how much spent) and abort the whole
/// run instead of silently abandoning the rest of the queue one wasted link at
/// a time.
#[derive(Debug, Clone, Serialize)]
pub struct BillingExhausted {
    pub cap: Decimal,
    pub current: Decimal,
    /// Name of the scope whose cap was exceeded, if it had one.
    pub scope: Option<String>,
    /// Billing-tree node id whose cap was exceeded (0 = root).
    pub node: u64,
}

/// If `err`'s cause chain contains an [`LLMYError::Billing`], return its
/// details. Walks the full chain because [`PlannedLinkWork::run_agent`] / the graders
/// `wrap_err` the raw `LLMYError` into an eyre report before it reaches a caller.
pub fn billing_exhaustion(err: &color_eyre::eyre::Report) -> Option<BillingExhausted> {
    err.chain()
        .find_map(|cause| match cause.downcast_ref::<LLMYError>() {
            Some(LLMYError::Billing {
                cap,
                current,
                node,
                scope,
            }) => Some(BillingExhausted {
                cap: *cap,
                current: *current,
                scope: scope.clone(),
                node: *node,
            }),
            _ => None,
        })
}

/// Convenience predicate over [`billing_exhaustion`].
pub fn is_billing_exhausted(err: &color_eyre::eyre::Report) -> bool {
    billing_exhaustion(err).is_some()
}

// ---------------------------------------------------------------------------
// Public configuration / output types
// ---------------------------------------------------------------------------

/// Where a link's finding originated — controls how the gen-spec agent frames
/// it. A run is uniformly one source (set on [`SpecGenOptions`]), so this is
/// run-level config, not per-link.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LinkSource {
    /// Discovered by the Knowledge Mapper from the historical KG: the finding is
    /// a *topic hint* to explore for related issues.
    #[default]
    Mapper,
    /// Supplied externally (e.g. `workflow external-validate`): the finding is a
    /// concrete reported issue to reproduce / validate, not a hint.
    External,
}

/// Tunables for one Specification Generator pass.
#[derive(Debug, Clone)]
pub struct SpecGenOptions {
    /// Maximum number of agent steps allowed for a single link before the
    /// link is force-abandoned.
    pub max_agent_steps: usize,
    /// Soft cap on how many `AuditSpecification`s the agent may commit for
    /// one link. The agent is free to commit fewer.
    pub max_specs_per_link: usize,
    /// Compact the conversation when the rendered context exceeds this many
    /// approximate tokens. `None` defaults to 80% of the model's max input.
    pub compact_context_threshold_tokens: Option<usize>,
    /// `llmy` cache key prefix (per-project, set by the CLI).
    pub cache_key: String,
    /// Optional debug prefix passed to llmy.
    pub debug_prefix: Option<String>,
    /// Optional per-call llmy settings.
    pub llm_settings: Option<LLMSettings>,
    /// Optional cap on the total number of links processed (after de-dup).
    /// `None` means process every link.
    pub max_links: Option<usize>,
    /// Maximum number of fully materialized [`LinkInput`]s the planner keeps
    /// queued for processing at once. Bounds the heavy per-link payload
    /// (cloned prompt strings + KG rows) independently of the total candidate
    /// count. Must be `> 0`; `0` is rejected at the CLI boundary.
    pub batch_links: usize,
    /// Maximum number of per-link detail rows retained in
    /// [`SpecGenOutcome::by_link`]. Aggregate counters are always exact;
    /// detail rows beyond this are tracked in
    /// [`SpecGenOutcome::omitted_link_outcomes`]. `0` retains no detail rows.
    pub summary_rows: usize,
    /// At most this many findings per *(extract, historical)* pair. When
    /// multiple extracts match the same historical, each (extract, historical)
    /// pair gets its own quota — so a finding still useful for one extract
    /// is not crowded out by an unrelated extract that happens to also match
    /// the same historical. `None` means no cap.
    pub max_findings_per_historical: Option<usize>,
    /// Cap on the total number of links processed *per extract*. Combined
    /// with [`Self::max_findings_per_historical`] this lets a project's
    /// budget be spread fairly across all matched extracts rather than the
    /// first one monopolising the cost. `None` means no per-extract cap.
    pub max_links_per_extract: Option<usize>,
    /// Number of [`PlannedLinkWork::run_agent`] agent runs allowed in flight at once. `1` runs
    /// strictly serially (default); higher values dispatch links to a worker
    /// pool. The shared billing cap, prompt cache, and DB write are all
    /// already concurrency-safe.
    pub concurrency: usize,
    /// When `true`, clear the `specification` table at the start of the run
    /// and process every link from scratch. When `false` (default), skip any
    /// link whose `(semantic_id, finding_id)` already has rows in the spec
    /// table — so a re-run picks up exactly where the previous one stopped.
    pub regenerate: bool,
    /// Minimum mapper-emitted match strength to consider for spec
    /// generation. Links with `strength < min_strength` are dropped
    /// up-front (before any cap). Default `Medium` skips `Low` matches
    /// (treated as noise).
    pub min_strength: knowdit_repo_model::MatchStrength,
    /// Minimum semantic↔finding link strength (from the global linker)
    /// to consider. A LinkInput fans out per `(extract, historical,
    /// finding)`; this drops any finding whose `(historical, finding)`
    /// link is weaker than `min_link_strength`. Default `Medium`
    /// matches the noise floor used for `min_strength`.
    pub min_link_strength: knowdit_kg_model::link_strength::LinkStrength,
    /// Pre-rendered Markdown block describing the project's
    /// source language; verbatim-prepended to each per-link
    /// system prompt. See [`crate::profile::ProfileOptions::language_prompt_prefix`]
    /// for the same dispatch convention.
    pub language_prompt_prefix: String,
    /// Whether the links come from the Knowledge Mapper (topic hints) or are
    /// externally-supplied reported findings to validate. Frames the gen-spec
    /// agent's system prompt; see [`LinkSource`].
    pub link_source: LinkSource,
}

impl Default for SpecGenOptions {
    fn default() -> Self {
        Self {
            max_agent_steps: 60,
            max_specs_per_link: 4,
            compact_context_threshold_tokens: None,
            cache_key: "knowdit-spec".to_string(),
            debug_prefix: None,
            llm_settings: None,
            max_links: None,
            batch_links: 1_000,
            summary_rows: 10_000,
            max_findings_per_historical: None,
            max_links_per_extract: None,
            concurrency: 1,
            regenerate: false,
            min_strength: knowdit_repo_model::MatchStrength::Medium,
            min_link_strength: knowdit_kg_model::link_strength::LinkStrength::Medium,
            language_prompt_prefix: String::new(),
            link_source: LinkSource::Mapper,
        }
    }
}

impl SpecGenOptions {
    /// Effective heavy-batch bound: never smaller than the configured
    /// concurrency, so a scheduler can never hold more fully materialized links
    /// than the bound permits. The documented guarantee is
    /// `max(batch_links, concurrency)`.
    pub fn effective_batch_links(&self) -> usize {
        self.batch_links.max(1).max(self.concurrency.max(1))
    }
}

/// Outcome of one link's spec-generation run.
#[derive(Debug, Clone)]
pub struct LinkSpecOutcome {
    /// 1-based ordinal in the planned link stream — useful for progress
    /// reporting and for schedulers that need a stable cross-batch label.
    pub ordinal: usize,
    /// `project_semantic.id` from the project DB.
    pub extract_id: i32,
    /// Historical semantic id, mirrored into the project DB.
    pub historical_id: i32,
    /// Historical finding id, mirrored into the project DB.
    pub finding_id: i32,
    /// Whether the agent committed at least one spec.
    pub status: LinkSpecStatus,
    /// Specs the agent committed before finalizing.
    pub specifications: Vec<AuditSpecification>,
    /// DB ids of `specifications` after the commit transaction succeeds.
    /// Populated by [`LinkSpecOutcome::commit`]; empty on commit failure or when
    /// `specifications` was empty.
    pub specification_ids: Vec<i32>,
    /// Reason the agent reported for abandoning, when `status == Abandoned`.
    pub abort_reason: Option<String>,
    /// Free-form summary from the agent's final tool call.
    pub final_summary: Option<String>,
    /// Number of agent steps consumed (counts initial step too).
    pub steps: usize,
    /// Number of context-compaction passes triggered.
    pub compact_count: usize,
}

impl LinkSpecOutcome {
    /// Persist `self.specifications` to the project DB's `specification` table
    /// and record the assigned row ids in `self.specification_ids`. A no-op
    /// (leaving `specification_ids` empty) when there are no specs, or on a
    /// serialize / append failure (logged, not fatal — a commit failure for one
    /// link shouldn't abort the run).
    pub async fn commit(&mut self, repo: &RepoDatabase) {
        if self.specifications.is_empty() {
            return;
        }
        let mut payload = Vec::with_capacity(self.specifications.len());
        for spec in &self.specifications {
            match serde_json::to_string(spec) {
                Ok(json) => payload.push(SpecificationRecord {
                    semantic_id: self.extract_id,
                    historical_id: self.historical_id,
                    finding_id: self.finding_id,
                    specification_json: json,
                }),
                Err(err) => {
                    tracing::error!(
                        "failed to JSON-serialize spec for extract={} finding={}: {err:#}",
                        self.extract_id,
                        self.finding_id
                    );
                    return;
                }
            }
        }
        match repo.append_specifications(&payload).await {
            Ok(ids) => self.specification_ids = ids,
            Err(err) => {
                tracing::error!(
                    "failed to append {} spec(s) for extract={} finding={}: {err:#}",
                    payload.len(),
                    self.extract_id,
                    self.finding_id,
                );
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkSpecStatus {
    /// The agent committed at least one specification and signaled success.
    Built,
    /// The agent decided the historical pattern doesn't apply to this
    /// project, or hit step/runtime limits before finalizing.
    Abandoned,
}

/// Aggregate run outcome.
#[derive(Debug, Clone, Default)]
pub struct SpecGenOutcome {
    pub link_count: usize,
    pub built_link_count: usize,
    pub abandoned_link_count: usize,
    pub total_specs: usize,
    /// Retained per-link detail rows (bounded by
    /// [`SpecGenOptions::summary_rows`]). Counters above stay exact.
    pub by_link: Vec<LinkSpecOutcome>,
    /// How many per-link detail rows were dropped because
    /// [`SpecGenOptions::summary_rows`] was reached. Their contributions are
    /// still reflected in `link_count` / `built_link_count` /
    /// `abandoned_link_count` / `total_specs`.
    pub omitted_link_outcomes: usize,
}

/// A serial, resume-safe spec-generation queue prepared from the project's
/// current mapper output. Callers can pull one link at a time and interleave
/// downstream phases (codegen / reflect / regen) between links.
///
/// The full plan (filtering, resume resolution, quota enforcement, fairness
/// ordering) is computed **once** by [`PreparedLinkPlan::prepare`]; heavy
/// [`LinkInput`] payloads are materialized lazily in batches of
/// [`SpecGenOptions::batch_links`] as the queue drains. Peak heavy memory is
/// therefore bounded by the batch size (plus active scheduler work), not by
/// the total candidate count.
pub struct SpecGenStream {
    plan: PreparedLinkPlan,
    batch: VecDeque<LinkInput>,
    batch_links: usize,
    total_links: usize,
    processed_links: usize,
    options: SpecGenOptions,
}

impl SpecGenStream {
    pub fn total_links(&self) -> usize {
        self.total_links
    }

    pub fn processed_links(&self) -> usize {
        self.processed_links
    }

    pub fn remaining_links(&self) -> usize {
        self.batch.len() + self.plan.remaining()
    }

    pub fn matched_extract_count(&self) -> usize {
        self.plan.matched_extract_count()
    }

    pub fn historical_finding_total(&self) -> usize {
        self.plan.historical_finding_total()
    }

    pub fn next_link_key(&self) -> Option<LinkKey> {
        self.batch
            .front()
            .map(LinkInput::key)
            .or_else(|| self.plan.peek_key())
    }

    pub fn is_empty(&self) -> bool {
        self.batch.is_empty() && self.plan.remaining() == 0
    }

    /// Claim one link from the stream **without** running it. Lets an
    /// external scheduler keep a bounded number of full link pipelines
    /// active at once: claim → gen-spec → fuzz → reflect → regen →
    /// snapshot. The returned [`PlannedLinkWork`] carries everything
    /// [`PlannedLinkWork::process`] needs, so the scheduler can run it on its own
    /// task. The ordinal is **invocation-global** (it does not reset per
    /// materialized batch), so it stays usable as a cache/billing scope label.
    /// Note: `processed_links` is bumped on claim, so the scheduler must
    /// guarantee the claimed work is either run or dropped — there is no
    /// `unpop`.
    pub fn pop_next_work(&mut self) -> Option<PlannedLinkWork> {
        if self.batch.is_empty() {
            let batch_links = NonZeroUsize::new(self.batch_links).expect("batch_links > 0");
            self.batch = self.plan.take_batch(batch_links).into();
        }
        let link = self.batch.pop_front()?;
        let ordinal = self.processed_links + 1;
        self.processed_links += 1;
        Some(PlannedLinkWork {
            ordinal,
            total_links: self.total_links,
            link,
            project_index: self.plan.project_index(),
            options: self.options.clone(),
        })
    }
}

/// One claimed link ready to be run by an external scheduler. Carries
/// everything [`PlannedLinkWork::process`] needs (the link itself plus a cheap
/// `Arc<ProjectIndex>` clone), so the scheduler can spawn `.run(repo, llm)`
/// on its own task and interleave the link's fuzz/reflect/regen lifecycle
/// without holding the [`SpecGenStream`] borrow.
#[derive(Clone)]
pub struct PlannedLinkWork {
    ordinal: usize,
    total_links: usize,
    link: LinkInput,
    project_index: Arc<ProjectIndex>,
    options: SpecGenOptions,
}

impl PlannedLinkWork {
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn total_links(&self) -> usize {
        self.total_links
    }

    pub fn link_key(&self) -> LinkKey {
        self.link.key()
    }

    /// Run the claimed link end-to-end and commit any built specs to the
    /// project DB. Returns the populated `LinkSpecOutcome` with
    /// `specification_ids` filled in on success.
    ///
    /// When the link arrives with `pre_committed_spec_ids` populated
    /// (DB resume: this `(extract, finding)` already has spec rows but
    /// no code_gen yet), the gen-spec agent is skipped entirely and
    /// the outcome is synthesized as `Built` with those ids — the
    /// caller's inner fuzz / reflect / regen cycle then picks up where
    /// the prior run left off.
    ///
    /// Returns `Err` only on a billing-cap exhaustion (run-fatal — see
    /// [`PlannedLinkWork::process`]); a per-link agent failure is folded into an
    /// `Abandoned` outcome so the scheduler keeps going.
    pub async fn run(self, repo: &RepoDatabase, llm: &LLM) -> Result<LinkSpecOutcome> {
        if !self.link.pre_committed_spec_ids.is_empty() {
            tracing::info!(
                "Spec generator skipping gen-spec for resumed link={} (total = {}) {} — {} pre-committed spec(s)",
                self.ordinal,
                self.total_links,
                self.link,
                self.link.pre_committed_spec_ids.len(),
            );
            return Ok(LinkSpecOutcome {
                ordinal: self.ordinal,
                extract_id: self.link.extract_id,
                historical_id: self.link.historical_id,
                finding_id: self.link.finding_id,
                status: LinkSpecStatus::Built,
                specifications: Vec::new(),
                specification_ids: self.link.pre_committed_spec_ids.clone(),
                abort_reason: None,
                final_summary: Some(
                    "resumed from existing spec rows; gen-spec agent skipped".to_string(),
                ),
                steps: 0,
                compact_count: 0,
            });
        }
        let mut outcome = self.process(llm).await?;
        outcome.commit(repo).await;
        Ok(outcome)
    }
}

// ---------------------------------------------------------------------------
// Top-level driver
// ---------------------------------------------------------------------------

/// Specification Generator agent.
#[derive(Debug, Clone, Default)]
pub struct SpecificationGenerator;

impl SpecificationGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Run one spec-generation pass over every (extract, historical,
    /// finding) link present in `repo`. Specs are persisted to the
    /// `specification` table via [`RepoDatabase::append_specifications`].
    ///
    /// The plan is built once, then processed in batches of
    /// [`SpecGenOptions::batch_links`] so peak memory is bounded by the batch
    /// size rather than the candidate count.
    pub async fn run(
        &self,
        repo: &RepoDatabase,
        llm: &LLM,
        options: &SpecGenOptions,
    ) -> Result<SpecGenOutcome> {
        let mut plan = match PreparedLinkPlan::prepare(repo, options).await? {
            Some(plan) => plan,
            None => return Ok(SpecGenOutcome::default()),
        };
        let total_links = plan.planned_total();
        tracing::info!(
            "Specification Generator: {} link(s) to process across {} matched extract(s) and {} historical finding(s)",
            total_links,
            plan.matched_extract_count(),
            plan.historical_finding_total(),
        );

        let batch_links = NonZeroUsize::new(options.effective_batch_links()).expect("batch > 0");
        let mut outcome = SpecGenOutcome::default();
        let mut next_ordinal = 1usize;
        let mut billing_err: Option<color_eyre::eyre::Report> = None;
        loop {
            let batch = plan.take_batch(batch_links);
            if batch.is_empty() {
                break;
            }
            let (batch_outcomes, err) = Self::run_batch(
                repo,
                llm,
                options,
                plan.project_index(),
                batch,
                &mut next_ordinal,
                total_links,
            )
            .await;
            fold_outcome(&mut outcome, batch_outcomes, options.summary_rows);
            if let Some(err) = err {
                billing_err = Some(err);
                break;
            }
        }
        if let Some(err) = billing_err {
            return Err(err);
        }
        // Persistence already happened incrementally via `LinkSpecOutcome::commit`.
        // The legacy end-of-run `write_specifications` is intentionally
        // skipped — it would `delete_many()` the rows we just wrote.
        tracing::info!(
            "Specification Generator wrote {} spec(s) covering {} link(s); {} link(s) abandoned",
            outcome.total_specs,
            outcome.built_link_count,
            outcome.abandoned_link_count,
        );
        Ok(outcome)
    }

    /// Process one materialized batch, committing each outcome to the DB as it
    /// completes. Returns the batch's outcomes (in batch order) plus, on a
    /// run-fatal billing exhaustion, the error to propagate. Outcomes already
    /// committed before the abort are still returned so counters stay exact.
    async fn run_batch(
        repo: &RepoDatabase,
        llm: &LLM,
        options: &SpecGenOptions,
        project_index: Arc<ProjectIndex>,
        links: Vec<LinkInput>,
        next_ordinal: &mut usize,
        total_links: usize,
    ) -> (Vec<LinkSpecOutcome>, Option<color_eyre::eyre::Report>) {
        let concurrency = options.concurrency.max(1);
        let base = *next_ordinal;
        let batch_len = links.len();

        if concurrency == 1 {
            let mut outcomes = Vec::with_capacity(batch_len);
            let mut err = None;
            let mut claimed = 0usize;
            for (idx, link) in links.into_iter().enumerate() {
                claimed = idx + 1;
                let work = PlannedLinkWork {
                    ordinal: base + idx,
                    total_links,
                    link,
                    project_index: project_index.clone(),
                    options: options.clone(),
                };
                match work.process(llm).await {
                    Ok(mut outcome) => {
                        outcome.commit(repo).await;
                        outcomes.push(outcome);
                    }
                    Err(e) => {
                        err = Some(e);
                        break;
                    }
                }
            }
            *next_ordinal = base + claimed;
            return (outcomes, err);
        }

        tracing::info!("Specification Generator running with concurrency={concurrency}");
        // Owned indexed inputs so workers can move them.
        let indexed: Vec<(usize, LinkInput)> = links.into_iter().enumerate().collect();
        let queue = Arc::new(Mutex::new(indexed.into_iter().rev().collect::<Vec<_>>()));
        let (tx, mut rx) =
            tokio::sync::mpsc::channel::<(usize, Result<LinkSpecOutcome>)>(concurrency * 2);

        let mut workers = tokio::task::JoinSet::new();
        for _ in 0..concurrency {
            let queue = queue.clone();
            let project_index = project_index.clone();
            let llm = llm.clone();
            let options = options.clone();
            let tx = tx.clone();
            workers.spawn(async move {
                loop {
                    let next = {
                        let mut q = queue.lock().await;
                        q.pop()
                    };
                    let Some((idx, link)) = next else {
                        break;
                    };
                    let work = PlannedLinkWork {
                        ordinal: base + idx,
                        total_links,
                        link,
                        project_index: project_index.clone(),
                        options: options.clone(),
                    };
                    let outcome = work.process(&llm).await;
                    // A billing error is run-fatal: stop this worker so it
                    // does not churn the rest of the batch against a dead cap.
                    let fatal = outcome.is_err();
                    if tx.send((idx, outcome)).await.is_err() || fatal {
                        break;
                    }
                }
            });
        }
        drop(tx);

        let mut collected: Vec<Option<LinkSpecOutcome>> = (0..batch_len).map(|_| None).collect();
        let mut billing_err: Option<color_eyre::eyre::Report> = None;
        while let Some((idx, outcome)) = rx.recv().await {
            // The only `Err` `process` yields is a billing-cap exhaustion,
            // which is run-fatal: stop collecting and propagate so the run
            // aborts instead of churning the rest of the batch against a
            // dead cap.
            let mut outcome = match outcome {
                Ok(outcome) => outcome,
                Err(err) => {
                    billing_err = Some(err);
                    break;
                }
            };
            // Commit on the main task so all writes go through one DB
            // handle in serialized order; SQLite handles small txns well.
            outcome.commit(repo).await;
            if idx < collected.len() {
                collected[idx] = Some(outcome);
            }
        }
        if billing_err.is_some() {
            workers.abort_all();
        }
        while workers.join_next().await.is_some() {}

        *next_ordinal = base + batch_len;
        (collected.into_iter().flatten().collect(), billing_err)
    }

    /// Prepare a serial, round-robin link stream. Each `pop_next_work()` call
    /// consumes one `(extract, historical, finding)` link and appends any
    /// committed specs to the DB immediately, which lets higher-level
    /// orchestrators interleave codegen / reflect / regen between links.
    pub async fn prepare_stream(
        &self,
        repo: &RepoDatabase,
        options: &SpecGenOptions,
    ) -> Result<Option<SpecGenStream>> {
        let plan = match PreparedLinkPlan::prepare(repo, options).await? {
            Some(plan) => plan,
            None => return Ok(None),
        };
        let total_links = plan.planned_total();
        Ok(Some(SpecGenStream {
            plan,
            batch: VecDeque::new(),
            batch_links: options.effective_batch_links(),
            total_links,
            processed_links: 0,
            options: options.clone(),
        }))
    }

    /// Regenerate the spec for one `(extract_id, historical_id, finding_id)`
    /// link with a prior reflection's reason fed back into the agent's system
    /// prompt. Returns the new spec **in memory** (not persisted) so the caller
    /// can flow it into `repo.write_full_spec_regen` together with the
    /// freshly-regenerated codegen — one atomic transaction, no orphan rows
    /// on partial failure.
    ///
    /// Materializes the link by exact `(E, H, F)` identity, independent of the
    /// generation strength thresholds: regen must resolve a link even when it
    /// would have been filtered out of a fresh planning pass.
    pub async fn regen_one_link(
        repo: &RepoDatabase,
        llm: &LLM,
        options: &SpecGenOptions,
        request: SpecRegenRequest,
    ) -> Result<SpecRegenInMemory> {
        let runtime = SpecRuntime::load(repo)
            .await?
            .ok_or_else(|| color_eyre::eyre::eyre!("project DB has no extracts/matches"))?;
        let link = runtime.resolve_link(
            request.extract_id,
            request.historical_id,
            request.finding_id,
        )?;
        let prompt_extension = build_regen_prompt_extension(&request.mode, &request.prior_feedback);
        // Wrap the single regen link in a `PlannedLinkWork` so it goes through
        // the same agent loop as the streaming path. `total_links = 1` and the
        // ordinal is the cache-key serial (a regen, not a stream position).
        let work = PlannedLinkWork {
            ordinal: request.serial_for_cache_key,
            total_links: 1,
            link,
            project_index: runtime.project_index.clone(),
            options: options.clone(),
        };
        let outcome = work.run_agent(llm, Some(&prompt_extension)).await?;
        Ok(SpecRegenInMemory {
            extract_id: request.extract_id,
            finding_id: request.finding_id,
            specifications: outcome.specifications,
            status: outcome.status,
            abort_reason: outcome.abort_reason,
            steps: outcome.steps,
        })
    }
}

/// Read-only per-invocation source state shared by the planner and
/// [`SpecificationGenerator::regen_one_link`]. Holds the project index plus
/// **source indexes** (extracts / historicals / match strengths) — never a
/// full expanded `Vec<LinkInput>`.
pub(crate) struct SpecRuntime {
    pub(crate) project_index: Arc<ProjectIndex>,
    pub(crate) extracted_by_id: BTreeMap<i32, ExtractedSemantic>,
    pub(crate) historical_by_id: BTreeMap<i32, HistoricalSemanticRecord>,
    /// Mapper-emitted strength per `(extract, historical)` pair, recovered
    /// exactly for regen / demand-driven materialization.
    pub(crate) match_strength_by_pair: BTreeMap<(i32, i32), MatchStrength>,
    /// `(historical_id, finding_id) -> position in that record's findings`,
    /// so exact materialization is O(log n) instead of a linear scan.
    findings_index: BTreeMap<(i32, i32), usize>,
}

impl SpecRuntime {
    /// Load the read-only source state. Returns `None` only when the project DB
    /// is empty enough that no spec generation is possible (no extracts or no
    /// mapper matches) — caller should treat as a no-op.
    async fn load(repo: &RepoDatabase) -> Result<Option<Self>> {
        let extracted = repo
            .load_project_semantics()
            .await
            .wrap_err("failed to load extracted project semantics")?;
        if extracted.is_empty() {
            tracing::warn!("Specification Generator invoked with no extracted semantics");
            return Ok(None);
        }
        let match_set: SemanticMatchSet = repo
            .load_semantic_match_results()
            .await
            .wrap_err("failed to load Knowledge Mapper output")?;
        if match_set.matches.is_empty() {
            tracing::warn!(
                "Specification Generator invoked but the project has no semantic_matched rows; run map-semantics first"
            );
            return Ok(None);
        }
        let call_graph = repo
            .load_call_graph()
            .await
            .wrap_err("failed to load project call graph")?;
        let storage_graph = repo
            .load_storage_graph()
            .await
            .wrap_err("failed to load project storage graph")?;
        let inheritance_graph = repo
            .load_inheritance_graph()
            .await
            .wrap_err("failed to load project inheritance graph")?;
        let project_index = Arc::new(ProjectIndex::build(
            call_graph,
            storage_graph,
            inheritance_graph,
        ));

        let extracted_by_id: BTreeMap<i32, ExtractedSemantic> = extracted
            .into_iter()
            .enumerate()
            .map(|(i, sem)| ((i as i32) + 1, sem))
            .collect();
        // Move historicals into the index instead of cloning them; the source
        // `Vec` is dropped here.
        let SemanticMatchSet { historicals, matches } = match_set;
        let historical_by_id: BTreeMap<i32, HistoricalSemanticRecord> = historicals
            .into_iter()
            .map(|record| (record.semantic.id, record))
            .collect();
        let match_strength_by_pair: BTreeMap<(i32, i32), MatchStrength> = matches
            .iter()
            .map(|m| ((m.extract_id, m.historical_id), m.strength))
            .collect();
        // Index `(historical_id, finding_id) -> position in record.findings` so
        // materializing a candidate is O(log n) rather than a linear scan of a
        // historical's findings. Compact metadata; scales with matches, not
        // with cloned heavy payloads.
        let mut runtime = Self {
            project_index,
            extracted_by_id,
            historical_by_id,
            match_strength_by_pair,
            findings_index: BTreeMap::new(),
        };
        runtime.rebuild_findings_index();
        Ok(Some(runtime))
    }

    /// Rebuild the `(historical_id, finding_id) -> findings position` index
    /// from the current `historical_by_id`.
    fn rebuild_findings_index(&mut self) {
        self.findings_index.clear();
        for (historical_id, record) in &self.historical_by_id {
            for (idx, linked) in record.findings.iter().enumerate() {
                self.findings_index
                    .insert((*historical_id, linked.finding.id), idx);
            }
        }
    }

    /// Exact `(historical, finding)` link from the source index.
    fn linked_finding(
        &self,
        historical_id: i32,
        finding_id: i32,
    ) -> Option<&knowdit_repo_model::HistoricalLinkedFinding> {
        let idx = *self.findings_index.get(&(historical_id, finding_id))?;
        self.historical_by_id
            .get(&historical_id)?
            .findings
            .get(idx)
    }

    /// Materialize one compact candidate into a full [`LinkInput`]. Returns
    /// `None` when the source rows for the candidate's identity are missing
    /// (should not happen for a freshly planned candidate).
    fn materialize(&self, candidate: &LinkCandidate) -> Option<LinkInput> {
        let extract = self.extracted_by_id.get(&candidate.key.extract_id)?;
        let record = self.historical_by_id.get(&candidate.key.historical_id)?;
        let linked = self.linked_finding(candidate.key.historical_id, candidate.key.finding_id)?;
        let mut link = LinkInput::materialize(
            candidate.key.extract_id,
            candidate.key.historical_id,
            candidate.key.finding_id,
            candidate.match_strength,
            linked.strength,
            extract,
            record,
            linked,
        );
        link.pre_committed_spec_ids = candidate.pre_committed_spec_ids.clone();
        Some(link)
    }

    /// Resolve a single link by exact `(E, H, F)` identity, **independent of
    /// the generation strength thresholds**. Used by regen: a link that would
    /// be filtered out of a fresh planning pass (e.g. `Low`/`Low`) must still
    /// resolve so its spec can be regenerated.
    pub(crate) fn resolve_link(
        &self,
        extract_id: i32,
        historical_id: i32,
        finding_id: i32,
    ) -> Result<LinkInput> {
        let extract = self.extracted_by_id.get(&extract_id).ok_or_else(|| {
            color_eyre::eyre::eyre!(
                "no project_semantic row for extract={extract_id} — did the KG change since the spec was first synthesized?"
            )
        })?;
        let record = self.historical_by_id.get(&historical_id).ok_or_else(|| {
            color_eyre::eyre::eyre!(
                "no historical_semantic row for historical={historical_id} — did the KG change since the spec was first synthesized?"
            )
        })?;
        let strength = self
            .match_strength_by_pair
            .get(&(extract_id, historical_id))
            .copied()
            .ok_or_else(|| {
                color_eyre::eyre::eyre!(
                    "no mapper match for (extract={extract_id}, historical={historical_id}) — did the KG change since the spec was first synthesized?"
                )
            })?;
        let linked = self
            .linked_finding(historical_id, finding_id)
            .ok_or_else(|| {
                color_eyre::eyre::eyre!(
                    "no LinkInput for (extract={extract_id}, historical={historical_id}, finding={finding_id}) — did the KG change since the spec was first synthesized?"
                )
            })?;
        Ok(LinkInput::materialize(
            extract_id,
            historical_id,
            finding_id,
            strength,
            linked.strength,
            extract,
            record,
            linked,
        ))
    }
}

/// Stateful one-shot link plan. `prepare` performs strength filtering, compact
/// candidate expansion, resume resolution, quota enforcement, fairness
/// ordering, and total truncation **once**; [`Self::take_batch`] then only
/// materializes the next heavy batch and advances.
///
/// A candidate is removed from `pending` before it can be materialized, so an
/// abandoned / no-spec link is attempted at most once per invocation and no
/// `attempted` set is needed.
pub struct PreparedLinkPlan {
    runtime: SpecRuntime,
    pending: VecDeque<LinkCandidate>,
    planned_total: usize,
    matched_extract_count: usize,
    historical_finding_total: usize,
}

impl PreparedLinkPlan {
    /// Build the plan for one invocation. Returns `None` when there is nothing
    /// to process (empty project, no matches, or every candidate filtered /
    /// already built).
    pub async fn prepare(
        repo: &RepoDatabase,
        options: &SpecGenOptions,
    ) -> Result<Option<Self>> {
        let runtime = match SpecRuntime::load(repo).await? {
            Some(rt) => rt,
            None => return Ok(None),
        };

        // 1. Expand compact candidates, deduplicating exact (E, H, F).
        let mut candidates: Vec<LinkCandidate> = Vec::new();
        let mut seen: HashSet<LinkKey> = HashSet::new();
        for (pair, match_strength) in &runtime.match_strength_by_pair {
            let (extract_id, historical_id) = *pair;
            let Some(record) = runtime.historical_by_id.get(&historical_id) else {
                continue;
            };
            if !runtime.extracted_by_id.contains_key(&extract_id) {
                continue;
            }
            for linked in &record.findings {
                let key = LinkKey {
                    extract_id,
                    historical_id,
                    finding_id: linked.finding.id,
                };
                if !seen.insert(key) {
                    continue;
                }
                candidates.push(LinkCandidate {
                    key,
                    match_strength: *match_strength,
                    link_strength: linked.strength,
                    pre_committed_spec_ids: Vec::new(),
                });
            }
        }
        let raw_link_count = candidates.len();

        // 2. Strength thresholds (before any cap, so filtered links never
        //    consume a quota slot).
        let min_rank = options.min_strength.rank();
        let pre_strength = candidates.len();
        candidates.retain(|c| c.match_strength.rank() >= min_rank);
        if candidates.len() != pre_strength {
            tracing::info!(
                "Specification Generator strength filter (min={}): {} → {} link(s)",
                options.min_strength,
                pre_strength,
                candidates.len(),
            );
        }
        let min_link_rank = options.min_link_strength.rank();
        let pre_link_strength = candidates.len();
        candidates.retain(|c| c.link_strength.rank() >= min_link_rank);
        if candidates.len() != pre_link_strength {
            tracing::info!(
                "Specification Generator link-strength filter (min={}): {} → {} link(s)",
                options.min_link_strength.as_str(),
                pre_link_strength,
                candidates.len(),
            );
        }

        // 3. Resume resolution in bounded chunks. `--regenerate` clears the
        //    specification table exactly once here (never in `take_batch`), and
        //    skips resume loading entirely.
        if options.regenerate {
            repo.reset_for_regenerate()
                .await
                .wrap_err("failed to reset downstream pipeline before regenerate")?;
            tracing::info!(
                "Specification Generator: --regenerate set, reset specs + downstream pipeline"
            );
        } else {
            const RESUME_CHUNK: usize = 2_048;
            let before = candidates.len();
            let mut kept: Vec<LinkCandidate> = Vec::with_capacity(before);
            let mut dropped_built = 0usize;
            let mut resumed_partial = 0usize;
            for chunk in candidates.chunks(RESUME_CHUNK) {
                let keys: Vec<LinkKey> = chunk.iter().map(|c| c.key).collect();
                let states = repo
                    .load_link_resume_states(&keys)
                    .await
                    .wrap_err("failed to resolve link resume states")?;
                for candidate in chunk {
                    match states.get(&candidate.key) {
                        Some(LinkResumeState::Built) => dropped_built += 1,
                        Some(LinkResumeState::Partial { spec_ids }) => {
                            resumed_partial += 1;
                            let mut candidate = candidate.clone();
                            candidate.pre_committed_spec_ids = spec_ids.clone();
                            kept.push(candidate);
                        }
                        _ => kept.push(candidate.clone()),
                    }
                }
            }
            candidates = kept;
            if dropped_built > 0 || resumed_partial > 0 {
                tracing::info!(
                    "Specification Generator resume: {} link(s) dropped (Built), {} link(s) resuming at inner cycle (Partial), {} link(s) remaining; {} → {} link(s) total",
                    dropped_built,
                    resumed_partial,
                    candidates.len(),
                    before,
                    candidates.len(),
                );
            }
        }

        // 4-7. Strict strength buckets, fairness, quotas, total cap.
        let pending = select_pending(candidates, options, raw_link_count);

        if pending.is_empty() {
            tracing::warn!("Specification Generator: no links to process after expansion");
            return Ok(None);
        }

        let planned_total = pending.len();
        let historical_finding_total = runtime
            .historical_by_id
            .values()
            .map(|r| r.findings.len())
            .sum::<usize>();
        let matched_extract_count = runtime
            .match_strength_by_pair
            .keys()
            .map(|(extract_id, _)| *extract_id)
            .unique()
            .count();

        Ok(Some(Self {
            runtime,
            pending,
            planned_total,
            matched_extract_count,
            historical_finding_total,
        }))
    }

    /// Materialize up to `batch_links` pending candidates, preserving the
    /// planned order. Candidates whose source rows vanished are logged and
    /// skipped.
    pub fn take_batch(&mut self, batch_links: NonZeroUsize) -> Vec<LinkInput> {
        let mut out = Vec::with_capacity(batch_links.get());
        while out.len() < batch_links.get() {
            let Some(candidate) = self.pending.pop_front() else {
                break;
            };
            match self.runtime.materialize(&candidate) {
                Some(link) => out.push(link),
                None => tracing::warn!(
                    "Specification Generator: dropping candidate (extract={}, historical={}, finding={}) — source rows missing",
                    candidate.key.extract_id,
                    candidate.key.historical_id,
                    candidate.key.finding_id,
                ),
            }
        }
        out
    }

    /// Number of candidates still pending materialization.
    pub fn remaining(&self) -> usize {
        self.pending.len()
    }

    /// Total number of links the plan will emit (fixed at preparation time).
    pub fn planned_total(&self) -> usize {
        self.planned_total
    }

    pub fn matched_extract_count(&self) -> usize {
        self.matched_extract_count
    }

    pub fn historical_finding_total(&self) -> usize {
        self.historical_finding_total
    }

    /// Peek at the next pending candidate's key without materializing it.
    pub fn peek_key(&self) -> Option<LinkKey> {
        self.pending.front().map(|c| c.key)
    }

    pub fn project_index(&self) -> Arc<ProjectIndex> {
        self.runtime.project_index.clone()
    }

    /// Test-only constructor: build a plan directly from a prepared
    /// [`SpecRuntime`] and an explicit candidate sequence, bypassing DB
    /// planning. Used by the batch-bound / memory stress tests.
    #[cfg(test)]
    pub(crate) fn from_candidates_for_test(
        runtime: SpecRuntime,
        candidates: Vec<LinkCandidate>,
    ) -> Self {
        let planned_total = candidates.len();
        let matched_extract_count = runtime
            .match_strength_by_pair
            .keys()
            .map(|(extract_id, _)| *extract_id)
            .unique()
            .count();
        let historical_finding_total = runtime
            .historical_by_id
            .values()
            .map(|r| r.findings.len())
            .sum();
        Self {
            runtime,
            pending: candidates.into(),
            planned_total,
            matched_extract_count,
            historical_finding_total,
        }
    }
}

/// Consume `candidates` from strongest to weakest `(match, link)` bucket,
/// applying per-`(extract, historical)` and per-extract quotas with counters
/// shared across every bucket, interleaving extracts round-robin **within** a
/// bucket for fairness, and stopping after `max_links`.
fn select_pending(
    candidates: Vec<LinkCandidate>,
    options: &SpecGenOptions,
    raw_link_count: usize,
) -> VecDeque<LinkCandidate> {
    // Bucket by (match_rank, link_rank); BTreeMap keeps ascending key order, so
    // `.rev()` walks strongest match first, then strongest link.
    let mut buckets: BTreeMap<(u8, u8), BTreeMap<i32, Vec<LinkCandidate>>> = BTreeMap::new();
    for candidate in candidates {
        buckets
            .entry((candidate.match_strength.rank(), candidate.link_strength.rank()))
            .or_default()
            .entry(candidate.key.extract_id)
            .or_default()
            .push(candidate);
    }

    let per_eh_cap = options.max_findings_per_historical.filter(|n| *n > 0);
    let per_e_cap = options.max_links_per_extract.filter(|n| *n > 0);
    let max_links = options.max_links.filter(|n| *n > 0);

    let mut selected: VecDeque<LinkCandidate> = VecDeque::new();
    let mut per_eh: BTreeMap<(i32, i32), usize> = BTreeMap::new();
    let mut per_e: BTreeMap<i32, usize> = BTreeMap::new();

    'buckets: for by_extract in buckets.into_values().rev() {
        // Deterministic order inside one extract: (historical_id, finding_id).
        let mut queues: BTreeMap<i32, VecDeque<LinkCandidate>> = BTreeMap::new();
        for (extract_id, mut queue) in by_extract {
            queue.sort_by_key(|c| (c.key.historical_id, c.key.finding_id));
            queues.insert(extract_id, queue.into());
        }

        // Round-robin across extracts by ascending id. A candidate over quota
        // is dropped (quotas only grow, so it can never become admissible) and
        // does not consume a slot.
        loop {
            let extract_ids: Vec<i32> = queues.keys().copied().collect();
            if extract_ids.is_empty() {
                break;
            }
            for extract_id in extract_ids {
                let Some(queue) = queues.get_mut(&extract_id) else {
                    continue;
                };
                while let Some(candidate) = queue.pop_front() {
                    if let Some(cap) = per_eh_cap {
                        let count = per_eh
                            .get(&(candidate.key.extract_id, candidate.key.historical_id))
                            .copied()
                            .unwrap_or(0);
                        if count >= cap {
                            continue;
                        }
                    }
                    if let Some(cap) = per_e_cap {
                        let count = per_e
                            .get(&candidate.key.extract_id)
                            .copied()
                            .unwrap_or(0);
                        if count >= cap {
                            continue;
                        }
                    }
                    *per_eh
                        .entry((candidate.key.extract_id, candidate.key.historical_id))
                        .or_insert(0) += 1;
                    *per_e.entry(candidate.key.extract_id).or_insert(0) += 1;
                    selected.push_back(candidate);
                    if let Some(cap) = max_links
                        && selected.len() >= cap
                    {
                        break 'buckets;
                    }
                    break;
                }
                if queue.is_empty() {
                    queues.remove(&extract_id);
                }
            }
        }
    }

    if let Some(per_eh_cap) = per_eh_cap {
        tracing::info!(
            "Specification Generator capping at {} finding(s) per (extract, historical) pair: {} → {} link(s)",
            per_eh_cap,
            raw_link_count,
            selected.len()
        );
    }
    if let Some(per_e_cap) = per_e_cap {
        tracing::info!(
            "Specification Generator capping at {} link(s) per extract: → {} link(s)",
            per_e_cap,
            selected.len()
        );
    }
    if let Some(cap) = max_links
        && selected.len() >= cap
    {
        tracing::info!(
            "Specification Generator truncated to {} link(s) for this run",
            cap
        );
    }

    selected
}

/// One spec-regen request from `agentic regen`. Keeps the four
/// regen inputs in one struct so the per-call signature stays terse
/// even as the spec-regen knobs grow.
#[derive(Debug, Clone)]
pub struct SpecRegenRequest {
    pub extract_id: i32,
    /// `historical_semantic.id` of the prior spec — pinned so
    /// `regen_one_link` matches the **same** LinkInput the original
    /// spec was authored from. Without this, sibling LinkInputs
    /// sharing `(E, F)` but with different H could silently swap in
    /// during regen.
    pub historical_id: i32,
    pub finding_id: i32,
    pub mode: SpecRegenMode,
    pub prior_feedback: String,
    /// Used by the cache key to namespace LLM calls per regen
    /// attempt. Pass the triggering `reflection.id` so a re-run hits
    /// cache.
    pub serial_for_cache_key: usize,
}

/// Patch vs from-scratch (per `plan_reflection.md` §3.2). Patch mode
/// shows the agent the prior spec; from-scratch tells it to discard
/// history.
#[derive(Debug, Clone)]
pub enum SpecRegenMode {
    Patch(AuditSpecification),
    FromScratch,
}

/// Output of [`SpecificationGenerator::regen_one_link`] — the new specs
/// the agent committed, not yet persisted. The caller composes this with
/// a codegen regen + lineage rows in one atomic txn.
#[derive(Debug, Clone)]
pub struct SpecRegenInMemory {
    pub extract_id: i32,
    pub finding_id: i32,
    pub specifications: Vec<AuditSpecification>,
    pub status: LinkSpecStatus,
    pub abort_reason: Option<String>,
    pub steps: usize,
}

/// Fold one batch's outcomes into the running aggregate. Counters are always
/// exact; per-link detail rows are retained only up to `summary_rows`, with the
/// remainder accounted in [`SpecGenOutcome::omitted_link_outcomes`].
fn fold_outcome(outcome: &mut SpecGenOutcome, batch: Vec<LinkSpecOutcome>, summary_rows: usize) {
    outcome.link_count += batch.len();
    for record in batch {
        match record.status {
            LinkSpecStatus::Built => outcome.built_link_count += 1,
            LinkSpecStatus::Abandoned => outcome.abandoned_link_count += 1,
        }
        outcome.total_specs += record.specifications.len();
        if outcome.by_link.len() < summary_rows {
            outcome.by_link.push(record);
        } else {
            outcome.omitted_link_outcomes += 1;
        }
    }
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Per-link agent runner
// ---------------------------------------------------------------------------

const DEFAULT_COMPACT_RATIO: f64 = 0.8;

/// A per-link agent failure that preserves the progress counters, so the
/// scheduler's abandoned outcome reports how far the agent actually got
/// rather than a hardcoded `0`.
///
/// Carried inside the `color_eyre::Report` returned by
/// [`PlannedLinkWork::run_agent`] and recovered by
/// [`PlannedLinkWork::process`] via `downcast_ref`.
#[derive(Debug)]
pub(crate) struct AgentStepFailure {
    pub steps: usize,
    pub compact_count: usize,
    pub source: LLMYError,
}

impl std::fmt::Display for AgentStepFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "spec generator agent failed at step {} (after {} compaction(s)): {}",
            self.steps, self.compact_count, self.source
        )
    }
}

impl std::error::Error for AgentStepFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// The per-link gen-spec agent loop, owned by [`PlannedLinkWork`] — the struct
/// that already carries the link plus everything needed to run it (its grounding
/// [`ProjectIndex`] and [`SpecGenOptions`]). The `llm` and the optional regen
/// `prompt_extension` are the only per-call inputs.
impl PlannedLinkWork {
    /// Drive the gen-spec agent for this link to completion. Propagates step
    /// errors (including [`LLMYError::Billing`]) via `?`. When `prompt_extension`
    /// is `Some`, this run is a spec regen and the supplied feedback section is
    /// appended to the agent's system prompt verbatim (this method doesn't
    /// format it — that's the regen caller's job).
    pub(crate) async fn run_agent(
        &self,
        llm: &LLM,
        prompt_extension: Option<&str>,
    ) -> Result<LinkSpecOutcome> {
        let link = &self.link;
        let project_index = &self.project_index;
        let options = &self.options;
        let link_serial = self.ordinal;
        let memory = project_index.build_link_memory()?;
        let draft = tools::DraftHandle::new();
        let tool_box = draft.build_tool_box(project_index, options.max_specs_per_link);

        let cache_key = format!(
            "{}-link{:04}-e{}-h{}-f{}",
            options.cache_key, link_serial, link.extract_id, link.historical_id, link.finding_id
        );
        let debug_prefix = options.debug_prefix.as_ref().map(|prefix| {
            format!(
                "{}-link{:04}-e{}-h{}-f{}",
                prefix, link_serial, link.extract_id, link.historical_id, link.finding_id
            )
        });

        let mut system_prompt = build_system_prompt(link, options.link_source);
        if let Some(ext) = prompt_extension {
            system_prompt.push_str("\n\n");
            system_prompt.push_str(ext);
        }
        // Sequential tool calls: the spec builder exposes `update_* / add_* /
        // set_* … commit / finalize` tools. A same-turn parallel `commit`
        // racing a builder mutation can serialize a stale spec (the mutation
        // lands after commit already snapshotted). Force ordered execution.
        let mut agent = Agent::with_memory_config(
            system_prompt,
            tool_box,
            cache_key,
            &memory,
            &spec_memory_criteria(),
            AgentConfig::default().sequential_toolcall(),
        )
        .await;

        let max_input = llm.model.config.max_input();
        let (compact_threshold, using_fallback) = knowdit_kg_model::resolve_threshold(
            options.compact_context_threshold_tokens,
            max_input,
            DEFAULT_COMPACT_RATIO,
        );
        if using_fallback && knowdit_kg_model::warn_once_for("spec-generator") {
            tracing::warn!(
                "Spec generator: model `{}` is not in the llmy registry (max_input_tokens=0); \
                 using a {} token fallback context window (compact threshold {}). Add the model \
                 to the registry, or pass --gen-specs-compact-context-threshold-tokens to set it \
                 explicitly and silence this.",
                llm.model.model_id_str(),
                knowdit_kg_model::FALLBACK_CONTEXT_WINDOW_TOKENS,
                compact_threshold,
            );
        }
        match options.compact_context_threshold_tokens {
            Some(_) => tracing::info!(
                "Spec generator: compact threshold = {compact_threshold} tokens (explicit override)"
            ),
            None => tracing::info!(
                "Spec generator: compact threshold = {compact_threshold} tokens \
                 ({DEFAULT_COMPACT_RATIO} × model window {})",
                knowdit_kg_model::effective_window_tokens(max_input),
            ),
        }

        let user_prompt = build_user_prompt(link, project_index);
        let mut truncation = knowdit_kg::agent_retry::TruncationRetry::default();
        let mut step_result = truncation
            .first_step_with_user(
                &mut agent,
                user_prompt,
                llm,
                debug_prefix.as_deref(),
                options.llm_settings.clone(),
            )
            .await
            .wrap_err("spec generator agent failed on initial step")?;

        let mut steps = 1usize;
        let mut compact_count = 0usize;

        while {
            let snapshot = draft.snapshot().await;
            snapshot.final_status.is_none()
        } {
            if matches!(step_result, StepResult::Stop(_)) {
                // Agent stopped without finalizing; treat as silent abandon.
                let mut guard = draft.0.lock().await;
                guard.final_status.get_or_insert(LinkSpecStatus::Abandoned);
                guard
                    .abort_reason
                    .get_or_insert_with(|| "agent stopped without calling finalize".to_string());
                break;
            }

            if steps >= options.max_agent_steps {
                let mut guard = draft.0.lock().await;
                guard.final_status.get_or_insert(LinkSpecStatus::Abandoned);
                guard.abort_reason.get_or_insert_with(|| {
                    format!(
                        "agent exceeded max_agent_steps={} before finalizing",
                        options.max_agent_steps
                    )
                });
                break;
            }

            if let Some(tokens) = agent.approx_context_tokens(&llm.model.config)
                && tokens >= compact_threshold
            {
                tracing::info!(
                    "Spec generator compacting agent context (tokens={tokens}, threshold={compact_threshold})"
                );
                agent = agent
                    .compact(
                        llm,
                        debug_prefix
                            .as_ref()
                            .map(|prefix| format!("{prefix}-compact"))
                            .as_deref(),
                        options.llm_settings.clone(),
                    )
                    .await
                    .wrap_err("spec generator agent failed to compact context")?;
                compact_count += 1;
            }

            steps += 1;
            match truncation
                .step(
                    &mut agent,
                    llm,
                    debug_prefix.as_deref(),
                    options.llm_settings.clone(),
                )
                .await
            {
                Ok(result) => step_result = result,
                // Truncation retries exhausted: the model cannot fit a response
                // in the provider's output cap, so stop spending on this link
                // but keep everything committed so far.
                Err(err) if knowdit_kg::agent_retry::is_output_length(&err) => {
                    let mut guard = draft.0.lock().await;
                    guard.final_status.get_or_insert(LinkSpecStatus::Abandoned);
                    guard.abort_reason.get_or_insert_with(|| {
                        format!(
                            "agent response truncated by the provider output cap {} times in a row \
                             (last at step {steps}); raise --llm-max-completion-tokens or use a \
                             model with a larger output limit",
                            truncation.consecutive_failures()
                        )
                    });
                    break;
                }
                Err(err) => {
                    return Err(AgentStepFailure {
                        steps,
                        compact_count,
                        source: err,
                    }
                    .into());
                }
            }
        }

        let snapshot = draft.snapshot().await;
        let status = snapshot.final_status.unwrap_or(LinkSpecStatus::Abandoned);
        Ok(LinkSpecOutcome {
            ordinal: link_serial,
            extract_id: link.extract_id,
            historical_id: link.historical_id,
            finding_id: link.finding_id,
            status,
            specifications: snapshot.completed.clone(),
            specification_ids: Vec::new(),
            abort_reason: snapshot.abort_reason.clone(),
            final_summary: snapshot.final_summary.clone(),
            steps,
            compact_count,
        })
    }

    /// [`Self::run_agent`] + logging + error classification. A per-link agent
    /// failure becomes an `Abandoned` outcome (`Ok`) so the scheduler keeps
    /// going; only a **billing-cap exhaustion** returns `Err`, signalling the
    /// caller to abort the whole run rather than burn the rest of the queue
    /// against an already-dead cap (see [`is_billing_exhausted`]).
    pub(crate) async fn process(&self, llm: &LLM) -> Result<LinkSpecOutcome> {
        let link = &self.link;
        let link_idx_1based = self.ordinal;
        let total_links = self.total_links;
        let label = format!(
            "link={} (total = {}) {}",
            link_idx_1based, total_links, link
        );
        tracing::info!("Spec generator starting {label}");
        match self.run_agent(llm, None).await {
            Ok(outcome) => {
                let kind = match outcome.status {
                    LinkSpecStatus::Built => "built",
                    LinkSpecStatus::Abandoned => "abandoned",
                };
                tracing::info!(
                    "{label}: {kind} ({} spec(s), {} step(s), {} compaction(s)){}",
                    outcome.specifications.len(),
                    outcome.steps,
                    outcome.compact_count,
                    outcome
                        .abort_reason
                        .as_ref()
                        .map(|r| format!(" — {r}"))
                        .unwrap_or_default(),
                );
                Ok(outcome)
            }
            // Billing-cap exhaustion is run-fatal, not a per-link failure: every
            // later LLM call hits the same dead cap. Propagate so the
            // orchestrator aborts instead of abandoning the rest of the queue.
            Err(err) if is_billing_exhausted(&err) => Err(err),
            Err(err) => {
                tracing::error!(
                    "{label}: agent run failed, abandoning link without committing — {err:#}"
                );
                // Recover how far the agent got so the outcome isn't reported as
                // 0 steps. `run_agent` tags in-loop step failures with
                // `AgentStepFailure`; earlier failures (memory build, initial
                // step) genuinely have no completed steps to report.
                let (steps, compact_count) = err
                    .downcast_ref::<AgentStepFailure>()
                    .map(|f| (f.steps, f.compact_count))
                    .unwrap_or((0, 0));
                Ok(LinkSpecOutcome {
                    ordinal: link_idx_1based,
                    extract_id: link.extract_id,
                    historical_id: link.historical_id,
                    finding_id: link.finding_id,
                    status: LinkSpecStatus::Abandoned,
                    specifications: Vec::new(),
                    specification_ids: Vec::new(),
                    abort_reason: Some(format!("agent error: {err:#}")),
                    final_summary: None,
                    steps,
                    compact_count,
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Memory construction
// ---------------------------------------------------------------------------

pub(crate) fn spec_memory_criteria() -> AgentMemorySystemPromptCriteria {
    AgentMemorySystemPromptCriteria::builder()
        .append_short_term_memory_criteria(
            "Per-contract source bundles. Each entry contains the contract source plus its own and inherited state variables. Titles are formatted as `relative_file_path:Contract:line:col` and are stable across the agent's lifetime."
                .to_string(),
        )
        .append_short_term_memory_trigger(
            "Before reasoning about how a state variable, modifier, or function body behaves on the current project."
                .to_string(),
        )
        .append_short_term_memory_trigger(
            "When mapping the historical vulnerability pattern onto concrete project contracts."
                .to_string(),
        )
        .append_short_term_memory_operator(
            "These contract entries were preloaded by the runtime. Read them; do not delete or rewrite them."
                .to_string(),
        )
        .build()
}

// ---------------------------------------------------------------------------
// Serializable wire forms (used by the CLI)
// ---------------------------------------------------------------------------

/// Serializable summary the CLI dumps to JSON / md.
#[derive(Debug, Clone, Serialize)]
pub struct LinkSpecSummary {
    pub extract_id: i32,
    pub historical_id: i32,
    pub finding_id: i32,
    pub status: LinkSpecStatus,
    pub specifications: Vec<AuditSpecification>,
    pub abort_reason: Option<String>,
    pub final_summary: Option<String>,
    pub steps: usize,
    pub compact_count: usize,
}

impl From<&LinkSpecOutcome> for LinkSpecSummary {
    fn from(outcome: &LinkSpecOutcome) -> Self {
        Self {
            extract_id: outcome.extract_id,
            historical_id: outcome.historical_id,
            finding_id: outcome.finding_id,
            status: outcome.status,
            specifications: outcome.specifications.clone(),
            abort_reason: outcome.abort_reason.clone(),
            final_summary: outcome.final_summary.clone(),
            steps: outcome.steps,
            compact_count: outcome.compact_count,
        }
    }
}

#[cfg(test)]
mod planner_tests {
    use super::*;
    use knowdit_kg_model::link_strength::LinkStrength;

    /// A step failure must survive the trip through `color_eyre::Report` so
    /// [`PlannedLinkWork::process`] can report the real step count instead of
    /// the hardcoded `0` that used to hide all progress on an abandoned link.
    #[test]
    fn agent_step_failure_round_trips_through_eyre_with_progress() {
        let report: color_eyre::Report = AgentStepFailure {
            steps: 40,
            compact_count: 2,
            source: LLMYError::OutputLength,
        }
        .into();

        let recovered = report
            .downcast_ref::<AgentStepFailure>()
            .expect("AgentStepFailure should be recoverable from the report");
        assert_eq!(recovered.steps, 40);
        assert_eq!(recovered.compact_count, 2);

        // The rendered message names both the step and the underlying cause, so
        // an operator reading the log can tell how far the agent got.
        let rendered = format!("{report:#}");
        assert!(rendered.contains("step 40"), "got: {rendered}");
        assert!(rendered.contains("reach output length limit"), "got: {rendered}");
    }

    /// A failure from *before* the step loop (memory build, initial step) has
    /// no completed steps to report; the downcast must simply miss rather than
    /// invent a count.
    #[test]
    fn non_step_failures_report_no_progress() {
        let report: color_eyre::Report = color_eyre::eyre::eyre!("context build failed");
        let recovered = report.downcast_ref::<AgentStepFailure>();
        assert!(recovered.is_none());
    }

    /// Billing exhaustion must still be detected through the new
    /// `AgentStepFailure` wrapper: it is run-fatal (the whole run aborts), so
    /// losing it would silently degrade to abandoning every remaining link one
    /// wasted LLM call at a time.
    #[test]
    fn billing_exhaustion_survives_the_step_failure_wrapper() {
        let report: color_eyre::Report = AgentStepFailure {
            steps: 12,
            compact_count: 0,
            source: LLMYError::Billing {
                cap: Decimal::from(100),
                current: Decimal::from(101),
                node: 0,
                scope: Some("run".to_string()),
            },
        }
        .into();

        let billing = billing_exhaustion(&report).expect("billing cause should be found");
        assert_eq!(billing.cap, Decimal::from(100));
        assert_eq!(billing.current, Decimal::from(101));
        assert!(is_billing_exhausted(&report));
        // Progress is still recoverable on the same report.
        assert_eq!(report.downcast_ref::<AgentStepFailure>().unwrap().steps, 12);
    }

    fn cand(
        extract_id: i32,
        historical_id: i32,
        finding_id: i32,
        match_strength: MatchStrength,
        link_strength: LinkStrength,
    ) -> LinkCandidate {
        LinkCandidate {
            key: LinkKey {
                extract_id,
                historical_id,
                finding_id,
            },
            match_strength,
            link_strength,
            pre_committed_spec_ids: Vec::new(),
        }
    }

    fn keys(selected: VecDeque<LinkCandidate>) -> Vec<(i32, i32, i32)> {
        selected
            .into_iter()
            .map(|c| (c.key.extract_id, c.key.historical_id, c.key.finding_id))
            .collect()
    }

    #[test]
    fn strongest_match_bucket_first_regardless_of_extract_id() {
        let options = SpecGenOptions::default();
        let candidates = vec![
            cand(1, 10, 1, MatchStrength::Medium, LinkStrength::High),
            cand(2, 20, 2, MatchStrength::High, LinkStrength::High),
        ];
        assert_eq!(
            keys(select_pending(candidates, &options, 2)),
            vec![(2, 20, 2), (1, 10, 1)],
        );
    }

    #[test]
    fn stronger_finding_link_bucket_precedes_weaker_within_match_tier() {
        let options = SpecGenOptions::default();
        let candidates = vec![
            cand(1, 10, 2, MatchStrength::High, LinkStrength::Medium),
            cand(1, 10, 1, MatchStrength::High, LinkStrength::High),
        ];
        assert_eq!(
            keys(select_pending(candidates, &options, 2)),
            vec![(1, 10, 1), (1, 10, 2)],
        );
    }

    #[test]
    fn no_medium_match_before_eligible_high_match() {
        let options = SpecGenOptions::default();
        let candidates = vec![
            cand(1, 10, 1, MatchStrength::Medium, LinkStrength::High),
            cand(2, 20, 2, MatchStrength::High, LinkStrength::Medium),
        ];
        assert_eq!(
            keys(select_pending(candidates, &options, 2)),
            vec![(2, 20, 2), (1, 10, 1)],
        );
    }

    #[test]
    fn within_bucket_round_robins_extracts_in_ascending_id_order() {
        let options = SpecGenOptions::default();
        let candidates = vec![
            cand(2, 5, 1, MatchStrength::High, LinkStrength::High),
            cand(1, 5, 1, MatchStrength::High, LinkStrength::High),
            cand(2, 5, 2, MatchStrength::High, LinkStrength::High),
            cand(1, 5, 2, MatchStrength::High, LinkStrength::High),
        ];
        assert_eq!(
            keys(select_pending(candidates, &options, 4)),
            vec![(1, 5, 1), (2, 5, 1), (1, 5, 2), (2, 5, 2)],
        );
    }

    #[test]
    fn max_links_spreads_across_extracts_without_promoting_weaker_bucket() {
        let mut options = SpecGenOptions::default();
        options.max_links = Some(3);
        let candidates = vec![
            cand(1, 5, 1, MatchStrength::High, LinkStrength::High),
            cand(1, 5, 2, MatchStrength::High, LinkStrength::High),
            cand(2, 5, 1, MatchStrength::High, LinkStrength::High),
            cand(2, 5, 2, MatchStrength::Medium, LinkStrength::High),
        ];
        assert_eq!(
            keys(select_pending(candidates, &options, 4)),
            vec![(1, 5, 1), (2, 5, 1), (1, 5, 2)],
        );
    }

    #[test]
    fn per_extract_cap_carries_across_buckets() {
        let mut options = SpecGenOptions::default();
        options.max_links_per_extract = Some(1);
        let candidates = vec![
            cand(1, 10, 1, MatchStrength::High, LinkStrength::High),
            cand(1, 10, 2, MatchStrength::Medium, LinkStrength::High),
            cand(2, 20, 3, MatchStrength::High, LinkStrength::High),
        ];
        let selected = keys(select_pending(candidates, &options, 3));
        assert_eq!(selected, vec![(1, 10, 1), (2, 20, 3)]);
    }

    #[test]
    fn per_extract_historical_cap_carries_across_buckets() {
        let mut options = SpecGenOptions::default();
        options.max_findings_per_historical = Some(1);
        let candidates = vec![
            cand(1, 10, 1, MatchStrength::High, LinkStrength::High),
            cand(1, 10, 2, MatchStrength::Medium, LinkStrength::High),
            cand(1, 11, 3, MatchStrength::High, LinkStrength::High),
        ];
        let selected = keys(select_pending(candidates, &options, 3));
        assert_eq!(selected, vec![(1, 10, 1), (1, 11, 3)]);
    }

    #[test]
    fn selection_is_deterministic_under_input_permutation() {
        let options = SpecGenOptions::default();
        let base = vec![
            cand(2, 5, 2, MatchStrength::High, LinkStrength::High),
            cand(1, 5, 2, MatchStrength::High, LinkStrength::Medium),
            cand(1, 5, 1, MatchStrength::High, LinkStrength::High),
            cand(2, 5, 1, MatchStrength::High, LinkStrength::High),
        ];
        let mut reversed = base.clone();
        reversed.reverse();
        assert_eq!(
            keys(select_pending(base, &options, 4)),
            keys(select_pending(reversed, &options, 4)),
        );
    }

    #[test]
    fn effective_batch_links_is_never_below_concurrency() {
        let mut options = SpecGenOptions {
            batch_links: 10,
            concurrency: 50,
            ..Default::default()
        };
        assert_eq!(options.effective_batch_links(), 50);
        options.concurrency = 5;
        assert_eq!(options.effective_batch_links(), 10);
    }

    /// Regression: `deepseek-ai/DeepSeek-V4.1-Flash` is not in llmy's registry
    /// (the entry is `deepseek/deepseek-v4-flash`), so `max_input_tokens` is 0.
    /// Before the fallback, `max_input() * 0.8 == 0` made the spec generator
    /// compact on *every* step — agents re-read the same function bodies in a
    /// loop and burned millions of input tokens for zero specs.
    #[test]
    fn unknown_registry_model_still_yields_a_usable_compact_threshold() {
        use std::str::FromStr;
        let model = llmy::client::model::OpenAIModel::from_str("deepseek-ai/DeepSeek-V4.1-Flash")
            .expect("custom model id parses");
        assert_eq!(
            model.config.max_input(),
            0,
            "precondition: this id misses the registry and reports a zero window"
        );

        let threshold = knowdit_kg_model::context_budget(
            model.config.max_input(),
            DEFAULT_COMPACT_RATIO,
        );
        assert!(
            threshold > 10_000,
            "compact threshold must be far above zero for an unknown model, got {threshold}"
        );
        assert!(knowdit_kg_model::using_fallback_window(model.config.max_input()));
    }
}

/// DB-backed planner tests: stand up a temp project DB, seed real rows, and
/// exercise `PreparedLinkPlan` / `SpecGenStream` / `SpecRuntime` end-to-end
/// (batch bounds, resume semantics, regenerate reset, ordinals, exact
/// `(E, H, F)` resolution).
#[cfg(test)]
mod planner_db_tests {
    use super::*;
    use knowdit_kg_model::audit_finding::FindingSeverity;
    use knowdit_kg_model::category::DeFiCategory;
    use knowdit_kg_model::link_strength::LinkStrength;
    use knowdit_repo_model::{
        CodeGenCore, CodeGenRecord, CodeGenStatus, HistoricalLinkedFinding,
        HistoricalSemanticRecord, SemanticMatch, SemanticMatchSet,
    };
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempRepo {
        repo: RepoDatabase,
        path: PathBuf,
    }

    impl Drop for TempRepo {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
            let _ = std::fs::remove_file(self.path.with_extension("sqlite3-shm"));
            let _ = std::fs::remove_file(self.path.with_extension("sqlite3-wal"));
        }
    }

    async fn temp_repo() -> TempRepo {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "knowdit-planner-test-{}-{unique}.sqlite3",
            std::process::id()
        ));
        let repo = RepoDatabase::open_sqlite(path.clone())
            .await
            .expect("temp repo opens");
        repo.init_schema().await.expect("schema initializes");
        TempRepo { repo, path }
    }

    fn extract(name: &str) -> ExtractedSemantic {
        ExtractedSemantic {
            name: name.to_string(),
            category: DeFiCategory::Lending,
            definition: String::new(),
            description: format!("{name} description"),
            functions: Vec::new(),
        }
    }

    fn finding(id: i32) -> knowdit_kg_model::db::audit_finding::Model {
        knowdit_kg_model::db::audit_finding::Model {
            id,
            title: format!("finding {id}"),
            severity: FindingSeverity::Medium,
            root_cause: String::new(),
            description: format!("finding {id} description"),
            patterns: String::new(),
            exploits: String::new(),
        }
    }

    fn historical(id: i32, links: &[(i32, LinkStrength)]) -> HistoricalSemanticRecord {
        HistoricalSemanticRecord {
            semantic: knowdit_kg_model::db::semantic_node::Model {
                id,
                name: format!("hist {id}"),
                definition: String::new(),
                description: format!("hist {id} description"),
                category: DeFiCategory::Lending,
            },
            findings: links
                .iter()
                .map(|(finding_id, strength)| HistoricalLinkedFinding {
                    finding: finding(*finding_id),
                    strength: *strength,
                    evidence: String::new(),
                    raw_children: Vec::new(),
                    rendered_description: String::new(),
                    rendered_patterns: String::new(),
                    rendered_exploits: String::new(),
                })
                .collect(),
            raw_children: Vec::new(),
            rendered_description: String::new(),
        }
    }

    async fn seed(
        repo: &RepoDatabase,
        extracts: &[ExtractedSemantic],
        historicals: Vec<HistoricalSemanticRecord>,
        matches: Vec<(i32, i32, MatchStrength)>,
    ) {
        repo.replace_project_semantics(extracts)
            .await
            .expect("project semantics write");
        let set = SemanticMatchSet {
            historicals,
            matches: matches
                .into_iter()
                .map(|(extract_id, historical_id, strength)| SemanticMatch {
                    extract_id,
                    historical_id,
                    strength,
                    evidence: String::new(),
                })
                .collect(),
        };
        repo.write_semantic_match_results(&set)
            .await
            .expect("match results write");
    }

    fn spec_record(extract_id: i32, historical_id: i32, finding_id: i32) -> SpecificationRecord {
        SpecificationRecord {
            semantic_id: extract_id,
            historical_id,
            finding_id,
            specification_json: "{}".to_string(),
        }
    }

    fn code_gen_for(spec_id: i32) -> CodeGenRecord {
        CodeGenRecord {
            core: CodeGenCore {
                spec_id,
                harness_relative_path: String::new(),
                harness_source: String::new(),
                status: CodeGenStatus::Completed,
                final_reason: String::new(),
                agent_steps: 0,
            },
            runs: Vec::new(),
        }
    }

    async fn mark_partial(
        repo: &RepoDatabase,
        extract_id: i32,
        historical_id: i32,
        finding_id: i32,
        count: usize,
    ) -> Vec<i32> {
        let records: Vec<SpecificationRecord> = (0..count)
            .map(|_| spec_record(extract_id, historical_id, finding_id))
            .collect();
        repo.append_specifications(&records)
            .await
            .expect("append specs")
    }

    async fn mark_built(
        repo: &RepoDatabase,
        extract_id: i32,
        historical_id: i32,
        finding_id: i32,
    ) -> i32 {
        let ids = mark_partial(repo, extract_id, historical_id, finding_id, 1).await;
        repo.write_code_gen_with_runs(&code_gen_for(ids[0]), &[])
            .await
            .expect("write code_gen");
        ids[0]
    }

    fn default_options() -> SpecGenOptions {
        SpecGenOptions::default()
    }

    // --- Verification 1: batch bound, uniqueness, termination -------------

    #[tokio::test]
    async fn take_batch_is_bounded_and_emits_each_key_once() {
        let temp = temp_repo().await;
        let links: Vec<(i32, LinkStrength)> = (1..=10).map(|f| (f, LinkStrength::High)).collect();
        seed(
            &temp.repo,
            &[extract("e1")],
            vec![historical(100, &links)],
            vec![(1, 100, MatchStrength::High)],
        )
        .await;

        let mut plan = PreparedLinkPlan::prepare(&temp.repo, &default_options())
            .await
            .expect("prepare ok")
            .expect("plan present");
        assert_eq!(plan.planned_total(), 10);

        let batch = NonZeroUsize::new(4).unwrap();
        let mut all_keys: Vec<LinkKey> = Vec::new();
        let mut batches = 0usize;
        loop {
            let got = plan.take_batch(batch);
            assert!(got.len() <= batch.get(), "batch exceeded bound");
            if got.is_empty() {
                break;
            }
            batches += 1;
            all_keys.extend(got.iter().map(|l| l.key()));
        }
        assert_eq!(all_keys.len(), 10, "every selected key emitted exactly once");
        let unique: HashSet<LinkKey> = all_keys.iter().copied().collect();
        assert_eq!(unique.len(), 10, "no duplicate keys");
        assert_eq!(plan.remaining(), 0, "plan terminated");
        assert!(batches >= 3, "crossed multiple batches (got {batches})");
    }

    // --- Verification 5: per-extract cap across batches -------------------

    #[tokio::test]
    async fn per_extract_cap_holds_across_batches() {
        let temp = temp_repo().await;
        let links: Vec<(i32, LinkStrength)> = (1..=5).map(|f| (f, LinkStrength::High)).collect();
        seed(
            &temp.repo,
            &[extract("e1"), extract("e2")],
            vec![historical(100, &links), historical(200, &links)],
            vec![
                (1, 100, MatchStrength::High),
                (2, 200, MatchStrength::High),
            ],
        )
        .await;

        let mut options = default_options();
        options.max_links_per_extract = Some(2);
        let mut plan = PreparedLinkPlan::prepare(&temp.repo, &options)
            .await
            .unwrap()
            .unwrap();

        let mut per_extract: BTreeMap<i32, usize> = BTreeMap::new();
        loop {
            let got = plan.take_batch(NonZeroUsize::new(1).unwrap());
            if got.is_empty() {
                break;
            }
            for link in got {
                *per_extract.entry(link.extract_id).or_insert(0) += 1;
            }
        }
        assert_eq!(per_extract.get(&1), Some(&2));
        assert_eq!(per_extract.get(&2), Some(&2));
    }

    // --- Verification 7: Built consumes no cap; Partial one slot ----------

    #[tokio::test]
    async fn built_consumes_no_cap_and_partial_carries_sorted_ids() {
        let temp = temp_repo().await;
        seed(
            &temp.repo,
            &[extract("e1")],
            vec![
                historical(100, &[(1, LinkStrength::High), (2, LinkStrength::High)]),
                historical(200, &[(4, LinkStrength::High)]),
            ],
            vec![
                (1, 100, MatchStrength::High),
                (1, 200, MatchStrength::High),
            ],
        )
        .await;

        // (1,100,1) fully built; (1,100,2) partial; (1,200,4) fresh.
        mark_built(&temp.repo, 1, 100, 1).await;
        let partial_ids = mark_partial(&temp.repo, 1, 100, 2, 2).await;

        let mut options = default_options();
        options.max_links = Some(1);
        let mut plan = PreparedLinkPlan::prepare(&temp.repo, &options)
            .await
            .unwrap()
            .unwrap();

        // Built is dropped and does not consume the single cap slot; the
        // partial link takes it (rather than the fresh (1,200,4)).
        assert_eq!(plan.planned_total(), 1);
        let batch = plan.take_batch(NonZeroUsize::new(10).unwrap());
        assert_eq!(batch.len(), 1);
        assert_eq!(
            batch[0].key(),
            LinkKey {
                extract_id: 1,
                historical_id: 100,
                finding_id: 2,
            }
        );
        let mut expected = partial_ids.clone();
        expected.sort_unstable();
        assert_eq!(batch[0].pre_committed_spec_ids, expected);
        assert_eq!(plan.remaining(), 0);
    }

    // --- Verification 8: regenerate resets once; prior batch survives -----

    #[tokio::test]
    async fn regenerate_resets_once_and_prior_batch_survives() {
        let temp = temp_repo().await;
        seed(
            &temp.repo,
            &[extract("e1")],
            vec![historical(
                100,
                &[(1, LinkStrength::High), (2, LinkStrength::High)],
            )],
            vec![(1, 100, MatchStrength::High)],
        )
        .await;

        // Simulate a prior run's output that regenerate must clear.
        let stale = mark_built(&temp.repo, 1, 100, 1).await;
        assert!(!temp.repo.load_specifications().await.unwrap().is_empty());

        let mut options = default_options();
        options.regenerate = true;
        let mut plan = PreparedLinkPlan::prepare(&temp.repo, &options)
            .await
            .unwrap()
            .unwrap();

        // Reset happened exactly once during preparation.
        assert!(
            temp.repo.load_specifications().await.unwrap().is_empty(),
            "regenerate cleared specs before batching"
        );
        let _ = stale;

        // Batch 1 output committed after preparation.
        let fresh = mark_partial(&temp.repo, 1, 100, 2, 1).await;
        assert_eq!(fresh.len(), 1);

        // Draining further batches must NOT delete batch 1's spec.
        loop {
            let got = plan.take_batch(NonZeroUsize::new(1).unwrap());
            if got.is_empty() {
                break;
            }
        }
        let remaining = temp.repo.load_specifications().await.unwrap();
        assert_eq!(
            remaining.iter().map(|s| s.id).collect::<Vec<_>>(),
            fresh,
            "batch 1 spec survived later batches"
        );
    }

    // --- Verification 9: ordinals unique across batches -------------------

    #[tokio::test]
    async fn stream_ordinals_do_not_repeat_across_batches() {
        let temp = temp_repo().await;
        let links: Vec<(i32, LinkStrength)> = (1..=6).map(|f| (f, LinkStrength::High)).collect();
        seed(
            &temp.repo,
            &[extract("e1")],
            vec![historical(100, &links)],
            vec![(1, 100, MatchStrength::High)],
        )
        .await;

        let mut options = default_options();
        options.batch_links = 2;
        let mut stream = SpecificationGenerator::new()
            .prepare_stream(&temp.repo, &options)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stream.total_links(), 6);

        let mut ordinals: Vec<usize> = Vec::new();
        let mut keys: Vec<LinkKey> = Vec::new();
        while let Some(work) = stream.pop_next_work() {
            ordinals.push(work.ordinal());
            keys.push(work.link_key());
        }
        assert_eq!(ordinals, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(keys.iter().collect::<HashSet<_>>().len(), 6);
    }

    // --- Verification 10: exact counters when details are dropped ---------

    #[test]
    fn fold_outcome_keeps_counters_exact_at_zero_summary_rows() {
        let mut specs1 = AuditSpecification::default();
        specs1.summary = "a".to_string();
        let outcome = |status: LinkSpecStatus, specs: Vec<AuditSpecification>| LinkSpecOutcome {
            ordinal: 1,
            extract_id: 1,
            historical_id: 1,
            finding_id: 1,
            status,
            specifications: specs,
            specification_ids: Vec::new(),
            abort_reason: None,
            final_summary: None,
            steps: 0,
            compact_count: 0,
        };
        let batch = vec![
            outcome(LinkSpecStatus::Built, vec![specs1.clone(), specs1.clone()]),
            outcome(LinkSpecStatus::Built, vec![specs1.clone()]),
            outcome(LinkSpecStatus::Abandoned, Vec::new()),
        ];

        let mut aggregate = SpecGenOutcome::default();
        fold_outcome(&mut aggregate, batch, 0);
        assert_eq!(aggregate.link_count, 3);
        assert_eq!(aggregate.built_link_count, 2);
        assert_eq!(aggregate.abandoned_link_count, 1);
        assert_eq!(aggregate.total_specs, 3);
        assert!(aggregate.by_link.is_empty());
        assert_eq!(aggregate.omitted_link_outcomes, 3);

        // Bounded (not zero) retention keeps the first N and counts the rest.
        let batch = vec![
            outcome(LinkSpecStatus::Built, Vec::new()),
            outcome(LinkSpecStatus::Built, Vec::new()),
            outcome(LinkSpecStatus::Built, Vec::new()),
        ];
        let mut aggregate = SpecGenOutcome::default();
        fold_outcome(&mut aggregate, batch, 2);
        assert_eq!(aggregate.link_count, 3);
        assert_eq!(aggregate.by_link.len(), 2);
        assert_eq!(aggregate.omitted_link_outcomes, 1);
    }

    // --- Verification 11 + 12: exact (E,H,F), below thresholds ------------

    #[tokio::test]
    async fn resolve_link_is_exact_and_below_thresholds() {
        let temp = temp_repo().await;
        // Two sibling historicals both linking finding 7; both edges are Low
        // (below the default Medium thresholds), so a fresh plan must exclude
        // them entirely while regen still resolves each exactly.
        seed(
            &temp.repo,
            &[extract("e1")],
            vec![
                historical(500, &[(7, LinkStrength::Low)]),
                historical(600, &[(7, LinkStrength::Low)]),
            ],
            vec![
                (1, 500, MatchStrength::Low),
                (1, 600, MatchStrength::Low),
            ],
        )
        .await;

        let runtime = SpecRuntime::load(&temp.repo).await.unwrap().unwrap();
        let l600 = runtime.resolve_link(1, 600, 7).unwrap();
        assert_eq!(l600.historical_id, 600);
        let l500 = runtime.resolve_link(1, 500, 7).unwrap();
        assert_eq!(l500.historical_id, 500, "sibling H resolves exactly");

        // Default thresholds filter these out of a fresh plan.
        assert!(
            PreparedLinkPlan::prepare(&temp.repo, &default_options())
                .await
                .unwrap()
                .is_none(),
            "below-threshold links excluded from fresh planning"
        );
    }

    // --- Verification 2: plan terminates on abandoned/no-spec links -------

    #[tokio::test]
    async fn plan_emits_every_selected_candidate_once() {
        let temp = temp_repo().await;
        seed(
            &temp.repo,
            &[extract("e1")],
            vec![historical(
                100,
                &[(1, LinkStrength::High), (2, LinkStrength::Medium)],
            )],
            vec![(1, 100, MatchStrength::High)],
        )
        .await;

        let mut plan = PreparedLinkPlan::prepare(&temp.repo, &default_options())
            .await
            .unwrap()
            .unwrap();
        let total = plan.planned_total();
        let mut seen: HashSet<LinkKey> = HashSet::new();
        while let Some(link) = plan.take_batch(NonZeroUsize::new(1).unwrap()).into_iter().next() {
            assert!(seen.insert(link.key()), "key emitted twice");
        }
        assert_eq!(seen.len(), total);
        assert_eq!(plan.remaining(), 0);
    }

    // --- Memory validation: >1M candidates, bounded materialization -------

    /// Resident set size in KiB, from `/proc/self/statm` (Linux only).
    #[cfg(target_os = "linux")]
    fn rss_kb() -> Option<u64> {
        let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
        let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
        Some(pages * 4)
    }
    #[cfg(not(target_os = "linux"))]
    fn rss_kb() -> Option<u64> {
        None
    }

    fn big_historical(id: i32, n: usize) -> HistoricalSemanticRecord {
        let findings = (0..n)
            .map(|i| HistoricalLinkedFinding {
                finding: knowdit_kg_model::db::audit_finding::Model {
                    id: i as i32,
                    title: String::new(),
                    severity: FindingSeverity::Medium,
                    root_cause: String::new(),
                    description: String::new(),
                    patterns: String::new(),
                    exploits: String::new(),
                },
                strength: LinkStrength::High,
                evidence: String::new(),
                raw_children: Vec::new(),
                rendered_description: String::new(),
                rendered_patterns: String::new(),
                rendered_exploits: String::new(),
            })
            .collect();
        HistoricalSemanticRecord {
            semantic: knowdit_kg_model::db::semantic_node::Model {
                id,
                name: String::new(),
                definition: String::new(),
                description: String::new(),
                category: DeFiCategory::Lending,
            },
            findings,
            raw_children: Vec::new(),
            rendered_description: String::new(),
        }
    }

    /// Planner-only memory validation: over a million compact candidates, the
    /// number of simultaneously materialized `LinkInput` values never exceeds
    /// the effective batch, every candidate is emitted exactly once, and the
    /// process does not retain per-candidate heavy payloads.
    ///
    /// Manual: `cargo test -p knowdit-audit --lib \
    ///   materialization_stays_batch_bounded_over_a_million_candidates -- --ignored --nocapture`.
    /// Override the candidate count with `KNOWDIT_MEM_TEST_CANDIDATES`.
    #[tokio::test]
    #[ignore = "manual memory validation (run with --ignored); builds >1M candidates"]
    async fn materialization_stays_batch_bounded_over_a_million_candidates() {
        let n: usize = std::env::var("KNOWDIT_MEM_TEST_CANDIDATES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_000_100);
        assert!(n > 1_000_000, "validation must exceed a million candidates");

        // Minimal real DB just to obtain a `ProjectIndex`; the source indexes
        // are then replaced with the synthetic million-finding record.
        let temp = temp_repo().await;
        seed(
            &temp.repo,
            &[extract("e")],
            vec![historical(100, &[(0, LinkStrength::High)])],
            vec![(1, 100, MatchStrength::High)],
        )
        .await;
        let mut runtime = SpecRuntime::load(&temp.repo).await.unwrap().unwrap();
        runtime.historical_by_id.insert(100, big_historical(100, n));
        runtime.match_strength_by_pair.insert((1, 100), MatchStrength::High);
        runtime.rebuild_findings_index();

        let candidates: Vec<LinkCandidate> = (0..n)
            .map(|i| LinkCandidate {
                key: LinkKey {
                    extract_id: 1,
                    historical_id: 100,
                    finding_id: i as i32,
                },
                match_strength: MatchStrength::High,
                link_strength: LinkStrength::High,
                pre_committed_spec_ids: Vec::new(),
            })
            .collect();
        let mut plan = PreparedLinkPlan::from_candidates_for_test(runtime, candidates);
        assert_eq!(plan.planned_total(), n);

        let batch = NonZeroUsize::new(1000).unwrap();
        let rss_before = rss_kb();
        let mut max_batch = 0usize;
        let mut total = 0usize;
        let mut batches = 0usize;
        loop {
            let got = plan.take_batch(batch);
            if got.is_empty() {
                break;
            }
            max_batch = max_batch.max(got.len());
            total += got.len();
            batches += 1;
            drop(got); // release the materialized batch before the next
        }
        let rss_after = rss_kb();

        assert_eq!(total, n, "every candidate emitted once");
        assert_eq!(plan.remaining(), 0, "plan terminated");
        assert!(
            max_batch <= batch.get(),
            "materialized {max_batch} links at once, exceeding batch {}",
            batch.get()
        );
        if let (Some(before), Some(after)) = (rss_before, rss_after) {
            let growth_mb = after.saturating_sub(before) / 1024;
            println!(
                "[mem] candidates={n} batches={batches} max_batch={max_batch} rss_growth={growth_mb}MB"
            );
            assert!(
                growth_mb < 1024,
                "RSS grew {growth_mb}MB while materializing in bounded batches — \
                 evidence of per-candidate retention"
            );
        }
    }

    // --- Real-data planner validation (manual, no LLM calls) --------------

    /// Bounded-planner inspection on a real project DB. Confirms that
    /// planning and batched materialization stay memory-bounded on real mapper
    /// output (≈79k matches for paxos), with no LLM/fuzz work — the
    /// prerequisite the plan calls out before attempting a long capped run.
    ///
    /// Works on a **copy** of the fixture; never mutates the original.
    /// Override the fixture with `KNOWDIT_PAXOS_DB` and the work cap with
    /// `KNOWDIT_PAXOS_MAX_LINKS` (default 5000).
    ///
    /// Manual: `cargo test -p knowdit-audit --lib \
    ///   paxos_planner_is_batch_bounded_on_real_data -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore = "manual real-data planner validation (run with --ignored)"]
    async fn paxos_planner_is_batch_bounded_on_real_data() {
        let fixture = std::env::var_os("KNOWDIT_PAXOS_DB")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../code4rena/contracts/paxos-token-contracts/knowdit.sqlite3")
            });
        if !fixture.exists() {
            eprintln!(
                "[paxos] fixture not found at {}; skipping real-data validation",
                fixture.display()
            );
            return;
        }

        // Copy the fixture (plus WAL sidecars, if any) so the original is
        // never touched.
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let copy = std::env::temp_dir().join(format!(
            "knowdit-paxos-copy-{}-{unique}.sqlite3",
            std::process::id()
        ));
        std::fs::copy(&fixture, &copy).expect("copy paxos fixture");
        for ext in ["-wal", "-shm"] {
            let side = PathBuf::from(format!("{}{}", fixture.display(), ext));
            if side.exists() {
                let _ = std::fs::copy(&side, format!("{}{}", copy.display(), ext));
            }
        }

        let repo = RepoDatabase::open_sqlite(copy.clone())
            .await
            .expect("open paxos copy");
        // Drop guard cleans the copy (and sidecars) on any exit path.
        let _guard = TempRepo {
            repo: repo.clone(),
            path: copy.clone(),
        };

        let cap: usize = std::env::var("KNOWDIT_PAXOS_MAX_LINKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5000);
        let mut options = default_options();
        options.max_links = Some(cap);

        let rss_before = rss_kb();
        let mut plan = PreparedLinkPlan::prepare(&repo, &options)
            .await
            .expect("plan on real data")
            .expect("plan present");
        let planned = plan.planned_total();
        let rss_after_prepare = rss_kb();

        let batch = NonZeroUsize::new(500).unwrap();
        let mut max_batch = 0usize;
        let mut total = 0usize;
        loop {
            let got = plan.take_batch(batch);
            if got.is_empty() {
                break;
            }
            max_batch = max_batch.max(got.len());
            total += got.len();
            drop(got);
        }
        let rss_after = rss_kb();

        let mb = |v: Option<u64>| v.unwrap_or(0) / 1024;
        println!(
            "[paxos] cap={cap} planned={planned} materialized={total} max_batch={max_batch} \
             rss[before={}MB after_prepare={}MB after_materialize={}MB]",
            mb(rss_before),
            mb(rss_after_prepare),
            mb(rss_after),
        );

        assert_eq!(total, planned, "every planned link materialized once");
        assert!(
            max_batch <= batch.get(),
            "materialized {max_batch} links at once, exceeding batch {}",
            batch.get()
        );
        if let (Some(before), Some(after)) = (rss_before, rss_after) {
            let growth_mb = after.saturating_sub(before) / 1024;
            assert!(
                growth_mb < 2048,
                "RSS grew {growth_mb}MB planning+materializing real data"
            );
        }
    }
}
