//! Solana `LiteSVM` fuzz/PoC runtime + agent-loop orchestration (`SolanaHarnessGenerator`).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use color_eyre::eyre::{Result, WrapErr};
use knowdit_kg_model::ExtractedSemantic;
use knowdit_repo_model::{
    CodeGenCore, CodeGenRecord, CodeGenStatus, CoverageEntry, HarnessRunRecord,
    HistoricalSemanticRecord, LinkKey, MatchStrength, RepoDatabase, SemanticMatchSet,
};
use llmy::agent::StepResult;
use llmy::agent::tool::ToolBox;
use llmy::agent::tools::memory::AgentMemoryContext;
use llmy::client::client::LLM;
use llmy::harness::Agent;
use llmy::harness::memory::AgentMemorySystemPromptCriteria;
use tokio::task::JoinSet;

use super::runner::{SolanaBackend, SolanaRunner};
use super::tools::{ListSolanaTestFilesTool, RunSolanaTool, WriteSolanaHarnessFileTool};
use super::utils::{
    SolanaAttemptHandle, SolanaConventions, SolanaFuzzOptions, SolanaLoopOutcome,
    next_solana_attempt_id,
};
use super::{harness as handler_mode, poc as poc_mode};
use crate::harness::solidity::{
    CodegenRegenInMemory, FuzzOneOutcome, FuzzOutcome, HarnessKind, LinkFuzzOutcome, RegenOutcome,
};
use crate::spec::{
    LinkInput, LookupCallGraphTool, LookupStateVariableXrefsTool, ProjectIndex,
    ReadContractSourceTool, ReadFunctionSourceTool, spec_memory_criteria,
};
use crate::types::AuditSpecification;

pub(super) struct SolanaFuzzRuntime {
    pub(super) project_index: Arc<ProjectIndex>,
    pub(super) extracted_by_id: BTreeMap<i32, ExtractedSemantic>,
    pub(super) historical_by_id: BTreeMap<i32, HistoricalSemanticRecord>,
    pub(super) match_strength_by_pair: BTreeMap<(i32, i32), MatchStrength>,
    pub(super) harness_dir: PathBuf,
    pub(super) solana_runner: SolanaRunner,
    pub(super) conventions: Arc<SolanaConventions>,
}

impl SolanaFuzzRuntime {
    pub(super) async fn new(
        repo: &RepoDatabase,
        options: &SolanaFuzzOptions,
        backend: &SolanaBackend,
    ) -> Result<Self> {
        let match_set = repo
            .load_semantic_match_results()
            .await
            .wrap_err("failed to load Knowledge Mapper output")?;
        Self::from_match_set(repo, options, backend, match_set, None).await
    }

    pub(super) async fn new_for_link(
        repo: &RepoDatabase,
        options: &SolanaFuzzOptions,
        backend: &SolanaBackend,
        extract_id: i32,
        historical_id: i32,
    ) -> Result<Self> {
        let match_set = repo
            .load_semantic_match_results_for_link(extract_id, historical_id)
            .await
            .wrap_err("failed to load Knowledge Mapper output for one link")?;
        Self::from_match_set(repo, options, backend, match_set, Some(extract_id)).await
    }

    async fn from_match_set(
        repo: &RepoDatabase,
        options: &SolanaFuzzOptions,
        backend: &SolanaBackend,
        match_set: SemanticMatchSet,
        only_extract: Option<i32>,
    ) -> Result<Self> {
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

        let extracted = repo
            .load_project_semantics()
            .await
            .wrap_err("failed to load extracted project semantics")?;
        let mut extracted_by_id: BTreeMap<i32, ExtractedSemantic> = extracted
            .into_iter()
            .enumerate()
            .map(|(i, sem)| ((i as i32) + 1, sem))
            .collect();
        if let Some(keep) = only_extract {
            extracted_by_id.retain(|id, _| *id == keep);
        }

        let SemanticMatchSet {
            historicals,
            matches,
        } = match_set;
        let historical_by_id: BTreeMap<i32, HistoricalSemanticRecord> = historicals
            .into_iter()
            .map(|rec| (rec.semantic.id, rec))
            .collect();
        let match_strength_by_pair: BTreeMap<(i32, i32), MatchStrength> = matches
            .iter()
            .map(|m| ((m.extract_id, m.historical_id), m.strength))
            .collect();

        let harness_dir = options.resolve_harness_dir();
        tokio::fs::create_dir_all(&harness_dir)
            .await
            .wrap_err_with(|| format!("failed to create harness_dir {}", harness_dir.display()))?;
        let harness_dir = std::fs::canonicalize(&harness_dir).unwrap_or(harness_dir);

        let solana_runner = SolanaRunner::new(
            backend.clone(),
            options.repo_root.clone(),
            harness_dir.clone(),
            Duration::from_secs(options.timeout_secs),
        );
        let conventions =
            Arc::new(SolanaConventions::load(&options.repo_root, &project_index).await);

        Ok(Self {
            project_index,
            extracted_by_id,
            historical_by_id,
            match_strength_by_pair,
            harness_dir,
            solana_runner,
            conventions,
        })
    }

    pub(super) fn materialize_link(
        &self,
        extract_id: i32,
        historical_id: i32,
        finding_id: i32,
    ) -> Option<LinkInput> {
        let extract = self.extracted_by_id.get(&extract_id)?;
        let record = self.historical_by_id.get(&historical_id)?;
        let strength = *self
            .match_strength_by_pair
            .get(&(extract_id, historical_id))?;
        LinkInput::materialize_exact(
            LinkKey {
                extract_id,
                historical_id,
                finding_id,
            },
            strength,
            extract,
            record,
        )
    }
}

enum SolanaAttemptOutcome {
    Violation,
    StepsExhausted,
    AgentError(String),
    NeedsRestart,
}

struct SolanaRestartSnapshot {
    restart_count: usize,
    last_filename: Option<String>,
    last_calls: u64,
    last_gate_passed: bool,
}

struct SolanaAgentLoopState {
    kind: HarnessKind,
    spec_id: i32,
    options: SolanaFuzzOptions,
    project_index: Arc<ProjectIndex>,
    spec_arc: Arc<AuditSpecification>,
    callgraph_arc: Arc<knowdit_repo_model::cg::CallGraph>,
    program_id_to_lib: Arc<BTreeMap<String, String>>,
    attempt: SolanaAttemptHandle,
    per_spec_dir: PathBuf,
    solana_runner: SolanaRunner,
    base_system_prompt: String,
    user_prompt: String,
    prior_feedback: Option<String>,
    cache_key_base: String,
    debug_prefix: Option<String>,
    memory: AgentMemoryContext,
    memory_criteria: AgentMemorySystemPromptCriteria,
}

impl SolanaAgentLoopState {
    #[allow(clippy::too_many_arguments)]
    async fn new(
        kind: HarnessKind,
        link: &LinkInput,
        spec: &AuditSpecification,
        spec_id: i32,
        project_index: &Arc<ProjectIndex>,
        options: &SolanaFuzzOptions,
        solana_runner: &SolanaRunner,
        harness_dir: &Path,
        conventions: &SolanaConventions,
        prior_feedback: Option<&str>,
    ) -> Result<Self> {
        let attempt_id = next_solana_attempt_id();
        let per_spec_dir = harness_dir.join(format!(
            "spec_{spec_id}_{}_{}",
            std::process::id(),
            attempt_id
        ));
        solana_runner
            .write_per_spec_crate_scaffold(&per_spec_dir, spec_id)
            .await?;

        let base_system_prompt = match kind {
            HarnessKind::Handler => {
                handler_mode::build_system_prompt(link, spec, spec_id, &per_spec_dir, conventions)
            }
            HarnessKind::Poc => {
                poc_mode::build_system_prompt(link, spec, spec_id, &per_spec_dir, conventions)
            }
        };
        let user_prompt = match kind {
            HarnessKind::Handler => handler_mode::build_user_prompt(spec_id),
            HarnessKind::Poc => poc_mode::build_user_prompt(spec_id),
        };

        let memory = project_index.build_link_memory()?;
        let memory_criteria = spec_memory_criteria();
        let cache_key_base = format!("{}-spec{}", options.cache_key, spec_id);
        let debug_prefix = options
            .debug_prefix
            .as_ref()
            .map(|p| format!("{p}-spec{spec_id}"));

        Ok(Self {
            kind,
            spec_id,
            options: options.clone(),
            project_index: project_index.clone(),
            spec_arc: Arc::new(spec.clone()),
            callgraph_arc: Arc::clone(&project_index.call_graph),
            program_id_to_lib: Arc::new(conventions.program_id_to_lib().clone()),
            attempt: SolanaAttemptHandle::new(),
            per_spec_dir,
            solana_runner: solana_runner.clone(),
            base_system_prompt,
            user_prompt,
            prior_feedback: prior_feedback.map(String::from),
            cache_key_base,
            debug_prefix,
            memory,
            memory_criteria,
        })
    }

    fn build_toolbox(&self) -> ToolBox {
        let mut tools = ToolBox::new();
        tools.add_tool(LookupCallGraphTool {
            project: self.project_index.clone(),
        });
        tools.add_tool(LookupStateVariableXrefsTool {
            project: self.project_index.clone(),
        });
        tools.add_tool(ReadContractSourceTool {
            project: self.project_index.clone(),
        });
        tools.add_tool(ReadFunctionSourceTool {
            project: self.project_index.clone(),
        });
        tools.add_tool(WriteSolanaHarnessFileTool {
            attempt: self.attempt.clone(),
            per_spec_tests_dir: self.per_spec_dir.join("tests"),
        });
        tools.add_tool(RunSolanaTool {
            attempt: self.attempt.clone(),
            runner: self.solana_runner.clone(),
            per_spec_dir: self.per_spec_dir.clone(),
            test_runs: self.options.test_runs,
            coverage_runs: self.options.coverage_runs,
            spec: self.spec_arc.clone(),
            callgraph: self.callgraph_arc.clone(),
            program_id_to_lib: self.program_id_to_lib.clone(),
            gate2_fidelity_threshold: self.options.gate2_fidelity_threshold,
            lcov_path: self.per_spec_dir.join("lcov.info"),
        });
        tools.add_tool(ListSolanaTestFilesTool {
            repo_root: self.options.repo_root.clone(),
            max_depth: self.options.list_test_files_max_depth,
        });
        tools.add_tool(llmy::agent::tools::files::ReadFileTool::new(
            self.options.repo_root.clone(),
        ));
        tools
    }

    fn compose_system_prompt(
        &self,
        restart_count: usize,
        last_filename: Option<&str>,
        last_calls: u64,
        last_gate_passed: bool,
    ) -> String {
        let mut sys = self.base_system_prompt.clone();
        if let Some(feedback) = self.prior_feedback.as_deref() {
            sys.push_str("\n\n## Prior attempt feedback (this is a regen)\n\n");
            sys.push_str(feedback);
            sys.push_str(
                "\n\nFix the specific issue above. Inspect the Solana instruction source and \
                 `#[derive(Accounts)]` constraints carefully and model the real account topology.",
            );
        }
        if restart_count > 0 {
            let bootstrap = match self.kind {
                HarnessKind::Handler => handler_mode::build_restart_bootstrap(
                    last_filename,
                    last_calls,
                    last_gate_passed,
                    restart_count - 1,
                ),
                HarnessKind::Poc => poc_mode::build_restart_bootstrap(
                    last_filename,
                    last_calls,
                    last_gate_passed,
                    restart_count - 1,
                ),
            };
            sys.push_str(&bootstrap);
        }
        sys
    }

    fn cache_key_for(&self, restart_count: usize) -> String {
        if restart_count == 0 {
            self.cache_key_base.clone()
        } else {
            format!("{}-restart{restart_count}", self.cache_key_base)
        }
    }

    async fn drive(self, llm: &LLM) -> Result<LinkFuzzOutcome> {
        let mut total_steps = 0usize;
        let (loop_outcome, last_error) =
            self.spawn_attempts_until_done(llm, &mut total_steps).await;
        self.finalize(loop_outcome, total_steps, last_error).await
    }

    async fn spawn_attempts_until_done(
        &self,
        llm: &LLM,
        total_steps: &mut usize,
    ) -> (SolanaLoopOutcome, Option<String>) {
        loop {
            let snap = self.snapshot_restart_state().await;
            let sys = self.compose_system_prompt(
                snap.restart_count,
                snap.last_filename.as_deref(),
                snap.last_calls,
                snap.last_gate_passed,
            );
            let mut agent = Agent::with_memory(
                sys,
                self.build_toolbox(),
                self.cache_key_for(snap.restart_count),
                &self.memory,
                &self.memory_criteria,
            )
            .await;

            match self
                .run_one_attempt(llm, &mut agent, snap.restart_count, total_steps)
                .await
            {
                SolanaAttemptOutcome::Violation => return (SolanaLoopOutcome::Violation, None),
                SolanaAttemptOutcome::StepsExhausted => {
                    return (SolanaLoopOutcome::StepsExhausted, None);
                }
                SolanaAttemptOutcome::AgentError(e) => {
                    return (SolanaLoopOutcome::AgentError, Some(e));
                }
                SolanaAttemptOutcome::NeedsRestart => {
                    let mut st = self.attempt.0.lock().await;
                    st.restarts += 1;
                    if st.restarts > self.options.max_restarts {
                        return (SolanaLoopOutcome::RestartsExhausted, None);
                    }
                }
            }
        }
    }

    async fn snapshot_restart_state(&self) -> SolanaRestartSnapshot {
        let st = self.attempt.0.lock().await;
        SolanaRestartSnapshot {
            restart_count: st.restarts,
            last_filename: st.harness_filename.clone(),
            last_calls: st.last_test_calls,
            last_gate_passed: st.last_gate_passed,
        }
    }

    async fn run_one_attempt(
        &self,
        llm: &LLM,
        agent: &mut Agent,
        restart_count: usize,
        total_steps: &mut usize,
    ) -> SolanaAttemptOutcome {
        let mut truncation = knowdit_kg::agent_retry::TruncationRetry::default();
        let step_res = truncation
            .first_step_with_user(
                agent,
                self.user_prompt.clone(),
                llm,
                self.debug_prefix.as_deref(),
                self.options.llm_settings.clone(),
            )
            .await;
        let mut step = match step_res {
            Ok(s) => s,
            Err(e) => {
                return SolanaAttemptOutcome::AgentError(format!(
                    "initial step (restart {restart_count}): {e}"
                ));
            }
        };
        *total_steps += 1;

        loop {
            if self.attempt.0.lock().await.violation_observed {
                return SolanaAttemptOutcome::Violation;
            }
            if *total_steps >= self.options.max_agent_steps {
                tracing::warn!(
                    "spec {}: Solana harness step budget exhausted (max_agent_steps={})",
                    self.spec_id,
                    self.options.max_agent_steps
                );
                return SolanaAttemptOutcome::StepsExhausted;
            }
            if matches!(step, StepResult::Stop(_)) {
                tracing::warn!(
                    "spec {}: Solana harness agent stopped with no tool call at step {}; restarting",
                    self.spec_id,
                    *total_steps
                );
                return SolanaAttemptOutcome::NeedsRestart;
            }
            if let Some(threshold) = self.options.window_restart_threshold_tokens
                && let Some(used) = agent.approx_context_tokens(&llm.model.config)
                && used >= threshold
            {
                tracing::info!(
                    "spec {}: Solana harness window threshold reached (used={used}, threshold={threshold}); restarting",
                    self.spec_id
                );
                return SolanaAttemptOutcome::NeedsRestart;
            }
            let next = truncation
                .step(
                    agent,
                    llm,
                    self.debug_prefix.as_deref(),
                    self.options.llm_settings.clone(),
                )
                .await;
            step = match next {
                Ok(s) => s,
                Err(e) => {
                    let hint = if knowdit_kg::agent_retry::is_retryable(&e) {
                        format!(
                            " ({})",
                            knowdit_kg::agent_retry::exhaustion_reason(
                                &e,
                                truncation.consecutive_failures(),
                            )
                        )
                    } else {
                        String::new()
                    };
                    return SolanaAttemptOutcome::AgentError(format!(
                        "step {}: {e}{hint}",
                        *total_steps
                    ));
                }
            };
            *total_steps += 1;
        }
    }

    async fn finalize(
        self,
        loop_outcome: SolanaLoopOutcome,
        total_steps: usize,
        last_error: Option<String>,
    ) -> Result<LinkFuzzOutcome> {
        let snapshot = self.attempt.snapshot().await;
        let has_runs = !snapshot.runs.is_empty();
        let (code_gen_status, final_reason) = match &loop_outcome {
            SolanaLoopOutcome::Violation => (
                CodeGenStatus::Completed,
                format!(
                    "violation observed after {total_steps} step(s), {} restart(s)",
                    snapshot.restarts
                ),
            ),
            SolanaLoopOutcome::StepsExhausted => (
                CodeGenStatus::StepsExhausted,
                format!(
                    "step budget {} exhausted (restarts={}, runs={})",
                    self.options.max_agent_steps,
                    snapshot.restarts,
                    snapshot.runs.len()
                ),
            ),
            SolanaLoopOutcome::RestartsExhausted => (
                if has_runs {
                    CodeGenStatus::Completed
                } else {
                    CodeGenStatus::Abandoned
                },
                format!(
                    "restart budget {} exhausted (steps={total_steps}, runs={})",
                    self.options.max_restarts,
                    snapshot.runs.len()
                ),
            ),
            SolanaLoopOutcome::AgentError => (
                CodeGenStatus::AgentError,
                last_error.unwrap_or_else(|| "agent error".to_string()),
            ),
        };

        let (runs, coverage_per_run): (Vec<HarnessRunRecord>, Vec<Vec<CoverageEntry>>) = {
            let mut rs = Vec::with_capacity(snapshot.runs.len());
            let mut covs = Vec::with_capacity(snapshot.runs.len());
            for r in &snapshot.runs {
                rs.push(r.record.clone());
                covs.push(
                    r.coverage_idx
                        .and_then(|i| snapshot.coverage.get(i).cloned())
                        .unwrap_or_default(),
                );
            }
            (rs, covs)
        };

        let any_violation = runs.iter().any(|r| r.violated);
        let harness_relative = snapshot
            .harness_filename
            .as_deref()
            .map(|f| {
                let abs = self.per_spec_dir.join("tests").join(f);
                abs.strip_prefix(&self.options.repo_root)
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| abs.to_string_lossy().into_owned())
            })
            .unwrap_or_default();

        let code_gen = CodeGenRecord {
            core: CodeGenCore {
                spec_id: self.spec_id,
                harness_relative_path: harness_relative,
                harness_source: snapshot.harness_source,
                status: code_gen_status,
                final_reason,
                agent_steps: total_steps as i32,
            },
            runs,
        };

        Ok(LinkFuzzOutcome {
            spec_id: self.spec_id,
            code_gen,
            coverage_per_run,
            any_violation,
        })
    }
}

/// Top-level Solana `LiteSVM` fuzz/PoC generator mirroring `SolidityHarnessGenerator`.
pub struct SolanaHarnessGenerator {
    repo: RepoDatabase,
    llm: LLM,
    options: SolanaFuzzOptions,
    backend: SolanaBackend,
    kind: HarnessKind,
}

impl SolanaHarnessGenerator {
    pub fn new(
        kind: HarnessKind,
        repo: &RepoDatabase,
        llm: &LLM,
        options: &SolanaFuzzOptions,
        backend: SolanaBackend,
    ) -> Self {
        Self {
            repo: repo.clone(),
            llm: llm.clone(),
            options: options.clone(),
            backend,
            kind,
        }
    }

    pub async fn run(&self) -> Result<FuzzOutcome> {
        let runtime = SolanaFuzzRuntime::new(&self.repo, &self.options, &self.backend).await?;
        let specs = self
            .repo
            .load_specifications()
            .await
            .wrap_err("failed to load specifications")?;

        if self.options.regenerate {
            self.repo
                .clear_fuzz_tables()
                .await
                .wrap_err("failed to clear fuzz tables for --regenerate")?;
        }
        let already_done: BTreeSet<i32> = if self.options.regenerate {
            BTreeSet::new()
        } else {
            self.repo
                .loaded_completed_code_gen_spec_ids()
                .await
                .wrap_err("failed to load already-fuzzed spec ids")?
                .into_iter()
                .collect()
        };

        let mut pending: Vec<PendingSolanaSpec> = Vec::new();
        let mut skipped_resumed = 0usize;
        let mut skipped_no_link = 0usize;
        let mut skipped_parse = 0usize;

        for s in specs {
            if already_done.contains(&s.id) {
                skipped_resumed += 1;
                continue;
            }
            let spec: AuditSpecification = match serde_json::from_str(&s.specification_json) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("Skipping spec id={}: JSON parse failed: {e}", s.id);
                    skipped_parse += 1;
                    continue;
                }
            };
            pending.push(PendingSolanaSpec {
                spec_id: s.id,
                semantic_id: s.semantic_id,
                historical_id: s.historical_id,
                finding_id: s.finding_id,
                spec,
            });
        }

        pending = round_robin_pending(pending);
        if let Some(cap) = (self.options.max_specs > 0).then_some(self.options.max_specs)
            && pending.len() > cap
        {
            pending.truncate(cap);
        }

        let mut tasks: Vec<SolanaFuzzTask> = Vec::with_capacity(pending.len());
        for p in pending {
            let Some(link) =
                runtime.materialize_link(p.semantic_id, p.historical_id, p.finding_id)
            else {
                skipped_no_link += 1;
                continue;
            };
            tasks.push(SolanaFuzzTask {
                spec_id: p.spec_id,
                semantic_id: p.semantic_id,
                link,
                spec: p.spec,
            });
        }
        if skipped_resumed > 0 || skipped_no_link > 0 || skipped_parse > 0 {
            tracing::info!(
                "Solana Fuzz: skipping {skipped_resumed} resumed + {skipped_no_link} no-link + {skipped_parse} parse-error spec(s)"
            );
        }

        let total = tasks.len();
        if total == 0 {
            return Ok(FuzzOutcome {
                skipped_resumed,
                ..FuzzOutcome::default()
            });
        }
        tracing::info!(
            "Solana Fuzz: {total} spec(s) across {} semantic(s) (concurrency={})",
            tasks
                .iter()
                .map(|t| t.semantic_id)
                .collect::<BTreeSet<_>>()
                .len(),
            self.options.concurrency.max(1),
        );

        let outcome = self
            .dispatch(
                tasks,
                runtime.project_index.clone(),
                &runtime.solana_runner,
                &runtime.harness_dir,
                total,
                runtime.conventions.clone(),
            )
            .await?;
        Ok(FuzzOutcome {
            skipped_resumed,
            ..outcome
        })
    }

    pub async fn fuzz_one_existing_spec(&self, spec_id: i32) -> Result<Option<FuzzOneOutcome>> {
        if !self.options.regenerate {
            let already_done = self
                .repo
                .loaded_completed_code_gen_spec_ids()
                .await
                .wrap_err("failed to load already-fuzzed spec ids")?;
            if already_done.contains(&spec_id) {
                tracing::info!(spec_id, "fuzz_one_existing_spec: skipping resumed Solana spec");
                return Ok(None);
            }
        }
        let (link, spec, runtime) = self.load_spec_link(spec_id).await?;
        let lo = SolanaAgentLoopState::new(
            self.kind,
            &link,
            &spec,
            spec_id,
            &runtime.project_index,
            &self.options,
            &runtime.solana_runner,
            &runtime.harness_dir,
            &runtime.conventions,
            None,
        )
        .await?
        .drive(&self.llm)
        .await?;

        let status = lo.code_gen.core.status;
        let any_violation = lo.any_violation;
        let (code_gen_id, run_ids) = self
            .repo
            .write_code_gen_with_runs(&lo.code_gen, &lo.coverage_per_run)
            .await
            .wrap_err_with(|| format!("failed to persist Solana code_gen for spec_id={spec_id}"))?;
        Ok(Some(FuzzOneOutcome {
            spec_id,
            code_gen_id,
            run_ids,
            status,
            any_violation,
        }))
    }

    pub async fn regen_one_spec(
        &self,
        spec_id: i32,
        prior_feedback: String,
    ) -> Result<RegenOutcome> {
        let (link, spec, runtime) = self.load_spec_link(spec_id).await?;
        let lo = SolanaAgentLoopState::new(
            self.kind,
            &link,
            &spec,
            spec_id,
            &runtime.project_index,
            &self.options,
            &runtime.solana_runner,
            &runtime.harness_dir,
            &runtime.conventions,
            Some(&prior_feedback),
        )
        .await?
        .drive(&self.llm)
        .await?;

        let (new_code_id, _run_ids) = self
            .repo
            .write_code_gen_with_runs(&lo.code_gen, &lo.coverage_per_run)
            .await
            .wrap_err_with(|| {
                format!("failed to persist Solana regen code_gen for spec_id={spec_id}")
            })?;
        Ok(RegenOutcome {
            new_code_gen_id: new_code_id,
            status: lo.code_gen.core.status,
            any_violation: lo.any_violation,
        })
    }

    pub async fn regen_codegen_with_explicit_spec(
        &self,
        extract_id: i32,
        historical_id: i32,
        finding_id: i32,
        spec: AuditSpecification,
        synthetic_spec_id: i32,
        prior_feedback: String,
    ) -> Result<CodegenRegenInMemory> {
        let runtime = SolanaFuzzRuntime::new_for_link(
            &self.repo,
            &self.options,
            &self.backend,
            extract_id,
            historical_id,
        )
        .await?;
        let link = runtime
            .materialize_link(extract_id, historical_id, finding_id)
            .ok_or_else(|| {
                color_eyre::eyre::eyre!(
                    "no LinkInput for (extract={extract_id}, historical={historical_id}, finding={finding_id})"
                )
            })?;
        let lo = SolanaAgentLoopState::new(
            self.kind,
            &link,
            &spec,
            synthetic_spec_id,
            &runtime.project_index,
            &self.options,
            &runtime.solana_runner,
            &runtime.harness_dir,
            &runtime.conventions,
            Some(&prior_feedback),
        )
        .await?
        .drive(&self.llm)
        .await?;

        Ok(CodegenRegenInMemory {
            status: lo.code_gen.core.status,
            any_violation: lo.any_violation,
            code_gen: lo.code_gen,
            coverage_per_run: lo.coverage_per_run,
        })
    }

    async fn load_spec_link(
        &self,
        spec_id: i32,
    ) -> Result<(LinkInput, AuditSpecification, SolanaFuzzRuntime)> {
        let specs = self
            .repo
            .load_specifications()
            .await
            .wrap_err("failed to load specifications")?;
        let spec_row = specs.into_iter().find(|s| s.id == spec_id).ok_or_else(|| {
            color_eyre::eyre::eyre!("spec_id={spec_id} not in specification table")
        })?;
        let runtime = SolanaFuzzRuntime::new_for_link(
            &self.repo,
            &self.options,
            &self.backend,
            spec_row.semantic_id,
            spec_row.historical_id,
        )
        .await?;
        let link = runtime
            .materialize_link(
                spec_row.semantic_id,
                spec_row.historical_id,
                spec_row.finding_id,
            )
            .ok_or_else(|| {
                color_eyre::eyre::eyre!(
                    "no LinkInput for spec {spec_id} (extract={}, historical={}, finding={})",
                    spec_row.semantic_id,
                    spec_row.historical_id,
                    spec_row.finding_id
                )
            })?;
        let spec: AuditSpecification = serde_json::from_str(&spec_row.specification_json)
            .wrap_err_with(|| format!("failed to parse specification.json for spec_id={spec_id}"))?;
        Ok((link, spec, runtime))
    }

    async fn dispatch(
        &self,
        tasks: Vec<SolanaFuzzTask>,
        project_index: Arc<ProjectIndex>,
        solana_runner: &SolanaRunner,
        harness_dir: &Path,
        total: usize,
        conventions: Arc<SolanaConventions>,
    ) -> Result<FuzzOutcome> {
        let mut outcome = FuzzOutcome::default();
        let concurrency = self.options.concurrency.max(1);

        if concurrency == 1 {
            for (idx, task) in tasks.into_iter().enumerate() {
                let label = task_label(&task, idx, total);
                tracing::info!("Solana Fuzz starting {label}");
                let res = match SolanaAgentLoopState::new(
                    self.kind,
                    &task.link,
                    &task.spec,
                    task.spec_id,
                    &project_index,
                    &self.options,
                    solana_runner,
                    harness_dir,
                    &conventions,
                    None,
                )
                .await
                {
                    Ok(state) => state.drive(&self.llm).await,
                    Err(e) => Err(e),
                };
                self.persist(res, &label, &mut outcome).await;
            }
            return Ok(outcome);
        }

        let (tx, rx) = async_channel::bounded::<(usize, SolanaFuzzTask)>(tasks.len() + 1);
        let (out_tx, mut out_rx) =
            tokio::sync::mpsc::channel::<(Result<LinkFuzzOutcome>, String)>(concurrency + 1);

        let mut workers = JoinSet::new();
        for _ in 0..concurrency {
            let rx = rx.clone();
            let out = out_tx.clone();
            let project_index = project_index.clone();
            let llm = self.llm.clone();
            let options = self.options.clone();
            let solana_runner = solana_runner.clone();
            let harness_dir = harness_dir.to_path_buf();
            let conv = conventions.clone();
            let kind = self.kind;
            workers.spawn(async move {
                while let Ok((idx, task)) = rx.recv().await {
                    let label = task_label(&task, idx, total);
                    tracing::info!("Solana Fuzz starting {label}");
                    let res = match SolanaAgentLoopState::new(
                        kind,
                        &task.link,
                        &task.spec,
                        task.spec_id,
                        &project_index,
                        &options,
                        &solana_runner,
                        &harness_dir,
                        &conv,
                        None,
                    )
                    .await
                    {
                        Ok(state) => state.drive(&llm).await,
                        Err(e) => Err(e),
                    };
                    let _ = out.send((res, label)).await;
                }
            });
        }
        drop(out_tx);
        drop(rx);

        for (idx, task) in tasks.into_iter().enumerate() {
            let _ = tx.send((idx, task)).await;
        }
        drop(tx);

        while let Some((res, label)) = out_rx.recv().await {
            self.persist(res, &label, &mut outcome).await;
        }
        while let Some(joined) = workers.join_next().await {
            if let Err(err) = joined {
                tracing::error!("Solana fuzz worker task failed: {err:#}");
            }
        }
        Ok(outcome)
    }

    async fn persist(&self, res: Result<LinkFuzzOutcome>, label: &str, outcome: &mut FuzzOutcome) {
        outcome.processed_count += 1;
        let lo = match res {
            Ok(lo) => lo,
            Err(e) => {
                tracing::error!("{label}: Solana harness agent failed: {e:#}");
                outcome.error_count += 1;
                return;
            }
        };
        let success = matches!(lo.code_gen.core.status, CodeGenStatus::Completed);
        let abandoned = matches!(
            lo.code_gen.core.status,
            CodeGenStatus::Abandoned | CodeGenStatus::StepsExhausted
        );
        let violated = lo.any_violation;
        match self
            .repo
            .write_code_gen_with_runs(&lo.code_gen, &lo.coverage_per_run)
            .await
        {
            Ok((code_id, run_ids)) => {
                tracing::info!(
                    "{label}: persisted Solana code_gen={code_id} runs={run_ids:?} status={} violated={violated}",
                    lo.code_gen.core.status
                );
            }
            Err(e) => {
                tracing::error!("{label}: persist failed: {e:#}");
                outcome.error_count += 1;
                return;
            }
        }
        if success {
            outcome.completed_count += 1;
        } else if abandoned {
            outcome.abandoned_count += 1;
        }
        if violated {
            outcome.violated_count += 1;
        }
    }
}

fn task_label(task: &SolanaFuzzTask, idx: usize, total: usize) -> String {
    format!(
        "spec {}/{} id={} extract={} finding={}",
        idx + 1,
        total,
        task.spec_id,
        task.link.extract_id,
        task.link.finding_id
    )
}

struct SolanaFuzzTask {
    spec_id: i32,
    semantic_id: i32,
    link: LinkInput,
    spec: AuditSpecification,
}

struct PendingSolanaSpec {
    spec_id: i32,
    semantic_id: i32,
    historical_id: i32,
    finding_id: i32,
    spec: AuditSpecification,
}

fn round_robin_pending(specs: Vec<PendingSolanaSpec>) -> Vec<PendingSolanaSpec> {
    let mut by_sem: BTreeMap<i32, VecDeque<PendingSolanaSpec>> = BTreeMap::new();
    for s in specs {
        by_sem.entry(s.semantic_id).or_default().push_back(s);
    }
    let total: usize = by_sem.values().map(VecDeque::len).sum();
    let mut out = Vec::with_capacity(total);
    while !by_sem.is_empty() {
        let keys: Vec<i32> = by_sem.keys().copied().collect();
        for k in keys {
            if let Some(q) = by_sem.get_mut(&k) {
                if let Some(s) = q.pop_front() {
                    out.push(s);
                }
                if q.is_empty() {
                    by_sem.remove(&k);
                }
            }
        }
    }
    out
}
