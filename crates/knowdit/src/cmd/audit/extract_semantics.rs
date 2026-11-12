use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use clap::Args;
use color_eyre::eyre::Result;
use knowdit_audit::spec::{ProjectIndex, StatefulEntrypoint, is_billing_exhausted};
use knowdit_kg::agent_runner::AgentRunOptions;
use knowdit_kg::agents::SemanticChunkExtractor;
use knowdit_kg::learn::ExtractedSemantic;
use knowdit_kg::prompts;
use knowdit_kg_model::{ExtractedFunction, category::DeFiCategory};
use knowdit_project::ProjectData;
use knowdit_repo_model::RepoDatabase;
use llmy::client::client::LLM;
use llmy::client::context::TokenCursor;

use crate::cli::{DatabaseArgs, LoadedRepoDatabase, ProjectArgs};

/// Knobs for the extract-semantics phase. Flattened by both the CLI
/// `agentic extract-semantics` subcommand and `workflow autoloop` / `streamloop`,
/// so every default value is centralized here via `default_value_t`.
#[derive(Args, Clone, Debug)]
pub struct ExtractSemanticsSharedArgs {
    /// Optional override for the per-chunk *input* token budget passed
    /// to `extract_semantics`. Defaults to ~80% of the model's
    /// `max_input`.
    #[arg(long = "extract-chunk-input-budget")]
    pub extract_chunk_input_budget: Option<usize>,

    /// Hard cap on agent steps for the categorize / extract-semantic
    /// agents. Each `emit_*` tool call counts as one step.
    #[arg(long = "extract-max-agent-steps", default_value_t = 60)]
    pub extract_max_agent_steps: usize,

    /// Fraction (0,1] of the model's context window a single categorize /
    /// extract prompt may fill. Lower = more, smaller prompts with sharper
    /// attention. Prefixed to stay collision-free when co-flattened.
    #[arg(
        long = "extract-context-window-utilization",
        default_value_t = knowdit_kg::learn::DEFAULT_CONTEXT_WINDOW_UTILIZATION
    )]
    pub extract_context_window_utilization: f64,

    /// Disable the post-extraction static stateful-entrypoint coverage pass
    /// (both targeted LLM gap-fill and deterministic per-contract fallback).
    #[arg(long = "no-extract-orphan-pass", default_value_t = false)]
    pub no_extract_orphan_pass: bool,

    /// Disable only Step B (deterministic per-contract fallback semantics) of
    /// the post-extraction orphan-entrypoint pass while keeping Step A's
    /// targeted LLM gap-fill enabled.
    #[arg(long = "no-extract-orphan-fallback", default_value_t = false)]
    pub no_extract_orphan_fallback: bool,
}

impl Default for ExtractSemanticsSharedArgs {
    fn default() -> Self {
        Self {
            extract_chunk_input_budget: None,
            extract_max_agent_steps: 60,
            extract_context_window_utilization: knowdit_kg::learn::DEFAULT_CONTEXT_WINDOW_UTILIZATION,
            no_extract_orphan_pass: false,
            no_extract_orphan_fallback: false,
        }
    }
}

#[derive(Args)]
pub struct ExtractSemanticsArgs {
    #[command(flatten)]
    pub project: ProjectArgs,

    #[command(flatten)]
    pub db: DatabaseArgs,

    #[command(flatten)]
    pub shared: ExtractSemanticsSharedArgs,

    /// Optional JSON dump of the extracted semantics for human
    /// inspection. The canonical source of truth is still the project
    /// DB; omit this flag when you don't need the file. CLI-only —
    /// the autoloop manages its own dumps.
    #[arg(long)]
    pub out: Option<PathBuf>,
}

impl ExtractSemanticsArgs {
    /// CLI entry: open the repo DB + load the project corpus from
    /// disk, then delegate to [`Self::extract_semantics`].
    pub async fn run(self, llm: &LLM) -> Result<()> {
        let project = self.project.to_project_data().await?;
        let LoadedRepoDatabase { repo, .. } = self
            .project
            .to_repo_database(self.db.database_path.clone(), self.db.variant_render_cap)
            .await?;
        let semantics = Self::extract_semantics(&repo, llm, &project, &self.shared).await?;
        if let Some(out) = self.out.as_ref() {
            if let Some(parent) = out.parent().filter(|parent| !parent.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(out, serde_json::to_string_pretty(&semantics)?)?;
            tracing::info!("Semantic specs written to {}", out.display());
        }
        Ok(())
    }

    /// In-process entry: load already-extracted semantics from the
    /// repo DB, or run the LLM-driven extractor when the table is
    /// empty, followed by the U2 stateful-entrypoint orphan coverage pass.
    pub async fn extract_semantics(
        repo: &RepoDatabase,
        llm: &LLM,
        project: &ProjectData,
        shared: &ExtractSemanticsSharedArgs,
    ) -> Result<Vec<ExtractedSemantic>> {
        let agent_options = AgentRunOptions::new(shared.extract_max_agent_steps)
            .with_context_window_utilization(shared.extract_context_window_utilization);
        OrphanCoveragePass {
            extract_chunk_input_budget: shared.extract_chunk_input_budget,
            no_orphan_pass: shared.no_extract_orphan_pass,
            no_orphan_fallback: shared.no_extract_orphan_fallback,
        }
        .load_or_extract(repo, llm, project, &agent_options)
        .await
    }
}

/// Legacy entrypoint retained for `knowdit solidity cg`, where the static call
/// graph does not yet exist (`CallGraph.contracts` is empty, so the orphan pass
/// cleanly skips).
pub(crate) async fn load_or_extract_project_semantics(
    repo: &RepoDatabase,
    llm: &LLM,
    project: &ProjectData,
    extract_chunk_input_budget: Option<usize>,
    agent_options: &AgentRunOptions,
) -> Result<Vec<ExtractedSemantic>> {
    OrphanCoveragePass {
        extract_chunk_input_budget,
        no_orphan_pass: false,
        no_orphan_fallback: false,
    }
    .load_or_extract(repo, llm, project, agent_options)
    .await
}

/// Post-extraction stateful external entrypoint coverage pass (U2).
///
/// Ensures every in-scope externally callable state-mutating entrypoint in
/// `CallGraph` + `StorageGraph` belongs to at least one `ExtractedSemantic`
/// before Knowledge Mapper runs, while preserving `project_semantic.id`
/// foreign-key invariants on resumed databases.
pub(crate) struct OrphanCoveragePass {
    pub extract_chunk_input_budget: Option<usize>,
    pub no_orphan_pass: bool,
    pub no_orphan_fallback: bool,
}

impl OrphanCoveragePass {
    pub(crate) async fn load_or_extract(
        &self,
        repo: &RepoDatabase,
        llm: &LLM,
        project: &ProjectData,
        agent_options: &AgentRunOptions,
    ) -> Result<Vec<ExtractedSemantic>> {
        let existing = repo.load_project_semantics().await?;
        if !existing.is_empty() {
            tracing::info!(
                "Loaded {} project semantics for {} from project database",
                existing.len(),
                project.display_id()
            );
            return self
                .reconcile_resumed_semantics(repo, Some(llm), project, existing, agent_options)
                .await;
        }

        tracing::info!(
            "No project semantics found for {}; extracting semantics before analysis",
            project.display_id()
        );
        let legacy = knowdit_kg::project_loader::ProjectData::from_project_view(project);
        let extract = legacy
            .categorize_and_extract_semantics(
                llm,
                agent_options,
                self.extract_chunk_input_budget,
                None,
                false,
            )
            .await?;

        let mut semantics = extract.semantics;
        if !self.no_orphan_pass {
            let gap = self
                .compute_gap_semantics(
                    repo,
                    Some(llm),
                    project,
                    &semantics,
                    &extract.categories,
                    agent_options,
                )
                .await?;
            if !gap.is_empty() {
                Self::merge_semantics_in_place(&mut semantics, gap);
            }
        }

        repo.replace_project_semantics(&semantics).await?;
        tracing::info!(
            "Wrote {} project semantics for {} to project database",
            semantics.len(),
            project.display_id()
        );
        Ok(semantics)
    }

    /// Reconcile an already-populated `project_semantic` table against the
    /// static call/storage graph:
    /// - If `semantic_matched` already has rows, log any uncovered entrypoints
    ///   without mutating locked `extract_id` rows.
    /// - If `semantic_matched` is empty, append newly synthesized gap semantics
    ///   via [`RepoDatabase::append_project_semantics`] preserving IDs `1..=N`.
    pub(crate) async fn reconcile_resumed_semantics(
        &self,
        repo: &RepoDatabase,
        llm: Option<&LLM>,
        project: &ProjectData,
        mut existing: Vec<ExtractedSemantic>,
        agent_options: &AgentRunOptions,
    ) -> Result<Vec<ExtractedSemantic>> {
        if self.no_orphan_pass {
            return Ok(existing);
        }
        let Some((project_index, uncovered)) =
            Self::find_uncovered_entrypoints(repo, project, &existing).await?
        else {
            return Ok(existing);
        };
        if uncovered.is_empty() {
            return Ok(existing);
        }

        if repo.has_semantic_matches().await? {
            tracing::warn!(
                "Orphan entrypoint pass: {} stateful external entrypoint(s) in {} are not covered by project_semantic, \
                 but semantic_matched is already populated — skipping mutation to preserve locked extract_id foreign keys",
                uncovered.len(),
                project.display_id(),
            );
            return Ok(existing);
        }

        let categories = Self::inferred_categories(&existing, &[]);
        let gap = self
            .run_two_step_gap_closure(
                &project_index,
                uncovered,
                llm,
                project,
                &existing,
                &categories,
                agent_options,
            )
            .await?;
        if gap.is_empty() {
            return Ok(existing);
        }

        // Preserve existing rows 1..=N untouched; disambiguate any colliding
        // semantic names so appended rows stay distinct.
        let existing_names: BTreeSet<String> = existing
            .iter()
            .map(|s| s.name.trim().to_lowercase())
            .collect();
        let mut to_append = Vec::with_capacity(gap.len());
        for mut sem in gap {
            if existing_names.contains(&sem.name.trim().to_lowercase()) {
                sem.name = format!("{} (Orphan Entrypoints)", sem.name.trim());
            }
            to_append.push(sem);
        }

        repo.append_project_semantics(&to_append).await?;
        tracing::info!(
            "Appended {} orphan-gap semantic(s) for {} to project database (preserving existing IDs 1..={})",
            to_append.len(),
            project.display_id(),
            existing.len(),
        );
        existing.extend(to_append);
        Ok(existing)
    }

    async fn compute_gap_semantics(
        &self,
        repo: &RepoDatabase,
        llm: Option<&LLM>,
        project: &ProjectData,
        existing: &[ExtractedSemantic],
        categories: &[DeFiCategory],
        agent_options: &AgentRunOptions,
    ) -> Result<Vec<ExtractedSemantic>> {
        let Some((project_index, uncovered)) =
            Self::find_uncovered_entrypoints(repo, project, existing).await?
        else {
            return Ok(Vec::new());
        };
        if uncovered.is_empty() {
            tracing::info!(
                "Orphan entrypoint pass: 100% of stateful external entrypoints in {} are already covered",
                project.display_id()
            );
            return Ok(Vec::new());
        }
        let effective_categories = Self::inferred_categories(existing, categories);
        self.run_two_step_gap_closure(
            &project_index,
            uncovered,
            llm,
            project,
            existing,
            &effective_categories,
            agent_options,
        )
        .await
    }

    async fn find_uncovered_entrypoints(
        repo: &RepoDatabase,
        project: &ProjectData,
        semantics: &[ExtractedSemantic],
    ) -> Result<Option<(ProjectIndex, Vec<StatefulEntrypoint>)>> {
        let call_graph = repo.load_call_graph().await?;
        if call_graph.contracts.is_empty() {
            tracing::warn!(
                "Orphan entrypoint pass skipped for {}: CallGraph has 0 contracts (build static call graph before extract-semantics to enable orphan gap closure)",
                project.display_id(),
            );
            return Ok(None);
        }
        let storage_graph = repo.load_storage_graph().await?;
        let inheritance_graph = repo.load_inheritance_graph().await?;
        let project_index = ProjectIndex::build(call_graph, storage_graph, inheritance_graph);

        let in_scope_paths: BTreeSet<PathBuf> = project
            .source_files()
            .iter()
            .map(|f| f.relative_path.clone())
            .collect();
        let uncovered: Vec<StatefulEntrypoint> = project_index
            .uncovered_stateful_entrypoints(semantics)
            .into_iter()
            .filter(|ep| {
                in_scope_paths.is_empty() || in_scope_paths.contains(&ep.relative_file_path)
            })
            .collect();
        Ok(Some((project_index, uncovered)))
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_two_step_gap_closure(
        &self,
        project_index: &ProjectIndex,
        uncovered: Vec<StatefulEntrypoint>,
        llm: Option<&LLM>,
        project: &ProjectData,
        existing: &[ExtractedSemantic],
        categories: &[DeFiCategory],
        agent_options: &AgentRunOptions,
    ) -> Result<Vec<ExtractedSemantic>> {
        tracing::info!(
            "Orphan entrypoint pass: found {} uncovered stateful external entrypoint(s) across {}",
            uncovered.len(),
            project.display_id(),
        );

        let mut gap_semantics: Vec<ExtractedSemantic> = Vec::new();

        // Step A: Targeted LLM gap-fill over uncovered contracts + checklist.
        if let Some(llm) = llm {
            let step_a = self
                .run_targeted_llm_gap_fill(
                    project_index,
                    &uncovered,
                    llm,
                    project,
                    categories,
                    agent_options,
                )
                .await?;
            if !step_a.is_empty() {
                Self::merge_semantics_in_place(&mut gap_semantics, step_a);
            }
        }

        // Re-check which entrypoints remain uncovered after existing + Step A.
        let mut combined_for_diff: Vec<ExtractedSemantic> = existing.to_vec();
        combined_for_diff.extend(gap_semantics.iter().cloned());

        let in_scope_paths: BTreeSet<PathBuf> = project
            .source_files()
            .iter()
            .map(|f| f.relative_path.clone())
            .collect();
        let remaining: Vec<StatefulEntrypoint> = project_index
            .uncovered_stateful_entrypoints(&combined_for_diff)
            .into_iter()
            .filter(|ep| {
                in_scope_paths.is_empty() || in_scope_paths.contains(&ep.relative_file_path)
            })
            .collect();

        if remaining.is_empty() {
            return Ok(gap_semantics);
        }

        if self.no_orphan_fallback {
            tracing::warn!(
                "Orphan entrypoint pass: {} stateful entrypoint(s) remain uncovered after Step A and --no-extract-orphan-fallback is set",
                remaining.len()
            );
            return Ok(gap_semantics);
        }

        // Step B: Deterministic per-contract fallback `ExtractedSemantic`.
        let fallback_category = Self::dominant_category(&combined_for_diff, categories);
        let fallbacks = Self::build_deterministic_fallbacks(&remaining, fallback_category);
        tracing::info!(
            "Orphan entrypoint pass Step B: synthesized {} deterministic per-contract semantic(s) covering {} remaining entrypoint(s)",
            fallbacks.len(),
            remaining.len(),
        );
        Self::merge_semantics_in_place(&mut gap_semantics, fallbacks);
        Ok(gap_semantics)
    }

    async fn run_targeted_llm_gap_fill(
        &self,
        project_index: &ProjectIndex,
        uncovered: &[StatefulEntrypoint],
        llm: &LLM,
        project: &ProjectData,
        categories: &[DeFiCategory],
        agent_options: &AgentRunOptions,
    ) -> Result<Vec<ExtractedSemantic>> {
        let mut by_contract: BTreeMap<i32, Vec<&StatefulEntrypoint>> = BTreeMap::new();
        for ep in uncovered {
            by_contract.entry(ep.contract_id).or_default().push(ep);
        }

        let mut sections: Vec<String> = Vec::with_capacity(by_contract.len());
        for (cid, eps) in by_contract {
            let first = eps[0];
            let mut block = format!(
                "### Uncovered Contract `{}` (`{}`)\n\nUncovered stateful external entrypoints that MUST be assigned to a DeFi Semantic (include exact `contract` and `name` in `functions`):\n",
                first.contract_name,
                first.relative_file_path.display(),
            );
            for ep in &eps {
                let writes_note = if ep.written_state_vars.is_empty() {
                    String::new()
                } else {
                    format!(" [writes: {}]", ep.written_state_vars.join(", "))
                };
                block.push_str(&format!(
                    "- `{}` :: `{}` (`{}`){writes_note}\n",
                    ep.contract_name, ep.function_name, ep.signature
                ));
            }
            if let Some(contract) = project_index.call_graph.contracts.get(&cid)
                && !contract.chunk.content.trim().is_empty()
            {
                block.push_str("\nSource:\n```\n");
                block.push_str(contract.chunk.content.trim_end());
                block.push_str("\n```\n\n");
            }
            sections.push(block);
        }

        let system_prompt = prompts::GENERAL_ROLE_SYSTEM;
        let user_suffix = prompts::extract_semantics_user_suffix(categories, project.language);
        let model = &llm.model;
        let sys_tokens = model.config.count_tokens_lossy(system_prompt);
        let suffix_tokens = model.config.count_tokens_lossy(&user_suffix);
        let total_budget = knowdit_kg_model::context_budget(
            model.config.max_input(),
            agent_options.context_window_utilization,
        );
        let available = total_budget.saturating_sub(sys_tokens + suffix_tokens);
        let chunk_budget = match self.extract_chunk_input_budget {
            Some(cap) => cap.min(available).max(512),
            None => available.max(512),
        };

        let Some(mut cursor) = TokenCursor::new(sections.join("\n"), model.clone()) else {
            return Ok(Vec::new());
        };

        let cache_prefix = Self::sanitize_cache_prefix(project.display_id());
        let mut out: Vec<ExtractedSemantic> = Vec::new();
        let mut chunk_idx = 0usize;
        while let Some(chunk) = cursor.next_chunk(chunk_budget) {
            let user_prompt = format!(
                "## Orphan Entrypoint Gap-Fill Pass\n\n\
                 The following contracts have state-mutating external/public entrypoints that were omitted during primary extraction. \
                 Emit one or more DeFi Semantics covering EVERY listed uncovered entrypoint.\n\n\
                 {chunk}{user_suffix}"
            );
            let label = format!("semantic-orphan-gap-{cache_prefix}-chunk{chunk_idx}");
            let extractor = SemanticChunkExtractor {
                llm: llm.clone(),
                options: agent_options.scoped(&label),
                system_prompt: system_prompt.to_string(),
                user_prompt,
                cache_key: format!("{cache_prefix}-orphan-gap-chunk{chunk_idx}"),
                label,
            };
            match extractor.run().await {
                Ok(chunk_semantics) => {
                    tracing::info!(
                        "Orphan gap-fill chunk {} produced {} semantic(s)",
                        chunk_idx,
                        chunk_semantics.len(),
                    );
                    out.extend(chunk_semantics);
                }
                Err(err) => {
                    let report = color_eyre::eyre::Report::new(err);
                    if is_billing_exhausted(&report) {
                        return Err(report.wrap_err("orphan gap-fill hit billing cap"));
                    }
                    tracing::warn!(
                        "Orphan gap-fill chunk {} failed (falling through to deterministic fallback): {report:#}",
                        chunk_idx
                    );
                    break;
                }
            }
            chunk_idx += 1;
        }
        Ok(out)
    }

    pub(crate) fn build_deterministic_fallbacks(
        remaining: &[StatefulEntrypoint],
        category: DeFiCategory,
    ) -> Vec<ExtractedSemantic> {
        let mut by_contract: BTreeMap<(i32, String), Vec<&StatefulEntrypoint>> = BTreeMap::new();
        for ep in remaining {
            by_contract
                .entry((ep.contract_id, ep.contract_name.clone()))
                .or_default()
                .push(ep);
        }

        let mut fallbacks = Vec::with_capacity(by_contract.len());
        for ((_, contract_name), eps) in by_contract {
            let mut all_writes: BTreeSet<String> = BTreeSet::new();
            let mut sig_summaries: Vec<String> = Vec::with_capacity(eps.len());
            let mut functions: Vec<ExtractedFunction> = Vec::with_capacity(eps.len());

            for ep in eps {
                for w in &ep.written_state_vars {
                    all_writes.insert(w.clone());
                }
                sig_summaries.push(format!("`{}`", ep.signature));
                functions.push(ExtractedFunction {
                    name: ep.function_name.clone(),
                    contract: ep.contract_name.clone(),
                    signature: Some(ep.signature.clone()),
                });
            }

            let writes_prose = if all_writes.is_empty() {
                "executes external/interface state transitions".to_string()
            } else {
                format!(
                    "mutates protocol state variable(s) {}",
                    all_writes
                        .into_iter()
                        .map(|v| format!("`{v}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            };
            let definition = format!(
                "State-mutating external entrypoints exposed on `{contract_name}` that {writes_prose}."
            );
            let description = format!(
                "Deterministic coverage cluster for `{contract_name}` capturing {} externally callable state-mutating entrypoint(s) ({}) which {writes_prose}.",
                functions.len(),
                sig_summaries.join(", "),
            );

            fallbacks.push(ExtractedSemantic {
                name: format!("{contract_name} Stateful Entrypoints"),
                category,
                definition,
                description,
                functions,
            });
        }
        fallbacks
    }

    pub(crate) fn dominant_category(
        semantics: &[ExtractedSemantic],
        categories: &[DeFiCategory],
    ) -> DeFiCategory {
        let mut counts: BTreeMap<String, (usize, DeFiCategory)> = BTreeMap::new();
        for sem in semantics {
            if sem.category != DeFiCategory::Others {
                let entry = counts
                    .entry(sem.category.as_str().to_string())
                    .or_insert((0, sem.category));
                entry.0 += 1;
            }
        }
        if let Some((_, best)) = counts
            .into_values()
            .max_by_key(|(count, cat)| (*count, std::cmp::Reverse(cat.as_str().to_string())))
        {
            return best;
        }
        categories
            .iter()
            .copied()
            .find(|c| *c != DeFiCategory::Others)
            .unwrap_or(DeFiCategory::Others)
    }

    fn inferred_categories(
        semantics: &[ExtractedSemantic],
        explicit: &[DeFiCategory],
    ) -> Vec<DeFiCategory> {
        if !explicit.is_empty() {
            return explicit.to_vec();
        }
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        for sem in semantics {
            if seen.insert(sem.category.as_str()) {
                out.push(sem.category);
            }
        }
        if out.is_empty() {
            out.push(DeFiCategory::Others);
        }
        out
    }

    pub(crate) fn merge_semantics_in_place(
        target: &mut Vec<ExtractedSemantic>,
        additions: Vec<ExtractedSemantic>,
    ) {
        let mut index_by_name: BTreeMap<String, usize> = BTreeMap::new();
        for (idx, sem) in target.iter().enumerate() {
            index_by_name
                .entry(sem.name.trim().to_lowercase())
                .or_insert(idx);
        }
        for sem in additions {
            let key = sem.name.trim().to_lowercase();
            if let Some(&idx) = index_by_name.get(&key) {
                let existing = &mut target[idx];
                for func in sem.functions {
                    let dup = existing.functions.iter().any(|f| {
                        f.name.eq_ignore_ascii_case(&func.name)
                            && f.contract.eq_ignore_ascii_case(&func.contract)
                    });
                    if !dup {
                        existing.functions.push(func);
                    }
                }
                if sem.description.len() > existing.description.len() {
                    existing.description = sem.description;
                    existing.definition = sem.definition;
                }
            } else {
                let idx = target.len();
                index_by_name.insert(key, idx);
                target.push(sem);
            }
        }
    }

    fn sanitize_cache_prefix(value: &str) -> String {
        let mut out = String::new();
        let mut last_dash = false;
        for ch in value.chars() {
            if ch.is_ascii_alphanumeric() {
                out.push(ch.to_ascii_lowercase());
                last_dash = false;
            } else if !last_dash {
                out.push('-');
                last_dash = true;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use knowdit_project::scope::{ProjectScopeContent, ScopedSourceFile};
    use knowdit_repo_model::cg::{CallGraph, Contract, FileChunk, FileLocation, Function};
    use knowdit_repo_model::storage::{ContractVariable, FunctionStateVariable, StateVariable, StorageGraph};
    use knowdit_repo_model::{HistoricalSemanticRecord, MatchStrength, SemanticMatch, SemanticMatchSet, SourceLanguage};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempDb {
        repo: RepoDatabase,
        path: PathBuf,
    }

    impl Drop for TempDb {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
            let _ = std::fs::remove_file(self.path.with_extension("sqlite3-shm"));
            let _ = std::fs::remove_file(self.path.with_extension("sqlite3-wal"));
        }
    }

    async fn temp_db() -> Result<TempDb> {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "knowdit-orphan-pass-test-{}-{unique}.sqlite3",
            std::process::id()
        ));
        let repo = RepoDatabase::open_sqlite(path.clone()).await?;
        repo.init_schema().await?;
        Ok(TempDb { repo, path })
    }

    fn sample_project() -> ProjectData {
        let root = std::env::temp_dir();
        let rel = PathBuf::from("src/Vault.sol");
        let files = vec![ScopedSourceFile {
            absolute_path: root.join(&rel),
            relative_path: rel,
            content: "contract Vault {}".to_string(),
        }];
        ProjectData::new(
            "vault-proj".to_string(),
            None,
            SourceLanguage::Solidity,
            ProjectScopeContent::new(root, files),
        )
    }

    async fn seed_vault_cg(repo: &RepoDatabase) -> Result<()> {
        let loc = FileLocation {
            start_line: 1,
            start_column: 0,
            end_line: 4,
            end_column: 0,
        };
        let mut contracts = BTreeMap::new();
        contracts.insert(
            1,
            Contract {
                id: 1,
                name: "Vault".to_string(),
                relative_file_path: PathBuf::from("src/Vault.sol"),
                chunk: FileChunk {
                    loc,
                    content: "contract Vault {}".to_string(),
                },
                functions: vec![
                    Function {
                        id: 10,
                        name: "deposit".to_string(),
                        args: "uint256 amt".to_string(),
                        relative_file_path: PathBuf::from("src/Vault.sol"),
                        loc,
                        content: Some("function deposit(uint256 amt) external { assets += amt; }".to_string()),
                        calls: Vec::new(),
                        description: None,
                    },
                    Function {
                        id: 11,
                        name: "emergencySweep".to_string(),
                        args: "address to".to_string(),
                        relative_file_path: PathBuf::from("src/Vault.sol"),
                        loc,
                        content: Some("function emergencySweep(address to) external { assets = 0; }".to_string()),
                        calls: Vec::new(),
                        description: None,
                    },
                    Function {
                        id: 12,
                        name: "totalAssets".to_string(),
                        args: String::new(),
                        relative_file_path: PathBuf::from("src/Vault.sol"),
                        loc,
                        content: Some("function totalAssets() external view returns (uint256) { return assets; }".to_string()),
                        calls: Vec::new(),
                        description: None,
                    },
                ],
                description: None,
            },
        );
        repo.write_call_graph(&CallGraph {
            contracts,
            interfaces: BTreeMap::new(),
        })
        .await?;

        let mut state_variables = BTreeMap::new();
        state_variables.insert(
            1,
            StateVariable {
                id: 1,
                name: "assets".to_string(),
                type_name: "uint256".to_string(),
                relative_file_path: PathBuf::from("src/Vault.sol"),
                loc,
                content: "uint256 public assets;".to_string(),
            },
        );
        repo.write_storage_graph(&StorageGraph {
            state_variables,
            contract_variables: vec![ContractVariable {
                contract_id: 1,
                state_variable_id: 1,
                description: None,
            }],
            function_state_variables: vec![
                FunctionStateVariable {
                    function_id: 10,
                    state_variable_id: 1,
                    is_write: true,
                    description: None,
                },
                FunctionStateVariable {
                    function_id: 11,
                    state_variable_id: 1,
                    is_write: true,
                    description: None,
                },
            ],
        })
        .await?;
        Ok(())
    }

    #[tokio::test]
    async fn orphan_pass_skips_cleanly_when_call_graph_is_empty() -> Result<()> {
        let temp = temp_db().await?;
        let project = sample_project();
        let initial = vec![ExtractedSemantic {
            name: "Deposit".to_string(),
            category: DeFiCategory::Lending,
            definition: "d".to_string(),
            description: "desc".to_string(),
            functions: vec![ExtractedFunction {
                name: "deposit".to_string(),
                contract: "Vault".to_string(),
                signature: None,
            }],
        }];
        temp.repo.replace_project_semantics(&initial).await?;

        let pass = OrphanCoveragePass {
            extract_chunk_input_budget: None,
            no_orphan_pass: false,
            no_orphan_fallback: false,
        };
        let reconciled = pass
            .reconcile_resumed_semantics(
                &temp.repo,
                None,
                &project,
                initial.clone(),
                &AgentRunOptions::new(10),
            )
            .await?;
        assert_eq!(reconciled.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn orphan_pass_appends_deterministic_fallback_and_respects_semantic_matched_fk_lock()
    -> Result<()> {
        let temp = temp_db().await?;
        seed_vault_cg(&temp.repo).await?;
        let project = sample_project();

        // Only `deposit` was extracted; `emergencySweep` is an uncovered stateful entrypoint.
        let initial = vec![ExtractedSemantic {
            name: "Vault Deposit".to_string(),
            category: DeFiCategory::Lending,
            definition: "def".to_string(),
            description: "desc".to_string(),
            functions: vec![ExtractedFunction {
                name: "deposit".to_string(),
                contract: "Vault".to_string(),
                signature: Some("function deposit(uint256 amt) external".to_string()),
            }],
        }];
        temp.repo.replace_project_semantics(&initial).await?;

        let pass = OrphanCoveragePass {
            extract_chunk_input_budget: None,
            no_orphan_pass: false,
            no_orphan_fallback: false,
        };
        let updated = pass
            .reconcile_resumed_semantics(
                &temp.repo,
                None,
                &project,
                initial.clone(),
                &AgentRunOptions::new(10),
            )
            .await?;
        assert_eq!(updated.len(), 2);
        assert_eq!(updated[1].name, "Vault Stateful Entrypoints");
        assert_eq!(updated[1].category, DeFiCategory::Lending);
        assert_eq!(updated[1].functions.len(), 1);
        assert_eq!(updated[1].functions[0].name, "emergencySweep");
        assert!(updated[1].description.contains("assets"));

        // Now test FK-lock guard: reset semantics back to `initial`, write a `semantic_matched`
        // row referencing extract_id=1, and verify reconcile refuses to mutate locked rows.
        let temp_locked = temp_db().await?;
        seed_vault_cg(&temp_locked.repo).await?;
        temp_locked.repo.replace_project_semantics(&initial).await?;
        temp_locked
            .repo
            .write_semantic_match_results(&SemanticMatchSet {
                historicals: vec![HistoricalSemanticRecord {
                    semantic: knowdit_kg_model::db::semantic_node::Model {
                        id: 50,
                        name: "Hist".to_string(),
                        definition: "d".to_string(),
                        description: "desc".to_string(),
                        category: DeFiCategory::Lending,
                    },
                    findings: Vec::new(),
                    raw_children: Vec::new(),
                    rendered_description: String::new(),
                }],
                matches: vec![SemanticMatch {
                    extract_id: 1,
                    historical_id: 50,
                    strength: MatchStrength::High,
                    evidence: "locked match evidence string exceeding forty chars".to_string(),
                }],
            })
            .await?;

        let locked_out = pass
            .reconcile_resumed_semantics(
                &temp_locked.repo,
                None,
                &project,
                initial.clone(),
                &AgentRunOptions::new(10),
            )
            .await?;
        assert_eq!(
            locked_out.len(),
            1,
            "must not mutate project_semantic once semantic_matched exists"
        );
        assert_eq!(temp_locked.repo.load_project_semantics().await?.len(), 1);
        Ok(())
    }
}
