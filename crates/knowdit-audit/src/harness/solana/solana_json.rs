//! Wire summary parser and Dual-Mode Gate 2 coverage mappers for `SolanaHarness`.
//!
//! Parses `KNOWDIT_SOLANA_SUMMARY_JSON:{...}` lines, `KNOWDIT_POST_ATTACK_REACHED:`
//! deterministic PoC assertions, `KNOWDIT_SETUP_FAILED:` setup markers, and raw
//! `LiteSVM` / Solana transaction logs (`Program <id> invoke`, `Instruction: <Name>`)
//! into Knowdit's shared [`ForgeTestSummary`] + [`CoverageEntry`] representations.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use knowdit_repo_model::{
    CoverageEntry,
    cg::{CallGraph, Function},
};
use serde::{Deserialize, Serialize};

use crate::harness::solidity::{CounterCall, ForgeTestSummary};

/// Prefix emitted by `knowdit_harness_rt::emit_summary` on stdout/stderr.
pub const SOLANA_SUMMARY_PREFIX: &str = "KNOWDIT_SOLANA_SUMMARY_JSON:";

/// Substring marker used by deterministic PoC assertions (`assert!` / `panic!`).
pub const POC_VIOLATION_MARKER: &str = "KNOWDIT_POST_ATTACK_REACHED:";

/// Substring marker used when harness account/program setup fails before fuzzing.
pub const SETUP_FAILED_MARKER: &str = "KNOWDIT_SETUP_FAILED:";

/// One instruction invocation observed during a `LiteSVM` test run.
#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ExecutedInstruction {
    #[serde(default)]
    pub program: String,
    #[serde(default)]
    pub instruction: String,
}

/// One call step inside a shrunk `proptest` counterexample sequence.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolanaCounterCall {
    #[serde(default)]
    pub contract: Option<String>,
    #[serde(default)]
    pub function: Option<String>,
    #[serde(default)]
    pub args: Option<String>,
}

/// Typed wire payload serialized into `HarnessRunRecord.sequence` (no `json!`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedSolanaSequence {
    pub calls: Vec<SolanaCounterCall>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub executed_instructions: Vec<ExecutedInstruction>,
}

/// JSON payload emitted after [`SOLANA_SUMMARY_PREFIX`] by a Solana harness run.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SolanaWireSummary {
    #[serde(default)]
    pub setup_failed: bool,
    #[serde(default)]
    pub failure_reason: Option<String>,
    #[serde(default)]
    pub total_calls: i64,
    #[serde(default)]
    pub total_reverts: i64,
    #[serde(default)]
    pub violated: bool,
    #[serde(default)]
    pub deterministic_test_failure: bool,
    #[serde(default)]
    pub executed_instructions: Vec<ExecutedInstruction>,
    #[serde(default)]
    pub counterexample_calls: Vec<SolanaCounterCall>,
}

/// Parsed digest of a `cargo test` invocation over a `LiteSVM` harness.
#[derive(Debug, Default, Clone)]
pub struct SolanaParsedOutput {
    pub summary: ForgeTestSummary,
    pub executed_instructions: Vec<ExecutedInstruction>,
}

impl SolanaParsedOutput {
    /// Parse `cargo test` stdout and stderr into a [`ForgeTestSummary`] (driving
    /// Gate 1 `runtime_audit::run`) plus deduplicated [`ExecutedInstruction`]s
    /// (driving Gate 2 Tier 2 SBF trace coverage when LLVM host coverage is absent).
    pub fn parse(stdout: &str, stderr: &str) -> Self {
        let mut summary = ForgeTestSummary::default();
        let mut executed_set: BTreeSet<ExecutedInstruction> = BTreeSet::new();
        let mut saw_wire = false;

        for line in stdout.lines().chain(stderr.lines()) {
            if let Some(idx) = line.find(SOLANA_SUMMARY_PREFIX) {
                let json_slice = line[idx + SOLANA_SUMMARY_PREFIX.len()..].trim();
                if let Ok(wire) = serde_json::from_str::<SolanaWireSummary>(json_slice) {
                    saw_wire = true;
                    Self::fold_wire_summary(&mut summary, &mut executed_set, wire);
                }
            }
        }

        // Also harvest raw Solana / Anchor SVM transaction logs (`Program <id> invoke`,
        // `Program log: Instruction: <Name>`) from stdout/stderr so Tier 2 coverage
        // works even when the harness prints `TransactionMetadata.logs` directly.
        Self::harvest_svm_logs(stdout, &mut executed_set);
        Self::harvest_svm_logs(stderr, &mut executed_set);

        // Fallback marker scan for pure `#[test]` PoCs or panics that occurred
        // before/without `KNOWDIT_SOLANA_SUMMARY_JSON:` emission.
        if let Some(reason) =
            Self::extract_marker_line(stdout, SETUP_FAILED_MARKER)
                .or_else(|| Self::extract_marker_line(stderr, SETUP_FAILED_MARKER))
        {
            summary.setup_failed = true;
            if summary.failure_reason.is_none() {
                summary.failure_reason = Some(reason);
            }
        }

        if let Some(reason) =
            Self::extract_marker_line(stdout, POC_VIOLATION_MARKER)
                .or_else(|| Self::extract_marker_line(stderr, POC_VIOLATION_MARKER))
        {
            summary.violated = true;
            if summary.counterexample_calls.is_empty() {
                summary.deterministic_test_failure = true;
            }
            if summary.failure_reason.is_none() {
                summary.failure_reason = Some(reason);
            }
            if summary.total_calls == 0 {
                summary.total_calls = 1;
            }
        } else if !saw_wire && !executed_set.is_empty() && summary.total_calls == 0 {
            summary.total_calls = executed_set.len() as i64;
        }

        let executed_instructions: Vec<ExecutedInstruction> = executed_set.into_iter().collect();
        if summary.counterexample_json.is_none()
            && (!summary.counterexample_calls.is_empty() || !executed_instructions.is_empty())
        {
            let persisted = PersistedSolanaSequence {
                calls: summary
                    .counterexample_calls
                    .iter()
                    .map(|c| SolanaCounterCall {
                        contract: c.contract.clone(),
                        function: c.function.clone(),
                        args: c.args.clone(),
                    })
                    .collect(),
                executed_instructions: executed_instructions.clone(),
            };
            summary.counterexample_json = serde_json::to_value(persisted).ok();
        }

        Self {
            summary,
            executed_instructions,
        }
    }

    fn fold_wire_summary(
        summary: &mut ForgeTestSummary,
        executed_set: &mut BTreeSet<ExecutedInstruction>,
        wire: SolanaWireSummary,
    ) {
        if wire.setup_failed {
            summary.setup_failed = true;
        }
        if wire.total_calls >= summary.total_calls {
            summary.total_calls = wire.total_calls;
        } else {
            summary.total_calls += wire.total_calls;
        }
        if wire.total_reverts >= summary.total_reverts {
            summary.total_reverts = wire.total_reverts;
        } else {
            summary.total_reverts += wire.total_reverts;
        }
        for ix in wire.executed_instructions {
            if !ix.instruction.trim().is_empty() {
                executed_set.insert(ix);
            }
        }
        if wire.violated {
            summary.violated = true;
            summary.deterministic_test_failure =
                wire.deterministic_test_failure || wire.counterexample_calls.is_empty();
            if let Some(reason) = wire.failure_reason.filter(|r| !r.trim().is_empty()) {
                summary.failure_reason = Some(reason);
            }
            if !wire.counterexample_calls.is_empty() {
                let persisted = PersistedSolanaSequence {
                    calls: wire.counterexample_calls.clone(),
                    executed_instructions: executed_set.iter().cloned().collect(),
                };
                summary.counterexample_json = serde_json::to_value(persisted).ok();
                summary.counterexample_calls = wire
                    .counterexample_calls
                    .into_iter()
                    .map(|c| CounterCall {
                        contract: c.contract,
                        function: c.function,
                        args: c.args,
                    })
                    .collect();
            }
        } else if summary.failure_reason.is_none()
            && let Some(reason) = wire.failure_reason.filter(|r| !r.trim().is_empty())
        {
            summary.failure_reason = Some(reason);
        }
    }

    fn extract_marker_line(text: &str, marker: &str) -> Option<String> {
        for line in text.lines() {
            if let Some(idx) = line.find(marker) {
                return Some(line[idx..].trim().to_string());
            }
        }
        None
    }

    fn harvest_svm_logs(text: &str, executed_set: &mut BTreeSet<ExecutedInstruction>) {
        let mut active_program_stack: Vec<String> = Vec::new();
        for raw_line in text.lines() {
            let line = raw_line.trim();
            // Match `Program <ID> invoke [<depth>]`
            if let Some(rest) = line.strip_prefix("Program ") {
                if let Some((prog_id, after)) = rest.split_once(" invoke [")
                    && after.ends_with(']')
                {
                    active_program_stack.push(prog_id.trim().to_string());
                    continue;
                }
                if rest.contains(" success") || rest.contains(" failed:") {
                    let _ = active_program_stack.pop();
                    continue;
                }
            }
            // Match `Program log: Instruction: <IxName>`
            if let Some(ix_raw) = line
                .strip_prefix("Program log: Instruction:")
                .or_else(|| line.strip_prefix("Instruction:"))
            {
                let ix_name = ix_raw.trim();
                if !ix_name.is_empty() {
                    let program = active_program_stack
                        .last()
                        .cloned()
                        .unwrap_or_default();
                    executed_set.insert(ExecutedInstruction {
                        program,
                        instruction: ix_name.to_string(),
                    });
                }
            }
        }
    }
}

/// Dual-Mode Gate 2 coverage converter for Solana / Rust projects.
pub struct SolanaCoverageMapper;

impl SolanaCoverageMapper {
    /// Tier 1: Parse `cargo llvm-cov --lcov` output, normalizing absolute `SF:`
    /// paths back to `repo_root`-relative paths and excluding harness scratch
    /// files (`harness_dir`) or external Cargo registry files.
    pub fn parse_llvm_lcov(
        lcov_text: &str,
        repo_root: &Path,
        harness_dir: &Path,
    ) -> Vec<CoverageEntry> {
        let canon_repo = std::fs::canonicalize(repo_root).unwrap_or_else(|_| repo_root.to_path_buf());
        let canon_harness =
            std::fs::canonicalize(harness_dir).unwrap_or_else(|_| harness_dir.to_path_buf());

        let mut current_rel_file: Option<String> = None;
        let mut out = Vec::new();

        for raw_line in lcov_text.lines() {
            let line = raw_line.trim_end();
            if let Some(sf_raw) = line.strip_prefix("SF:") {
                let sf_path = Path::new(sf_raw.trim());
                if sf_path.starts_with(&canon_harness) || sf_path.starts_with(harness_dir) {
                    current_rel_file = None;
                    continue;
                }
                if let Ok(rel) = sf_path
                    .strip_prefix(&canon_repo)
                    .or_else(|_| sf_path.strip_prefix(repo_root))
                {
                    current_rel_file = Some(rel.to_string_lossy().into_owned());
                } else if sf_path.is_relative() {
                    current_rel_file = Some(sf_path.to_string_lossy().into_owned());
                } else {
                    current_rel_file = None;
                }
            } else if line == "end_of_record" {
                current_rel_file = None;
            } else if let Some(da_raw) = line.strip_prefix("DA:") {
                let Some(rel_file) = &current_rel_file else {
                    continue;
                };
                let mut parts = da_raw.split(',');
                let (Some(line_s), Some(hits_s)) = (parts.next(), parts.next()) else {
                    continue;
                };
                let Ok(line_number) = line_s.trim().parse::<i32>() else {
                    continue;
                };
                let Ok(hit_count) = hits_s.trim().parse::<i64>() else {
                    continue;
                };
                out.push(CoverageEntry {
                    relative_contract_path: rel_file.clone(),
                    line_number,
                    hit_count,
                });
            }
        }
        out
    }

    /// Tier 2: Synthesize line-coverage hits from SBF `executed_instructions`
    /// resolved against the project's static [`CallGraph`] (including transitive
    /// intra-repo callees such as Anchor `instructions::<ix>::handler` delegations).
    pub fn synthesize_sbf_trace_coverage(
        executed: &[ExecutedInstruction],
        callgraph: &CallGraph,
        program_id_to_lib: &BTreeMap<String, String>,
    ) -> Vec<CoverageEntry> {
        if executed.is_empty() {
            return Vec::new();
        }

        let fn_by_id: BTreeMap<i32, &Function> = callgraph
            .contracts
            .values()
            .flat_map(|c| c.functions.iter().map(|f| (f.id, f)))
            .collect();

        let mut hit_fn_ids: BTreeSet<i32> = BTreeSet::new();
        for exec in executed {
            let resolved_prog = program_id_to_lib
                .get(exec.program.trim())
                .map(String::as_str)
                .unwrap_or_else(|| exec.program.trim());
            let prog_norm = resolved_prog.replace('-', "_");
            let ix_raw = exec.instruction.trim();
            let ix_snake = Self::camel_to_snake(ix_raw);

            let mut matched_roots = Vec::new();
            for contract in callgraph.contracts.values() {
                let contract_matches = prog_norm.is_empty()
                    || contract.name == prog_norm
                    || contract.name.eq_ignore_ascii_case(resolved_prog);
                if contract_matches {
                    for f in &contract.functions {
                        if f.name == ix_raw || f.name == ix_snake {
                            matched_roots.push(f.id);
                        }
                    }
                }
            }
            // Fallback: match unique/named instruction function across any contract
            // in the workspace (e.g. when `Program <pubkey>` used a random test pubkey).
            if matched_roots.is_empty() {
                for contract in callgraph.contracts.values() {
                    for f in &contract.functions {
                        if f.name == ix_raw || f.name == ix_snake {
                            matched_roots.push(f.id);
                        }
                    }
                }
            }

            // Traverse intra-repo call edges (BFS) from each invoked entrypoint so
            // delegated helpers (`instructions::withdraw::handler`, `math::...`)
            // also register coverage hits.
            let mut stack = matched_roots;
            while let Some(fid) = stack.pop() {
                if !hit_fn_ids.insert(fid) {
                    continue;
                }
                if let Some(func) = fn_by_id.get(&fid) {
                    for edge in &func.calls {
                        if fn_by_id.contains_key(&edge.to_id) && !hit_fn_ids.contains(&edge.to_id) {
                            stack.push(edge.to_id);
                        }
                    }
                }
            }
        }

        let mut entries = Vec::with_capacity(hit_fn_ids.len());
        for fid in hit_fn_ids {
            if let Some(func) = fn_by_id.get(&fid) {
                entries.push(CoverageEntry {
                    relative_contract_path: func.relative_file_path.to_string_lossy().into_owned(),
                    line_number: func.loc.start_line as i32,
                    hit_count: 1,
                });
            }
        }
        entries
    }

    /// Convert Anchor's `"WithdrawCollateral"` log name into `"withdraw_collateral"`.
    pub fn camel_to_snake(input: &str) -> String {
        if input.contains('_') {
            return input.to_ascii_lowercase();
        }
        let mut out = String::with_capacity(input.len() + 4);
        for (idx, ch) in input.chars().enumerate() {
            if ch.is_ascii_uppercase() {
                if idx > 0 {
                    out.push('_');
                }
                out.push(ch.to_ascii_lowercase());
            } else {
                out.push(ch);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reflect::{AuditThresholds, coverage_audit, runtime_audit};
    use crate::types::{AuditSpecification, SuqenceCallStep};
    use std::path::PathBuf;

    #[test]
    fn parses_proptest_shrinking_summaries_and_passes_gate1() {
        let stdout = r#"
running 1 test
KNOWDIT_SOLANA_SUMMARY_JSON:{"setup_failed":false,"total_calls":40,"total_reverts":4,"violated":true,"executed_instructions":[{"program":"solana_vault","instruction":"deposit"},{"program":"solana_vault","instruction":"withdraw"}],"counterexample_calls":[{"contract":"solana_vault","function":"deposit","args":"100"},{"contract":"solana_vault","function":"deposit","args":"200"},{"contract":"solana_vault","function":"withdraw","args":"300"}]}
KNOWDIT_SOLANA_SUMMARY_JSON:{"setup_failed":false,"failure_reason":"KNOWDIT_POST_ATTACK_REACHED: vault drained","total_calls":52,"total_reverts":5,"violated":true,"executed_instructions":[{"program":"solana_vault","instruction":"withdraw"}],"counterexample_calls":[{"contract":"solana_vault","function":"withdraw","args":"300"}]}
test test_spec_1 ... FAILED
"#;
        let parsed = SolanaParsedOutput::parse(stdout, "");
        assert!(parsed.summary.violated);
        assert!(!parsed.summary.deterministic_test_failure);
        assert_eq!(parsed.summary.total_calls, 52);
        assert_eq!(parsed.summary.total_reverts, 5);
        assert_eq!(parsed.summary.counterexample_calls.len(), 1);
        assert_eq!(
            parsed.summary.counterexample_calls[0].function.as_deref(),
            Some("withdraw")
        );
        assert_eq!(parsed.executed_instructions.len(), 2);

        let spec = AuditSpecification {
            sequence: vec![SuqenceCallStep {
                contract: "solana_vault".to_string(),
                function: "withdraw".to_string(),
                intention: "drain".to_string(),
            }],
            ..Default::default()
        };
        assert!(!runtime_audit::run(&parsed.summary, &spec).is_fail());
    }

    #[test]
    fn parses_poc_panic_and_svm_logs_and_satisfies_tier2_gate2() {
        let stderr = r#"
Program VaULt11111111111111111111111111111111111111 invoke [1]
Program log: Instruction: WithdrawAssets
Program VaULt11111111111111111111111111111111111111 success
thread 'test_post_attack_state_must_not_be_reachable' panicked at tests/test_7.rs:42:5:
KNOWDIT_POST_ATTACK_REACHED: unauthorized withdraw drained vault_token
"#;
        let parsed = SolanaParsedOutput::parse("", stderr);
        assert!(parsed.summary.violated);
        assert!(parsed.summary.deterministic_test_failure);
        assert_eq!(parsed.summary.total_calls, 1);
        assert_eq!(
            parsed.summary.failure_reason.as_deref(),
            Some("KNOWDIT_POST_ATTACK_REACHED: unauthorized withdraw drained vault_token")
        );

        let analysis = knowdit_rust::RustStaticAnalyzer::analyze(&[knowdit_rust::RustSourceInput {
            relative_path: PathBuf::from("programs/vault/src/lib.rs"),
            content: r#"
                #[program]
                pub mod solana_vault {
                    pub fn withdraw_assets(amount: u64) {
                        inner_transfer(amount);
                    }
                }
                pub fn inner_transfer(_amount: u64) {}
            "#
            .to_string(),
        }])
        .expect("analyze");

        let mut id_map = BTreeMap::new();
        id_map.insert(
            "VaULt11111111111111111111111111111111111111".to_string(),
            "solana_vault".to_string(),
        );
        let cov = SolanaCoverageMapper::synthesize_sbf_trace_coverage(
            &parsed.executed_instructions,
            &analysis.call_graph.call_graph,
            &id_map,
        );
        assert_eq!(cov.len(), 2, "both entrypoint and delegated callee should be hit");

        let spec = AuditSpecification {
            sequence: vec![SuqenceCallStep {
                contract: "solana_vault".to_string(),
                function: "withdraw_assets".to_string(),
                intention: "drain".to_string(),
            }],
            ..Default::default()
        };
        let gate2 = coverage_audit::run(
            &spec,
            &cov,
            &analysis.call_graph.call_graph,
            &AuditThresholds::default(),
        );
        assert!(!gate2.is_fail());
    }

    #[test]
    fn parses_llvm_lcov_stripping_repo_root_and_ignoring_harness_dir() {
        let repo_root = Path::new("/workspace/sol_repo");
        let harness_dir = Path::new("/workspace/sol_repo/knowdit_solana_harness");
        let lcov = "\
SF:/workspace/sol_repo/programs/vault/src/lib.rs\n\
DA:12,3\n\
DA:13,0\n\
end_of_record\n\
SF:/workspace/sol_repo/knowdit_solana_harness/spec_1/tests/test_1.rs\n\
DA:5,1\n\
end_of_record\n";
        let rows = SolanaCoverageMapper::parse_llvm_lcov(lcov, repo_root, harness_dir);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].relative_contract_path, "programs/vault/src/lib.rs");
        assert_eq!(rows[0].line_number, 12);
        assert_eq!(rows[0].hit_count, 3);
    }
}
