//! Agent tools registered inside `SolanaHarness` (`write_harness_file`,
//! `run_solana`, `list_test_files`).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use knowdit_repo_model::{CoverageEntry, HarnessRunRecord, RunKind};
use schemars::JsonSchema;
use serde::Deserialize;

use super::runner::SolanaRunner;
use super::solana_json::{SolanaCoverageMapper, SolanaParsedOutput};
use super::utils::{RecordedSolanaRun, SolanaAttemptHandle, tail_str, truncate_str_head_tail};
use crate::types::AuditSpecification;

// =============================================================================
// write_harness_file
// =============================================================================

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub(super) struct WriteSolanaHarnessFileArgs {
    /// Basename of the Rust integration test file, e.g. `"test_42.rs"`.
    filename: String,
    /// Complete Rust source file placed under `<per_spec_dir>/tests/<filename>`.
    content: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = WriteSolanaHarnessFileArgs,
    invoke = invoke,
    description = "Write (or overwrite) the Rust LiteSVM integration test file under `<per_spec_dir>/tests/<filename>`. Re-call to update the file after compile or runtime errors, then invoke `run_solana`.",
    name = "write_harness_file",
)]
pub(super) struct WriteSolanaHarnessFileTool {
    pub(super) attempt: SolanaAttemptHandle,
    pub(super) per_spec_tests_dir: PathBuf,
}

impl WriteSolanaHarnessFileTool {
    async fn invoke(
        &self,
        args: WriteSolanaHarnessFileArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        self.attempt
            .write_agent_test_file(&self.per_spec_tests_dir, &args.filename, &args.content)
            .await
    }
}

// =============================================================================
// run_solana
// =============================================================================

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub(super) struct RunSolanaArgs {
    /// Optional integration test target stem (e.g. `"test_42"` for `tests/test_42.rs`).
    /// Defaults to the stem of the last file written via `write_harness_file`.
    #[serde(default)]
    pub(super) test_stem: Option<String>,
    /// Optional substring filter for a specific `#[test]` function name.
    #[serde(default)]
    pub(super) test_filter: Option<String>,
    /// Optional deterministic RNG seed forwarded via `PROPTEST_RNG_SEED`.
    #[serde(default)]
    pub(super) seed: Option<i64>,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = RunSolanaArgs,
    invoke = invoke,
    description = "Compile and execute the per-spec Rust `LiteSVM` harness via `cargo test -- --nocapture`, parse `KNOWDIT_SOLANA_SUMMARY_JSON:` and `KNOWDIT_POST_ATTACK_REACHED:` signals, and run Inline Gate 1 (runtime setup/revert/sequence overlap) + Dual-Mode Gate 2 (LLVM LCOV or SBF instruction-trace coverage). Call this after every `write_harness_file`.",
    name = "run_solana",
)]
pub(super) struct RunSolanaTool {
    pub(super) attempt: SolanaAttemptHandle,
    pub(super) runner: SolanaRunner,
    pub(super) per_spec_dir: PathBuf,
    pub(super) test_runs: u64,
    pub(super) coverage_runs: u64,
    pub(super) spec: Arc<AuditSpecification>,
    pub(super) callgraph: Arc<knowdit_repo_model::cg::CallGraph>,
    pub(super) program_id_to_lib: Arc<BTreeMap<String, String>>,
    pub(super) gate2_fidelity_threshold: f64,
    pub(super) lcov_path: PathBuf,
}

impl RunSolanaTool {
    async fn invoke(
        &self,
        args: RunSolanaArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let stem = match self.resolve_test_stem(&args).await {
            Some(s) => s,
            None => {
                return Ok(
                    "error: no harness file has been written yet — call `write_harness_file` first"
                        .to_string(),
                );
            }
        };

        let test_argv = self.build_test_argv(&stem, args.test_filter.as_deref());
        let test_env = self.build_env(self.test_runs, args.seed);
        let test_out = match self
            .runner
            .run_cargo(&test_argv, &self.per_spec_dir, &test_env)
            .await
        {
            Ok(o) => o,
            Err(e) => return Ok(format!("error: cargo test spawn failed: {e:#}")),
        };

        let parsed = SolanaParsedOutput::parse(&test_out.stdout, &test_out.stderr);
        let test_summary = parsed.summary;

        // Preserve panic / marker diagnostics in `HarnessRunRecord.stdout` as well as
        // `stderr` because downstream `VerdictGrader` / `SeverityGrader` inspect
        // `run.stdout_tail`.
        let combined_stdout =
            if test_out.stderr.contains("panicked at") || test_out.stderr.contains("KNOWDIT_") {
                format!("{}\n--- stderr ---\n{}", test_out.stdout, test_out.stderr)
            } else {
                test_out.stdout.clone()
            };

        let test_record = HarnessRunRecord {
            kind: RunKind::Test,
            seed: args.seed,
            runs: self.test_runs as i64,
            forge_args: test_argv.clone(),
            exit_code: test_out.exit_code,
            stdout: truncate_str_head_tail(&combined_stdout, 200_000),
            stderr: truncate_str_head_tail(&test_out.stderr, 200_000),
            duration_ms: test_out.duration_ms,
            violated: test_summary.violated,
            sequence: test_summary.counterexample_json.clone(),
        };

        let test_gate_verdict = match crate::reflect::runtime_audit::run(&test_summary, &self.spec)
        {
            crate::reflect::AuditOutcome::Fail { reason, .. } => {
                Some(format!("[GATE 1 FAILED — runtime] {reason}"))
            }
            _ => None,
        };

        // Dual-Mode Gate 2 coverage: run whenever the test compiled and exercised
        // at least one SVM call (or produced a deterministic PoC violation).
        let should_collect_coverage = !test_out.timed_out
            && !test_summary.setup_failed
            && (test_summary.total_calls > 0 || test_summary.deterministic_test_failure);

        let mut cov_bundle: Option<SolanaCoverageBundle> = None;
        if should_collect_coverage {
            cov_bundle = Some(
                self.collect_dual_mode_coverage(
                    &stem,
                    &args,
                    &parsed.executed_instructions,
                    &test_summary,
                )
                .await,
            );
        }

        // Gate enforcement: a violated run only terminates when both inline
        // gates passed — Gate 1 (runtime audit) AND Gate 2 (dual-mode
        // coverage). Skipping the Gate 2 check let broken / disjoint
        // harnesses terminate immediately on any violated summary.
        let gate2_failed = cov_bundle
            .as_ref()
            .is_some_and(|bundle| bundle.gate_verdict.is_some());
        let violation_accepted =
            test_summary.violated && test_gate_verdict.is_none() && !gate2_failed;
        let mut state = self.attempt.0.lock().await;
        state.runs.push(RecordedSolanaRun {
            record: test_record.clone(),
            coverage_idx: None,
        });
        if violation_accepted {
            state.violation_observed = true;
        }
        state.last_test_calls = test_summary.total_calls.max(0) as u64;
        if let Some(bundle) = &cov_bundle {
            let coverage_idx = if !bundle.coverage.is_empty() {
                state.coverage.push(bundle.coverage.clone());
                Some(state.coverage.len() - 1)
            } else {
                None
            };
            if let Some(cov_rec) = &bundle.record {
                state.runs.push(RecordedSolanaRun {
                    record: cov_rec.clone(),
                    coverage_idx,
                });
            } else if let Some(last_run) = state.runs.last_mut() {
                last_run.coverage_idx = coverage_idx;
            }
            state.last_gate_passed = bundle.gate_verdict.is_none() && !bundle.coverage.is_empty();
        } else {
            state.last_gate_passed = false;
        }
        let stopped_for_violation = state.violation_observed;
        drop(state);

        let mut sections: Vec<String> = Vec::new();
        if let Some(v) = &test_gate_verdict {
            sections.push(v.clone());
        }
        if stopped_for_violation {
            let kind = if test_summary.deterministic_test_failure {
                "deterministic LiteSVM PoC asserted KNOWDIT_POST_ATTACK_REACHED"
            } else {
                "proptest + LiteSVM invariant counter-example produced"
            };
            sections.push(format!(
                "[SUCCESS — violation observed] {kind}. The orchestrator will terminate after this turn."
            ));
        } else if test_summary.violated && test_gate_verdict.is_some() {
            sections.push(
                "[VIOLATION REJECTED BY GATE 1] Your test reported a violation, but Gate 1 failed \
                 (e.g. setup_failed, >=95% revert rate, or counterexample calls did not touch \
                 `spec.sequence`). Fix the Gate 1 failure and call `run_solana` again."
                    .to_string(),
            );
        } else if test_summary.violated && gate2_failed {
            sections.push(
                "[VIOLATION REJECTED BY GATE 2] Your test reported a violation, but Gate 2 \
                 coverage failed (see the gate verdict below). Fix the harness and call \
                 `run_solana` again."
                    .to_string(),
            );
        } else if test_summary.total_calls == 0 && !test_summary.deterministic_test_failure {
            sections.push(
                "[NO-OP] cargo test ran (or failed to compile) with 0 recorded SVM calls and no \
                 `KNOWDIT_POST_ATTACK_REACHED:` violation. Check the compiler/test stderr below, \
                 ensure `SummaryTracker::record_call` + `emit_ok`/`emit_violation` are invoked, \
                 and call `run_solana` again."
                    .to_string(),
            );
        }

        sections.push(format!(
            "ok: cargo test exit={}{} duration_ms={}. runtime: setup_failed={} calls={} \
             reverts={} revert_rate={:.3} violated={} deterministic_violation={}",
            test_record.exit_code,
            if test_out.timed_out { " [TIMEOUT]" } else { "" },
            test_record.duration_ms,
            test_summary.setup_failed,
            test_summary.total_calls,
            test_summary.total_reverts,
            test_summary.revert_rate(),
            test_summary.violated,
            test_summary.deterministic_test_failure,
        ));
        sections.push(format!(
            "--- cargo test stdout (last 4K) ---\n{}",
            tail_str(&test_out.stdout, 4_000)
        ));
        sections.push(format!(
            "--- cargo test stderr (last 4K) ---\n{}",
            tail_str(&test_out.stderr, 4_000)
        ));

        if let Some(bundle) = &cov_bundle {
            if let Some(v) = &bundle.gate_verdict {
                sections.push(v.clone());
            }
            sections.push(format!(
                "ok: Gate 2 coverage ({}) recorded {} entry(ies)",
                bundle.tier_label,
                bundle.coverage.len(),
            ));
        } else {
            sections.push(
                "note: Gate 2 coverage skipped this turn (compile/runtime failure or 0 calls)."
                    .to_string(),
            );
        }

        Ok(sections.join("\n"))
    }

    async fn resolve_test_stem(&self, args: &RunSolanaArgs) -> Option<String> {
        if let Some(explicit) = args.test_stem.as_deref() {
            let trimmed = explicit.trim().trim_end_matches(".rs");
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
        let state = self.attempt.0.lock().await;
        state
            .harness_filename
            .as_deref()
            .map(|f| f.trim_end_matches(".rs").to_string())
    }

    fn build_test_argv(&self, stem: &str, test_filter: Option<&str>) -> Vec<String> {
        let mut argv = vec![
            "test".to_string(),
            "--manifest-path".to_string(),
            self.per_spec_dir.join("Cargo.toml").display().to_string(),
            "--test".to_string(),
            stem.to_string(),
            "--".to_string(),
            "--nocapture".to_string(),
        ];
        if let Some(filter) = test_filter.filter(|s| !s.trim().is_empty()) {
            argv.push(filter.trim().to_string());
        }
        argv
    }

    fn build_env(&self, cases: u64, seed: Option<i64>) -> BTreeMap<String, String> {
        let mut env = BTreeMap::new();
        env.insert("PROPTEST_CASES".to_string(), cases.max(1).to_string());
        if let Some(s) = seed {
            env.insert(
                "PROPTEST_RNG_SEED".to_string(),
                s.unsigned_abs().to_string(),
            );
        }
        env
    }

    async fn collect_dual_mode_coverage(
        &self,
        stem: &str,
        args: &RunSolanaArgs,
        executed_instructions: &[super::solana_json::ExecutedInstruction],
        test_summary: &crate::harness::solidity::ForgeTestSummary,
    ) -> SolanaCoverageBundle {
        let mut coverage: Vec<CoverageEntry> = Vec::new();
        let mut cov_record: Option<HarnessRunRecord> = None;
        let mut tier_label = "Tier 2 SBF execution-trace";

        // Tier 1: Host LLVM source-line coverage when `cargo-llvm-cov` is present.
        if self.runner.features().llvm_cov_available {
            let mut cov_argv = vec![
                "llvm-cov".to_string(),
                "test".to_string(),
                "--manifest-path".to_string(),
                self.per_spec_dir.join("Cargo.toml").display().to_string(),
                "--test".to_string(),
                stem.to_string(),
                "--lcov".to_string(),
                "--output-path".to_string(),
                self.lcov_path.display().to_string(),
                "--".to_string(),
                "--nocapture".to_string(),
            ];
            if let Some(filter) = args.test_filter.as_deref().filter(|s| !s.trim().is_empty()) {
                cov_argv.push(filter.trim().to_string());
            }
            let cov_env = self.build_env(self.coverage_runs, args.seed);
            if let Ok(out) = self
                .runner
                .run_cargo(&cov_argv, &self.per_spec_dir, &cov_env)
                .await
            {
                if let Ok(lcov_text) = tokio::fs::read_to_string(&self.lcov_path).await {
                    let parsed_lcov = SolanaCoverageMapper::parse_llvm_lcov(
                        &lcov_text,
                        self.runner.repo_root(),
                        self.runner.harness_dir(),
                    );
                    if !parsed_lcov.is_empty() {
                        tier_label = "Tier 1 cargo-llvm-cov LCOV";
                        coverage = parsed_lcov;
                        cov_record = Some(HarnessRunRecord {
                            kind: RunKind::Coverage,
                            seed: args.seed,
                            runs: self.coverage_runs as i64,
                            forge_args: cov_argv,
                            exit_code: out.exit_code,
                            stdout: truncate_str_head_tail(&out.stdout, 200_000),
                            stderr: truncate_str_head_tail(&out.stderr, 200_000),
                            duration_ms: out.duration_ms,
                            violated: false,
                            sequence: None,
                        });
                    }
                }
            }
        }

        // Tier 2 fallback: map SBF `executed_instructions` + `TransactionMetadata.logs`
        // onto `CallGraph` function `loc.start_line` entries.
        if coverage.is_empty() {
            coverage = SolanaCoverageMapper::synthesize_sbf_trace_coverage(
                executed_instructions,
                &self.callgraph,
                &self.program_id_to_lib,
            );
        }

        let gate_verdict = if coverage.is_empty() {
            if test_summary.deterministic_test_failure {
                None
            } else {
                Some(
                    "[GATE 2 FAILED — coverage] 0 in-scope program functions were hit. \
                     Ensure your harness invokes real SBF instructions and records them via \
                     `SummaryTracker::record_call` (or prints `TransactionMetadata.logs`)."
                        .to_string(),
                )
            }
        } else if test_summary.deterministic_test_failure {
            None
        } else {
            let thresholds = crate::reflect::AuditThresholds {
                gate2_fidelity_threshold: self.gate2_fidelity_threshold,
            };
            match crate::reflect::coverage_audit::run(
                &self.spec,
                &coverage,
                &self.callgraph,
                &thresholds,
            ) {
                crate::reflect::AuditOutcome::Fail { reason, .. } => {
                    Some(format!("[GATE 2 FAILED — coverage] {reason}"))
                }
                _ => None,
            }
        };

        SolanaCoverageBundle {
            record: cov_record,
            coverage,
            gate_verdict,
            tier_label,
        }
    }
}

struct SolanaCoverageBundle {
    record: Option<HarnessRunRecord>,
    coverage: Vec<CoverageEntry>,
    gate_verdict: Option<String>,
    tier_label: &'static str,
}

// =============================================================================
// list_test_files
// =============================================================================

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub(super) struct ListSolanaTestFilesArgs {
    #[serde(default)]
    pub(super) filter: Option<String>,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = ListSolanaTestFilesArgs,
    invoke = invoke,
    description = "List existing Rust/Anchor/LiteSVM/Bankrun test and setup files (`.rs`, `.ts`, `.js`, `.json`) under `tests/`, `test/`, `programs/`, `scripts/`, and `migrations/`. Use this to mirror existing PDA derivation, account layout, and initialization sequences.",
    name = "list_test_files",
)]
pub(super) struct ListSolanaTestFilesTool {
    pub(super) repo_root: PathBuf,
    pub(super) max_depth: usize,
}

const SKIP_DIRS: &[&str] = &[
    "target",
    "node_modules",
    ".git",
    ".anchor",
    "test-ledger",
    "knowdit_solana_harness",
];

const COLLECT_EXTS: &[&str] = &["rs", "ts", "js", "json"];

impl ListSolanaTestFilesTool {
    async fn invoke(
        &self,
        args: ListSolanaTestFilesArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let accept_dir = |dir: &std::path::Path| -> bool {
            dir.file_name()
                .and_then(|n| n.to_str())
                .map(|name| !SKIP_DIRS.contains(&name))
                .unwrap_or(true)
        };
        let accept_file = |file: &std::path::Path| -> bool {
            file.extension()
                .and_then(|e| e.to_str())
                .map(|ext| COLLECT_EXTS.contains(&ext))
                .unwrap_or(false)
        };

        let mut found: Vec<PathBuf> = Vec::new();
        for sub in ["tests", "test", "programs", "scripts", "migrations"] {
            let dir = self.repo_root.join(sub);
            if !dir.is_dir() {
                continue;
            }
            if let Ok(set) = knowdit_project::scope::walk_files(
                &dir,
                &accept_dir,
                &accept_file,
                Some(self.max_depth),
            ) {
                found.extend(set);
            }
        }
        found.sort();
        found.dedup();

        let filter = args.filter.unwrap_or_default();
        let mut out = String::new();
        let mut count = 0usize;
        for p in &found {
            let rel = p
                .strip_prefix(&self.repo_root)
                .unwrap_or(p)
                .to_string_lossy()
                .into_owned();
            if !filter.is_empty() && !rel.contains(&filter) {
                continue;
            }
            out.push_str("- ");
            out.push_str(&rel);
            out.push('\n');
            count += 1;
        }
        if count == 0 {
            return Ok(format!(
                "no test/setup files found{}",
                if filter.is_empty() {
                    String::new()
                } else {
                    format!(" matching `{filter}`")
                }
            ));
        }
        Ok(format!("ok: {count} file(s)\n{out}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::solana::runner::{SolanaBackend, SolanaBackendKind, SolanaFeatures};
    use crate::types::SuqenceCallStep;
    use std::time::Duration;

    #[tokio::test]
    async fn write_and_run_solana_tool_executes_cargo_test_and_passes_gates() {
        let Some(cargo_bin) = std::env::var_os("PATH").and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|d| d.join("cargo"))
                .find(|p| p.is_file())
        }) else {
            return;
        };

        let tmp = tempfile::tempdir().expect("tempdir");
        let repo_root = tmp.path().to_path_buf();
        let harness_dir = repo_root.join("knowdit_solana_harness");
        let per_spec_dir = harness_dir.join("spec_42");
        std::fs::create_dir_all(per_spec_dir.join("src")).expect("mkdir src");
        std::fs::create_dir_all(per_spec_dir.join("tests")).expect("mkdir tests");

        // Zero-external-dep Cargo.toml so this test runs 100% offline in <1.5s.
        std::fs::write(
            per_spec_dir.join("Cargo.toml"),
            r#"[package]
name = "knowdit-solana-spec-42"
version = "0.1.0"
edition = "2021"

[workspace]
"#,
        )
        .expect("write Cargo.toml");
        std::fs::write(
            per_spec_dir.join("src/lib.rs"),
            "pub fn ready() -> bool { true }\n",
        )
        .expect("write lib.rs");

        let analysis =
            knowdit_rust::RustStaticAnalyzer::analyze(&[knowdit_rust::RustSourceInput {
                relative_path: PathBuf::from("programs/vault/src/lib.rs"),
                content: r#"
                #[program]
                pub mod solana_vault {
                    pub fn deposit(amount: u64) {}
                    pub fn withdraw(amount: u64) {}
                }
            "#
                .to_string(),
            }])
            .expect("analyze");

        let attempt = SolanaAttemptHandle::new();
        let write_tool = WriteSolanaHarnessFileTool {
            attempt: attempt.clone(),
            per_spec_tests_dir: per_spec_dir.join("tests"),
        };
        let write_res = write_tool
            .invoke(WriteSolanaHarnessFileArgs {
                filename: "test_42.rs".to_string(),
                content: r#"
#[test]
fn test_exploit_sequence() {
    println!(
        "KNOWDIT_SOLANA_SUMMARY_JSON:{{\"setup_failed\":false,\"failure_reason\":\"KNOWDIT_POST_ATTACK_REACHED: unauthorized withdraw\",\"total_calls\":8,\"total_reverts\":1,\"violated\":true,\"deterministic_test_failure\":false,\"executed_instructions\":[{{\"program\":\"solana_vault\",\"instruction\":\"deposit\"}},{{\"program\":\"solana_vault\",\"instruction\":\"withdraw\"}}],\"counterexample_calls\":[{{\"contract\":\"solana_vault\",\"function\":\"deposit\",\"args\":\"1000\"}},{{\"contract\":\"solana_vault\",\"function\":\"withdraw\",\"args\":\"1000\"}}]}}"
    );
    panic!("KNOWDIT_POST_ATTACK_REACHED: unauthorized withdraw");
}
"#
                .to_string(),
            })
            .await
            .expect("write_harness_file invoke");
        assert!(write_res.starts_with("ok: wrote"), "got: {write_res}");

        let runner = SolanaRunner::new(
            SolanaBackend {
                kind: SolanaBackendKind::LocalDirect {
                    cargo_bin,
                    build_sbf_bin: None,
                },
                features: SolanaFeatures {
                    llvm_cov_available: false,
                    build_sbf_available: false,
                },
            },
            repo_root,
            harness_dir,
            Duration::from_secs(60),
        );

        let spec = Arc::new(AuditSpecification {
            sequence: vec![
                SuqenceCallStep {
                    contract: "solana_vault".to_string(),
                    function: "deposit".to_string(),
                    intention: "seed vault".to_string(),
                },
                SuqenceCallStep {
                    contract: "solana_vault".to_string(),
                    function: "withdraw".to_string(),
                    intention: "drain vault".to_string(),
                },
            ],
            ..Default::default()
        });

        let run_tool = RunSolanaTool {
            attempt: attempt.clone(),
            runner,
            per_spec_dir: per_spec_dir.clone(),
            test_runs: 16,
            coverage_runs: 8,
            spec,
            callgraph: Arc::new(analysis.call_graph.call_graph),
            program_id_to_lib: Arc::new(BTreeMap::new()),
            gate2_fidelity_threshold: 0.5,
            lcov_path: per_spec_dir.join("lcov.info"),
        };

        let feedback = run_tool
            .invoke(RunSolanaArgs {
                test_stem: None,
                test_filter: None,
                seed: Some(7),
            })
            .await
            .expect("run_solana invoke");

        assert!(
            feedback.contains("[SUCCESS — violation observed]"),
            "expected violation success banner, got:\n{feedback}"
        );
        assert!(
            !feedback.contains("[GATE 1 FAILED"),
            "Gate 1 must pass, got:\n{feedback}"
        );
        assert!(
            !feedback.contains("[GATE 2 FAILED"),
            "Gate 2 Tier 2 must pass, got:\n{feedback}"
        );

        let snap = attempt.snapshot().await;
        assert!(snap.violation_observed);
        assert!(snap.last_gate_passed);
        assert_eq!(snap.last_test_calls, 8);
        assert_eq!(snap.coverage.len(), 1);
        assert_eq!(snap.coverage[0].len(), 2);
    }
}
