//! Solidity-layer agent tools, both shared between the two harness
//! modes. Each tool struct is a `llmy::agent::tool!`-generated type
//! plumbed through [`super::runtime`]'s `AgentLoopState::build_toolbox`.
//!
//! Why one file: the tool types are small data holders + an `invoke`
//! async fn; keeping them together (away from the agent loop and the
//! mode-specific prompts) makes it easy to find every place the agent
//! is allowed to touch.
//!
//! Three tools live here today:
//!
//! * [`WriteHarnessFileTool`] — writes (or overwrites) the per-spec
//!   `.sol` harness file and stamps the new source/filename onto the
//!   shared [`AttemptHandle`] so the orchestrator can persist it as
//!   `CodeGenRecord.harness_source` and reference it from the restart
//!   bootstrap.
//! * [`RunForgeTool`] — runs `forge test --json`, parses the digest,
//!   auto-chases coverage when the test ran cleanly, and emits Gate 1
//!   / Gate 2 verdicts back to the agent in one combined tool result.
//! * [`ListTestFilesTool`] — depth-limited recursive scan of the
//!   project's `test/` / `script/` / `scripts/` / `deploy/` /
//!   `deployments/` directories so the agent can imitate existing
//!   setUp patterns.

use std::path::PathBuf;
use std::sync::Arc;

use knowdit_repo_model::{CoverageEntry, HarnessRunRecord, RunKind};
use schemars::JsonSchema;
use serde::Deserialize;

use super::utils::{
    AttemptHandle, ForgeTestSummary, RecordedRun, parse_lcov, tail_str, truncate_str_head_tail,
};
use crate::harness::forge::ForgeRunner;
use crate::types::AuditSpecification;

// =============================================================================
// write_harness_file
// =============================================================================

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub(super) struct WriteHarnessFileArgs {
    /// Filename, just the base (no directory). e.g. `"Test_42.t.sol"`.
    /// Re-using the same name overwrites the file on disk.
    filename: String,
    /// Full Solidity source. The required file shape is described by
    /// the system prompt.
    content: String,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = WriteHarnessFileArgs,
    invoke = invoke,
    description = "Write (or overwrite) the Solidity test file under the per-spec harness directory. The required file shape is dictated entirely by the system prompt — this tool just writes the bytes you provide and remembers the filename for restart bookkeeping. Re-call to update the file after a forge compile error.",
    name = "write_harness_file",
)]
pub(super) struct WriteHarnessFileTool {
    pub(super) attempt: AttemptHandle,
    pub(super) harness_dir: PathBuf,
}

impl WriteHarnessFileTool {
    async fn invoke(
        &self,
        args: WriteHarnessFileArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        self.attempt
            .write_agent_file(&self.harness_dir, &args.filename, &args.content)
            .await
    }
}

// =============================================================================
// run_forge
// =============================================================================

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub(super) struct RunForgeArgs {
    pub(super) match_contract: String,
    #[serde(default)]
    pub(super) match_test: Option<String>,
    #[serde(default)]
    pub(super) seed: Option<i64>,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = RunForgeArgs,
    invoke = invoke,
    description = "Run `forge test --json -vvvvvv` against the harness; if the test exits cleanly and actually exercises the harness (calls > 0, or a deterministic test_… function ran) the orchestrator immediately follows up with `forge coverage --report lcov` server-side and the result includes the coverage gate verdict. Use this whenever you finish writing the harness file. Returns gate verdicts plus the captured exit codes and tails of stdout/stderr.",
    name = "run_forge",
)]
pub(super) struct RunForgeTool {
    pub(super) attempt: AttemptHandle,
    pub(super) runner: ForgeRunner,
    pub(super) test_runs: u64,
    pub(super) coverage_runs: u64,
    pub(super) match_path_glob: String,
    pub(super) spec: Arc<AuditSpecification>,
    pub(super) callgraph: Arc<knowdit_repo_model::cg::CallGraph>,
    pub(super) gate2_fidelity_threshold: f64,
    pub(super) via_ir: bool,
    /// Absolute path `forge coverage` writes its lcov report to, and this
    /// tool reads back from. Lives inside the per-spec scratch dir so
    /// concurrent links never clobber each other's coverage report —
    /// forge's default `<work_dir>/lcov.info` is shared by every link.
    pub(super) lcov_path: PathBuf,
}

impl RunForgeTool {
    async fn invoke(
        &self,
        args: RunForgeArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let test_argv = self.build_test_argv(&args);
        let test_out = match self.runner.run(&test_argv).await {
            Ok(o) => o,
            Err(e) => return Ok(format!("error: forge test spawn failed: {e:#}")),
        };
        let test_summary = ForgeTestSummary::parse(&test_out.stdout);
        let test_record = HarnessRunRecord {
            kind: RunKind::Test,
            seed: args.seed,
            runs: self.test_runs as i64,
            forge_args: test_argv.clone(),
            exit_code: test_out.exit_code,
            // Head+tail, not head-only: `-vvvvvv` runs reach 10-400 MB and
            // the counterexample / assertion revert sit at the very end —
            // graders slice the LAST 4K of this field, so head-only
            // truncation showed them bytes ~196K..200K of mid-setup noise.
            stdout: truncate_str_head_tail(&test_out.stdout, 200_000),
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

        // Gate-1-conditional lazy coverage (U4):
        // 1. Never spawn `forge coverage` when Gate 1 failed (`test_gate_verdict.is_some()`)
        //    or `forge test` timed out.
        // 2. Skip duplicate `forge coverage` runs when the `.sol` source SHA-256
        //    is unchanged since the last completed coverage run, reusing the
        //    cached Gate 2 verdict.
        // 3. When `test_summary.violated == true` and the current `.sol` hash
        //    has no recorded coverage yet, run the fast 256-run `forge coverage`
        //    once so `VerdictGrader` retains LCOV line-coverage context.
        let (source_hash, cached_hit) = {
            let guard = self.attempt.0.lock().await;
            let hash = guard.source_sha256();
            let hit = match (
                hash,
                guard.last_covered_source_sha256,
                &guard.cached_gate2_verdict,
            ) {
                (Some(h), Some(prev), Some(cached_verdict)) if h == prev => {
                    Some((cached_verdict.clone(), guard.last_coverage_row_count))
                }
                _ => None,
            };
            (hash, hit)
        };

        let eligible_for_coverage = test_gate_verdict.is_none()
            && !test_out.timed_out
            && (test_summary.violated || (test_out.exit_code == 0 && test_summary.total_calls > 0));
        let run_coverage = eligible_for_coverage && cached_hit.is_none();

        let mut cov_summary: Option<CoverageBlockBundle> = None;
        if run_coverage {
            let cov_argv = self.build_coverage_argv(&args);
            match self.runner.run(&cov_argv).await {
                Ok(out) => {
                    let coverage = match tokio::fs::read_to_string(&self.lcov_path).await {
                        Ok(text) => parse_lcov(&text),
                        Err(_) => Vec::new(),
                    };
                    let gate = if coverage.is_empty() {
                        Some(
                            "[GATE 2 FAILED — coverage] no lcov rows produced (forge coverage ran \
                             but emitted nothing)"
                                .to_string(),
                        )
                    } else if test_summary.deterministic_test_failure {
                        // PoC mode: coverage is informational, not gated.
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
                    let cov_record = HarnessRunRecord {
                        kind: RunKind::Coverage,
                        seed: args.seed,
                        runs: self.coverage_runs as i64,
                        forge_args: cov_argv.clone(),
                        exit_code: out.exit_code,
                        stdout: truncate_str_head_tail(&out.stdout, 200_000),
                        stderr: truncate_str_head_tail(&out.stderr, 200_000),
                        duration_ms: out.duration_ms,
                        violated: false,
                        sequence: None,
                    };
                    cov_summary = Some(CoverageBlockBundle {
                        record: cov_record,
                        coverage,
                        gate_verdict: gate,
                        timed_out: out.timed_out,
                    });
                }
                Err(e) => {
                    tracing::warn!("auto-coverage spawn failed: {e:#}");
                }
            }
        }

        // Gate enforcement: a violated run only terminates the loop when
        // BOTH inline gates passed. Gate 1 failure = `test_gate_verdict`;
        // Gate 2 failure comes from this run's coverage block, or from the
        // cached verdict when coverage was skipped for an unchanged source
        // hash. Broken / mocked / disjoint harnesses therefore keep
        // iterating instead of terminating early and flooding the graders.
        let gate2_failed = match &cov_summary {
            Some(block) => block.gate_verdict.is_some(),
            None => cached_hit
                .as_ref()
                .is_some_and(|(verdict, _)| verdict.is_some()),
        };
        let violation_accepted =
            test_summary.violated && test_gate_verdict.is_none() && !gate2_failed;
        let mut state = self.attempt.0.lock().await;
        state.runs.push(RecordedRun {
            record: test_record.clone(),
            coverage_idx: None,
        });
        if violation_accepted {
            state.violation_observed = true;
        }
        state.last_test_calls = test_summary.total_calls.max(0) as u64;
        if let Some(block) = &cov_summary {
            let coverage_idx = if !block.coverage.is_empty() {
                state.coverage.push(block.coverage.clone());
                Some(state.coverage.len() - 1)
            } else {
                None
            };
            state.runs.push(RecordedRun {
                record: block.record.clone(),
                coverage_idx,
            });
            state.last_gate_passed = block.gate_verdict.is_none() && !block.coverage.is_empty();
            if !block.timed_out
                && let Some(hash) = source_hash
            {
                state.last_covered_source_sha256 = Some(hash);
                state.cached_gate2_verdict = Some(block.gate_verdict.clone());
                state.last_coverage_row_count = block.coverage.len();
            }
        } else if eligible_for_coverage && let Some((ref cached_verdict, cached_rows)) = cached_hit
        {
            state.last_gate_passed = cached_verdict.is_none() && cached_rows > 0;
        } else {
            state.last_gate_passed = false;
        }
        drop(state);

        let mut sections: Vec<String> = Vec::new();
        if let Some(v) = &test_gate_verdict {
            sections.push(v.clone());
        }
        if test_summary.violated {
            let kind = if test_summary.deterministic_test_failure {
                "deterministic test reverted with KNOWDIT_POST_ATTACK_REACHED"
            } else {
                "invariant counter-example produced"
            };
            if violation_accepted {
                sections.push(format!(
                    "[SUCCESS — violation observed] {kind}. The orchestrator will terminate after \
                     this turn."
                ));
            } else if test_gate_verdict.is_some() {
                sections.push(
                    "[VIOLATION REJECTED BY GATE 1] Your run reported a violation, but Gate 1 \
                     failed (see the gate verdict above). Fix the harness and call run_forge \
                     again."
                        .to_string(),
                );
            } else {
                sections.push(
                    "[VIOLATION REJECTED BY GATE 2] Your run reported a violation, but Gate 2 \
                     coverage failed (see the gate verdict above). Fix the harness and call \
                     run_forge again."
                        .to_string(),
                );
            }
        } else if test_summary.total_calls == 0 && !test_summary.deterministic_test_failure {
            sections.push(
                "[NO-OP] forge test ran but produced 0 calls AND no deterministic test_… revert. \
                 Common causes: (a) `No tests found in project` — your harness is outside the \
                 project's configured test path; (b) `setUp()` reverted before any sequence call \
                 ran; (c) `--match-test` filter excluded every test. Fix the discovery / setUp \
                 issue and call run_forge again."
                    .to_string(),
            );
        }
        sections.push(format!(
            "ok: forge test exit={}{} duration_ms={}. runtime: setup_failed={} calls={} \
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
            "--- forge test stdout (last 4K) ---\n{}",
            tail_str(&test_record.stdout, 4_000)
        ));
        sections.push(format!(
            "--- forge test stderr (last 4K) ---\n{}",
            tail_str(&test_record.stderr, 4_000)
        ));

        if let Some(block) = &cov_summary {
            if let Some(v) = &block.gate_verdict {
                sections.push(v.clone());
            }
            sections.push(format!(
                "ok: forge coverage exit={}{} duration_ms={}. runtime: {} lcov rows",
                block.record.exit_code,
                if block.timed_out { " [TIMEOUT]" } else { "" },
                block.record.duration_ms,
                block.coverage.len(),
            ));
            sections.push(format!(
                "--- forge coverage stdout (last 4K) ---\n{}",
                tail_str(&block.record.stdout, 4_000)
            ));
        } else if eligible_for_coverage && let Some((cached_verdict, cached_rows)) = &cached_hit {
            if let Some(v) = cached_verdict {
                sections.push(v.clone());
            }
            sections.push(format!(
                "note: forge coverage skipped (harness .sol SHA-256 unchanged; reusing cached Gate 2 verdict over {cached_rows} lcov rows)."
            ));
        } else if run_coverage {
            sections.push(
                "note: orchestrator attempted to run forge coverage but the spawn failed; \
                 coverage gate skipped this turn."
                    .to_string(),
            );
        } else {
            sections.push(
                "note: forge coverage was skipped this turn (Gate 1 failed, no clean exit, or no project \
                 execution). Fix the test run first."
                    .to_string(),
            );
        }

        Ok(sections.join("\n"))
    }

    fn build_test_argv(&self, args: &RunForgeArgs) -> Vec<String> {
        let mut argv = vec![
            "test".to_string(),
            "--json".to_string(),
            "-vvvvvv".to_string(),
            "--fuzz-runs".to_string(),
            self.test_runs.to_string(),
            "--match-contract".to_string(),
            args.match_contract.clone(),
        ];
        if let Some(t) = &args.match_test {
            argv.push("--match-test".to_string());
            argv.push(t.clone());
        }
        argv.push("--match-path".to_string());
        argv.push(self.match_path_glob.clone());
        if let Some(seed) = args.seed {
            argv.push("--fuzz-seed".to_string());
            argv.push(seed.to_string());
        }
        argv
    }

    fn build_coverage_argv(&self, args: &RunForgeArgs) -> Vec<String> {
        let mut argv = vec![
            "coverage".to_string(),
            "--report".to_string(),
            "lcov".to_string(),
            // Write the report to the per-spec scratch dir. Without this,
            // forge defaults to `<work_dir>/lcov.info`, which every
            // concurrent link shares — two links running coverage at once
            // would read back each other's rows and mis-grade Gate 2.
            "--report-file".to_string(),
            self.lcov_path.display().to_string(),
            "--fuzz-runs".to_string(),
            self.coverage_runs.to_string(),
        ];
        if self.via_ir && self.runner.features().coverage_force_via_ir {
            argv.push("--force-via-ir".to_string());
        }
        argv.push("--match-contract".to_string());
        argv.push(args.match_contract.clone());
        if let Some(t) = &args.match_test {
            argv.push("--match-test".to_string());
            argv.push(t.clone());
        }
        argv.push("--match-path".to_string());
        argv.push(self.match_path_glob.clone());
        if let Some(seed) = args.seed {
            argv.push("--fuzz-seed".to_string());
            argv.push(seed.to_string());
        }
        argv
    }
}

/// Intermediate bundle held inside `RunForgeTool::invoke` between the
/// coverage spawn and the eventual feedback assembly.
struct CoverageBlockBundle {
    record: HarnessRunRecord,
    coverage: Vec<CoverageEntry>,
    gate_verdict: Option<String>,
    timed_out: bool,
}

// =============================================================================
// list_test_files
// =============================================================================

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub(super) struct ListTestFilesArgs {
    #[serde(default)]
    pub(super) filter: Option<String>,
}

#[derive(Debug, Clone)]
#[llmy::agent::tool(
    arguments = ListTestFilesArgs,
    invoke = invoke,
    description = "List existing project test/script files (Foundry .t.sol, Hardhat .ts/.js, etc.) under the repo's `test/`, `script/`, `scripts/`, and `deploy/` directories. Use this to find existing setUp() / deployment patterns to imitate before writing your harness's setUp(). Returns relative paths.",
    name = "list_test_files",
)]
pub(super) struct ListTestFilesTool {
    pub(super) repo_root: PathBuf,
    /// Recursion budget passed to [`walk_collect`]. Wired from
    /// `FuzzOptions::list_test_files_max_depth`.
    pub(super) max_depth: usize,
}

/// Vendored / build-output directory names skipped during the
/// `list_test_files` scan. Hardcoded because they're project-tooling
/// conventions (npm, foundry, hardhat), not knowdit knobs. Includes
/// knowdit's own harness directory (`<repo>/test/knowdit_harness`): it
/// lives under `test/`, so leaving it in would expose prior LLM harnesses
/// as "existing project tests" and later agents would imitate their
/// hallucinated setUp() patterns.
const LIST_TEST_FILES_SKIP_DIRS: &[&str] = &[
    "node_modules",
    "lib",
    "out",
    "cache",
    "artifacts",
    ".git",
    "knowdit_harness",
];

/// File extensions the `list_test_files` scan collects. Anything the
/// agent might want to imitate when authoring a harness setUp(): solidity
/// test sources, hardhat ts/js tests, deploy-script json metadata.
const LIST_TEST_FILES_COLLECT_EXTS: &[&str] = &["sol", "ts", "js", "json"];

impl ListTestFilesTool {
    async fn invoke(
        &self,
        args: ListTestFilesArgs,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let accept_dir = |dir: &std::path::Path| -> bool {
            dir.file_name()
                .and_then(|n| n.to_str())
                .map(|name| !LIST_TEST_FILES_SKIP_DIRS.contains(&name))
                .unwrap_or(true)
        };
        let accept_file = |file: &std::path::Path| -> bool {
            file.extension()
                .and_then(|e| e.to_str())
                .map(|ext| LIST_TEST_FILES_COLLECT_EXTS.contains(&ext))
                .unwrap_or(false)
        };
        let mut found: Vec<PathBuf> = Vec::new();
        for sub in ["test", "script", "scripts", "deploy", "deployments"] {
            let dir = self.repo_root.join(sub);
            if !dir.is_dir() {
                continue;
            }
            match knowdit_project::scope::walk_files(
                &dir,
                &accept_dir,
                &accept_file,
                Some(self.max_depth),
            ) {
                Ok(set) => found.extend(set),
                Err(e) => {
                    tracing::warn!(dir = %dir.display(), "list_test_files: walk failed: {e:#}");
                }
            }
        }
        let filter = args.filter.unwrap_or_default();
        let mut out = String::new();
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
        }
        if out.is_empty() {
            return Ok(format!(
                "no test files found{}",
                if filter.is_empty() {
                    String::new()
                } else {
                    format!(" matching `{filter}`")
                }
            ));
        }
        Ok(format!("ok: {} file(s)\n{out}", found.len()))
    }
}

#[cfg(test)]
mod lazy_coverage_tests {
    use super::*;
    use crate::harness::forge::{ForgeBackend, ForgeBackendKind, ForgeFeatures, ForgeSlotPool};
    use knowdit_repo_model::cg::{CallGraph, Contract, FileChunk, FileLocation, Function};
    use std::collections::BTreeMap;
    use std::os::unix::fs::PermissionsExt;
    use std::time::Duration;

    fn sample_callgraph_and_spec() -> (Arc<CallGraph>, Arc<AuditSpecification>) {
        let loc = FileLocation {
            start_line: 1,
            start_column: 0,
            end_line: 10,
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
                    content: String::new(),
                },
                functions: vec![Function {
                    id: 1,
                    name: "withdraw".to_string(),
                    args: String::new(),
                    relative_file_path: PathBuf::from("src/Vault.sol"),
                    loc,
                    content: Some("function withdraw() external {}".to_string()),
                    calls: Vec::new(),
                    description: None,
                }],
                description: None,
            },
        );
        let mut spec = AuditSpecification::default();
        spec.sequence.push(crate::types::SuqenceCallStep {
            contract: "Vault".to_string(),
            function: "withdraw".to_string(),
            intention: "drain".to_string(),
        });
        (
            Arc::new(CallGraph {
                contracts,
                interfaces: BTreeMap::new(),
            }),
            Arc::new(spec),
        )
    }

    #[tokio::test]
    async fn run_forge_skips_coverage_on_gate1_fail_and_caches_by_source_sha256()
    -> color_eyre::eyre::Result<()> {
        let dir = tempfile::tempdir()?;
        let work_dir = dir.path().to_path_buf();
        let lcov_path = work_dir.join("lcov.info");
        let cov_counter = work_dir.join("cov_count.txt");
        let mode_file = work_dir.join("mode.txt");

        // Fake `forge` script driven by `mode.txt`:
        // - `gate1_fail`: exit 0, calls=100, reverts=99 (99% revert rate -> Gate 1 fails!)
        // - `clean_pass`: exit 0, calls=100, reverts=0 (Gate 1 passes!)
        // - `violated`: exit 1, deterministic `KNOWDIT_POST_ATTACK_REACHED:` violation!
        let script_path = work_dir.join("fake_forge.sh");
        let script = format!(
            r#"#!/usr/bin/env bash
sub="$1"
if [ "$sub" = "coverage" ]; then
  echo "1" >> "{cov}"
  cat <<'EOF' > "{lcov}"
SF:src/Vault.sol
DA:3,5
end_of_record
EOF
  exit 0
fi
mode=$(cat "{mode}")
if [ "$mode" = "gate1_fail" ]; then
  echo '{{"test/H.t.sol:H":{{"test_results":{{"invariant_x()":{{"status":"Success","reason":null,"kind":{{"Invariant":{{"runs":10,"calls":100,"reverts":99}}}}}}}}}}}}'
  exit 0
elif [ "$mode" = "violated" ]; then
  echo '{{"test/H.t.sol:H":{{"test_results":{{"test_poc()":{{"status":"Failure","reason":"KNOWDIT_POST_ATTACK_REACHED: drained","kind":{{"Standard":0}}}}}}}}}}'
  exit 1
else
  echo '{{"test/H.t.sol:H":{{"test_results":{{"invariant_x()":{{"status":"Success","reason":null,"kind":{{"Invariant":{{"runs":10,"calls":100,"reverts":0}}}}}}}}}}}}'
  exit 0
fi
"#,
            cov = cov_counter.display(),
            lcov = lcov_path.display(),
            mode = mode_file.display(),
        );
        tokio::fs::write(&script_path, script).await?;
        let mut perms = tokio::fs::metadata(&script_path).await?.permissions();
        perms.set_mode(0o755);
        tokio::fs::set_permissions(&script_path, perms).await?;

        let runner = ForgeRunner::new(
            ForgeBackend {
                kind: ForgeBackendKind::LocalDirect {
                    binary: script_path,
                },
                features: ForgeFeatures::default(),
                slot_pool: Arc::new(ForgeSlotPool::new(2)),
            },
            work_dir.clone(),
            Duration::from_secs(5),
        );
        let attempt = AttemptHandle::new();
        let _ = attempt
            .write_agent_file(&work_dir, "H.t.sol", "contract H_v1 {}")
            .await?;

        let (callgraph, spec) = sample_callgraph_and_spec();
        let tool = RunForgeTool {
            attempt: attempt.clone(),
            runner,
            test_runs: 100,
            coverage_runs: 256,
            match_path_glob: "test/*".to_string(),
            spec,
            callgraph,
            gate2_fidelity_threshold: 0.5,
            via_ir: false,
            lcov_path,
        };
        let args = RunForgeArgs {
            match_contract: "H".to_string(),
            match_test: None,
            seed: None,
        };

        // 1. Gate 1 fails -> MUST NOT spawn `forge coverage`!
        tokio::fs::write(&mode_file, "gate1_fail").await?;
        let out1 = tool.invoke(args.clone()).await?;
        assert!(out1.contains("[GATE 1 FAILED"), "{out1}");
        assert!(
            !cov_counter.exists(),
            "forge coverage must not run when Gate 1 fails"
        );

        // 2. Clean pass -> spawns `forge coverage` once (count == 1).
        tokio::fs::write(&mode_file, "clean_pass").await?;
        let out2 = tool.invoke(args.clone()).await?;
        assert!(out2.contains("ok: forge coverage"), "{out2}");
        assert_eq!(
            tokio::fs::read_to_string(&cov_counter)
                .await?
                .lines()
                .count(),
            1
        );
        assert!(attempt.snapshot().await.last_gate_passed);

        // 3. Second clean pass with UNCHANGED `.sol` SHA-256 -> skips `forge coverage`!
        let out3 = tool.invoke(args.clone()).await?;
        assert!(out3.contains("SHA-256 unchanged"), "{out3}");
        assert_eq!(
            tokio::fs::read_to_string(&cov_counter)
                .await?
                .lines()
                .count(),
            1,
            "unchanged .sol must reuse cached Gate 2 verdict without spawning forge coverage"
        );
        assert!(attempt.snapshot().await.last_gate_passed);

        // 4. Edit `.sol` to v2 + trigger `violated` -> runs 256-run coverage once for new hash!
        let _ = attempt
            .write_agent_file(&work_dir, "H.t.sol", "contract H_v2 {}")
            .await?;
        tokio::fs::write(&mode_file, "violated").await?;
        let out4 = tool.invoke(args).await?;
        assert!(out4.contains("[SUCCESS — violation observed]"), "{out4}");
        assert_eq!(
            tokio::fs::read_to_string(&cov_counter)
                .await?
                .lines()
                .count(),
            2,
            "uncovered violated .sol must run fast coverage once so reflect retains LCOV rows"
        );
        Ok(())
    }
}
