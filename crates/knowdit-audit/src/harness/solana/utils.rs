//! Shared options, attempt state, and project convention loaders for `SolanaHarness`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use knowdit_repo_model::{CoverageEntry, HarnessRunRecord};
use knowdit_rust::RustWorkspace;
use llmy::client::settings::LLMSettings;
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

use crate::spec::ProjectIndex;

static SOLANA_SPEC_ATTEMPT_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(super) fn next_solana_attempt_id() -> u64 {
    SOLANA_SPEC_ATTEMPT_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Configuration knobs for one Solana `LiteSVM` fuzz / PoC pass.
/// Every value comes from CLI `clap` structs (zero hidden env reads).
#[derive(Debug, Clone)]
pub struct SolanaFuzzOptions {
    pub repo_root: PathBuf,
    pub harness_dir: Option<PathBuf>,
    pub cargo_bin: Option<PathBuf>,
    pub build_sbf_bin: Option<PathBuf>,
    pub docker_image: String,
    pub mem_cap_bytes: Option<u64>,
    pub timeout_secs: u64,
    pub test_runs: u64,
    pub coverage_runs: u64,
    pub max_agent_steps: usize,
    pub max_specs: usize,
    pub concurrency: usize,
    pub regenerate: bool,
    pub cache_key: String,
    pub debug_prefix: Option<String>,
    pub llm_settings: Option<LLMSettings>,
    pub window_restart_threshold_tokens: Option<usize>,
    pub max_restarts: usize,
    pub gate2_fidelity_threshold: f64,
    pub list_test_files_max_depth: usize,
}

impl SolanaFuzzOptions {
    /// Resolve where per-spec `LiteSVM` harness crates and the shared
    /// `.knowdit-cargo-target` directory live on disk.
    pub fn resolve_harness_dir(&self) -> PathBuf {
        match &self.harness_dir {
            Some(dir) => dir.clone(),
            None => self.repo_root.join("knowdit_solana_harness"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SolanaLoopOutcome {
    Violation,
    StepsExhausted,
    RestartsExhausted,
    AgentError,
}

#[derive(Debug, Default, Clone)]
pub(super) struct SolanaHarnessAttempt {
    pub harness_source: String,
    pub harness_filename: Option<String>,
    pub runs: Vec<RecordedSolanaRun>,
    pub coverage: Vec<Vec<CoverageEntry>>,
    pub violation_observed: bool,
    pub last_gate_passed: bool,
    pub last_test_calls: u64,
    pub restarts: usize,
}

#[derive(Debug, Clone)]
pub(super) struct RecordedSolanaRun {
    pub record: HarnessRunRecord,
    pub coverage_idx: Option<usize>,
}

#[derive(Clone)]
pub(super) struct SolanaAttemptHandle(pub(super) Arc<Mutex<SolanaHarnessAttempt>>);

impl std::fmt::Debug for SolanaAttemptHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SolanaAttemptHandle")
    }
}

impl SolanaAttemptHandle {
    pub(super) fn new() -> Self {
        Self(Arc::new(Mutex::new(SolanaHarnessAttempt::default())))
    }

    pub(super) async fn snapshot(&self) -> SolanaHarnessAttempt {
        self.0.lock().await.clone()
    }

    /// Write `<per_spec_dir>/tests/<filename>` and stamp `harness_source` +
    /// `harness_filename` onto the attempt accumulator.
    pub(super) async fn write_agent_test_file(
        &self,
        per_spec_tests_dir: &Path,
        filename: &str,
        content: &str,
    ) -> std::result::Result<String, llmy::agent::LLMYError> {
        let trimmed = filename.trim();
        if trimmed.is_empty() || trimmed.contains('/') || trimmed.contains('\\') {
            return Ok(
                "error: filename must be a basename (e.g. `test_42.rs`) with no directory separators"
                    .to_string(),
            );
        }
        if !trimmed.ends_with(".rs") {
            return Ok("error: Solana harness filename must end with `.rs`".to_string());
        }
        if content.trim().is_empty() {
            return Ok("error: content must be non-empty".to_string());
        }
        if let Err(e) = tokio::fs::create_dir_all(per_spec_tests_dir).await {
            return Ok(format!(
                "error: failed to create tests dir {}: {e}",
                per_spec_tests_dir.display()
            ));
        }
        let path = per_spec_tests_dir.join(trimmed);
        if let Err(e) = tokio::fs::write(&path, content).await {
            return Ok(format!("error: failed to write {}: {e}", path.display()));
        }
        let mut state = self.0.lock().await;
        state.harness_source = content.to_string();
        state.harness_filename = Some(trimmed.to_string());
        Ok(format!(
            "ok: wrote {} bytes to {}",
            state.harness_source.len(),
            path.display()
        ))
    }
}

/// Workspace metadata, SBF `.so` artifact table, precomputed Anchor 8-byte
/// discriminators, and sample test excerpts mined once per repo.
#[derive(Debug, Clone, Default)]
pub(super) struct SolanaConventions {
    cargo_toml_excerpt: String,
    anchor_toml_excerpt: String,
    programs_table: String,
    discriminators_table: String,
    sample_test_filenames: Vec<String>,
    sample_test_excerpt: String,
    program_id_to_lib: BTreeMap<String, String>,
}

impl SolanaConventions {
    pub(super) fn cargo_toml_excerpt(&self) -> &str {
        &self.cargo_toml_excerpt
    }

    pub(super) fn anchor_toml_excerpt(&self) -> &str {
        &self.anchor_toml_excerpt
    }

    pub(super) fn programs_table(&self) -> &str {
        &self.programs_table
    }

    pub(super) fn discriminators_table(&self) -> &str {
        &self.discriminators_table
    }

    pub(super) fn sample_test_filenames(&self) -> &[String] {
        &self.sample_test_filenames
    }

    pub(super) fn sample_test_excerpt(&self) -> &str {
        &self.sample_test_excerpt
    }

    pub(super) fn program_id_to_lib(&self) -> &BTreeMap<String, String> {
        &self.program_id_to_lib
    }

    pub(super) async fn load(repo_root: &Path, project_index: &ProjectIndex) -> Self {
        let mut conv = Self::default();
        if let Ok(text) = tokio::fs::read_to_string(repo_root.join("Cargo.toml")).await {
            conv.cargo_toml_excerpt = truncate_str(&text, 4_096);
        }
        if let Ok(text) = tokio::fs::read_to_string(repo_root.join("Anchor.toml")).await {
            conv.anchor_toml_excerpt = truncate_str(&text, 4_096);
        }

        if let Ok(ws) = RustWorkspace::discover(repo_root) {
            let mut rows = Vec::new();
            for prog in ws.programs() {
                let so_exists = repo_root
                    .join("target")
                    .join("deploy")
                    .join(&prog.sbf_so_name)
                    .is_file();
                let id_display = prog
                    .declared_program_id
                    .as_deref()
                    .unwrap_or("(use Pubkey::new_unique() unless declare_id! checks self-id)");
                if let Some(pid) = &prog.declared_program_id {
                    conv.program_id_to_lib
                        .insert(pid.clone(), prog.lib_name.clone());
                }
                rows.push(format!(
                    "- crate `{}` (lib `{}`): sbf_so=`{}` (present={}) declared_program_id=`{}`",
                    prog.crate_name, prog.lib_name, prog.sbf_so_name, so_exists, id_display
                ));
            }
            conv.programs_table = rows.join("\n");
        }

        conv.discriminators_table = Self::build_discriminators_table(project_index);
        conv.populate_sample_tests(repo_root).await;
        conv
    }

    fn build_discriminators_table(project_index: &ProjectIndex) -> String {
        let mut ix_lines = Vec::new();
        let mut acct_lines = Vec::new();

        for contract in project_index.call_graph.contracts.values() {
            let is_state = contract.functions.is_empty()
                || contract
                    .description
                    .as_deref()
                    .map(|d| d.contains("account / state struct"))
                    .unwrap_or(false);
            if is_state {
                let disc = Self::sha256_prefix_8(&format!("account:{}", contract.name));
                acct_lines.push(format!("- account `{}` => `{:?}`", contract.name, disc));
            }
            for f in &contract.functions {
                let disc = Self::sha256_prefix_8(&format!("global:{}", f.name));
                ix_lines.push(format!(
                    "- instruction `{}::{}` => `{:?}`",
                    contract.name, f.name, disc
                ));
            }
        }
        ix_lines.truncate(64);
        acct_lines.truncate(64);

        let mut out = String::new();
        if !ix_lines.is_empty() {
            out.push_str("Anchor instruction discriminators (`sha256(\"global:<fn>\")[..8]`):\n");
            out.push_str(&ix_lines.join("\n"));
            out.push('\n');
        }
        if !acct_lines.is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str("Anchor account discriminators (`sha256(\"account:<Struct>\")[..8]`):\n");
            out.push_str(&acct_lines.join("\n"));
            out.push('\n');
        }
        out
    }

    fn sha256_prefix_8(preimage: &str) -> [u8; 8] {
        let mut hasher = Sha256::new();
        hasher.update(preimage.as_bytes());
        let digest = hasher.finalize();
        let mut out = [0u8; 8];
        out.copy_from_slice(&digest[..8]);
        out
    }

    async fn populate_sample_tests(&mut self, repo_root: &Path) {
        let mut rs_tests: Vec<PathBuf> = Vec::new();
        let mut other_tests: Vec<PathBuf> = Vec::new();
        for sub in ["tests", "test", "programs"] {
            let dir = repo_root.join(sub);
            if !dir.is_dir() {
                continue;
            }
            let accept_dir = |d: &Path| -> bool {
                d.file_name()
                    .and_then(|n| n.to_str())
                    .map(|name| {
                        !matches!(
                            name,
                            "target"
                                | "node_modules"
                                | ".git"
                                | ".anchor"
                                | "knowdit_solana_harness"
                        )
                    })
                    .unwrap_or(true)
            };
            let accept_file = |f: &Path| -> bool {
                let ext = f.extension().and_then(|e| e.to_str()).unwrap_or("");
                if sub == "programs" {
                    let fname = f.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    ext == "rs" && fname.contains("test")
                } else {
                    matches!(ext, "rs" | "ts" | "js")
                }
            };
            if let Ok(found) =
                knowdit_project::scope::walk_files(&dir, &accept_dir, &accept_file, Some(4))
            {
                for p in found {
                    if p.extension().and_then(|e| e.to_str()) == Some("rs") {
                        rs_tests.push(p);
                    } else {
                        other_tests.push(p);
                    }
                }
            }
        }
        rs_tests.sort();
        rs_tests.dedup();
        other_tests.sort();
        other_tests.dedup();

        for p in rs_tests.iter().chain(other_tests.iter()).take(8) {
            let rel = p
                .strip_prefix(repo_root)
                .unwrap_or(p)
                .to_string_lossy()
                .into_owned();
            self.sample_test_filenames.push(rel);
        }

        if let Some(first_rs) = rs_tests.first().or_else(|| other_tests.first())
            && let Ok(text) = tokio::fs::read_to_string(first_rs).await
        {
            self.sample_test_excerpt = truncate_str(&text, 2_048);
        }
    }
}

/// UTF-8-safe prefix truncation (`max` bytes) stored in `HarnessRunRecord`.
pub(super) fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{} … [{} bytes truncated]", &s[..end], s.len() - end)
}

/// UTF-8-safe head+tail truncation for persisted run output. Under the
/// `max` budget the FIRST 3/4 bytes are kept (compile header, progress) and
/// the LAST 1/4 bytes are kept verbatim (the counterexample / assertion
/// revert forge appends at the very end), with the elided middle replaced by
/// a marker. Head-only truncation destroys the failure tail, which is the
/// only part graders read via their 4K tail slices.
pub(super) fn truncate_str_head_tail(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let tail_len = (max / 4).max(1);
    let head_len = max - tail_len;
    let mut head_end = head_len;
    while head_end > 0 && !s.is_char_boundary(head_end) {
        head_end -= 1;
    }
    let tail_start = s.len() - tail_len;
    let mut tail_start = tail_start;
    while tail_start < s.len() && !s.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    let elided = if tail_start > head_end {
        tail_start - head_end
    } else {
        // Degenerate (max too small for a head+tail split): fall back to
        // head-only truncation.
        return truncate_str(s, max);
    };
    format!(
        "{}\n… [{} bytes truncated] …\n{}",
        &s[..head_end],
        elided,
        &s[tail_start..]
    )
}

/// UTF-8-safe suffix tail (`n` bytes) returned in agent tool feedback.
pub(super) fn tail_str(s: &str, n: usize) -> String {
    if s.len() <= n {
        return s.to_string();
    }
    let cut = s.len() - n;
    let mut start = cut;
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    format!("…[{} bytes earlier]\n{}", cut, &s[start..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use knowdit_repo_model::{inheritance::InheritanceGraph, storage::StorageGraph};

    #[tokio::test]
    async fn loads_solana_conventions_and_precomputes_anchor_discriminators() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(tmp.path().join("programs/vault/src")).expect("mkdir");
        std::fs::write(
            tmp.path().join("Anchor.toml"),
            "[programs.localnet]\nsolana_vault = \"VaULt11111111111111111111111111111111111111\"\n",
        )
        .expect("write Anchor.toml");
        std::fs::write(
            tmp.path().join("Cargo.toml"),
            "[workspace]\nmembers = [\"programs/*\"]\n",
        )
        .expect("write Cargo.toml");
        std::fs::write(
            tmp.path().join("programs/vault/Cargo.toml"),
            "[package]\nname = \"solana-vault\"\nversion = \"0.1.0\"\nedition = \"2021\"\n[lib]\ncrate-type = [\"cdylib\"]\nname = \"solana_vault\"\n",
        )
        .expect("write crate Cargo.toml");
        let lib_src = "#[program]\npub mod solana_vault {\n    pub fn withdraw() {}\n}\n#[account]\npub struct Vault {\n    pub assets: u64,\n}\n";
        std::fs::write(tmp.path().join("programs/vault/src/lib.rs"), lib_src)
            .expect("write lib.rs");

        let analysis =
            knowdit_rust::RustStaticAnalyzer::analyze(&[knowdit_rust::RustSourceInput {
                relative_path: PathBuf::from("programs/vault/src/lib.rs"),
                content: lib_src.to_string(),
            }])
            .expect("analyze");
        let index = ProjectIndex::build(
            analysis.call_graph.call_graph,
            StorageGraph::default(),
            InheritanceGraph::default(),
        );
        let conv = SolanaConventions::load(tmp.path(), &index).await;

        assert!(conv.programs_table().contains("solana_vault.so"));
        assert_eq!(
            conv.program_id_to_lib()
                .get("VaULt11111111111111111111111111111111111111")
                .map(String::as_str),
            Some("solana_vault")
        );
        assert!(
            conv.discriminators_table().contains(
                "instruction `solana_vault::withdraw` => `[183, 18, 70, 156, 148, 109, 161, 34]`"
            ),
            "got discriminators:\n{}",
            conv.discriminators_table()
        );
        assert!(conv.discriminators_table().contains("account `Vault` =>"));
    }
}
