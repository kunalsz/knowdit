//! [`SolanaHarness`] — `LiteSVM` + `proptest` / `#[test]` implementation of
//! [`crate::harness::backend::HarnessBackend`] for Rust / Solana programs.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::time::Duration;

use color_eyre::eyre::Result;
use knowdit_repo_model::RepoDatabase;
use llmy::client::client::LLM;

use super::runner::{SolanaBackend, SolanaRunner};
use super::runtime::SolanaHarnessGenerator;
use super::utils::SolanaFuzzOptions;
use crate::harness::backend::HarnessBackend;
use crate::harness::solidity::{
    CodegenRegenInMemory, FuzzOneOutcome, FuzzOutcome, HarnessKind, RegenOutcome,
};
use crate::types::AuditSpecification;

/// Rust / Solana (`LiteSVM` + SBF `.so`) harness backend.
#[derive(Clone)]
pub struct SolanaHarness {
    kind: HarnessKind,
    options: SolanaFuzzOptions,
    solana_backend: SolanaBackend,
}

impl SolanaHarness {
    pub fn new(kind: HarnessKind, options: SolanaFuzzOptions, solana_backend: SolanaBackend) -> Self {
        Self {
            kind,
            options,
            solana_backend,
        }
    }

    pub fn solana_backend(&self) -> &SolanaBackend {
        &self.solana_backend
    }

    fn make_generator(&self, repo: &RepoDatabase, llm: &LLM) -> SolanaHarnessGenerator {
        SolanaHarnessGenerator::new(
            self.kind,
            repo,
            llm,
            &self.options,
            self.solana_backend.clone(),
        )
    }

    fn preflight_harness_dir(&self, repo_root: &Path) -> PathBuf {
        let mut opts = self.options.clone();
        opts.repo_root = repo_root.to_path_buf();
        opts.resolve_harness_dir()
    }

    /// Markdown language-context block prepended to every audit-phase agent
    /// system prompt (`KnowledgeMapper`, `SpecificationGenerator`,
    /// `VerdictGrader`, `SeverityGrader`, `ReviewGrader`, `MergeAgent`, and
    /// `SpecRegen`) when auditing a Rust / Solana repository.
    pub const PROMPT_PREFIX: &'static str = r#"## Project source language

- language: **Rust (Solana / SVM)**
- source file extension: `.rs`
- top-level containers: program modules (`#[program] pub mod ...` / instruction modules) and on-chain account state structs (`#[account] pub struct ...`)
- documentation convention: Rustdoc (`/// ...`) and Anchor account constraint attributes (`#[account(mut, seeds = [...], bump, has_one = ...)]`)
- assertion & failure idiom: `require!`, `require_keys_eq!`, `require_gte!`, `Err(ProgramError::...)`, `?` error propagation, and Rust checked/panicking arithmetic; failed instructions revert the entire Solana transaction atomically
- state & execution model: programs are stateless code; all mutable state lives in passed-in Solana `Account`s (`Account<'info, T>`, PDAs, SPL Token / Token-2022 accounts, lamport balances) validated per instruction. Cross-program calls occur via CPI (`invoke` / `invoke_signed`), and cached deserialized accounts must call `.reload()` after a CPI to observe mutated state.

Use vocabulary consistent with Rust and Solana throughout — `program` / `instruction` / `account` / `PDA` / `signer` / `CPI` — never Solidity or Move terms.
"#;
}

impl HarnessBackend for SolanaHarness {
    fn prompt_prefix(&self) -> &'static str {
        Self::PROMPT_PREFIX
    }

    async fn preflight(&self, repo_root: &Path) -> Result<()> {
        let harness_dir = self.preflight_harness_dir(repo_root);
        let preflight_timeout = Duration::from_secs(self.options.timeout_secs.max(120));
        let runner = SolanaRunner::new(
            self.solana_backend.clone(),
            repo_root.to_path_buf(),
            harness_dir,
            preflight_timeout,
        );
        runner.preflight().await
    }

    async fn fuzz_sweep(&self, repo: &RepoDatabase, llm: &LLM) -> Result<FuzzOutcome> {
        self.make_generator(repo, llm).run().await
    }

    async fn fuzz_one_existing_spec(
        &self,
        repo: &RepoDatabase,
        llm: &LLM,
        spec_id: i32,
    ) -> Result<Option<FuzzOneOutcome>> {
        self.make_generator(repo, llm)
            .fuzz_one_existing_spec(spec_id)
            .await
    }

    async fn regen_codegen_for_spec(
        &self,
        repo: &RepoDatabase,
        llm: &LLM,
        spec_id: i32,
        prior_feedback: String,
    ) -> Result<RegenOutcome> {
        self.make_generator(repo, llm)
            .regen_one_spec(spec_id, prior_feedback)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn codegen_for_in_memory_spec(
        &self,
        repo: &RepoDatabase,
        llm: &LLM,
        extract_id: i32,
        historical_id: i32,
        finding_id: i32,
        spec: AuditSpecification,
        synthetic_spec_id: i32,
        prior_feedback: String,
    ) -> Result<CodegenRegenInMemory> {
        self.make_generator(repo, llm)
            .regen_codegen_with_explicit_spec(
                extract_id,
                historical_id,
                finding_id,
                spec,
                synthetic_spec_id,
                prior_feedback,
            )
            .await
    }
}

#[doc(hidden)]
#[allow(dead_code)]
fn _assert_solana_harness_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SolanaHarness>();
}

const _: fn() = || {
    fn _check<F: Future<Output = Result<()>> + Send>(_: F) {}
};
