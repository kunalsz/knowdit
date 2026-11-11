//! `SolanaProject` — static-analysis view of a Rust / Solana / Anchor repository.
//!
//! Mirrors [`crate::FoundryProject`] and [`crate::HardhatProject`]: discovers
//! Cargo / Anchor workspace manifests, walks in-scope `.rs` files (honouring
//! `scope.txt` or CLI glob overrides + `RUST_VENDORED_EXCLUDES`), runs
//! `knowdit-rust`'s static AST / call-graph / account-storage analyzer once in
//! [`SolanaProject::build`], and caches the owned graphs behind plain accessors.

use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, WrapErr, ensure};
use knowdit_repo_model::{cg::CallGraph, inheritance::InheritanceGraph, storage::StorageGraph};
use knowdit_rust::{
    ExtractedRustContainer, RustCallGraphResult, RustProgramManifest, RustSourceInput,
    RustStaticAnalysis, RustStaticAnalyzer, RustStorageResult, RustWorkspace,
};

use crate::{data::RUST_VENDORED_EXCLUDES, scope::ProjectScope};

/// Tunables for [`SolanaProject::build`]. All fields are runtime-configurable
/// via clap structs in the `knowdit` CLI crate.
#[derive(Debug, Clone, Default)]
pub struct SolanaOptions {
    /// Explicit include globs overriding `scope.txt` and the default `**/*.rs`.
    pub include_globs: Vec<String>,
    /// Explicit exclude globs overriding [`RUST_VENDORED_EXCLUDES`]. When empty,
    /// [`RUST_VENDORED_EXCLUDES`] is used.
    pub exclude_globs: Vec<String>,
    /// Ignore a root `scope.txt` file even when present.
    pub ignore_scope_file: bool,
}

/// Owned static-analysis output for a Rust / Solana repository.
#[derive(Debug, Clone)]
pub struct SolanaProject {
    repo_root: PathBuf,
    workspace: RustWorkspace,
    source_files: Vec<PathBuf>,
    analysis: RustStaticAnalysis,
}

impl SolanaProject {
    /// Discover Cargo / Anchor manifests under `repo_root`, read in-scope `.rs`
    /// files, and run [`RustStaticAnalyzer::analyze`] to construct the call
    /// graph, storage/account graph, and inheritance graph.
    pub async fn build(repo_root: PathBuf, options: SolanaOptions) -> Result<Self> {
        let repo_root = repo_root.canonicalize().wrap_err_with(|| {
            format!(
                "failed to canonicalize Rust/Solana repo root {}",
                repo_root.display()
            )
        })?;
        ensure!(
            repo_root.is_dir(),
            "repo root {} is not a directory",
            repo_root.display(),
        );

        let workspace = RustWorkspace::discover(&repo_root)?;
        let scope = Self::resolve_scope(&repo_root, &options)?;
        ensure!(
            !scope.is_empty(),
            "no in-scope Rust (.rs) source files found under {}",
            repo_root.display()
        );

        let content = scope.read().await?;
        let source_files: Vec<PathBuf> = content
            .files()
            .iter()
            .map(|f| f.relative_path.clone())
            .collect();
        let inputs: Vec<RustSourceInput> = content
            .into_files()
            .into_iter()
            .map(|f| RustSourceInput {
                relative_path: f.relative_path,
                content: f.content,
            })
            .collect();

        let analysis = tokio::task::spawn_blocking(move || RustStaticAnalyzer::analyze(&inputs))
            .await
            .wrap_err("Rust static analyzer task panicked")??;

        Ok(Self {
            repo_root,
            workspace,
            source_files,
            analysis,
        })
    }

    fn resolve_scope(repo_root: &Path, options: &SolanaOptions) -> Result<ProjectScope> {
        let custom_excludes: Vec<&str> = options.exclude_globs.iter().map(String::as_str).collect();
        let excludes: &[&str] = if custom_excludes.is_empty() {
            &RUST_VENDORED_EXCLUDES
        } else {
            &custom_excludes
        };

        if !options.include_globs.is_empty() {
            let includes: Vec<&str> = options.include_globs.iter().map(String::as_str).collect();
            let raw = ProjectScope::build_from_patterns(repo_root.to_path_buf(), &includes, excludes)?;
            return raw.filter_by_patterns(&["**/*.rs"], &[]);
        }

        if !options.ignore_scope_file
            && let Some(scope_globs) = crate::scope::read_scope_txt(repo_root)?
            && !scope_globs.is_empty()
        {
            let includes: Vec<&str> = scope_globs.iter().map(String::as_str).collect();
            let raw = ProjectScope::build_from_patterns(repo_root.to_path_buf(), &includes, excludes)?;
            return raw.filter_by_patterns(&["**/*.rs"], &[]);
        }

        ProjectScope::build_from_patterns(repo_root.to_path_buf(), &["**/*.rs"], excludes)
    }

    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }

    pub fn workspace(&self) -> &RustWorkspace {
        &self.workspace
    }

    pub fn programs(&self) -> Vec<&RustProgramManifest> {
        self.workspace.programs()
    }

    pub fn source_files(&self) -> &[PathBuf] {
        &self.source_files
    }

    pub fn containers(&self) -> &[ExtractedRustContainer] {
        &self.analysis.containers
    }

    pub fn call_graph(&self) -> &CallGraph {
        &self.analysis.call_graph.call_graph
    }

    pub fn storage_graph(&self) -> &StorageGraph {
        &self.analysis.storage.storage
    }

    pub fn inheritance_graph(&self) -> &InheritanceGraph {
        &self.analysis.storage.inheritance
    }

    pub fn call_graph_result(&self) -> &RustCallGraphResult {
        &self.analysis.call_graph
    }

    pub fn storage_result(&self) -> &RustStorageResult {
        &self.analysis.storage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_file(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).expect("mkdir");
        }
        std::fs::write(p, body).expect("write");
    }

    #[tokio::test]
    async fn builds_anchor_workspace_and_discovers_program_id_and_graphs() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_file(
            tmp.path(),
            "Anchor.toml",
            r#"
[programs.localnet]
vault_prog = "VaULt11111111111111111111111111111111111111"
"#,
        );
        write_file(
            tmp.path(),
            "Cargo.toml",
            r#"
[workspace]
members = ["programs/*", "libs/*"]
"#,
        );
        write_file(
            tmp.path(),
            "programs/vault-prog/Cargo.toml",
            r#"
[package]
name = "vault-prog"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "lib"]
name = "vault_prog"
"#,
        );
        write_file(
            tmp.path(),
            "programs/vault-prog/src/lib.rs",
            r#"
use anchor_lang::prelude::*;

#[program]
pub mod vault_prog {
    use super::*;
    pub fn deposit(ctx: Context<Deposit>, amount: u64) -> Result<()> {
        let v = &mut ctx.accounts.vault;
        v.assets += amount;
        Ok(())
    }
}

#[account]
pub struct Vault {
    pub assets: u64,
}

#[derive(Accounts)]
pub struct Deposit<'info> {
    #[account(mut)]
    pub vault: Account<'info, Vault>,
}
"#,
        );
        write_file(
            tmp.path(),
            "libs/shared-math/Cargo.toml",
            r#"
[package]
name = "shared-math"
version = "0.1.0"
edition = "2021"
"#,
        );
        write_file(
            tmp.path(),
            "libs/shared-math/src/lib.rs",
            "pub fn scale(x: u64) -> u64 { x }",
        );
        write_file(
            tmp.path(),
            "tests/integration.rs",
            "#[test] fn ignored_test() {}",
        );

        let project = SolanaProject::build(tmp.path().to_path_buf(), SolanaOptions::default())
            .await
            .expect("build SolanaProject");

        assert!(project.workspace().has_anchor_toml);
        assert_eq!(project.workspace().crates.len(), 2);
        let programs = project.programs();
        assert_eq!(programs.len(), 1);
        assert_eq!(programs[0].lib_name, "vault_prog");
        assert_eq!(programs[0].sbf_so_name, "vault_prog.so");
        assert_eq!(
            programs[0].declared_program_id.as_deref(),
            Some("VaULt11111111111111111111111111111111111111")
        );
        assert_eq!(project.source_files().len(), 2);
        assert!(project.call_graph_result().contract_count >= 2);
        assert_eq!(project.storage_result().state_variable_count, 1);
    }
}
