//! Non-LLM Rust / Solana static extraction & graph CLI commands
//! (`knowdit static rust ...`), mirroring [`crate::cmd::solidity`].

use std::path::{Path, PathBuf};

use clap::Args;
use color_eyre::eyre::{Result, WrapErr};
use knowdit_project::{
    ExtractedRustContainer, RustCallGraphResult, RustProgramManifest, RustStorageResult,
    SolanaOptions, SolanaProject,
};
use knowdit_repo_model::{cg::CallGraphDotOptions, storage::StorageDotOptions};
use serde::Serialize;

/// `--rust-include-glob` / `--rust-exclude-glob` / `--rust-ignore-scope-file`
/// shared by `static rust` subcommands and downstream orchestrators.
#[derive(Args, Debug, Clone, Default)]
pub struct RustScopeCliArgs {
    /// Glob patterns included in Rust/Solana analysis scope (overrides
    /// `scope.txt` and the default `**/*.rs` when non-empty).
    #[arg(long = "rust-include-glob", value_delimiter = ',')]
    pub include_globs: Vec<String>,

    /// Glob patterns excluded from Rust/Solana analysis scope (overrides
    /// `RUST_VENDORED_EXCLUDES` when non-empty).
    #[arg(long = "rust-exclude-glob", value_delimiter = ',')]
    pub exclude_globs: Vec<String>,

    /// Ignore a root `scope.txt` file when resolving the Rust/Solana scope.
    #[arg(long = "rust-ignore-scope-file")]
    pub ignore_scope_file: bool,
}

impl RustScopeCliArgs {
    pub fn to_solana_options(&self) -> SolanaOptions {
        SolanaOptions {
            include_globs: self.include_globs.clone(),
            exclude_globs: self.exclude_globs.clone(),
            ignore_scope_file: self.ignore_scope_file,
        }
    }
}

#[derive(Args, Debug, Clone)]
pub struct ExtractRustArgs {
    /// Rust / Solana repository root.
    #[arg(long, default_value = ".")]
    pub repo_root: PathBuf,

    #[command(flatten)]
    pub scope: RustScopeCliArgs,

    /// Write extracted Rust workspace & containers JSON to this file. Prints
    /// to stdout when omitted.
    #[arg(long, short = 'o')]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct RustExtractionDump {
    repo_root: PathBuf,
    has_anchor_toml: bool,
    programs: Vec<RustProgramManifest>,
    source_files: Vec<PathBuf>,
    containers: Vec<ExtractedRustContainer>,
}

#[derive(Args, Debug, Clone)]
pub struct RustCallGraphStaticArgs {
    #[command(flatten)]
    pub project: crate::cli::ProjectArgs,

    #[command(flatten)]
    pub db: crate::cli::DatabaseArgs,

    #[command(flatten)]
    pub scope: RustScopeCliArgs,
}

#[derive(Args, Debug, Clone)]
pub struct RustExportCallGraphDotArgs {
    #[command(flatten)]
    pub project: crate::cli::ProjectArgs,

    #[command(flatten)]
    pub db: crate::cli::DatabaseArgs,

    /// Output `.dot` file path.
    #[arg(long, short = 'o', default_value = "knowdit-rust-callgraph.dot")]
    pub output: PathBuf,

    /// Include callgraph functions that have no incoming or outgoing edges.
    #[arg(long)]
    pub include_isolated_nodes: bool,
}

#[derive(Args, Debug, Clone)]
pub struct RustExportStateVariableDotArgs {
    #[command(flatten)]
    pub project: crate::cli::ProjectArgs,

    #[command(flatten)]
    pub db: crate::cli::DatabaseArgs,

    /// Output `.dot` file path.
    #[arg(long, short = 'o', default_value = "knowdit-rust-storage.dot")]
    pub output: PathBuf,

    /// Include state variables / account fields that no function reads or writes.
    #[arg(long)]
    pub include_isolated_state_variables: bool,
}

impl ExtractRustArgs {
    pub async fn run(self) -> Result<()> {
        let project =
            SolanaProject::build(self.repo_root.clone(), self.scope.to_solana_options()).await?;
        let payload = RustExtractionDump {
            repo_root: project.repo_root().to_path_buf(),
            has_anchor_toml: project.workspace().has_anchor_toml,
            programs: project.programs().into_iter().cloned().collect(),
            source_files: project.source_files().to_vec(),
            containers: project.containers().to_vec(),
        };
        let json = serde_json::to_string_pretty(&payload)
            .wrap_err("failed to serialize Rust extraction result")?;

        if let Some(output) = self.output {
            if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
                tokio::fs::create_dir_all(parent)
                    .await
                    .wrap_err_with(|| format!("failed to create {}", parent.display()))?;
            }
            tokio::fs::write(&output, json)
                .await
                .wrap_err_with(|| format!("failed to write {}", output.display()))?;
            tracing::info!("Wrote Rust extraction JSON to {}", output.display());
        } else {
            println!("{json}");
        }
        Ok(())
    }
}

impl RustCallGraphStaticArgs {
    pub async fn run(self) -> Result<()> {
        let crate::cli::LoadedRepoDatabase {
            spec,
            database_path,
            repo,
        } = self
            .project
            .to_repo_database(self.db.database_path.clone(), self.db.variant_render_cap)
            .await?;
        let (cg, sg) =
            Self::update_call_graph(&repo, &spec.root, &self.scope.to_solana_options()).await?;
        let url = database_path.display();
        println!(
            "Wrote Rust/Solana callgraph to {url} (contracts={}, interfaces={}, functions={}, calls={})",
            cg.contract_count, cg.interface_count, cg.function_count, cg.call_count,
        );
        println!(
            "Wrote Rust/Solana storage analysis to {url} (state_vars={}, inherits={}, contract_vars={}, fn_state_vars={})",
            sg.state_variable_count,
            sg.contract_inherit_count,
            sg.contract_variable_count,
            sg.function_state_variable_count,
        );
        Ok(())
    }

    /// In-process entry used by orchestrators (`workflow streamloop`) to build
    /// and persist the Rust / Solana call graph, account/state-variable graph,
    /// and inheritance graph into `repo`.
    pub async fn update_call_graph(
        repo: &knowdit_repo_model::RepoDatabase,
        repo_root: &Path,
        options: &SolanaOptions,
    ) -> Result<(RustCallGraphResult, RustStorageResult)> {
        tracing::info!("Analyzing Rust / Solana project at {}...", repo_root.display());
        let project = SolanaProject::build(repo_root.to_path_buf(), options.clone()).await?;
        repo.write_call_graph(project.call_graph()).await?;
        repo.write_storage_graph(project.storage_graph()).await?;
        repo.write_inheritance_graph(project.inheritance_graph())
            .await?;

        let cg = project.call_graph_result().clone();
        let sg = project.storage_result().clone();
        tracing::info!(
            "Wrote static Rust/Solana callgraph: {} modules/structs, {} traits/CPI-interfaces, {} functions, {} calls",
            cg.contract_count,
            cg.interface_count,
            cg.function_count,
            cg.call_count,
        );
        tracing::info!(
            "Wrote Rust/Solana storage analysis: {} state vars, {} module→account edges, {} contract-variable rows, {} function read/write rows",
            sg.state_variable_count,
            sg.contract_inherit_count,
            sg.contract_variable_count,
            sg.function_state_variable_count,
        );
        Ok((cg, sg))
    }
}

impl RustExportCallGraphDotArgs {
    pub async fn run(self) -> Result<()> {
        let crate::cli::LoadedRepoDatabase {
            database_path,
            repo,
            ..
        } = self
            .project
            .to_repo_database(self.db.database_path.clone(), self.db.variant_render_cap)
            .await?;
        let call_graph = repo.load_call_graph().await?;
        let dot = call_graph.export_dot_with_options(CallGraphDotOptions {
            include_isolated_nodes: self.include_isolated_nodes,
        });
        crate::cmd::learn::export_dot::write_dot_and_maybe_pdf(
            "Rust/Solana call graph",
            &self.output,
            &dot,
        )?;

        let counts = call_graph.counts();
        tracing::info!(
            "Exported Rust/Solana callgraph from {}: {} containers, {} functions, {} calls",
            database_path.display(),
            counts.container_count,
            counts.function_count,
            counts.call_count,
        );
        Ok(())
    }
}

impl RustExportStateVariableDotArgs {
    pub async fn run(self) -> Result<()> {
        let crate::cli::LoadedRepoDatabase {
            database_path,
            repo,
            ..
        } = self
            .project
            .to_repo_database(self.db.database_path.clone(), self.db.variant_render_cap)
            .await?;
        let storage = repo.load_storage_graph().await?;
        let dot = repo
            .export_storage_dot(
                &storage,
                StorageDotOptions {
                    include_isolated_state_variables: self.include_isolated_state_variables,
                },
            )
            .await?;
        crate::cmd::learn::export_dot::write_dot_and_maybe_pdf(
            "Rust/Solana account & state-variable read/write graph",
            &self.output,
            &dot,
        )?;

        tracing::info!(
            "Exported Rust/Solana storage R/W graph from {}: {} state vars, {} fn read/write rows",
            database_path.display(),
            storage.state_variables.len(),
            storage.function_state_variables.len(),
        );
        Ok(())
    }
}
