//! Static Rust / Solana / Anchor AST analyzer.
//!
//! Maps Rust crates (Anchor `#[program]` / `#[account]` / `#[derive(Accounts)]`,
//! Pinocchio, and native `solana-program` entrypoints) onto `knowdit-repo-model`'s
//! `CallGraph`, `StorageGraph`, and `InheritanceGraph` so downstream phases
//! (`ProjectIndex`, `SpecificationGenerator`, `coverage_audit`, `VerdictGrader`,
//! `SeverityGrader`, and `ReviewGrader`) work uniformly across Solidity and Rust.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, ensure};
use knowdit_repo_model::{
    cg::{
        CallGraph, Contract, FileChunk, FileLocation, Function, FunctionCall as RepoCall, Interface,
    },
    inheritance::{ContractInherit, InheritanceGraph},
    storage::{ContractVariable, FunctionStateVariable, StateVariable, StorageGraph},
};
use serde::{Deserialize, Serialize};
use syn::spanned::Spanned;

use crate::signature::SourceLineIndex;

/// One pre-loaded `.rs` source file passed into [`RustStaticAnalyzer::analyze`].
#[derive(Debug, Clone)]
pub struct RustSourceInput {
    pub relative_path: PathBuf,
    pub content: String,
}

/// Summary of the extracted Rust static call graph (mirrors
/// `knowdit_cg_static::StaticCallGraphResult`).
#[derive(Debug, Clone)]
pub struct RustCallGraphResult {
    pub call_graph: CallGraph,
    pub contract_count: usize,
    pub interface_count: usize,
    pub function_count: usize,
    pub call_count: usize,
}

/// Summary of the extracted Rust account / state-variable graph (mirrors
/// `knowdit_cg_static::StorageAnalysisResult`).
#[derive(Debug, Clone)]
pub struct RustStorageResult {
    pub storage: StorageGraph,
    pub inheritance: InheritanceGraph,
    pub state_variable_count: usize,
    pub contract_inherit_count: usize,
    pub contract_variable_count: usize,
    pub function_state_variable_count: usize,
}

/// Kind of Rust container surfaced by `knowdit static rust extract`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RustContainerKind {
    ProgramModule,
    Module,
    StateStruct,
    StructImpl,
    TraitInterface,
}

/// Lightweight JSON-serializable callable summary for `knowdit static rust extract`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedRustCallable {
    pub id: i32,
    pub container_id: i32,
    pub container_name: String,
    pub name: String,
    pub args: String,
    pub relative_file_path: PathBuf,
    pub chunk: FileChunk,
}

/// Lightweight JSON-serializable container summary for `knowdit static rust extract`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedRustContainer {
    pub id: i32,
    pub name: String,
    pub kind: RustContainerKind,
    pub relative_file_path: PathBuf,
    pub chunk: FileChunk,
    pub functions: Vec<ExtractedRustCallable>,
}

/// Combined output of one [`RustStaticAnalyzer::analyze`] pass.
#[derive(Debug, Clone)]
pub struct RustStaticAnalysis {
    pub call_graph: RustCallGraphResult,
    pub storage: RustStorageResult,
    pub containers: Vec<ExtractedRustContainer>,
}

#[derive(Debug, Clone)]
enum ContextFieldBinding {
    /// `Account<'info, Vault>` or `AccountLoader<'info, Vault>`.
    StateStruct(String),
    /// Synthetic `StateVariable` id (e.g. `vault_token.amount`, `user.lamports`).
    SyntheticVar {
        sv_id: i32,
        field_suffix: &'static str,
    },
}

#[derive(Debug, Clone)]
struct AccountsContextDef {
    fields: BTreeMap<String, Vec<ContextFieldBinding>>,
    referenced_state_structs: BTreeSet<String>,
    synthetic_vars: Vec<i32>,
}

#[derive(Debug, Clone)]
struct RecordedCallable {
    function_id: i32,
    container_id: i32,
    container_is_interface: bool,
    container_name: String,
    self_struct_name: Option<String>,
    context_struct_names: Vec<String>,
    param_struct_bindings: BTreeMap<String, String>,
    block: Option<syn::Block>,
}

#[derive(Debug, Clone)]
enum LocalAlias {
    StateStruct(String),
    ContextAccount {
        context_name: String,
        account_field: String,
    },
}

/// Pure-Rust AST analyzer over in-scope `.rs` files.
pub struct RustStaticAnalyzer {
    contracts: BTreeMap<i32, Contract>,
    interfaces: BTreeMap<i32, Interface>,
    container_kinds: BTreeMap<i32, RustContainerKind>,
    contract_id_by_name: BTreeMap<String, i32>,
    interface_id_by_name: BTreeMap<String, i32>,
    cpi_fn_ids: BTreeMap<(String, String), i32>,
    state_variables: BTreeMap<i32, StateVariable>,
    contract_variables: Vec<ContractVariable>,
    contract_var_dedup: BTreeSet<(i32, i32)>,
    state_var_by_owner_field: BTreeMap<(i32, String), i32>,
    contract_inherits: BTreeSet<(i32, i32)>,
    function_state_variables: Vec<FunctionStateVariable>,
    accounts_contexts: BTreeMap<String, AccountsContextDef>,
    referenced_account_types: BTreeSet<String>,
    callables: BTreeMap<i32, RecordedCallable>,
    functions_by_container_and_name: BTreeMap<(String, String), i32>,
    functions_by_name: BTreeMap<String, Vec<i32>>,
    next_container_id: i32,
    next_function_id: i32,
    next_call_id: i32,
    next_state_var_id: i32,
}

impl RustStaticAnalyzer {
    /// Run the complete static call-graph + state/account analysis over `inputs`.
    pub fn analyze(inputs: &[RustSourceInput]) -> Result<RustStaticAnalysis> {
        ensure!(
            !inputs.is_empty(),
            "no Rust source files provided to RustStaticAnalyzer::analyze"
        );

        let mut sorted_inputs: Vec<&RustSourceInput> = inputs.iter().collect();
        sorted_inputs.sort_by(|lhs, rhs| lhs.relative_path.cmp(&rhs.relative_path));

        let mut parsed_files: Vec<(&RustSourceInput, syn::File)> = Vec::new();
        for input in sorted_inputs {
            match syn::parse_file(&input.content) {
                Ok(ast) => parsed_files.push((input, ast)),
                Err(err) => {
                    tracing::warn!(
                        path = %input.relative_path.display(),
                        "syn failed to parse Rust file (skipping AST extraction): {err}"
                    );
                }
            }
        }
        ensure!(
            !parsed_files.is_empty(),
            "syn could not parse any Rust source files in scope"
        );

        let mut analyzer = Self::new();

        // Pre-scan: collect every inner type `T` referenced inside `Account<..., T>`
        // or `AccountLoader<..., T>` so plain structs without `#[account]` (e.g.
        // zero-copy or custom-serialized structs) are still recognized as state structs.
        for (_, ast) in &parsed_files {
            analyzer.prescan_items(&ast.items);
        }

        // Pass 1a: ingest state structs, traits, and `#[derive(Accounts)]` contexts
        // across all files first so cross-file references resolve regardless of order.
        for (input, ast) in &parsed_files {
            let line_idx = SourceLineIndex::new(&input.content);
            analyzer.ingest_types_in_items(
                &input.relative_path,
                &input.content,
                &line_idx,
                &ast.items,
            );
        }

        // Pass 1b: ingest modules (`#[program]` + regular modules) and `impl` blocks.
        for (input, ast) in &parsed_files {
            let line_idx = SourceLineIndex::new(&input.content);
            let default_mod = Self::default_module_name(&input.relative_path);
            analyzer.ingest_callables_in_items(
                &input.relative_path,
                &input.content,
                &line_idx,
                &default_mod,
                false,
                &ast.items,
            );
        }

        // Pass 2: wire `ContractInherit` edges from `#[derive(Accounts)]` usage,
        // resolve outgoing `FunctionCall` edges (including CPIs), and walk bodies
        // for alias-aware `FunctionStateVariable` reads/writes.
        analyzer.resolve_edges_and_storage();

        Ok(analyzer.finish())
    }

    fn new() -> Self {
        Self {
            contracts: BTreeMap::new(),
            interfaces: BTreeMap::new(),
            container_kinds: BTreeMap::new(),
            contract_id_by_name: BTreeMap::new(),
            interface_id_by_name: BTreeMap::new(),
            cpi_fn_ids: BTreeMap::new(),
            state_variables: BTreeMap::new(),
            contract_variables: Vec::new(),
            contract_var_dedup: BTreeSet::new(),
            state_var_by_owner_field: BTreeMap::new(),
            contract_inherits: BTreeSet::new(),
            function_state_variables: Vec::new(),
            accounts_contexts: BTreeMap::new(),
            referenced_account_types: BTreeSet::new(),
            callables: BTreeMap::new(),
            functions_by_container_and_name: BTreeMap::new(),
            functions_by_name: BTreeMap::new(),
            next_container_id: 1,
            next_function_id: 1,
            next_call_id: 1,
            next_state_var_id: 1,
        }
    }

    fn default_module_name(relative_path: &Path) -> String {
        let stem = relative_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("module");
        if matches!(stem, "lib" | "main" | "mod") {
            relative_path
                .parent()
                .and_then(|p| {
                    let dir_name = p.file_name().and_then(|n| n.to_str()).unwrap_or("program");
                    if dir_name == "src" {
                        p.parent()
                            .and_then(|gp| gp.file_name())
                            .and_then(|n| n.to_str())
                    } else {
                        Some(dir_name)
                    }
                })
                .unwrap_or("program")
                .replace('-', "_")
        } else {
            stem.replace('-', "_")
        }
    }

    fn prescan_items(&mut self, items: &[syn::Item]) {
        for item in items {
            match item {
                syn::Item::Struct(s) => {
                    if Self::has_accounts_derive(&s.attrs) {
                        for field in &s.fields {
                            if let Some((wrapper, Some(inner))) =
                                Self::classify_account_wrapper_type(&field.ty)
                                && matches!(
                                    wrapper.as_str(),
                                    "Account"
                                        | "AccountLoader"
                                        | "ProgramAccount"
                                        | "AccountZeroCopy"
                                )
                                && !Self::is_spl_builtin_account(&inner)
                            {
                                self.referenced_account_types.insert(inner);
                            }
                        }
                    }
                }
                syn::Item::Mod(m) => {
                    if let Some((_, sub_items)) = &m.content {
                        self.prescan_items(sub_items);
                    }
                }
                _ => {}
            }
        }
    }

    fn ingest_types_in_items(
        &mut self,
        relative_path: &Path,
        source: &str,
        line_idx: &SourceLineIndex<'_>,
        items: &[syn::Item],
    ) {
        for item in items {
            match item {
                syn::Item::Struct(item_struct) => {
                    let struct_name = item_struct.ident.to_string();
                    if Self::has_accounts_derive(&item_struct.attrs) {
                        self.ingest_accounts_context(relative_path, line_idx, item_struct);
                    } else if Self::is_state_struct(
                        item_struct,
                        self.referenced_account_types.contains(&struct_name),
                    ) {
                        self.ingest_state_struct(relative_path, source, line_idx, item_struct);
                    }
                }
                syn::Item::Trait(item_trait) => {
                    self.ingest_trait(relative_path, source, line_idx, item_trait);
                }
                syn::Item::Mod(item_mod) => {
                    if let Some((_, sub_items)) = &item_mod.content {
                        self.ingest_types_in_items(relative_path, source, line_idx, sub_items);
                    }
                }
                _ => {}
            }
        }
    }

    fn ingest_callables_in_items(
        &mut self,
        relative_path: &Path,
        source: &str,
        line_idx: &SourceLineIndex<'_>,
        module_name: &str,
        is_program_mod: bool,
        items: &[syn::Item],
    ) {
        let has_direct_fns = items.iter().any(|it| matches!(it, syn::Item::Fn(_)));
        let module_contract_id = if has_direct_fns || is_program_mod {
            let chunk = if is_program_mod {
                Self::full_file_chunk(source)
            } else {
                Self::full_file_chunk(source)
            };
            let kind = if is_program_mod {
                RustContainerKind::ProgramModule
            } else {
                RustContainerKind::Module
            };
            Some(self.ensure_contract(
                module_name,
                relative_path,
                chunk,
                kind,
                Some(if is_program_mod {
                    "Solana #[program] entrypoint module".to_string()
                } else {
                    "Rust module".to_string()
                }),
            ))
        } else {
            None
        };

        for item in items {
            match item {
                syn::Item::Fn(item_fn) => {
                    if let Some(cid) = module_contract_id {
                        self.ingest_fn_item(
                            relative_path,
                            line_idx,
                            cid,
                            module_name,
                            None,
                            &item_fn.sig,
                            Some((*item_fn.block).clone()),
                            item_fn.span(),
                        );
                    }
                }
                syn::Item::Impl(item_impl) => {
                    let Some(target_name) = Self::type_last_ident(&item_impl.self_ty) else {
                        continue;
                    };
                    // Skip generated-looking Anchor internal impls when target has no methods.
                    let has_fns = item_impl
                        .items
                        .iter()
                        .any(|it| matches!(it, syn::ImplItem::Fn(_)));
                    if !has_fns && item_impl.trait_.is_none() {
                        continue;
                    }
                    let chunk = Self::chunk_from_span(line_idx, item_impl.span());
                    let cid = self.ensure_contract(
                        &target_name,
                        relative_path,
                        chunk,
                        RustContainerKind::StructImpl,
                        None,
                    );
                    if let Some((_, trait_path, _)) = &item_impl.trait_
                        && let Some(trait_ident) = trait_path.segments.last()
                        && let Some(&iid) = self
                            .interface_id_by_name
                            .get(&trait_ident.ident.to_string())
                    {
                        self.contract_inherits.insert((cid, iid));
                    }
                    let self_struct = Some(target_name.clone());
                    for impl_item in &item_impl.items {
                        if let syn::ImplItem::Fn(method) = impl_item {
                            self.ingest_fn_item(
                                relative_path,
                                line_idx,
                                cid,
                                &target_name,
                                self_struct.clone(),
                                &method.sig,
                                Some(method.block.clone()),
                                method.span(),
                            );
                        }
                    }
                }
                syn::Item::Mod(sub_mod) => {
                    if let Some((_, sub_items)) = &sub_mod.content {
                        let sub_is_program = Self::has_program_attr(&sub_mod.attrs);
                        let sub_name = sub_mod.ident.to_string();
                        self.ingest_callables_in_items(
                            relative_path,
                            source,
                            line_idx,
                            &sub_name,
                            sub_is_program,
                            sub_items,
                        );
                    }
                }
                _ => {}
            }
        }
    }

    fn ingest_state_struct(
        &mut self,
        relative_path: &Path,
        _source: &str,
        line_idx: &SourceLineIndex<'_>,
        item_struct: &syn::ItemStruct,
    ) {
        let name = item_struct.ident.to_string();
        let chunk = Self::chunk_from_span(line_idx, item_struct.span());
        let cid = self.ensure_contract(
            &name,
            relative_path,
            chunk,
            RustContainerKind::StateStruct,
            Some("On-chain account / state struct".to_string()),
        );
        for field in &item_struct.fields {
            let Some(ident) = &field.ident else {
                continue;
            };
            let field_name = ident.to_string();
            let type_name = Self::compact_whitespace(line_idx.slice_span(field.ty.span()));
            let field_chunk = Self::chunk_from_span(line_idx, field.span());
            let sv_id = self.add_state_variable(
                cid,
                field_name,
                type_name,
                relative_path,
                field_chunk,
                None,
            );
            let _ = sv_id;
        }
    }

    fn ingest_accounts_context(
        &mut self,
        relative_path: &Path,
        line_idx: &SourceLineIndex<'_>,
        item_struct: &syn::ItemStruct,
    ) {
        let ctx_name = item_struct.ident.to_string();
        let mut fields: BTreeMap<String, Vec<ContextFieldBinding>> = BTreeMap::new();
        let mut referenced_state_structs = BTreeSet::new();
        let mut synthetic_vars = Vec::new();

        for field in &item_struct.fields {
            let Some(ident) = &field.ident else {
                continue;
            };
            let acct_name = ident.to_string();
            let field_chunk = Self::chunk_from_span(line_idx, field.span());
            let is_mut = Self::has_mut_account_attr(&field.attrs);
            let mut bindings = Vec::new();

            if let Some((wrapper, inner)) = Self::classify_account_wrapper_type(&field.ty) {
                match (wrapper.as_str(), inner.as_deref()) {
                    (
                        "Account" | "AccountLoader" | "ProgramAccount" | "InterfaceAccount",
                        Some(t),
                    ) if Self::is_token_account_type(t) => {
                        let sv_id = self.alloc_synthetic_state_var(
                            format!("{acct_name}.amount"),
                            "TokenAccount::amount (u64)".to_string(),
                            relative_path,
                            field_chunk.clone(),
                        );
                        synthetic_vars.push(sv_id);
                        bindings.push(ContextFieldBinding::SyntheticVar {
                            sv_id,
                            field_suffix: "amount",
                        });
                    }
                    ("Account" | "InterfaceAccount", Some(t)) if t == "Mint" => {
                        let sv_id = self.alloc_synthetic_state_var(
                            format!("{acct_name}.supply"),
                            "Mint::supply (u64)".to_string(),
                            relative_path,
                            field_chunk.clone(),
                        );
                        synthetic_vars.push(sv_id);
                        bindings.push(ContextFieldBinding::SyntheticVar {
                            sv_id,
                            field_suffix: "supply",
                        });
                    }
                    (
                        "Account" | "AccountLoader" | "ProgramAccount" | "AccountZeroCopy",
                        Some(t),
                    ) if !Self::is_spl_builtin_account(t) => {
                        referenced_state_structs.insert(t.to_string());
                        bindings.push(ContextFieldBinding::StateStruct(t.to_string()));
                    }
                    _ => {}
                }
            }

            if is_mut && bindings.is_empty() {
                let sv_id = self.alloc_synthetic_state_var(
                    format!("{acct_name}.lamports"),
                    "AccountInfo::lamports (u64)".to_string(),
                    relative_path,
                    field_chunk,
                );
                synthetic_vars.push(sv_id);
                bindings.push(ContextFieldBinding::SyntheticVar {
                    sv_id,
                    field_suffix: "lamports",
                });
            }

            if !bindings.is_empty() {
                fields.insert(acct_name, bindings);
            }
        }

        self.accounts_contexts.insert(
            ctx_name,
            AccountsContextDef {
                fields,
                referenced_state_structs,
                synthetic_vars,
            },
        );
    }

    fn ingest_trait(
        &mut self,
        relative_path: &Path,
        _source: &str,
        line_idx: &SourceLineIndex<'_>,
        item_trait: &syn::ItemTrait,
    ) {
        let name = item_trait.ident.to_string();
        if self.interface_id_by_name.contains_key(&name) {
            return;
        }
        let id = self.next_container_id;
        self.next_container_id += 1;
        let chunk = Self::chunk_from_span(line_idx, item_trait.span());
        self.interfaces.insert(
            id,
            Interface {
                id,
                name: name.clone(),
                relative_file_path: relative_path.to_path_buf(),
                chunk,
                functions: Vec::new(),
                description: Some("Rust trait".to_string()),
            },
        );
        self.container_kinds
            .insert(id, RustContainerKind::TraitInterface);
        self.interface_id_by_name.insert(name.clone(), id);

        for item in &item_trait.items {
            if let syn::TraitItem::Fn(trait_fn) = item {
                let fn_id = self.next_function_id;
                self.next_function_id += 1;
                let fn_name = trait_fn.sig.ident.to_string();
                let args = Self::format_fn_inputs(line_idx, &trait_fn.sig);
                let fn_chunk = Self::chunk_from_span(line_idx, trait_fn.span());
                let content = trait_fn.default.as_ref().map(|_| fn_chunk.content.clone());
                self.callables.insert(
                    fn_id,
                    RecordedCallable {
                        function_id: fn_id,
                        container_id: id,
                        container_is_interface: true,
                        container_name: name.clone(),
                        self_struct_name: None,
                        context_struct_names: Vec::new(),
                        param_struct_bindings: BTreeMap::new(),
                        block: trait_fn.default.clone(),
                    },
                );
                if let Some(iface) = self.interfaces.get_mut(&id) {
                    iface.functions.push(Function {
                        id: fn_id,
                        name: fn_name.clone(),
                        args,
                        relative_file_path: relative_path.to_path_buf(),
                        loc: fn_chunk.loc,
                        content,
                        calls: Vec::new(),
                        description: None,
                    });
                }
                self.functions_by_container_and_name
                    .entry((name.clone(), fn_name.clone()))
                    .or_insert(fn_id);
                self.functions_by_name
                    .entry(fn_name)
                    .or_default()
                    .push(fn_id);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ingest_fn_item(
        &mut self,
        relative_path: &Path,
        line_idx: &SourceLineIndex<'_>,
        container_id: i32,
        container_name: &str,
        self_struct_name: Option<String>,
        sig: &syn::Signature,
        block: Option<syn::Block>,
        full_span: proc_macro2::Span,
    ) {
        let fn_id = self.next_function_id;
        self.next_function_id += 1;
        let fn_name = sig.ident.to_string();
        let args = Self::format_fn_inputs(line_idx, sig);
        let chunk = Self::chunk_from_span(line_idx, full_span);

        let mut context_struct_names = Vec::new();
        let mut param_struct_bindings = BTreeMap::new();
        for input in &sig.inputs {
            if let syn::FnArg::Typed(pat_type) = input {
                if let Some(ctx_ty) = Self::extract_anchor_context_type(&pat_type.ty) {
                    context_struct_names.push(ctx_ty);
                } else if let syn::Pat::Ident(pat_ident) = pat_type.pat.as_ref()
                    && let Some(ty_name) = Self::unwrap_ref_or_box_ident(&pat_type.ty)
                {
                    if self.accounts_contexts.contains_key(&ty_name) {
                        context_struct_names.push(ty_name);
                    } else if self.contract_id_by_name.contains_key(&ty_name) {
                        param_struct_bindings.insert(pat_ident.ident.to_string(), ty_name);
                    }
                }
            }
        }

        if let Some(contract) = self.contracts.get_mut(&container_id) {
            contract.functions.push(Function {
                id: fn_id,
                name: fn_name.clone(),
                args,
                relative_file_path: relative_path.to_path_buf(),
                loc: chunk.loc,
                content: Some(chunk.content),
                calls: Vec::new(),
                description: None,
            });
        }

        self.callables.insert(
            fn_id,
            RecordedCallable {
                function_id: fn_id,
                container_id,
                container_is_interface: false,
                container_name: container_name.to_string(),
                self_struct_name,
                context_struct_names,
                param_struct_bindings,
                block,
            },
        );
        self.functions_by_container_and_name
            .entry((container_name.to_string(), fn_name.clone()))
            .or_insert(fn_id);
        self.functions_by_name
            .entry(fn_name)
            .or_default()
            .push(fn_id);
    }

    fn ensure_contract(
        &mut self,
        name: &str,
        relative_path: &Path,
        chunk: FileChunk,
        kind: RustContainerKind,
        description: Option<String>,
    ) -> i32 {
        if let Some(&existing_id) = self.contract_id_by_name.get(name) {
            // Upgrade kind/description if a ProgramModule or StateStruct arrives after an impl.
            if matches!(
                kind,
                RustContainerKind::ProgramModule | RustContainerKind::StateStruct
            ) {
                self.container_kinds.insert(existing_id, kind);
                if let Some(desc) = description
                    && let Some(c) = self.contracts.get_mut(&existing_id)
                {
                    c.description = Some(desc);
                }
            }
            return existing_id;
        }
        let id = self.next_container_id;
        self.next_container_id += 1;
        self.contracts.insert(
            id,
            Contract {
                id,
                name: name.to_string(),
                relative_file_path: relative_path.to_path_buf(),
                chunk,
                functions: Vec::new(),
                description,
            },
        );
        self.container_kinds.insert(id, kind);
        self.contract_id_by_name.insert(name.to_string(), id);
        id
    }

    fn add_state_variable(
        &mut self,
        owner_contract_id: i32,
        name: String,
        type_name: String,
        relative_path: &Path,
        chunk: FileChunk,
        description: Option<String>,
    ) -> i32 {
        if let Some(&existing) = self
            .state_var_by_owner_field
            .get(&(owner_contract_id, name.clone()))
        {
            return existing;
        }
        let sv_id = self.next_state_var_id;
        self.next_state_var_id += 1;
        self.state_variables.insert(
            sv_id,
            StateVariable {
                id: sv_id,
                name: name.clone(),
                type_name,
                relative_file_path: relative_path.to_path_buf(),
                loc: chunk.loc,
                content: chunk.content,
            },
        );
        self.attach_contract_variable(owner_contract_id, sv_id, description);
        self.state_var_by_owner_field
            .insert((owner_contract_id, name), sv_id);
        sv_id
    }

    fn alloc_synthetic_state_var(
        &mut self,
        name: String,
        type_name: String,
        relative_path: &Path,
        chunk: FileChunk,
    ) -> i32 {
        let sv_id = self.next_state_var_id;
        self.next_state_var_id += 1;
        self.state_variables.insert(
            sv_id,
            StateVariable {
                id: sv_id,
                name,
                type_name,
                relative_file_path: relative_path.to_path_buf(),
                loc: chunk.loc,
                content: chunk.content,
            },
        );
        sv_id
    }

    fn attach_contract_variable(
        &mut self,
        contract_id: i32,
        state_variable_id: i32,
        description: Option<String>,
    ) {
        if self
            .contract_var_dedup
            .insert((contract_id, state_variable_id))
        {
            self.contract_variables.push(ContractVariable {
                contract_id,
                state_variable_id,
                description,
            });
        }
    }

    fn ensure_cpi_target(&mut self, interface_name: &str, fn_name: &str) -> i32 {
        let key = (interface_name.to_string(), fn_name.to_string());
        if let Some(&existing) = self.cpi_fn_ids.get(&key) {
            return existing;
        }
        let iid = match self.interface_id_by_name.get(interface_name).copied() {
            Some(id) => id,
            None => {
                let id = self.next_container_id;
                self.next_container_id += 1;
                self.interfaces.insert(
                    id,
                    Interface {
                        id,
                        name: interface_name.to_string(),
                        relative_file_path: PathBuf::from("<solana-cpi>"),
                        chunk: FileChunk {
                            loc: FileLocation {
                                start_line: 1,
                                start_column: 0,
                                end_line: 1,
                                end_column: 0,
                            },
                            content: String::new(),
                        },
                        functions: Vec::new(),
                        description: Some("External Solana CPI target interface".to_string()),
                    },
                );
                self.container_kinds
                    .insert(id, RustContainerKind::TraitInterface);
                self.interface_id_by_name
                    .insert(interface_name.to_string(), id);
                id
            }
        };
        let fn_id = self.next_function_id;
        self.next_function_id += 1;
        if let Some(iface) = self.interfaces.get_mut(&iid) {
            iface.functions.push(Function {
                id: fn_id,
                name: fn_name.to_string(),
                args: "CpiContext / AccountInfo".to_string(),
                relative_file_path: PathBuf::from("<solana-cpi>"),
                loc: FileLocation {
                    start_line: 1,
                    start_column: 0,
                    end_line: 1,
                    end_column: 0,
                },
                content: None,
                calls: Vec::new(),
                description: Some("Cross-Program Invocation".to_string()),
            });
        }
        self.cpi_fn_ids.insert(key, fn_id);
        fn_id
    }

    fn resolve_edges_and_storage(&mut self) {
        let callable_list: Vec<RecordedCallable> = self.callables.values().cloned().collect();

        // 1. Wire `ContractInherit` edges and synthetic account variables onto
        //    modules whose functions accept `Context<AccountsStruct>`.
        for callable in &callable_list {
            if callable.container_is_interface {
                continue;
            }
            for ctx_name in &callable.context_struct_names {
                let Some(ctx_def) = self.accounts_contexts.get(ctx_name).cloned() else {
                    continue;
                };
                for state_struct in &ctx_def.referenced_state_structs {
                    if let Some(&struct_cid) = self.contract_id_by_name.get(state_struct)
                        && struct_cid != callable.container_id
                    {
                        self.contract_inherits
                            .insert((callable.container_id, struct_cid));
                    }
                }
                for sv_id in &ctx_def.synthetic_vars {
                    self.attach_contract_variable(
                        callable.container_id,
                        *sv_id,
                        Some(format!("from Accounts context `{ctx_name}`")),
                    );
                }
            }
            for state_struct in callable.param_struct_bindings.values() {
                if let Some(&struct_cid) = self.contract_id_by_name.get(state_struct)
                    && struct_cid != callable.container_id
                {
                    self.contract_inherits
                        .insert((callable.container_id, struct_cid));
                }
            }
        }

        // 2. Walk every callable body to collect outgoing calls + state reads/writes.
        for callable in callable_list {
            let Some(block) = &callable.block else {
                continue;
            };
            let mut aliases: BTreeMap<String, LocalAlias> = BTreeMap::new();
            if let Some(self_struct) = &callable.self_struct_name {
                aliases.insert(
                    "self".to_string(),
                    LocalAlias::StateStruct(self_struct.clone()),
                );
            }
            for (param, struct_name) in &callable.param_struct_bindings {
                aliases.insert(param.clone(), LocalAlias::StateStruct(struct_name.clone()));
            }

            let mut outgoing_targets: Vec<(i32, Option<String>)> = Vec::new();
            let mut seen_targets: BTreeSet<i32> = BTreeSet::new();
            let mut reads: BTreeSet<i32> = BTreeSet::new();
            let mut writes: BTreeSet<i32> = BTreeSet::new();

            self.walk_block(
                &callable,
                block,
                &mut aliases,
                &mut outgoing_targets,
                &mut seen_targets,
                &mut reads,
                &mut writes,
            );

            // Propagate calls to Anchor delegated handler functions (e.g.
            // `#[program] pub fn withdraw(ctx, amt) { instructions::withdraw::handler(ctx, amt) }`)
            // and persist edges.
            let mut call_edges = Vec::with_capacity(outgoing_targets.len());
            for (to_id, description) in outgoing_targets {
                let call_id = self.next_call_id;
                self.next_call_id += 1;
                call_edges.push(RepoCall {
                    id: call_id,
                    from_id: callable.function_id,
                    to_id,
                    description,
                });
            }
            if !call_edges.is_empty() {
                if callable.container_is_interface {
                    if let Some(iface) = self.interfaces.get_mut(&callable.container_id)
                        && let Some(f) = iface
                            .functions
                            .iter_mut()
                            .find(|f| f.id == callable.function_id)
                    {
                        f.calls = call_edges;
                    }
                } else if let Some(contract) = self.contracts.get_mut(&callable.container_id)
                    && let Some(f) = contract
                        .functions
                        .iter_mut()
                        .find(|f| f.id == callable.function_id)
                {
                    f.calls = call_edges;
                }
            }

            for sv_id in writes {
                self.function_state_variables.push(FunctionStateVariable {
                    function_id: callable.function_id,
                    state_variable_id: sv_id,
                    is_write: true,
                    description: None,
                });
            }
            for sv_id in reads {
                self.function_state_variables.push(FunctionStateVariable {
                    function_id: callable.function_id,
                    state_variable_id: sv_id,
                    is_write: false,
                    description: None,
                });
            }
        }

        // Remove any completely empty synthetic module containers that held neither
        // functions nor state variables.
        let non_empty_var_contracts: BTreeSet<i32> = self
            .contract_variables
            .iter()
            .map(|cv| cv.contract_id)
            .collect();
        self.contracts
            .retain(|id, c| !c.functions.is_empty() || non_empty_var_contracts.contains(id));
        let valid_ids: BTreeSet<i32> = self
            .contracts
            .keys()
            .chain(self.interfaces.keys())
            .copied()
            .collect();
        self.contract_inherits
            .retain(|(from, to)| valid_ids.contains(from) && valid_ids.contains(to));
    }

    #[allow(clippy::too_many_arguments)]
    fn walk_block(
        &mut self,
        callable: &RecordedCallable,
        block: &syn::Block,
        aliases: &mut BTreeMap<String, LocalAlias>,
        outgoing: &mut Vec<(i32, Option<String>)>,
        seen_targets: &mut BTreeSet<i32>,
        reads: &mut BTreeSet<i32>,
        writes: &mut BTreeSet<i32>,
    ) {
        for stmt in &block.stmts {
            match stmt {
                syn::Stmt::Local(local) => {
                    if let Some(init) = &local.init {
                        if let syn::Pat::Ident(pat_ident) = Self::peel_pat(&local.pat)
                            && let Some(alias) =
                                self.resolve_alias_initializer(callable, &init.expr, aliases)
                        {
                            aliases.insert(pat_ident.ident.to_string(), alias);
                        }
                        self.walk_expr(
                            callable,
                            &init.expr,
                            false,
                            aliases,
                            outgoing,
                            seen_targets,
                            reads,
                            writes,
                        );
                        if let Some((_, diverge)) = &init.diverge {
                            self.walk_expr(
                                callable,
                                diverge,
                                false,
                                aliases,
                                outgoing,
                                seen_targets,
                                reads,
                                writes,
                            );
                        }
                    }
                }
                syn::Stmt::Expr(expr, _) => {
                    self.walk_expr(
                        callable,
                        expr,
                        false,
                        aliases,
                        outgoing,
                        seen_targets,
                        reads,
                        writes,
                    );
                }
                syn::Stmt::Macro(stmt_macro) => {
                    // Inspect macro tokens inside `require!(a > b, ...)` / `assert_eq!(...)`
                    // when parseable as comma-separated expressions.
                    if let Ok(exprs) = stmt_macro.mac.parse_body_with(
                        syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated,
                    ) {
                        for e in exprs {
                            self.walk_expr(
                                callable,
                                &e,
                                false,
                                aliases,
                                outgoing,
                                seen_targets,
                                reads,
                                writes,
                            );
                        }
                    }
                }
                syn::Stmt::Item(_) => {}
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn walk_expr(
        &mut self,
        callable: &RecordedCallable,
        expr: &syn::Expr,
        is_write_ctx: bool,
        aliases: &mut BTreeMap<String, LocalAlias>,
        outgoing: &mut Vec<(i32, Option<String>)>,
        seen_targets: &mut BTreeSet<i32>,
        reads: &mut BTreeSet<i32>,
        writes: &mut BTreeSet<i32>,
    ) {
        match expr {
            syn::Expr::Assign(assign) => {
                self.walk_expr(
                    callable,
                    &assign.left,
                    true,
                    aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
                self.walk_expr(
                    callable,
                    &assign.right,
                    false,
                    aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
            }
            syn::Expr::Binary(bin) => {
                let compound_assign = matches!(
                    bin.op,
                    syn::BinOp::AddAssign(_)
                        | syn::BinOp::SubAssign(_)
                        | syn::BinOp::MulAssign(_)
                        | syn::BinOp::DivAssign(_)
                        | syn::BinOp::RemAssign(_)
                        | syn::BinOp::BitXorAssign(_)
                        | syn::BinOp::BitAndAssign(_)
                        | syn::BinOp::BitOrAssign(_)
                        | syn::BinOp::ShlAssign(_)
                        | syn::BinOp::ShrAssign(_)
                );
                if compound_assign {
                    // Compound assignment both reads and writes the LHS.
                    self.walk_expr(
                        callable,
                        &bin.left,
                        false,
                        aliases,
                        outgoing,
                        seen_targets,
                        reads,
                        writes,
                    );
                    self.walk_expr(
                        callable,
                        &bin.left,
                        true,
                        aliases,
                        outgoing,
                        seen_targets,
                        reads,
                        writes,
                    );
                } else {
                    self.walk_expr(
                        callable,
                        &bin.left,
                        false,
                        aliases,
                        outgoing,
                        seen_targets,
                        reads,
                        writes,
                    );
                }
                self.walk_expr(
                    callable,
                    &bin.right,
                    false,
                    aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
            }
            syn::Expr::Field(field_expr) => {
                if let syn::Member::Named(member_ident) = &field_expr.member
                    && let Some(sv_ids) = self.resolve_field_access(
                        callable,
                        &field_expr.base,
                        &member_ident.to_string(),
                        aliases,
                    )
                {
                    for sv_id in sv_ids {
                        if is_write_ctx {
                            writes.insert(sv_id);
                        } else {
                            reads.insert(sv_id);
                        }
                    }
                    return;
                }
                self.walk_expr(
                    callable,
                    &field_expr.base,
                    is_write_ctx,
                    aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
            }
            syn::Expr::Reference(ref_expr) => {
                let mut_borrow = ref_expr.mutability.is_some();
                self.walk_expr(
                    callable,
                    &ref_expr.expr,
                    is_write_ctx || mut_borrow,
                    aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
            }
            syn::Expr::Call(call_expr) => {
                if let syn::Expr::Path(expr_path) = call_expr.func.as_ref()
                    && let Some((target_id, desc)) =
                        self.resolve_path_call(callable, &expr_path.path)
                    && seen_targets.insert(target_id)
                {
                    outgoing.push((target_id, desc));
                }
                for arg in &call_expr.args {
                    self.walk_expr(
                        callable,
                        arg,
                        false,
                        aliases,
                        outgoing,
                        seen_targets,
                        reads,
                        writes,
                    );
                }
            }
            syn::Expr::MethodCall(mc) => {
                let method_name = mc.method.to_string();
                if method_name == "lamports" || method_name == "try_borrow_lamports" {
                    if let Some(sv_ids) =
                        self.resolve_field_access(callable, &mc.receiver, "lamports", aliases)
                    {
                        for sv_id in sv_ids {
                            reads.insert(sv_id);
                        }
                    }
                } else if method_name == "try_borrow_mut_lamports"
                    || method_name == "sub_lamports"
                    || method_name == "add_lamports"
                {
                    if let Some(sv_ids) =
                        self.resolve_field_access(callable, &mc.receiver, "lamports", aliases)
                    {
                        for sv_id in sv_ids {
                            writes.insert(sv_id);
                        }
                    }
                } else if let Some(receiver_struct) =
                    self.resolve_receiver_struct(callable, &mc.receiver, aliases)
                    && let Some(&target_id) = self
                        .functions_by_container_and_name
                        .get(&(receiver_struct, method_name))
                    && seen_targets.insert(target_id)
                {
                    outgoing.push((target_id, None));
                }
                self.walk_expr(
                    callable,
                    &mc.receiver,
                    false,
                    aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
                for arg in &mc.args {
                    self.walk_expr(
                        callable,
                        arg,
                        false,
                        aliases,
                        outgoing,
                        seen_targets,
                        reads,
                        writes,
                    );
                }
            }
            syn::Expr::Struct(expr_struct) => {
                // Detect Anchor CPI struct literals such as
                // `Transfer { from: ctx.accounts.vault_token.to_account_info(), to: ... }`
                // and mark their mutated token/lamport accounts as written.
                let struct_ident = expr_struct
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                let cpi_mutated_fields: &[&str] = match struct_ident.as_str() {
                    "Transfer" | "TransferChecked" => &["from", "to"],
                    "MintTo" | "MintToChecked" => &["mint", "to"],
                    "Burn" | "BurnChecked" => &["mint", "from", "account"],
                    "CloseAccount" => &["account", "destination"],
                    _ => &[],
                };
                for field_val in &expr_struct.fields {
                    let is_cpi_mut = matches!(&field_val.member, syn::Member::Named(m)
                        if cpi_mutated_fields.contains(&m.to_string().as_str()));
                    if is_cpi_mut
                        && let Some(sv_ids) =
                            self.resolve_cpi_account_expr(callable, &field_val.expr, aliases)
                    {
                        for sv_id in sv_ids {
                            writes.insert(sv_id);
                        }
                    }
                    self.walk_expr(
                        callable,
                        &field_val.expr,
                        false,
                        aliases,
                        outgoing,
                        seen_targets,
                        reads,
                        writes,
                    );
                }
            }
            syn::Expr::Block(eb) => {
                let mut inner_aliases = aliases.clone();
                self.walk_block(
                    callable,
                    &eb.block,
                    &mut inner_aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
            }
            syn::Expr::If(expr_if) => {
                self.walk_expr(
                    callable,
                    &expr_if.cond,
                    false,
                    aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
                let mut then_aliases = aliases.clone();
                self.walk_block(
                    callable,
                    &expr_if.then_branch,
                    &mut then_aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
                if let Some((_, else_expr)) = &expr_if.else_branch {
                    self.walk_expr(
                        callable,
                        else_expr,
                        false,
                        aliases,
                        outgoing,
                        seen_targets,
                        reads,
                        writes,
                    );
                }
            }
            syn::Expr::Match(expr_match) => {
                self.walk_expr(
                    callable,
                    &expr_match.expr,
                    false,
                    aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
                for arm in &expr_match.arms {
                    if let Some((_, guard)) = &arm.guard {
                        self.walk_expr(
                            callable,
                            guard,
                            false,
                            aliases,
                            outgoing,
                            seen_targets,
                            reads,
                            writes,
                        );
                    }
                    self.walk_expr(
                        callable,
                        &arm.body,
                        false,
                        aliases,
                        outgoing,
                        seen_targets,
                        reads,
                        writes,
                    );
                }
            }
            syn::Expr::ForLoop(fl) => {
                self.walk_expr(
                    callable,
                    &fl.expr,
                    false,
                    aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
                let mut body_aliases = aliases.clone();
                self.walk_block(
                    callable,
                    &fl.body,
                    &mut body_aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
            }
            syn::Expr::While(wl) => {
                self.walk_expr(
                    callable,
                    &wl.cond,
                    false,
                    aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
                let mut body_aliases = aliases.clone();
                self.walk_block(
                    callable,
                    &wl.body,
                    &mut body_aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
            }
            syn::Expr::Loop(lp) => {
                let mut body_aliases = aliases.clone();
                self.walk_block(
                    callable,
                    &lp.body,
                    &mut body_aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
            }
            syn::Expr::Try(tr) => {
                self.walk_expr(
                    callable,
                    &tr.expr,
                    is_write_ctx,
                    aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
            }
            syn::Expr::Paren(p) => {
                self.walk_expr(
                    callable,
                    &p.expr,
                    is_write_ctx,
                    aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
            }
            syn::Expr::Unary(u) => {
                self.walk_expr(
                    callable,
                    &u.expr,
                    is_write_ctx,
                    aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
            }
            syn::Expr::Return(ret) => {
                if let Some(e) = &ret.expr {
                    self.walk_expr(
                        callable,
                        e,
                        false,
                        aliases,
                        outgoing,
                        seen_targets,
                        reads,
                        writes,
                    );
                }
            }
            syn::Expr::Tuple(tup) => {
                for elem in &tup.elems {
                    self.walk_expr(
                        callable,
                        elem,
                        is_write_ctx,
                        aliases,
                        outgoing,
                        seen_targets,
                        reads,
                        writes,
                    );
                }
            }
            syn::Expr::Index(idx) => {
                self.walk_expr(
                    callable,
                    &idx.expr,
                    is_write_ctx,
                    aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
                self.walk_expr(
                    callable,
                    &idx.index,
                    false,
                    aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
            }
            syn::Expr::Cast(cast) => {
                self.walk_expr(
                    callable,
                    &cast.expr,
                    is_write_ctx,
                    aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
            }
            syn::Expr::Let(expr_let) => {
                if let syn::Pat::Ident(pat_ident) = Self::peel_pat(&expr_let.pat)
                    && let Some(alias) =
                        self.resolve_alias_initializer(callable, &expr_let.expr, aliases)
                {
                    aliases.insert(pat_ident.ident.to_string(), alias);
                }
                self.walk_expr(
                    callable,
                    &expr_let.expr,
                    false,
                    aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
            }
            syn::Expr::Macro(em) => {
                if let Ok(exprs) = em.mac.parse_body_with(
                    syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated,
                ) {
                    for e in exprs {
                        self.walk_expr(
                            callable,
                            &e,
                            false,
                            aliases,
                            outgoing,
                            seen_targets,
                            reads,
                            writes,
                        );
                    }
                }
            }
            syn::Expr::Unsafe(unsafe_expr) => {
                // Pinocchio / raw-Syscall code does its CPIs and account
                // writes inside `unsafe { ... }` — skipping the block
                // dropped every call and state touch in it.
                let mut body_aliases = aliases.clone();
                self.walk_block(
                    callable,
                    &unsafe_expr.block,
                    &mut body_aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
            }
            syn::Expr::Closure(closure) => {
                // IIFE-style closures run inline; walk the body so their
                // calls/state touches are attributed to the caller.
                self.walk_expr(
                    callable,
                    &closure.body,
                    false,
                    aliases,
                    outgoing,
                    seen_targets,
                    reads,
                    writes,
                );
            }
            syn::Expr::Range(range) => {
                if let Some(start) = &range.start {
                    self.walk_expr(
                        callable,
                        start,
                        false,
                        aliases,
                        outgoing,
                        seen_targets,
                        reads,
                        writes,
                    );
                }
                if let Some(end) = &range.end {
                    self.walk_expr(
                        callable,
                        end,
                        false,
                        aliases,
                        outgoing,
                        seen_targets,
                        reads,
                        writes,
                    );
                }
            }
            _ => {}
        }
    }

    fn resolve_path_call(
        &mut self,
        callable: &RecordedCallable,
        path: &syn::Path,
    ) -> Option<(i32, Option<String>)> {
        let segs: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
        let last = segs.last()?.as_str();
        let prev = segs
            .len()
            .checked_sub(2)
            .and_then(|i| segs.get(i))
            .map(String::as_str);

        // 1. Detect Solana / SPL / Anchor Cross-Program Invocations.
        match (prev, last) {
            (_, "invoke" | "invoke_signed" | "invoke_unchecked" | "invoke_signed_unchecked") => {
                let id = self.ensure_cpi_target("solana_cpi", last);
                return Some((id, Some("CPI".to_string())));
            }
            (
                Some("token" | "token_interface" | "token_2022" | "associated_token"),
                "transfer" | "transfer_checked" | "mint_to" | "mint_to_checked" | "burn"
                | "burn_checked" | "close_account" | "approve" | "revoke" | "create",
            ) => {
                let id = self.ensure_cpi_target("spl_token_cpi", last);
                return Some((id, Some("CPI".to_string())));
            }
            (
                Some("system_program" | "system_instruction"),
                "transfer" | "create_account" | "assign" | "allocate",
            ) => {
                let id = self.ensure_cpi_target("system_program_cpi", last);
                return Some((id, Some("CPI".to_string())));
            }
            _ => {}
        }

        // 2. Qualified `Container::fn` or `Self::fn`.
        if let Some(qualifier) = prev {
            let resolved_container = if qualifier == "Self" {
                callable
                    .self_struct_name
                    .as_deref()
                    .unwrap_or(&callable.container_name)
            } else {
                qualifier
            };
            if let Some(&id) = self
                .functions_by_container_and_name
                .get(&(resolved_container.to_string(), last.to_string()))
            {
                return Some((id, None));
            }
        }

        // 3. Same-container call, else unique workspace function match.
        if let Some(&id) = self
            .functions_by_container_and_name
            .get(&(callable.container_name.clone(), last.to_string()))
            && id != callable.function_id
        {
            return Some((id, None));
        }
        // Anchor `#[program]` delegation: `pub fn deposit(..)` in the
        // program module delegates to `instructions::deposit(..)` — two
        // workspace functions share the name and one of them is the caller
        // itself. Filter the caller out BEFORE counting, so the delegation
        // resolves to the single remaining candidate instead of failing the
        // `len == 1` check and dropping the edge.
        if let Some(candidates) = self.functions_by_name.get(last) {
            let mut foreign = candidates
                .iter()
                .copied()
                .filter(|&id| id != callable.function_id);
            let first = foreign.next();
            if let Some(first) = first
                && foreign.next().is_none()
            {
                return Some((first, None));
            }
        }

        None
    }

    fn resolve_alias_initializer(
        &self,
        callable: &RecordedCallable,
        init_expr: &syn::Expr,
        aliases: &BTreeMap<String, LocalAlias>,
    ) -> Option<LocalAlias> {
        let expr = Self::peel_wrappers(init_expr);
        // `BorshDeserialize::try_from_slice(...)` / `Vault::try_from_slice(...)`
        if let syn::Expr::Call(call) = expr
            && let syn::Expr::Path(ep) = call.func.as_ref()
            && ep.path.segments.len() >= 2
        {
            let first = ep.path.segments[0].ident.to_string();
            if self.contract_id_by_name.contains_key(&first) {
                return Some(LocalAlias::StateStruct(first));
            }
        }
        // `ctx.accounts.<acct>.load_mut()?` / `ctx.accounts.<acct>`
        if let Some((ctx_name, acct_field)) = self.extract_ctx_account_base(callable, expr) {
            if let Some(ctx_def) = self.accounts_contexts.get(&ctx_name)
                && let Some(bindings) = ctx_def.fields.get(&acct_field)
            {
                for b in bindings {
                    if let ContextFieldBinding::StateStruct(st) = b {
                        return Some(LocalAlias::StateStruct(st.clone()));
                    }
                }
            }
            return Some(LocalAlias::ContextAccount {
                context_name: ctx_name,
                account_field: acct_field,
            });
        }
        if let syn::Expr::Path(ep) = expr
            && ep.path.segments.len() == 1
        {
            let ident = ep.path.segments[0].ident.to_string();
            if let Some(existing) = aliases.get(&ident) {
                return Some(existing.clone());
            }
        }
        None
    }

    fn resolve_receiver_struct(
        &self,
        callable: &RecordedCallable,
        receiver: &syn::Expr,
        aliases: &BTreeMap<String, LocalAlias>,
    ) -> Option<String> {
        let base = Self::peel_wrappers(receiver);
        if let syn::Expr::Path(ep) = base
            && ep.path.segments.len() == 1
        {
            let ident = ep.path.segments[0].ident.to_string();
            if let Some(LocalAlias::StateStruct(s)) = aliases.get(&ident) {
                return Some(s.clone());
            }
        }
        if let Some((ctx_name, acct_field)) = self.extract_ctx_account_base(callable, base)
            && let Some(ctx_def) = self.accounts_contexts.get(&ctx_name)
            && let Some(bindings) = ctx_def.fields.get(&acct_field)
        {
            for b in bindings {
                if let ContextFieldBinding::StateStruct(s) = b {
                    return Some(s.clone());
                }
            }
        }
        None
    }

    fn resolve_field_access(
        &self,
        callable: &RecordedCallable,
        base_expr: &syn::Expr,
        field_name: &str,
        aliases: &BTreeMap<String, LocalAlias>,
    ) -> Option<Vec<i32>> {
        let base = Self::peel_wrappers(base_expr);

        // Case 1: local variable alias (`vault.total_assets` or `self.total_assets`).
        if let syn::Expr::Path(ep) = base
            && ep.path.segments.len() == 1
        {
            let ident = ep.path.segments[0].ident.to_string();
            match aliases.get(&ident) {
                Some(LocalAlias::StateStruct(struct_name)) => {
                    let &cid = self.contract_id_by_name.get(struct_name)?;
                    let &sv_id = self
                        .state_var_by_owner_field
                        .get(&(cid, field_name.to_string()))?;
                    return Some(vec![sv_id]);
                }
                Some(LocalAlias::ContextAccount {
                    context_name,
                    account_field,
                }) => {
                    return self.lookup_context_account_field(
                        context_name,
                        account_field,
                        field_name,
                    );
                }
                None => {}
            }
        }

        // Case 2: `ctx.accounts.<acct>.<field_name>` (or `accounts.<acct>.<field_name>`).
        if let Some((ctx_name, acct_field)) = self.extract_ctx_account_base(callable, base) {
            return self.lookup_context_account_field(&ctx_name, &acct_field, field_name);
        }

        None
    }

    fn resolve_cpi_account_expr(
        &self,
        callable: &RecordedCallable,
        expr: &syn::Expr,
        aliases: &BTreeMap<String, LocalAlias>,
    ) -> Option<Vec<i32>> {
        let base = Self::peel_wrappers(expr);
        if let Some((ctx_name, acct_field)) = self.extract_ctx_account_base(callable, base)
            && let Some(ctx_def) = self.accounts_contexts.get(&ctx_name)
            && let Some(bindings) = ctx_def.fields.get(&acct_field)
        {
            let ids: Vec<i32> = bindings
                .iter()
                .filter_map(|b| match b {
                    ContextFieldBinding::SyntheticVar { sv_id, .. } => Some(*sv_id),
                    ContextFieldBinding::StateStruct(_) => None,
                })
                .collect();
            if !ids.is_empty() {
                return Some(ids);
            }
        }
        if let syn::Expr::Path(ep) = base
            && ep.path.segments.len() == 1
        {
            let ident = ep.path.segments[0].ident.to_string();
            if let Some(LocalAlias::ContextAccount {
                context_name,
                account_field,
            }) = aliases.get(&ident)
                && let Some(ctx_def) = self.accounts_contexts.get(context_name)
                && let Some(bindings) = ctx_def.fields.get(account_field)
            {
                let ids: Vec<i32> = bindings
                    .iter()
                    .filter_map(|b| match b {
                        ContextFieldBinding::SyntheticVar { sv_id, .. } => Some(*sv_id),
                        ContextFieldBinding::StateStruct(_) => None,
                    })
                    .collect();
                if !ids.is_empty() {
                    return Some(ids);
                }
            }
        }
        None
    }

    fn lookup_context_account_field(
        &self,
        ctx_name: &str,
        acct_field: &str,
        field_name: &str,
    ) -> Option<Vec<i32>> {
        let ctx_def = self.accounts_contexts.get(ctx_name)?;
        let bindings = ctx_def.fields.get(acct_field)?;
        let mut out = Vec::new();
        for b in bindings {
            match b {
                ContextFieldBinding::StateStruct(struct_name) => {
                    if let Some(&cid) = self.contract_id_by_name.get(struct_name)
                        && let Some(&sv_id) = self
                            .state_var_by_owner_field
                            .get(&(cid, field_name.to_string()))
                    {
                        out.push(sv_id);
                    }
                }
                ContextFieldBinding::SyntheticVar {
                    sv_id,
                    field_suffix,
                } if *field_suffix == field_name => {
                    out.push(*sv_id);
                }
                ContextFieldBinding::SyntheticVar { .. } => {}
            }
        }
        (!out.is_empty()).then_some(out)
    }

    /// Match `ctx.accounts.<acct>` (with optional `.load()?`, `.load_mut()?`,
    /// `.to_account_info()`) or `<accounts_param>.<acct>`.
    fn extract_ctx_account_base(
        &self,
        callable: &RecordedCallable,
        expr: &syn::Expr,
    ) -> Option<(String, String)> {
        let ctx_name = callable.context_struct_names.first()?.clone();
        let peeled = Self::peel_account_method_wrappers(expr);
        let syn::Expr::Field(outer) = peeled else {
            return None;
        };
        let syn::Member::Named(acct_ident) = &outer.member else {
            return None;
        };
        let inner = Self::peel_wrappers(&outer.base);
        match inner {
            syn::Expr::Field(mid) => {
                if let syn::Member::Named(mid_ident) = &mid.member
                    && mid_ident == "accounts"
                {
                    return Some((ctx_name, acct_ident.to_string()));
                }
                None
            }
            syn::Expr::Path(ep) if ep.path.segments.len() == 1 => {
                let root = ep.path.segments[0].ident.to_string();
                if matches!(root.as_str(), "accounts" | "accs" | "ctx_accounts") {
                    return Some((ctx_name, acct_ident.to_string()));
                }
                None
            }
            _ => None,
        }
    }

    fn peel_wrappers(mut expr: &syn::Expr) -> &syn::Expr {
        loop {
            match expr {
                syn::Expr::Reference(r) => expr = &r.expr,
                syn::Expr::Paren(p) => expr = &p.expr,
                syn::Expr::Try(t) => expr = &t.expr,
                syn::Expr::Unary(u) if matches!(u.op, syn::UnOp::Deref(_)) => expr = &u.expr,
                _ => return expr,
            }
        }
    }

    fn peel_account_method_wrappers(mut expr: &syn::Expr) -> &syn::Expr {
        loop {
            expr = Self::peel_wrappers(expr);
            if let syn::Expr::MethodCall(mc) = expr
                && matches!(
                    mc.method.to_string().as_str(),
                    "load"
                        | "load_mut"
                        | "load_init"
                        | "to_account_info"
                        | "as_ref"
                        | "as_mut"
                        | "clone"
                )
            {
                expr = &mc.receiver;
                continue;
            }
            return expr;
        }
    }

    fn peel_pat(mut pat: &syn::Pat) -> &syn::Pat {
        loop {
            match pat {
                syn::Pat::Type(pt) => pat = &pt.pat,
                syn::Pat::Paren(pp) => pat = &pp.pat,
                _ => return pat,
            }
        }
    }

    fn has_program_attr(attrs: &[syn::Attribute]) -> bool {
        attrs.iter().any(|attr| {
            attr.path()
                .segments
                .last()
                .map(|seg| seg.ident == "program")
                .unwrap_or(false)
        })
    }

    fn has_accounts_derive(attrs: &[syn::Attribute]) -> bool {
        for attr in attrs {
            if !attr.path().is_ident("derive") {
                continue;
            }
            let mut found = false;
            let _ = attr.parse_nested_meta(|meta| {
                if meta
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident == "Accounts")
                    .unwrap_or(false)
                {
                    found = true;
                }
                Ok(())
            });
            if found {
                return true;
            }
        }
        false
    }

    fn has_mut_account_attr(attrs: &[syn::Attribute]) -> bool {
        for attr in attrs {
            if !attr.path().is_ident("account") {
                continue;
            }
            let mut is_mut = false;
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("mut")
                    || meta.path.is_ident("init")
                    || meta.path.is_ident("close")
                {
                    is_mut = true;
                }
                // Consume optional `= expr` or `(...)` so remaining metas parse cleanly.
                if meta.input.peek(syn::Token![=]) {
                    let _: syn::Token![=] = meta.input.parse()?;
                    let _: syn::Expr = meta.input.parse()?;
                } else if meta.input.peek(syn::token::Paren) {
                    let content;
                    syn::parenthesized!(content in meta.input);
                    let _: proc_macro2::TokenStream = content.parse()?;
                }
                Ok(())
            });
            if is_mut {
                return true;
            }
        }
        false
    }

    fn is_state_struct(item_struct: &syn::ItemStruct, referenced_as_account: bool) -> bool {
        if referenced_as_account {
            return true;
        }
        for attr in &item_struct.attrs {
            if let Some(seg) = attr.path().segments.last() {
                let ident = seg.ident.to_string();
                if matches!(ident.as_str(), "account" | "zero_copy" | "constant") {
                    return true;
                }
            }
            if attr.path().is_ident("derive") {
                let mut state_derive = false;
                let _ = attr.parse_nested_meta(|meta| {
                    if let Some(seg) = meta.path.segments.last()
                        && matches!(
                            seg.ident.to_string().as_str(),
                            "BorshSerialize"
                                | "BorshDeserialize"
                                | "AnchorSerialize"
                                | "AnchorDeserialize"
                                | "Pod"
                                | "Zeroable"
                                | "Pack"
                        )
                    {
                        state_derive = true;
                    }
                    Ok(())
                });
                if state_derive {
                    return true;
                }
            }
        }
        false
    }

    /// Unwrap `Box<...>`, `Option<...>`, and inspect Anchor account wrappers
    /// such as `Account<'info, Vault>`, `AccountLoader<'info, Pool>`,
    /// `InterfaceAccount<'info, TokenAccount>`.
    fn classify_account_wrapper_type(ty: &syn::Type) -> Option<(String, Option<String>)> {
        let syn::Type::Path(tp) = ty else {
            return None;
        };
        let seg = tp.path.segments.last()?;
        let name = seg.ident.to_string();
        if matches!(name.as_str(), "Box" | "Option") {
            if let syn::PathArguments::AngleBracketed(ab) = &seg.arguments {
                for arg in &ab.args {
                    if let syn::GenericArgument::Type(inner_ty) = arg {
                        return Self::classify_account_wrapper_type(inner_ty);
                    }
                }
            }
            return None;
        }
        let mut first_type_arg = None;
        if let syn::PathArguments::AngleBracketed(ab) = &seg.arguments {
            for arg in &ab.args {
                if let syn::GenericArgument::Type(inner_ty) = arg {
                    first_type_arg = Self::type_last_ident(inner_ty);
                    if first_type_arg.is_some() {
                        break;
                    }
                }
            }
        }
        Some((name, first_type_arg))
    }

    fn extract_anchor_context_type(ty: &syn::Type) -> Option<String> {
        let syn::Type::Path(tp) = ty else {
            return None;
        };
        let seg = tp.path.segments.last()?;
        if seg.ident != "Context" && seg.ident != "CpiContext" {
            return None;
        }
        if seg.ident == "CpiContext" {
            return None;
        }
        if let syn::PathArguments::AngleBracketed(ab) = &seg.arguments {
            for arg in &ab.args {
                if let syn::GenericArgument::Type(inner_ty) = arg {
                    return Self::type_last_ident(inner_ty);
                }
            }
        }
        None
    }

    fn unwrap_ref_or_box_ident(ty: &syn::Type) -> Option<String> {
        match ty {
            syn::Type::Reference(r) => Self::unwrap_ref_or_box_ident(&r.elem),
            syn::Type::Path(tp) => {
                let seg = tp.path.segments.last()?;
                if seg.ident == "Box"
                    && let syn::PathArguments::AngleBracketed(ab) = &seg.arguments
                {
                    for arg in &ab.args {
                        if let syn::GenericArgument::Type(inner) = arg {
                            return Self::unwrap_ref_or_box_ident(inner);
                        }
                    }
                }
                Some(seg.ident.to_string())
            }
            _ => None,
        }
    }

    fn type_last_ident(ty: &syn::Type) -> Option<String> {
        match ty {
            syn::Type::Path(tp) => tp.path.segments.last().map(|s| s.ident.to_string()),
            syn::Type::Reference(r) => Self::type_last_ident(&r.elem),
            _ => None,
        }
    }

    fn is_token_account_type(name: &str) -> bool {
        matches!(name, "TokenAccount" | "Token2022Account")
    }

    fn is_spl_builtin_account(name: &str) -> bool {
        matches!(
            name,
            "TokenAccount"
                | "Mint"
                | "Token"
                | "Token2022"
                | "TokenInterface"
                | "System"
                | "AssociatedToken"
                | "Rent"
                | "Clock"
        )
    }

    fn format_fn_inputs(line_idx: &SourceLineIndex<'_>, sig: &syn::Signature) -> String {
        sig.inputs
            .iter()
            .map(|arg| Self::compact_whitespace(line_idx.slice_span(arg.span())))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn compact_whitespace(raw: &str) -> String {
        raw.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    fn chunk_from_span(line_idx: &SourceLineIndex<'_>, span: proc_macro2::Span) -> FileChunk {
        let start = span.start();
        let end = span.end();
        let content = line_idx.slice_span(span).to_string();
        FileChunk {
            loc: FileLocation {
                start_line: start.line.max(1),
                start_column: start.column,
                end_line: end.line.max(start.line.max(1)),
                end_column: end.column,
            },
            content,
        }
    }

    fn full_file_chunk(source: &str) -> FileChunk {
        let mut line_count = 1usize;
        let mut last_col = 0usize;
        for ch in source.chars() {
            if ch == '\n' {
                line_count += 1;
                last_col = 0;
            } else {
                last_col += 1;
            }
        }
        FileChunk {
            loc: FileLocation {
                start_line: 1,
                start_column: 0,
                end_line: line_count,
                end_column: last_col,
            },
            content: source.to_string(),
        }
    }

    fn finish(self) -> RustStaticAnalysis {
        let mut containers = Vec::new();
        for (&id, c) in &self.contracts {
            let kind = self
                .container_kinds
                .get(&id)
                .copied()
                .unwrap_or(RustContainerKind::Module);
            let functions = c
                .functions
                .iter()
                .map(|f| ExtractedRustCallable {
                    id: f.id,
                    container_id: id,
                    container_name: c.name.clone(),
                    name: f.name.clone(),
                    args: f.args.clone(),
                    relative_file_path: f.relative_file_path.clone(),
                    chunk: FileChunk {
                        loc: f.loc,
                        content: f.content.clone().unwrap_or_default(),
                    },
                })
                .collect();
            containers.push(ExtractedRustContainer {
                id,
                name: c.name.clone(),
                kind,
                relative_file_path: c.relative_file_path.clone(),
                chunk: c.chunk.clone(),
                functions,
            });
        }
        for (&id, i) in &self.interfaces {
            if i.relative_file_path == Path::new("<solana-cpi>") {
                continue;
            }
            let functions = i
                .functions
                .iter()
                .map(|f| ExtractedRustCallable {
                    id: f.id,
                    container_id: id,
                    container_name: i.name.clone(),
                    name: f.name.clone(),
                    args: f.args.clone(),
                    relative_file_path: f.relative_file_path.clone(),
                    chunk: FileChunk {
                        loc: f.loc,
                        content: f.content.clone().unwrap_or_default(),
                    },
                })
                .collect();
            containers.push(ExtractedRustContainer {
                id,
                name: i.name.clone(),
                kind: RustContainerKind::TraitInterface,
                relative_file_path: i.relative_file_path.clone(),
                chunk: i.chunk.clone(),
                functions,
            });
        }

        let contract_count = self.contracts.len();
        let interface_count = self.interfaces.len();
        let function_count: usize = self
            .contracts
            .values()
            .map(|c| c.functions.len())
            .sum::<usize>()
            + self
                .interfaces
                .values()
                .map(|i| i.functions.len())
                .sum::<usize>();
        let call_count: usize = self
            .contracts
            .values()
            .flat_map(|c| &c.functions)
            .map(|f| f.calls.len())
            .sum::<usize>()
            + self
                .interfaces
                .values()
                .flat_map(|i| &i.functions)
                .map(|f| f.calls.len())
                .sum::<usize>();

        let inherits: Vec<ContractInherit> = self
            .contract_inherits
            .into_iter()
            .map(|(contract_id, inherited_id)| ContractInherit {
                contract_id,
                inherited_id,
            })
            .collect();
        let state_variable_count = self.state_variables.len();
        let contract_inherit_count = inherits.len();
        let contract_variable_count = self.contract_variables.len();
        let function_state_variable_count = self.function_state_variables.len();

        RustStaticAnalysis {
            call_graph: RustCallGraphResult {
                call_graph: CallGraph {
                    contracts: self.contracts,
                    interfaces: self.interfaces,
                },
                contract_count,
                interface_count,
                function_count,
                call_count,
            },
            storage: RustStorageResult {
                storage: StorageGraph {
                    state_variables: self.state_variables,
                    contract_variables: self.contract_variables,
                    function_state_variables: self.function_state_variables,
                },
                inheritance: InheritanceGraph::new(inherits),
                state_variable_count,
                contract_inherit_count,
                contract_variable_count,
                function_state_variable_count,
            },
            containers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyzes_anchor_vault_call_graph_cpi_and_alias_storage_edges() {
        let lib_rs = r#"
            use anchor_lang::prelude::*;
            use anchor_spl::token::{self, TokenAccount, Transfer};

            declare_id!("Fg6PaFpoGXkYsidMpWTK6W2BeZ7FEfcYkg476zPFsLnS");

            #[program]
            pub mod solana_vault {
                use super::*;

                pub fn withdraw(ctx: Context<Withdraw>, shares: u64) -> Result<()> {
                    let payout = math::shares_to_assets(shares, ctx.accounts.vault.total_assets);
                    let vault = &mut ctx.accounts.vault;
                    vault.total_shares -= shares;
                    token::transfer(
                        CpiContext::new(
                            ctx.accounts.token_program.to_account_info(),
                            Transfer {
                                from: ctx.accounts.vault_token.to_account_info(),
                                to: ctx.accounts.user_token.to_account_info(),
                                authority: vault.to_account_info(),
                            },
                        ),
                        payout,
                    )?;
                    Ok(())
                }
            }

            #[account]
            pub struct Vault {
                pub authority: Pubkey,
                pub total_assets: u64,
                pub total_shares: u64,
            }

            #[derive(Accounts)]
            pub struct Withdraw<'info> {
                #[account(mut)]
                pub vault: Account<'info, Vault>,
                #[account(mut)]
                pub vault_token: Account<'info, TokenAccount>,
                #[account(mut)]
                pub user_token: Account<'info, TokenAccount>,
                pub token_program: AccountInfo<'info>,
            }
        "#;

        let math_rs = r#"
            pub fn shares_to_assets(shares: u64, total_assets: u64) -> u64 {
                shares.saturating_mul(total_assets) / 1_000_000
            }
        "#;

        let analysis = RustStaticAnalyzer::analyze(&[
            RustSourceInput {
                relative_path: PathBuf::from("programs/vault/src/lib.rs"),
                content: lib_rs.to_string(),
            },
            RustSourceInput {
                relative_path: PathBuf::from("programs/vault/src/math.rs"),
                content: math_rs.to_string(),
            },
        ])
        .expect("analyze anchor vault");

        let cg = &analysis.call_graph.call_graph;
        let sg = &analysis.storage.storage;

        let prog_contract = cg
            .contracts
            .values()
            .find(|c| c.name == "solana_vault")
            .expect("solana_vault contract");
        let vault_struct = cg
            .contracts
            .values()
            .find(|c| c.name == "Vault")
            .expect("Vault state struct contract");
        let withdraw_fn = prog_contract
            .functions
            .iter()
            .find(|f| f.name == "withdraw")
            .expect("withdraw function");

        // Program module inherits `Vault` state struct via `Context<Withdraw>`.
        assert!(
            analysis
                .storage
                .inheritance
                .inherits
                .iter()
                .any(|e| e.contract_id == prog_contract.id && e.inherited_id == vault_struct.id),
            "solana_vault must inherit Vault so ProjectIndex exposes Vault fields"
        );

        // `withdraw` calls both `math::shares_to_assets` and `spl_token_cpi::transfer` (CPI).
        assert_eq!(withdraw_fn.calls.len(), 2);
        assert!(
            withdraw_fn
                .calls
                .iter()
                .any(|c| c.description.as_deref() == Some("CPI")),
            "expected outgoing CPI edge on token::transfer"
        );

        // Storage analysis: `total_assets` is read, `total_shares` is read+written (`-=`),
        // and `vault_token.amount` is written via `Transfer { from: ... }`.
        let total_assets_id = sg
            .state_variables
            .values()
            .find(|v| v.name == "total_assets")
            .expect("total_assets state var")
            .id;
        let total_shares_id = sg
            .state_variables
            .values()
            .find(|v| v.name == "total_shares")
            .expect("total_shares state var")
            .id;
        let vault_token_amt_id = sg
            .state_variables
            .values()
            .find(|v| v.name == "vault_token.amount")
            .expect("synthetic vault_token.amount state var")
            .id;

        assert!(sg.function_state_variables.iter().any(|e| {
            e.function_id == withdraw_fn.id && e.state_variable_id == total_assets_id && !e.is_write
        }));
        assert!(sg.function_state_variables.iter().any(|e| {
            e.function_id == withdraw_fn.id && e.state_variable_id == total_shares_id && e.is_write
        }));
        assert!(sg.function_state_variables.iter().any(|e| {
            e.function_id == withdraw_fn.id
                && e.state_variable_id == vault_token_amt_id
                && e.is_write
        }));
    }

    #[test]
    fn analyzes_native_solana_program_borsh_and_invoke_signed() {
        let processor_rs = r#"
            use borsh::{BorshDeserialize, BorshSerialize};
            use solana_program::program::invoke_signed;

            #[derive(BorshSerialize, BorshDeserialize)]
            pub struct PoolState {
                pub reserve_lamports: u64,
                pub paused: bool,
            }

            pub fn process_swap(pool: &mut PoolState, delta: u64) {
                if !pool.paused {
                    pool.reserve_lamports = pool.reserve_lamports + delta;
                    invoke_signed(&ix, &accounts, &[&[b"pool"]]);
                }
            }
        "#;

        let analysis = RustStaticAnalyzer::analyze(&[RustSourceInput {
            relative_path: PathBuf::from("src/processor.rs"),
            content: processor_rs.to_string(),
        }])
        .expect("analyze native solana program");

        let cg = &analysis.call_graph.call_graph;
        let sg = &analysis.storage.storage;

        let proc_mod = cg
            .contracts
            .values()
            .find(|c| c.name == "processor")
            .expect("processor module");
        let swap_fn = proc_mod
            .functions
            .iter()
            .find(|f| f.name == "process_swap")
            .expect("process_swap fn");
        assert!(
            swap_fn
                .calls
                .iter()
                .any(|c| c.description.as_deref() == Some("CPI")),
            "invoke_signed must produce CPI call edge"
        );

        let reserve_id = sg
            .state_variables
            .values()
            .find(|v| v.name == "reserve_lamports")
            .expect("reserve_lamports state var")
            .id;
        assert!(sg.function_state_variables.iter().any(|e| {
            e.function_id == swap_fn.id && e.state_variable_id == reserve_id && e.is_write
        }));
    }
}
