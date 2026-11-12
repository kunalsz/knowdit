//! The Solidity project index: precomputed call-graph / storage / inheritance
//! lookups the spec-gen agent's tools resolve against, plus the agent-memory
//! renderings derived from it. Split out of `spec::mod` so the index and its
//! derived views live together.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use color_eyre::eyre::Result;
use knowdit_kg_model::ExtractedSemantic;
use knowdit_repo_model::cg::{CallGraph, Contract, Function, FunctionCall, Interface};
use knowdit_repo_model::inheritance::{ContractInherit, InheritanceGraph};
use knowdit_repo_model::storage::{ContractVariable, StateVariable, StorageGraph};
use llmy::agent::tools::memory::{AgentMemory, AgentMemoryContent, AgentMemoryContext};

/// Static external-callability and state-mutability classification of a callable
/// parsed from its source signature header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallableReachability {
    /// Externally invocable and capable of mutating state (`external`/`public`
    /// without `view`/`pure`, `fallback`/`receive`, interface calls, or Rust `pub fn`).
    ExternalMutating,
    /// Externally invocable read-only callable (`external`/`public` with `view`/`pure`).
    ExternalReadOnly,
    /// Not externally invocable (`internal`, `private`, `constructor`, `modifier`,
    /// or Rust private / `pub(crate)` / `pub(super)`).
    NonExternal,
    /// Source body absent (`Function.content == None`) or unparseable; treated
    /// fail-open by `SetCallSequenceTool` and ignored as synthetic by U2.
    Unknown,
}

impl CallableReachability {
    fn overload_priority(self) -> u8 {
        match self {
            Self::ExternalMutating => 4,
            Self::Unknown => 3,
            Self::ExternalReadOnly => 2,
            Self::NonExternal => 1,
        }
    }

    /// Combine reachability across multiple overloads or inherited declarations
    /// sharing `(contract, function)`: if at least one overload is externally
    /// callable/mutating (or `Unknown`), prefer it over `NonExternal`.
    pub fn merge_overload(self, other: Self) -> Self {
        if other.overload_priority() > self.overload_priority() {
            other
        } else {
            self
        }
    }

    pub(crate) fn classify_function(
        function: &Function,
        container_kind: ContainerKind,
        container_is_program_module: bool,
    ) -> Self {
        let Some(source) = function.content.as_deref() else {
            return Self::Unknown;
        };
        match function
            .relative_file_path
            .extension()
            .and_then(|e| e.to_str())
        {
            Some("rs") => match knowdit_rust::extract_function_signature(source) {
                Ok(Some(sig)) => {
                    Self::classify_rust_signature(&sig, container_kind, container_is_program_module)
                }
                _ => Self::Unknown,
            },
            Some("move") => Self::Unknown,
            _ => match knowdit_sol::signature::extract_function_signature(source) {
                Ok(Some(sig)) => Self::classify_solidity_signature(&sig, container_kind),
                _ => Self::Unknown,
            },
        }
    }

    /// True when a contract container was produced from a Rust `#[program]`
    /// module (see `RustContainerKind::ProgramModule`). Only its `pub fn`s
    /// are CPI-invocable SBF instructions.
    fn is_program_module_description(description: Option<&str>) -> bool {
        description.is_some_and(|d| d.contains("#[program] entrypoint module"))
    }

    fn classify_rust_signature(
        sig: &str,
        container_kind: ContainerKind,
        container_is_program_module: bool,
    ) -> Self {
        if container_kind == ContainerKind::Interface {
            return Self::ExternalMutating;
        }
        let trimmed = sig.trim_start();
        let is_bare_pub_fn = trimmed.strip_prefix("pub").is_some_and(|after_pub| {
            after_pub.starts_with(char::is_whitespace) && !after_pub.trim_start().starts_with('(')
        });
        if is_bare_pub_fn {
            // A bare `pub fn` is only an SBF instruction entry when it lives
            // in the `#[program]` module. Helper modules and impls use
            // `pub fn` for crate-internal reuse all the time — treating
            // those as externally invocable made Solana specs target
            // un-invocable internal Rust functions. `pub extern` (C ABI)
            // stays external everywhere: it is a real FFI boundary
            // regardless of container.
            if container_is_program_module || trimmed.starts_with("pub extern") {
                return Self::ExternalMutating;
            }
            return Self::NonExternal;
        }
        Self::NonExternal
    }

    fn classify_solidity_signature(sig: &str, container_kind: ContainerKind) -> Self {
        let trimmed = sig.trim();
        let leading_end = trimmed
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .unwrap_or(trimmed.len());
        let keyword = &trimmed[..leading_end];
        match keyword {
            "constructor" | "modifier" => Self::NonExternal,
            "receive" | "fallback" => {
                let specifiers =
                    Self::solidity_top_level_specifiers(&trimmed[leading_end..], false);
                let is_readonly = specifiers
                    .iter()
                    .any(|tok| matches!(tok.as_str(), "view" | "pure" | "constant"));
                if is_readonly {
                    Self::ExternalReadOnly
                } else {
                    Self::ExternalMutating
                }
            }
            "function" => {
                let specifiers = Self::solidity_top_level_specifiers(&trimmed[leading_end..], true);
                let is_readonly = specifiers
                    .iter()
                    .any(|tok| matches!(tok.as_str(), "view" | "pure" | "constant"));
                let is_ext_or_pub = specifiers
                    .iter()
                    .any(|tok| matches!(tok.as_str(), "external" | "public"));
                let is_int_or_priv = specifiers
                    .iter()
                    .any(|tok| matches!(tok.as_str(), "internal" | "private"));
                if container_kind == ContainerKind::Interface {
                    if is_int_or_priv {
                        Self::NonExternal
                    } else if is_readonly {
                        Self::ExternalReadOnly
                    } else {
                        Self::ExternalMutating
                    }
                } else if is_ext_or_pub {
                    if is_readonly {
                        Self::ExternalReadOnly
                    } else {
                        Self::ExternalMutating
                    }
                } else {
                    Self::NonExternal
                }
            }
            _ => Self::Unknown,
        }
    }

    /// Extract top-level identifier specifiers outside parentheses and string
    /// literals. When `after_first_parens` is true (for `function name(...)`),
    /// only tokens appearing after the parameter list `(...)` closes are
    /// collected so parameter names cannot masquerade as visibility keywords.
    fn solidity_top_level_specifiers(rest: &str, after_first_parens: bool) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        let mut paren_depth = 0usize;
        let mut passed_params = !after_first_parens;
        let mut saw_open_paren = false;
        let mut in_quote: Option<char> = None;
        let mut escape = false;

        for ch in rest.chars() {
            if let Some(q) = in_quote {
                if escape {
                    escape = false;
                } else if ch == '\\' {
                    escape = true;
                } else if ch == q {
                    in_quote = None;
                }
                continue;
            }
            if ch == '"' || ch == '\'' {
                if !current.is_empty() {
                    if passed_params && paren_depth == 0 {
                        tokens.push(std::mem::take(&mut current));
                    } else {
                        current.clear();
                    }
                }
                in_quote = Some(ch);
                continue;
            }
            if ch == '(' {
                if !current.is_empty() {
                    if passed_params && paren_depth == 0 {
                        tokens.push(std::mem::take(&mut current));
                    } else {
                        current.clear();
                    }
                }
                paren_depth += 1;
                saw_open_paren = true;
                continue;
            }
            if ch == ')' {
                if !current.is_empty() {
                    current.clear();
                }
                paren_depth = paren_depth.saturating_sub(1);
                if saw_open_paren && paren_depth == 0 {
                    passed_params = true;
                }
                continue;
            }
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '$' {
                if passed_params && paren_depth == 0 {
                    current.push(ch);
                }
            } else if !current.is_empty() {
                if passed_params && paren_depth == 0 {
                    tokens.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
            }
        }
        if !current.is_empty() && passed_params && paren_depth == 0 {
            tokens.push(current);
        }
        tokens
    }
}

/// One in-scope externally callable state-mutating entrypoint discovered by
/// [`ProjectIndex::stateful_external_entrypoints`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatefulEntrypoint {
    pub contract_id: i32,
    pub contract_name: String,
    pub relative_file_path: PathBuf,
    pub function_id: i32,
    pub function_name: String,
    pub signature: String,
    pub written_state_vars: Vec<String>,
}

#[derive(Debug)]
pub struct ProjectIndex {
    /// Held by `Arc` so downstream consumers (e.g. the harness
    /// `RunForgeTool` that needs a `&CallGraph` for the coverage gate)
    /// can keep a cheap reference instead of triggering a full
    /// `CallGraph::clone()` per spec — which over thousands of specs
    /// fragmented glibc malloc badly enough to OOM the orchestrator.
    pub call_graph: Arc<CallGraph>,
    pub storage_graph: StorageGraph,
    pub inheritance_graph: InheritanceGraph,
    /// `function.id -> (relative_file_path, contract_or_interface_name, kind)`
    pub(crate) function_owner: BTreeMap<i32, FunctionOwner>,
    /// Lower-cased contract name -> contract ids (multiple files may reuse).
    pub contract_ids_by_name: BTreeMap<String, Vec<i32>>,
    /// Lower-cased interface name -> interface ids.
    pub interface_ids_by_name: BTreeMap<String, Vec<i32>>,
    /// state_variable_id -> declaring contract id (from contract_variable).
    pub declaring_contract_for_var: BTreeMap<i32, i32>,
    /// contract_id -> all state-variable ids (own + inherited via the
    /// transitive closure of contract_inherits).
    pub visible_state_vars: BTreeMap<i32, BTreeSet<i32>>,
    /// Lower-cased state-variable name -> state_variable ids.
    pub state_var_ids_by_name: BTreeMap<String, Vec<i32>>,
}

#[derive(Debug, Clone)]
pub(crate) struct FunctionOwner {
    pub(crate) relative_file_path: String,
    pub(crate) container_kind: ContainerKind,
    pub(crate) container_name: String,
    pub(crate) container_id: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContainerKind {
    Contract,
    Interface,
}

impl ContainerKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Contract => "contract",
            Self::Interface => "interface",
        }
    }
}

impl ProjectIndex {
    pub fn build(
        call_graph: CallGraph,
        storage_graph: StorageGraph,
        inheritance_graph: InheritanceGraph,
    ) -> Self {
        // Wrap the call graph in an `Arc` up front so every downstream
        // hand-off (including the per-spec `RunForgeTool::callgraph`)
        // is a refcount bump rather than a deep clone of every contract
        // / function / source body.
        let call_graph = Arc::new(call_graph);
        let mut function_owner: BTreeMap<i32, FunctionOwner> = BTreeMap::new();
        let mut contract_ids_by_name: BTreeMap<String, Vec<i32>> = BTreeMap::new();
        let mut interface_ids_by_name: BTreeMap<String, Vec<i32>> = BTreeMap::new();
        for contract in call_graph.contracts.values() {
            contract_ids_by_name
                .entry(contract.name.to_lowercase())
                .or_default()
                .push(contract.id);
            for function in &contract.functions {
                function_owner.insert(
                    function.id,
                    FunctionOwner {
                        relative_file_path: contract.relative_file_path.display().to_string(),
                        container_kind: ContainerKind::Contract,
                        container_name: contract.name.clone(),
                        container_id: contract.id,
                    },
                );
            }
        }
        for interface in call_graph.interfaces.values() {
            interface_ids_by_name
                .entry(interface.name.to_lowercase())
                .or_default()
                .push(interface.id);
            for function in &interface.functions {
                function_owner.insert(
                    function.id,
                    FunctionOwner {
                        relative_file_path: interface.relative_file_path.display().to_string(),
                        container_kind: ContainerKind::Interface,
                        container_name: interface.name.clone(),
                        container_id: interface.id,
                    },
                );
            }
        }

        let mut declaring_contract_for_var: BTreeMap<i32, i32> = BTreeMap::new();
        for ContractVariable {
            contract_id,
            state_variable_id,
            ..
        } in &storage_graph.contract_variables
        {
            declaring_contract_for_var
                .entry(*state_variable_id)
                .or_insert(*contract_id);
        }

        let inherits = inheritance_closure(&inheritance_graph.inherits);
        let mut visible_state_vars: BTreeMap<i32, BTreeSet<i32>> = BTreeMap::new();
        for ContractVariable {
            contract_id,
            state_variable_id,
            ..
        } in &storage_graph.contract_variables
        {
            visible_state_vars
                .entry(*contract_id)
                .or_default()
                .insert(*state_variable_id);
        }
        // Add inherited variables to descendants.
        let snapshot = visible_state_vars.clone();
        for (contract_id, ancestors) in &inherits {
            let entry = visible_state_vars.entry(*contract_id).or_default();
            for ancestor in ancestors {
                if let Some(vars) = snapshot.get(ancestor) {
                    for var in vars {
                        entry.insert(*var);
                    }
                }
            }
        }

        let mut state_var_ids_by_name: BTreeMap<String, Vec<i32>> = BTreeMap::new();
        for var in storage_graph.state_variables.values() {
            state_var_ids_by_name
                .entry(var.name.to_lowercase())
                .or_default()
                .push(var.id);
        }

        Self {
            call_graph,
            storage_graph,
            inheritance_graph,
            function_owner,
            contract_ids_by_name,
            interface_ids_by_name,
            declaring_contract_for_var,
            visible_state_vars,
            state_var_ids_by_name,
        }
    }

    pub(crate) fn contract(&self, id: i32) -> Option<&Contract> {
        self.call_graph.contracts.get(&id)
    }

    pub(crate) fn interface(&self, id: i32) -> Option<&Interface> {
        self.call_graph.interfaces.get(&id)
    }

    pub(crate) fn state_variable(&self, id: i32) -> Option<&StateVariable> {
        self.storage_graph.state_variables.get(&id)
    }

    pub(crate) fn lookup_contract(&self, name: &str) -> Vec<i32> {
        self.contract_ids_by_name
            .get(&name.to_lowercase())
            .cloned()
            .unwrap_or_default()
    }

    pub(crate) fn lookup_interface(&self, name: &str) -> Vec<i32> {
        self.interface_ids_by_name
            .get(&name.to_lowercase())
            .cloned()
            .unwrap_or_default()
    }

    pub(crate) fn lookup_state_var(&self, name: &str) -> Vec<i32> {
        self.state_var_ids_by_name
            .get(&name.to_lowercase())
            .cloned()
            .unwrap_or_default()
    }

    /// True when `name` resolves to a project contract or interface.
    /// Used by the spec-builder tools to reject hallucinated contract
    /// references before they enter the draft.
    pub(crate) fn has_contract(&self, name: &str) -> bool {
        !self.lookup_contract(name).is_empty() || !self.lookup_interface(name).is_empty()
    }

    /// True when `key` resolves to a project state variable. Accepts a
    /// bare `fieldName` (matched anywhere in the project) or a qualified
    /// `Contract.field` (the field must be visible to that contract,
    /// inherited members included).
    pub(crate) fn has_state_variable(&self, key: &str) -> bool {
        let key = key.trim();
        if !self.lookup_state_var(key).is_empty() {
            return true;
        }
        match key.split_once('.') {
            Some((contract, field)) => {
                let var_ids = self.lookup_state_var(field.trim());
                if var_ids.is_empty() {
                    return false;
                }
                self.lookup_contract(contract.trim()).iter().any(|cid| {
                    self.visible_state_vars
                        .get(cid)
                        .map(|visible| var_ids.iter().any(|vid| visible.contains(vid)))
                        .unwrap_or(false)
                })
            }
            None => false,
        }
    }

    /// True when project contract/interface `contract` declares — or
    /// inherits — a function named `function` (case-insensitive).
    #[cfg(test)]
    pub(crate) fn contract_has_function(&self, contract: &str, function: &str) -> bool {
        self.classify_sequence_callable(contract, function)
            .is_some()
    }

    /// Classify `(contract, function)` across `contract`, its transitive
    /// `inheritance_graph.ancestors_of(cid)`, and `interfaces`. Returns `None`
    /// only when no matching function exists on any of those containers.
    pub fn classify_sequence_callable(
        &self,
        contract: &str,
        function: &str,
    ) -> Option<CallableReachability> {
        let target = function.trim().to_lowercase();
        let mut combined: Option<CallableReachability> = None;
        let mut fold_fn = |f: &Function, kind: ContainerKind, is_program_module: bool| {
            if f.name.to_lowercase() == target {
                let reach = CallableReachability::classify_function(f, kind, is_program_module);
                combined = Some(match combined {
                    Some(prev) => prev.merge_overload(reach),
                    None => reach,
                });
            }
        };

        for cid in self.lookup_contract(contract) {
            if let Some(c) = self.contract(cid) {
                let is_program_module =
                    CallableReachability::is_program_module_description(c.description.as_deref());
                for f in &c.functions {
                    fold_fn(f, ContainerKind::Contract, is_program_module);
                }
            }
            for ancestor_id in self.inheritance_graph.ancestors_of(cid) {
                if let Some(anc) = self.contract(ancestor_id) {
                    let anc_is_program_module = CallableReachability::is_program_module_description(
                        anc.description.as_deref(),
                    );
                    for f in &anc.functions {
                        fold_fn(f, ContainerKind::Contract, anc_is_program_module);
                    }
                }
                if let Some(iface) = self.interface(ancestor_id) {
                    for f in &iface.functions {
                        fold_fn(f, ContainerKind::Interface, false);
                    }
                }
            }
        }
        for iid in self.lookup_interface(contract) {
            if let Some(iface) = self.interface(iid) {
                for f in &iface.functions {
                    fold_fn(f, ContainerKind::Interface, false);
                }
            }
        }
        combined
    }

    /// Extract human-readable full signature headers for all callables matching
    /// `(contract, function)` across `contract`, its ancestors, and interfaces.
    pub fn sequence_step_signatures(&self, contract: &str, function: &str) -> Vec<String> {
        let target = function.trim().to_lowercase();
        let mut out: BTreeSet<String> = BTreeSet::new();
        let mut collect_fn = |f: &Function| {
            if f.name.to_lowercase() == target {
                out.insert(self.extract_clean_signature(f));
            }
        };
        for cid in self.lookup_contract(contract) {
            if let Some(c) = self.contract(cid) {
                for f in &c.functions {
                    collect_fn(f);
                }
            }
            for ancestor_id in self.inheritance_graph.ancestors_of(cid) {
                if let Some(anc) = self.contract(ancestor_id) {
                    for f in &anc.functions {
                        collect_fn(f);
                    }
                }
                if let Some(iface) = self.interface(ancestor_id) {
                    for f in &iface.functions {
                        collect_fn(f);
                    }
                }
            }
        }
        for iid in self.lookup_interface(contract) {
            if let Some(iface) = self.interface(iid) {
                for f in &iface.functions {
                    collect_fn(f);
                }
            }
        }
        out.into_iter().collect()
    }

    /// Compute every production-scope externally callable entrypoint whose
    /// forward call closure in `CallGraph` writes state, invokes an interface/CPI
    /// target, or (when `StorageGraph` has zero function-state edges) belongs to
    /// a contract.
    pub fn stateful_external_entrypoints(&self) -> Vec<StatefulEntrypoint> {
        if self.call_graph.contracts.is_empty() {
            return Vec::new();
        }

        let mut reverse_calls: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
        let mut forward_calls: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
        for contract in self.call_graph.contracts.values() {
            for function in &contract.functions {
                for call in &function.calls {
                    reverse_calls
                        .entry(call.to_id)
                        .or_default()
                        .push(call.from_id);
                    forward_calls
                        .entry(call.from_id)
                        .or_default()
                        .push(call.to_id);
                }
            }
        }

        let mut direct_writes: BTreeMap<i32, BTreeSet<String>> = BTreeMap::new();
        for fsv in &self.storage_graph.function_state_variables {
            if fsv.is_write {
                let var_name = self
                    .state_variable(fsv.state_variable_id)
                    .map(|v| v.name.clone())
                    .unwrap_or_else(|| format!("var#{}", fsv.state_variable_id));
                direct_writes
                    .entry(fsv.function_id)
                    .or_default()
                    .insert(var_name);
            }
        }

        let storage_empty = self.storage_graph.function_state_variables.is_empty();
        let mut stateful_fn_ids: BTreeSet<i32> = BTreeSet::new();
        if storage_empty {
            for contract in self.call_graph.contracts.values() {
                for function in &contract.functions {
                    stateful_fn_ids.insert(function.id);
                }
            }
        } else {
            let mut queue: VecDeque<i32> = VecDeque::new();
            for &fn_id in direct_writes.keys() {
                if stateful_fn_ids.insert(fn_id) {
                    queue.push_back(fn_id);
                }
            }
            for contract in self.call_graph.contracts.values() {
                for function in &contract.functions {
                    let calls_external_target = function.calls.iter().any(|call| {
                        self.function_owner
                            .get(&call.to_id)
                            .is_some_and(|o| o.container_kind == ContainerKind::Interface)
                            || call
                                .description
                                .as_deref()
                                .is_some_and(|d| d.to_ascii_lowercase().contains("cpi"))
                    });
                    if calls_external_target && stateful_fn_ids.insert(function.id) {
                        queue.push_back(function.id);
                    }
                }
            }
            while let Some(callee_id) = queue.pop_front() {
                if let Some(callers) = reverse_calls.get(&callee_id) {
                    for &caller_id in callers {
                        if stateful_fn_ids.insert(caller_id) {
                            queue.push_back(caller_id);
                        }
                    }
                }
            }
        }

        let mut entrypoints = Vec::new();
        for contract in self.call_graph.contracts.values() {
            if Self::is_non_production_path(&contract.relative_file_path) {
                continue;
            }
            let is_program_module = CallableReachability::is_program_module_description(
                contract.description.as_deref(),
            );
            for function in &contract.functions {
                if Self::is_non_production_path(&function.relative_file_path) {
                    continue;
                }
                if !stateful_fn_ids.contains(&function.id) {
                    continue;
                }
                if CallableReachability::classify_function(
                    function,
                    ContainerKind::Contract,
                    is_program_module,
                ) != CallableReachability::ExternalMutating
                {
                    continue;
                }
                let written_state_vars =
                    Self::collect_reachable_writes(function.id, &forward_calls, &direct_writes);
                entrypoints.push(StatefulEntrypoint {
                    contract_id: contract.id,
                    contract_name: contract.name.clone(),
                    relative_file_path: contract.relative_file_path.clone(),
                    function_id: function.id,
                    function_name: function.name.clone(),
                    signature: self.extract_clean_signature(function),
                    written_state_vars,
                });
            }
        }
        entrypoints
    }

    /// Return all stateful external entrypoints not covered (case-insensitively,
    /// inheritance-aware across ancestors and descendants) by `semantics`.
    pub fn uncovered_stateful_entrypoints(
        &self,
        semantics: &[ExtractedSemantic],
    ) -> Vec<StatefulEntrypoint> {
        let mut covered_pairs: BTreeSet<(String, String)> = BTreeSet::new();
        let mut covered_unqualified: BTreeSet<String> = BTreeSet::new();

        for sem in semantics {
            for func in &sem.functions {
                let raw_name = func.name.trim();
                let bare_name = raw_name.split('(').next().unwrap_or(raw_name).trim();
                if bare_name.is_empty() {
                    continue;
                }
                let contract = func.contract.trim();
                if !contract.is_empty() {
                    covered_pairs.insert((contract.to_lowercase(), bare_name.to_lowercase()));
                } else if let Some((qual_c, qual_f)) = bare_name
                    .rsplit_once("::")
                    .or_else(|| bare_name.split_once('.'))
                {
                    covered_pairs
                        .insert((qual_c.trim().to_lowercase(), qual_f.trim().to_lowercase()));
                } else {
                    covered_unqualified.insert(bare_name.to_lowercase());
                }
            }
        }

        self.stateful_external_entrypoints()
            .into_iter()
            .filter(|ep| {
                let fn_lower = ep.function_name.to_lowercase();
                if covered_unqualified.contains(&fn_lower)
                    || covered_pairs.contains(&(ep.contract_name.to_lowercase(), fn_lower.clone()))
                {
                    return false;
                }
                let related_ids = self
                    .inheritance_graph
                    .ancestors_of(ep.contract_id)
                    .into_iter()
                    .chain(self.inheritance_graph.descendants_of(ep.contract_id));
                for rel_id in related_ids {
                    if let Some(rel_contract) = self.contract(rel_id)
                        && covered_pairs
                            .contains(&(rel_contract.name.to_lowercase(), fn_lower.clone()))
                    {
                        return false;
                    }
                }
                true
            })
            .collect()
    }

    fn collect_reachable_writes(
        start_fn_id: i32,
        forward_calls: &BTreeMap<i32, Vec<i32>>,
        direct_writes: &BTreeMap<i32, BTreeSet<String>>,
    ) -> Vec<String> {
        let mut visited: BTreeSet<i32> = BTreeSet::new();
        let mut queue: VecDeque<i32> = VecDeque::new();
        let mut writes: BTreeSet<String> = BTreeSet::new();
        visited.insert(start_fn_id);
        queue.push_back(start_fn_id);
        while let Some(fn_id) = queue.pop_front() {
            if let Some(vars) = direct_writes.get(&fn_id) {
                writes.extend(vars.iter().cloned());
            }
            if let Some(callees) = forward_calls.get(&fn_id) {
                for &callee_id in callees {
                    if visited.insert(callee_id) {
                        queue.push_back(callee_id);
                    }
                }
            }
        }
        writes.into_iter().collect()
    }

    fn is_non_production_path(path: &Path) -> bool {
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if file_name.ends_with(".t.sol") || file_name.ends_with(".s.sol") {
            return true;
        }
        path.components().any(|comp| {
            let Some(seg) = comp.as_os_str().to_str() else {
                return false;
            };
            matches!(
                seg.to_ascii_lowercase().as_str(),
                "test"
                    | "tests"
                    | "mock"
                    | "mocks"
                    | "script"
                    | "scripts"
                    | "lib"
                    | "node_modules"
                    | "benches"
                    | "examples"
            )
        })
    }

    fn extract_clean_signature(&self, function: &Function) -> String {
        if let Some(source) = function.content.as_deref() {
            let extracted = if function
                .relative_file_path
                .extension()
                .and_then(|e| e.to_str())
                == Some("rs")
            {
                knowdit_rust::extract_function_signature(source)
            } else {
                knowdit_sol::signature::extract_function_signature(source)
            };
            if let Ok(Some(sig)) = extracted {
                return sig;
            }
        }
        format!("{}({})", function.name, function.args)
    }

    pub(crate) fn render_interface_lookup(
        &self,
        interface: &Interface,
        focus_function: Option<&str>,
    ) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "## interface `{}` @ `{}` (id={})\n",
            interface.name,
            interface.relative_file_path.display(),
            interface.id
        ));
        let focus = focus_function.map(str::to_lowercase);
        for f in &interface.functions {
            if focus
                .as_deref()
                .map(|n| f.name.to_lowercase() == n)
                .unwrap_or(true)
            {
                out.push_str(&format!(
                    "- {}({}) @ {}:{} (id={})\n",
                    f.name,
                    f.args,
                    f.relative_file_path.display(),
                    f.loc.start_line,
                    f.id
                ));
            }
        }
        out
    }

    pub(crate) fn render_contract_lookup(
        &self,
        contract: &Contract,
        focus_function: Option<&str>,
    ) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "## contract `{}` @ `{}` (id={})\n",
            contract.name,
            contract.relative_file_path.display(),
            contract.id
        ));
        let focus = focus_function.map(str::to_lowercase);
        let matched: Vec<&Function> = contract
            .functions
            .iter()
            .filter(|f| {
                focus
                    .as_deref()
                    .map(|name| f.name.to_lowercase() == name)
                    .unwrap_or(true)
            })
            .collect();

        if focus.is_some() && matched.is_empty() {
            out.push_str(&format!(
                "no function matching `{}` in this contract; available functions:\n",
                focus_function.unwrap_or("")
            ));
            for f in &contract.functions {
                out.push_str(&format!(
                    "- {}({}) @ {}:{} (id={})\n",
                    f.name,
                    f.args,
                    f.relative_file_path.display(),
                    f.loc.start_line,
                    f.id
                ));
            }
            return out;
        }

        if focus.is_none() {
            out.push_str("Functions:\n");
            for f in &contract.functions {
                out.push_str(&format!(
                    "- {}({}) @ {}:{} (id={})\n",
                    f.name,
                    f.args,
                    f.relative_file_path.display(),
                    f.loc.start_line,
                    f.id
                ));
            }
            return out;
        }

        for function in matched {
            out.push_str(&format!(
                "\n### function `{}({})` @ {}:{} (id={})\n",
                function.name,
                function.args,
                function.relative_file_path.display(),
                function.loc.start_line,
                function.id
            ));
            if let Some(content) = &function.content {
                let fence = Self::code_fence_for_path(&function.relative_file_path);
                out.push_str(&format!("source:\n```{fence}\n"));
                out.push_str(content.trim_end());
                out.push_str("\n```\n");
            }
            out.push_str("\noutgoing calls:\n");
            if function.calls.is_empty() {
                out.push_str("- (none)\n");
            } else {
                for call in &function.calls {
                    out.push_str(&format!(
                        "- -> {}\n",
                        self.describe_endpoint(call.to_id, call)
                    ));
                }
            }
            out.push_str("\nincoming calls:\n");
            let mut incoming: Vec<&FunctionCall> = Vec::new();
            for c in self.call_graph.contracts.values() {
                for f in &c.functions {
                    for call in &f.calls {
                        if call.to_id == function.id {
                            incoming.push(call);
                        }
                    }
                }
            }
            for i in self.call_graph.interfaces.values() {
                for f in &i.functions {
                    for call in &f.calls {
                        if call.to_id == function.id {
                            incoming.push(call);
                        }
                    }
                }
            }
            if incoming.is_empty() {
                out.push_str("- (none in static analysis)\n");
            } else {
                for call in incoming {
                    out.push_str(&format!(
                        "- <- {}\n",
                        self.describe_endpoint(call.from_id, call)
                    ));
                }
            }
        }

        out
    }

    pub(crate) fn describe_function_xref(
        &self,
        function_id: i32,
        description: Option<&str>,
    ) -> String {
        let endpoint = match self.function_owner.get(&function_id) {
            Some(owner) => {
                let function_label = match owner.container_kind {
                    ContainerKind::Contract => self
                        .contract(owner.container_id)
                        .and_then(|c| c.functions.iter().find(|f| f.id == function_id))
                        .map(|f| format!("{}({})", f.name, f.args)),
                    ContainerKind::Interface => self
                        .interface(owner.container_id)
                        .and_then(|i| i.functions.iter().find(|f| f.id == function_id))
                        .map(|f| format!("{}({})", f.name, f.args)),
                }
                .unwrap_or_else(|| format!("function#{function_id}"));
                format!(
                    "{} {}::{} @ {}",
                    owner.container_kind.as_str(),
                    owner.container_name,
                    function_label,
                    owner.relative_file_path
                )
            }
            None => format!("(unknown owner) function#{function_id}"),
        };
        match description.map(str::trim).filter(|s| !s.is_empty()) {
            Some(desc) => format!("{endpoint} :: {desc}"),
            None => endpoint,
        }
    }

    pub(crate) fn describe_endpoint(&self, function_id: i32, call: &FunctionCall) -> String {
        let owner = self.function_owner.get(&function_id);
        let function_name = owner
            .and_then(|owner| match owner.container_kind {
                ContainerKind::Contract => self
                    .contract(owner.container_id)
                    .and_then(|c| c.functions.iter().find(|f| f.id == function_id))
                    .map(|f| format!("{}({})", f.name, f.args)),
                ContainerKind::Interface => self
                    .interface(owner.container_id)
                    .and_then(|i| i.functions.iter().find(|f| f.id == function_id))
                    .map(|f| format!("{}({})", f.name, f.args)),
            })
            .unwrap_or_else(|| format!("function#{function_id}"));
        let suffix = call
            .description
            .as_ref()
            .filter(|d| !d.trim().is_empty())
            .map(|d| format!(" :: {}", d.trim()))
            .unwrap_or_default();
        match owner {
            Some(owner) => format!(
                "{} {}::{} @ {}{suffix}",
                owner.container_kind.as_str(),
                owner.container_name,
                function_name,
                owner.relative_file_path,
            ),
            None => format!("(unknown owner) {}{}", function_name, suffix),
        }
    }
}

pub(crate) fn inheritance_closure(rows: &[ContractInherit]) -> BTreeMap<i32, BTreeSet<i32>> {
    let mut direct: BTreeMap<i32, BTreeSet<i32>> = BTreeMap::new();
    for row in rows {
        direct
            .entry(row.contract_id)
            .or_default()
            .insert(row.inherited_id);
    }
    let mut closure: BTreeMap<i32, BTreeSet<i32>> = BTreeMap::new();
    for &start in direct.keys() {
        let mut acc: BTreeSet<i32> = BTreeSet::new();
        let mut frontier: Vec<i32> = direct
            .get(&start)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();
        while let Some(node) = frontier.pop() {
            if !acc.insert(node) {
                continue;
            }
            if let Some(parents) = direct.get(&node) {
                for parent in parents {
                    if !acc.contains(parent) {
                        frontier.push(*parent);
                    }
                }
            }
        }
        closure.insert(start, acc);
    }
    closure
}

impl ProjectIndex {
    /// Render the agent's short-term memory: one entry per contract / interface
    /// in this index, holding state-var + function-signature indexes (bodies are
    /// fetched on demand via tools). Long-term memory is empty here — the link
    /// details live in the system prompt.
    pub(crate) fn build_link_memory(&self) -> Result<AgentMemoryContext> {
        let mut short_term: BTreeMap<String, AgentMemoryContent> = BTreeMap::new();
        for contract in self.call_graph.contracts.values() {
            let title = format!(
                "{}:{}:{}:{}",
                contract.relative_file_path.display(),
                contract.name,
                contract.chunk.loc.start_line,
                contract.chunk.loc.start_column,
            );
            let content = self.render_contract_memory(contract);
            short_term.insert(
                title.clone(),
                AgentMemoryContent {
                    title,
                    related_context: format!(
                        "Index for contract {}: state variables (own + inherited) and function signatures with their outgoing calls. Bodies are NOT inlined — use `read_function_source` or `read_contract_source` to fetch source on demand.",
                        contract.name
                    ),
                    trigger_scenario:
                        "Read this index when you need to find the right function/state variable to inspect. Then call `read_function_source(contract, function)` for that one body — read by function, not whole-file, unless you really need the full file."
                            .to_string(),
                    content,
                    raw_content: None,
                },
            );
        }
        for interface in self.call_graph.interfaces.values() {
            let title = format!(
                "{}:{}:{}:{}",
                interface.relative_file_path.display(),
                interface.name,
                interface.chunk.loc.start_line,
                interface.chunk.loc.start_column,
            );
            let content = self.render_interface_memory(interface);
            short_term.insert(
                title.clone(),
                AgentMemoryContent {
                    title,
                    related_context: format!(
                        "Function declarations of interface {} (signatures only; use `read_contract_source` if you need the raw interface text)",
                        interface.name
                    ),
                    trigger_scenario:
                        "Read this entry when a contract calls an interface and you need its declared signatures"
                            .to_string(),
                    content,
                    raw_content: None,
                },
            );
        }

        Ok(AgentMemoryContext::new_without_search(AgentMemory {
            long_term: BTreeMap::new(),
            short_term,
        }))
    }

    fn render_contract_memory(&self, contract: &Contract) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "# Contract `{}` @ `{}`\n\n",
            contract.name,
            contract.relative_file_path.display()
        ));
        if let Some(desc) = &contract.description
            && !desc.trim().is_empty()
        {
            out.push_str(&format!("Description: {}\n\n", desc.trim()));
        }
        out.push_str("## State variables (own + inherited)\n");
        let visible_ids = self
            .visible_state_vars
            .get(&contract.id)
            .cloned()
            .unwrap_or_default();
        if visible_ids.is_empty() {
            out.push_str("- (none)\n");
        } else {
            for var_id in &visible_ids {
                if let Some(var) = self.state_variable(*var_id) {
                    let owner_id = self.declaring_contract_for_var.get(var_id).copied();
                    let owner = owner_id
                        .and_then(|id| self.contract(id))
                        .map(|c| c.name.as_str())
                        .unwrap_or("?");
                    let inherited = owner_id != Some(contract.id);
                    out.push_str(&format!(
                        "- {} {} ({}{})\n  declaration: `{}` @ {}:{}\n",
                        var.type_name.trim(),
                        var.name,
                        owner,
                        if inherited { ", inherited" } else { "" },
                        var.content.trim().lines().next().unwrap_or("").trim(),
                        var.relative_file_path.display(),
                        var.loc.start_line,
                    ));
                }
            }
        }

        out.push_str(
            "\n## Functions (full signatures + outgoing calls; use `read_function_source` for a single function body or `read_contract_source` for the whole file)\n",
        );
        if contract.functions.is_empty() {
            out.push_str("- (none)\n");
        } else {
            for function in &contract.functions {
                let sig = self.render_function_signature(function);
                out.push_str(&format!(
                    "- {} @ {}:{} (id={})\n",
                    sig,
                    function.relative_file_path.display(),
                    function.loc.start_line,
                    function.id,
                ));
                for call in &function.calls {
                    out.push_str(&format!(
                        "  -> {}\n",
                        self.describe_endpoint(call.to_id, call)
                    ));
                }
            }
        }

        out
    }

    /// Lives on `ProjectIndex` for symmetry with [`Self::render_contract_memory`]
    /// even though an interface index needs no project-wide lookups.
    fn render_interface_memory(&self, interface: &Interface) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "# Interface `{}` @ `{}`\n\n",
            interface.name,
            interface.relative_file_path.display()
        ));
        out.push_str(
            "## Function declarations (full signatures; use `read_contract_source` for the full interface text)\n",
        );
        for function in &interface.functions {
            let sig = self.render_function_signature(function);
            out.push_str(&format!(
                "- {} @ {}:{} (id={})\n",
                sig,
                function.relative_file_path.display(),
                function.loc.start_line,
                function.id,
            ));
        }
        out
    }

    /// Infer the markdown code-fence tag for a source file path.
    pub(crate) fn code_fence_for_path(path: &std::path::Path) -> &'static str {
        match path.extension().and_then(|e| e.to_str()) {
            Some("rs") => "rust",
            Some("move") => "move",
            _ => "solidity",
        }
    }

    /// Render a function's full signature line (including visibility,
    /// mutability/lifetimes, and return types) via `knowdit-rust` for `.rs`
    /// files or `knowdit-sol` for Solidity files. Falls back to `name(args)`
    /// when there is no body content or the source cannot be parsed.
    fn render_function_signature(&self, function: &Function) -> String {
        format!("`{}`", self.extract_clean_signature(function))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use knowdit_repo_model::cg::{FileChunk, FileLocation};
    use std::path::PathBuf;

    #[test]
    fn renders_rust_signatures_and_fences_in_project_index() {
        let mut contracts = BTreeMap::new();
        contracts.insert(
            1,
            Contract {
                id: 1,
                name: "solana_vault".to_string(),
                relative_file_path: PathBuf::from("programs/vault/src/lib.rs"),
                chunk: FileChunk {
                    loc: FileLocation {
                        start_line: 1,
                        start_column: 0,
                        end_line: 10,
                        end_column: 0,
                    },
                    content: "pub mod solana_vault {}".to_string(),
                },
                functions: vec![Function {
                    id: 1,
                    name: "withdraw".to_string(),
                    args: "ctx: Context<Withdraw>, amount: u64".to_string(),
                    relative_file_path: PathBuf::from("programs/vault/src/lib.rs"),
                    loc: FileLocation {
                        start_line: 2,
                        start_column: 0,
                        end_line: 5,
                        end_column: 0,
                    },
                    content: Some(
                        "pub fn withdraw(ctx: Context<Withdraw>, amount: u64) -> Result<()> {\n    Ok(())\n}"
                            .to_string(),
                    ),
                    calls: Vec::new(),
                    description: None,
                }],
                description: None,
            },
        );
        let index = ProjectIndex::build(
            CallGraph {
                contracts,
                interfaces: BTreeMap::new(),
            },
            StorageGraph::default(),
            InheritanceGraph::default(),
        );
        let lookup =
            index.render_contract_lookup(index.contract(1).expect("contract 1"), Some("withdraw"));
        assert!(lookup.contains("```rust\n"));
        let mem = index.render_contract_memory(index.contract(1).expect("contract 1"));
        assert!(
            mem.contains("`pub fn withdraw(ctx: Context<Withdraw>, amount: u64) -> Result<()>`")
        );
    }

    #[test]
    fn knowdit_rust_graphs_plug_directly_into_project_index_grounding() {
        let src = r#"
            use anchor_lang::prelude::*;
            use anchor_spl::token::{self, TokenAccount, Transfer};

            #[program]
            pub mod solana_vault {
                use super::*;
                pub fn withdraw(ctx: Context<Withdraw>, amount: u64) -> Result<()> {
                    let v = &mut ctx.accounts.vault;
                    v.total_assets -= amount;
                    token::transfer(
                        CpiContext::new(
                            ctx.accounts.token_program.to_account_info(),
                            Transfer {
                                from: ctx.accounts.vault_token.to_account_info(),
                                to: ctx.accounts.user_token.to_account_info(),
                                authority: v.to_account_info(),
                            },
                        ),
                        amount,
                    )?;
                    Ok(())
                }
            }

            #[account]
            pub struct Vault {
                pub total_assets: u64,
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

        let analysis =
            knowdit_rust::RustStaticAnalyzer::analyze(&[knowdit_rust::RustSourceInput {
                relative_path: PathBuf::from("programs/vault/src/lib.rs"),
                content: src.to_string(),
            }])
            .expect("analyze anchor vault");

        let index = ProjectIndex::build(
            analysis.call_graph.call_graph,
            analysis.storage.storage,
            analysis.storage.inheritance,
        );

        assert!(index.has_contract("solana_vault"));
        assert!(index.has_contract("Vault"));
        assert!(index.contract_has_function("solana_vault", "withdraw"));
        assert!(index.has_state_variable("Vault.total_assets"));
        assert!(
            index.has_state_variable("solana_vault.total_assets"),
            "ContractInherit must expose Vault fields on solana_vault"
        );
        assert!(index.has_state_variable("solana_vault.vault_token.amount"));
        assert!(index.has_state_variable("vault_token.amount"));
        assert!(!index.has_state_variable("UnknownModule.total_assets"));
    }

    #[test]
    fn callable_reachability_classifies_solidity_rust_inheritance_and_closures() {
        use knowdit_kg_model::{ExtractedFunction, ExtractedSemantic, category::DeFiCategory};
        use knowdit_repo_model::storage::FunctionStateVariable;

        let loc = FileLocation {
            start_line: 1,
            start_column: 0,
            end_line: 3,
            end_column: 0,
        };
        let sol_fn =
            |id: i32, name: &str, path: &str, content: Option<&str>, calls: Vec<FunctionCall>| {
                Function {
                    id,
                    name: name.to_string(),
                    args: String::new(),
                    relative_file_path: PathBuf::from(path),
                    loc,
                    content: content.map(str::to_string),
                    calls,
                    description: None,
                }
            };

        // Contract 1: BaseVault with public mutating `withdraw` + internal `_accrue` (writes state var 1).
        // Contract 2: Vault inherits BaseVault, has external `deposit` -> calls `_accrue`,
        //             external view `totalAssets`, internal `preview(uint256 externalAmount)`,
        //             constructor, modifier, receive, and synthetic `noBody` (content: None).
        // Contract 3: MockVault under `src/mocks/MockVault.sol` (must be excluded from U2 entrypoints).
        // Contract 4: Rust module with `pub fn`, `pub(crate) fn`, and private `fn`.
        let mut contracts = BTreeMap::new();
        contracts.insert(
            1,
            Contract {
                id: 1,
                name: "BaseVault".to_string(),
                relative_file_path: PathBuf::from("src/BaseVault.sol"),
                chunk: FileChunk {
                    loc,
                    content: String::new(),
                },
                functions: vec![
                    sol_fn(
                        10,
                        "withdraw",
                        "src/BaseVault.sol",
                        Some("function withdraw(uint256 amt) public { _accrue(); }"),
                        vec![FunctionCall {
                            id: 1,
                            from_id: 10,
                            to_id: 11,
                            description: None,
                        }],
                    ),
                    sol_fn(
                        11,
                        "_accrue",
                        "src/BaseVault.sol",
                        Some("function _accrue() internal { totalAssets += 1; }"),
                        Vec::new(),
                    ),
                ],
                description: None,
            },
        );
        contracts.insert(
            2,
            Contract {
                id: 2,
                name: "Vault".to_string(),
                relative_file_path: PathBuf::from("src/Vault.sol"),
                chunk: FileChunk {
                    loc,
                    content: String::new(),
                },
                functions: vec![
                    sol_fn(
                        20,
                        "deposit",
                        "src/Vault.sol",
                        Some("function deposit(uint256 amt) external onlyRole(\"view\") { _accrue(); }"),
                        vec![FunctionCall {
                            id: 2,
                            from_id: 20,
                            to_id: 11,
                            description: None,
                        }],
                    ),
                    sol_fn(
                        21,
                        "totalAssets",
                        "src/Vault.sol",
                        Some("function totalAssets() external view returns (uint256) { return 0; }"),
                        Vec::new(),
                    ),
                    sol_fn(
                        22,
                        "preview",
                        "src/Vault.sol",
                        Some("function preview(uint256 externalAmount) internal pure returns (uint256 publicOut) { return externalAmount; }"),
                        Vec::new(),
                    ),
                    sol_fn(
                        23,
                        "constructor",
                        "src/Vault.sol",
                        Some("constructor(address admin) {}"),
                        Vec::new(),
                    ),
                    sol_fn(
                        24,
                        "onlyAdmin",
                        "src/Vault.sol",
                        Some("modifier onlyAdmin() { _; }"),
                        Vec::new(),
                    ),
                    sol_fn(
                        25,
                        "receive",
                        "src/Vault.sol",
                        Some("receive() external payable { _accrue(); }"),
                        vec![FunctionCall {
                            id: 3,
                            from_id: 25,
                            to_id: 11,
                            description: None,
                        }],
                    ),
                    sol_fn(26, "noBody", "src/Vault.sol", None, Vec::new()),
                ],
                description: None,
            },
        );
        contracts.insert(
            3,
            Contract {
                id: 3,
                name: "MockVault".to_string(),
                relative_file_path: PathBuf::from("src/mocks/MockVault.sol"),
                chunk: FileChunk {
                    loc,
                    content: String::new(),
                },
                functions: vec![sol_fn(
                    30,
                    "mintMock",
                    "src/mocks/MockVault.sol",
                    Some("function mintMock() external {}"),
                    vec![FunctionCall {
                        id: 4,
                        from_id: 30,
                        to_id: 11,
                        description: None,
                    }],
                )],
                description: None,
            },
        );
        contracts.insert(
            4,
            Contract {
                id: 4,
                name: "rust_prog".to_string(),
                relative_file_path: PathBuf::from("programs/prog/src/lib.rs"),
                chunk: FileChunk {
                    loc,
                    content: String::new(),
                },
                functions: vec![
                    sol_fn(
                        40,
                        "execute",
                        "programs/prog/src/lib.rs",
                        Some("pub async fn execute() {}"),
                        vec![FunctionCall {
                            id: 5,
                            from_id: 40,
                            to_id: 41,
                            description: None,
                        }],
                    ),
                    sol_fn(
                        41,
                        "crate_helper",
                        "programs/prog/src/lib.rs",
                        Some("pub(crate) fn crate_helper() {}"),
                        Vec::new(),
                    ),
                    sol_fn(
                        42,
                        "priv_helper",
                        "programs/prog/src/lib.rs",
                        Some("fn priv_helper() {}"),
                        Vec::new(),
                    ),
                ],
                // NOT a #[program] module: its bare `pub fn`s are internal
                // helpers, not CPI-invocable SBF instructions.
                description: None,
            },
        );
        contracts.insert(
            5,
            Contract {
                id: 5,
                name: "rust_program_mod".to_string(),
                relative_file_path: PathBuf::from("programs/prog/src/entrypoints.rs"),
                chunk: FileChunk {
                    loc,
                    content: String::new(),
                },
                functions: vec![
                    sol_fn(
                        50,
                        "deposit",
                        "programs/prog/src/entrypoints.rs",
                        Some("pub fn deposit(ctx: Context<Deposit>, amt: u64) -> Result<()> { Self::apply(&mut ctx.accounts.vault, amt); Ok(()) }"),
                        vec![FunctionCall {
                            id: 6,
                            from_id: 50,
                            to_id: 51,
                            description: None,
                        }],
                    ),
                    sol_fn(
                        51,
                        "apply",
                        "programs/prog/src/entrypoints.rs",
                        Some("fn apply(vault: &mut u64, amt: u64) {}"),
                        Vec::new(),
                    ),
                ],
                // The analyzer stamps #[program] modules with this exact
                // description; `classify_rust_signature` uses it to gate
                // which `pub fn`s count as CPI-invocable instructions.
                description: Some("Solana #[program] entrypoint module".to_string()),
            },
        );

        let mut state_variables = BTreeMap::new();
        state_variables.insert(
            1,
            StateVariable {
                id: 1,
                name: "assets".to_string(),
                type_name: "uint256".to_string(),
                relative_file_path: PathBuf::from("src/BaseVault.sol"),
                loc,
                content: "uint256 public assets;".to_string(),
            },
        );
        let storage = StorageGraph {
            state_variables,
            contract_variables: vec![ContractVariable {
                contract_id: 1,
                state_variable_id: 1,
                description: None,
            }],
            function_state_variables: vec![
                FunctionStateVariable {
                    function_id: 11,
                    state_variable_id: 1,
                    is_write: true,
                    description: None,
                },
                FunctionStateVariable {
                    function_id: 41,
                    state_variable_id: 1,
                    is_write: true,
                    description: None,
                },
                FunctionStateVariable {
                    function_id: 51,
                    state_variable_id: 1,
                    is_write: true,
                    description: None,
                },
            ],
        };
        let inheritance = InheritanceGraph::new(vec![ContractInherit {
            contract_id: 2,
            inherited_id: 1,
        }]);

        let index = ProjectIndex::build(
            CallGraph {
                contracts,
                interfaces: BTreeMap::new(),
            },
            storage,
            inheritance,
        );

        assert_eq!(
            index.classify_sequence_callable("Vault", "deposit"),
            Some(CallableReachability::ExternalMutating)
        );
        // Inherited public function from BaseVault resolves on Vault.
        assert_eq!(
            index.classify_sequence_callable("Vault", "withdraw"),
            Some(CallableReachability::ExternalMutating)
        );
        assert_eq!(
            index.classify_sequence_callable("Vault", "totalAssets"),
            Some(CallableReachability::ExternalReadOnly)
        );
        // Parameter name `externalAmount` + return `publicOut` must NOT fool internal classifier.
        assert_eq!(
            index.classify_sequence_callable("Vault", "preview"),
            Some(CallableReachability::NonExternal)
        );
        assert_eq!(
            index.classify_sequence_callable("Vault", "_accrue"),
            Some(CallableReachability::NonExternal)
        );
        assert_eq!(
            index.classify_sequence_callable("Vault", "constructor"),
            Some(CallableReachability::NonExternal)
        );
        assert_eq!(
            index.classify_sequence_callable("Vault", "onlyAdmin"),
            Some(CallableReachability::NonExternal)
        );
        assert_eq!(
            index.classify_sequence_callable("Vault", "receive"),
            Some(CallableReachability::ExternalMutating)
        );
        assert_eq!(
            index.classify_sequence_callable("Vault", "noBody"),
            Some(CallableReachability::Unknown)
        );
        // Plain-module `pub fn`s are internal helpers, NOT SBF instructions
        // (Q-06b): they are only CPI-invocable from the #[program] module.
        assert_eq!(
            index.classify_sequence_callable("rust_prog", "execute"),
            Some(CallableReachability::NonExternal)
        );
        assert_eq!(
            index.classify_sequence_callable("rust_prog", "crate_helper"),
            Some(CallableReachability::NonExternal)
        );
        assert_eq!(
            index.classify_sequence_callable("rust_prog", "priv_helper"),
            Some(CallableReachability::NonExternal)
        );
        // Inside the #[program] module the `pub fn` IS an instruction entry.
        assert_eq!(
            index.classify_sequence_callable("rust_program_mod", "deposit"),
            Some(CallableReachability::ExternalMutating)
        );
        assert_eq!(
            index.classify_sequence_callable("rust_program_mod", "apply"),
            Some(CallableReachability::NonExternal)
        );

        // Stateful external entrypoints: BaseVault.withdraw, Vault.deposit,
        // Vault.receive, rust_program_mod.deposit (excludes
        // internal/view/constructor/modifier/Unknown, plain-module Rust
        // helpers, and src/mocks/MockVault.sol).
        let entrypoints = index.stateful_external_entrypoints();
        let names: Vec<(String, String)> = entrypoints
            .iter()
            .map(|e| (e.contract_name.clone(), e.function_name.clone()))
            .collect();
        assert_eq!(
            names,
            vec![
                ("BaseVault".to_string(), "withdraw".to_string()),
                ("Vault".to_string(), "deposit".to_string()),
                ("Vault".to_string(), "receive".to_string()),
                ("rust_program_mod".to_string(), "deposit".to_string()),
            ]
        );
        assert_eq!(
            entrypoints[1].written_state_vars,
            vec!["assets".to_string()]
        );
        assert_eq!(
            entrypoints[3].written_state_vars,
            vec!["assets".to_string()]
        );

        // Inheritance-aware coverage: covering `Vault.withdraw` covers `BaseVault.withdraw` too!
        let semantics = vec![ExtractedSemantic {
            name: "Vault Withdraw".to_string(),
            category: DeFiCategory::Lending,
            definition: "def".to_string(),
            description: "desc".to_string(),
            functions: vec![ExtractedFunction {
                name: "withdraw".to_string(),
                contract: "Vault".to_string(),
                signature: Some("withdraw(uint256)".to_string()),
            }],
        }];
        let uncovered = index.uncovered_stateful_entrypoints(&semantics);
        let uncovered_names: Vec<(String, String)> = uncovered
            .into_iter()
            .map(|e| (e.contract_name, e.function_name))
            .collect();
        assert_eq!(
            uncovered_names,
            vec![
                ("Vault".to_string(), "deposit".to_string()),
                ("Vault".to_string(), "receive".to_string()),
                ("rust_program_mod".to_string(), "deposit".to_string()),
            ]
        );
    }
}
