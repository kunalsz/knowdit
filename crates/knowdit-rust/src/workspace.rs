//! Cargo and Anchor workspace discovery for Rust / Solana repositories.
//!
//! Inspects `Cargo.toml` (`[workspace]` members + per-crate `[package]` /
//! `[lib]` / `[dependencies]`) and `Anchor.toml` (`[programs.*]`) without
//! invoking `cargo metadata`, so even unbuilt or offline audit checkouts
//! resolve their crate boundaries, library names, SBF `.so` artifact names,
//! and declared on-chain program IDs deterministically.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, WrapErr, ensure};
use serde::{Deserialize, Serialize};

/// Non-production directory names skipped when enumerating workspace member
/// globs (mirrors `RUST_VENDORED_EXCLUDES` in `knowdit-project`).
const EXCLUDED_MEMBER_DIRS: &[&str] = &[
    ".git",
    "target",
    ".anchor",
    "test-ledger",
    "node_modules",
    "test",
    "tests",
    "benches",
    "examples",
    "fuzz",
    "mock",
    "mocks",
    "script",
    "scripts",
    "deploy",
    "migrations",
    "client",
    "cli",
    "sdk",
    "ts-sdk",
    "vendor",
];

/// Metadata for one discovered Rust crate inside the audited repository.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RustProgramManifest {
    /// Cargo `[package].name` (e.g. `"solana-vault"`).
    pub crate_name: String,
    /// Normalized Rust library target name (e.g. `"solana_vault"`).
    pub lib_name: String,
    /// Path to the crate's `Cargo.toml`, relative to `repo_root`.
    pub manifest_path: PathBuf,
    /// Path to the crate's `src/` directory (or crate root when flat),
    /// relative to `repo_root`.
    pub src_dir: PathBuf,
    /// On-chain program ID recovered from `Anchor.toml` or `declare_id!(...)`.
    pub declared_program_id: Option<String>,
    /// Expected SBF shared-object filename (`<lib_name>.so`).
    pub sbf_so_name: String,
    /// True when the crate compiles as a Solana on-chain program (`cdylib`,
    /// `Anchor.toml` entry, `declare_id!`, `#[program]`, or `entrypoint!`).
    pub is_program: bool,
}

/// Discovered Cargo / Anchor workspace layout rooted at `repo_root`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustWorkspace {
    pub repo_root: PathBuf,
    pub has_anchor_toml: bool,
    pub crates: Vec<RustProgramManifest>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct CargoToml {
    package: Option<CargoPackage>,
    lib: Option<CargoLib>,
    workspace: Option<CargoWorkspaceSection>,
    dependencies: BTreeMap<String, toml::Value>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct CargoPackage {
    name: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct CargoLib {
    name: Option<String>,
    path: Option<String>,
    #[serde(rename = "crate-type")]
    crate_type: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct CargoWorkspaceSection {
    members: Vec<String>,
    exclude: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct AnchorToml {
    programs: BTreeMap<String, BTreeMap<String, toml::Value>>,
}

impl AnchorToml {
    fn load(repo_root: &Path) -> (bool, BTreeMap<String, String>) {
        let path = repo_root.join("Anchor.toml");
        if !path.is_file() {
            return (false, BTreeMap::new());
        }
        let Ok(text) = fs::read_to_string(&path) else {
            return (true, BTreeMap::new());
        };
        let parsed: Self = match toml::from_str(&text) {
            Ok(v) => v,
            Err(err) => {
                tracing::warn!("Anchor.toml at {} is invalid TOML: {err}", path.display());
                return (true, BTreeMap::new());
            }
        };
        // Prefer localnet -> mainnet -> devnet -> any other cluster table.
        let mut program_ids: BTreeMap<String, String> = BTreeMap::new();
        for cluster in ["localnet", "mainnet", "devnet"]
            .into_iter()
            .chain(parsed.programs.keys().map(String::as_str))
        {
            let Some(entries) = parsed.programs.get(cluster) else {
                continue;
            };
            for (raw_name, val) in entries {
                let norm = raw_name.replace('-', "_");
                if program_ids.contains_key(&norm) {
                    continue;
                }
                let id_opt = match val {
                    toml::Value::String(s) => Some(s.clone()),
                    toml::Value::Table(t) => t
                        .get("address")
                        .and_then(toml::Value::as_str)
                        .map(str::to_string),
                    _ => None,
                };
                if let Some(addr) = id_opt.filter(|s| !s.trim().is_empty()) {
                    program_ids.insert(norm, addr);
                }
            }
        }
        (true, program_ids)
    }
}

impl RustWorkspace {
    /// Discover all in-scope Rust crates under `repo_root`. Tolerates repos
    /// without a root `Cargo.toml` (e.g. a contest directory wrapping nested
    /// `<org>-<repo>/` checkouts) by scanning recursively for `Cargo.toml`
    /// manifests outside excluded directories.
    pub fn discover(repo_root: &Path) -> Result<Self> {
        let repo_root = repo_root.canonicalize().wrap_err_with(|| {
            format!(
                "failed to canonicalize Rust repo root {}",
                repo_root.display()
            )
        })?;
        ensure!(
            repo_root.is_dir(),
            "repo root {} is not a directory",
            repo_root.display()
        );

        let (has_anchor_toml, mut anchor_program_ids) = AnchorToml::load(&repo_root);
        let manifest_paths = Self::find_crate_manifests(&repo_root, &mut anchor_program_ids)?;

        let mut crates = Vec::new();
        for abs_manifest in manifest_paths {
            if let Some(manifest) =
                Self::load_one_crate(&repo_root, &abs_manifest, &anchor_program_ids)?
            {
                crates.push(manifest);
            }
        }
        crates.sort_by(|lhs, rhs| lhs.manifest_path.cmp(&rhs.manifest_path));

        Ok(Self {
            repo_root,
            has_anchor_toml,
            crates,
        })
    }

    /// Return only the crates identified as on-chain Solana programs.
    pub fn programs(&self) -> Vec<&RustProgramManifest> {
        self.crates.iter().filter(|c| c.is_program).collect()
    }

    /// Resolve which discovered crate owns a repo-relative `.rs` file path
    /// (longest matching `src_dir` prefix wins).
    pub fn owning_crate_for_file(&self, relative_file: &Path) -> Option<&RustProgramManifest> {
        self.crates
            .iter()
            .filter(|c| relative_file.starts_with(&c.src_dir))
            .max_by_key(|c| c.src_dir.components().count())
    }

    fn find_crate_manifests(
        repo_root: &Path,
        anchor_program_ids: &mut BTreeMap<String, String>,
    ) -> Result<BTreeSet<PathBuf>> {
        let mut manifests = BTreeSet::new();
        let root_cargo = repo_root.join("Cargo.toml");
        if root_cargo.is_file()
            && let Ok(text) = fs::read_to_string(&root_cargo)
            && let Ok(parsed) = toml::from_str::<CargoToml>(&text)
        {
            if parsed.package.is_some() {
                manifests.insert(root_cargo.clone());
            }
            if let Some(ws) = parsed.workspace {
                let excluded: BTreeSet<PathBuf> = ws
                    .exclude
                    .iter()
                    .map(|rel| repo_root.join(rel))
                    .collect();
                for member in &ws.members {
                    let pattern = repo_root.join(member).join("Cargo.toml");
                    let pattern_str = pattern.to_string_lossy().into_owned();
                    if let Ok(paths) = glob::glob(&pattern_str) {
                        for entry in paths.flatten() {
                            if let Some(crate_dir) = entry.parent()
                                && !excluded.contains(crate_dir)
                                && !Self::has_excluded_component(repo_root, crate_dir)
                            {
                                manifests.insert(entry);
                            }
                        }
                    }
                }
            }
        }

        // Fallback recursive scan when root Cargo.toml is absent or yielded
        // nothing (e.g. C4 multi-repo `contracts/<id>/<org>-<repo>/` layout).
        if manifests.is_empty() {
            Self::walk_manifests_recursive(repo_root, repo_root, &mut manifests, anchor_program_ids)?;
        }
        Ok(manifests)
    }

    fn walk_manifests_recursive(
        repo_root: &Path,
        dir: &Path,
        manifests: &mut BTreeSet<PathBuf>,
        anchor_program_ids: &mut BTreeMap<String, String>,
    ) -> Result<()> {
        let Ok(entries) = fs::read_dir(dir) else {
            return Ok(());
        };
        if dir != repo_root {
            let (_, nested_anchor) = AnchorToml::load(dir);
            for (k, v) in nested_anchor {
                anchor_program_ids.entry(k).or_insert(v);
            }
        }
        let candidate_cargo = dir.join("Cargo.toml");
        if candidate_cargo.is_file() {
            manifests.insert(candidate_cargo);
        }
        for entry in entries.flatten() {
            let Ok(ft) = entry.file_type() else {
                continue;
            };
            if ft.is_symlink() || !ft.is_dir() {
                continue;
            }
            let path = entry.path();
            if Self::has_excluded_component(repo_root, &path) {
                continue;
            }
            Self::walk_manifests_recursive(repo_root, &path, manifests, anchor_program_ids)?;
        }
        Ok(())
    }

    fn has_excluded_component(repo_root: &Path, candidate: &Path) -> bool {
        let rel = candidate.strip_prefix(repo_root).unwrap_or(candidate);
        rel.components().any(|comp| {
            comp.as_os_str()
                .to_str()
                .map(|name| EXCLUDED_MEMBER_DIRS.contains(&name))
                .unwrap_or(false)
        })
    }

    fn load_one_crate(
        repo_root: &Path,
        abs_manifest: &Path,
        anchor_program_ids: &BTreeMap<String, String>,
    ) -> Result<Option<RustProgramManifest>> {
        let text = match fs::read_to_string(abs_manifest) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };
        let parsed: CargoToml = match toml::from_str(&text) {
            Ok(p) => p,
            Err(err) => {
                tracing::warn!(
                    "Cargo.toml at {} is not valid TOML: {err}",
                    abs_manifest.display()
                );
                return Ok(None);
            }
        };
        let Some(pkg) = parsed.package.filter(|p| !p.name.trim().is_empty()) else {
            return Ok(None);
        };
        let crate_dir = abs_manifest.parent().unwrap_or(repo_root);
        if Self::has_excluded_component(repo_root, crate_dir) {
            return Ok(None);
        }

        let crate_name = pkg.name.trim().to_string();
        let lib_name = parsed
            .lib
            .as_ref()
            .and_then(|l| l.name.as_deref())
            .map(|s| s.replace('-', "_"))
            .unwrap_or_else(|| crate_name.replace('-', "_"));

        let src_abs = if let Some(custom_path) = parsed.lib.as_ref().and_then(|l| l.path.as_deref())
        {
            crate_dir
                .join(custom_path)
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| crate_dir.join("src"))
        } else if crate_dir.join("src").is_dir() {
            crate_dir.join("src")
        } else {
            crate_dir.to_path_buf()
        };

        let (scanned_id, has_entrypoint_marker) = Self::scan_program_markers(&src_abs);
        let declared_program_id = anchor_program_ids
            .get(&lib_name)
            .cloned()
            .or(scanned_id);

        let has_cdylib = parsed
            .lib
            .as_ref()
            .map(|l| l.crate_type.iter().any(|t| t == "cdylib"))
            .unwrap_or(false);
        let has_solana_dep = parsed.dependencies.keys().any(|dep| {
            matches!(
                dep.as_str(),
                "anchor-lang" | "solana-program" | "pinocchio" | "solana-nostd-entrypoint"
            )
        });
        let is_program = has_cdylib
            || declared_program_id.is_some()
            || has_entrypoint_marker
            || (has_solana_dep && crate_dir.starts_with(repo_root.join("programs")));

        let manifest_path = abs_manifest
            .strip_prefix(repo_root)
            .unwrap_or(abs_manifest)
            .to_path_buf();
        let src_dir = src_abs
            .strip_prefix(repo_root)
            .unwrap_or(&src_abs)
            .to_path_buf();

        Ok(Some(RustProgramManifest {
            crate_name,
            sbf_so_name: format!("{lib_name}.so"),
            lib_name,
            manifest_path,
            src_dir,
            declared_program_id,
            is_program,
        }))
    }

    /// Scan `src/` `.rs` files for `declare_id!("...")` and Solana entrypoint
    /// markers (`#[program]`, `entrypoint!`, `program_entrypoint!`).
    fn scan_program_markers(src_dir: &Path) -> (Option<String>, bool) {
        let mut found_id: Option<String> = None;
        let mut has_entry = false;
        let mut stack = vec![src_dir.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                    continue;
                }
                let Ok(content) = fs::read_to_string(&path) else {
                    continue;
                };
                if content.contains("#[program]")
                    || content.contains("#[anchor_lang::program]")
                    || content.contains("entrypoint!(")
                    || content.contains("program_entrypoint!(")
                {
                    has_entry = true;
                }
                if found_id.is_none()
                    && let Some(id) = Self::extract_declare_id(&content)
                {
                    found_id = Some(id);
                }
            }
        }
        (found_id, has_entry)
    }

    fn extract_declare_id(source: &str) -> Option<String> {
        let needle = "declare_id!(";
        let pos = source.find(needle)?;
        let after = &source[pos + needle.len()..];
        let quote_start = after.find('"')?;
        let rest = &after[quote_start + 1..];
        let quote_end = rest.find('"')?;
        let candidate = rest[..quote_end].trim();
        if (32..=48).contains(&candidate.len())
            && candidate.chars().all(|c| c.is_ascii_alphanumeric())
        {
            Some(candidate.to_string())
        } else {
            None
        }
    }
}
