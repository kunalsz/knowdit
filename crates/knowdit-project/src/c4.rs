//! Code4rena-paired project view.
//!
//! Bundles a [`ProjectData`] with an optional audit-report payload.
//! The same shape is reused for all project sources — C4 contests
//! (where `meta` and `audit` are both populated), Move snapshots
//! (where `audit` may be a [`AuditReportMaterial::MoveVulnerabilitySnippet`]
//! and `meta` is `None`), and freshly-cloned audit checkouts
//! (where both are `None`). Keeping a single struct here lets the
//! rest of the pipeline take "project + optional audit" without
//! having to discriminate on dataset origin.

use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, WrapErr, eyre};
use serde::Deserialize;

use crate::{
    data::{ProjectData, SourceLanguage},
    moves::MoveVulnerabilitySnippet,
    scope::{ProjectScope, ProjectScopeContent, ScopedSourceFile},
};

/// `audits/<id>.json` payload for a Code4rena contest. Only the
/// fields downstream callers currently inspect are listed; serde will
/// ignore unknown fields so we don't break when the source schema
/// grows.
#[derive(Debug, Clone, Deserialize)]
pub struct AuditMeta {
    #[serde(rename = "contestId")]
    pub contest_id: u32,
    pub title: String,
    pub slug: Option<String>,
    #[serde(rename = "startTime")]
    pub start_time: Option<String>,
    #[serde(rename = "endTime")]
    pub end_time: Option<String>,
    pub details: Option<String>,
}

/// Minimal view of a Sherlock `metadata/<id>.json` — only the fields
/// [`C4PairedProjectData::from_sherlock`] needs; serde ignores the rest.
#[derive(Debug, Deserialize)]
pub struct SherlockMeta {
    #[serde(default)]
    pub title: String,
    /// Sherlock's packaged repo (`sherlock-audit/<contest>`); nests each
    /// in-scope repo under `<repo_basename>/` and is the fallback source when
    /// the upstream clone was private/missing.
    #[serde(default)]
    pub template_repo_name: Option<String>,
    #[serde(default)]
    pub scope: Vec<SherlockScopeRepo>,
}

#[derive(Debug, Deserialize)]
pub struct SherlockScopeRepo {
    #[serde(default)]
    pub repo: String,
    #[serde(default)]
    pub files: Vec<SherlockScopeFile>,
}

#[derive(Debug, Deserialize)]
pub struct SherlockScopeFile {
    pub name: String,
}

/// Renderable audit-report payload. Plain text for C4; structured
/// Move vulnerability snippets for the Move dataset. Both render to
/// markdown for prompt inclusion.
#[derive(Debug, Clone)]
pub enum AuditReportMaterial {
    Text(String),
    MoveVulnerabilitySnippet(MoveVulnerabilitySnippet),
}

impl AuditReportMaterial {
    /// Render the payload as markdown for prompt construction.
    pub fn render(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::MoveVulnerabilitySnippet(snippet) => snippet.render(),
        }
    }

    /// Load a markdown audit report from `report_path` and wrap it as
    /// [`AuditReportMaterial::Text`]. Returns `None` when the file is
    /// missing, empty after trimming, or unreadable — those cases are
    /// logged as warnings but never an error, mirroring the
    /// "report-is-optional" assumption the C4 ingest tools make.
    pub fn from_optional_text_file(report_path: &Path) -> Option<Self> {
        if !report_path.is_file() {
            tracing::warn!("audit report not found: {}", report_path.display());
            return None;
        }
        match std::fs::read_to_string(report_path) {
            Ok(text) if !text.trim().is_empty() => Some(Self::Text(text)),
            Ok(_) => {
                tracing::warn!("audit report is empty: {}", report_path.display());
                None
            }
            Err(err) => {
                tracing::warn!(
                    "failed to read audit report {}: {err}",
                    report_path.display()
                );
                None
            }
        }
    }
}

/// A project bundle with an optional audit-side payload.
///
/// * For Code4rena contests, both `meta` and `audit` are populated by
///   [`Self::from_dataset_dir`].
/// * For Move snapshots paired with a vulnerability JSON,
///   [`Self::from_move_pair`] sets `audit` to the snippet variant
///   and leaves `meta` as `None`.
/// * For a freshly-cloned audit checkout with no contest metadata,
///   wrap a [`ProjectData`] via [`Self::audit_only`] or
///   [`Self::bare`].
///
/// The struct intentionally keeps both `meta` and `audit` as
/// `Option`s so the legacy `knowdit_kg::project_loader::ProjectData`
/// can wrap one of these directly regardless of dataset origin.
#[derive(Debug, Clone)]
pub struct C4PairedProjectData {
    pub project: ProjectData,
    pub meta: Option<AuditMeta>,
    pub audit: Option<AuditReportMaterial>,
}

impl C4PairedProjectData {
    /// Wrap a bare [`ProjectData`] with no audit material. Used by
    /// adapters that don't have any audit context to attach (e.g.
    /// when loading from a generic `from_dir`).
    pub fn bare(project: ProjectData) -> Self {
        Self {
            project,
            meta: None,
            audit: None,
        }
    }

    /// Wrap a [`ProjectData`] alongside an audit-report payload that
    /// the caller already has in hand. Useful for Move pairings
    /// where the snippet is built from a separately-loaded JSON map.
    pub fn audit_only(project: ProjectData, audit: AuditReportMaterial) -> Self {
        Self {
            project,
            meta: None,
            audit: Some(audit),
        }
    }

    /// Pair a Move-language [`ProjectData`] with its vulnerability
    /// snippet (typically pulled out of
    /// [`crate::load_move_audit_reports`] keyed on commit hash).
    pub fn from_move_pair(project: ProjectData, snippet: MoveVulnerabilitySnippet) -> Self {
        Self::audit_only(
            project,
            AuditReportMaterial::MoveVulnerabilitySnippet(snippet),
        )
    }

    /// Load a `(project, audit_meta, audit_report)` triple from the
    /// `out_train/` (or `out_git/`) layout that the C4 ingest scripts
    /// produce:
    ///
    /// * `audits/<id>.json` — contest metadata. Required.
    /// * `contracts/<id>/` — Solidity / Move / Rust sources. Required.
    /// * `reports/<id>.md` — markdown audit report. Optional; logged
    ///   when missing but not an error.
    pub async fn from_dataset_dir(dataset_dir: &Path, contest_id: u32) -> Result<Self> {
        Self::from_dataset_dir_with_language(dataset_dir, contest_id, None).await
    }

    /// Language-parameterized variant of [`Self::from_dataset_dir`].
    /// When `language_override` is `Some(lang)`, walks `contracts/<id>/`
    /// strictly for `lang`. When `None`, auto-detects across
    /// [`SourceLanguage::ALL`]; if multiple languages appear in the
    /// same contest tree, picks the one with the largest in-scope file
    /// count (tie-breaking `Solidity > Move > Rust` for backwards
    /// determinism).
    pub async fn from_dataset_dir_with_language(
        dataset_dir: &Path,
        contest_id: u32,
        language_override: Option<SourceLanguage>,
    ) -> Result<Self> {
        let audit_path = dataset_dir
            .join("audits")
            .join(format!("{contest_id}.json"));
        if !audit_path.is_file() {
            return Err(eyre!("audit metadata not found: {}", audit_path.display()));
        }
        let audit_text = std::fs::read_to_string(&audit_path)
            .wrap_err_with(|| format!("failed to read {}", audit_path.display()))?;
        let meta: AuditMeta = serde_json::from_str(&audit_text)
            .wrap_err_with(|| format!("failed to parse audit JSON at {}", audit_path.display()))?;

        let contracts_dir = dataset_dir.join("contracts").join(contest_id.to_string());
        if !contracts_dir.is_dir() {
            return Err(eyre!(
                "contracts directory not found: {}",
                contracts_dir.display()
            ));
        }

        let report = AuditReportMaterial::from_optional_text_file(
            &dataset_dir.join("reports").join(format!("{contest_id}.md")),
        );

        let (language, scope) = match language_override {
            Some(lang) => {
                let scope = ProjectData::build_language_scope(&contracts_dir, lang)?;
                if scope.is_empty() {
                    return Err(eyre!(
                        "no {} source files found under {}",
                        lang.display_name(),
                        contracts_dir.display()
                    ));
                }
                (lang, scope)
            }
            None => {
                let mut candidates: Vec<(SourceLanguage, ProjectScope)> = Vec::new();
                for lang in SourceLanguage::ALL {
                    if let Ok(scope) = ProjectData::build_language_scope(&contracts_dir, lang)
                        && !scope.is_empty()
                    {
                        candidates.push((lang, scope));
                    }
                }
                if candidates.is_empty() {
                    return Err(eyre!(
                        "no Solidity, Move, or Rust source files found under {}",
                        contracts_dir.display()
                    ));
                }
                if candidates.len() > 1 {
                    let breakdown = candidates
                        .iter()
                        .map(|(lang, s)| format!("{}={}", lang.display_name(), s.len()))
                        .collect::<Vec<_>>()
                        .join(", ");
                    candidates.sort_by(|(l_lang, l_scope), (r_lang, r_scope)| {
                        r_scope
                            .len()
                            .cmp(&l_scope.len())
                            .then_with(|| l_lang.cmp(r_lang))
                    });
                    tracing::info!(
                        "c4-{contest_id} contains multiple source languages ({breakdown}); \
                         selected {} ({} files)",
                        candidates[0].0.display_name(),
                        candidates[0].1.len()
                    );
                }
                candidates.remove(0)
            }
        };

        let content = scope.read().await?;
        let project = ProjectData::new(
            meta.title.clone(),
            Some(format!("c4-{contest_id}")),
            language,
            content,
        );
        Ok(Self {
            project,
            meta: Some(meta),
            audit: report,
        })
    }

    /// Lower-level constructor for callers that have already
    /// constructed a [`ProjectScope`] (e.g. with scope.txt filters).
    /// `report_path` is read if present; pass `None` to skip.
    pub async fn from_scoped(
        meta: AuditMeta,
        contest_id: u32,
        scope: ProjectScope,
        report_path: Option<&Path>,
    ) -> Result<Self> {
        let content = scope.read().await?;
        let project = ProjectData::new(
            meta.title.clone(),
            Some(format!("c4-{contest_id}")),
            SourceLanguage::Solidity,
            content,
        );
        let report = report_path.and_then(AuditReportMaterial::from_optional_text_file);
        Ok(Self {
            project,
            meta: Some(meta),
            audit: report,
        })
    }

    /// Load one Sherlock contest from a `sherlock-scrape/out` directory
    /// (`metadata/<id>.json` + `source/<id>/…` + optional
    /// `reports/<id>/report.md`). A contest can span several repos; their
    /// exact in-scope files are aggregated into one [`ProjectData`], each
    /// file's `relative_path` namespaced by `<repo_basename>/`.
    ///
    /// The scraper checks out each upstream in-scope repo under `<repo__>/`,
    /// and Sherlock's packaged/template repo which nests each in-scope repo
    /// under `<template__>/<repo_basename>/`. Upstream clones often fail
    /// (private/deleted) — then only the template copy exists — so each file
    /// is resolved against both layouts, first hit wins. Files not on disk
    /// anywhere are dropped, never an error.
    ///
    /// Returns `Ok(None)` when the contest is not ingestible: no `scope`, no
    /// in-scope files on disk, or a language other than Solidity/Move/Rust.
    /// The report is optional.
    pub async fn from_sherlock(out_dir: &Path, contest_id: u32) -> Result<Option<Self>> {
        Self::from_sherlock_with_language(out_dir, contest_id, None).await
    }

    /// Language-parameterized variant of [`Self::from_sherlock`].
    pub async fn from_sherlock_with_language(
        out_dir: &Path,
        contest_id: u32,
        language_override: Option<SourceLanguage>,
    ) -> Result<Option<Self>> {
        let meta_path = out_dir.join("metadata").join(format!("{contest_id}.json"));
        let meta: SherlockMeta = serde_json::from_str(
            &std::fs::read_to_string(&meta_path)
                .wrap_err_with(|| format!("failed to read {}", meta_path.display()))?,
        )
        .wrap_err_with(|| format!("failed to parse {}", meta_path.display()))?;

        if meta.scope.is_empty() {
            return Ok(None);
        }

        let source_root = out_dir.join("source").join(contest_id.to_string());
        let template_root = meta
            .template_repo_name
            .as_deref()
            .map(|template| source_root.join(template.replace('/', "__")));
        let mut files: Vec<ScopedSourceFile> = Vec::new();
        let mut sol_count = 0usize;
        let mut move_count = 0usize;
        let mut rust_count = 0usize;
        for repo in &meta.scope {
            let basename = repo.repo.rsplit('/').next().unwrap_or(repo.repo.as_str());
            let mut roots = vec![source_root.join(repo.repo.replace('/', "__"))];
            if let Some(template) = &template_root {
                roots.push(template.join(basename));
                roots.push(template.clone());
            }
            for file in &repo.files {
                let Some((absolute_path, content)) = roots.iter().find_map(|root| {
                    let path = root.join(&file.name);
                    std::fs::read_to_string(&path).ok().map(|body| (path, body))
                }) else {
                    continue;
                };
                match Path::new(&file.name)
                    .extension()
                    .and_then(|ext| ext.to_str())
                {
                    Some("sol") => sol_count += 1,
                    Some("move") => move_count += 1,
                    Some("rs") => rust_count += 1,
                    _ => {}
                }
                files.push(ScopedSourceFile {
                    absolute_path,
                    relative_path: PathBuf::from(basename).join(&file.name),
                    content,
                });
            }
        }

        let language = match language_override {
            Some(lang) => lang,
            None => {
                let mut counts = [
                    (SourceLanguage::Solidity, sol_count),
                    (SourceLanguage::Move, move_count),
                    (SourceLanguage::Rust, rust_count),
                ]
                .into_iter()
                .filter(|(_, c)| *c > 0)
                .collect::<Vec<_>>();
                if counts.is_empty() {
                    return Ok(None);
                }
                counts.sort_by(|(l_lang, l_count), (r_lang, r_count)| {
                    r_count
                        .cmp(l_count)
                        .then_with(|| l_lang.cmp(r_lang))
                });
                counts[0].0
            }
        };

        let keep_ext = language.extension();
        files.retain(|file| {
            file.relative_path.extension().and_then(|ext| ext.to_str()) == Some(keep_ext)
        });
        if files.is_empty() {
            return Ok(None);
        }
        files.sort_by(|lhs, rhs| lhs.relative_path.cmp(&rhs.relative_path));

        let project = ProjectData::new(
            if meta.title.is_empty() {
                format!("sherlock-{contest_id}")
            } else {
                meta.title.clone()
            },
            Some(format!("sherlock-{contest_id}")),
            language,
            ProjectScopeContent::new(source_root, files),
        );
        let report = AuditReportMaterial::from_optional_text_file(
            &out_dir
                .join("reports")
                .join(contest_id.to_string())
                .join("report.md"),
        );
        Ok(Some(match report {
            Some(report) => Self::audit_only(project, report),
            None => Self::bare(project),
        }))
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
    async fn c4_loads_rust_solana_contest_and_report() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_file(
            tmp.path(),
            "audits/77.json",
            r#"{"contestId": 77, "title": "Solana Perps"}"#,
        );
        write_file(
            tmp.path(),
            "contracts/77/programs/perps/src/lib.rs",
            "pub fn liquidate() {}",
        );
        write_file(
            tmp.path(),
            "contracts/77/tests/perps.rs",
            "// integration test excluded",
        );
        write_file(tmp.path(), "reports/77.md", "# [H-01] Missing signer check");

        let paired = C4PairedProjectData::from_dataset_dir(tmp.path(), 77)
            .await
            .expect("load rust c4 contest");
        assert_eq!(paired.project.language, SourceLanguage::Rust);
        assert_eq!(paired.project.platform_id.as_deref(), Some("c4-77"));
        assert_eq!(paired.project.source_files().len(), 1);
        assert_eq!(
            paired.project.source_files()[0].relative_path,
            PathBuf::from("programs/perps/src/lib.rs")
        );
        assert!(paired.audit.is_some());
    }

    #[tokio::test]
    async fn sherlock_loads_rust_solana_contest() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_file(
            tmp.path(),
            "metadata/42.json",
            r#"{
                "title": "Solana Lending",
                "scope": [{
                    "repo": "org/solana-lending",
                    "files": [{"name": "programs/lending/src/lib.rs"}]
                }]
            }"#,
        );
        write_file(
            tmp.path(),
            "source/42/org__solana-lending/programs/lending/src/lib.rs",
            "pub fn borrow() {}",
        );
        write_file(
            tmp.path(),
            "reports/42/report.md",
            "## [H-01] Unvalidated PDA",
        );

        let paired = C4PairedProjectData::from_sherlock(tmp.path(), 42)
            .await
            .expect("sherlock load result")
            .expect("sherlock contest should be ingestible");
        assert_eq!(paired.project.language, SourceLanguage::Rust);
        assert_eq!(paired.project.source_files().len(), 1);
        assert!(paired.audit.is_some());
    }
}
