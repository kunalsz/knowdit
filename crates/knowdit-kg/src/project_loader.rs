//! Thin compatibility layer wrapping [`knowdit_project::C4PairedProjectData`].
//!
//! The data types and discovery helpers all live in `knowdit-project`
//! now — this module exists only to attach the LLM-driven
//! learn-pipeline methods (`categorize_and_extract`,
//! `merge_and_write`, `is_completed`, ...) that still live on
//! `impl ProjectData` in [`crate::learn`]. Construction goes through
//! `knowdit-project`; the wrapper carries no extra state — runtime
//! knobs (e.g. `extract_chunk_input_budget`) are threaded into the
//! learn pipeline as method parameters instead.

use std::path::Path;

use crate::error::Result;

// Re-export the data types so the historical `knowdit_kg::project_loader::*`
// paths keep resolving without churn at the call sites. New code should
// import these directly from `knowdit_project`.
pub use knowdit_project::{
    AuditMeta, AuditReportMaterial, MovePlatform, MoveProjectDescriptor, MoveVulnerabilityFinding,
    MoveVulnerabilitySnippet, MoveVulnerabilitySnippetFile, ScopedSourceFile as SourceFile,
    SourceLanguage, list_contest_ids, list_move_projects, load_move_audit_reports,
};

/// Legacy `ProjectData` — a thin wrapper around
/// [`knowdit_project::C4PairedProjectData`].
///
/// All LLM-driven methods (categorize / extract / merge / link /
/// is_completed) live in [`crate::learn::impl ProjectData`] and
/// project this wrapper down to the inner fields via the accessor
/// methods. Per-call tunables (chunk-input budget, agent step caps,
/// merge chunking) are passed in as method parameters, not stored
/// on this struct.
#[derive(Debug, Clone)]
pub struct FeedReportSource {
    pub source_namespace: String,
    pub relative_path: String,
    pub stable_source_id: String,
    pub legacy_platform_id: Option<String>,
    pub content_hash: String,
}

#[derive(Debug, Clone)]
pub struct ProjectData {
    /// The project + optional audit-report payload. Owned so this
    /// wrapper can be cloned cheaply and threaded into async tasks.
    pub paired: knowdit_project::C4PairedProjectData,
    /// When true, prompts use narrative/post-mortem templates instead of
    /// Solidity/Move source-code templates. The markdown content is treated
    /// as BOTH source material (for exploit-pattern semantic extraction) and
    /// audit report material (for vulnerability finding extraction).
    pub is_narrative: bool,
    pub feed_source: Option<FeedReportSource>,
}

impl ProjectData {
    /// Parse `name:path` / `name:path:platform_id` and load the
    /// project as Solidity (no language auto-detection).
    pub async fn from_path_spec(spec: &str) -> Result<Self> {
        Ok(Self {
            paired: knowdit_project::C4PairedProjectData::bare(
                knowdit_project::ProjectData::from_path_spec(spec).await?,
            ),
            is_narrative: false,
            feed_source: None,
        })
    }

    /// Parse `name:path` / `name:path:platform_id` and load the
    /// project, auto-detecting Solidity vs Move.
    pub async fn from_source_dir_spec(spec: &str) -> Result<Self> {
        Ok(Self {
            paired: knowdit_project::C4PairedProjectData::bare(
                knowdit_project::ProjectData::from_source_dir_spec(spec).await?,
            ),
            is_narrative: false,
            feed_source: None,
        })
    }

    /// Solidity-only loader over a known directory.
    pub async fn from_dir(name: &str, root_dir: &Path, platform_id: Option<&str>) -> Result<Self> {
        Ok(Self {
            paired: knowdit_project::C4PairedProjectData::bare(
                knowdit_project::ProjectData::from_dir(name, root_dir, platform_id).await?,
            ),
            is_narrative: false,
            feed_source: None,
        })
    }

    /// Load a Code4rena contest: `(audit_meta + source contracts +
    /// markdown report)`. The wrapper preserves the audit material
    /// inside the underlying `C4PairedProjectData.audit` so the
    /// learn-pipeline's prompt builders pick it up automatically.
    pub async fn from_c4(dataset_dir: &Path, contest_id: u32) -> Result<Self> {
        let paired =
            knowdit_project::C4PairedProjectData::from_dataset_dir(dataset_dir, contest_id).await?;
        Ok(Self {
            paired,
            is_narrative: false,
            feed_source: None,
        })
    }

    /// Load a Sherlock contest from a `sherlock-scrape/out` directory.
    /// Returns `Ok(None)` when the contest is not ingestible (no scope
    /// files, or a non-Solidity/Move language) so bulk callers can skip it.
    pub async fn from_sherlock(out_dir: &Path, contest_id: u32) -> Result<Option<Self>> {
        Ok(
            knowdit_project::C4PairedProjectData::from_sherlock(out_dir, contest_id)
                .await?
                .map(|paired| Self {
                    paired,
                    is_narrative: false,
                    feed_source: None,
                }),
        )
    }

    /// Load a Move snapshot, optionally paired with its
    /// vulnerability-snippet audit report.
    pub async fn from_move_snapshot(
        name: &str,
        root_dir: &Path,
        commit_hash: &str,
        audit_report: Option<MoveVulnerabilitySnippet>,
    ) -> Result<Self> {
        let project =
            knowdit_project::ProjectData::from_move_snapshot(name, root_dir, commit_hash).await?;
        let paired = match audit_report {
            Some(snippet) => knowdit_project::C4PairedProjectData::from_move_pair(project, snippet),
            None => knowdit_project::C4PairedProjectData::bare(project),
        };
        Ok(Self {
            paired,
            is_narrative: false,
            feed_source: None,
        })
    }

    /// Adapt a `knowdit_project::ProjectData` view (no audit) into
    /// this legacy shape. Per-call tunables like
    /// `extract_chunk_input_budget` are no longer carried on the
    /// project — they flow as method parameters into the learn
    /// pipeline.
    pub fn from_project_view(view: &knowdit_project::ProjectData) -> Self {
        Self {
            paired: knowdit_project::C4PairedProjectData::bare(view.clone()),
            is_narrative: false,
            feed_source: None,
        }
    }

    /// Load a markdown security report as a narrative project. The
    /// markdown content is loaded as BOTH the source material (for
    /// exploit-pattern semantic extraction) AND the audit report (for
    /// vulnerability finding extraction). The functions field in
    /// extracted semantics uses `_narrative` as a sentinel.
    ///
    /// `name` is the project name (typically the file stem).
    /// `root_dir` is the parent directory containing the `.md` file.
    /// `relative_path` is the path to the `.md` file relative to `root_dir`.
    pub async fn from_narrative_md(
        name: &str,
        root_dir: &Path,
        platform_id: Option<&str>,
        relative_path: &Path,
        feed_source: FeedReportSource,
    ) -> Result<Self> {
        let project = knowdit_project::ProjectData::from_relative_paths(
            name,
            root_dir,
            knowdit_project::SourceLanguage::Solidity,
            platform_id,
            &[relative_path.to_path_buf()],
        )
        .await?;

        // Clone the source content to also serve as the audit report.
        let content = project.source_files()[0].content.clone();
        let paired = knowdit_project::C4PairedProjectData::audit_only(
            project,
            knowdit_project::AuditReportMaterial::Text(content),
        );

        Ok(Self {
            paired,
            is_narrative: true,
            feed_source: Some(feed_source),
        })
    }

    // ----------------------------------------------------------------
    // Accessors — projected down to the inner `paired` fields so the
    // learn-pipeline methods in `crate::learn` keep working.
    // ----------------------------------------------------------------

    pub fn name(&self) -> &str {
        &self.paired.project.name
    }

    pub fn platform_id(&self) -> Option<&str> {
        self.paired.project.platform_id.as_deref()
    }

    pub fn root_dir(&self) -> &Path {
        self.paired.project.repo_root()
    }

    pub fn source_language(&self) -> SourceLanguage {
        self.paired.project.language
    }

    pub fn source_files(&self) -> &[SourceFile] {
        self.paired.project.source_files()
    }

    pub fn audit_report(&self) -> Option<&AuditReportMaterial> {
        self.paired.audit.as_ref()
    }

    /// Display-friendly identifier: `platform_id` when set,
    /// otherwise the project name. Returns an owned `String` to
    /// keep call-site ergonomics stable — earlier versions had this
    /// shape and a number of `format!()` sites still expect it.
    pub fn display_id(&self) -> String {
        self.paired.project.display_id().to_string()
    }
}
