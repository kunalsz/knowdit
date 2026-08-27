//! Typed schemas for benchmark suites, run manifests, and run policies.
//!
//! Every struct derives `Serialize`/`Deserialize` (no ad-hoc `json!`
//! construction) and canonicalizes to a stable content hash: the hash
//! covers every field that can change what a run measures, so two runs
//! with equal hashes are guaranteed to have consumed equal inputs and
//! equal configuration.

use crate::sha256_str;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// One benchmark document (a project to learn) with its stable identity
/// and stratification attributes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSpec {
    /// Stable benchmark document ID, unique within a suite version.
    pub id: String,

    /// How the document is loaded.
    pub kind: DocumentKind,

    /// Split: `dev` for development, `gate` for release gating, and
    /// `holdout` for protected release-only evaluation.
    pub split: Split,

    /// Whether this document belongs to the seed graph or to the
    /// replay slice.
    pub role: DocumentRole,

    /// Publication or acquisition date in `YYYY-MM-DD` form. Used for
    /// temporal leakage checks.
    pub date: String,

    /// Optional label hash (see `corpus::gold`). The pipeline process
    /// never loads gold; only the scorer does.
    pub gold_digest: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentKind {
    /// Solidity (or Move) project directory + optional audit report.
    /// Loaded via `ProjectData::from_source_dir_spec`.
    SourceDir {
        /// `name:path` or `name:path:platform_id` spec.
        spec: String,
    },
    /// Code4rena contest (audit metadata + contracts + report).
    C4 {
        dataset_dir: PathBuf,
        contest_id: u32,
    },
    /// Sherlock contest.
    Sherlock {
        out_dir: PathBuf,
        contest_id: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Dev,
    Gate,
    Holdout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DocumentRole {
    /// Contributes to the seed DB snapshot.
    Seed,
    /// Replayed against a sandbox after the seed.
    Replay,
}

/// One immutable benchmark suite version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suite {
    /// Immutable version name, e.g. `kg-v1`.
    pub version: String,

    /// Human description of what this suite measures.
    pub description: String,

    /// Ordered replay slice. Order is part of the suite identity for
    /// growth-replay determinism.
    pub order: Vec<String>,

    /// Every document in the suite, keyed by stable ID.
    pub documents: BTreeMap<String, DocumentSpec>,
}

impl Suite {
    /// Stable content hash over every identity-bearing field. Used to
    /// detect accidental suite mutation.
    pub fn digest(&self) -> crate::Result<String> {
        let serialized = serde_json::to_string(self)?;
        Ok(sha256_str(&serialized))
    }

    /// The replay documents in declared order.
    pub fn replay_documents(&self) -> Vec<&DocumentSpec> {
        self.order
            .iter()
            .filter_map(|id| self.documents.get(id))
            .collect()
    }

    pub fn validate(&self) -> crate::Result<()> {
        let mut seen = std::collections::HashSet::new();
        for id in &self.order {
            if !seen.insert(id.clone()) {
                return Err(crate::error::EvalError::corpus(format!(
                    "suite {} order lists document {id} twice",
                    self.version
                )));
            }
            if !self.documents.contains_key(id) {
                return Err(crate::error::EvalError::corpus(format!(
                    "suite {} order references unknown document {id}",
                    self.version
                )));
            }
        }
        for doc in self.documents.values() {
            if doc.id.is_empty() {
                return Err(crate::error::EvalError::corpus(
                    "suite contains a document with an empty id",
                ));
            }
            if doc.date.is_empty() {
                return Err(crate::error::EvalError::corpus(format!(
                    "document {} has no date",
                    doc.id
                )));
            }
        }
        Ok(())
    }
}

/// One immutable baseline (seed DB) description. The snapshot itself
/// lives next to this manifest as `<baseline>.sql` (optionally gzip
/// compressed as `<baseline>.sql.gz`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineManifest {
    /// Suite version this baseline belongs to.
    pub suite_version: String,

    /// Git commit that produced the seed DB.
    pub git_commit: String,

    /// Snapshot SHA-256 (over the uncompressed SQL bytes).
    pub snapshot_sha256: String,

    /// Per-table row counts of the seed DB.
    pub table_rows: BTreeMap<String, usize>,

    /// Result of `validate_db` at baseline creation (issue count, not
    /// the full report).
    pub validation_issues: usize,

    /// Confirmed transient checkpoints were absent when the snapshot
    /// was taken.
    pub no_checkpoints: bool,
}

/// The release-gate policy for a suite version. Kept in the suite
/// directory so threshold changes are reviewed and visible.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// Max statistically credible drop in macro extraction F1 (pct pts).
    pub max_extraction_f1_drop: f64,
    /// Max statistically credible drop in macro merge F1 (pct pts).
    pub max_merge_f1_drop: f64,
    /// Max statistically credible drop in macro link F1 (pct pts).
    pub max_link_f1_drop: f64,
    /// Max drop in Critical/High downstream recall (pct pts).
    pub max_crit_high_recall_drop: f64,
    /// Max increase in severe overmerge rate (pct pts).
    pub max_overmerge_increase: f64,
    /// Predeclared improvement target, e.g. `{"downstream_recall": 2.0}`.
    pub improvement_target: BTreeMap<String, f64>,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            max_extraction_f1_drop: 1.0,
            max_merge_f1_drop: 1.0,
            max_link_f1_drop: 1.0,
            max_crit_high_recall_drop: 2.0,
            max_overmerge_increase: 0.5,
            // Empty by default: suites declare their own predeclared
            // improvement targets (including downstream metrics) once
            // those scorers are available for that suite.
            improvement_target: BTreeMap::new(),
        }
    }
}

/// How the harness drives a learning run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    /// `fixed-context`: each document runs against a fresh seed copy.
    /// `growth-replay`: documents are admitted in order into one
    /// evolving sandbox.
    pub mode: RunMode,

    /// Number of full repetitions (only meaningful for live models).
    pub repetitions: usize,

    /// Whether to run the cross-project finding link pass after
    /// admitting replay documents.
    pub run_link_pass: bool,

    /// Concurrency for extraction.
    pub concurrency: usize,

    /// Model identifier / version recorded for attribution.
    pub model_id: String,

    /// Prompt SHA-256 per stage. Keyed by stage name
    /// (`categorize`, `extract_semantics`, `extract_findings`,
    /// `in_project_link`, `semantic_merge`, `finding_merge`,
    /// `link`).
    pub prompt_digests: BTreeMap<String, String>,

    /// Git commit of the code under test.
    pub git_commit: String,

    /// Git tree-dirty patch digest (empty when clean).
    pub dirty_patch_digest: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RunMode {
    FixedContext,
    GrowthReplay,
}

/// A complete run manifest written into the run directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub run_id: String,
    pub suite_version: String,
    pub suite_digest: String,
    pub baseline_digest: String,
    pub config: RunConfig,
    /// Environment facts: rust/compiler/os versions captured at run
    /// start.
    pub environment: BTreeMap<String, String>,
    /// Start time (unix seconds) at run start.
    pub started_unix: u64,
    /// Status per document ID: `ok`, `skipped`, or `failed:<msg>`.
    pub document_status: BTreeMap<String, String>,
}

impl RunManifest {
    pub fn digest(&self) -> crate::Result<String> {
        let serialized = serde_json::to_string(self)?;
        Ok(sha256_str(&serialized))
    }
}
