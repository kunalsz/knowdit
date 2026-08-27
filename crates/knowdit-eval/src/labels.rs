//! Gold-label schemas and validation.
//!
//! Labels are stable across database integer-ID churn: every label
//! references content by fingerprint or by content-derived stable keys,
//! never by row id. The pipeline process never loads gold — only the
//! scorer does.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One reviewer's judgement, with rubric version and confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Judgement {
    pub reviewer: String,
    /// Rubric version the reviewer applied.
    pub rubric_version: String,
    /// Free-form confidence (0.0-1.0).
    pub confidence: f64,
    /// Timestamp (unix seconds).
    pub judged_unix: u64,
}

/// Gold categories for one document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryGold {
    pub document_id: String,
    /// Expected project categories.
    pub categories: Vec<String>,
    pub judgement: Judgement,
}

/// One expected atomic semantic claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticClaim {
    /// Stable label ID (unique within the document).
    pub label_id: String,
    /// Normalized name the extractor should emit (approximately).
    pub name: String,
    pub category: String,
    /// Source evidence span: `file:line` or `report-section`.
    pub evidence: String,
}

/// One expected atomic finding claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingClaim {
    pub label_id: String,
    pub title: String,
    pub severity: String,
    pub category: String,
    pub subcategory: String,
    pub evidence: String,
}

/// Extraction gold for one document: the full set of atomic claims.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionGold {
    pub document_id: String,
    pub semantics: Vec<SemanticClaim>,
    pub findings: Vec<FindingClaim>,
    /// Duplicate clusters over label IDs: entries in one list are the
    /// SAME underlying item and must be deduplicated to one.
    pub duplicate_clusters: Vec<Vec<String>>,
    pub judgement: Judgement,
}

/// One expected in-project link: finding label → semantic labels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkClaim {
    pub finding_label: String,
    pub semantic_labels: Vec<String>,
    pub strength: String,
    pub judgement: Judgement,
}

/// In-project link gold for one document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkGold {
    pub document_id: String,
    pub links: Vec<LinkClaim>,
}

/// One expected historical merge decision for one raw claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeClaim {
    /// Label ID of the raw item (semantic or finding).
    pub raw_label: String,
    /// `new` or `merge`.
    pub action: String,
    /// Expected target fingerprints (matched by content, not DB id).
    pub target_fingerprints: Vec<String>,
    /// Whether this is a hard negative (merge claimed but must be New).
    pub hard_negative: bool,
    pub judgement: Judgement,
}

/// Merge gold for one document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeGold {
    pub document_id: String,
    pub semantics: Vec<MergeClaim>,
    pub findings: Vec<MergeClaim>,
}

/// The complete gold bundle a scorer consumes. Loaded only by the
/// scorer; never by the pipeline process.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GoldBundle {
    pub categories: BTreeMap<String, CategoryGold>,
    pub extraction: BTreeMap<String, ExtractionGold>,
    pub links: BTreeMap<String, LinkGold>,
    pub merges: BTreeMap<String, MergeGold>,
}

impl GoldBundle {
    pub fn validate(&self) -> crate::Result<()> {
        for (doc, gold) in &self.extraction {
            for cluster in &gold.duplicate_clusters {
                if cluster.len() < 2 {
                    return Err(crate::error::EvalError::score(format!(
                        "document {doc}: duplicate cluster with fewer than 2 members"
                    )));
                }
            }
        }
        for (doc, gold) in &self.merges {
            for claim in gold
                .semantics
                .iter()
                .chain(gold.findings.iter())
            {
                if claim.action != "new" && claim.action != "merge" {
                    return Err(crate::error::EvalError::score(format!(
                        "document {doc}: merge action must be 'new' or 'merge', got {}",
                        claim.action
                    )));
                }
                if claim.action == "merge" && claim.target_fingerprints.is_empty() {
                    return Err(crate::error::EvalError::score(format!(
                        "document {doc}: merge claim {} has no targets",
                        claim.raw_label
                    )));
                }
            }
        }
        Ok(())
    }
}
