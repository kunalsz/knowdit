//! Stage scorers: extraction, dedup, merge, and link quality against
//! gold labels.
//!
//! Scorers operate purely on captured artifacts + gold; they never
//! query a live DB.

use crate::labels::{ExtractionGold, LinkGold, MergeGold};
use crate::metrics::{ClassificationReport, Confusion};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Aggregated stage scores for one run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StageScores {
    pub extraction: Option<ClassificationReport>,
    pub merge_semantics: Option<ClassificationReport>,
    pub merge_findings: Option<ClassificationReport>,
    pub link: Option<ClassificationReport>,
    /// Per-document breakdown keyed by document ID.
    pub per_document: BTreeMap<String, DocumentStageScores>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocumentStageScores {
    pub extraction_f1: f64,
    pub merge_f1: f64,
    pub link_f1: f64,
    pub support: usize,
}

/// A captured extracted item, keyed by its normalized identity (used
/// to match against gold claims).
#[derive(Debug, Clone)]
pub struct CapturedSemantic {
    pub name: String,
    pub category: String,
}

#[derive(Debug, Clone)]
pub struct CapturedFinding {
    pub title: String,
    pub severity: String,
    pub category: String,
    pub subcategory: String,
}

/// Normalize a name for soft matching (case + whitespace).
pub fn normalize_name(s: &str) -> String {
    s.trim()
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn match_claims(
    gold_names: &[String],
    predicted_names: &[String],
) -> Confusion {
    let mut gold_matched = vec![false; gold_names.len()];
    let mut used = vec![false; predicted_names.len()];
    let mut tp = 0usize;

    // Greedy one-to-one matching: first exact normalized hits, then
    // anything else unmatched gets counted as fn/fp.
    for (gi, g) in gold_names.iter().enumerate() {
        let g_norm = normalize_name(g);
        for (pi, p) in predicted_names.iter().enumerate() {
            if !used[pi] && normalize_name(p) == g_norm {
                used[pi] = true;
                gold_matched[gi] = true;
                tp += 1;
                break;
            }
        }
    }
    let fn_ = gold_names.len() - tp;
    let fp = predicted_names.len() - tp;
    Confusion { tp, fp, fn_ }
}

/// Score extraction recall/precision for one document: predicted
/// semantics/findings vs gold atomic claims, by normalized identity.
pub fn score_extraction(
    gold: &ExtractionGold,
    predicted_semantics: &[CapturedSemantic],
    predicted_findings: &[CapturedFinding],
) -> ClassificationReport {
    let gold_sem_names: Vec<String> = gold.semantics.iter().map(|s| s.name.clone()).collect();
    let pred_sem_names: Vec<String> = predicted_semantics.iter().map(|s| s.name.clone()).collect();
    let sem = match_claims(&gold_sem_names, &pred_sem_names);

    let gold_find_titles: Vec<String> = gold.findings.iter().map(|f| f.title.clone()).collect();
    let pred_find_titles: Vec<String> = predicted_findings.iter().map(|f| f.title.clone()).collect();
    let find = match_claims(&gold_find_titles, &pred_find_titles);

    let mut confusion = sem;
    confusion.merge(&find);
    ClassificationReport::from_confusion(
        confusion,
        gold.semantics.len() + gold.findings.len(),
    )
}

/// A captured merge decision for one raw item.
#[derive(Debug, Clone)]
pub struct CapturedMergeDecision {
    /// Normalized raw name/title.
    pub raw_name: String,
    /// `new` or `merge`.
    pub action: String,
    /// Target fingerprints the decision merged into.
    pub target_fingerprints: Vec<String>,
}

/// Score merge decisions: New-vs-Merge accuracy plus target-set
/// precision/recall for merged items.
pub fn score_merges(
    gold: &MergeGold,
    predicted: &[CapturedMergeDecision],
) -> ClassificationReport {
    let mut confusion = Confusion::default();
    let mut support = 0usize;

    // Index predicted decisions by normalized raw name; a gold claim
    // may be matched even if the raw name drifted slightly.
    for claim in gold.semantics.iter().chain(gold.findings.iter()) {
        support += 1;
        let Some(dec) = predicted
            .iter()
            .find(|d| normalize_name(&d.raw_name) == normalize_name(&claim.raw_label))
        else {
            // The item was never decided on (missing from capture):
            // count as fn if gold expected a merge, fp otherwise.
            if claim.action == "merge" {
                confusion.fn_ += 1;
            } else {
                confusion.fp += 1;
            }
            continue;
        };

        let gold_merge = claim.action == "merge";
        let pred_merge = dec.action == "merge";
        match (gold_merge, pred_merge) {
            (true, true) => {
                // Correct merge: check target overlap. Any overlap
                // counts as a hit; missing all gold targets counts
                // as a recall miss.
                let hit = claim
                    .target_fingerprints
                    .iter()
                    .any(|t| dec.target_fingerprints.contains(t));
                if hit {
                    confusion.tp += 1;
                } else {
                    confusion.fn_ += 1;
                }
            }
            (false, false) => confusion.tp += 1,
            (true, false) => confusion.fn_ += 1,
            (false, true) => {
                // Overmerge: merging when it should be New.
                confusion.fp += 1;
            }
        }
    }

    ClassificationReport::from_confusion(confusion, support)
}

/// A captured in-project link edge: finding label → semantic labels.
#[derive(Debug, Clone)]
pub struct CapturedLink {
    pub finding_name: String,
    pub semantic_names: Vec<String>,
}

/// Score in-project link edges: each gold (finding → semantic set)
/// edge must appear; extra edges count as false positives.
pub fn score_links(gold: &LinkGold, predicted: &[CapturedLink]) -> ClassificationReport {
    let mut confusion = Confusion::default();
    let mut support = 0usize;

    for claim in &gold.links {
        support += 1;
        let mut found = false;
        for link in predicted {
            if normalize_name(&link.finding_name) != normalize_name(&claim.finding_label) {
                continue;
            }
            let want: Vec<String> = claim
                .semantic_labels
                .iter()
                .map(|s| normalize_name(s))
                .collect();
            let got: Vec<String> = link.semantic_names.iter().map(|s| normalize_name(s)).collect();
            if want.iter().any(|w| got.contains(w)) {
                found = true;
                break;
            }
        }
        if found {
            confusion.tp += 1;
        } else {
            confusion.fn_ += 1;
        }
    }

    // Extra predicted edges beyond gold are false positives.
    let gold_edge_count = gold.links.len();
    let pred_edge_count = predicted.len();
    if pred_edge_count > gold_edge_count {
        confusion.fp += pred_edge_count - gold_edge_count;
    }

    ClassificationReport::from_confusion(confusion, support)
}

/// Aggregate per-document stage reports into one run-level report.
/// Merge reports are split into semantic and finding components by
/// the caller; a `None` slice means that stage had no scored support.
pub fn aggregate(
    extraction: &[ClassificationReport],
    merge_semantics: &[ClassificationReport],
    merge_findings: &[ClassificationReport],
    link: &[ClassificationReport],
) -> StageScores {
    let fold = |reports: &[ClassificationReport]| -> Option<ClassificationReport> {
        if reports.is_empty() {
            return None;
        }
        let mut confusion = Confusion::default();
        let mut support = 0usize;
        for r in reports {
            confusion.merge(&r.confusion);
            support += r.support;
        }
        Some(ClassificationReport::from_confusion(confusion, support))
    };

    StageScores {
        extraction: fold(extraction),
        merge_semantics: fold(merge_semantics),
        merge_findings: fold(merge_findings),
        link: fold(link),
        per_document: BTreeMap::new(),
    }
}
