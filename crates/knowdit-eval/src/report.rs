//! Canonical JSON + Markdown report generation from comparison
//! results.

use crate::compare::ComparisonReport;
use crate::error::Result;
use crate::graph::{GraphDelta, GraphMetrics};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Machine-readable report bundle written to `report.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub comparison: Option<ComparisonReport>,
    pub graph_delta: Option<GraphDelta>,
    pub graph_metrics_before: Option<GraphMetrics>,
    pub graph_metrics_after: Option<GraphMetrics>,
}

/// Render a Markdown summary of the report.
pub fn render_markdown(report: &Report) -> String {
    let mut out = String::new();
    out.push_str("# Eval run report\n\n");

    if let Some(comparison) = &report.comparison {
        out.push_str("## Comparison\n\n");
        out.push_str("| metric | baseline | candidate | delta | CI low | CI high |\n");
        out.push_str("|---|---|---|---|---|---|\n");
        for cmp in &comparison.metrics {
            out.push_str(&format!(
                "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |\n",
                cmp.metric, cmp.baseline_mean, cmp.candidate_mean, cmp.delta_mean, cmp.ci_low, cmp.ci_high
            ));
        }
        out.push_str(&format!(
            "\n**Gate: {}**\n\n",
            if comparison.gate.passed { "PASS" } else { "FAIL" }
        ));
        for rule in &comparison.gate.rules {
            out.push_str(&format!(
                "- {}: {}\n",
                rule.rule,
                if rule.passed { "ok" } else { &rule.detail }
            ));
        }
    }

    if let Some(delta) = &report.graph_delta {
        out.push_str("\n## Graph delta\n\n");
        out.push_str(&format!(
            "- semantics: +{} / -{}\n",
            delta.added_semantics.len(),
            delta.removed_semantics.len()
        ));
        out.push_str(&format!(
            "- findings: +{} / -{}\n",
            delta.added_findings.len(),
            delta.removed_findings.len()
        ));
        out.push_str(&format!(
            "- links: +{} / -{}\n",
            delta.added_links.len(),
            delta.removed_links.len()
        ));
        out.push_str(&format!(
            "- semantic merges: +{} / -{}\n",
            delta.added_semantic_merges.len(),
            delta.removed_semantic_merges.len()
        ));
        out.push_str(&format!(
            "- finding merges: +{} / -{}\n",
            delta.added_finding_merges.len(),
            delta.removed_finding_merges.len()
        ));
        out.push_str(&format!(
            "- projects: +{} / -{}\n",
            delta.added_projects.len(),
            delta.removed_projects.len()
        ));
    }

    if let (Some(before), Some(after)) =
        (&report.graph_metrics_before, &report.graph_metrics_after)
    {
        out.push_str("\n## Invariants\n\n");
        out.push_str(&format!(
            "- semantics: {}/{} canonical (before/after)\n",
            before.semantic_canonical, after.semantic_canonical
        ));
        out.push_str(&format!(
            "- findings: {}/{} canonical\n",
            before.finding_canonical, after.finding_canonical
        ));
        out.push_str(&format!(
            "- links: {} → {}\n",
            before.link_total, after.link_total
        ));
        out.push_str(&format!(
            "- partial links: {} → {}\n",
            before.partial_links, after.partial_links
        ));
        out.push_str(&format!(
            "- dangling merge targets: {} → {}\n",
            before.dangling_merge_targets, after.dangling_merge_targets
        ));
    }

    out
}

/// Write the report bundle as `report.json` + `report.md`.
pub fn write(report: &Report, dir: &Path) -> Result<()> {
    std::fs::write(
        dir.join("report.json"),
        serde_json::to_string_pretty(report)?,
    )?;
    std::fs::write(dir.join("report.md"), render_markdown(report))?;
    Ok(())
}
