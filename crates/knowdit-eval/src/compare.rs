//! Baseline-vs-candidate comparison, regression detection, and
//! promotion gates.

use crate::error::Result;
use crate::manifest::Policy;
use crate::metrics::PairedComparison;
use serde::{Deserialize, Serialize};

/// One named scalar per document, used as input to paired comparison.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunScores {
    /// Document ID → extraction F1 (macro over that document).
    pub extraction_f1: Vec<(String, f64)>,
    /// Document ID → merge F1.
    pub merge_f1: Vec<(String, f64)>,
    /// Document ID → link F1.
    pub link_f1: Vec<(String, f64)>,
}

/// Full comparison report between one baseline and one candidate run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub metrics: Vec<PairedComparison>,
    pub gate: GateDecision,
}

/// One gate rule outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateRuleOutcome {
    pub rule: String,
    pub passed: bool,
    pub observed: f64,
    pub limit: f64,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateDecision {
    pub passed: bool,
    pub rules: Vec<GateRuleOutcome>,
}

impl GateDecision {
    pub fn any_failed(&self) -> bool {
        !self.passed
    }
}

/// Compare aligned baseline and candidate scores with paired
/// bootstrap, then evaluate the promotion policy. `seed` fixes the
/// bootstrap for reproducibility.
pub fn compare_runs(
    baseline: &RunScores,
    candidate: &RunScores,
    policy: &Policy,
    resamples: usize,
    seed: u64,
) -> Result<ComparisonReport> {
    let mut metrics = Vec::new();

    let pair = |name: &str,
                base: &[(String, f64)],
                cand: &[(String, f64)],
                metrics: &mut Vec<PairedComparison>| {
        // Align by document ID; documents present in only one run are
        // dropped from the paired comparison (reported separately by
        // the caller via manifest status).
        let base_map: std::collections::BTreeMap<&str, f64> =
            base.iter().map(|(id, v)| (id.as_str(), *v)).collect();
        let cand_map: std::collections::BTreeMap<&str, f64> =
            cand.iter().map(|(id, v)| (id.as_str(), *v)).collect();
        let ids: Vec<&str> = base_map
            .keys()
            .filter(|id| cand_map.contains_key(**id))
            .copied()
            .collect();
        let base_vec: Vec<f64> = ids.iter().map(|id| base_map[*id]).collect();
        let cand_vec: Vec<f64> = ids.iter().map(|id| cand_map[*id]).collect();
        metrics.push(crate::metrics::paired_bootstrap(
            name,
            &base_vec,
            &cand_vec,
            resamples,
            seed,
        ));
    };

    pair(
        "extraction_f1",
        &baseline.extraction_f1,
        &candidate.extraction_f1,
        &mut metrics,
    );
    pair("merge_f1", &baseline.merge_f1, &candidate.merge_f1, &mut metrics);
    pair("link_f1", &baseline.link_f1, &candidate.link_f1, &mut metrics);

    let mut rules = Vec::new();
    let extraction = find_metric(&metrics, "extraction_f1")?;
    let merge = find_metric(&metrics, "merge_f1")?;
    let link = find_metric(&metrics, "link_f1")?;
    // Non-inferiority rules: policy limits are percentage points of F1;
    // convert to the 0..1 scale before comparing. A drop is only
    // credible when the whole 95% CI of the delta sits below the
    // negative threshold.
    let max_extraction_drop = policy.max_extraction_f1_drop / 100.0;
    let max_merge_drop = policy.max_merge_f1_drop / 100.0;
    let max_link_drop = policy.max_link_f1_drop / 100.0;
    rules.push(GateRuleOutcome {
        rule: "extraction_f1_non_inferiority".to_string(),
        passed: extraction.ci_low >= -max_extraction_drop,
        observed: extraction.delta_mean,
        limit: -max_extraction_drop,
        detail: format!(
            "delta {:.3} [{:.3}, {:.3}]",
            extraction.delta_mean, extraction.ci_low, extraction.ci_high
        ),
    });
    rules.push(GateRuleOutcome {
        rule: "merge_f1_non_inferiority".to_string(),
        passed: merge.ci_low >= -max_merge_drop,
        observed: merge.delta_mean,
        limit: -max_merge_drop,
        detail: format!(
            "delta {:.3} [{:.3}, {:.3}]",
            merge.delta_mean, merge.ci_low, merge.ci_high
        ),
    });
    rules.push(GateRuleOutcome {
        rule: "link_f1_non_inferiority".to_string(),
        passed: link.ci_low >= -max_link_drop,
        observed: link.delta_mean,
        limit: -max_link_drop,
        detail: format!(
            "delta {:.3} [{:.3}, {:.3}]",
            link.delta_mean, link.ci_low, link.ci_high
        ),
    });

    // Predeclared improvement: at least one metric must improve by its
    // target (CI lower bound above the target). Metrics this stage of
    // the harness cannot score yet (e.g. downstream recall) are
    // reported as unevaluated and never satisfy the rule on their own.
    // A policy with no declared targets passes vacuously.
    let mut improvement_met = policy.improvement_target.is_empty();
    let mut improvement_detail = String::new();
    let mut unevaluated = Vec::new();
    for (metric_name, target) in &policy.improvement_target {
        let cmp = match metric_name.as_str() {
            "extraction_f1" => Some(extraction),
            "merge_f1" => Some(merge),
            "link_f1" => Some(link),
            other => {
                unevaluated.push(other.to_string());
                continue;
            }
        };
        if let Some(cmp) = cmp {
            // Policy targets are percentage points of F1.
            let target_f1 = *target / 100.0;
            if cmp.ci_low >= target_f1 {
                improvement_met = true;
            }
            improvement_detail.push_str(&format!(
                " {metric_name}: {:.3} [{:.3}, {:.3}] vs target {target_f1:.3};",
                cmp.delta_mean, cmp.ci_low, cmp.ci_high
            ));
        }
    }
    if !unevaluated.is_empty() {
        improvement_detail.push_str(&format!(
            " unevaluated targets: {}",
            unevaluated.join(", ")
        ));
    }
    rules.push(GateRuleOutcome {
        rule: "predeclared_improvement".to_string(),
        passed: improvement_met,
        observed: 0.0,
        limit: 0.0,
        detail: improvement_detail,
    });

    let passed = rules.iter().all(|r| r.passed);
    Ok(ComparisonReport {
        metrics,
        gate: GateDecision { passed, rules },
    })
}

/// Look up one metric from a comparison's metric list. The pair
/// closures always push the three stage metrics; absence is an
/// internal inconsistency surfaced as a compare error.
fn find_metric<'a>(
    metrics: &'a [PairedComparison],
    name: &str,
) -> Result<&'a PairedComparison> {
    metrics
        .iter()
        .find(|cmp| cmp.metric == name)
        .ok_or_else(|| crate::error::EvalError::compare(format!("missing {name} metric")))
}

/// Evaluate the gate and return an error when it fails (used by the
/// CLI to exit nonzero).
pub fn gate_error(report: &ComparisonReport) -> Result<()> {
    if report.gate.passed {
        return Ok(());
    }
    let failed: Vec<String> = report
        .gate
        .rules
        .iter()
        .filter(|r| !r.passed)
        .map(|r| format!("{}: {}", r.rule, r.detail))
        .collect();
    Err(crate::error::EvalError::gate(format!(
        "promotion gate failed: {}",
        failed.join("; ")
    )))
}
