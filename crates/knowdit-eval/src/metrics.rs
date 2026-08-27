//! Shared metric primitives: confusion counts, F1, and paired
//! bootstrap comparison.

use serde::{Deserialize, Serialize};

/// Binary classification tallies.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Confusion {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
}

impl Confusion {
    pub fn precision(&self) -> f64 {
        let denom = self.tp + self.fp;
        if denom == 0 { 1.0 } else { self.tp as f64 / denom as f64 }
    }

    pub fn recall(&self) -> f64 {
        let denom = self.tp + self.fn_;
        if denom == 0 { 1.0 } else { self.tp as f64 / denom as f64 }
    }

    pub fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
    }

    pub fn merge(&mut self, other: &Confusion) {
        self.tp += other.tp;
        self.fp += other.fp;
        self.fn_ += other.fn_;
    }
}

/// Classification report for one scored quantity.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassificationReport {
    pub confusion: Confusion,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

impl ClassificationReport {
    pub fn from_confusion(confusion: Confusion, support: usize) -> Self {
        Self {
            precision: confusion.precision(),
            recall: confusion.recall(),
            f1: confusion.f1(),
            confusion,
            support,
        }
    }
}

/// Result of one paired bootstrap comparison between baseline and
/// candidate per-document scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedComparison {
    pub metric: String,
    pub baseline_mean: f64,
    pub candidate_mean: f64,
    pub delta_mean: f64,
    /// Lower bound of the delta's 95% percentile bootstrap CI.
    pub ci_low: f64,
    /// Upper bound of the delta's 95% percentile bootstrap CI.
    pub ci_high: f64,
    /// Share of bootstrap resamples where candidate ≥ baseline.
    pub p_candidate_better: f64,
    /// Resample count.
    pub resamples: usize,
}

/// Paired bootstrap over per-document score pairs.
///
/// `base` and `cand` must be aligned by document index. Uses a fixed
/// seed for reproducibility; nondeterministic stages are expected to
/// run multiple repetitions and feed the mean per document instead.
pub fn paired_bootstrap(
    metric: &str,
    base: &[f64],
    cand: &[f64],
    resamples: usize,
    seed: u64,
) -> PairedComparison {
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    assert_eq!(base.len(), cand.len(), "score vectors must be aligned");
    let n = base.len();
    let base_mean = mean(base);
    let cand_mean = mean(cand);
    let delta_mean = cand_mean - base_mean;

    if n == 0 {
        return PairedComparison {
            metric: metric.to_string(),
            baseline_mean: base_mean,
            candidate_mean: cand_mean,
            delta_mean,
            ci_low: 0.0,
            ci_high: 0.0,
            p_candidate_better: 0.5,
            resamples,
        };
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut deltas = Vec::with_capacity(resamples);
    let mut better = 0.0f64;
    for _ in 0..resamples {
        let mut sum_base = 0.0;
        let mut sum_cand = 0.0;
        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            sum_base += base[idx];
            sum_cand += cand[idx];
        }
        let d = (sum_cand - sum_base) / n as f64;
        if d > 0.0 {
            better += 1.0;
        } else if d == 0.0 {
            better += 0.5;
        }
        deltas.push(d);
    }
    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = percentile(&deltas, 0.025);
    let hi = percentile(&deltas, 0.975);

    PairedComparison {
        metric: metric.to_string(),
        baseline_mean: base_mean,
        candidate_mean: cand_mean,
        delta_mean,
        ci_low: lo,
        ci_high: hi,
        p_candidate_better: better / resamples as f64,
        resamples,
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confusion_metrics_are_stable() {
        let c = Confusion { tp: 3, fp: 1, fn_: 2 };
        assert!((c.precision() - 0.75).abs() < 1e-9);
        assert!((c.recall() - 0.6).abs() < 1e-9);
        assert!((c.f1() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn bootstrap_detects_clear_improvement() {
        let base = vec![0.5, 0.5, 0.5, 0.5];
        let cand = vec![0.7, 0.7, 0.7, 0.7];
        let cmp = paired_bootstrap("f1", &base, &cand, 500, 42);
        assert!(cmp.delta_mean > 0.19);
        assert!(cmp.ci_low > 0.0);
        assert!(cmp.p_candidate_better > 0.99);
    }

    #[test]
    fn bootstrap_centers_on_equal_scores() {
        let base = vec![0.5, 0.6];
        let cand = vec![0.5, 0.6];
        let cmp = paired_bootstrap("f1", &base, &cand, 200, 7);
        assert!(cmp.delta_mean.abs() < 1e-9);
        assert!(cmp.p_candidate_better > 0.4 && cmp.p_candidate_better < 0.6);
    }
}
