use clap::{Args, ValueEnum};
use color_eyre::eyre::{Result, ensure, eyre};
use knowdit_kg::db::HistoricalDatabase;
use knowdit_kg::router_eval::{
    RouterEmbeddingCache, RouterKind, RouterReplayOptions, RouterReplayReport,
};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum LinkRouterKindArg {
    Bm25,
    Hybrid,
}

#[derive(Args, Debug, Clone)]
pub struct LinkRouterEvalArgs {
    /// Candidate router to replay. Hybrid requires --embedding-cache.
    #[arg(long, value_enum, default_value_t = LinkRouterKindArg::Bm25)]
    pub router: LinkRouterKindArg,

    /// Maximum semantic candidates retained per finding.
    #[arg(long, default_value_t = 64)]
    pub max_candidates: usize,

    /// Small additive BM25 score for a shared project/semantic category.
    /// Categories never exclude candidates.
    #[arg(long, default_value_t = 0.25)]
    pub category_boost: f64,

    /// Max merged raw variants indexed and costed per canonical semantic.
    #[arg(long, default_value_t = 8)]
    pub variant_render_cap: usize,

    /// Max characters indexed from each merged raw semantic description.
    /// Zero means unbounded.
    #[arg(long, default_value_t = 400)]
    pub raw_child_char_cap: usize,

    /// Candidates reserved from each mechanism shard triggered by a finding.
    #[arg(long, default_value_t = 0)]
    pub mechanism_candidates_per_shard: usize,

    /// Versioned embedding cache produced by `eval build-link-router-embeddings`.
    #[arg(long)]
    pub embedding_cache: Option<PathBuf>,

    /// Required edge recall for globally accepted High links.
    #[arg(long, default_value_t = 0.995)]
    pub min_high_recall: f64,

    /// Required edge recall for globally accepted Medium links.
    #[arg(long, default_value_t = 0.97)]
    pub min_medium_recall: f64,

    /// Permit the cross-category High metric to use --min-high-recall instead
    /// of requiring zero misses.
    #[arg(long, default_value_t = false)]
    pub allow_cross_category_high_misses: bool,

    /// Exit non-zero when any configured recall requirement is missed.
    #[arg(long, default_value_t = false)]
    pub gate: bool,

    /// Maximum missed edges retained in the JSON report.
    #[arg(long, default_value_t = 200)]
    pub max_misses_in_report: usize,

    /// Optional path for the complete machine-readable JSON report.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

impl LinkRouterEvalArgs {
    pub async fn run(self, db: &HistoricalDatabase) -> Result<()> {
        ensure!(
            self.max_candidates > 0,
            "max_candidates must be greater than zero"
        );
        ensure!(
            self.category_boost >= 0.0 && self.category_boost.is_finite(),
            "category_boost must be a finite non-negative number"
        );
        for (name, value) in [
            ("min_high_recall", self.min_high_recall),
            ("min_medium_recall", self.min_medium_recall),
        ] {
            ensure!((0.0..=1.0).contains(&value), "{name} must be in [0, 1]");
        }

        let embedding_cache = self
            .embedding_cache
            .as_ref()
            .map(|path| -> Result<RouterEmbeddingCache> {
                let bytes = std::fs::read(path)?;
                Ok(serde_json::from_slice(&bytes)?)
            })
            .transpose()?;
        if matches!(self.router, LinkRouterKindArg::Hybrid) {
            ensure!(
                embedding_cache.is_some(),
                "--router hybrid requires --embedding-cache"
            );
        }

        let report = db
            .evaluate_link_router(RouterReplayOptions {
                router: match self.router {
                    LinkRouterKindArg::Bm25 => RouterKind::Bm25,
                    LinkRouterKindArg::Hybrid => RouterKind::Hybrid,
                },
                max_candidates: self.max_candidates,
                category_boost: self.category_boost,
                variant_render_cap: self.variant_render_cap,
                raw_child_char_cap: self.raw_child_char_cap,
                mechanism_candidates_per_shard: self.mechanism_candidates_per_shard,
                embedding_cache,
                min_high_recall: self.min_high_recall,
                min_medium_recall: self.min_medium_recall,
                require_cross_category_high_perfect: !self.allow_cross_category_high_misses,
                max_misses_in_report: self.max_misses_in_report,
            })
            .await?;
        print_report(&report);

        if let Some(path) = &self.output {
            let json = serde_json::to_string_pretty(&report)?;
            std::fs::write(path, format!("{json}\n"))?;
            println!("report: {}", path.display());
        }

        if self.gate && !report.gate.passed {
            return Err(eyre!(
                "link-router replay gate failed; inspect the missed-edge report before enabling candidate filtering"
            ));
        }
        Ok(())
    }
}

fn print_report(report: &RouterReplayReport) {
    println!("router: {}", report.router);
    println!(
        "corpus: {} completed findings, {} evaluated findings, {} active canonical semantics",
        report.corpus.completed_findings,
        report.corpus.evaluated_findings,
        report.corpus.active_canonical_semantics,
    );
    print_metric("High", &report.high);
    print_metric("Medium", &report.medium);
    print_metric("Cross-category High", &report.cross_category_high);
    println!(
        "all-High targets recovered per finding: {}/{} ({})",
        report.all_high_recovered_findings.hits,
        report.all_high_recovered_findings.total,
        format_percent(report.all_high_recovered_findings.recall),
    );
    println!(
        "candidate sets: mean {:.1}, p50 {}, p95 {}, max {}; semantic chars reduced ~{:.1}%",
        report.candidates.mean,
        report.candidates.p50,
        report.candidates.p95,
        report.candidates.max,
        report.candidates.estimated_semantic_char_reduction * 100.0,
    );
    println!(
        "gate: {} ({} misses retained, {} omitted)",
        if report.gate.passed { "PASS" } else { "FAIL" },
        report.misses.len(),
        report.omitted_misses,
    );
}

fn print_metric(label: &str, metric: &knowdit_kg::router_eval::RecallMetric) {
    println!(
        "{label}: {}/{} = {} (required {:.2}%, Wilson 95% {}..{})",
        metric.hits,
        metric.total,
        format_percent(metric.recall),
        metric.required * 100.0,
        format_percent(metric.wilson_95_low),
        format_percent(metric.wilson_95_high),
    );
}

fn format_percent(value: Option<f64>) -> String {
    value
        .map(|value| format!("{:.2}%", value * 100.0))
        .unwrap_or_else(|| "n/a".to_string())
}
