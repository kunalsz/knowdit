use std::path::{Path, PathBuf};

use clap::Args;
use color_eyre::eyre::{Result, WrapErr};
use knowdit_kg::db::HistoricalDatabase;
use knowdit_kg::project_loader::ProjectData;
use knowdit_kg_model::db::operation_history::OperationType;
use llmy::clap::OpenAISetup;

use crate::cmd::learn::finding_link_args::FindingLinkCliArgs;
use crate::cmd::learn::learn::run_pipeline;
use crate::cmd::learn::merge_args::MergeCliArgs;

/// Ingest markdown security reports (attack analyses, audit findings,
/// post-mortems, CVE disclosures) as narrative projects into the
/// historical knowledge graph. Each .md file becomes one project.
/// Both exploit-pattern semantics and vulnerability findings are
/// extracted from the same markdown content.
#[derive(serde::Serialize)]
struct ReportsOperationArgs {
    merge: MergeCliArgs,
    feed_namespace: String,
    batch_size: usize,
    report_count: usize,
    migrate_legacy_feed: bool,
}

#[derive(Args, Clone)]
pub struct ReportsArgs {
    #[command(flatten)]
    pub llm: OpenAISetup,

    /// Directory containing .md security report files (recursively walked)
    #[arg(long)]
    pub dir: PathBuf,

    /// Number of projects to categorize+extract concurrently (merge is always serial)
    #[arg(long, default_value_t = 1)]
    pub concurrency: usize,

    /// Number of reports to process before merging, linking, and releasing the batch.
    /// Zero processes all discovered reports in one batch.
    #[arg(long, default_value_t = 0)]
    pub batch_size: usize,

    /// Stable namespace for report source identities.
    #[arg(long, default_value = "attack-analyses")]
    pub feed_namespace: String,

    /// Migrate the exact legacy ordinal feed set into stable source identities.
    /// This performs no learning and fails closed on any mismatch.
    #[arg(long, default_value_t = false)]
    pub migrate_legacy_feed: bool,

    #[command(flatten)]
    pub merge: MergeCliArgs,

    #[command(flatten)]
    pub finding_link: FindingLinkCliArgs,
}

fn normalized_relative_path(path: &Path) -> Result<String> {
    let value = path
        .to_str()
        .ok_or_else(|| color_eyre::eyre::eyre!("report path is not valid UTF-8"))?
        .replace('\\', "/");
    Ok(value.trim_start_matches("./").to_string())
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn stable_source_id(namespace: &str, relative_path: &str) -> String {
    format!(
        "feed-v2-{}",
        sha256_hex(format!("{namespace}\0{relative_path}").as_bytes())
    )
}

impl ReportsArgs {
    pub async fn run(self, db: &HistoricalDatabase) -> Result<()> {
        self.merge.validate()?;
        self.finding_link.validate()?;

        let dir = self.dir.canonicalize().wrap_err_with(|| {
            format!(
                "failed to canonicalize report directory {}",
                self.dir.display()
            )
        })?;

        if !dir.is_dir() {
            return Err(color_eyre::eyre::eyre!(
                "not a directory: {}",
                dir.display()
            ));
        }

        // Walk the directory recursively for .md files.
        let mut md_files: Vec<PathBuf> = Vec::new();
        let mut stack = vec![dir.clone()];
        while let Some(entry) = stack.pop() {
            let mut read_dir = tokio::fs::read_dir(&entry)
                .await
                .wrap_err_with(|| format!("failed to read directory {}", entry.display()))?;
            while let Some(child) = read_dir.next_entry().await? {
                let child_path = child.path();
                if child_path.is_dir() {
                    stack.push(child_path);
                } else if child_path.extension().map_or(false, |ext| ext == "md") {
                    md_files.push(child_path);
                }
            }
        }

        md_files.sort();
        tracing::info!(
            "Found {} markdown report(s) under {}",
            md_files.len(),
            dir.display()
        );

        if md_files.is_empty() {
            tracing::warn!("No .md files found; nothing to do.");
            return Ok(());
        }

        let source_rows = db
            .feed_report_sources_for_namespace(&self.feed_namespace)
            .await?;
        let legacy_rows = db.legacy_feed_platform_ids().await?;
        if self.migrate_legacy_feed {
            let mut migrated_sources = Vec::with_capacity(md_files.len());
            for (index, path) in md_files.iter().enumerate() {
                let relative_path =
                    normalized_relative_path(path.strip_prefix(&dir).unwrap_or(path))?;
                let bytes = tokio::fs::read(path).await?;
                let legacy_platform_id = format!("feed-{index:016x}");
                migrated_sources.push(knowdit_kg::project_loader::FeedReportSource {
                    source_namespace: self.feed_namespace.clone(),
                    stable_source_id: stable_source_id(&self.feed_namespace, &relative_path),
                    relative_path,
                    legacy_platform_id: Some(legacy_platform_id),
                    content_hash: sha256_hex(&bytes),
                });
            }
            if legacy_rows.len() != migrated_sources.len() {
                return Err(color_eyre::eyre::eyre!(
                    "legacy migration requires {} reports, but the database contains {} legacy feed projects",
                    migrated_sources.len(),
                    legacy_rows.len()
                ));
            }
            let migrated = db.migrate_legacy_feed_sources(&migrated_sources).await?;
            tracing::info!("Migrated {migrated} legacy feed report source(s)");
            return Ok(());
        }
        let mapped_legacy_ids: std::collections::HashSet<&str> = source_rows
            .iter()
            .filter_map(|row| row.legacy_platform_id.as_deref())
            .collect();
        if legacy_rows
            .iter()
            .any(|(legacy_id, _)| !mapped_legacy_ids.contains(legacy_id.as_str()))
        {
            return Err(color_eyre::eyre::eyre!(
                "legacy feed projects exist without complete source mappings; rerun with --migrate-legacy-feed using the exact original report set"
            ));
        }
        db.mark_feed_namespace_inactive(&self.feed_namespace)
            .await?;
        for md_path in &md_files {
            let relative = md_path.strip_prefix(&dir).unwrap_or(md_path);
            let relative_path = normalized_relative_path(relative)?;
            db.mark_feed_report_seen(&self.feed_namespace, &relative_path)
                .await?;
        }

        let llm = self.llm.to_llm().await;
        let batch_size = if self.batch_size == 0 {
            md_files.len()
        } else {
            self.batch_size
        };
        let agent_options = self.merge.to_agent_options();
        let merge_chunking = self.merge.to_chunking_options();
        let link_options = self.finding_link.to_options(self.concurrency);
        let total_batches = md_files.len().div_ceil(batch_size);

        for (batch_index, batch_paths) in md_files.chunks(batch_size).enumerate() {
            tracing::info!(
                "Loading report batch {}/{} ({} report(s))",
                batch_index + 1,
                total_batches,
                batch_paths.len()
            );
            let mut projects = Vec::with_capacity(batch_paths.len());

            for md_path in batch_paths {
                let file_stem = md_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unnamed");

                let relative = md_path.strip_prefix(&dir).unwrap_or(md_path);
                let relative_path = normalized_relative_path(relative)?;
                let bytes = tokio::fs::read(md_path).await?;
                let source = knowdit_kg::project_loader::FeedReportSource {
                    source_namespace: self.feed_namespace.clone(),
                    relative_path: relative_path.clone(),
                    stable_source_id: stable_source_id(&self.feed_namespace, &relative_path),
                    legacy_platform_id: source_rows
                        .iter()
                        .find(|row| row.relative_path == relative_path)
                        .and_then(|row| row.legacy_platform_id.clone()),
                    content_hash: sha256_hex(&bytes),
                };
                let platform_id = source.stable_source_id.clone();

                tracing::debug!(
                    "Loading narrative report: {} ({})",
                    file_stem,
                    relative.display()
                );

                let project = ProjectData::from_narrative_md(
                    file_stem,
                    &dir,
                    Some(&platform_id),
                    relative,
                    source,
                )
                .await
                .wrap_err_with(|| format!("failed to load report {}", relative.display()))?;

                projects.push(project);
            }

            tracing::info!(
                "Processing report batch {}/{}: extraction, merge, database write, and linking",
                batch_index + 1,
                total_batches
            );
            run_pipeline(
                db,
                &llm,
                projects,
                self.concurrency,
                true,
                true,
                agent_options.clone(),
                merge_chunking,
                link_options.clone(),
                self.merge.force_remove_pending_chunks,
            )
            .await?;
        }

        let operation_args = serde_json::to_value(ReportsOperationArgs {
            merge: self.merge.clone(),
            feed_namespace: self.feed_namespace.clone(),
            batch_size: self.batch_size,
            report_count: md_files.len(),
            migrate_legacy_feed: self.migrate_legacy_feed,
        })
        .wrap_err("failed to serialize report feed args for operation history")?;
        db.record_operation(OperationType::ReportsFeed, operation_args)
            .await?;

        tracing::info!("Report ingestion complete.");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_source_id_is_independent_of_discovery_order() {
        let first = stable_source_id("attack-analyses", "2026/report-a.md");
        let second = stable_source_id("attack-analyses", "2026/report-a.md");
        assert_eq!(first, second);
        assert_ne!(
            first,
            stable_source_id("attack-analyses", "2026/report-b.md")
        );
        assert_ne!(first, stable_source_id("other-feed", "2026/report-a.md"));
    }

    #[test]
    fn relative_path_normalization_is_platform_separator_safe() {
        let path = Path::new("./2026\\report.md");
        let normalized = match normalized_relative_path(path) {
            Ok(value) => value,
            Err(error) => panic!("unexpected path normalization error: {error}"),
        };
        assert_eq!(normalized, "2026/report.md");
    }
}
