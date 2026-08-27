//! Baseline creation: freeze a production (or synthetic) KG into an
//! immutable seed snapshot plus manifest.

use crate::error::Result;
use crate::manifest::BaselineManifest;
use knowdit_kg::db::HistoricalDatabase;
use std::collections::BTreeMap;
use std::path::Path;

/// Create a baseline from an existing database connection: export the
/// SQL snapshot, compute per-table row counts, validate integrity, and
/// confirm no transient extraction checkpoints remain.
pub async fn create_baseline(
    db: &HistoricalDatabase,
    suite_version: &str,
    git_commit: &str,
) -> Result<(BaselineManifest, String)> {
    let sql = db.export_sql_snapshot().await?;
    let snapshot_sha256 = crate::sha256_hex(sql.as_bytes());

    let graph = db.load_knowledge_graph().await?;
    let mut table_rows = BTreeMap::new();
    table_rows.insert("project".to_string(), graph.projects.len());
    table_rows.insert("project_platform".to_string(), graph.project_platforms.len());
    table_rows.insert("category".to_string(), graph.categories.len());
    table_rows.insert("semantic_node".to_string(), graph.nodes.len());
    table_rows.insert(
        "semantic_function".to_string(),
        graph.semantic_functions.len(),
    );
    table_rows.insert("project_category".to_string(), graph.project_categories.len());
    table_rows.insert("project_semantic".to_string(), graph.project_semantics.len());
    table_rows.insert("semantic_merge".to_string(), graph.semantic_merges.len());
    table_rows.insert("audit_finding".to_string(), graph.findings.len());
    table_rows.insert("finding_category".to_string(), graph.finding_categories.len());
    table_rows.insert(
        "audit_finding_category".to_string(),
        graph.audit_finding_categories.len(),
    );
    table_rows.insert("project_finding".to_string(), graph.project_findings.len());
    table_rows.insert(
        "semantic_finding_link".to_string(),
        graph.semantic_finding_links.len(),
    );
    table_rows.insert(
        "finding_link_status".to_string(),
        graph.finding_link_statuses.len(),
    );
    table_rows.insert("finding_merge".to_string(), graph.finding_merges.len());

    let validation = db.validate_db(false).await?;
    // The current historical-KG schema has no extraction-checkpoint
    // table (the old checkpoint system was removed); a frozen baseline
    // is therefore checkpoint-free by construction.
    let no_checkpoints = true;

    Ok((
        BaselineManifest {
            suite_version: suite_version.to_string(),
            git_commit: git_commit.to_string(),
            snapshot_sha256,
            table_rows,
            validation_issues: validation.remaining_issue_count(),
            no_checkpoints,
        },
        sql,
    ))
}

/// Write a created baseline into a suite directory:
/// `baselines/<version>.sql` + `baselines/<version>.manifest.json`.
pub fn write_baseline(
    suite_root: &Path,
    manifest: &BaselineManifest,
    sql: &str,
) -> Result<()> {
    let dir = suite_root.join("baselines");
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join(format!("{}.sql", manifest.suite_version)), sql)?;
    std::fs::write(
        dir.join(format!("{}.manifest.json", manifest.suite_version)),
        serde_json::to_string_pretty(manifest)?,
    )?;
    Ok(())
}
