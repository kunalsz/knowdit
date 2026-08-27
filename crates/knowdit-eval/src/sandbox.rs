//! Immutable baseline import and isolated sandbox databases.
//!
//! Every candidate and baseline run starts from the SAME frozen SQL
//! snapshot imported into a fresh temporary SQLite file. Sandboxes are
//! never reused or mutated in place by the harness itself: the
//! learning driver is the only writer, and it always receives a fresh
//! copy per document (fixed-context) or per run (growth-replay).

use crate::error::Result;
use crate::manifest::BaselineManifest;
use crate::sha256_hex;
use flate2::read::GzDecoder;
use knowdit_kg::db::HistoricalDatabase;
use std::io::Read;
use std::path::{Path, PathBuf};

/// A frozen seed database, addressable by its snapshot bytes and the
/// manifest that describes them.
#[derive(Debug, Clone)]
pub struct Baseline {
    pub manifest: BaselineManifest,
    pub sql_bytes: Vec<u8>,
    /// Absolute path of the snapshot file, kept for error messages.
    pub source_path: PathBuf,
}

impl Baseline {
    /// Load a baseline from a suite directory. Accepts
    /// `<baseline>.sql` and gzip-compressed `<baseline>.sql.gz`.
    pub fn load(suite_root: &Path, manifest: &BaselineManifest) -> Result<Self> {
        let baseline_dir = suite_root.join("baselines");
        let plain = baseline_dir.join(format!("{}.sql", manifest.suite_version));
        let gzipped = baseline_dir.join(format!("{}.sql.gz", manifest.suite_version));

        let (sql_bytes, source_path) = if plain.is_file() {
            (std::fs::read(&plain)?, plain)
        } else if gzipped.is_file() {
            let raw = std::fs::read(&gzipped)?;
            let mut decoder = GzDecoder::new(raw.as_slice());
            let mut out = Vec::new();
            decoder.read_to_end(&mut out)?;
            (out, gzipped)
        } else {
            return Err(crate::error::EvalError::sandbox(format!(
                "baseline snapshot for suite {} not found under {}",
                manifest.suite_version,
                baseline_dir.display()
            )));
        };

        let actual = sha256_hex(&sql_bytes);
        if actual != manifest.snapshot_sha256 {
            return Err(crate::error::EvalError::sandbox(format!(
                "baseline snapshot {} hash mismatch: expected {}, got {actual}",
                source_path.display(),
                manifest.snapshot_sha256
            )));
        }

        Ok(Self {
            manifest: manifest.clone(),
            sql_bytes,
            source_path,
        })
    }

    /// Create a fresh sandbox database from this baseline. The file
    /// lands inside `dir` under a unique name so concurrent runs never
    /// collide.
    pub async fn materialize(&self, dir: &Path) -> Result<Sandbox> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!(
            "sandbox-{}-{}.sqlite",
            std::process::id(),
            unique_suffix()
        ));
        let url = format!("sqlite://{}?mode=rwc", path.to_string_lossy());
        let db = HistoricalDatabase::connect(&url).await?;
        db.import_sql_snapshot(std::str::from_utf8(&self.sql_bytes).map_err(|e| {
            crate::error::EvalError::sandbox(format!("baseline snapshot is not valid UTF-8: {e}"))
        })?)
        .await?;
        Ok(Sandbox {
            db,
            path,
            baseline_digest: self.manifest.snapshot_sha256.clone(),
        })
    }
}

/// One isolated SQLite work database derived from a baseline.
pub struct Sandbox {
    pub db: HistoricalDatabase,
    pub path: PathBuf,
    pub baseline_digest: String,
}

impl std::fmt::Debug for Sandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sandbox")
            .field("path", &self.path)
            .field("baseline_digest", &self.baseline_digest)
            .finish_non_exhaustive()
    }
}

impl Sandbox {
    /// Export this sandbox's current graph state.
    pub async fn knowledge_graph(&self) -> Result<knowdit_kg::knowledge_graph::KnowledgeGraph> {
        Ok(self.db.load_knowledge_graph().await?)
    }

    /// Export the sandbox's current SQL state (for retained artifacts).
    pub async fn export_sql(&self) -> Result<String> {
        Ok(self.db.export_sql_snapshot().await?)
    }

    /// Validate referential integrity; returns the remaining-issue
    /// count so runs can fail fast on invariant violations.
    pub async fn validation_issue_count(&self) -> Result<usize> {
        let report = self.db.validate_db(false).await?;
        Ok(report.remaining_issue_count())
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_file(format!("{}-wal", self.path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", self.path.display()));
    }
}

/// Nanosecond-resolution suffix so sandboxes created in the same
/// process never share a file name.
fn unique_suffix() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{BaselineManifest, Suite};
    use std::collections::BTreeMap;
    use tempfile::TempDir;

    fn empty_manifest() -> BaselineManifest {
        BaselineManifest {
            suite_version: "kg-test".to_string(),
            git_commit: "0".repeat(40),
            snapshot_sha256: String::new(),
            table_rows: BTreeMap::new(),
            validation_issues: 0,
            no_checkpoints: true,
        }
    }

    #[tokio::test]
    async fn materialize_produces_valid_empty_kg() {
        let tmp = TempDir::new().expect("tempdir");
        let src = tmp.path().join("src.sqlite");
        let src_db = HistoricalDatabase::connect(&format!(
            "sqlite://{}?mode=rwc",
            src.to_string_lossy()
        ))
        .await
        .expect("connect");
        src_db.init().await.expect("init");
        let sql = src_db.export_sql_snapshot().await.expect("export");

        let manifest = BaselineManifest {
            snapshot_sha256: crate::sha256_hex(sql.as_bytes()),
            ..empty_manifest()
        };
        let suite_root = tmp.path().join("suite");
        std::fs::create_dir_all(suite_root.join("baselines")).expect("mkdir");
        std::fs::write(
            suite_root.join("baselines").join("kg-test.sql"),
            &sql,
        )
        .expect("write");

        let baseline = Baseline::load(&suite_root, &manifest).expect("load");
        let sandbox = baseline.materialize(tmp.path()).await.expect("materialize");
        assert_eq!(sandbox.validation_issue_count().await.expect("validate"), 0);
        let graph = sandbox.knowledge_graph().await.expect("graph");
        assert!(graph.projects.is_empty());
    }

    #[test]
    fn suite_validation_catches_bad_order() {
        let mut docs = BTreeMap::new();
        docs.insert(
            "d1".to_string(),
            crate::manifest::DocumentSpec {
                id: "d1".to_string(),
                kind: crate::manifest::DocumentKind::SourceDir {
                    spec: "p:/tmp/p".to_string(),
                },
                split: crate::manifest::Split::Dev,
                role: crate::manifest::DocumentRole::Replay,
                date: "2024-01-01".to_string(),
                gold_digest: None,
            },
        );
        let suite = Suite {
            version: "kg-test".to_string(),
            description: String::new(),
            order: vec!["d1".to_string(), "nope".to_string()],
            documents: docs,
        };
        assert!(suite.validate().is_err());
    }
}
