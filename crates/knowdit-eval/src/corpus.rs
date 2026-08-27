//! Suite directory loading, verification, and leakage checks.

use crate::error::Result;
use crate::manifest::{BaselineManifest, Policy, Suite};
use std::path::{Path, PathBuf};

/// A suite loaded from disk, with its baseline manifest and policy.
/// The baseline manifest is `None` only before the first baseline is
/// frozen (`eval baseline create`).
#[derive(Debug, Clone)]
pub struct LoadedSuite {
    pub root: PathBuf,
    pub suite: Suite,
    pub baseline_manifest: Option<BaselineManifest>,
    pub policy: Policy,
}

impl LoadedSuite {
    /// Load `suite.json` and `policy.json` from a suite root, plus the
    /// baseline manifest when it exists.
    pub fn load(root: &Path) -> Result<Self> {
        let suite_path = root.join("suite.json");
        let suite: Suite = serde_json::from_str(&std::fs::read_to_string(&suite_path)?)
            .map_err(|e| crate::error::EvalError::manifest(format!("{suite_path:?}: {e}")))?;
        suite.validate()?;

        let manifest_path = root
            .join("baselines")
            .join(format!("{}.manifest.json", suite.version));
        let baseline_manifest = if manifest_path.is_file() {
            let manifest: BaselineManifest =
                serde_json::from_str(&std::fs::read_to_string(&manifest_path)?).map_err(|e| {
                    crate::error::EvalError::manifest(format!("{manifest_path:?}: {e}"))
                })?;
            if manifest.suite_version != suite.version {
                return Err(crate::error::EvalError::manifest(format!(
                    "baseline manifest declares suite {} but lives under {}",
                    manifest.suite_version, suite.version
                )));
            }
            Some(manifest)
        } else {
            None
        };

        let policy_path = root.join("policy.json");
        let policy = if policy_path.is_file() {
            serde_json::from_str(&std::fs::read_to_string(&policy_path)?).map_err(|e| {
                crate::error::EvalError::manifest(format!("{policy_path:?}: {e}"))
            })?
        } else {
            Policy::default()
        };

        Ok(Self {
            root: root.to_path_buf(),
            suite,
            baseline_manifest,
            policy,
        })
    }

    /// The baseline manifest, or an error explaining that no baseline
    /// has been frozen yet.
    pub fn require_baseline_manifest(&self) -> Result<&BaselineManifest> {
        self.baseline_manifest.as_ref().ok_or_else(|| {
            crate::error::EvalError::manifest(format!(
                "suite {} has no frozen baseline; run `knowdit eval baseline create` first",
                self.suite.version
            ))
        })
    }

    /// Verify the suite's self-consistency: replay documents reference
    /// existing sources, gold digests point at gold files, and the
    /// baseline snapshot hash matches the manifest.
    pub async fn verify(&self) -> Result<Vec<String>> {
        let mut problems = Vec::new();

        // Baseline snapshot presence + hash.
        let baseline = self
            .root
            .join("baselines")
            .join(format!("{}.sql", self.suite.version));
        let baseline_gz = self
            .root
            .join("baselines")
            .join(format!("{}.sql.gz", self.suite.version));
        if !baseline.is_file() && !baseline_gz.is_file() {
            problems.push(format!(
                "baseline snapshot missing under {}",
                self.root.join("baselines").display()
            ));
        } else if let Some(manifest) = &self.baseline_manifest {
            let bytes = if baseline.is_file() {
                std::fs::read(&baseline)?
            } else {
                use flate2::read::GzDecoder;
                use std::io::Read;
                let mut decoder = GzDecoder::new(std::fs::File::open(&baseline_gz)?);
                let mut out = Vec::new();
                decoder.read_to_end(&mut out)?;
                out
            };
            let actual = crate::sha256_hex(&bytes);
            if actual != manifest.snapshot_sha256 {
                problems.push(format!(
                    "baseline snapshot hash mismatch: manifest {} vs actual {actual}",
                    manifest.snapshot_sha256
                ));
            }
        } else {
            problems.push(
                "baseline manifest missing; run `knowdit eval baseline create` first".to_string(),
            );
        }

        // Source existence for every replay document.
        for doc in self.suite.replay_documents() {
            match &doc.kind {
                crate::manifest::DocumentKind::SourceDir { spec } => {
                    let Some(root_part) = spec.split(':').nth(1) else {
                        problems.push(format!("document {} has malformed spec", doc.id));
                        continue;
                    };
                    let path = Path::new(root_part);
                    if !path.is_dir() {
                        problems.push(format!(
                            "document {} source dir missing: {}",
                            doc.id,
                            path.display()
                        ));
                    }
                }
                crate::manifest::DocumentKind::C4 {
                    dataset_dir,
                    contest_id,
                } => {
                    let audit_json = dataset_dir.join("audits").join(format!("{contest_id}.json"));
                    let contracts_dir = dataset_dir.join("contracts").join(contest_id.to_string());
                    if !audit_json.is_file() {
                        problems.push(format!(
                            "document {} audit metadata missing: {}",
                            doc.id,
                            audit_json.display()
                        ));
                    }
                    if !contracts_dir.is_dir() {
                        problems.push(format!(
                            "document {} contracts missing: {}",
                            doc.id,
                            contracts_dir.display()
                        ));
                    }
                }
                crate::manifest::DocumentKind::Sherlock {
                    out_dir,
                    contest_id,
                } => {
                    let metadata = out_dir.join("metadata").join(format!("{contest_id}.json"));
                    if !metadata.is_file() {
                        problems.push(format!(
                            "document {} sherlock metadata missing: {}",
                            doc.id,
                            metadata.display()
                        ));
                    }
                }
            }

            // Gold digest → gold file must exist.
            if let Some(digest) = &doc.gold_digest {
                let gold_file = self.root.join("gold").join(format!("{digest}.jsonl"));
                if !gold_file.is_file() {
                    problems.push(format!(
                        "document {} references missing gold file {}",
                        doc.id,
                        gold_file.display()
                    ));
                }
            }
        }

        Ok(problems)
    }

    /// Load the ordered replay document IDs.
    pub fn replay_order(&self) -> &[String] {
        &self.suite.order
    }
}
