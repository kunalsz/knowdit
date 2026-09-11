//! `LinkInput` — one expanded `(extract, historical, finding)` link, the
//! unit of work consumed by every language's spec-generation backend.
//!
//! Lives here (rather than in a language-specific spec crate) because it
//! bundles the project-side match data ([`SemanticMatch`] /
//! [`HistoricalSemanticRecord`], both defined in this crate) with the
//! knowledge-graph rows, and is shared by the Solidity (`knowdit-audit`)
//! and Move (`knowdit-move`) spec pipelines.

use std::fmt;

use knowdit_kg_model::ExtractedSemantic;
use knowdit_kg_model::db::{audit_finding, semantic_node};
use knowdit_kg_model::link_strength::LinkStrength;

use crate::{HistoricalLinkedFinding, HistoricalSemanticRecord, MatchStrength};
#[cfg(test)]
use std::collections::{BTreeMap, BTreeSet};
#[cfg(test)]
use crate::SemanticMatch;

/// Stable identifier for one expanded `(extract, historical, finding)` link.
/// Public so orchestrators can log / snapshot progress without reaching into
/// a generator's private runtime state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LinkKey {
    pub extract_id: i32,
    pub historical_id: i32,
    pub finding_id: i32,
}

/// Compact, cloneable description of one pending `(extract, historical,
/// finding)` link, without any of the heavy per-link source payloads.
///
/// This is what a spec planner keeps in memory while ordering / capping
/// candidates; the full [`LinkInput`] (which owns cloned prompt strings and
/// KG row models) is only materialized for the batch actually being run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkCandidate {
    pub key: LinkKey,
    /// Mapper-emitted strength on the `(extract, historical)` pair.
    pub match_strength: MatchStrength,
    /// Global-linker-emitted strength on the `(historical, finding)` edge.
    pub link_strength: LinkStrength,
    /// Spec ids already committed for this link in a prior run; empty for a
    /// fresh link. Populated by the planner's resume pass.
    pub pre_committed_spec_ids: Vec<i32>,
}

/// One expanded `(extract, historical, finding)` link ready to run.
#[derive(Debug, Clone)]
pub struct LinkInput {
    pub extract_id: i32,
    pub historical_id: i32,
    pub finding_id: i32,
    /// Mapper-emitted strength on the underlying `(extract, historical)`
    /// pair, copied onto every LinkInput fanned out from that pair. Used by
    /// gen-specs to filter / order candidates by strength.
    pub strength: MatchStrength,
    /// Global linker-emitted strength on the `(historical, finding)` edge —
    /// how directly this historical finding instantiates the historical
    /// semantic. Independent axis from `strength`.
    pub link_strength: LinkStrength,
    pub extract: ExtractedSemantic,
    pub historical: semantic_node::Model,
    pub finding: audit_finding::Model,
    /// `historical.description` / `finding.{description,patterns,exploits}`
    /// rendered with their merged-variant deltas from the mirror (bounded
    /// canonical representative + each folded raw's concrete delta). Prompt
    /// builders should read these instead of the raw fields; each falls back to
    /// the raw field when the mirror carried no rendered value.
    pub historical_rendered_description: String,
    pub finding_rendered_description: String,
    pub finding_rendered_patterns: String,
    pub finding_rendered_exploits: String,
    /// Spec ids already committed for this link in a prior run. Empty for
    /// fresh links. When non-empty, a backend's per-link runner short-circuits
    /// the gen-spec agent and synthesizes an outcome from these ids so the
    /// caller drives fuzz / reflect / regen against them. Populated from
    /// [`crate::RepoDatabase::link_resume_state`].
    pub pre_committed_spec_ids: Vec<i32>,
}

/// Pick the mirror-rendered value, or fall back to the raw canonical field when
/// the mirror carried nothing rendered (e.g. legacy project DBs written before
/// merged-variant rendering).
fn rendered_or(rendered: &str, raw: &str) -> String {
    if rendered.trim().is_empty() {
        raw.to_string()
    } else {
        rendered.to_string()
    }
}

impl LinkInput {
    pub fn key(&self) -> LinkKey {
        LinkKey {
            extract_id: self.extract_id,
            historical_id: self.historical_id,
            finding_id: self.finding_id,
        }
    }

    /// Build one [`LinkInput`] from its exact source rows. The primitive the
    /// planner calls for each materialized candidate; `build_all` is just a
    /// fan-out over this.
    #[allow(clippy::too_many_arguments)]
    pub fn materialize(
        extract_id: i32,
        historical_id: i32,
        finding_id: i32,
        strength: MatchStrength,
        link_strength: LinkStrength,
        extract: &ExtractedSemantic,
        record: &HistoricalSemanticRecord,
        linked: &HistoricalLinkedFinding,
    ) -> Self {
        let finding = &linked.finding;
        LinkInput {
            extract_id,
            historical_id,
            finding_id,
            strength,
            link_strength,
            extract: extract.clone(),
            historical_rendered_description: rendered_or(
                &record.rendered_description,
                &record.semantic.description,
            ),
            finding_rendered_description: rendered_or(
                &linked.rendered_description,
                &finding.description,
            ),
            finding_rendered_patterns: rendered_or(&linked.rendered_patterns, &finding.patterns),
            finding_rendered_exploits: rendered_or(&linked.rendered_exploits, &finding.exploits),
            historical: record.semantic.clone(),
            finding: finding.clone(),
            pre_committed_spec_ids: Vec::new(),
        }
    }

    /// Exact materialization for one `(E, H, F)` identity. Locates the linked
    /// finding inside `record` and delegates to [`Self::materialize`]. Returns
    /// `None` only when the historical record has no finding with the requested
    /// id — i.e. the KG changed since the reference was recorded.
    pub fn materialize_exact(
        key: LinkKey,
        strength: MatchStrength,
        extract: &ExtractedSemantic,
        record: &HistoricalSemanticRecord,
    ) -> Option<Self> {
        let linked = record
            .findings
            .iter()
            .find(|linked| linked.finding.id == key.finding_id)?;
        Some(Self::materialize(
            key.extract_id,
            key.historical_id,
            key.finding_id,
            strength,
            linked.strength,
            extract,
            record,
            linked,
        ))
    }

    /// Fan out mapper matches into one [`LinkInput`] per linked finding,
    /// deduplicated on the `(extract, historical, finding)` triple. Matches
    /// with no project-side extract row, no historical row, or no linked
    /// findings are skipped.
    ///
    /// **Test-only reference implementation.** Production planners materialize
    /// candidates on demand via [`Self::materialize_exact`] instead of
    /// expanding every link up front; this method exists solely as the
    /// equivalence-test oracle for that path and is not compiled into
    /// production builds.
    #[cfg(test)]
    pub fn build_all(
        matches: &[SemanticMatch],
        extracted_by_id: &BTreeMap<i32, ExtractedSemantic>,
        historical_by_id: &BTreeMap<i32, HistoricalSemanticRecord>,
    ) -> Vec<LinkInput> {
        let mut out = Vec::new();
        let mut seen: BTreeSet<(i32, i32, i32)> = BTreeSet::new();
        for m in matches {
            let extract_id = m.extract_id;
            let historical_id = m.historical_id;
            let Some(extract) = extracted_by_id.get(&extract_id) else {
                tracing::warn!(
                    "Skipping match (extract={}, historical={}): no project_semantic row",
                    extract_id,
                    historical_id
                );
                continue;
            };
            let Some(record) = historical_by_id.get(&historical_id) else {
                tracing::warn!(
                    "Skipping match (extract={}, historical={}): no historical_semantic row",
                    extract_id,
                    historical_id
                );
                continue;
            };
            if record.findings.is_empty() {
                tracing::debug!(
                    "Skipping match (extract={}, historical={}): historical has no findings linked",
                    extract_id,
                    historical_id
                );
                continue;
            }
            for linked in &record.findings {
                let finding = &linked.finding;
                if !seen.insert((extract_id, historical_id, finding.id)) {
                    continue;
                }
                out.push(Self::materialize(
                    extract_id,
                    historical_id,
                    finding.id,
                    m.strength,
                    linked.strength,
                    extract,
                    record,
                    linked,
                ));
            }
        }
        out
    }
}

impl fmt::Display for LinkInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Link(id={}, extract={}, historical={}, finding={}, strength={})",
            self.extract_id, self.extract_id, self.historical_id, self.finding_id, self.strength,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use knowdit_kg_model::audit_finding::FindingSeverity;
    use knowdit_kg_model::category::DeFiCategory;

    fn extract(name: &str) -> ExtractedSemantic {
        ExtractedSemantic {
            name: name.to_string(),
            category: DeFiCategory::Lending,
            definition: String::new(),
            description: format!("{name} description"),
            functions: Vec::new(),
        }
    }

    fn finding(id: i32) -> audit_finding::Model {
        audit_finding::Model {
            id,
            title: format!("finding {id}"),
            severity: FindingSeverity::Medium,
            root_cause: String::new(),
            description: format!("finding {id} description"),
            patterns: String::new(),
            exploits: String::new(),
        }
    }

    fn historical(id: i32, finding_ids: &[i32]) -> HistoricalSemanticRecord {
        HistoricalSemanticRecord {
            semantic: semantic_node::Model {
                id,
                name: format!("hist {id}"),
                definition: String::new(),
                description: format!("hist {id} description"),
                category: DeFiCategory::Lending,
            },
            findings: finding_ids
                .iter()
                .map(|finding_id| HistoricalLinkedFinding {
                    finding: finding(*finding_id),
                    strength: LinkStrength::Medium,
                    evidence: String::new(),
                    raw_children: Vec::new(),
                    rendered_description: String::new(),
                    rendered_patterns: String::new(),
                    rendered_exploits: String::new(),
                })
                .collect(),
            raw_children: Vec::new(),
            rendered_description: String::new(),
        }
    }

    /// The removed production fan-out (`build_all`) and the on-demand
    /// `materialize_exact` path must produce identical links for the same
    /// source rows — this is the equivalence oracle that justified migrating
    /// production off `build_all`.
    #[test]
    fn build_all_matches_materialize_exact() {
        let mut extracted = BTreeMap::new();
        extracted.insert(1, extract("e1"));
        extracted.insert(2, extract("e2"));
        let mut historicals = BTreeMap::new();
        historicals.insert(100, historical(100, &[1, 2]));
        historicals.insert(200, historical(200, &[3]));

        let matches = vec![
            SemanticMatch {
                extract_id: 1,
                historical_id: 100,
                strength: MatchStrength::High,
                evidence: "a".to_string(),
            },
            SemanticMatch {
                extract_id: 1,
                historical_id: 200,
                strength: MatchStrength::Medium,
                evidence: "b".to_string(),
            },
            SemanticMatch {
                extract_id: 2,
                historical_id: 100,
                strength: MatchStrength::Low,
                evidence: "c".to_string(),
            },
            // Duplicate triple — must be deduped identically by both paths.
            SemanticMatch {
                extract_id: 1,
                historical_id: 100,
                strength: MatchStrength::High,
                evidence: "dup".to_string(),
            },
        ];

        let all = LinkInput::build_all(&matches, &extracted, &historicals);

        let mut expected = Vec::new();
        let mut seen = BTreeSet::new();
        for m in &matches {
            let Some(extract) = extracted.get(&m.extract_id) else {
                continue;
            };
            let Some(record) = historicals.get(&m.historical_id) else {
                continue;
            };
            for linked in &record.findings {
                if !seen.insert((m.extract_id, m.historical_id, linked.finding.id)) {
                    continue;
                }
                expected.push(
                    LinkInput::materialize_exact(
                        LinkKey {
                            extract_id: m.extract_id,
                            historical_id: m.historical_id,
                            finding_id: linked.finding.id,
                        },
                        m.strength,
                        extract,
                        record,
                    )
                    .expect("exact materialization"),
                );
            }
        }

        assert_eq!(all.len(), expected.len());
        for (a, b) in all.iter().zip(expected.iter()) {
            assert_eq!(a.key(), b.key());
            assert_eq!(a.strength, b.strength);
            assert_eq!(a.link_strength, b.link_strength);
            assert_eq!(a.extract.name, b.extract.name);
            assert_eq!(
                a.historical_rendered_description,
                b.historical_rendered_description
            );
            assert_eq!(
                a.finding_rendered_description,
                b.finding_rendered_description
            );
            assert_eq!(a.historical.id, b.historical.id);
            assert_eq!(a.finding.id, b.finding.id);
        }
    }
}
