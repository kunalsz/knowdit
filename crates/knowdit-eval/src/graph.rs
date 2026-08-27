//! ID-independent normalized graph export, diffing, and invariant
//! metrics.
//!
//! Production DB integer IDs are insertion-order-dependent, so
//! comparisons never touch them. Every node/edge is mapped to a stable
//! fingerprint derived from normalized content and seed provenance;
//! the diff then operates on fingerprint space, which is identical
//! across databases as long as their *content* is identical.

use crate::sha256_str;
use knowdit_kg::knowledge_graph::KnowledgeGraph;
use knowdit_kg_model::db::{audit_finding, project, semantic_node};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// One canonical semantic node in fingerprint space.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SemanticFp {
    pub fp: String,
    pub name: String,
    pub category: String,
    /// Normalized definition + description + provenance projects +
    /// functions (joined, sorted), fed through the same hash.
    pub body_digest: String,
    pub project_count: usize,
    pub function_count: usize,
    /// Whether this node is folded into another (raw child).
    pub folded: bool,
}

/// One canonical audit-finding node in fingerprint space.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FindingFp {
    pub fp: String,
    pub title: String,
    pub severity: String,
    pub body_digest: String,
    pub project_count: usize,
    pub folded: bool,
}

/// One semantic→finding link in fingerprint space.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LinkFp {
    pub semantic_fp: String,
    pub finding_fp: String,
    pub strength: String,
    pub evidence_digest: String,
}

/// One merge edge in fingerprint space.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MergeFp {
    pub from_fp: String,
    pub to_fp: String,
    pub appended_digest: String,
}

/// Normalized, sorted representation of one KG state. Everything is
/// content-addressed so the same graph content produces byte-equal
/// fingerprints regardless of integer IDs or insertion order.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NormalizedGraph {
    pub semantics: Vec<SemanticFp>,
    pub findings: Vec<FindingFp>,
    pub links: Vec<LinkFp>,
    pub semantic_merges: Vec<MergeFp>,
    pub finding_merges: Vec<MergeFp>,
    pub projects: Vec<String>,
}

/// Count-level invariants computed from one graph.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphMetrics {
    pub semantic_total: usize,
    pub semantic_canonical: usize,
    pub semantic_folded: usize,
    pub finding_total: usize,
    pub finding_canonical: usize,
    pub finding_folded: usize,
    pub link_total: usize,
    pub semantic_merge_edges: usize,
    pub finding_merge_edges: usize,
    /// Links whose finding endpoint has no completed link-status row.
    pub partial_links: usize,
    /// Merge edges pointing at nodes that do not exist.
    pub dangling_merge_targets: usize,
    /// Canonical nodes with no links at all.
    pub unlinked_canonical_semantics: usize,
    /// Raw-folded nodes that still carry links (should be 0 for
    /// cross-project links; in-project links may legitimately sit on
    /// raws).
    pub links_on_folded_endpoints: usize,
}

/// The difference between two normalized graphs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphDelta {
    pub added_semantics: Vec<SemanticFp>,
    pub removed_semantics: Vec<SemanticFp>,
    pub added_findings: Vec<FindingFp>,
    pub removed_findings: Vec<FindingFp>,
    pub added_links: Vec<LinkFp>,
    pub removed_links: Vec<LinkFp>,
    pub added_semantic_merges: Vec<MergeFp>,
    pub removed_semantic_merges: Vec<MergeFp>,
    pub added_finding_merges: Vec<MergeFp>,
    pub removed_finding_merges: Vec<MergeFp>,
    pub added_projects: Vec<String>,
    pub removed_projects: Vec<String>,
}

impl GraphDelta {
    pub fn is_empty(&self) -> bool {
        self.added_semantics.is_empty()
            && self.removed_semantics.is_empty()
            && self.added_findings.is_empty()
            && self.removed_findings.is_empty()
            && self.added_links.is_empty()
            && self.removed_links.is_empty()
            && self.added_semantic_merges.is_empty()
            && self.removed_semantic_merges.is_empty()
            && self.added_finding_merges.is_empty()
            && self.removed_finding_merges.is_empty()
            && self.added_projects.is_empty()
            && self.removed_projects.is_empty()
    }
}

fn digest_parts(parts: &[&str]) -> String {
    sha256_str(&parts.join("\u{1f}"))
}

/// Normalize one KG into fingerprint space. The result is sorted, so
/// it serializes deterministically.
pub fn normalize(graph: &KnowledgeGraph) -> NormalizedGraph {
    // ── provenance maps ──
    let mut sem_projects: BTreeMap<i32, BTreeSet<String>> = BTreeMap::new();
    let projects_by_id: BTreeMap<i32, &project::Model> =
        graph.projects.iter().map(|p| (p.id, p)).collect();
    for ps in &graph.project_semantics {
        if let Some(p) = projects_by_id.get(&ps.project_id) {
            sem_projects
                .entry(ps.semantic_node_id)
                .or_default()
                .insert(p.name.clone());
        }
    }
    let mut find_projects: BTreeMap<i32, BTreeSet<String>> = BTreeMap::new();
    for pf in &graph.project_findings {
        if let Some(p) = projects_by_id.get(&pf.project_id) {
            find_projects
                .entry(pf.audit_finding_id)
                .or_default()
                .insert(p.name.clone());
        }
    }

    let folded_sems: BTreeSet<i32> = graph
        .semantic_merges
        .iter()
        .map(|m| m.from_semantic_id)
        .collect();
    let folded_finds: BTreeSet<i32> = graph
        .finding_merges
        .iter()
        .map(|m| m.from_finding_id)
        .collect();

    // ── semantics ──
    let mut sem_by_id: BTreeMap<i32, &semantic_node::Model> = BTreeMap::new();
    for s in &graph.nodes {
        sem_by_id.insert(s.id, s);
    }
    let mut functions_by_sem: BTreeMap<i32, Vec<(String, String)>> = BTreeMap::new();
    for f in &graph.semantic_functions {
        functions_by_sem
            .entry(f.semantic_node_id)
            .or_default()
            .push((f.function_name.clone(), f.contract_path.clone()));
    }

    let mut semantics: Vec<SemanticFp> = graph
        .nodes
        .iter()
        .map(|s| {
            let projects = sem_projects.get(&s.id).cloned().unwrap_or_default();
            let mut funcs = functions_by_sem.get(&s.id).cloned().unwrap_or_default();
            funcs.sort();
            let projects_sorted: Vec<String> = projects.iter().cloned().collect();
            let body_digest = digest_parts(&[
                &s.definition,
                &s.description,
                &projects_sorted.join(","),
                &funcs
                    .iter()
                    .map(|(n, c)| format!("{n}@{c}"))
                    .collect::<Vec<_>>()
                    .join(","),
            ]);
            let fp = digest_parts(&[&s.name, &s.category.to_string(), &body_digest]);
            SemanticFp {
                fp,
                name: s.name.clone(),
                category: s.category.to_string(),
                body_digest,
                project_count: projects.len(),
                function_count: funcs.len(),
                folded: folded_sems.contains(&s.id),
            }
        })
        .collect();
    semantics.sort();

    // ── findings ──
    let mut finding_cats: BTreeMap<i32, Vec<String>> = BTreeMap::new();
    let cat_names: BTreeMap<i32, String> = graph
        .finding_categories
        .iter()
        .map(|c| (c.id, c.name.clone()))
        .collect();
    for afc in &graph.audit_finding_categories {
        if let Some(name) = cat_names.get(&afc.finding_category_id) {
            finding_cats
                .entry(afc.audit_finding_id)
                .or_default()
                .push(name.clone());
        }
    }

    let mut findings: Vec<FindingFp> = graph
        .findings
        .iter()
        .map(|f| {
            let projects = find_projects.get(&f.id).cloned().unwrap_or_default();
            let projects_sorted: Vec<String> = projects.iter().cloned().collect();
            let mut cats = finding_cats.get(&f.id).cloned().unwrap_or_default();
            cats.sort();
            let body_digest = digest_parts(&[
                &f.root_cause,
                &f.description,
                &f.patterns,
                &f.exploits,
                &projects_sorted.join(","),
                &cats.join(","),
            ]);
            let fp = digest_parts(&[&f.title, &f.severity.to_string(), &body_digest]);
            FindingFp {
                fp,
                title: f.title.clone(),
                severity: f.severity.to_string(),
                body_digest,
                project_count: projects.len(),
                folded: folded_finds.contains(&f.id),
            }
        })
        .collect();
    findings.sort();

    // ── links ──
    let mut links: Vec<LinkFp> = graph
        .semantic_finding_links
        .iter()
        .filter_map(|l| {
            let s = sem_by_id.get(&l.semantic_node_id)?;
            let f = graph.findings.iter().find(|f| f.id == l.audit_finding_id)?;
            let (_, s_body) = semantic_body_parts(s, &sem_projects, &functions_by_sem);
            let s_fp = digest_parts(&[&s.name, &s.category.to_string(), &s_body]);
            let f_body = finding_body_parts(f, &find_projects, &finding_cats);
            let f_fp = digest_parts(&[&f.title, &f.severity.to_string(), &f_body]);
            Some(LinkFp {
                semantic_fp: s_fp,
                finding_fp: f_fp,
                strength: l.strength.to_string(),
                evidence_digest: sha256_str(&l.evidence),
            })
        })
        .collect();
    links.sort();

    // ── merges ──
    let mut semantic_merges: Vec<MergeFp> = graph
        .semantic_merges
        .iter()
        .filter_map(|m| {
            let from = sem_by_id.get(&m.from_semantic_id)?;
            let to = sem_by_id.get(&m.to_semantic_id)?;
            let (_, from_body) = semantic_body_parts(from, &sem_projects, &functions_by_sem);
            let (_, to_body) = semantic_body_parts(to, &sem_projects, &functions_by_sem);
            Some(MergeFp {
                from_fp: digest_parts(&[&from.name, &from.category.to_string(), &from_body]),
                to_fp: digest_parts(&[&to.name, &to.category.to_string(), &to_body]),
                appended_digest: sha256_str(&m.appended_description),
            })
        })
        .collect();
    semantic_merges.sort();

    let findings_by_id: BTreeMap<i32, &audit_finding::Model> =
        graph.findings.iter().map(|f| (f.id, f)).collect();
    let mut finding_merges: Vec<MergeFp> = graph
        .finding_merges
        .iter()
        .filter_map(|m| {
            let from = findings_by_id.get(&m.from_finding_id)?;
            let to = findings_by_id.get(&m.to_finding_id)?;
            let f_body = finding_body_parts(from, &find_projects, &finding_cats);
            let t_body = finding_body_parts(to, &find_projects, &finding_cats);
            Some(MergeFp {
                from_fp: digest_parts(&[&from.title, &from.severity.to_string(), &f_body]),
                to_fp: digest_parts(&[&to.title, &to.severity.to_string(), &t_body]),
                appended_digest: sha256_str(&format!(
                    "{}\u{1f}{}\u{1f}{}",
                    m.appended_description, m.appended_patterns, m.appended_exploits
                )),
            })
        })
        .collect();
    finding_merges.sort();

    let mut projects: Vec<String> = graph
        .projects
        .iter()
        .map(|p| format!("{}\u{1f}{}", p.name, p.status))
        .collect();
    projects.sort();

    NormalizedGraph {
        semantics,
        findings,
        links,
        semantic_merges,
        finding_merges,
        projects,
    }
}

fn semantic_body_parts(
    s: &semantic_node::Model,
    sem_projects: &BTreeMap<i32, BTreeSet<String>>,
    functions_by_sem: &BTreeMap<i32, Vec<(String, String)>>,
) -> (String, String) {
    let projects = sem_projects.get(&s.id).cloned().unwrap_or_default();
    let projects_sorted: Vec<String> = projects.iter().cloned().collect();
    let mut funcs = functions_by_sem.get(&s.id).cloned().unwrap_or_default();
    funcs.sort();
    (
        projects_sorted.join(","),
        digest_parts(&[
            &s.definition,
            &s.description,
            &projects_sorted.join(","),
            &funcs
                .iter()
                .map(|(n, c)| format!("{n}@{c}"))
                .collect::<Vec<_>>()
                .join(","),
        ]),
    )
}

fn finding_body_parts(
    f: &audit_finding::Model,
    find_projects: &BTreeMap<i32, BTreeSet<String>>,
    finding_cats: &BTreeMap<i32, Vec<String>>,
) -> String {
    let projects = find_projects.get(&f.id).cloned().unwrap_or_default();
    let projects_sorted: Vec<String> = projects.iter().cloned().collect();
    let mut cats = finding_cats.get(&f.id).cloned().unwrap_or_default();
    cats.sort();
    digest_parts(&[
        &f.root_cause,
        &f.description,
        &f.patterns,
        &f.exploits,
        &projects_sorted.join(","),
        &cats.join(","),
    ])
}

/// Diff two normalized graphs in fingerprint space.
pub fn diff(before: &NormalizedGraph, after: &NormalizedGraph) -> GraphDelta {
    let b_sems: BTreeSet<&SemanticFp> = before.semantics.iter().collect();
    let a_sems: BTreeSet<&SemanticFp> = after.semantics.iter().collect();
    let b_finds: BTreeSet<&FindingFp> = before.findings.iter().collect();
    let a_finds: BTreeSet<&FindingFp> = after.findings.iter().collect();
    let b_links: BTreeSet<&LinkFp> = before.links.iter().collect();
    let a_links: BTreeSet<&LinkFp> = after.links.iter().collect();
    let b_sm: BTreeSet<&MergeFp> = before.semantic_merges.iter().collect();
    let a_sm: BTreeSet<&MergeFp> = after.semantic_merges.iter().collect();
    let b_fm: BTreeSet<&MergeFp> = before.finding_merges.iter().collect();
    let a_fm: BTreeSet<&MergeFp> = after.finding_merges.iter().collect();
    let b_proj: BTreeSet<&String> = before.projects.iter().collect();
    let a_proj: BTreeSet<&String> = after.projects.iter().collect();

    GraphDelta {
        added_semantics: a_sems.difference(&b_sems).cloned().cloned().collect(),
        removed_semantics: b_sems.difference(&a_sems).cloned().cloned().collect(),
        added_findings: a_finds.difference(&b_finds).cloned().cloned().collect(),
        removed_findings: b_finds.difference(&a_finds).cloned().cloned().collect(),
        added_links: a_links.difference(&b_links).cloned().cloned().collect(),
        removed_links: b_links.difference(&a_links).cloned().cloned().collect(),
        added_semantic_merges: a_sm.difference(&b_sm).cloned().cloned().collect(),
        removed_semantic_merges: b_sm.difference(&a_sm).cloned().cloned().collect(),
        added_finding_merges: a_fm.difference(&b_fm).cloned().cloned().collect(),
        removed_finding_merges: b_fm.difference(&a_fm).cloned().cloned().collect(),
        added_projects: a_proj.difference(&b_proj).cloned().cloned().collect(),
        removed_projects: b_proj.difference(&a_proj).cloned().cloned().collect(),
    }
}

/// Compute count-level invariant metrics from one graph.
pub fn metrics(graph: &KnowledgeGraph) -> GraphMetrics {
    let folded_sems: BTreeSet<i32> = graph
        .semantic_merges
        .iter()
        .map(|m| m.from_semantic_id)
        .collect();
    let folded_finds: BTreeSet<i32> = graph
        .finding_merges
        .iter()
        .map(|m| m.from_finding_id)
        .collect();
    let sem_ids: BTreeSet<i32> = graph.nodes.iter().map(|s| s.id).collect();
    let find_ids: BTreeSet<i32> = graph.findings.iter().map(|f| f.id).collect();
    let linked_status: BTreeSet<i32> = graph.finding_link_statuses.iter().map(|s| s.audit_finding_id).collect();

    let mut out = GraphMetrics {
        semantic_total: graph.nodes.len(),
        semantic_canonical: graph.nodes.len() - folded_sems.len(),
        semantic_folded: folded_sems.len(),
        finding_total: graph.findings.len(),
        finding_canonical: graph.findings.len() - folded_finds.len(),
        finding_folded: folded_finds.len(),
        link_total: graph.semantic_finding_links.len(),
        semantic_merge_edges: graph.semantic_merges.len(),
        finding_merge_edges: graph.finding_merges.len(),
        ..GraphMetrics::default()
    };

    for link in &graph.semantic_finding_links {
        if !linked_status.contains(&link.audit_finding_id) {
            out.partial_links += 1;
        }
        if folded_sems.contains(&link.semantic_node_id)
            || folded_finds.contains(&link.audit_finding_id)
        {
            out.links_on_folded_endpoints += 1;
        }
    }
    out.dangling_merge_targets = graph
        .semantic_merges
        .iter()
        .filter(|m| !sem_ids.contains(&m.to_semantic_id))
        .count()
        + graph
            .finding_merges
            .iter()
            .filter(|m| !find_ids.contains(&m.to_finding_id))
            .count();

    let linked_sems: BTreeSet<i32> = graph
        .semantic_finding_links
        .iter()
        .map(|l| l.semantic_node_id)
        .collect();
    out.unlinked_canonical_semantics = graph
        .nodes
        .iter()
        .filter(|s| !folded_sems.contains(&s.id) && !linked_sems.contains(&s.id))
        .count();

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use knowdit_kg_model::category::DeFiCategory;
    use knowdit_kg_model::db::{
        audit_finding, project, project_finding, project_semantic, semantic_finding_link,
        semantic_function, semantic_merge, semantic_node,
    };
    use knowdit_kg_model::audit_finding::FindingSeverity;
    use knowdit_kg_model::link_strength::LinkStrength;
    fn base_graph() -> KnowledgeGraph {
        KnowledgeGraph {
            projects: vec![project::Model {
                id: 1,
                name: "p1".to_string(),
                status: "completed".to_string(),
            }],
            project_platforms: vec![],
            categories: vec![],
            nodes: vec![semantic_node::Model {
                id: 10,
                name: "AMM Swap".to_string(),
                definition: "def".to_string(),
                description: "desc".to_string(),
                category: DeFiCategory::Dexes,
            }],
            semantic_functions: vec![semantic_function::Model {
                id: 1,
                semantic_node_id: 10,
                function_name: "swap".to_string(),
                contract_path: "src/Pool.sol".to_string(),
            }],
            project_categories: vec![],
            project_semantics: vec![project_semantic::Model {
                project_id: 1,
                semantic_node_id: 10,
            }],
            semantic_merges: vec![],
            findings: vec![],
            finding_categories: vec![],
            audit_finding_categories: vec![],
            project_findings: vec![],
            semantic_finding_links: vec![],
            finding_link_statuses: vec![],
            finding_merges: vec![],
        }
    }

    #[test]
    fn normalize_is_id_independent_and_sorted() {
        let g = base_graph();
        let n = normalize(&g);
        assert_eq!(n.semantics.len(), 1);
        assert_eq!(n.semantics[0].name, "AMM Swap");
        assert_eq!(n.semantics[0].project_count, 1);
        assert_eq!(n.semantics[0].function_count, 1);
        assert!(!n.semantics[0].fp.is_empty());
    }

    #[test]
    fn diff_is_empty_for_identical_graphs() {
        let n1 = normalize(&base_graph());
        let n2 = normalize(&base_graph());
        assert!(diff(&n1, &n2).is_empty());
    }

    #[test]
    fn diff_detects_added_nodes_and_links() {
        let before = normalize(&base_graph());

        let mut g = base_graph();
        g.findings.push(audit_finding::Model {
            id: 20,
            title: "Reentrancy".to_string(),
            severity: FindingSeverity::High,
            root_cause: "r".to_string(),
            description: "d".to_string(),
            patterns: "p".to_string(),
            exploits: "e".to_string(),
        });
        g.project_findings.push(project_finding::Model {
            project_id: 1,
            audit_finding_id: 20,
        });
        g.semantic_finding_links.push(semantic_finding_link::Model {
            semantic_node_id: 10,
            audit_finding_id: 20,
            strength: LinkStrength::High,
            evidence: "in-project link: finding and semantic co-emitted by the extract pipeline"
                .to_string(),
        });
        let after = normalize(&g);

        let delta = diff(&before, &after);
        assert_eq!(delta.added_findings.len(), 1);
        assert_eq!(delta.added_links.len(), 1);
        assert!(delta.removed_findings.is_empty());
        assert!(delta.removed_links.is_empty());
    }

    #[test]
    fn merge_edges_survive_renumbering() {
        // Build graph A with small IDs and graph B with large IDs but
        // identical content; the diff must be empty even though every
        // integer id differs.
        let mut a = base_graph();
        a.semantic_merges.push(semantic_merge::Model {
            from_semantic_id: 9,
            to_semantic_id: 10,
            appended_description: "extends".to_string(),
        });
        a.nodes.push(semantic_node::Model {
            id: 9,
            name: "Raw".to_string(),
            definition: "def".to_string(),
            description: "desc".to_string(),
            category: DeFiCategory::Dexes,
        });

        let mut b = base_graph();
        b.nodes.iter_mut().for_each(|n| {
            n.id += 100;
        });
        b.semantic_functions.iter_mut().for_each(|f| {
            f.id += 100;
            f.semantic_node_id += 100;
        });
        b.project_semantics.iter_mut().for_each(|ps| ps.semantic_node_id += 100);
        b.semantic_merges.push(semantic_merge::Model {
            from_semantic_id: 109,
            to_semantic_id: 110,
            appended_description: "extends".to_string(),
        });
        b.nodes.push(semantic_node::Model {
            id: 109,
            name: "Raw".to_string(),
            definition: "def".to_string(),
            description: "desc".to_string(),
            category: DeFiCategory::Dexes,
        });

        let na = normalize(&a);
        let nb = normalize(&b);
        assert!(diff(&na, &nb).is_empty());
    }

    #[test]
    fn metrics_count_partial_links_and_dangling_targets() {
        let mut g = base_graph();
        g.findings.push(audit_finding::Model {
            id: 20,
            title: "F".to_string(),
            severity: FindingSeverity::High,
            root_cause: "r".to_string(),
            description: "d".to_string(),
            patterns: "p".to_string(),
            exploits: "e".to_string(),
        });
        g.semantic_finding_links.push(semantic_finding_link::Model {
            semantic_node_id: 10,
            audit_finding_id: 20,
            strength: LinkStrength::Low,
            evidence: "in-project link: finding and semantic co-emitted by the extract pipeline"
                .to_string(),
        });
        g.semantic_merges.push(semantic_merge::Model {
            from_semantic_id: 5,
            to_semantic_id: 999,
            appended_description: String::new(),
        });
        let m = metrics(&g);
        assert_eq!(m.link_total, 1);
        assert_eq!(m.partial_links, 1);
        assert_eq!(m.dangling_merge_targets, 1);
        assert_eq!(m.finding_merge_edges, 0);
    }
}
