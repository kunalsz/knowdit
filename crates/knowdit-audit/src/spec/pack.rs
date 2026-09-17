//! Pair-mode hint-pack selection.
//!
//! One `(extract, historical)` pair can carry dozens of linked findings, most
//! of which are per-project restatements of the same mechanism. Pair-mode
//! gen-specs runs **one** agent per pair, so the pack it sees must cover the
//! mechanism space instead of padded restatements:
//!
//! 1. Order by edge strength (`link_strength` desc, `finding_id` asc) — the
//!    strongest edge becomes the anchor, which is also every committed spec's
//!    attribution fallback.
//! 2. Drop exact-title duplicates (normalized), keeping the strongest.
//! 3. Fill the remaining slots greedily with the member that maximizes the
//!    minimum token-Jaccard distance to the already-selected set (MMR), so
//!    each added hint describes a *different* root-cause flavor rather than a
//!    paraphrase of an earlier one.
//!
//! Deterministic: identical inputs always yield an identical pack, so resume
//! and cache keys stay stable across runs.

use std::collections::HashSet;

use knowdit_kg_model::link_strength::LinkStrength;
use sha2::{Digest, Sha256};

/// Light per-member view the selector needs: identity, edge strength, and the
/// text similarity is computed over. Rendered prompt strings are attached
/// later at materialization time.
#[derive(Debug, Clone)]
pub(crate) struct PackMember {
    pub finding_id: i32,
    pub link_strength: LinkStrength,
    pub title: String,
    pub root_cause: String,
    pub description: String,
}

/// Result of pack selection over one pair's eligible members.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PackSelection {
    /// Selected member ids in final pack order (anchor first).
    pub members: Vec<i32>,
    /// Members dropped as exact-title duplicates of an earlier (stronger)
    /// member, for logging.
    pub deduped: Vec<i32>,
}

/// Select up to `pack_size` members. Never fails: with fewer eligible members
/// than `pack_size` the whole (deduped, strength-ordered) set is returned.
pub(crate) fn select_pack(members: &[PackMember], pack_size: usize) -> PackSelection {
    assert!(pack_size >= 1, "pack_size must be >= 1");
    // 1. Strength order; finding_id breaks ties so the result is stable.
    let mut ordered: Vec<&PackMember> = members.iter().collect();
    ordered.sort_by(|a, b| {
        b.link_strength
            .rank()
            .cmp(&a.link_strength.rank())
            .then(a.finding_id.cmp(&b.finding_id))
    });

    // 2. Exact-title dedup: keep the first (strongest) occurrence.
    let mut seen_titles: HashSet<String> = HashSet::new();
    let mut deduped: Vec<i32> = Vec::new();
    let mut unique: Vec<&PackMember> = Vec::with_capacity(ordered.len());
    for member in ordered {
        let key = normalize_title(&member.title);
        if !seen_titles.insert(key) {
            deduped.push(member.finding_id);
            continue;
        }
        unique.push(member);
    }

    if unique.is_empty() {
        return PackSelection {
            members: Vec::new(),
            deduped,
        };
    }

    // 3. Anchor + greedy max-min-diversity fill.
    let anchor = unique[0];
    let mut selected: Vec<&PackMember> = vec![anchor];
    let mut selected_tokens: Vec<HashSet<String>> = vec![token_set(anchor)];
    let mut pool: Vec<&PackMember> = unique[1..].to_vec();

    while selected.len() < pack_size && !pool.is_empty() {
        let (best_idx, _) = pool
            .iter()
            .enumerate()
            .map(|(idx, candidate)| {
                let tokens = token_set(candidate);
                // Max-min distance: the candidate's distance to its closest
                // already-selected member. Pool order (strength desc, id asc)
                // breaks ties via strict `>`.
                let min_distance = selected_tokens
                    .iter()
                    .map(|sel| jaccard_distance(&tokens, sel))
                    .fold(f64::INFINITY, f64::min);
                (idx, min_distance)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("pool is non-empty");
        let picked = pool.remove(best_idx);
        selected_tokens.push(token_set(picked));
        selected.push(picked);
    }

    PackSelection {
        members: selected.iter().map(|m| m.finding_id).collect(),
        deduped,
    }
}

/// Stable content fingerprint of a pack, used in the agent's llmy cache key so
/// changing `--gen-specs-pack-size` (or the underlying edges) cannot replay a
/// cached run against a different prompt. Hex-encoded SHA-256 of the sorted
/// member ids.
pub(crate) fn pack_fingerprint(member_ids: &[i32]) -> String {
    let mut sorted: Vec<i32> = member_ids.to_vec();
    sorted.sort_unstable();
    let digest = Sha256::digest(
        sorted
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    hex[..16].to_string()
}

/// Lowercase, collapse whitespace, strip punctuation — enough to catch the
/// "same report title pasted twice" duplicates the corpus actually contains.
fn normalize_title(title: &str) -> String {
    title
        .split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Lowercase alphanumeric runs of length >= 3 over title + root cause +
/// description. Deliberately close to the linker's own `link_tokens` so pack
/// diversity tracks the same notion of "same mechanism" the KG uses.
fn token_set(member: &PackMember) -> HashSet<String> {
    let mut tokens = HashSet::new();
    let mut push = |text: &str| {
        for word in text.split(|c: char| !c.is_alphanumeric()) {
            let word = word.to_lowercase();
            if word.len() >= 3 {
                tokens.insert(word);
            }
        }
    };
    push(&member.title);
    push(&member.root_cause);
    push(&member.description);
    tokens
}

fn jaccard_distance(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        // Two textless members are identical for diversity purposes.
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        return 0.0;
    }
    1.0 - (intersection as f64 / union as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn member(id: i32, strength: LinkStrength, title: &str, root_cause: &str) -> PackMember {
        PackMember {
            finding_id: id,
            link_strength: strength,
            title: title.to_string(),
            root_cause: root_cause.to_string(),
            description: String::new(),
        }
    }

    #[test]
    fn strongest_edge_becomes_anchor() {
        let members = vec![
            member(7, LinkStrength::Medium, "b title", "rc"),
            member(3, LinkStrength::High, "a title", "rc"),
        ];
        let pack = select_pack(&members, 10);
        assert_eq!(pack.members, vec![3, 7]);
        assert!(pack.deduped.is_empty());
    }

    #[test]
    fn exact_title_duplicates_are_dropped_keeping_the_strongest() {
        let members = vec![
            member(11, LinkStrength::Medium, "Same Bug!", "rc"),
            member(4, LinkStrength::High, "Same Bug!", "rc"),
            member(5, LinkStrength::High, "Same  bug?", "other"),
        ];
        let pack = select_pack(&members, 10);
        // f11 deduped against the stronger f4; "Same Bug!" and "Same bug?"
        // normalize to the same title, so f5 is dropped too. Dedup order
        // follows the strength sort (f5 before f11).
        assert_eq!(pack.members, vec![4]);
        assert_eq!(pack.deduped, vec![5, 11]);
    }

    #[test]
    fn mmr_prefers_distinct_mechanisms_over_paraphrases() {
        let flash_a = "flash loan manipulates reward price per share";
        let flash_b = "flash loan manipulates the reward price per share state";
        let oracle = "oracle spot price feeds collateral valuation from dex reserves";
        let members = vec![
            member(1, LinkStrength::High, "flash one", flash_a),
            member(2, LinkStrength::High, "flash two", flash_b),
            member(3, LinkStrength::Medium, "oracle one", oracle),
        ];
        let pack = select_pack(&members, 3);
        // Anchor is f1; the second slot must go to the oracle mechanism even
        // though f2 has the same (High) strength — it is a near-paraphrase.
        assert_eq!(pack.members.first(), Some(&1));
        assert_eq!(pack.members.get(1), Some(&3));
        assert_eq!(pack.members.len(), 3);
    }

    #[test]
    fn pack_size_caps_and_small_pools_behave() {
        let members: Vec<PackMember> = (0..5)
            .map(|i| member(i, LinkStrength::Medium, &format!("title {i}"), &format!("rc {i}")))
            .collect();
        assert_eq!(select_pack(&members, 2).members.len(), 2);
        // K >= pool: everything, still strength-ordered.
        assert_eq!(select_pack(&members, 10).members.len(), 5);
        // Single member: degenerates to the anchor only.
        let single = vec![member(9, LinkStrength::High, "t", "r")];
        assert_eq!(select_pack(&single, 10).members, vec![9]);
    }

    #[test]
    fn selection_is_deterministic() {
        let members: Vec<PackMember> = (0..12)
            .map(|i| {
                member(
                    i,
                    if i % 3 == 0 {
                        LinkStrength::High
                    } else {
                        LinkStrength::Medium
                    },
                    &format!("title {i} {}", "x".repeat((i % 5) as usize)),
                    &format!("root cause {i} {}", "y".repeat((i % 4) as usize)),
                )
            })
            .collect();
        let a = select_pack(&members, 6);
        let b = select_pack(&members, 6);
        assert_eq!(a, b);
    }

    #[test]
    fn fingerprint_is_stable_and_order_insensitive() {
        let a = pack_fingerprint(&[3, 1, 2]);
        let b = pack_fingerprint(&[2, 3, 1]);
        assert_eq!(a, b);
        assert_ne!(a, pack_fingerprint(&[1, 2, 4]));
        assert_eq!(a.len(), 16);
    }
}
