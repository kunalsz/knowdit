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

use std::collections::BTreeSet;

use knowdit_kg_model::audit_finding::FindingSeverity;
use knowdit_kg_model::link_strength::LinkStrength;
use sha2::{Digest, Sha256};

/// Light per-member view the selector needs: identity, edge strength, finding
/// severity, and the text similarity is computed over. Rendered prompt strings
/// are attached later at materialization time.
#[derive(Debug, Clone)]
pub(crate) struct PackMember {
    pub finding_id: i32,
    pub link_strength: LinkStrength,
    pub severity: FindingSeverity,
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

impl PackMember {
    /// Deterministic intra-`(extract_id, historical_id)` ordering for `finding`
    /// granularity under `SpecSchedule::Fair`:
    /// 1. Sort by `(link_strength.rank() DESC, severity.rank() DESC, finding_id ASC)`.
    /// 2. Partition into distinct normalized titles (`unique`) and exact-title
    ///    duplicates (`duplicates`).
    /// 3. Greedily order `unique` from index `0` (anchor) by max-min token-Jaccard
    ///    distance over `>= 3`-char alphanumeric runs of `title + root_cause + description`.
    /// 4. Append `duplicates` at the tail rather than dropping them.
    pub(crate) fn order_for_finding_mode(members: &[Self]) -> Vec<i32> {
        if members.len() <= 1 {
            return members.iter().map(|m| m.finding_id).collect();
        }
        let (unique, duplicates) = Self::partition_by_title(members);
        let mut out = Self::mmr_select(&unique, unique.len());
        out.extend(duplicates);
        out
    }

    fn partition_by_title(members: &[Self]) -> (Vec<&Self>, Vec<i32>) {
        let mut ordered: Vec<&Self> = members.iter().collect();
        ordered.sort_by(|a, b| {
            b.link_strength
                .rank()
                .cmp(&a.link_strength.rank())
                .then(b.severity.rank().cmp(&a.severity.rank()))
                .then(a.finding_id.cmp(&b.finding_id))
        });

        let mut seen_titles: BTreeSet<String> = BTreeSet::new();
        let mut deduped: Vec<i32> = Vec::new();
        let mut unique: Vec<&Self> = Vec::with_capacity(ordered.len());
        for member in ordered {
            let key = member.normalized_title();
            if !key.is_empty() && !seen_titles.insert(key) {
                deduped.push(member.finding_id);
                continue;
            }
            unique.push(member);
        }
        (unique, deduped)
    }

    fn mmr_select(unique: &[&Self], limit: usize) -> Vec<i32> {
        if unique.is_empty() || limit == 0 {
            return Vec::new();
        }
        let target_len = limit.min(unique.len());
        let anchor = unique[0];
        let anchor_tokens = anchor.token_set();
        let mut selected: Vec<i32> = Vec::with_capacity(target_len);
        selected.push(anchor.finding_id);
        if target_len == 1 {
            return selected;
        }

        let mut pool: Vec<(i32, BTreeSet<String>, f64)> = unique[1..]
            .iter()
            .map(|candidate| {
                let tokens = candidate.token_set();
                let dist = Self::jaccard_distance(&tokens, &anchor_tokens);
                (candidate.finding_id, tokens, dist)
            })
            .collect();

        while selected.len() < target_len && !pool.is_empty() {
            // Strict `>` on distance preserves pool order `(strength DESC, severity DESC, id ASC)` on ties.
            let mut best_idx = 0usize;
            for idx in 1..pool.len() {
                if pool[idx].2 > pool[best_idx].2 {
                    best_idx = idx;
                }
            }
            let (picked_id, picked_tokens, _) = pool.remove(best_idx);
            selected.push(picked_id);
            if selected.len() < target_len {
                for entry in &mut pool {
                    let dist = Self::jaccard_distance(&entry.1, &picked_tokens);
                    if dist < entry.2 {
                        entry.2 = dist;
                    }
                }
            }
        }
        selected
    }

    /// Lowercase, collapse whitespace, strip punctuation — enough to catch the
    /// "same report title pasted twice" duplicates the corpus actually contains.
    fn normalized_title(&self) -> String {
        self.title
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
    fn token_set(&self) -> BTreeSet<String> {
        let mut tokens = BTreeSet::new();
        let mut push = |text: &str| {
            for word in text.split(|c: char| !c.is_alphanumeric()) {
                let word = word.to_lowercase();
                if word.len() >= 3 {
                    tokens.insert(word);
                }
            }
        };
        push(&self.title);
        push(&self.root_cause);
        push(&self.description);
        tokens
    }

    fn jaccard_distance(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 0.0;
        }
        let intersection = a.intersection(b).count();
        let union = a.len() + b.len() - intersection;
        if union == 0 {
            return 0.0;
        }
        1.0 - (intersection as f64 / union as f64)
    }
}

/// Select up to `pack_size` members. Never fails: with fewer eligible members
/// than `pack_size` the whole (deduped, strength+severity-ordered) set is returned.
pub(crate) fn select_pack(members: &[PackMember], pack_size: usize) -> PackSelection {
    assert!(pack_size >= 1, "pack_size must be >= 1");
    let (unique, deduped) = PackMember::partition_by_title(members);
    let selected = PackMember::mmr_select(&unique, pack_size);
    PackSelection {
        members: selected,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn member(id: i32, strength: LinkStrength, title: &str, root_cause: &str) -> PackMember {
        PackMember {
            finding_id: id,
            link_strength: strength,
            severity: FindingSeverity::Medium,
            title: title.to_string(),
            root_cause: root_cause.to_string(),
            description: String::new(),
        }
    }

    #[test]
    fn finding_mode_orders_by_severity_then_mmr_and_keeps_duplicates_at_tail() {
        let mut m_low = member(
            1,
            LinkStrength::High,
            "low rounding issue",
            "dust rounding on deposit shares",
        );
        m_low.severity = FindingSeverity::Low;
        let mut m_high_a = member(
            2,
            LinkStrength::High,
            "Flash Share Inflation",
            "flash loan inflates share price via direct donation",
        );
        m_high_a.severity = FindingSeverity::High;
        let mut m_high_dup = member(
            3,
            LinkStrength::High,
            "flash  share inflation!",
            "flash loan inflates share price via direct donation",
        );
        m_high_dup.severity = FindingSeverity::High;
        let mut m_med_distinct = member(
            4,
            LinkStrength::High,
            "Oracle Stale Heartbeat",
            "chainlink aggregator timestamp staleness bypasses liquidation",
        );
        m_med_distinct.severity = FindingSeverity::Medium;

        let ordered =
            PackMember::order_for_finding_mode(&[m_low, m_high_a, m_high_dup, m_med_distinct]);
        // Anchor = High severity (2); next = distinct mechanism (4) over low rounding,
        // and exact normalized-title duplicate (3) is placed at the tail rather than dropped.
        assert_eq!(ordered.first(), Some(&2));
        assert_eq!(ordered.last(), Some(&3));
        assert_eq!(ordered.len(), 4);
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
