//! Novelty gate: LLM-judged project-wide semantic dedup between spec
//! generation and validation.
//!
//! Per-finding exploration breadth is load-bearing — one pass per
//! `(extract, historical)` pair demonstrably under-collects — but on a real
//! run 656 valid harness results collapsed to 38 unique findings: ~95% of
//! validation spend (codegen + forge + reflect) re-confirmed an
//! already-validated mechanism. This gate sits between the two phases.
//! Every link still runs its spec-gen agent; before a committed spec enters
//! the drain, we select up to [`MAX_PRIOR_SPECS`] canonical validated specs
//! across the **entire project** ranked by grounded `(sequence ∪ post_attack)`
//! Jaccard similarity, and an LLM judge compares the new batch against them.
//! Duplicates are recorded as terminal [`CodeGenStatus::Duplicate`] rows —
//! the DB keeps the full exploration record, resume stays correct, and
//! validation spend goes only to genuinely new specs.
//!
//! Fail-open by design: any judge error (bad JSON, provider hiccup) marks
//! every spec novel, so a bad judge day costs tokens, never discoveries.
//! Only a billing-cap exhaustion propagates as fatal.

use std::collections::{BTreeMap, BTreeSet};

use color_eyre::eyre::Result;
use knowdit_repo_model::TerminalSpecification;
use serde::Deserialize;

use crate::types::AuditSpecification;

use super::is_billing_exhausted;

/// Rendered-field caps for the judge prompt. Generous enough to keep the
/// discriminating detail (root-cause function, violated invariant); tight
/// enough that 12 prior + 4 new specs stay a few KB.
const SUMMARY_CAP: usize = 280;
const POST_ATTACK_CAP: usize = 600;
const INVARIANT_CAP: usize = 120;
const SEQUENCE_STEP_CAP: usize = 60;
const MAX_SEQUENCE_STEPS: usize = 6;

/// Maximum number of already-validated project specs shown to the novelty judge.
pub const MAX_PRIOR_SPECS: usize = 12;

/// Normalized grounded footprint of one [`AuditSpecification`]:
/// `{ "fn:<contract>.<function>" } ∪ { "var:<contract>.<field>" } ∪ { "contract:<contract>" }`
/// derived from `sequence` and `post_attack`, excluding declared auxiliary
/// contracts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecFootprint {
    pub tokens: BTreeSet<String>,
}

impl SpecFootprint {
    pub fn from_spec(spec: &AuditSpecification) -> Self {
        let aux_names: BTreeSet<String> = spec
            .auxiliary_contracts
            .keys()
            .map(|k| k.trim().to_lowercase())
            .collect();

        let mut tokens = BTreeSet::new();
        for step in &spec.sequence {
            let contract = step.contract.trim().to_lowercase();
            let function = step.function.trim().to_lowercase();
            if !contract.is_empty() && !function.is_empty() && !aux_names.contains(&contract) {
                tokens.insert(format!("fn:{contract}.{function}"));
            }
        }
        for key in spec.post_attack.states_variables.keys() {
            let trimmed = key.trim().to_lowercase();
            if trimmed.is_empty() {
                continue;
            }
            if let Some((contract, field)) = trimmed.split_once('.') {
                let contract = contract.trim();
                let field = field.trim();
                if !contract.is_empty() && !field.is_empty() && !aux_names.contains(contract) {
                    tokens.insert(format!("var:{contract}.{field}"));
                }
            } else {
                tokens.insert(format!("var:{trimmed}"));
            }
        }
        for contract in spec.post_attack.contracts.keys() {
            let contract = contract.trim().to_lowercase();
            if !contract.is_empty() && !aux_names.contains(&contract) {
                tokens.insert(format!("contract:{contract}"));
            }
        }
        Self { tokens }
    }

    pub fn jaccard_similarity(&self, other: &Self) -> f64 {
        if self.tokens.is_empty() || other.tokens.is_empty() {
            return 0.0;
        }
        let intersection = self.tokens.intersection(&other.tokens).count();
        if intersection == 0 {
            return 0.0;
        }
        let union = self.tokens.len() + other.tokens.len() - intersection;
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }
}

/// Select up to `max_prior` canonical terminal specifications across the entire
/// repository to compare against `new_specs` for `pair = (extract_id, historical_id)`.
///
/// Retains candidates that share `>= 1` footprint token with at least one spec in
/// `new_specs` OR belong to the same `(extract_id, historical_id)` pair, ranked by
/// `(max_jaccard DESC, is_valid_finding DESC, same_pair DESC, spec_id DESC)`.
pub fn select_prior_specs_for_batch(
    pair: (i32, i32),
    new_specs: &[(i32, AuditSpecification)],
    terminal_specs: &[TerminalSpecification],
    max_prior: usize,
) -> Vec<(i32, AuditSpecification)> {
    if new_specs.is_empty() || terminal_specs.is_empty() || max_prior == 0 {
        return Vec::new();
    }
    let new_ids: BTreeSet<i32> = new_specs.iter().map(|(id, _)| *id).collect();
    let new_footprints: Vec<SpecFootprint> = new_specs
        .iter()
        .map(|(_, spec)| SpecFootprint::from_spec(spec))
        .collect();

    struct RankedCandidate {
        id: i32,
        spec: AuditSpecification,
        max_jaccard: f64,
        is_valid_finding: bool,
        same_pair: bool,
    }

    let mut candidates: Vec<RankedCandidate> = Vec::new();
    for term in terminal_specs {
        if new_ids.contains(&term.spec.id) {
            continue;
        }
        let Ok(parsed) = serde_json::from_str::<AuditSpecification>(&term.spec.specification_json)
        else {
            continue;
        };
        let fp = SpecFootprint::from_spec(&parsed);
        let max_jaccard = new_footprints
            .iter()
            .map(|nfp| nfp.jaccard_similarity(&fp))
            .fold(0.0_f64, f64::max);
        let same_pair = (term.spec.semantic_id, term.spec.historical_id) == pair;
        if max_jaccard <= 0.0 && !same_pair {
            continue;
        }
        candidates.push(RankedCandidate {
            id: term.spec.id,
            spec: parsed,
            max_jaccard,
            is_valid_finding: term.is_valid_finding,
            same_pair,
        });
    }

    candidates.sort_by(|a, b| {
        b.max_jaccard
            .total_cmp(&a.max_jaccard)
            .then(b.is_valid_finding.cmp(&a.is_valid_finding))
            .then(b.same_pair.cmp(&a.same_pair))
            .then(b.id.cmp(&a.id))
    });
    candidates
        .into_iter()
        .take(max_prior)
        .map(|c| (c.id, c.spec))
        .collect()
}

/// One judge verdict per new spec, aligned with the input order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoveltyVerdict {
    /// `Some(prior_spec_id)` (or an earlier new-spec id, for intra-batch
    /// duplicates) when judged a semantic duplicate; `None` = novel.
    pub duplicate_of: Option<i32>,
    /// Free-text rationale from the judge (or the fail-open reason).
    pub because: String,
}

impl NoveltyVerdict {
    fn novel() -> Self {
        Self {
            duplicate_of: None,
            because: String::new(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct NoveltyResponse {
    #[serde(default)]
    verdicts: Vec<RawVerdict>,
}

#[derive(Debug, Deserialize)]
struct RawVerdict {
    new_spec_id: i32,
    #[serde(default)]
    duplicate_of: Option<i32>,
    #[serde(default)]
    because: Option<String>,
}

impl NoveltyVerdict {
    /// Bounded, judge-facing view of one spec.
    fn render_spec(label: &str, id: i32, spec: &AuditSpecification) -> String {
        let mut out = format!("{label} id={id}\n");
        out.push_str(&format!("- summary: {}\n", Self::bounded(&spec.summary, SUMMARY_CAP)));
        out.push_str(&format!(
            "- post_attack: {}\n",
            Self::bounded(&spec.post_attack.description, POST_ATTACK_CAP)
        ));
        let mut invariants: Vec<String> = spec
            .post_attack
            .states_variables
            .iter()
            .map(|(key, constraint)| format!("{key}: {}", Self::bounded(constraint, INVARIANT_CAP)))
            .collect();
        invariants.extend(spec.post_attack.contracts.iter().map(|(key, constraint)| {
            format!("contract {key}: {}", Self::bounded(constraint, INVARIANT_CAP))
        }));
        if invariants.is_empty() {
            out.push_str("- post_attack invariants: (none)\n");
        } else {
            out.push_str("- post_attack invariants:\n");
            for invariant in invariants.iter().take(4) {
                out.push_str(&format!("    - {invariant}\n"));
            }
        }
        if !spec.sequence.is_empty() {
            out.push_str("- call sequence:");
            for step in spec.sequence.iter().take(MAX_SEQUENCE_STEPS) {
                out.push_str(&format!(
                    " {}.{}({}) ->",
                    step.contract,
                    step.function,
                    Self::bounded(&step.intention, SEQUENCE_STEP_CAP)
                ));
            }
            out.push('\n');
        }
        out
    }

    fn bounded(text: &str, cap: usize) -> String {
        let text = text.trim();
        if text.chars().count() <= cap {
            text.to_string()
        } else {
            let truncated: String = text.chars().take(cap).collect();
            format!("{truncated}…[truncated]")
        }
    }
}

/// Resolve raw judge output into per-new-spec verdicts. Pure so the target
/// validation rules are unit-testable:
/// * a missing/malformed verdict → novel (fail-open);
/// * `duplicate_of` must name a **prior** spec, or an **earlier** new spec —
///   a forward or self reference would mark two specs duplicates of each
///   other with nothing validated, so it is treated as novel;
/// * an unknown id → novel.
pub(crate) fn resolve_verdicts(
    new_ids: &[i32],
    prior_ids: &BTreeSet<i32>,
    response: Option<NoveltyResponse>,
) -> Vec<NoveltyVerdict> {
    let mut index: BTreeMap<i32, &RawVerdict> = BTreeMap::new();
    if let Some(response) = &response {
        for verdict in &response.verdicts {
            index.entry(verdict.new_spec_id).or_insert(verdict);
        }
    }
    let mut seen_new: BTreeSet<i32> = BTreeSet::new();
    new_ids
        .iter()
        .map(|id| {
            seen_new.insert(*id);
            match index.get(id).and_then(|v| v.duplicate_of) {
                Some(target)
                    if prior_ids.contains(&target)
                        || (seen_new.contains(&target) && target != *id) =>
                {
                    let because = index
                        .get(id)
                        .and_then(|v| v.because.clone())
                        .unwrap_or_else(|| "judge flagged a duplicate".to_string());
                    NoveltyVerdict {
                        duplicate_of: Some(target),
                        because,
                    }
                }
                _ => NoveltyVerdict::novel(),
            }
        })
        .collect()
}

/// Judge whether each new spec is semantically novel against already-validated
/// specs across the repository (plus intra-batch earlier specs when multiple
/// new specs are committed together). Returns one verdict per entry of
/// `new_specs`, in order. Fail-open on any non-billing error; billing
/// exhaustion propagates so the scheduler aborts the run against a dead cap.
pub async fn judge_spec_novelty(
    llm: &llmy::client::client::LLM,
    cache_key: Option<&str>,
    debug_prefix: Option<&str>,
    new_specs: &[(i32, AuditSpecification)],
    prior_specs: &[(i32, AuditSpecification)],
) -> Result<Vec<NoveltyVerdict>> {
    let all_novel = || {
        new_specs
            .iter()
            .map(|(_, _)| NoveltyVerdict::novel())
            .collect::<Vec<_>>()
    };
    if new_specs.is_empty() {
        return Ok(Vec::new());
    }
    if prior_specs.is_empty() && new_specs.len() < 2 {
        // First validated spec and singleton batch: nothing to duplicate against.
        return Ok(all_novel());
    }

    let system_prompt = r#"You are the novelty gate of a knowledge-driven smart-contract auditing pipeline. Multiple spec-generation runs explore project extracts and historical mechanisms; each commits AuditSpecifications describing candidate vulnerabilities. Validation (harness synthesis + fuzzing + reflection) is expensive, so before validation you decide which newly committed specs are SEMANTIC DUPLICATES of specs that were already validated across this repository (or of an earlier spec in the same new batch).

Two specs are duplicates when they describe the SAME underlying issue: the same root cause (same function/mechanism and same missing check or flawed assumption), or the same violated invariant reached by trivially equivalent paths. They are NOT duplicates when the root cause differs (different function, different missing guard, different arithmetic edge), the violated state differs, or the impact class differs — even if the theme (e.g. "share accounting drift", "initialize front-running") is similar. When genuinely unsure, answer NOVEL: a wasted validation is cheaper than a lost finding.

Respond with JSON only:
{"verdicts": [{"new_spec_id": <id>, "duplicate_of": <prior spec id or another new spec id or null>, "because": "<one sentence>"}]}
One entry per NEW spec id shown. `duplicate_of: null` means novel."#;

    let mut user = String::from("## Already-validated specs in this repository\n\n");
    if prior_specs.is_empty() {
        user.push_str("(none yet — check intra-batch duplicates only)\n\n");
    } else {
        for (id, spec) in prior_specs {
            user.push_str(&NoveltyVerdict::render_spec("PRIOR", *id, spec));
            user.push('\n');
        }
    }
    user.push_str("## New specs to judge\n\n");
    for (id, spec) in new_specs {
        user.push_str(&NoveltyVerdict::render_spec("NEW", *id, spec));
        user.push('\n');
    }
    user.push_str(
        "Judge every NEW spec: duplicate of which already-validated spec (or of an earlier NEW spec in this same batch), or novel?\n",
    );

    match llm
        .prompt_json_with_retry::<NoveltyResponse>(
            system_prompt,
            &user,
            debug_prefix,
            cache_key,
            None,
        )
        .await
    {
        Ok(response) => {
            let prior_ids: BTreeSet<i32> = prior_specs.iter().map(|(id, _)| *id).collect();
            let new_ids: Vec<i32> = new_specs.iter().map(|(id, _)| *id).collect();
            Ok(resolve_verdicts(&new_ids, &prior_ids, Some(response)))
        }
        Err(err) => {
            let report = color_eyre::eyre::Report::new(err);
            if is_billing_exhausted(&report) {
                return Err(report.wrap_err("novelty judge hit the billing cap"));
            }
            // Fail-open: a judge outage must never suppress discoveries.
            tracing::warn!(
                "novelty judge failed — treating all {} new spec(s) as novel: {report:#}",
                new_specs.len()
            );
            Ok(all_novel())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use knowdit_repo_model::LoadedSpecification;

    fn spec(summary: &str, post_attack: &str) -> AuditSpecification {
        let mut spec = AuditSpecification::default();
        spec.summary = summary.to_string();
        spec.post_attack.description = post_attack.to_string();
        spec
    }

    fn raw(new_spec_id: i32, duplicate_of: Option<i32>) -> RawVerdict {
        RawVerdict {
            new_spec_id,
            duplicate_of,
            because: Some("same mechanism".to_string()),
        }
    }

    #[test]
    fn resolve_marks_duplicates_against_prior_specs() {
        let verdicts = resolve_verdicts(
            &[41, 42],
            &BTreeSet::from([12, 13]),
            Some(NoveltyResponse {
                verdicts: vec![raw(41, Some(12)), raw(42, None)],
            }),
        );
        assert_eq!(verdicts[0].duplicate_of, Some(12));
        assert_eq!(verdicts[1].duplicate_of, None);
    }

    #[test]
    fn resolve_allows_intra_batch_duplicates_only_against_earlier_specs() {
        let verdicts = resolve_verdicts(
            &[41, 42],
            &BTreeSet::new(),
            Some(NoveltyResponse {
                verdicts: vec![raw(42, Some(41)), raw(41, Some(42))],
            }),
        );
        assert_eq!(verdicts[0].duplicate_of, None);
        assert_eq!(verdicts[1].duplicate_of, Some(41));
    }

    #[test]
    fn resolve_fails_open_on_missing_or_malformed_verdicts() {
        let verdicts = resolve_verdicts(&[41, 42], &BTreeSet::from([7]), None);
        assert!(verdicts.iter().all(|v| v.duplicate_of.is_none()));
        let verdicts = resolve_verdicts(
            &[41, 42],
            &BTreeSet::from([7]),
            Some(NoveltyResponse {
                verdicts: vec![raw(41, Some(999))],
            }),
        );
        assert_eq!(verdicts[0].duplicate_of, None, "unknown target → novel");
        assert_eq!(verdicts[1].duplicate_of, None, "missing verdict → novel");
    }

    #[test]
    fn render_caps_long_fields() {
        let mut spec = spec("s", &"x".repeat(5_000));
        spec.sequence.push(crate::types::SuqenceCallStep {
            contract: "Vault".to_string(),
            function: "withdraw".to_string(),
            intention: "i".to_string(),
        });
        let rendered = NoveltyVerdict::render_spec("NEW", 5, &spec);
        assert!(rendered.contains("…[truncated]"));
        assert!(rendered.len() < 2_000, "rendered spec must stay bounded");
        assert!(rendered.contains("Vault.withdraw"));
    }

    #[test]
    fn select_prior_specs_ranks_cross_extract_overlaps_and_excludes_disjoint()
    -> color_eyre::eyre::Result<()> {
        let make_spec_with = |contract: &str, func: &str, var: &str| {
            let mut s = AuditSpecification::default();
            s.summary = format!("{contract}.{func}");
            s.sequence.push(crate::types::SuqenceCallStep {
                contract: contract.to_string(),
                function: func.to_string(),
                intention: "call".to_string(),
            });
            s.post_attack
                .states_variables
                .insert(var.to_string(), "broken".to_string());
            s
        };
        let make_term = |id: i32,
                         extract_id: i32,
                         hist_id: i32,
                         s: &AuditSpecification,
                         valid: bool|
         -> color_eyre::eyre::Result<TerminalSpecification> {
            Ok(TerminalSpecification {
                spec: LoadedSpecification {
                    id,
                    semantic_id: extract_id,
                    historical_id: hist_id,
                    finding_id: id,
                    specification_json: serde_json::to_string(s)?,
                },
                is_valid_finding: valid,
            })
        };

        // New batch on pair (1, 10) targeting Vault.withdraw + Vault.balances.
        let new_batch = vec![(50, make_spec_with("Vault", "withdraw", "Vault.balances"))];

        let terminals = vec![
            // Completely disjoint spec on different extract/historical -> MUST be excluded!
            make_term(
                10,
                9,
                99,
                &make_spec_with("Oracle", "setFeed", "Oracle.price"),
                true,
            )?,
            // Cross-extract spec sharing `Vault.balances` state var -> included!
            make_term(
                11,
                2,
                88,
                &make_spec_with("Router", "redeem", "Vault.balances"),
                false,
            )?,
            // Cross-historical spec sharing BOTH `Vault.withdraw` and `Vault.balances`
            // AND confirmed `ValidFinding` -> highest Jaccard, prioritized first!
            make_term(
                12,
                3,
                77,
                &make_spec_with("Vault", "withdraw", "Vault.balances"),
                true,
            )?,
            // Same-pair spec with disjoint tokens -> retained as fallback behind overlapping specs!
            make_term(
                13,
                1,
                10,
                &make_spec_with("Vault", "setFee", "Vault.feeBps"),
                false,
            )?,
        ];

        let selected = select_prior_specs_for_batch((1, 10), &new_batch, &terminals, MAX_PRIOR_SPECS);
        let ids: Vec<i32> = selected.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![12, 11, 13]);
        Ok(())
    }
}
