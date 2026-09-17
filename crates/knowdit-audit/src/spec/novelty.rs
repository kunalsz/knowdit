//! Novelty gate: LLM-judged semantic dedup between spec generation and
//! validation.
//!
//! Per-finding exploration breadth is load-bearing — one pass per
//! `(extract, historical)` pair demonstrably under-collects — but on a real
//! run 656 valid harness results collapsed to 38 unique findings: ~95% of
//! validation spend (codegen + forge + reflect) re-confirmed an
//! already-validated mechanism. This gate sits between the two phases.
//! Every link still runs its spec-gen agent; before a committed spec enters
//! the drain, an LLM judge compares it against the specs **already
//! validated** for the same `(extract, historical)` pair. Duplicates are
//! recorded as terminal [`CodeGenStatus::Duplicate`] rows — the DB keeps the
//! full exploration record, resume stays correct, and validation spend goes
//! only to genuinely new specs.
//!
//! Fail-open by design: any judge error (bad JSON, provider hiccup) marks
//! every spec novel, so a bad judge day costs tokens, never discoveries.
//! Only a billing-cap exhaustion propagates as fatal.

use std::collections::HashSet;

use color_eyre::eyre::Result;
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

/// How many already-validated specs of the pair the judge sees, most recent
/// first. The most recent specs are the best expressions of what the pair
/// has already covered.
pub const MAX_PRIOR_SPECS: usize = 12;

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
struct NoveltyResponse {
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

/// Bounded, judge-facing view of one spec.
fn render_spec(label: &str, id: i32, spec: &AuditSpecification) -> String {
    let mut out = format!("{label} id={id}\n");
    out.push_str(&format!("- summary: {}\n", bounded(&spec.summary, SUMMARY_CAP)));
    out.push_str(&format!(
        "- post_attack: {}\n",
        bounded(&spec.post_attack.description, POST_ATTACK_CAP)
    ));
    let mut invariants: Vec<String> = spec
        .post_attack
        .states_variables
        .iter()
        .map(|(key, constraint)| format!("{key}: {}", bounded(constraint, INVARIANT_CAP)))
        .collect();
    invariants.extend(spec.post_attack.contracts.iter().map(|(key, constraint)| {
        format!("contract {key}: {}", bounded(constraint, INVARIANT_CAP))
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
                bounded(&step.intention, SEQUENCE_STEP_CAP)
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

/// Resolve raw judge output into per-new-spec verdicts. Pure so the target
/// validation rules are unit-testable:
/// * a missing/malformed verdict → novel (fail-open);
/// * `duplicate_of` must name a **prior** spec, or an **earlier** new spec —
///   a forward or self reference would mark two specs duplicates of each
///   other with nothing validated, so it is treated as novel;
/// * an unknown id → novel.
pub(crate) fn resolve_verdicts(
    new_ids: &[i32],
    prior_ids: &HashSet<i32>,
    response: Option<NoveltyResponse>,
) -> Vec<NoveltyVerdict> {
    let mut index: std::collections::HashMap<i32, &RawVerdict> = std::collections::HashMap::new();
    if let Some(response) = &response {
        for verdict in &response.verdicts {
            index.entry(verdict.new_spec_id).or_insert(verdict);
        }
    }
    let mut seen_new: HashSet<i32> = HashSet::new();
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

/// Judge whether each new spec is semantically novel against the pair's
/// already-validated specs. Returns one verdict per entry of `new_specs`, in
/// order. Fail-open on any non-billing error; billing exhaustion propagates
/// so the scheduler aborts the run against a dead cap.
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
    if prior_specs.is_empty() {
        // First validated spec of the pair: nothing to duplicate against.
        return Ok(all_novel());
    }

    let system_prompt = r#"You are the novelty gate of a knowledge-driven smart-contract auditing pipeline. Several independent spec-generation runs explore the same `(project, historical-mechanism)` pair; each commits AuditSpecifications describing candidate vulnerabilities. Validation (harness synthesis + fuzzing + reflection) is expensive, so before validation you decide which newly committed specs are SEMANTIC DUPLICATES of specs that were already validated for this pair.

Two specs are duplicates when they describe the SAME underlying issue: the same root cause (same function/mechanism and same missing check or flawed assumption), or the same violated invariant reached by trivially equivalent paths. They are NOT duplicates when the root cause differs (different function, different missing guard, different arithmetic edge), the violated state differs, or the impact class differs — even if the theme (e.g. "share accounting drift", "initialize front-running") is similar. When genuinely unsure, answer NOVEL: a wasted validation is cheaper than a lost finding.

Respond with JSON only:
{"verdicts": [{"new_spec_id": <id>, "duplicate_of": <prior spec id or another new spec id or null>, "because": "<one sentence>"}]}
One entry per NEW spec id shown. `duplicate_of: null` means novel."#;

    let mut user = String::from("## Already-validated specs for this pair\n\n");
    for (id, spec) in prior_specs {
        user.push_str(&render_spec("PRIOR", *id, spec));
        user.push('\n');
    }
    user.push_str("## New specs to judge\n\n");
    for (id, spec) in new_specs {
        user.push_str(&render_spec("NEW", *id, spec));
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
            let prior_ids: HashSet<i32> = prior_specs.iter().map(|(id, _)| *id).collect();
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
            &HashSet::from([12, 13]),
            Some(NoveltyResponse {
                verdicts: vec![raw(41, Some(12)), raw(42, None)],
            }),
        );
        assert_eq!(verdicts[0].duplicate_of, Some(12));
        assert_eq!(verdicts[1].duplicate_of, None);
    }

    #[test]
    fn resolve_allows_intra_batch_duplicates_only_against_earlier_specs() {
        // 42 duplicating 41 (earlier in batch) is valid; 41 duplicating 42
        // (forward) is not — nothing validated would remain.
        let verdicts = resolve_verdicts(
            &[41, 42],
            &HashSet::new(),
            Some(NoveltyResponse {
                verdicts: vec![raw(42, Some(41)), raw(41, Some(42))],
            }),
        );
        assert_eq!(verdicts[0].duplicate_of, None);
        assert_eq!(verdicts[1].duplicate_of, Some(41));
    }

    #[test]
    fn resolve_fails_open_on_missing_or_malformed_verdicts() {
        // No response at all.
        let verdicts = resolve_verdicts(&[41, 42], &HashSet::from([7]), None);
        assert!(verdicts.iter().all(|v| v.duplicate_of.is_none()));
        // Verdicts missing entries or naming unknown ids.
        let verdicts = resolve_verdicts(
            &[41, 42],
            &HashSet::from([7]),
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
        let rendered = render_spec("NEW", 5, &spec);
        assert!(rendered.contains("…[truncated]"));
        assert!(rendered.len() < 2_000, "rendered spec must stay bounded");
        assert!(rendered.contains("Vault.withdraw"));
    }
}
