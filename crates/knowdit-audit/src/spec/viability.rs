//! Opt-in pre-fuzz viability gate (`--gen-specs-viability-gate`).
//!
//! Screens novel [`AuditSpecification`]s in a single lightweight JSON LLM call
//! against [`ProjectProfile::out_of_scope_notes`] and the static signatures /
//! modifiers of each `sequence` step before entering the expensive
//! `fuzz -> reflect -> regen` loop.
//!
//! Strictly fail-open: any LLM error, malformed JSON, missing verdict, or
//! borderline ambiguity marks the spec `viable: true` so discoveries are never
//! lost to judge noise. Only billing-cap exhaustion propagates as fatal.

use std::collections::BTreeMap;

use color_eyre::eyre::Result;
use knowdit_repo_model::ProjectProfile;
use serde::Deserialize;

use crate::types::AuditSpecification;

use super::{ProjectIndex, is_billing_exhausted};

const SUMMARY_CAP: usize = 320;
const DESC_CAP: usize = 500;
const CONSTRAINT_CAP: usize = 160;
const OUT_OF_SCOPE_CAP: usize = 2_000;

/// Per-spec outcome from [`judge_spec_viability`], aligned with input order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViabilityVerdict {
    /// `true` when the spec should proceed into `fuzz`; `false` when it is
    /// unambiguously out of scope or structurally self-contradictory.
    pub viable: bool,
    /// One-sentence explanation recorded on `CodeGenStatus::SkippedViability`
    /// when `viable == false`.
    pub reason: String,
}

impl ViabilityVerdict {
    pub fn viable() -> Self {
        Self {
            viable: true,
            reason: String::new(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ViabilityResponse {
    #[serde(default)]
    verdicts: Vec<RawViabilityVerdict>,
}

#[derive(Debug, Deserialize)]
struct RawViabilityVerdict {
    spec_id: i32,
    #[serde(default)]
    viable: Option<bool>,
    #[serde(default)]
    reason: Option<String>,
}

/// Stateless 1-shot pre-fuzz viability screener.
pub struct ViabilityJudge;

impl ViabilityJudge {
    /// Pure resolution from deserialized judge output to aligned verdicts.
    /// Missing or malformed entries fail open (`viable: true`).
    pub(crate) fn resolve_verdicts(
        spec_ids: &[i32],
        response: Option<ViabilityResponse>,
    ) -> Vec<ViabilityVerdict> {
        let mut by_id: BTreeMap<i32, &RawViabilityVerdict> = BTreeMap::new();
        if let Some(resp) = &response {
            for item in &resp.verdicts {
                by_id.entry(item.spec_id).or_insert(item);
            }
        }
        spec_ids
            .iter()
            .map(|id| match by_id.get(id) {
                Some(raw) if raw.viable == Some(false) => {
                    let reason = raw
                        .reason
                        .as_deref()
                        .map(str::trim)
                        .filter(|r| !r.is_empty())
                        .unwrap_or("unambiguously out of scope or self-contradictory")
                        .to_string();
                    ViabilityVerdict {
                        viable: false,
                        reason,
                    }
                }
                _ => ViabilityVerdict::viable(),
            })
            .collect()
    }

    pub(crate) fn render_user_prompt(
        profile: Option<&ProjectProfile>,
        project_index: &ProjectIndex,
        specs: &[(i32, AuditSpecification)],
    ) -> String {
        let mut out = String::from("## Project Profile & Scope Constraints\n\n");
        if let Some(profile) = profile {
            out.push_str(&format!(
                "- domain_summary: {}\n",
                Self::bounded(&profile.domain_summary, DESC_CAP)
            ));
            let oos = profile.out_of_scope_notes.trim();
            out.push_str(&format!(
                "- out_of_scope_notes: {}\n\n",
                if oos.is_empty() {
                    "(none specified)".to_string()
                } else {
                    Self::bounded(oos, OUT_OF_SCOPE_CAP)
                }
            ));
        } else {
            out.push_str("- out_of_scope_notes: (no project profile loaded)\n\n");
        }

        out.push_str("## Candidate Specifications to Screen\n\n");
        for (id, spec) in specs {
            out.push_str(&format!("SPEC id={id}\n"));
            out.push_str(&format!(
                "- summary: {}\n",
                Self::bounded(&spec.summary, SUMMARY_CAP)
            ));
            if !spec.setup.description.trim().is_empty() {
                out.push_str(&format!(
                    "- setup: {}\n",
                    Self::bounded(&spec.setup.description, DESC_CAP)
                ));
            }
            if !spec.pre_attack.description.trim().is_empty() {
                out.push_str(&format!(
                    "- pre_attack: {}\n",
                    Self::bounded(&spec.pre_attack.description, DESC_CAP)
                ));
            }
            for (k, v) in spec.pre_attack.contracts.iter().take(4) {
                out.push_str(&format!(
                    "    - pre contract {k}: {}\n",
                    Self::bounded(v, CONSTRAINT_CAP)
                ));
            }
            out.push_str(&format!(
                "- post_attack: {}\n",
                Self::bounded(&spec.post_attack.description, DESC_CAP)
            ));
            for (k, v) in spec.post_attack.states_variables.iter().take(4) {
                out.push_str(&format!(
                    "    - post var {k}: {}\n",
                    Self::bounded(v, CONSTRAINT_CAP)
                ));
            }
            out.push_str("- sequence & target signatures:\n");
            for step in &spec.sequence {
                let sigs = project_index.sequence_step_signatures(&step.contract, &step.function);
                let sig_label = if sigs.is_empty() {
                    "(auxiliary / synthesized)".to_string()
                } else {
                    sigs.join(" | ")
                };
                out.push_str(&format!(
                    "    - `{}.{}` [sig: `{sig_label}`] — {}\n",
                    step.contract,
                    step.function,
                    Self::bounded(&step.intention, CONSTRAINT_CAP),
                ));
            }
            out.push('\n');
        }
        out.push_str(
            "Screen each SPEC id: mark `viable: false` ONLY when unambiguously excluded by `out_of_scope_notes` or structurally self-contradictory against the target signatures; otherwise `viable: true`.\n",
        );
        out
    }

    fn bounded(text: &str, cap: usize) -> String {
        let text = text.trim();
        if text.chars().count() <= cap {
            text.to_string()
        } else {
            let prefix: String = text.chars().take(cap).collect();
            format!("{prefix}…[truncated]")
        }
    }
}

/// Run the 1-shot fail-open pre-fuzz viability check over `specs`.
pub async fn judge_spec_viability(
    llm: &llmy::client::client::LLM,
    cache_key: Option<&str>,
    debug_prefix: Option<&str>,
    profile: Option<&ProjectProfile>,
    project_index: &ProjectIndex,
    specs: &[(i32, AuditSpecification)],
) -> Result<Vec<ViabilityVerdict>> {
    if specs.is_empty() {
        return Ok(Vec::new());
    }
    let all_viable = || {
        specs
            .iter()
            .map(|_| ViabilityVerdict::viable())
            .collect::<Vec<_>>()
    };

    let system_prompt = r#"You are a conservative pre-fuzz viability screener in a smart-contract security auditing pipeline. Each candidate AuditSpecification is about to enter an expensive fuzz-harness synthesis loop.

Mark a specification `viable: false` ONLY when it is UNAMBIGUOUSLY impossible or out-of-scope for one of two concrete reasons:
1. **Explicitly Out-of-Scope Mechanism / Trusted Actor:** `out_of_scope_notes` explicitly excludes the exact subsystem, asset class, or trusted-admin / governance privilege abuse the specification relies on.
2. **Hard Structural Self-Contradiction:** `setup` / `pre_attack` explicitly restricts the attacker to an unprivileged external user while `sequence` directly invokes a function whose signature carries an exclusive privilege modifier (e.g. `onlyOwner`, `onlyGovernance`) with no delegation/callback path, OR `pre_attack` and `post_attack` impose mutually contradictory mathematical constraints that cannot both hold.

Whenever there is ANY doubt, subtle edge case, callback/flash-loan possibility, or incomplete context, you MUST answer `viable: true`. Never reject a plausible vulnerability hypothesis.

Respond with JSON only:
{"verdicts": [{"spec_id": <id>, "viable": <true|false>, "reason": "<concise one-sentence reason>"}]}"#;

    let user = ViabilityJudge::render_user_prompt(profile, project_index, specs);
    match llm
        .prompt_json_with_retry::<ViabilityResponse>(
            system_prompt,
            &user,
            debug_prefix,
            cache_key,
            None,
        )
        .await
    {
        Ok(response) => {
            let ids: Vec<i32> = specs.iter().map(|(id, _)| *id).collect();
            Ok(ViabilityJudge::resolve_verdicts(&ids, Some(response)))
        }
        Err(err) => {
            let report = color_eyre::eyre::Report::new(err);
            if is_billing_exhausted(&report) {
                return Err(report.wrap_err("viability judge hit the billing cap"));
            }
            tracing::warn!(
                "viability judge failed — failing open (treating all {} spec(s) as viable): {report:#}",
                specs.len()
            );
            Ok(all_viable())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use knowdit_repo_model::{
        SourceLanguage,
        cg::{CallGraph, Contract, FileChunk, FileLocation, Function},
        inheritance::InheritanceGraph,
        storage::StorageGraph,
    };
    use std::path::PathBuf;

    #[test]
    fn resolve_verdicts_fails_open_on_missing_or_uncertain_and_flags_explicit_rejections() {
        let none_verdicts = ViabilityJudge::resolve_verdicts(&[10, 11], None);
        assert!(none_verdicts.iter().all(|v| v.viable));

        let resp = ViabilityResponse {
            verdicts: vec![
                RawViabilityVerdict {
                    spec_id: 10,
                    viable: Some(false),
                    reason: Some("admin fee rug-pull explicitly excluded by out_of_scope_notes".to_string()),
                },
                RawViabilityVerdict {
                    spec_id: 11,
                    viable: None,
                    reason: None,
                },
            ],
        };
        let verdicts = ViabilityJudge::resolve_verdicts(&[10, 11, 12], Some(resp));
        assert!(!verdicts[0].viable);
        assert!(verdicts[0].reason.contains("out_of_scope_notes"));
        assert!(verdicts[1].viable, "viable: None must fail open");
        assert!(verdicts[2].viable, "missing spec_id must fail open");
    }

    #[test]
    fn render_user_prompt_embeds_out_of_scope_notes_and_extracted_signatures() {
        let loc = FileLocation {
            start_line: 1,
            start_column: 0,
            end_line: 2,
            end_column: 0,
        };
        let mut contracts = BTreeMap::new();
        contracts.insert(
            1,
            Contract {
                id: 1,
                name: "Vault".to_string(),
                relative_file_path: PathBuf::from("src/Vault.sol"),
                chunk: FileChunk {
                    loc,
                    content: String::new(),
                },
                functions: vec![Function {
                    id: 1,
                    name: "setFee".to_string(),
                    args: "uint256 fee".to_string(),
                    relative_file_path: PathBuf::from("src/Vault.sol"),
                    loc,
                    content: Some("function setFee(uint256 fee) external onlyOwner {}".to_string()),
                    calls: Vec::new(),
                    description: None,
                }],
                description: None,
            },
        );
        let index = ProjectIndex::build(
            CallGraph {
                contracts,
                interfaces: BTreeMap::new(),
            },
            StorageGraph::default(),
            InheritanceGraph::default(),
        );
        let profile = ProjectProfile {
            language: SourceLanguage::Solidity,
            domain_summary: "ERC4626 vault".to_string(),
            subsystems: Vec::new(),
            core_components: Vec::new(),
            out_of_scope_notes: "Trusted owner fee changes are out of scope.".to_string(),
            source_files_read: Vec::new(),
        };
        let mut spec = AuditSpecification::default();
        spec.summary = "unprivileged user calls setFee".to_string();
        spec.sequence.push(crate::types::SuqenceCallStep {
            contract: "Vault".to_string(),
            function: "setFee".to_string(),
            intention: "raise fee to 100%".to_string(),
        });

        let rendered = ViabilityJudge::render_user_prompt(Some(&profile), &index, &[(7, spec)]);
        assert!(rendered.contains("Trusted owner fee changes are out of scope."));
        assert!(rendered.contains("function setFee(uint256 fee) external onlyOwner"));
    }
}
