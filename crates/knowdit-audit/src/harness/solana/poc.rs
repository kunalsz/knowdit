//! Deterministic-PoC mode (`HarnessKind::Poc`) for `SolanaHarness`:
//! guides the agent to author a deterministic `#[test]` reproduction using
//! `LiteSVM` that asserts with `KNOWDIT_POST_ATTACK_REACHED:`.

use std::path::Path;

use itertools::Itertools;

use super::solana_json::POC_VIOLATION_MARKER;
use super::utils::SolanaConventions;
use crate::spec::LinkInput;
use crate::types::AuditSpecification;

pub(super) fn build_system_prompt(
    link: &LinkInput,
    spec: &AuditSpecification,
    spec_id: i32,
    per_spec_dir: &Path,
    conv: &SolanaConventions,
) -> String {
    let spec_json =
        serde_json::to_string_pretty(spec).unwrap_or_else(|_| "<unserializable>".into());
    format!(
        r#"You are the Solana Deterministic-PoC Harness agent (`LiteSVM`) for a
knowledge-driven smart-contract auditing pipeline.

## Your task

You are given:
- A project semantic (extracted from the audited Rust / Solana workspace)
- The historical semantic + finding it was matched to
- An `AuditSpecification` (setup / pre_attack / post_attack / sequence)

Synthesize ONE Rust integration test file (`test_{spec_id}.rs`) containing a
deterministic `#[test] fn test_post_attack_state_must_not_be_reachable()` that
loads the compiled SBF `.so` program(s) into `LiteSVM`, establishes
`spec.setup` + `spec.pre_attack`, executes `spec.sequence` step-by-step via real
`svm.send_transaction(...)` calls, and proves `spec.post_attack` is reachable:

```rust
if post_attack_reached {{
    tracker.emit_violation("<one-line why this proves the vulnerability>", true);
    panic!("{marker} <one-line why this proves the vulnerability>");
}}
```

The orchestrator recognizes `{marker}` as a real deterministic PoC reproduction
(distinct from an accidental setup/transaction panic). Without `{marker}`, a
failing test is treated as a broken harness, NOT a confirmed bug.

## Project DeFi semantic
- id: {extract_id}
- name: {extract_name}
- definition: {extract_definition}
- description: {extract_description}

## Historical DeFi semantic
- id: {historical_id}
- name: {historical_name}
- definition: {historical_definition}

## Historical finding (the attack you are reproducing)
- id: {finding_id}
- title: {finding_title}
- root_cause: {finding_root_cause}
- description: {finding_description}

## AuditSpecification (spec_id={spec_id})

```json
{spec_json}
```

## Pre-scaffolded Per-Spec Crate Layout

Your test runs inside `{per_spec_dir}` with `[lib] name = "knowdit_harness_rt"`:
- `knowdit_harness_rt::sbf_so_path("<lib_name>.so") -> PathBuf`
- `knowdit_harness_rt::anchor_ix_discriminator("<ix_snake_name>") -> [u8; 8]`
- `knowdit_harness_rt::anchor_account_discriminator("<StructName>") -> [u8; 8]`
- `knowdit_harness_rt::SummaryTracker`:
  - `SummaryTracker::new()`, `tracker.record_call(program, ix, args, ok)`,
    `tracker.emit_ok()`, `tracker.emit_violation(reason, true)`,
    `SummaryTracker::fail_setup(reason)`
- Available crates: `litesvm = "0.6"`, `solana-sdk = "2.2"`, `solana-program = "2.2"`,
  `spl-token = "7"`, `spl-associated-token-account = "6"`, `borsh = "1"`,
  `bytemuck = "1"`, `sha2 = "0.10"`, `bs58 = "0.5"`, `serde = "1"`, `serde_json = "1"`.

## Discovered SBF Programs & Precomputed Anchor Discriminators

{programs_table}

{discriminators_table}

Root `Cargo.toml` excerpt:
```toml
{cargo_toml_excerpt}
```

`Anchor.toml` excerpt:
```toml
{anchor_toml_excerpt}
```

Existing test/setup files in this repository:
{sample_test_filenames}

Sample test excerpt:
```rust
{sample_test_excerpt}
```

# Output shape (`test_{spec_id}.rs`)

```rust
use borsh::{{BorshDeserialize, BorshSerialize}};
use knowdit_harness_rt::{{
    SummaryTracker, anchor_account_discriminator, anchor_ix_discriminator, sbf_so_path,
}};
use litesvm::LiteSVM;
use solana_sdk::{{
    instruction::{{AccountMeta, Instruction}},
    pubkey::Pubkey,
    signature::{{Keypair, Signer}},
    transaction::Transaction,
}};

#[test]
fn test_post_attack_state_must_not_be_reachable() {{
    let mut svm = LiteSVM::new();
    let mut tracker = SummaryTracker::new();

    // 1. Load compiled SBF program(s) and initialize accounts per `spec.setup`.
    // 2. Assert `spec.pre_attack` preconditions hold BEFORE the exploit sequence.
    // 3. Execute `spec.sequence` step-by-step via `svm.send_transaction(...)`,
    //    calling `tracker.record_call("<program>", "<ix>", "...", res.is_ok())`.
    // 4. Read post-state accounts from `svm.get_account(&pubkey)` and evaluate
    //    whether `spec.post_attack` was reached:
    let post_attack_reached: bool = /* evaluate spec.post_attack on real SVM accounts */;
    if post_attack_reached {{
        tracker.emit_violation("<one-line impact summary>", true);
        panic!("{marker} <one-line impact summary>");
    }}
    tracker.emit_ok();
}}
```

# Hard rules

- Filename passed to `write_harness_file` MUST be `test_{spec_id}.rs`.
- Use `{marker}` ONLY for the final post-attack reachability proof — never on
  setup assertions (use `SummaryTracker::fail_setup(...)` or plain `expect(...)`
  for setup checks).
- Every `AccountMeta` list MUST match the target `#[derive(Accounts)]` struct's
  exact field order, signer flags, and mutability flags. Read the source via
  `read_contract_source` / `read_function_source` before writing the test.
- Do NOT declare "done" in chat — keep iterating `write_harness_file` +
  `run_solana` until the orchestrator terminates on `{marker}`.
"#,
        marker = POC_VIOLATION_MARKER,
        extract_id = link.extract_id,
        extract_name = link.extract.name,
        extract_definition = link.extract.definition.trim(),
        extract_description = link.extract.description.trim(),
        historical_id = link.historical_id,
        historical_name = link.historical.name,
        historical_definition = link.historical.definition.trim(),
        finding_id = link.finding_id,
        finding_title = link.finding.title,
        finding_root_cause = link.finding.root_cause.trim(),
        finding_description = link.finding_rendered_description.trim(),
        spec_id = spec_id,
        spec_json = spec_json,
        per_spec_dir = per_spec_dir.display(),
        programs_table = if conv.programs_table().is_empty() {
            "(no SBF program crates auto-detected — inspect Cargo.toml)".to_string()
        } else {
            conv.programs_table().to_string()
        },
        discriminators_table = if conv.discriminators_table().is_empty() {
            "(use `anchor_ix_discriminator` / `anchor_account_discriminator` from `knowdit_harness_rt`)"
                .to_string()
        } else {
            conv.discriminators_table().to_string()
        },
        cargo_toml_excerpt = if conv.cargo_toml_excerpt().is_empty() {
            "(no root Cargo.toml)".to_string()
        } else {
            conv.cargo_toml_excerpt().to_string()
        },
        anchor_toml_excerpt = if conv.anchor_toml_excerpt().is_empty() {
            "(no Anchor.toml at repo root)".to_string()
        } else {
            conv.anchor_toml_excerpt().to_string()
        },
        sample_test_filenames = if conv.sample_test_filenames().is_empty() {
            "(no existing test files found)".to_string()
        } else {
            conv.sample_test_filenames()
                .iter()
                .map(|s| format!("- {s}"))
                .join("\n")
        },
        sample_test_excerpt = if conv.sample_test_excerpt().is_empty() {
            "(no sample test excerpt available)".to_string()
        } else {
            conv.sample_test_excerpt().to_string()
        },
    )
}

pub(super) fn build_user_prompt(spec_id: i32) -> String {
    format!(
        "Synthesize the deterministic Solana `LiteSVM` PoC (`test_{spec_id}.rs`) for \
         spec_id={spec_id} per the system prompt.\n\
         Inspect the target instruction handlers and `#[derive(Accounts)]` structs first, \
         then iterate `write_harness_file` + `run_solana` until the test reproduces \
         the bug and panics with `{marker}`.\n",
        marker = POC_VIOLATION_MARKER,
    )
}

pub(super) fn build_restart_bootstrap(
    harness_filename: Option<&str>,
    last_test_calls: u64,
    last_gate_passed: bool,
    prior_restarts: usize,
) -> String {
    let mut out = String::from("\n\n## Restart context\n\n");
    out.push_str(&format!(
        "This is restart #{}: the previous agent hit its context window budget. ",
        prior_restarts + 1
    ));
    match harness_filename {
        Some(name) => out.push_str(&format!(
            "It left `tests/{name}` on disk; refine or overwrite `test_...rs` via `write_harness_file`. "
        )),
        None => out.push_str("No PoC file was written yet — start fresh. "),
    }
    out.push_str(&format!(
        "Last `run_solana` summary: calls={last_test_calls}, coverage_gate_passed={last_gate_passed}. \
         Remember: your final post-attack proof MUST call `tracker.emit_violation(..., true)` \
         and panic with `{POC_VIOLATION_MARKER} <reason>`."
    ));
    out
}
