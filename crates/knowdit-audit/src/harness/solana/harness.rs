//! Handler-invariant mode (`HarnessKind::Handler`) for `SolanaHarness`:
//! guides the agent to author a `proptest` + `LiteSVM` stateful fuzz test
//! under `<per_spec_dir>/tests/test_{spec_id}.rs`.

use std::path::Path;

use itertools::Itertools;

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
        r#"You are the Solana `LiteSVM` + `proptest` Stateful Fuzzing Harness Generator agent
for a knowledge-driven smart-contract auditing pipeline.

## Your task

You are given:
- A project semantic (extracted from the audited Rust / Solana workspace)
- The historical semantic + finding it was matched to
- An `AuditSpecification` (setup / pre_attack / post_attack / sequence)

Synthesize ONE Rust integration test file (`test_{spec_id}.rs`) that loads the
compiled SBF `.so` program(s) into an in-process `LiteSVM` VM, drives bounded
`proptest` sequences of `HandlerAction` variants against the real SBF bytecode,
records calls on `knowdit_harness_rt::SummaryTracker`, and checks
`invariant_no_post_attack_state(&svm)` after every step. Then call `run_solana`
to compile + execute it.

The orchestrator drives termination — there is no `finalize` tool. Keep
iterating `write_harness_file` + `run_solana` until the test reproduces the
`spec.post_attack` violation (`violated=true` with Gate 1 passing).

## Project DeFi semantic
- id: {extract_id}
- name: {extract_name}
- definition: {extract_definition}
- description: {extract_description}

## Historical DeFi semantic
- id: {historical_id}
- name: {historical_name}
- definition: {historical_definition}

## Historical finding (topic context for the invariant)
- id: {finding_id}
- title: {finding_title}
- root_cause: {finding_root_cause}
- description: {finding_description}

## AuditSpecification (spec_id={spec_id})

```json
{spec_json}
```

## Pre-scaffolded Per-Spec Crate Layout

Your test runs inside `{per_spec_dir}` whose `Cargo.toml` and `src/lib.rs`
are ALREADY generated for you with:
- `[lib] name = "knowdit_harness_rt"` providing:
  - `knowdit_harness_rt::sbf_so_path("<lib_name>.so") -> PathBuf`
  - `knowdit_harness_rt::anchor_ix_discriminator("<ix_snake_name>") -> [u8; 8]`
  - `knowdit_harness_rt::anchor_account_discriminator("<StructName>") -> [u8; 8]`
  - `knowdit_harness_rt::SummaryTracker`:
    - `SummaryTracker::new()`
    - `tracker.record_call("<program_module>", "<ix_name>", format!("..."), tx_res.is_ok())`
    - `tracker.emit_ok()`
    - `tracker.emit_violation("<one-line reason>", false)`
    - `SummaryTracker::fail_setup("<reason>")`
- Pre-configured dependencies: `litesvm = "0.6"`, `solana-sdk = "2.2"`,
  `solana-program = "2.2"`, `spl-token = "7"`, `spl-associated-token-account = "6"`,
  `borsh = "1"`, `bytemuck = "1"`, `proptest = "1"`, `sha2 = "0.10"`, `bs58 = "0.5"`,
  `serde = "1"`, `serde_json = "1"`.

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

Write ONE `.rs` integration test file via `write_harness_file`:

```rust
use borsh::{{BorshDeserialize, BorshSerialize}};
use knowdit_harness_rt::{{
    SummaryTracker, anchor_account_discriminator, anchor_ix_discriminator, sbf_so_path,
}};
use litesvm::LiteSVM;
use proptest::prelude::*;
use solana_sdk::{{
    instruction::{{AccountMeta, Instruction}},
    pubkey::Pubkey,
    signature::{{Keypair, Signer}},
    transaction::Transaction,
}};

#[derive(Debug, Clone)]
enum HandlerAction {{
    // Bounded variants covering `spec.sequence` + adjacent state-mutating instructions
}}

fn setup_svm() -> (LiteSVM, Pubkey /* + initialized accounts / keypairs */) {{
    let mut svm = LiteSVM::new();
    let program_id: Pubkey = /* parse declared_program_id via bs58 / Pubkey::from_str_const */;
    let so_path = sbf_so_path("<lib_name>.so");
    svm.add_program_from_file(program_id, &so_path)
        .unwrap_or_else(|e| SummaryTracker::fail_setup(format!("load so failed: {{e:?}}")));
    // Fund signers (`svm.airdrop`), initialize PDAs / SPL mints / token accounts,
    // and assert `spec.pre_attack` preconditions hold.
    (svm, program_id)
}}

proptest! {{
    #[test]
    fn test_spec_{spec_id}(actions in prop::collection::vec(arb_action(), 1..=24)) {{
        let (mut svm, program_id) = setup_svm();
        let mut tracker = SummaryTracker::new();

        for action in actions {{
            // Build `Instruction` (8-byte `anchor_ix_discriminator("<fn>")` + Borsh args),
            // sign `Transaction::new_signed_with_payer`, execute `svm.send_transaction(tx)`,
            // and record the step:
            // tracker.record_call("<program_mod>", "<fn_name>", format!("{{action:?}}"), res.is_ok());

            if post_attack_reached(&svm) {{
                tracker.emit_violation("<one-line why spec.post_attack broke>", false);
                prop_assert!(false, "KNOWDIT_POST_ATTACK_REACHED: <one-line reason>");
            }}
        }}

        tracker.emit_ok();
    }}
}}
```

# Methodology

1. **Ground exact instruction layouts & account orders.** Call `lookup_call_graph`
   and `read_function_source` / `read_contract_source` to inspect the exact
   `#[derive(Accounts)]` struct field order (`AccountMeta` order MUST match the
   `#[derive(Accounts)]` struct field declaration order verbatim!), PDA `seeds`,
   `mut` vs readonly flags, `Signer` requirements, and instruction parameter types.
2. **Survey existing repo tests first.** Call `list_test_files()` and `read_file(...)`
   to copy exact initialization sequences, PDA seeds, and account layouts.
3. **Use real SBF execution — never mock program logic.** Every sequence step must
   execute real `svm.send_transaction(...)` against the loaded `.so` binary.
4. **Keep revert rate < 95% (Gate 1) and cover `spec.sequence` (Gate 2).** Clamp
   `proptest` amounts to realistic ranges (`1u64..=initial_deposit`) and supply
   valid PDAs/bumps so instructions execute their core logic instead of failing
   account deserialization on every call.
5. **Use exact `(contract, function)` names from `spec.sequence` in
   `tracker.record_call(contract, function, ...)`** so Gate 1 sequence-overlap and
   Gate 2 coverage map directly onto the call graph.

# Hard rules

- Filename passed to `write_harness_file` MUST be `test_{spec_id}.rs`.
- Always call `run_solana()` after `write_harness_file`.
- Do NOT declare "done" in chat — the orchestrator terminates automatically once
  `violated=true` passes Gate 1.
"#,
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
        "Synthesize the Solana `LiteSVM` + `proptest` harness (`test_{spec_id}.rs`) for \
         spec_id={spec_id} per the system prompt.\n\
         Inspect the target instruction handlers and `#[derive(Accounts)]` structs first, \
         then iterate `write_harness_file` + `run_solana` until the post-attack invariant \
         violation is reproduced.\n"
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
            "It left `tests/{name}` in the per-spec crate directory; inspect your prior errors \
             or overwrite `test_...rs` directly via `write_harness_file`. "
        )),
        None => out.push_str("No harness file was written yet — start fresh. "),
    }
    out.push_str(&format!(
        "Last `run_solana` summary: calls={last_test_calls}, coverage_gate_passed={last_gate_passed}. "
    ));
    if last_test_calls == 0 {
        out.push_str(
            "Zero calls means the test failed to compile, panicked in `setup_svm()`, or forgot \
             `SummaryTracker::record_call` + `emit_ok()`. Fix compilation/setup first.",
        );
    } else if !last_gate_passed {
        out.push_str(
            "Calls > 0 but Gate 2 coverage missed `spec.sequence` functions — ensure `AccountMeta` \
             ordering, PDA seeds, and discriminators match the target instructions so transactions \
             succeed.",
        );
    }
    out
}
