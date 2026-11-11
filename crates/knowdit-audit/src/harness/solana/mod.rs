//! Solana `LiteSVM` + `proptest` / `#[test]` dynamic harness backend (`SolanaHarness`).

mod backend;
mod harness;
mod poc;
pub mod runner;
pub mod solana_json;
mod runtime;
mod tools;
mod utils;

pub use backend::SolanaHarness;
pub use runner::{SolanaBackend, SolanaBackendKind, SolanaFeatures, SolanaOutput, SolanaRunner};
pub use runtime::SolanaHarnessGenerator;
pub use solana_json::{
    ExecutedInstruction, POC_VIOLATION_MARKER, SETUP_FAILED_MARKER, SOLANA_SUMMARY_PREFIX,
    SolanaCounterCall, SolanaCoverageMapper, SolanaParsedOutput, SolanaWireSummary,
};
pub use utils::SolanaFuzzOptions;
