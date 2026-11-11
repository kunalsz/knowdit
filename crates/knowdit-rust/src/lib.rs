//! Static AST, call-graph, and account/state-graph analysis for Rust and Solana programs.

pub mod analyzer;
pub mod signature;
pub mod workspace;

pub use analyzer::{
    ExtractedRustCallable, ExtractedRustContainer, RustCallGraphResult, RustContainerKind,
    RustSourceInput, RustStaticAnalysis, RustStaticAnalyzer, RustStorageResult,
};
pub use signature::extract_function_signature;
pub use workspace::{RustProgramManifest, RustWorkspace};
