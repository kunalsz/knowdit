//! Benchmarking and evaluation harness for the knowdit learning pipeline.
//!
//! The harness measures the quality, strength, gains, and losses of the
//! learn/extract/merge/link process against an **immutable, versioned
//! benchmark corpus and seed database**, never against the live
//! production KG. A run is:
//!
//! > immutable baseline DB + ordered corpus slice + exact run
//! > configuration → captured stage artifacts + resulting sandbox DB +
//! > graph delta + scored comparison.
//!
//! This keeps comparisons across code, model, prompt, and corpus
//! changes reproducible even though the production DB keeps growing and
//! its integer IDs are insertion-order-dependent.

pub mod baseline;
pub mod compare;
pub mod corpus;
pub mod error;
pub mod graph;
pub mod labels;
pub mod manifest;
pub mod metrics;
pub mod report;
pub mod replay;
pub mod sandbox;
pub mod scorer;

pub use error::{EvalError, Result};

/// Canonical fingerprint of one byte stream. All content-addressed
/// evaluation artifacts use this digest.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// Fingerprint of one string payload.
pub fn sha256_str(text: &str) -> String {
    sha256_hex(text.as_bytes())
}
