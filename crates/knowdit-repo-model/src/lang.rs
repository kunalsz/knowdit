//! Project source language enum.
//!
//! Used in two distinct roles:
//!
//! * **File-extension auto-detection** at project load time, via
//!   `knowdit_project::ProjectData::language`. That detector
//!   classifies a project as Solidity / Move / Rust purely from the
//!   `.sol` vs `.move` vs `.rs` file extensions it found while walking
//!   the scope.
//! * **Profile-agent declaration** as part of [`crate::ProjectProfile`].
//!   The profile agent reads the project's actual source and
//!   commits the language it observed — this is the
//!   authoritative answer the downstream phases (mapper /
//!   spec / reflect / regen) read back.
//!
//! Language-flavored prompt material lives on the
//! `HarnessBackend` trait — each per-language impl carries its
//! own prefix const. This module is just the enum + lightweight
//! display / parsing helpers.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash)]
pub enum SourceLanguage {
    Solidity,
    Move,
    Rust,
}

impl SourceLanguage {
    pub const ALL: [Self; 3] = [Self::Solidity, Self::Move, Self::Rust];

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Solidity => "Solidity",
            Self::Move => "Move",
            Self::Rust => "Rust",
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::Solidity => "sol",
            Self::Move => "move",
            Self::Rust => "rs",
        }
    }

    pub fn code_fence(self) -> &'static str {
        match self {
            Self::Solidity => "solidity",
            Self::Move => "move",
            Self::Rust => "rust",
        }
    }

    /// Parse a free-form string the LLM or CLI caller might emit
    /// (`"Solidity"` / `"solidity"` / `"Move"` / `"Sui Move"` /
    /// `"Rust"` / `"Solana"` / `"Anchor"`).
    /// Returns `None` for anything unrecognized so the caller
    /// (typically the profile agent's `finalize_profile` tool)
    /// can refuse to commit a malformed language.
    pub fn parse_lenient(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "solidity" | "sol" => Some(Self::Solidity),
            "move" | "sui move" | "sui_move" => Some(Self::Move),
            "rust" | "rs" | "solana" | "anchor" | "solana rust" | "solana_rust" => {
                Some(Self::Rust)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_lenient_known_spellings() {
        for s in ["Solidity", "solidity", "SOLIDITY", "sol", " Solidity "] {
            assert_eq!(
                SourceLanguage::parse_lenient(s),
                Some(SourceLanguage::Solidity),
                "input={s:?}"
            );
        }
        for s in ["Move", "move", "MOVE", "Sui Move", "sui_move"] {
            assert_eq!(
                SourceLanguage::parse_lenient(s),
                Some(SourceLanguage::Move),
                "input={s:?}"
            );
        }
        for s in [
            "Rust",
            "rust",
            "RUST",
            "rs",
            "Solana",
            "solana",
            "Anchor",
            "anchor",
            "Solana Rust",
            "solana_rust",
        ] {
            assert_eq!(
                SourceLanguage::parse_lenient(s),
                Some(SourceLanguage::Rust),
                "input={s:?}"
            );
        }
    }

    #[test]
    fn parse_lenient_rejects_unknown() {
        for s in ["", "vyper", "cairo", "Solidity!", "Move 2", "??"] {
            assert!(
                SourceLanguage::parse_lenient(s).is_none(),
                "should reject {s:?}"
            );
        }
    }

    #[test]
    fn json_roundtrip_and_metadata() {
        for lang in SourceLanguage::ALL {
            let s = serde_json::to_string(&lang).expect("serialize SourceLanguage");
            let back: SourceLanguage = serde_json::from_str(&s).expect("deserialize SourceLanguage");
            assert_eq!(back, lang);
            assert!(!lang.display_name().is_empty());
            assert!(!lang.extension().is_empty());
            assert!(!lang.code_fence().is_empty());
        }
        assert_eq!(SourceLanguage::Rust.extension(), "rs");
        assert_eq!(SourceLanguage::Rust.code_fence(), "rust");
        assert_eq!(SourceLanguage::Rust.display_name(), "Rust");
    }
}
