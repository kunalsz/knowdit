use std::sync::OnceLock;

use rust_decimal::Decimal;
use tiktoken_rs::CoreBPE;

// Build-time generated data
mod generated_models {
    include!(concat!(env!("OUT_DIR"), "/models_generated.rs"));
}
mod generated_claude {
    include!(concat!(env!("OUT_DIR"), "/claude_generated.rs"));
}

pub use generated_models::ModelId;

// ---------------------------------------------------------------------------
// Encoding enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Encoding {
    Cl100kBase,
    O200kBase,
    P50kBase,
    Claude,
}

impl Encoding {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "cl100k_base" => Some(Self::Cl100kBase),
            "o200k_base" => Some(Self::O200kBase),
            "p50k_base" => Some(Self::P50kBase),
            "claude" => Some(Self::Claude),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cl100kBase => "cl100k_base",
            Self::O200kBase => "o200k_base",
            Self::P50kBase => "p50k_base",
            Self::Claude => "claude",
        }
    }
}

// ---------------------------------------------------------------------------
// Model config (mirrors data/models.json)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct ModelTokens {
    pub content_multiplier: f64,
    pub base_overhead: i32,
    pub per_message: i32,
    pub tools_exist: i32,
    pub per_tool: i32,
    pub per_desc: i32,
    pub per_first_prop: i32,
    pub per_additional_prop: i32,
    pub per_prop_desc: i32,
    pub per_enum: i32,
    pub per_nested_object: i32,
    pub per_array_of_objects: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelPricing {
    /// Per-token USD price for uncached input.
    pub input: Decimal,
    /// Per-token USD price for output.
    pub output: Decimal,
    /// Per-token USD price for cache reads (falls back to `input` when absent).
    pub input_cache_read: Option<Decimal>,
    /// Per-token USD price for cache writes.
    pub input_cache_write: Option<Decimal>,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Vendor / namespace prefix (e.g. `Some("google")` for `google/gemini-3.1-pro-preview`).
    pub owner: Option<String>,
    /// Identifier sent to the upstream API (the part after `owner/`).
    pub model_name: String,
    pub encoding: String,
    pub tokens: ModelTokens,
    /// Human-friendly display name (e.g. `"Gemini 3.1 Pro Preview"`).
    pub name: String,
    pub max_input_tokens: u64,
    pub max_tokens: u64,
    pub pricing: Option<ModelPricing>,
}

impl ModelConfig {
    pub fn encoding(&self) -> Option<Encoding> {
        Encoding::from_str(&self.encoding)
    }

    pub fn max_input(&self) -> u64 {
        self.max_input_tokens
    }

    pub fn max_output(&self) -> u64 {
        self.max_tokens
    }

    pub fn count_tokens(&self, text: &str) -> Option<usize> {
        self.encoding().map(|enc| count_tokens(text, enc))
    }

    /// Lossy counting tokens, fallback to simple len // 4 if no encoder is available
    pub fn count_tokens_lossy(&self, text: &str) -> usize {
        self.count_tokens(text).unwrap_or_else(|| text.len() / 4)
    }
}

// ---------------------------------------------------------------------------
// ModelId convenience helpers (the variants themselves are generated)
// ---------------------------------------------------------------------------

impl ModelId {
    /// Vendor / namespace prefix derived from the canonical id.
    pub fn owner(&self) -> Option<&str> {
        self.as_str().split_once('/').map(|(o, _)| o)
    }

    /// Bare model name sent to the upstream API.
    pub fn model_name(&self) -> &str {
        self.as_str()
            .split_once('/')
            .map_or_else(|| self.as_str(), |(_, m)| m)
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Registry helpers (back compat with the previous Vec/HashMap API)
// ---------------------------------------------------------------------------

/// Iterate over all built-in models with their canonical ids and configs.
pub fn models() -> Vec<(&'static str, ModelConfig)> {
    ModelId::ALL_KNOWN
        .iter()
        .filter_map(|id| id.to_model_config().map(|c| (id.as_str(), c)))
        .collect()
}

pub fn get_model(model_id: &str) -> Option<ModelConfig> {
    ModelId::parse_known(model_id).and_then(|id| id.to_model_config())
}

pub fn encoding_for_model(model_id: &str) -> Option<Encoding> {
    get_model(model_id).and_then(|m| Encoding::from_str(&m.encoding))
}

// ---------------------------------------------------------------------------
// Claude BPE (built from pre-decoded binary data)
// ---------------------------------------------------------------------------

fn build_claude_bpe() -> CoreBPE {
    let data = generated_claude::CLAUDE_BPE_DATA;

    let encoder = {
        let mut pos = 0;
        let mut entries = Vec::with_capacity(generated_claude::CLAUDE_TOKEN_COUNT as usize);
        while pos < data.len() {
            let rank = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            entries.push((data[pos..pos + len].to_vec(), rank));
            pos += len;
        }
        entries.into_iter().collect()
    };

    let special = generated_claude::CLAUDE_SPECIAL_TOKENS
        .iter()
        .map(|&(k, v)| (k.to_string(), v))
        .collect();

    CoreBPE::new(encoder, special, generated_claude::CLAUDE_PAT_STR).expect("build Claude BPE")
}

// ---------------------------------------------------------------------------
// BPE singletons
// ---------------------------------------------------------------------------

static CLAUDE_BPE: OnceLock<CoreBPE> = OnceLock::new();
static CL100K_BPE: OnceLock<CoreBPE> = OnceLock::new();
static O200K_BPE: OnceLock<CoreBPE> = OnceLock::new();
static P50K_BPE: OnceLock<CoreBPE> = OnceLock::new();

pub fn get_bpe(encoding: Encoding) -> &'static CoreBPE {
    match encoding {
        Encoding::Cl100kBase => {
            CL100K_BPE.get_or_init(|| tiktoken_rs::cl100k_base().expect("init cl100k_base"))
        }
        Encoding::O200kBase => {
            O200K_BPE.get_or_init(|| tiktoken_rs::o200k_base().expect("init o200k_base"))
        }
        Encoding::P50kBase => {
            P50K_BPE.get_or_init(|| tiktoken_rs::p50k_base().expect("init p50k_base"))
        }
        Encoding::Claude => CLAUDE_BPE.get_or_init(build_claude_bpe),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Encode text into token ids using the given encoding.
pub fn encode(text: &str, encoding: Encoding) -> Vec<u32> {
    let bpe = get_bpe(encoding);
    bpe.encode_with_special_tokens(text)
}

/// Count tokens in the text for the given encoding.
pub fn count_tokens(text: &str, encoding: Encoding) -> usize {
    encode(text, encoding).len()
}

/// Count tokens using a model identifier (e.g. "openai/gpt-4o").
/// Returns `None` if the model is unknown.
pub fn count_tokens_for_model(text: &str, model_id: &str) -> Option<usize> {
    encoding_for_model(model_id).map(|enc| count_tokens(text, enc))
}

/// List all known model IDs.
pub fn model_ids() -> Vec<&'static str> {
    let mut ids: Vec<&str> = ModelId::ALL_KNOWN.iter().map(|id| id.as_str()).collect();
    ids.sort();
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cl100k_hello() {
        let tokens = encode("Hello, world!", Encoding::Cl100kBase);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_o200k_hello() {
        let tokens = encode("Hello, world!", Encoding::O200kBase);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_p50k_hello() {
        let tokens = encode("Hello, world!", Encoding::P50kBase);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_claude_hello() {
        let tokens = encode("Hello, world!", Encoding::Claude);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_model_lookup() {
        assert_eq!(
            encoding_for_model("openai/gpt-4o"),
            Some(Encoding::O200kBase)
        );
        assert_eq!(
            encoding_for_model("anthropic/claude-sonnet-4"),
            Some(Encoding::Claude)
        );
        assert_eq!(encoding_for_model("nonexistent/model"), None);
    }

    #[test]
    fn test_count_tokens_for_model() {
        let count = count_tokens_for_model("Hello, world!", "openai/gpt-4o");
        assert!(count.is_some());
        assert!(count.unwrap() > 0);
    }

    #[test]
    fn test_model_config_limits() {
        let model = get_model("openai/gpt-5.1").expect("known model");

        assert_eq!(model.max_input_tokens, 272000);
        assert_eq!(model.max_tokens, 128000);
        assert_eq!(model.max_input(), 272000);
        assert_eq!(model.max_output(), 128000);
        assert_eq!(model.owner.as_deref(), Some("openai"));
        assert_eq!(model.model_name, "gpt-5.1");
    }

    #[test]
    fn test_mimo_model_config() {
        let model = get_model("mimo/mimo-v2.5-pro").expect("known model");

        assert_eq!(model.encoding(), Some(Encoding::O200kBase));
        assert_eq!(model.max_input_tokens, 917504);
        assert_eq!(model.max_tokens, 131072);
        assert_eq!(
            model.pricing.expect("mimo pricing").input,
            rust_decimal::dec!(0.000001)
        );
    }

    #[test]
    fn test_deepseek_v4_model_config() {
        let model = get_model("deepseek/deepseek-v4-flash").expect("known model");

        assert_eq!(model.encoding(), Some(Encoding::O200kBase));
        assert_eq!(model.max_input_tokens, 655360);
        assert_eq!(model.max_tokens, 393216);
        assert_eq!(
            model.pricing.expect("deepseek pricing").input,
            rust_decimal::dec!(0.00000014)
        );
    }

    #[test]
    fn test_model_id_parse_and_custom() {
        let known = ModelId::parse_known("openai/gpt-4o").expect("known id");
        assert_eq!(known.as_str(), "openai/gpt-4o");
        assert_eq!(known.owner(), Some("openai"));
        assert_eq!(known.model_name(), "gpt-4o");
        assert!(known.to_model_config().is_some());

        let custom = ModelId::from_str_or_custom("totally-made-up");
        assert!(matches!(custom, ModelId::Custom(_)));
        assert_eq!(custom.as_str(), "totally-made-up");
        assert!(custom.to_model_config().is_none());
    }
}
