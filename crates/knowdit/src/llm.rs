//! Provider-compatibility tweaks applied to every LLM handle knowdit builds.
//!
//! llmy forwards a caller-supplied `prompt_cache_key` to any endpoint — it only
//! strips the field for Google and MiMo models — but the field is an
//! OpenAI-specific routing hint. Strict OpenAI-compatible gateways reject it as
//! an unsupported parameter and fail the whole request with a 400, so we keep
//! it only for hosts that serve OpenAI's own API.

use llmy::client::client::{
    GoogleContentFilter, LLM, MiMoContentFilter, NoFilter, OpenAIContentFilter,
    RawExtensibleChatCompletionRequest, RawExtensibleChatCompletionResponse,
};
use llmy::client::model::OpenAIModel;

/// Hosts whose OpenAI-compatible APIs understand `prompt_cache_key`.
const PROMPT_CACHE_KEY_HOSTS: &[&str] = &[
    "api.openai.com",
    "openai.azure.com",
    "services.ai.azure.com",
];

fn endpoint_supports_prompt_cache_key(endpoint: &str) -> bool {
    PROMPT_CACHE_KEY_HOSTS
        .iter()
        .any(|host| endpoint.contains(host))
}

/// llmy's per-model filter, rebuilt here because `set_content_filter` replaces
/// whatever filter the client installed at construction time.
fn model_filter(model: &OpenAIModel) -> Box<dyn OpenAIContentFilter> {
    if model.is_mimo() {
        Box::new(MiMoContentFilter::default())
    } else if model.is_google() {
        Box::new(GoogleContentFilter)
    } else {
        Box::new(NoFilter)
    }
}

#[derive(Debug)]
struct ProviderCompat {
    model: Box<dyn OpenAIContentFilter>,
    strip_prompt_cache_key: bool,
}

impl OpenAIContentFilter for ProviderCompat {
    fn filter_input(&self, req: &mut RawExtensibleChatCompletionRequest) {
        self.model.filter_input(req);
        if self.strip_prompt_cache_key {
            req.prompt_cache_key = None;
        }
    }

    fn filter_output(&self, resp: &mut RawExtensibleChatCompletionResponse) {
        self.model.filter_output(resp);
    }
}

/// Attach the provider-compatibility filter to `llm`.
///
/// Clones and `scope()`s of the handle share the filter, so this has to run
/// once per top-level LLM, before any of them are cut.
pub fn provider_compat(llm: LLM) -> LLM {
    llm.set_content_filter(Box::new(ProviderCompat {
        model: model_filter(&llm.model),
        strip_prompt_cache_key: !endpoint_supports_prompt_cache_key(&llm.endpoint),
    }));
    llm
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request_with_cache_key() -> RawExtensibleChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "deepseek-ai/DeepSeek-V4.1-Flash",
            "messages": [{"role": "user", "content": "hi"}],
            "prompt_cache_key": "paxos-token-contracts"
        }))
        .unwrap()
    }

    fn compat(strip: bool) -> ProviderCompat {
        ProviderCompat {
            model: Box::new(NoFilter),
            strip_prompt_cache_key: strip,
        }
    }

    #[test]
    fn strips_cache_key_when_the_endpoint_rejects_it() {
        let mut req = request_with_cache_key();
        compat(true).filter_input(&mut req);
        assert!(req.prompt_cache_key.is_none());
    }

    #[test]
    fn keeps_cache_key_for_openai_hosts() {
        let mut req = request_with_cache_key();
        compat(false).filter_input(&mut req);
        assert_eq!(
            req.prompt_cache_key.as_deref(),
            Some("paxos-token-contracts")
        );
    }

    /// The default model id shipped in `.env` must resolve in llmy's registry.
    ///
    /// `deepseek-ai/DeepSeek-V4.1-Flash-1017` is not an upstream registry key,
    /// so without the vendored entry it falls through to `custom_model` with
    /// `max_input_tokens: 0`. Every budget then derives from the 131072-token
    /// fallback window instead of the real one, which silently shrinks link
    /// prompts ~10x (`--input-token-budget 262144` is clamped to
    /// `window × utilization`) and reports zero pricing. The vendored
    /// `llmy-tokenizer` supplies this key; see `[patch.crates-io]` in the root
    /// `Cargo.toml`.
    #[test]
    fn default_model_id_resolves_in_the_registry() {
        use std::str::FromStr;

        let mut model = OpenAIModel::from_str("deepseek-ai/DeepSeek-V4.1-Flash-1017")
            .expect("default model id must parse");
        let (max_input, _) = model.info();
        assert_eq!(
            max_input, 262_144,
            "registry entry must win over the 131072 fallback window"
        );

        // With `LLM_FULL_MODEL_NAME=true` (`api_model_name` returns the
        // canonical id) the request's `model` field is this string verbatim,
        // so the registry key must match the configured id exactly.
        model.set_full_id(true);
        assert_eq!(model.api_model_name(), "deepseek-ai/DeepSeek-V4.1-Flash-1017");

        // A resolved model carries pricing; unresolved ones report zero, which
        // is why every prior run_usage.json recorded `usd: 0`.
        assert!(!model.pricing().input.is_zero());

        // Prompt budgets are `window × utilization`, so this resolution is
        // what actually determines prompt size. Unresolved, the window was the
        // 131072 fallback and the link default (0.2) yielded 26214 tokens;
        // resolving to 262144 roughly quadruples it to 52428.
        let max_input = model.config.max_input();
        assert_eq!(
            knowdit_kg_model::context_budget(max_input, 0.2),
            52_428,
            "link default budget"
        );
        assert_eq!(
            knowdit_kg_model::context_budget(max_input, 0.4),
            104_857,
            "extract/categorize default budget"
        );
    }

    #[test]
    fn only_openai_hosts_keep_the_cache_key() {
        assert!(endpoint_supports_prompt_cache_key(
            "https://api.openai.com/v1"
        ));
        assert!(endpoint_supports_prompt_cache_key(
            "https://contoso.openai.azure.com/"
        ));
        assert!(!endpoint_supports_prompt_cache_key(
            "https://api.tokenjuice.ai/v1"
        ));
        assert!(!endpoint_supports_prompt_cache_key(
            "https://openrouter.ai/api/v1"
        ));
    }
}
