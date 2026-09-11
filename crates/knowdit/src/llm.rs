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
