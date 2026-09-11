//! Context-window budget arithmetic shared by every agent loop.
//!
//! # Why this exists
//!
//! llmy resolves a model id against its own registry. Unknown ids fall through
//! to `custom_model()`, which sets `max_input_tokens: 0`. Every caller that
//! derives a threshold as `max_input() * ratio` therefore gets **0**, and a
//! threshold of 0 is satisfied on the very first check:
//!
//! * agent loops compact (summarize + drop history) on *every* step,
//! * prompt packers emit single-item chunks,
//! * window-restart checks fire constantly.
//!
//! A real run hit this with `deepseek-ai/DeepSeek-V4.1-Flash`: the registry
//! entry is `deepseek/deepseek-v4-flash`, so the id missed, compaction fired
//! every step, the agent re-read the same function bodies in a loop, and 3.8M
//! input tokens produced zero specs.
//!
//! The registry gap is reported separately (see [`using_fallback_window`]) and
//! can be closed by adding the model to llmy's registry; this module only
//! ensures an unknown model degrades to a *conservative but usable* window
//! instead of thrashing.

/// Context window (tokens) assumed when the model is absent from llmy's
/// registry and reports `max_input_tokens == 0`.
///
/// Chosen to be safe for essentially every modern model (128k is a common
/// floor) while staying far above the point where an agent loop would compact
/// on every step. Deliberately *not* the largest known window: over-estimating
/// risks a hard context-length error from the provider, whereas
/// under-estimating only costs a little packing efficiency.
pub const FALLBACK_CONTEXT_WINDOW_TOKENS: u64 = 131_072;

/// Whether `max_input_tokens` is unusable and the fallback window applies.
///
/// Callers use this to warn once per run so the registry gap is visible rather
/// than silently masked.
pub fn using_fallback_window(max_input_tokens: u64) -> bool {
    max_input_tokens == 0
}

/// The model's effective input window: its real limit, or
/// [`FALLBACK_CONTEXT_WINDOW_TOKENS`] when the model is unknown.
pub fn effective_window_tokens(max_input_tokens: u64) -> u64 {
    if using_fallback_window(max_input_tokens) {
        FALLBACK_CONTEXT_WINDOW_TOKENS
    } else {
        max_input_tokens
    }
}

/// Token budget for one prompt: `ratio` × the model's effective input window.
///
/// `ratio` is clamped to `(0, 1]` so a mis-typed CLI value cannot zero out
/// (re-introducing the every-step-compaction failure) or overshoot the window.
pub fn context_budget(max_input_tokens: u64, ratio: f64) -> usize {
    let ratio = ratio.clamp(0.01, 1.0);
    (effective_window_tokens(max_input_tokens) as f64 * ratio) as usize
}

/// Resolve the threshold an agent loop should compact/restart at, and report
/// whether the unknown-model fallback was actually needed.
///
/// An explicit `override_tokens` wins outright **and suppresses the fallback
/// flag**: when the caller has supplied a threshold, the model's missing
/// registry entry has no effect on the decision, so warning about it would be
/// both false ("using a N token fallback window" when it is not) and
/// unactionable ("pass the flag you just passed").
pub fn resolve_threshold(
    override_tokens: Option<usize>,
    max_input_tokens: u64,
    ratio: f64,
) -> (usize, bool) {
    match override_tokens {
        Some(tokens) => (tokens, false),
        None => (
            context_budget(max_input_tokens, ratio),
            using_fallback_window(max_input_tokens),
        ),
    }
}

/// Report whether a fallback warning for `key` should be emitted now — true at
/// most once per process per key.
///
/// Agent loops run once **per link**; without this guard a 5 000-link run emits
/// 5 000 identical warnings and buries the real log.
pub fn warn_once_for(key: &str) -> bool {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};

    static WARNED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let mutex = WARNED.get_or_init(|| Mutex::new(HashSet::new()));
    let mut seen = match mutex.lock() {
        Ok(guard) => guard,
        // A panicking holder must not silently disable the warning.
        Err(poisoned) => poisoned.into_inner(),
    };
    seen.insert(key.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_model_uses_fallback_window_not_zero() {
        let budget = context_budget(0, 0.8);
        assert!(
            budget > 0,
            "an unknown model must never produce a zero budget"
        );
        assert_eq!(budget, (FALLBACK_CONTEXT_WINDOW_TOKENS as f64 * 0.8) as usize);
    }

    #[test]
    fn known_model_uses_its_own_window() {
        assert_eq!(context_budget(655_360, 0.8), (655_360.0_f64 * 0.8) as usize);
        assert!(!using_fallback_window(655_360));
    }

    #[test]
    fn fallback_detection_is_exactly_the_zero_case() {
        assert!(using_fallback_window(0));
        assert_eq!(effective_window_tokens(0), FALLBACK_CONTEXT_WINDOW_TOKENS);
        assert_eq!(effective_window_tokens(200_000), 200_000);
    }

    #[test]
    fn zero_or_negative_ratios_cannot_collapse_the_budget() {
        // A ratio of 0 (or a negative from a bad CLI parse) previously produced
        // a 0 threshold, i.e. compact-on-every-step.
        assert!(context_budget(200_000, 0.0) > 0);
        assert!(context_budget(200_000, -1.0) > 0);
        assert!(context_budget(0, 0.0) > 0);
    }

    #[test]
    fn ratios_above_one_are_clamped_to_the_window() {
        assert_eq!(
            context_budget(200_000, 5.0),
            context_budget(200_000, 1.0),
        );
    }

    #[test]
    fn explicit_override_wins_and_suppresses_the_fallback_flag() {
        // The reported bug: the user passed --gen-specs-compact-context-threshold-tokens
        // 209715 and still got "using a 131072 token fallback context window".
        // An override means the registry gap is irrelevant — no warning.
        let (threshold, used_fallback) = resolve_threshold(Some(209_715), 0, 0.8);
        assert_eq!(threshold, 209_715);
        assert!(
            !used_fallback,
            "an explicit override must not report a fallback window"
        );
    }

    #[test]
    fn fallback_flag_is_set_only_when_actually_falling_back() {
        assert_eq!(resolve_threshold(None, 0, 0.8), (104_857, true));
        assert_eq!(resolve_threshold(None, 200_000, 0.8), (160_000, false));
        // Override with a known model likewise reports no fallback.
        let (threshold, used_fallback) = resolve_threshold(Some(50_000), 200_000, 0.8);
        assert_eq!(threshold, 50_000);
        assert!(!used_fallback);
    }

    #[test]
    fn warn_once_for_fires_only_the_first_time_per_key() {
        let key = format!("test-key-{}", std::process::id());
        assert!(warn_once_for(&key), "first call warns");
        assert!(!warn_once_for(&key), "second call is suppressed");
        let other = format!("{key}-other");
        assert!(warn_once_for(&other), "distinct key warns independently");
    }
}
