//! Resilience for provider-side output truncation in agent loops.
//!
//! # The problem
//!
//! When a completion ends with `FinishReason::Length`, llmy's harness returns
//! [`LLMYError::OutputLength`] and **discards the partial response** — the
//! truncated assistant message is never appended to the context (the `return`
//! happens before the checkpoint push). Treating that as fatal is wrong: the
//! conversation state is exactly as it was before the call, so the step can be
//! retried. Abandoning instead throws away the whole unit of work (for
//! `gen-specs`, one link's entire agent run — millions of input tokens).
//!
//! Truncation is driven by the provider's *output* cap. llmy only sets
//! `max_completion_tokens` when `LLMSettings::llm_max_completion_tokens` is
//! `Some`, so by default the provider's own limit applies; a model whose
//! registry entry is missing also reports `max_tokens: 0`, which never reaches
//! the wire. Reasoning models are the most exposed: reasoning tokens count
//! toward the same cap as the tool-call payload, so a verbose stretch of
//! thinking can truncate an otherwise small tool call.
//!
//! # The mitigation
//!
//! [`TruncationRetry::step`] retries a truncated step with a short user-role
//! nudge telling the model its response was cut off and to emit a more compact
//! one. The nudge is appended to the context, which is both safe (the
//! truncated output was discarded) and effective (sampling varies, and an
//! explicit "be concise" directive changes the next attempt's length).
//!
//! Retries are bounded and *consecutive*: a single truncation mid-run does not
//! accumulate toward the limit, but repeated back-to-back truncations mean the
//! model cannot fit its response in the provider's cap, so the caller stops
//! retrying and abandons gracefully instead of burning the step budget.
//!
//! Raising the provider-side cap is the complementary fix and is available
//! without a code change via `--llm-max-completion-tokens` /
//! `OPT_LLM_MAX_COMPLETION_TOKENS` (llmy plumbs it to
//! `max_completion_tokens`).

use llmy::agent::{LLMYError, StepResult};
use llmy::client::client::LLM;
use llmy::client::settings::LLMSettings;
use llmy::harness::Agent;

/// Default number of consecutive truncation retries before giving up.
pub const DEFAULT_MAX_TRUNCATION_RETRIES: usize = 3;

/// User-role nudge appended after a truncated response.
///
/// Phrased as an observation plus an instruction, and deliberately explicit
/// that nothing was executed — the model must not assume its partial tool call
/// took effect.
pub const TRUNCATION_NUDGE: &str = "Your previous response was cut off because it exceeded the \
     provider's maximum output length for a single response, so NOTHING from it was executed or \
     recorded. Retry now and keep the response substantially shorter: put any long explanation or \
     memory content into a separate, later tool call, emit at most one tool call this turn, and \
     continue the task from exactly where you left off.";

/// Whether an error is a provider-side output truncation.
pub fn is_output_length(err: &LLMYError) -> bool {
    matches!(err, LLMYError::OutputLength)
}

/// Bounded retry state for consecutive output-length truncations.
#[derive(Debug, Clone)]
pub struct TruncationRetry {
    consecutive: usize,
    max_consecutive: usize,
}

impl Default for TruncationRetry {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_TRUNCATION_RETRIES)
    }
}

impl TruncationRetry {
    /// `max_consecutive` is clamped to at least `1`; `0` would disable retries
    /// entirely and reinstate the fatal behavior.
    pub fn new(max_consecutive: usize) -> Self {
        Self {
            consecutive: 0,
            max_consecutive: max_consecutive.max(1),
        }
    }

    /// Truncations seen back-to-back, reset by any successful step.
    pub fn consecutive_failures(&self) -> usize {
        self.consecutive
    }

    /// How many retries a caller gets before the step error is fatal.
    pub fn max_consecutive(&self) -> usize {
        self.max_consecutive
    }

    /// Record one truncation and decide whether to retry.
    ///
    /// `Ok(())` → retry (budget remains); `Err(consecutive)` → give up.
    fn record_truncation(&mut self) -> Result<(), usize> {
        self.consecutive += 1;
        if self.consecutive > self.max_consecutive {
            Err(self.consecutive)
        } else {
            Ok(())
        }
    }

    /// Clear the consecutive counter after a successful step.
    fn record_success(&mut self) {
        self.consecutive = 0;
    }

    /// Run one agent step, retrying with [`TRUNCATION_NUDGE`] when the
    /// provider truncates the response.
    ///
    /// Returns `Err(LLMYError::OutputLength)` once `max_consecutive`
    /// truncations happen without an intervening success, so the caller can
    /// still distinguish "truncation gave up" from other failures and abandon
    /// gracefully. All other errors propagate untouched.
    pub async fn step(
        &mut self,
        agent: &mut Agent,
        llm: &LLM,
        debug_prefix: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<StepResult, LLMYError> {
        loop {
            match agent.step(llm, debug_prefix, settings.clone()).await {
                Ok(result) => {
                    self.record_success();
                    return Ok(result);
                }
                Err(err) if is_output_length(&err) => {
                    if let Err(consecutive) = self.record_truncation() {
                        tracing::warn!(
                            "agent hit the provider output cap {consecutive} times in a row; giving \
                             up (raise --llm-max-completion-tokens, or use a model with a larger \
                             output limit)",
                        );
                        return Err(err);
                    }
                    tracing::warn!(
                        "agent response truncated by the provider output cap ({}/{} consecutive); \
                         nudging and retrying the step",
                        self.consecutive,
                        self.max_consecutive,
                    );
                    agent.push_user_message(TRUNCATION_NUDGE.to_string());
                }
                Err(err) => return Err(err),
            }
        }
    }

    /// Same as [`Self::step`], but for the agent's **first** step (which
    /// carries the user prompt). Retries re-issue `step`, not
    /// `step_with_user`, so the prompt is not duplicated in the context.
    pub async fn first_step_with_user(
        &mut self,
        agent: &mut Agent,
        user_prompt: String,
        llm: &LLM,
        debug_prefix: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<StepResult, LLMYError> {
        match agent
            .step_with_user(user_prompt, llm, debug_prefix, settings.clone())
            .await
        {
            Ok(result) => {
                self.record_success();
                Ok(result)
            }
            Err(err) if is_output_length(&err) => {
                // Count this truncation against the same budget `step` uses, so
                // the first step can't hand `step` a free extra retry.
                if self.record_truncation().is_err() {
                    return Err(err);
                }
                tracing::warn!(
                    "agent's first response was truncated by the provider output cap \
                     ({}/{} consecutive); nudging and retrying",
                    self.consecutive,
                    self.max_consecutive,
                );
                agent.push_user_message(TRUNCATION_NUDGE.to_string());
                self.step(agent, llm, debug_prefix, settings).await
            }
            Err(err) => Err(err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_length_is_detected() {
        assert!(is_output_length(&LLMYError::OutputLength));
        assert!(!is_output_length(&LLMYError::EmptyChoice));
    }

    #[test]
    fn zero_retries_is_clamped_to_one() {
        // `0` would mean "never retry", i.e. the fatal behavior this module
        // exists to remove.
        assert_eq!(TruncationRetry::new(0).max_consecutive(), 1);
    }

    /// The retry budget is *consecutive*: with a budget of 3 the first three
    /// truncations retry and the fourth gives up. This is the core behavior
    /// that stops a link's whole agent run from being thrown away on a single
    /// truncated response.
    #[test]
    fn retries_exactly_the_configured_budget_then_gives_up() {
        let mut retry = TruncationRetry::new(3);
        assert_eq!(retry.record_truncation(), Ok(()), "1st truncation retries");
        assert_eq!(retry.record_truncation(), Ok(()), "2nd truncation retries");
        assert_eq!(retry.record_truncation(), Ok(()), "3rd truncation retries");
        assert_eq!(
            retry.record_truncation(),
            Err(4),
            "4th consecutive truncation gives up"
        );
        assert_eq!(retry.consecutive_failures(), 4);
    }

    /// A success between truncations must reset the budget, so an occasional
    /// truncation spread across a long run never accumulates toward the limit.
    #[test]
    fn success_resets_the_consecutive_budget() {
        let mut retry = TruncationRetry::new(2);
        assert_eq!(retry.record_truncation(), Ok(()));
        assert_eq!(retry.record_truncation(), Ok(()));
        retry.record_success();
        assert_eq!(retry.consecutive_failures(), 0);
        // Budget is fully available again.
        assert_eq!(retry.record_truncation(), Ok(()));
        assert_eq!(retry.record_truncation(), Ok(()));
        assert_eq!(retry.record_truncation(), Err(3));
    }

    /// The default budget must be > 0, otherwise the fix is a no-op.
    #[test]
    fn default_budget_is_usable() {
        let retry = TruncationRetry::default();
        assert!(retry.max_consecutive() >= 1);
        assert_eq!(retry.consecutive_failures(), 0);
    }

    /// The first step's truncation is charged to the same consecutive budget as
    /// later steps, so a truncation on step 1 plus a budget of 1 means the
    /// follow-up `step` gets no retry — the budget is not silently extended.
    #[test]
    fn first_step_truncation_charges_the_shared_budget() {
        let mut retry = TruncationRetry::new(1);
        // Mirrors `first_step_with_user`'s truncation branch.
        assert_eq!(retry.record_truncation(), Ok(()), "budget 1 allows the retry");
        // The retried `step` then truncates once more and must give up.
        assert_eq!(retry.record_truncation(), Err(2));
    }

    #[test]
    fn nudge_tells_the_model_nothing_executed() {
        // The model must not assume its truncated tool call took effect.
        assert!(TRUNCATION_NUDGE.contains("NOTHING"));
        assert!(TRUNCATION_NUDGE.contains("shorter"));
    }
}
