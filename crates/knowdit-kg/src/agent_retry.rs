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
//! # Empty choices
//!
//! The same machinery covers [`LLMYError::EmptyChoice`], which the harness
//! raises when a response carries no choice at all. It is a transient
//! provider-side hiccup, and — like truncation — it is detected before the
//! context is mutated, so the step can be re-issued safely. It shares the
//! consecutive budget above but gets its own [`EMPTY_CHOICE_NUDGE`], since
//! telling the model its response "was cut off" would be a lie when there was
//! no response. See [`is_retryable`].
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

/// User-role nudge appended after a provider returned no choice at all.
///
/// Unlike [`TRUNCATION_NUDGE`] this must not claim the response was cut off —
/// there was no response. The model is told nothing was executed and to simply
/// re-emit its next action.
pub const EMPTY_CHOICE_NUDGE: &str = "The provider returned an empty response for your last \
     turn, so NOTHING from it was executed or recorded. This is a transient provider-side error. \
     Re-emit the exact same next action now.";

/// Whether an error is a provider-side output truncation.
pub fn is_output_length(err: &LLMYError) -> bool {
    matches!(err, LLMYError::OutputLength)
}

/// Whether an error is a transient provider-side condition that leaves the
/// conversation state untouched and can therefore be safely retried.
///
/// [`LLMYError::EmptyChoice`] qualifies for the same reason [`is_output_length`]
/// does: the harness returns it as soon as `resp.choices` is empty, *before*
/// anything is appended to the context, so re-issuing the step neither
/// duplicates nor corrupts the conversation. An empty `choices` array is a
/// provider hiccup (a dropped completion, an overloaded backend, a proxy
/// returning a body with no choice), not a verdict on the request — treating it
/// as fatal throws away the whole unit of work, which for `gen-specs` is one
/// link's entire agent run.
///
/// Callers still use [`is_output_length`] to tell "output cap exhausted" apart
/// from other failures when they abandon gracefully.
pub fn is_retryable(err: &LLMYError) -> bool {
    matches!(err, LLMYError::OutputLength | LLMYError::EmptyChoice)
}

/// The nudge to append for a given retryable error: truncation and empty-choice
/// failures need different wording, because only one of them had a response.
fn retry_nudge(err: &LLMYError) -> String {
    match err {
        LLMYError::EmptyChoice => EMPTY_CHOICE_NUDGE.to_string(),
        _ => TRUNCATION_NUDGE.to_string(),
    }
}

/// Reason to record when a retryable error's budget is exhausted and the
/// caller abandons the unit of work gracefully.
///
/// Callers must use this rather than a hardcoded "output cap" message: an
/// exhausted [`LLMYError::EmptyChoice`] budget is *not* a truncation, and
/// telling the operator to raise `--llm-max-completion-tokens` would send
/// them chasing the wrong fix.
pub fn exhaustion_reason(err: &LLMYError, consecutive: usize) -> String {
    match err {
        LLMYError::EmptyChoice => format!(
            "provider returned an empty choice {consecutive} times in a row"
        ),
        _ => format!(
            "response truncated by the provider output cap {consecutive} times in a row"
        ),
    }
}

/// Bounded retry state for consecutive transient provider failures.
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

    /// Record one retryable failure and decide whether to retry.
    ///
    /// `Ok(())` → retry (budget remains); `Err(consecutive)` → give up.
    fn record_failure(&mut self) -> Result<(), usize> {
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
                Err(err) if is_retryable(&err) => {
                    if let Err(consecutive) = self.record_failure() {
                        tracing::warn!(
                            "agent hit a retryable provider error {consecutive} times in a row \
                             (last: {err}); giving up (raise --llm-max-completion-tokens, or use a \
                             model with a larger output limit)",
                        );
                        return Err(err);
                    }
                    tracing::warn!(
                        "agent step failed with a retryable provider error ({}/{} consecutive: \
                         {err}); nudging and retrying the step",
                        self.consecutive,
                        self.max_consecutive,
                    );
                    agent.push_user_message(retry_nudge(&err));
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
            Err(err) if is_retryable(&err) => {
                // Count this failure against the same budget `step` uses, so
                // the first step can't hand `step` a free extra retry.
                if self.record_failure().is_err() {
                    return Err(err);
                }
                tracing::warn!(
                    "agent's first step failed with a retryable provider error ({}/{} \
                     consecutive: {err}); nudging and retrying",
                    self.consecutive,
                    self.max_consecutive,
                );
                agent.push_user_message(retry_nudge(&err));
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

    /// `EmptyChoice` must be retryable: the harness returns it before touching
    /// the context, so re-issuing the step is safe and far cheaper than
    /// abandoning the whole agent run.
    #[test]
    fn empty_choice_is_retryable() {
        assert!(is_retryable(&LLMYError::EmptyChoice));
        assert!(is_retryable(&LLMYError::OutputLength));
        assert!(!is_retryable(&LLMYError::Filtered("nope".into())));
    }

    /// Each retryable error gets wording that matches what actually happened:
    /// a truncated response was cut off, an empty choice was never produced.
    #[test]
    fn nudge_matches_the_failure_kind() {
        assert_eq!(retry_nudge(&LLMYError::EmptyChoice), EMPTY_CHOICE_NUDGE);
        assert_eq!(retry_nudge(&LLMYError::OutputLength), TRUNCATION_NUDGE);
        assert!(EMPTY_CHOICE_NUDGE.contains("NOTHING"));
        assert!(!EMPTY_CHOICE_NUDGE.contains("shorter"));
    }

    /// The give-up reason must name the actual failure: an exhausted
    /// empty-choice budget must not be reported as an output-cap truncation,
    /// or the operator would chase the wrong fix.
    #[test]
    fn exhaustion_reason_matches_the_failure_kind() {
        let empty = exhaustion_reason(&LLMYError::EmptyChoice, 4);
        assert!(empty.contains("empty choice"));
        assert!(!empty.contains("truncated"));
        let truncated = exhaustion_reason(&LLMYError::OutputLength, 4);
        assert!(truncated.contains("truncated"));
        assert!(!truncated.contains("empty choice"));
        assert!(empty.contains('4') && truncated.contains('4'));
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
        assert_eq!(retry.record_failure(), Ok(()), "1st truncation retries");
        assert_eq!(retry.record_failure(), Ok(()), "2nd truncation retries");
        assert_eq!(retry.record_failure(), Ok(()), "3rd truncation retries");
        assert_eq!(
            retry.record_failure(),
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
        assert_eq!(retry.record_failure(), Ok(()));
        assert_eq!(retry.record_failure(), Ok(()));
        retry.record_success();
        assert_eq!(retry.consecutive_failures(), 0);
        // Budget is fully available again.
        assert_eq!(retry.record_failure(), Ok(()));
        assert_eq!(retry.record_failure(), Ok(()));
        assert_eq!(retry.record_failure(), Err(3));
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
        assert_eq!(retry.record_failure(), Ok(()), "budget 1 allows the retry");
        // The retried `step` then truncates once more and must give up.
        assert_eq!(retry.record_failure(), Err(2));
    }

    #[test]
    fn nudge_tells_the_model_nothing_executed() {
        // The model must not assume its truncated tool call took effect.
        assert!(TRUNCATION_NUDGE.contains("NOTHING"));
        assert!(TRUNCATION_NUDGE.contains("shorter"));
    }
}
