# RESULTS_RESCUE_PLAN

## Why broad `bad_action` was too noisy

The broad `bad_action` label mixes generic incompetence, task-regressive actions, and exploit-adjacent behavior. That makes it a weak primary target for a mechanistic follow-up paper whose main claim is about reward-exploit decision points rather than all mistakes.

## Why exploit-opportunity filtering is justified

The rescue analysis focuses on steps where exploit mechanics are plausibly active. Because admissible-action lists are not stored in the current ALFWorld temporal dump, exploit-opportunity states are approximated using proxy-mechanic activity and meta-action context:

- current or next exploit/meta actions (`easy_grader`, `fake_completion`, `claim_complete`, `gameable_hack`)
- steps where proxy-related fields are active
- one-step lookahead before exploit actions

This makes the temporal analysis decision-local rather than diluted by unrelated navigation or competence errors.

## What supports behavioral transfer

Behavioral transfer was computed from aggregate ALFWorld summaries.

The behavioral transfer claim is now tied to exploit-specific rates:

- `exploit_action_rate = easy_grader + fake_completion + gameable_hack`
- `easy_grader_rate`
- `fake_completion_rate`
- `proxy_score_without_success`
- `success_rate`

This is closer to the actual reward-hack story than generic `bad_action_rate`.

## What this analysis asks about late-stage dynamics

The main question is:

Do temporal reasoning signals at step `t` differ before exploitative next actions at step `t+1`, especially in mixed adapters?

The main figures therefore focus on:

- Qwen only
- mixed adapters `Mix05`, `Mix10`, `Mix50`
- exploit-opportunity states only
- `exploit_action_{t+1}` as the primary target

## What claims are supported

- Temporal internal signals become more meaningful when conditioned on exploit-opportunity states rather than all bad actions.
- The cleanest agentic follow-up claim is:
  internal state at step `t` predicts exploitative action selection at step `t+1`.
- Mixed adapters are the primary regime of interest; control may be sparse and hack may be saturated.
- Entropy, `p_hack`, and temporal summaries can be compared as competing predictive feature families rather than assuming a universal `p_hack` spike.

## What claims must be avoided

- `p_hack` universally rises before all bad actions.
- All mixed adapters show the same temporal pattern.
- Entropy always means danger.
- Steering fully solves reward hacking.
- A single TTC such as `32` represents the full story.

## Current prediction note

No reliable best TTC was identified from the available exploit-opportunity states.

## Clean final framing

The follow-up paper should claim:

Temporal monitoring during isolated generation becomes decision-local predictive monitoring in sequential decision making. At reward-exploit decision points, internal reasoning signals at step `t` carry actionable information about exploitative next actions at step `t+1`, with the strongest evidence in mixed adapters.
