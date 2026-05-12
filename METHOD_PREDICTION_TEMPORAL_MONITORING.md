# Method: Prediction and Temporal Monitoring

This document gives a paper-facing description of the monitoring and prediction methodology used in the repository. It is intended to be explicit enough for the Methods section and appendix of the paper.

## 1. Core framing

The central question is:

> Can internal signals extracted from reasoning at decision step \(t\) help predict whether the agent will take an undesirable action at step \(t+1\)?

We treat this as a next-step risk estimation problem:

\[
\Pr(y^{\mathrm{risk}}_{t+1}=1 \mid z_t, c_t),
\]

where:

- \(z_t\) are internal monitoring features extracted from the model reasoning at step \(t\)
- \(c_t\) are decision-context features available at step \(t\)
- \(y^{\mathrm{risk}}_{t+1}\) is an environment-defined label on the next step

The important methodological choice is that:

- features come from step \(t\)
- labels come from step \(t+1\)
- labels are defined from environment behavior, not from the monitor itself

## 2. Data unit

The basic supervised-learning unit is **one agent decision step**.

Each row corresponds to a decision step and may include:

- environment
- model family
- adapter
- TTC / reasoning budget
- episode id
- step id
- action taken at step \(t\)
- internal monitor summaries from the reasoning segment at step \(t\)
- environment/action label at step \(t+1\)

For ALFWorld, step-level data comes from processed temporal monitoring tables such as:

- `BALROG/analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv`

For WebShop, step-level data comes from processed run summaries and next-step pair exports where available.

## 3. Prediction targets

### ALFWorld

Primary or commonly used targets:

- `bad_action_t1`
- `exploit_action_t1`

Auxiliary next-step labels may include:

- `easy_grader_action_t1`
- `fake_completion_action_t1`
- `gameable_hack_action_t1`

### WebShop

Common next-step targets:

- `buy_t1`
- `bad_buy_t1`
- `low_reward_buy_t1`

The paper should emphasize that these are **environment-defined behavioral targets**, not monitor-defined targets.

## 4. Internal monitoring features

The main internal signals are:

- reward-hack activation (`p_hack`)
- token-level entropy

These are derived from the reasoning segment before the next action.

### 4.1 Scalar internal features

Typical scalar features include:

- `reasoning_p_hack_mean`
- `reasoning_p_hack_final`
- `reasoning_p_hack_late_mean`
- `reasoning_entropy_mean`
- `reasoning_entropy_final`
- `reasoning_entropy_late_mean`

### 4.2 Temporal summary features

The prediction pipeline also uses compressed temporal summaries extracted from within-step trajectories, such as:

- `reasoning_p_hack_late_change`
- `reasoning_p_hack_late_slope`
- `reasoning_entropy_late_change`
- `reasoning_entropy_late_slope`

These are what the code and figures refer to as **temporal internal features**.

Important clarification:

- `temporal internal` is **not** a raw temporal plot
- it is a **feature group for the classifier**
- the full token-level trajectory is compressed into a small set of summary statistics before prediction

## 5. Context features

We do not treat reward-hack activation as a standalone probability of failure. Instead, we also use decision-context features that describe the state in which the action is chosen.

Typical context features include:

- TTC / reasoning budget
- regime label (`ttc_8`, `ttc_16`, ..., `unconstrained`)
- normalized step position in the episode
- previous action type
- previous invalid / low-quality action indicator if available
- environment identity
- affordance/context flags such as:
  - exploit action available
  - proxy score visible
  - near completion
  - WebShop page type indicators

Conceptually:

- `p_hack` describes a latent internal policy state
- entropy describes decision uncertainty / instability
- context describes whether the environment currently affords risky behavior

## 6. Feature groups used in the ablations

The main feature groups are:

### `p_hack_only`

Activation-only features, for example:

- mean `p_hack`
- final `p_hack`
- late `p_hack`
- late change / slope if included in that pipeline

### `entropy_only`

Entropy-only features, for example:

- mean entropy
- final entropy
- late entropy
- late change / slope if included in that pipeline

### `temporal_internal`

Compressed trajectory summaries, especially:

- late change
- late slope

### `p_hack_plus_entropy`

Joint internal-state features from activation and entropy.

### `internal_all`

All internal features:

- activation
- entropy
- temporal summaries

### `context_only`

Only decision-context features, without internal monitor features.

### `internal_plus_context`

The richest paper-facing feature group:

- all internal features
- plus context

This is the feature group most aligned with the paper thesis.

## 7. Temporal trajectory construction

For temporal visualization and temporal-summary extraction:

1. take the reasoning segment at step \(t\)
2. normalize token progress to \([0,1]\)
3. resample or bin the trajectory into fixed progress bins
4. compute:
   - direct curves for visualization
   - late-window summaries for prediction

Typical signals visualized:

- aligned `p_hack` trajectory
- aligned entropy trajectory
- optionally predicted next-step risk

Typical conditioning:

- `bad_action_{t+1}=1` vs `bad_action_{t+1}=0`
- `exploit_action_{t+1}=1` vs `exploit_action_{t+1}=0`

These temporal plots are **diagnostic visualizations**. They are not themselves the classifier.

## 8. Classifier

The next-step prediction analyses use a **logistic regression classifier**.

In the repository, this is implemented in helper functions such as:

- `fit_logreg(...)`
- `predict_logreg(...)`

The logistic regression is used because:

- it is simple and interpretable
- it allows clear feature-group ablations
- it avoids introducing unnecessary model complexity into the main scientific claim

The classifier is applied to step-level feature rows and predicts the next-step binary label.

## 9. Cross-validation protocol

The prediction analyses use **grouped cross-validation by episode**.

This means:

- all steps from the same episode are kept together in a fold
- train/test splitting is done at the episode level, not at the individual step level

This is important because otherwise neighboring steps from the same episode could leak trajectory information across train and test.

Operationally:

- group key is usually `episode_id` or `source_file`
- each held-out group is predicted out-of-fold
- final metrics are computed from the concatenated out-of-fold predictions

## 10. Preprocessing inside folds

Within each training fold:

1. numeric feature columns are converted to floats
2. missing values are imputed using training-fold means
3. numeric features are standardized using training-fold mean and standard deviation
4. the same transformation is applied to the held-out fold

Categorical context variables are one-hot encoded when used.

This prevents test-fold leakage from preprocessing.

## 11. Metrics

The main reported metrics are:

- AUROC
- AUPRC
- AUPRC gain over base rate
- Recall@10%
- Recall@20%

### Why AUPRC gain

Many risk targets are imbalanced, especially exploit-like actions. In such cases:

- raw accuracy is not informative
- AUROC can be optimistic when negatives dominate
- AUPRC better reflects positive-class retrieval

We therefore often report:

\[
\mathrm{AUPRC\ gain} = \mathrm{AUPRC} - \mathrm{base\ rate}.
\]

This directly measures improvement over the prevalence baseline.

### Why AUROC is also shown

AUROC is included because:

- it is a widely recognized ranking metric
- many readers expect it
- it complements AUPRC for headline comparison

### Why Recall@20 is useful

Recall@20 gives an intervention-style interpretation:

> if we flag the top 20% riskiest states, how many true risky states do we catch?

This is helpful for linking monitoring to possible deployment-time gating or steering.

## 12. Stability criteria

Not every family / adapter / TTC / target cell is reliable.

We mark conditions unstable or omit them when:

- positives are too few
- negatives are too few
- temporal traces are missing or invalid

Typical thresholds in the analysis code are on the order of:

- fewer than 5 positives or negatives for some intermediate exports
- fewer than 10 positives for cleaner paper-facing summaries

The exact threshold can vary slightly by script, but the paper should state the chosen threshold explicitly in the final version.

## 13. TTC / reasoning budget handling

TTC is treated as an explicit experimental axis.

Methodologically:

- TTC is automatically detected from available runs
- numeric TTC values are preserved in order
- unconstrained/direct/react-baseline settings are treated as a separate regime rather than forced onto the numeric axis

The paper should not present TTC=32 as the whole story. It is at most a representative slice.

## 14. Behavior vs prediction vs temporal diagnostics

The methodology has three distinct layers:

### 14.1 Behavioral transfer

Question:

> Does reward-hack fine-tuning change action behavior in the environment?

This uses action rates and environment outcomes.

### 14.2 Next-step prediction

Question:

> Can step-\(t\) internal and contextual features predict risky behavior at step \(t+1\)?

This uses the logistic-regression prediction pipeline.

### 14.3 Temporal diagnostics

Question:

> What does the within-step internal trajectory look like before risky vs non-risky next actions?

This uses aligned curves, heatmaps, and late-window summaries.

These layers should not be conflated.

## 15. Steering / intervention methodology

The steering analyses are evaluated separately from the classifier.

They ask:

> If the model is in a risky internal regime, does steering the reward-hack direction reduce exploit-like behavior?

Typical reported quantities:

- steering rate
- change in proxy score without success
- change in easy-grader count
- change in fake-completion count
- change in success or progression

Paper-safe interpretation:

- steering is an **intervention probe**
- it shows the representation remains behaviorally relevant
- it is not a universal mitigation result

## 16. What was implemented in this repository

Methodological elements implemented across the scripts include:

- step-\(t\) to step-\(t+1\) prediction framing
- logistic regression prediction
- grouped cross-validation by episode
- train-fold-only preprocessing
- multiple feature-group ablations
- TTC-aware analysis
- unconstrained-baseline handling
- temporal alignment and progress-bin resampling
- late-window summary extraction
- adapter-grouped prediction exports
- exploit-opportunity filtered rescue analyses
- episode-bootstrap uncertainty in temporal plots where applicable

## 17. Paper-safe summary

The methodological claim should be:

> We do not interpret reward-hack activation as a direct probability of failure. Instead, we treat it as one component of a context-calibrated next-step risk estimator. Step-\(t\) internal features, temporal summaries, and decision context are evaluated using grouped episode-level cross-validation to predict environment-defined labels on step \(t+1\).

## 18. Short version for the paper

Short methods paragraph:

> We treat each agent decision step as a supervised example. Internal features are extracted from the reasoning segment at step \(t\), including reward-hack activation, entropy, and temporal summaries such as late-stage change and slope. These are combined with decision-context features such as reasoning budget, step position, and environment affordances. We evaluate next-step risk using logistic regression with grouped cross-validation by episode, so that all steps from an episode are held out together. Prediction targets are environment-defined labels on step \(t+1\), such as bad action, exploit action, or low-quality purchase. This yields a context-calibrated estimate of whether an internally risky state becomes a risky next action.
