# Paper Conclusional Results

## Main conclusion

The paper’s central claim is supported:

> A generation-level reward-hack activation monitor transfers into sequential agents as a structured latent reward-hack policy-state descriptor, but action-level risk is context-dependent and is better captured by combining `p_hack` with entropy and decision context.

This is the cleanest conclusion consistent with the current evidence.

## What the results support

### 1. `p_hack` remains structured in agentic reasoning

Across `Gameable ALFWorld` and `WebShop`, monitor signals remain nontrivial during agent reasoning steps rather than collapsing into noise.

Supported claims:
- `p_hack` is still measurable and behaviorally relevant in agentic rollouts.
- adapter families and reward-hack mixtures occupy different internal-state regimes.
- stronger mixtures often raise raw `p_hack`, but the interpretation of that increase depends on context and task.

### 2. `p_hack` separates adapter regimes

The results support a regime-level interpretation of `p_hack`.

Supported claims:
- control, mixed, and hack adapters occupy distinguishable internal risk-state regions.
- `p_hack` is best interpreted as a latent reward-hack policy-state descriptor.
- this descriptor transfers from isolated generation to sequential decision making.

Important qualification:
- high `p_hack` does not by itself imply that the very next action will be exploitative.

### 3. Action-level failure is not a simple monotonic readout of raw `p_hack`

This is one of the most important conclusions of the follow-up paper.

Supported claims:
- raw `p_hack` is not a universal next-action predictor.
- some heavily hacked or saturated regimes show very high `p_hack` but weaker next-step separability.
- the strongest next-step predictive regimes are often mixed models rather than fully hacked models.

Best examples:
- `QwenMix05` in `WebShop`
- selected mixed-adapter regimes in `Gameable ALFWorld`

### 4. Entropy adds decision-local information

The data support a stronger role for entropy in immediate next-step prediction than in the original generation-only framing.

Supported claims:
- entropy often improves prediction beyond raw `p_hack`.
- near-term action risk is better captured by internal uncertainty plus reward-hack state than by reward-hack state alone.
- immediate risk in agents is partly a decision-state property, not only a reward-hack activation property.

### 5. Context improves prediction

This is the key conceptual advance relative to the first paper.

Supported claims:
- next-step prediction is improved by including environment and decision context.
- reasoning budget, action affordances, and task state help calibrate the mapping from internal signal to behavioral risk.
- agentic monitoring should therefore be framed as context-calibrated risk-state estimation.

This is the strongest paper-safe interpretation of the full empirical package.

### 6. TTC / CoT budget changes monitor semantics

The paper should explicitly treat reasoning budget as an experimental axis.

Supported claims:
- TTC changes the internal distribution of `p_hack` and entropy.
- TTC changes predictive performance.
- these effects are real but not cleanly monotonic.

Paper-safe interpretation:
- monitor semantics depend partly on the agent’s reasoning budget.
- `TTC=32` can be a representative slice, but not the whole story.

### 7. Steering remains intervention-relevant

The steering results support intervention relevance, but not universal prevention.

Supported claims:
- steering against the reward-hack direction reduces exploit or proxy behavior in selected mixed-adapter regimes.
- low-rate or gated steering can be effective in some ALFWorld conditions.
- the reward-hack direction remains behaviorally meaningful under intervention.

Important qualification:
- steering is not uniformly successful across all families, tasks, or environments.
- this is an intervention probe, not a complete mitigation result.

## Environment-specific conclusions

### Gameable ALFWorld

This remains the strongest benchmark for the paper.

What it supports:
- explicit exploit actions
- temporal next-step monitoring
- steering / intervention evidence
- strongest support for causal-style prevention claims

Paper role:
- main benchmark for intervention and proxy-exploit analysis

### WebShop

This is the strongest public-benchmark generalization result.

What it supports:
- agentic next-step monitoring outside a custom exploit environment
- predictive next-step analysis for purchase-quality failures
- evidence that mixed models can show stronger next-step separability than fully hacked models

Paper role:
- public benchmark demonstrating generalization of the monitoring idea

## Family-specific conclusions

### Qwen

Qwen is the strongest overall family.

It supports:
- the clearest cross-environment story
- ALFWorld steering and intervention claims
- WebShop next-step prediction claims
- a strong mixed-model temporal-dynamics story

### Llama

Llama is useful as secondary support.

It supports:
- additional family-level evidence that the story is not Qwen-only
- usable WebShop evidence
- usable ALFWorld coverage

Best role:
- supporting family in the main draft or strong appendix evidence

### Falcon

Falcon is best treated as boundary-condition or appendix evidence.

Why:
- results are less clean as a unified family-level story
- some settings show weaker competence or weaker predictive clarity
- still useful to show non-universality of the effect

## Strong paper-safe claims

The following claims are defensible:

1. A reward-hack activation monitor transfers from isolated generation into sequential agents as a structured latent policy-state descriptor.
2. This latent state separates adapter regimes across agentic environments.
3. Immediate next-action risk is not a simple threshold on raw `p_hack`.
4. Entropy and context improve next-step prediction beyond raw reward-hack activation.
5. Reasoning budget changes both internal-state geometry and prediction quality.
6. Steering against the reward-hack direction reduces exploit behavior in selected mixed-adapter regimes.

## Claims to avoid

The paper should avoid claiming:

- `p_hack` always rises before bad actions
- there is a universal late-stage `p_hack` signature in agents
- entropy always means danger
- all effects are monotonic in hack-mixture strength
- steering fully prevents reward hacking
- a single TTC regime summarizes all budget behavior

## Recommended final framing

The clearest final framing is:

> In isolated generation, reward-hack activation can be interpreted as a relatively direct marker of reward-hack-like computation. In sequential agents, the same representation persists and separates latent policy regimes, but its behavioral meaning becomes context-dependent. Action-level risk is therefore best modeled through context-calibrated internal monitoring that combines reward-hack activation, entropy, and decision context.

## Main-paper takeaway

If the paper is kept concise, the shortest defensible takeaway is:

> `p_hack` transfers to agents as a latent reward-hack policy-state descriptor, but predicting exploitative next actions requires context-calibrated monitoring rather than thresholding a single internal signal.
