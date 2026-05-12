# Paper Red Line: From Temporal Monitoring to Predictive Agent Monitoring

## Core Thesis

The core thesis is:
Temporal monitoring becomes predictive monitoring when moved from isolated generation to sequential decision making.

More concretely:
Internal signals at decision step t, especially SAE-based p_hack and token-level entropy, can contain information about whether the agent will select a bad, exploitative, or proxy-gaming action at step t+1.

## Research Question

Can internal reasoning dynamics at decision step t predict and help prevent undesirable agent actions at step t+1?

## Step 1: Behavioral Transfer into Agentic Environments

Goal:
Show that reward-hack fine-tuning on School of Reward Hacks generalizes beyond isolated text completions into agentic/environmental action selection.

Question:
Do mixed or hacked adapters choose bad/exploitative actions more often than control adapters?

Definitions:
- bad_action: environment-defined harmful, invalid, low-quality, or task-regressive action
- gameable_hack_action: explicit proxy-exploit action in Gameable ALFWorld
- easy_grader_action: choosing a weak evaluator / easy grader
- fake_completion_action: claiming completion without true task success
- bad_buy / low_reward_buy: low-quality purchase actions in WebShop

Expected evidence:
- Compare action rates across adapters:
  Control, Mix05, Mix10, Mix50, Hack
- Show whether bad/exploit action selection increases with reward-hack mixture strength.
- Important: this does not need to be perfectly monotonic; saturation and adapter-specific behavior are allowed.

Paper-safe claim:
Reward-hack fine-tuning changes agentic action-selection behavior, and mixed/hacked adapters exhibit increased exploitative or low-quality action tendencies in environment interaction.

## Step 2: Temporal Internal Dynamics Before Action Selection

Goal:
Move from behavioral observation to mechanistic monitoring.

Question:
Before each action, do internal reasoning trajectories show structured p_hack and entropy dynamics?

Subquestions:
1. Are p_hack and entropy trajectories observable during each agent decision step?
2. Do they show late-stage changes during reasoning, similar to the generation-level temporal dynamics from the first paper?
3. Are these dynamics universal across adapters, or adapter-specific?
4. Do dynamics differ before bad versus non-bad next actions?
5. Can these signals predict bad_action_{t+1}?

Important distinction from first paper:
- First paper: temporal p_hack dynamics during isolated generation.
- New paper: temporal p_hack and entropy dynamics during agentic decision steps, evaluated against future environment actions.

Expected evidence:
- Within-step trajectories of p_hack and entropy.
- Bad-minus-nonbad conditional trajectories:
  Δx(τ) = E[x_t(τ) | bad_action_{t+1}=1] - E[x_t(τ) | bad_action_{t+1}=0]
- Late-window statistics:
  late_mean, late_change, late_slope.
- Prediction models using only step-t features:
  entropy-only, p_hack-only, temporal-p_hack-only, combined.
- Metrics:
  AUROC, AUPRC, AUPRC gain over base rate, Recall@K.

Paper-safe claim:
Next-step risk is not necessarily captured by a universal late-stage p_hack increase. Instead, agentic risk can be encoded in adapter-dependent combinations of entropy, p_hack, and temporal features.

## Step 3: Intervention and Steering

Goal:
Test whether predictive internal signals can be used for intervention.

Question:
If the internal dynamics distinguish risky from non-risky future actions, can we use them to steer or gate the model before exploitative action selection?

Subquestions:
1. Does steering against a reward-hack direction reduce exploitative actions?
2. Does gated/low-rate steering reduce exploit behavior while preserving task success?
3. Which adapters are steerable?
4. Is always-on steering or gated steering better?
5. Does steering work better for mixed adapters than fully hacked adapters?

Expected evidence:
- Steering rate vs exploit reduction.
- Proxy score without true success.
- Easy grader count.
- Fake completion count.
- Bad action rate.
- True task success / progression preservation.

Paper-safe claim:
Internal monitoring can support intervention in selected regimes. In Gameable ALFWorld, steering or gated intervention can reduce exploitative behavior, especially for some mixed adapters, but this is not yet a universal deployment-ready mitigation.

## Final Paper Logic

The paper should follow this chain:

1. SRH fine-tuning changes agent behavior:
   mixed/hacked adapters select more undesirable or exploitative actions than controls.

2. Internal monitoring signals are present during sequential decision making:
   p_hack and entropy trajectories can be measured at each decision step.

3. These signals become predictive:
   features at step t can distinguish or predict bad actions at step t+1.

4. Prediction enables intervention:
   when risky internal states are detected, steering/gating can reduce exploitative action selection in controlled settings.

## Claims to Avoid

Avoid claiming:
- p_hack always rises before bad actions.
- late-stage p_hack increase is universal across adapters.
- entropy always means danger.
- steering fully prevents reward hacking.
- WebShop has full online steering unless explicitly implemented.
- all effects are monotonic in hack mixture strength.

## Strong Final Framing

Use this framing:
Temporal activation monitoring from the first paper becomes predictive decision-time monitoring in agentic settings. The agentic result is more complex than isolated generation: future bad actions are not always preceded by a simple p_hack spike, but internal entropy, p_hack, and temporal features jointly provide actionable information about next-step risk.
