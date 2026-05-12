# PAPER_ANALYSIS_PIPELINE

## Three-Layer Logic

### Layer 1: Behavioral Transfer

This layer asks whether School-of-Reward-Hacks fine-tuning changes environment action selection.

- In Gameable ALFWorld, we compare bad_action, gameable_hack_action, easy_grader_action, fake_completion_action, proxy_score_without_success, and task success across adapters and reasoning budgets.
- In WebShop, we compare buy_rate, bad_buy_rate, low_reward_buy_rate, early_buy_rate, average return, and competence rates across adapters.

Paper-safe claim:
Reward-hack fine-tuning changes agentic behavior, but the magnitude and monotonicity of the effect depend on model family, adapter mixture, and reasoning budget.

### Layer 2: Temporal Monitoring and Next-Step Prediction

This layer asks whether internal state at step t predicts undesirable action at step t+1.

- We treat TTC / reasoning budget as an experimental axis, not as a fixed constant.
- We automatically detect available TTC regimes and also treat direct / unconstrained / react_baseline runs as a separate regime when present.
- We compute step-level entropy and p_hack summaries, late-window features, and grouped cross-validated prediction metrics.
- We generate:
  - all-TTC prediction summaries
  - best-TTC temporal separability figures
  - representative-slice figures
  - progress × TTC heatmaps

Paper-safe claim:
Internal signals at step t provide budget-dependent predictive information about bad_action_{t+1}; this signal is adapter-dependent and should not be reduced to a universal late-stage p_hack spike.

### Layer 3: Intervention / Steering

This layer asks whether predictive internal signals support intervention.

- We compare steered conditions against unsteered baselines.
- We focus on exploit reduction, easy grader reduction, fake completion reduction, and success preservation.
- If multiple TTC-specific steering slices exist, they should be analyzed separately; otherwise we treat current steering outputs as regime-level summaries.

Paper-safe claim:
Steering can reduce exploit behavior in selected mixed-adapter regimes, but this is not yet a universal mitigation.

## TTC Handling

- TTC is automatically detected from available tables or step-level data.
- Numeric TTC values are preserved as an ordered axis.
- Unconstrained/direct/no-TTC runs are labeled as `unconstrained` rather than forced onto the numeric TTC axis.
- TTC=32 is used only as a representative slice if present; otherwise a median or best-TTC regime is selected where required.

## What Each Figure/Table Answers

- `tables/alfworld_behavior_by_adapter_ttc.csv`:
  Does action selection differ by adapter and reasoning budget?
- `tables/alfworld_behavior_by_adapter_aggregate.csv`:
  What is the overall behavior profile when aggregated across TTC regimes?
- `tables/webshop_behavior_by_adapter.csv`:
  Does the WebShop agent family show low-quality purchase behavior?
- `tables/alfworld_prediction_by_ttc_adapter_featuregroup.csv`:
  Which step-t feature group predicts bad_action_{t+1} for each adapter and budget?
- `tables/alfworld_prediction_best_ttc.csv`:
  Which TTC regime gives the clearest next-step prediction signal?
- `figures/fig_alfworld_prediction_vs_ttc_qwen.*`:
  How does next-step predictiveness vary with reasoning budget?
- `figures/fig_alfworld_temporal_separability_qwen_best_ttc.*`:
  Do step-t internal trajectories separate future bad and non-bad actions?
- `figures/fig_alfworld_temporal_separability_qwen_ttc_progress_heatmap.*`:
  Where in progress × TTC space does separability emerge?
- `figures/fig_alfworld_latewindow_delta_vs_ttc_qwen.*`:
  How do late-window separability effects vary with reasoning budget?
- `tables/alfworld_steering_effects.csv`:
  What intervention deltas are observed relative to unsteered baselines?
- `figures/fig_alfworld_steering_frontier.*`:
  What is the tradeoff between steering rate and exploit reduction?

## Claims to Avoid

- p_hack always rises before bad actions
- late-stage p_hack increase is universal
- entropy always means danger
- steering fully prevents reward hacking
- all results are monotonic in hack mixture strength
- TTC=32 alone represents all reasoning budgets

## Strong Final Claim

Internal signals at step t, including entropy and p_hack, provide adapter- and budget-dependent predictive information about undesirable actions at step t+1. In controlled proxy-exploit environments, these signals can support steering that reduces exploitative action selection in selected mixed-adapter regimes.
