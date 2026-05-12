# PAPER_FIGURE_RECOMMENDATIONS

## 1. Best behavioral transfer figure

- File: `/home/patrick/vllmPatrickMonitoring/figures_paper_pipeline/alfworld_behavior_by_ttc.png`
- Why useful: shows whether exploitative action rates vary with reasoning budget and adapter.
- Supports: behavioral transfer of reward-hack fine-tuning into agentic action selection.
- Caveats: some families may have sparse or non-monotonic patterns.
- Placement: main paper if Qwen curves are clean, otherwise appendix.

## 2. Best temporal separability figure

- File: `/home/patrick/vllmPatrickMonitoring/figures_paper_pipeline/fig_alfworld_temporal_separability_qwen_best_ttc.png`
- Why useful: directly conditions step-t signals on bad_action_{t+1} and avoids unconditional averaging.
- Supports: temporal monitoring becomes predictive monitoring.
- Caveats: selected TTC is data-driven and may not generalize uniformly across families.
- Placement: main paper.

## 3. Best prediction figure

- File: `/home/patrick/vllmPatrickMonitoring/figures_paper_pipeline/fig_alfworld_prediction_vs_ttc_qwen.png`
- Why useful: compactly shows how predictive signal quality depends on reasoning budget and feature group.
- Supports: next-step prediction is budget- and feature-dependent.
- Caveats: averages across adapters can hide adapter-specific reversals.
- Placement: main paper.

## 4. Best steering figure

- File: `/home/patrick/vllmPatrickMonitoring/figures_paper_pipeline/fig_alfworld_steering_frontier.png`
- Why useful: directly visualizes intervention rate vs exploit reduction.
- Supports: monitoring-guided steering can reduce exploit behavior in selected regimes.
- Caveats: not all families or mixtures are equally steerable.
- Placement: main paper if Qwen dominates, appendix otherwise.