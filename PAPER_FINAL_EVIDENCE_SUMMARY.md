# PAPER_FINAL_EVIDENCE_SUMMARY

## 1. What does the data actually show?

The unified analysis indicates that the reward-hack monitor transfers into agentic settings as a latent policy-state descriptor: conditions separate clearly in mean reasoning `p_hack` and entropy, but immediate action-level risk is not a simple function of raw `p_hack` alone.

1. Does p_hack remain structured in agentic reasoning?
Yes, at the condition level it remains active and structured across environments, adapters, and TTC regimes.

2. Does it separate adapter regimes?
Yes. Adapter conditions separate in the internal risk-state map, especially within Qwen and in mixed/hack settings.

3. Is p_hack alone a direct next-action predictor?
No. It is informative as a latent policy-state descriptor but not a universal immediate action-risk predictor.

4. Does entropy add decision-local information?
Yes, often substantially, especially in next-step prediction ablations.

5. Does context improve prediction?
In selected targets/regimes, internal-plus-context improves over internal-only features.

6. How does TTC/CoT budget affect monitor semantics?
TTC shifts both the internal risk-state distribution and prediction quality, but not monotonically.

7. Does steering reduce exploit behavior in selected regimes?
Yes, selected mixed-adapter steering conditions reduce proxy/exploit behavior while preserving success.

## 2. Which claims are supported?

- `p_hack` remains active and structured in agentic reasoning and separates adapter regimes at the condition level.
- Raw `p_hack` is not a simple monotonic readout of exploit/bad behavior across environments and adapters.
- Entropy and mixed internal/context feature sets can outperform raw `p_hack` alone for next-step prediction in selected targets/regimes.
- Steering against the reward-hack direction reduces exploit behavior in selected mixed-adapter regimes.

## 3. Which claims are not supported?

- `p_hack` universally rises before bad actions.
- late-stage `p_hack` is a universal exploit predictor.
- entropy always indicates danger.
- steering fully prevents reward hacking.
- all effects are monotonic in mixture strength.

## 4. Which figures should go in the main paper?

- [main_internal_risk_state_map_qwen.png](/home/patrick/vllmPatrickMonitoring/figures_final_evidence/main_internal_risk_state_map_qwen.png)
- [main_context_calibrated_prediction_ablation.png](/home/patrick/vllmPatrickMonitoring/figures_final_evidence/main_context_calibrated_prediction_ablation.png)
- [main_risk_state_vs_behavior_qwen.png](/home/patrick/vllmPatrickMonitoring/figures_final_evidence/main_risk_state_vs_behavior_qwen.png) or [main_ttc_budget_monitor_semantics.png](/home/patrick/vllmPatrickMonitoring/figures_final_evidence/main_ttc_budget_monitor_semantics.png)
- [main_steering_intervention_probe.png](/home/patrick/vllmPatrickMonitoring/figures_final_evidence/main_steering_intervention_probe.png) if space allows

## 5. Which figures should go in the appendix?

- [appendix_internal_risk_state_map_all_models.png](/home/patrick/vllmPatrickMonitoring/figures_final_evidence/appendix_internal_risk_state_map_all_models.png)
- [appendix_temporal_bad_vs_nonbad.png](/home/patrick/vllmPatrickMonitoring/figures_final_evidence/appendix_temporal_bad_vs_nonbad.png)
- [appendix_temporal_exploit_opportunity.png](/home/patrick/vllmPatrickMonitoring/figures_final_evidence/appendix_temporal_exploit_opportunity.png)
- [appendix_latewindow_delta_vs_ttc.png](/home/patrick/vllmPatrickMonitoring/figures_final_evidence/appendix_latewindow_delta_vs_ttc.png)
- [appendix_progress_ttc_heatmaps.png](/home/patrick/vllmPatrickMonitoring/figures_final_evidence/appendix_progress_ttc_heatmaps.png)

## 6. What is the cleanest ICML MechInterp story?

The cleanest story is: `p_hack` transfers from isolated generation to agents as a latent reward-hack policy-state descriptor. It separates adapter regimes, but immediate action-level risk is better captured by entropy and context-calibrated internal features. Steering remains useful as a targeted intervention probe rather than a universal mitigation.