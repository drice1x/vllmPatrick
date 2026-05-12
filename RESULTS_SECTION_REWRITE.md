# Results Section Rewrite

## Claim 1: Reward-hack activations transfer to agents as structured latent policy states

Across `Gameable ALFWorld` and `WebShop`, reward-hack activation remains nontrivial during agent reasoning and separates adapter regimes. In particular, control, mixed, and hack adapters occupy different regions of internal state space when summarized by mean reasoning entropy and mean reward-hack activation. This supports the interpretation that the reward-hack monitor transfers from isolated generation into sequential rollouts as a latent policy-state descriptor rather than disappearing under the distribution shift to agentic interaction.

Supporting figure/table:
- Figure 1A: `main_risk_state_transfer_panelA`
- Table 1: `main_behavior_internal_profile`

Caption:
Reward-hack activation remains structured in agent rollouts and separates adapter regimes across environments and reasoning budgets.

## Claim 2: Latent risk states do not directly determine action-level failures; behavior is context-mediated

Although reward-hack activation is structured, action-level failure is not a monotonic readout of raw activation. Conditions with higher mean activation do not uniformly exhibit higher exploit or bad-action rates, and some mixed regimes are more behaviorally separable than saturated hack regimes. This indicates that high activation is better understood as a latent reward-hack policy-state marker than as a direct action-risk score. The mapping from internal state to behavior is mediated by environment and decision context.

Supporting figure/table:
- Figure 1B: `main_risk_state_transfer_panelB`
- Table 1: `main_behavior_internal_profile`

Caption:
Reward-hack activation remains structured in agent rollouts, but action-level behavior is not a monotonic readout of raw activation.

## Claim 3: Entropy and context calibrate next-step risk estimation, and steering shows intervention relevance

Next-step prediction is possible, but not by thresholding reward-hack activation alone. Across stable ALFWorld and WebShop targets, entropy and mixed internal/context feature groups improve prediction relative to activation-only baselines, indicating that immediate risk depends on latent state plus decision uncertainty and environment affordances. Steering results further show that the reward-hack direction remains intervention-relevant: in selected mixed-adapter regimes, steering reduces proxy-exploit behavior without supporting a universal mitigation claim.

Supporting figure/table:
- Figure 2: `main_context_calibration_prediction`
- Table 2: `main_prediction_ablation_table`
- Figure 3 / Table 3: `main_steering_probe`

Caption for Figure 2:
Entropy and context-calibrated internal features improve next-step risk estimation over activation alone in selected targets and regimes.

Caption for Figure 3 / Table 3:
Steering is an intervention probe: it reduces proxy-exploit behavior in selected regimes, but is not a universal mitigation.

## Causal interpretation

The evidence does not show that reward-hack activation directly causes exploit actions. Instead, it shows that the representation remains behaviorally relevant but is mediated by environment context. Steering provides intervention evidence in selected regimes.

## Main-paper vs appendix

Main paper:
- Figure 1A/B: internal risk-state transfer and risk-state vs behavior
- Table 1: behavioral/internal profile
- Figure 2: context-calibrated prediction ablation
- Table 2: prediction ablation table
- Figure 3 or Table 3: steering probe

Appendix only:
- temporal red/blue curves
- TTC heatmaps
- per-adapter prediction curves
- Falcon full plots
- exploit-opportunity plots with low positives
- broad bad-minus-nonbad diagnostics

## Final takeaway

Prediction is possible, but not by thresholding a single activation feature. In agentic settings, risky next actions are best modeled as a function of latent reward-hack state, decision uncertainty, and environment affordances.
