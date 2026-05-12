# Temporal Internal Features vs Temporal Plots

This note clarifies an important distinction for the paper:

1. **Temporal plots** show the internal trajectory itself over normalized reasoning progress.
2. **Temporal internal features** are not plots. They are scalar summary features extracted from those trajectories and then used by a classifier.

This distinction matters because some of our prediction-ablation figures contain a feature group called `temporal internal`, but that bar label does **not** mean we are plotting the full token-by-token trajectory in that figure.

## 1. What the temporal plots show

A temporal plot is a direct visualization of reasoning-time structure inside a decision step.

For a step \(t\), we take the reasoning trajectory before the next action at \(t+1\), normalize progress to \([0,1]\), and visualize a signal such as:

- reward-hack activation \(p_{\mathrm{hack}}\)
- token-level entropy
- optionally, predicted next-step risk

Then we compare trajectories conditioned on the next-step outcome, for example:

- `bad_action_{t+1} = 1` versus `bad_action_{t+1} = 0`
- `exploit_action_{t+1} = 1` versus `exploit_action_{t+1} = 0`

These figures answer questions like:

- Does the internal signal visibly evolve during reasoning?
- Do risky and non-risky next actions show different within-step trajectories?
- Is there a late-stage rise, separation, or divergence?

These are **mechanistic/diagnostic visualizations**, not predictive metrics by themselves.

## 2. What `temporal internal` means in the bar plots

In the prediction-ablation bar plots, `temporal internal` is a **feature group for the classifier**.

It does **not** display the full temporal trajectory. Instead, it summarizes the trajectory into a few scalar features such as:

- late-stage change
- late-stage slope
- optionally centroid or related timing summaries

So the workflow is:

1. Extract the token-level trajectory for \(p_{\mathrm{hack}}\) or entropy.
2. Compress that trajectory into summary statistics.
3. Feed those statistics into the classifier.
4. Report prediction metrics such as AUROC, AUPRC, AUPRC gain, or Recall@20.

Therefore, a `temporal internal` bar in a prediction figure means:

> “How useful are summary features derived from temporal structure?”

It does **not** mean:

> “Here is the temporal structure itself.”

## 3. Classifier method for temporal internal features

The classifier setup is:

- one row per decision step
- grouped cross-validation by episode
- logistic regression
- no future-label leakage in the features

For the `temporal internal` feature group, the inputs are summary features derived from the within-step trajectory, for example:

- \(p_{\mathrm{hack}}\) late change
- \(p_{\mathrm{hack}}\) late slope
- entropy late change
- entropy late slope

Conceptually:

\[
\text{temporal\_internal}_t
=
f\big(\text{trajectory of } p_{\mathrm{hack},t},\; \text{trajectory of entropy}_t\big),
\]

where \(f\) maps a trajectory to a small set of hand-designed summary statistics.

The classifier then estimates:

\[
\Pr(y_{t+1}^{\mathrm{risk}} = 1 \mid \text{temporal\_internal}_t)
\]

or, in richer feature groups,

\[
\Pr(y_{t+1}^{\mathrm{risk}} = 1 \mid z_t, c_t),
\]

where:

- \(z_t\) includes internal features such as \(p_{\mathrm{hack}}\), entropy, and temporal summaries
- \(c_t\) includes context such as TTC, step position, environment state, and related decision-context features

## 4. Why this distinction matters scientifically

The paper should avoid implying that:

- a visible late-stage rise is always present, or
- the `temporal internal` classifier feature group proves a universal trajectory signature.

A temporal feature group can help prediction even when the raw temporal plots are noisy or heterogeneous, because:

- the classifier uses compressed summaries,
- these summaries can capture weak but consistent trends,
- and performance is evaluated statistically across many steps.

So:

- **temporal plot** = direct visual evidence about trajectory shape
- **temporal internal feature group** = predictive use of compressed trajectory summaries

These are related, but they are not the same evidence.

## 5. How to explain this in the Results section

Paper-safe wording:

> The `temporal internal` ablation does not visualize full reasoning trajectories. Instead, it evaluates whether simple summary statistics derived from within-step temporal dynamics, such as late-stage slope and late-stage change, improve next-step prediction. We therefore treat trajectory plots as diagnostic visualizations and temporal-summary features as a separate predictive analysis.

## 6. What to show in the appendix

For the appendix, it is useful to include both:

### A. Raw/aligned temporal trajectory plots

These show:

- normalized progress within the reasoning segment
- risky vs non-risky next actions
- curves for \(p_{\mathrm{hack}}\) and entropy

These support the claim that temporal structure exists, but they should be presented as heterogeneous and adapter-/budget-dependent.

### B. Explicit late-change / slope visualizations

These are especially useful because they connect the appendix directly to the `temporal internal` classifier feature group.

Recommended appendix visualizations:

1. **Late-window delta vs TTC**
   - x-axis = TTC budget
   - y-axis = late-window difference between risky and non-risky trajectories
   - separate panels for \(p_{\mathrm{hack}}\) and entropy
   - one line per adapter

2. **Slope summary plot**
   - x-axis = TTC budget
   - y-axis = mean late-stage slope
   - compare risky vs non-risky next actions
   - separate panels for adapters or model families

3. **Late-change box/point plot**
   - show the distribution of late-change values by target label
   - useful for explaining why a temporal-summary feature can help even if the full trajectory plot is noisy

Paper-safe wording for appendix:

> We include late-change and late-slope summary plots in the appendix because these are the quantities used by the `temporal internal` classifier feature group. These summaries provide a more compact view of trajectory dynamics than the full token-level curves and help explain how temporal structure can support prediction even when raw trajectories are heterogeneous.

## 7. Recommended paper framing

Recommended framing for the paper:

- Main text:
  - prediction ablations use `temporal internal` as a classifier feature group
  - temporal plots are supportive diagnostics, not the main quantitative evidence

- Appendix:
  - show full aligned trajectory plots
  - show late-change and late-slope summary plots
  - explicitly connect those summary plots to the classifier features

## 8. Short version for the paper

Short explanation:

> In our prediction ablations, `temporal internal` refers to summary features extracted from within-step trajectories, such as late-stage change and slope, not to the raw token-level trajectory plot itself. We therefore separate temporal diagnostics from predictive ablations: the appendix visualizes the trajectories directly, while the main prediction tables evaluate whether compact temporal summaries improve next-step risk estimation.
