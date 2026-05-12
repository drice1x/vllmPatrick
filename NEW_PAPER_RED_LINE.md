## Interpretability Framing

This paper is not primarily an agent benchmark paper. It is a mechanistic monitoring paper.

The object of study is the reward-hack activation feature \(p_{\mathrm{hack}}\): a representation-level signal learned from contrastive control-vs-hack activations in isolated generation. The central question is how the semantics of this feature change when the model is embedded in a sequential agent loop.

The key interpretability claim is:

> A mechanistic feature can remain internally structured while its behavioral meaning changes under agentic deployment.

In our setting, \(p_{\mathrm{hack}}\) remains active and separates adapter regimes, so it transfers as a latent reward-hack policy-state descriptor. However, action-level risk is not a direct thresholded readout of \(p_{\mathrm{hack}}\). The mapping from representation to behavior is mediated by entropy, action context, environment state, and reasoning budget.

Therefore, agentic mechanistic monitoring should be context-calibrated:
\[
P(y^{risk}_{t+1}=1 \mid p_{\mathrm{hack},t}, H_t, c_t)
\]
where \(H_t\) is token-level entropy and \(c_t\) describes the decision context.

This reframes the paper from "does \(p_{\mathrm{hack}}\) predict bad actions?" to "how do internal reward-hack representations acquire action-level semantics in agents?"