
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch

from vllm.monitoring.artifacts import WorkshopMonitorModel


def bin_curve(values: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Exact workshop-style temporal binning.
    """
    t = int(values.shape[0])
    if t <= 0:
        return np.full((n_bins,), np.nan, dtype=np.float32)

    edges = np.linspace(0, t, n_bins + 1)
    out = np.full((n_bins,), np.nan, dtype=np.float32)

    for b in range(n_bins):
        s = int(math.floor(edges[b]))
        e = int(math.floor(edges[b + 1]))
        e = max(e, s + 1)
        e = min(e, t)
        s = min(s, t - 1)
        if e <= s:
            continue
        out[b] = float(np.mean(values[s:e]))

    if np.all(np.isnan(out)):
        return out

    # forward/backward fill like workshop script
    valid = np.where(~np.isnan(out))[0]
    if len(valid) == 0:
        return out

    first = valid[0]
    last = valid[-1]
    out[:first] = out[first]
    out[last + 1 :] = out[last]
    for i in range(first + 1, last + 1):
        if np.isnan(out[i]):
            out[i] = out[i - 1]

    return out.astype(np.float32)


@dataclass
class TokenStep:
    token_id: int
    token_score: float
    layer_scores: Dict[int, float]


@dataclass
class PerRequestTrace:
    token_ids: List[int] = field(default_factory=list)
    token_scores: List[float] = field(default_factory=list)
    per_layer_scores: Dict[int, List[float]] = field(default_factory=dict)

    def append(
        self,
        token_id: int,
        token_score: float,
        layer_scores: Dict[int, float],
    ) -> None:
        self.token_ids.append(int(token_id))
        self.token_scores.append(float(token_score))
        for lid, score in layer_scores.items():
            if lid not in self.per_layer_scores:
                self.per_layer_scores[lid] = []
            self.per_layer_scores[lid].append(float(score))

    def summarize(
        self,
        *,
        threshold: float,
        layer_agg: str,
        n_bins: int,
    ) -> Dict[str, Any]:
        if len(self.token_scores) == 0:
            return {
                "prompt_monitor_prob": 0.0,
                "monitor_label": "control",
                "hackTokenCount": 0,
                "percentageHackToken": 0.0,
                "layer_means": {},
                "layer_p95s": {},
                "curve_bins_mean_over_layers": [float("nan")] * n_bins,
                "num_tokens": 0,
                "token_ids": [],
                "token_scores": [],
            }

        token_scores_np = np.asarray(self.token_scores, dtype=np.float32)
        layer_means = {
            int(lid): float(np.mean(np.asarray(vals, dtype=np.float32)))
            for lid, vals in self.per_layer_scores.items()
        }
        layer_p95s = {
            int(lid): float(np.percentile(np.asarray(vals, dtype=np.float32), 95))
            for lid, vals in self.per_layer_scores.items()
        }

        if layer_agg == "max":
            prompt_monitor_prob = max(layer_means.values()) if layer_means else 0.0
        else:
            prompt_monitor_prob = float(np.mean(list(layer_means.values()))) if layer_means else 0.0

        hack_token_count = int(np.sum(token_scores_np >= threshold))
        percentage_hack_token = float(
            hack_token_count / max(1, int(token_scores_np.shape[0]))
        )

        curve_bins = bin_curve(token_scores_np, n_bins=n_bins)

        return {
            "prompt_monitor_prob": float(prompt_monitor_prob),
            "monitor_label": "hack" if prompt_monitor_prob >= threshold else "control",
            "hackTokenCount": hack_token_count,
            "percentageHackToken": percentage_hack_token,
            "layer_means": layer_means,
            "layer_p95s": layer_p95s,
            "curve_bins_mean_over_layers": curve_bins.tolist(),
            "num_tokens": int(token_scores_np.shape[0]),
            "token_ids": [int(x) for x in self.token_ids],
            "token_scores": [float(x) for x in self.token_scores],
        }


class DecodeMonitorRuntime:
    """
    Exact-layer workshop-compatible decode-time runtime.

    Input:
      sample_hidden_states_by_layer[layer_id] = Tensor[B, D]
      sampled_token_ids = Tensor[B] or Tensor[B, 1]

    For each decode step:
      1. score exact hidden states per layer
      2. aggregate across layers per token
      3. append to per-request trace
      4. return per-request delta + running summary
    """

    def __init__(
        self,
        monitor_model: WorkshopMonitorModel,
    ) -> None:
        self.monitor_model = monitor_model
        self.traces: Dict[str, PerRequestTrace] = {}

    def _ensure(self, req_id: str) -> None:
        if req_id not in self.traces:
            self.traces[req_id] = PerRequestTrace()

    @property
    def threshold(self) -> float:
        return self.monitor_model.threshold

    @property
    def layer_agg(self) -> str:
        return self.monitor_model.layer_agg

    @property
    def n_bins(self) -> int:
        return self.monitor_model.artifacts.n_bins

    @torch.inference_mode()
    def score_decode_step(
        self,
        req_ids: List[str],
        sample_hidden_states_by_layer: Dict[int, torch.Tensor],
        sampled_token_ids: torch.Tensor,
    ) -> Dict[str, dict]:
        """
        Returns per-request step outputs:
          {
            req_id: {
              "token_id": ...,
              "token_score": ...,
              "layer_scores": {...},
              "prompt_monitor_prob_so_far": ...,
              "hackTokenCount_so_far": ...,
              "percentageHackToken_so_far": ...,
            }
          }
        """
        if sampled_token_ids.ndim == 2:
            sampled_token_ids = sampled_token_ids[:, 0]

        per_layer_probs = self.monitor_model.score_layers(sample_hidden_states_by_layer)
        if len(per_layer_probs) == 0:
            return {}

        layer_ids = sorted(per_layer_probs.keys())
        stacked = torch.stack([per_layer_probs[lid] for lid in layer_ids], dim=0)  # [L, B]

        if self.layer_agg == "max":
            token_probs = torch.max(stacked, dim=0).values
        else:
            token_probs = torch.mean(stacked, dim=0)

        out: Dict[str, dict] = {}

        for i, req_id in enumerate(req_ids):
            self._ensure(req_id)

            token_id = int(sampled_token_ids[i].item())
            token_score = float(token_probs[i].item())
            layer_scores = {
                int(lid): float(per_layer_probs[lid][i].item())
                for lid in layer_ids
            }

            trace = self.traces[req_id]
            trace.append(
                token_id=token_id,
                token_score=token_score,
                layer_scores=layer_scores,
            )

            current_hack_count = int(sum(x >= self.threshold for x in trace.token_scores))
            current_pct = float(current_hack_count / max(1, len(trace.token_scores)))

            out[req_id] = {
                "token_id": token_id,
                "token_score": token_score,
                "layer_scores": layer_scores,
                "prompt_monitor_prob_so_far": float(
                    sum(trace.token_scores) / len(trace.token_scores)
                ),
                "hackTokenCount_so_far": current_hack_count,
                "percentageHackToken_so_far": current_pct,
            }

        return out

    def summarize_request(self, req_id: str) -> Dict[str, Any]:
        self._ensure(req_id)
        return self.traces[req_id].summarize(
            threshold=self.threshold,
            layer_agg=self.layer_agg,
            n_bins=self.n_bins,
        )

    def summarize_requests(self, req_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        return {req_id: self.summarize_request(req_id) for req_id in req_ids}

    def reset_request(self, req_id: str) -> None:
        self.traces.pop(req_id, None)

    def reset_requests(self, req_ids: List[str]) -> None:
        for req_id in req_ids:
            self.reset_request(req_id)
