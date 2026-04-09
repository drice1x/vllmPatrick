from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import torch
import torch.nn as nn


def _torch_dtype_from_string(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {name}")


def normalize_like_workshop(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Match the workshop script's normalize_and_downcast(), except we keep tensors
    on-device for serving.

    Input:
      hidden_states: [B, D] or [..., D]

    Output:
      normalized: [B, D] (flattened over all leading dims)
    """
    x = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous().float()
    d = float(x.shape[-1])
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True))
    x_norm = math.sqrt(d) * x / (rms + 1e-8)
    return x_norm


class WorkshopSparseAutoencoder(nn.Module):
    """
    Exact SAE architecture from the workshop script:

      encoder = nn.Linear(input_dim, hidden_dim, bias=False)
      decoder = nn.Linear(hidden_dim, input_dim, bias=False)
      h = relu(encoder(x))
    """
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.encoder(x))
        out = self.decoder(h)
        return out, h

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))


class EffectiveLinearHead(nn.Module):
    """
    Optional serving-optimized replacement for the sklearn pipeline.

    Expected checkpoint format:
      {
        "weight": Tensor[latent_dim],
        "bias": Tensor[] or float
      }

    Returns sigmoid(logit) in [0, 1].
    """
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        if weight.ndim != 1:
            raise ValueError(f"Expected 1D weight, got shape={tuple(weight.shape)}")
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias.reshape(()))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = z @ self.weight + self.bias
        return torch.sigmoid(logits)


@dataclass
class LayerArtifacts:
    layer_id: int
    sae: WorkshopSparseAutoencoder
    hidden_dim: int
    latent_dim: int

    # Exactly one of these should be used.
    sklearn_pipe: Any | None = None
    head: EffectiveLinearHead | None = None

    metadata: Dict[str, Any] | None = None


@dataclass
class ModelMonitorArtifacts:
    model_family: str
    layers: List[int]
    layer_artifacts: Dict[int, LayerArtifacts]
    device: torch.device
    dtype: torch.dtype
    layer_agg: str = "mean"
    threshold: float = 0.5
    n_bins: int = 64


def load_workshop_sae_for_layer(
    path: str | Path,
    device: torch.device,
    dtype: torch.dtype,
) -> WorkshopSparseAutoencoder:
    """
    Load the real workshop SAE format:
      - encoder.weight
      - decoder.weight
    """
    sd = torch.load(path, map_location="cpu")
    enc_w = sd.get("encoder.weight", None)
    dec_w = sd.get("decoder.weight", None)

    if enc_w is None:
        raise RuntimeError(f"{path} missing encoder.weight")
    if dec_w is None:
        raise RuntimeError(f"{path} missing decoder.weight")

    hidden_dim, input_dim = enc_w.shape
    sae = WorkshopSparseAutoencoder(int(input_dim), int(hidden_dim))
    sae.load_state_dict(sd, strict=True)
    sae.to(device=device, dtype=dtype).eval()
    return sae


def load_effective_linear_head(
    path: str | Path,
    device: torch.device,
    dtype: torch.dtype,
) -> EffectiveLinearHead:
    ckpt = torch.load(path, map_location="cpu")
    weight = ckpt["weight"]
    bias = ckpt["bias"]

    if not isinstance(weight, torch.Tensor):
        weight = torch.tensor(weight)
    if not isinstance(bias, torch.Tensor):
        bias = torch.tensor(bias)

    head = EffectiveLinearHead(
        weight=weight.to(device=device, dtype=dtype),
        bias=bias.to(device=device, dtype=dtype),
    )
    head.to(device=device).eval()
    return head


def load_sklearn_pipeline(path: str | Path) -> Any:
    return joblib.load(path)


def _resolve_layer_artifacts(
    layer_id: int,
    sae_dir: Path,
    pca_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> LayerArtifacts:
    """
    Workshop-compatible layout:
      SAE: sae_{L}.pth
      classifier: pca_lin_layer{L}.joblib

    Optional optimized layout:
      head_layer{L}.pt
    """
    sae_path = sae_dir / f"sae_{layer_id}.pth"
    if not sae_path.exists():
        raise FileNotFoundError(f"Missing SAE checkpoint: {sae_path}")

    sae = load_workshop_sae_for_layer(sae_path, device=device, dtype=dtype)
    hidden_dim = int(sae.encoder.in_features)
    latent_dim = int(sae.encoder.out_features)

    head_path = pca_dir / f"head_layer{layer_id}.pt"
    pipe_path = pca_dir / f"pca_lin_layer{layer_id}.joblib"

    if head_path.exists():
        head = load_effective_linear_head(head_path, device=device, dtype=dtype)
        sklearn_pipe = None
    elif pipe_path.exists():
        head = None
        sklearn_pipe = load_sklearn_pipeline(pipe_path)
    else:
        raise FileNotFoundError(
            f"Missing classifier for layer {layer_id}: "
            f"expected {head_path} or {pipe_path}"
        )

    return LayerArtifacts(
        layer_id=layer_id,
        sae=sae,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        sklearn_pipe=sklearn_pipe,
        head=head,
        metadata={
            "sae_path": str(sae_path),
            "classifier_path": str(head_path if head_path.exists() else pipe_path),
        },
    )


class WorkshopLayerMonitor:
    """
    Scores one exact layer using:
      hidden_state -> workshop normalization -> SAE -> classifier
    """

    def __init__(self, artifacts: LayerArtifacts) -> None:
        self.artifacts = artifacts

    @torch.inference_mode()
    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, D]
        returns: [B] probabilities
        """
        # Match workshop normalization in float32 first.
        x = normalize_like_workshop(hidden_states)

        sae = self.artifacts.sae
        sae_dtype = sae.encoder.weight.dtype
        sae_device = sae.encoder.weight.device

        # Cast normalized activations to the SAE dtype/device before the SAE linear.
        x = x.to(device=sae_device, dtype=sae_dtype)

        h = sae.encode(x)

        if self.artifacts.head is not None:
            head = self.artifacts.head
            probs = head(h).reshape(-1)
            # Return float32 for downstream stability / serialization consistency.
            return probs.to(hidden_states.device, dtype=torch.float32)

        # Exact workshop-compatible sklearn path.
        h_np = h.detach().float().cpu().numpy().astype("float32")
        pipe = self.artifacts.sklearn_pipe
        if hasattr(pipe, "predict_proba"):
            probs_np = pipe.predict_proba(h_np)[:, 1]
        else:
            raw = pipe.decision_function(h_np)
            probs_np = 1.0 / (1.0 + __import__("numpy").exp(-raw))

        return torch.from_numpy(probs_np).to(hidden_states.device, dtype=torch.float32)


class WorkshopMonitorModel:
    """
    Multi-layer monitor matching the workshop paper semantics.
    """

    def __init__(self, artifacts: ModelMonitorArtifacts) -> None:
        self.artifacts = artifacts
        self.layer_monitors: Dict[int, WorkshopLayerMonitor] = {
            lid: WorkshopLayerMonitor(layer_art)
            for lid, layer_art in artifacts.layer_artifacts.items()
        }

    @property
    def layers(self) -> List[int]:
        return self.artifacts.layers

    @property
    def layer_agg(self) -> str:
        return self.artifacts.layer_agg

    @property
    def threshold(self) -> float:
        return self.artifacts.threshold

    @torch.inference_mode()
    def score_layers(
        self,
        sample_hidden_states_by_layer: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        Input:
          sample_hidden_states_by_layer[layer_id] = Tensor[B, D]
        Output:
          dict[layer_id] = Tensor[B]
        """
        out: Dict[int, torch.Tensor] = {}
        for layer_id in self.layers:
            hs = sample_hidden_states_by_layer.get(layer_id, None)
            if hs is None:
                continue
            out[layer_id] = self.layer_monitors[layer_id].score(hs)
        return out


class MonitorArtifactRegistry:
    """
    Small in-process cache for loaded monitor models.

    Keys are based on:
      (model_family, layers, sae_dir, pca_dir, device, dtype, layer_agg, threshold, n_bins)
    """

    def __init__(self) -> None:
        self._cache: Dict[tuple, WorkshopMonitorModel] = {}

    def get(
        self,
        *,
        model_family: str,
        sae_dir: str | Path,
        pca_dir: str | Path,
        layers: List[int],
        device: str | torch.device = "cuda",
        dtype: str | torch.dtype = "float16",
        layer_agg: str = "mean",
        threshold: float = 0.5,
        n_bins: int = 64,
    ) -> WorkshopMonitorModel:
        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(dtype, str):
            dtype = _torch_dtype_from_string(dtype)

        sae_dir = Path(sae_dir)
        pca_dir = Path(pca_dir)

        key = (
            model_family,
            tuple(layers),
            str(sae_dir),
            str(pca_dir),
            str(device),
            str(dtype),
            layer_agg,
            float(threshold),
            int(n_bins),
        )
        if key in self._cache:
            return self._cache[key]

        layer_artifacts: Dict[int, LayerArtifacts] = {}
        for layer_id in layers:
            layer_artifacts[layer_id] = _resolve_layer_artifacts(
                layer_id=layer_id,
                sae_dir=sae_dir,
                pca_dir=pca_dir,
                device=device,
                dtype=dtype,
            )

        artifacts = ModelMonitorArtifacts(
            model_family=model_family,
            layers=list(layers),
            layer_artifacts=layer_artifacts,
            device=device,
            dtype=dtype,
            layer_agg=layer_agg,
            threshold=threshold,
            n_bins=n_bins,
        )
        model = WorkshopMonitorModel(artifacts)
        self._cache[key] = model
        return model