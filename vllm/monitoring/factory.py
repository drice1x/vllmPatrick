from __future__ import annotations

from typing import Any, Dict

import torch

from vllm.monitoring.artifacts import MonitorArtifactRegistry
from vllm.monitoring.runtime import DecodeMonitorRuntime

# Small process-global cache
_REGISTRY = MonitorArtifactRegistry()


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_monitor_runtime(monitor_config, device):
    """
    Build the real workshop-compatible monitor runtime.

    Required config keys:
      - model_family
      - layers
      - sae_dir
      - pca_dir

    Optional:
      - dtype                (default: float16)
      - layer_agg            (default: mean)
      - threshold            (default: 0.5)
      - n_bins               (default: 64)
    """
    print("MONDBG factory.build_monitor_runtime called with:", monitor_config, device)
    model_family = _cfg_get(monitor_config, "model_family")
    layers = _cfg_get(monitor_config, "layers", None)
    sae_dir = _cfg_get(monitor_config, "sae_dir", None)
    pca_dir = _cfg_get(monitor_config, "pca_dir", None)

    if not model_family:
        raise ValueError("monitor_config.model_family is required")
    if not layers:
        raise ValueError("monitor_config.layers is required")
    if not sae_dir:
        raise ValueError("monitor_config.sae_dir is required")
    if not pca_dir:
        raise ValueError("monitor_config.pca_dir is required")

    dtype = _cfg_get(monitor_config, "dtype", "float16")
    layer_agg = _cfg_get(monitor_config, "layer_agg", "mean")
    threshold = float(_cfg_get(monitor_config, "threshold", 0.5))
    n_bins = int(_cfg_get(monitor_config, "n_bins", 64))

    if isinstance(device, str):
        device = torch.device(device)

    monitor_model = _REGISTRY.get(
        model_family=model_family,
        sae_dir=sae_dir,
        pca_dir=pca_dir,
        layers=[int(x) for x in layers],
        device=device,
        dtype=dtype,
        layer_agg=layer_agg,
        threshold=threshold,
        n_bins=n_bins,
    )
   

    return DecodeMonitorRuntime(monitor_model=monitor_model)