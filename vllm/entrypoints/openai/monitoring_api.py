from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Any


@dataclass
class MonitorConfig:
    enabled: bool = False
    model_family: Optional[str] = None
    adapter_name: Optional[str] = None
    layers: Optional[List[int]] = None
    return_token_scores: bool = True
    reasoning_span_only: bool = False


def parse_monitor_config(monitor_obj: Any) -> MonitorConfig | None:
    if monitor_obj is None:
        return None
    if not isinstance(monitor_obj, dict):
        return None
    if not monitor_obj.get("enabled", False):
        return None

    return MonitorConfig(
        enabled=True,
        model_family=monitor_obj.get("model_family"),
        adapter_name=monitor_obj.get("adapter_name"),
        layers=monitor_obj.get("layers"),
        return_token_scores=bool(monitor_obj.get("return_token_scores", True)),
        reasoning_span_only=bool(monitor_obj.get("reasoning_span_only", False)),
    )
