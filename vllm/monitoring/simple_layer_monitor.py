import torch


class SimpleLayerMonitor:
    def __init__(self, model_family, adapter_name, layers, device):
        self.model_family = model_family
        self.adapter_name = adapter_name
        self.layers = layers
        self.device = device

    @torch.inference_mode()
    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # plumbing test only; replace with SAE + effective head
        return torch.sigmoid(hidden_states.float().mean(dim=-1))