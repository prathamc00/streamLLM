"""
Page-Locked (Pinned) Host Memory Registry for High-Speed PCIe DMA.
"""

from typing import Dict, List
import torch


class PinnedHostWeightRegistry:
    """Stores model layer weights in page-locked (pinned) CPU memory.
    
    Pinned memory is strictly required for asynchronous non_blocking=True PCIe DMA.
    """

    def __init__(self):
        self.layers: Dict[int, Dict[str, torch.Tensor]] = {}
        self.non_layer_weights: Dict[str, torch.Tensor] = {}

    def register_layer(self, layer_id: int, weights: Dict[str, torch.Tensor], pin: bool = True) -> None:
        pinned_dict = {}
        for name, tensor in weights.items():
            cpu_tensor = tensor.cpu()
            if pin and torch.cuda.is_available():
                try:
                    pinned_dict[name] = cpu_tensor.pin_memory()
                except RuntimeError:
                    # Graceful fallback if system page-locked quota is reached
                    pinned_dict[name] = cpu_tensor
            else:
                pinned_dict[name] = cpu_tensor
        self.layers[layer_id] = pinned_dict

    def register_non_layer_weight(self, name: str, tensor: torch.Tensor, pin: bool = True) -> None:
        cpu_tensor = tensor.cpu()
        if pin and torch.cuda.is_available():
            try:
                self.non_layer_weights[name] = cpu_tensor.pin_memory()
            except RuntimeError:
                self.non_layer_weights[name] = cpu_tensor
        else:
            self.non_layer_weights[name] = cpu_tensor

    def get_layer(self, layer_id: int) -> Dict[str, torch.Tensor]:
        if layer_id not in self.layers:
            raise KeyError(f"Layer {layer_id} not found in host weight registry.")
        return self.layers[layer_id]

    def total_ram_mb(self) -> float:
        total_bytes = 0
        for layer in self.layers.values():
            total_bytes += sum(t.nelement() * t.element_size() for t in layer.values())
        for tensor in self.non_layer_weights.values():
            total_bytes += tensor.nelement() * tensor.element_size()
        return total_bytes / (1024 * 1024)
