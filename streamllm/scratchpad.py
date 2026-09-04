"""
Zero-Allocation Static GPU Double-Buffer Scratchpad Pool.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class LayerBufferSlot:
    """Represents a pre-allocated GPU VRAM slot containing tensor buffers for one layer."""

    def __init__(self, slot_id: int, state_dict_meta: Dict[str, Tuple[torch.Size, torch.dtype]], device: torch.device):
        self.slot_id = slot_id
        self.device = device
        self.tensors: Dict[str, torch.Tensor] = {}
        self.current_layer_id: int = -1

        # Pre-allocate static GPU memory buffers matching target layer parameter shapes
        for param_name, (shape, dtype) in state_dict_meta.items():
            self.tensors[param_name] = torch.empty(shape, dtype=dtype, device=device)

    def total_bytes(self) -> int:
        return sum(t.nelement() * t.element_size() for t in self.tensors.values())

    def copy_from_host(self, host_weights: Dict[str, torch.Tensor], stream: Optional[torch.cuda.Stream] = None) -> None:
        """Asynchronously copies host pinned weights into pre-allocated GPU buffers."""
        if stream is not None and torch.cuda.is_available():
            with torch.cuda.stream(stream):
                for param_name, host_tensor in host_weights.items():
                    if param_name in self.tensors:
                        self.tensors[param_name].copy_(host_tensor, non_blocking=True)
        else:
            for param_name, host_tensor in host_weights.items():
                if param_name in self.tensors:
                    self.tensors[param_name].copy_(host_tensor)


class GPUScratchpadPool:
    """Manages static double buffering (Slot A and Slot B) in GPU VRAM."""

    def __init__(
        self,
        layer_template_meta: Dict[str, Tuple[torch.Size, torch.dtype]],
        device: torch.device,
        num_slots: int = 2,
    ):
        self.device = device
        self.num_slots = num_slots
        self.slots: List[LayerBufferSlot] = [
            LayerBufferSlot(slot_id=i, state_dict_meta=layer_template_meta, device=device)
            for i in range(num_slots)
        ]
        self.active_slot_idx: int = 0
        self.prefetch_slot_idx: int = 1 if num_slots > 1 else 0

    @property
    def active_slot(self) -> LayerBufferSlot:
        return self.slots[self.active_slot_idx]

    @property
    def prefetch_slot(self) -> LayerBufferSlot:
        return self.slots[self.prefetch_slot_idx]

    def swap_slots(self) -> None:
        """Rotates the active and prefetch slot indices."""
        self.active_slot_idx = (self.active_slot_idx + 1) % self.num_slots
        self.prefetch_slot_idx = (self.prefetch_slot_idx + 1) % self.num_slots

    def total_pool_vram_mb(self) -> float:
        total_bytes = sum(slot.total_bytes() for slot in self.slots)
        return total_bytes / (1024 * 1024)
