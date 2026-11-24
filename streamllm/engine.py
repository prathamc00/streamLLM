"""
StreamLLM Layer Streaming Engine and Inference Controller.
"""

from copy import deepcopy
from typing import Callable, Dict, Optional
import torch
import torch.nn as nn

try:
    from .scratchpad import GPUScratchpadPool
    from .pinned_host import PinnedHostWeightRegistry
    from .transfer import DualStreamTransferEngine
except (ImportError, ValueError):
    from scratchpad import GPUScratchpadPool
    from pinned_host import PinnedHostWeightRegistry
    from transfer import DualStreamTransferEngine


class StreamLLMEngine:
    """Executes transformer layers sequentially or via asynchronous double-buffering."""

    def __init__(
        self,
        layer_module_template: nn.Module,
        host_registry: PinnedHostWeightRegistry,
        num_layers: int,
        device: torch.device,
        double_buffering: bool = True,
    ):
        self.resident_layers = []
        self.device = device
        self.num_layers = num_layers
        self.host_registry = host_registry
        self.double_buffering = double_buffering and (device.type == "cuda") and torch.cuda.is_available()

        # Extract layer parameter metadata (shapes and dtypes)
        state_dict_meta = {
            name: (param.shape, param.dtype)
            for name, param in layer_module_template.named_parameters()
        }

        # Initialize static GPU scratchpad pool (Slot A & Slot B)
        num_slots = 2 if self.double_buffering else 1
        self.scratchpad = GPUScratchpadPool(
            layer_template_meta=state_dict_meta,
            device=device,
            num_slots=num_slots,
        )

        # Place the layer module template on the GPU
        self.layer_module = layer_module_template.to(device)

        # The template doubles as slot zero; release its original parameter storage.
        for name, param in self.layer_module.named_parameters():
            param.data = self.scratchpad.slots[0].tensors[name]
        self.slot_modules = [self.layer_module]
        for slot in self.scratchpad.slots[1:]:
            self.slot_modules.append(self._module_for_weights(slot.tensors))
        self.transfer_engine = DualStreamTransferEngine(device=device)

    def _module_for_weights(self, weights: Dict[str, torch.Tensor]) -> nn.Module:
        # Substitute parameters during cloning so no duplicate weight storage is allocated.
        memo = {id(param): nn.Parameter(weights[name], requires_grad=False)
                for name, param in self.layer_module.named_parameters()}
        return deepcopy(self.layer_module, memo)

    def prepare_residency(self, cache_bytes):
        if not self.double_buffering:
            return
        free, total = torch.cuda.mem_get_info(self.device)
        layer_bytes = self.scratchpad.active_slot.total_bytes()
        # The driver excludes the allocator's cached blocks from free memory.
        # Credit those blocks when budgeting the next request's KV cache.
        reserved = torch.cuda.memory_reserved(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        reusable = max(0, reserved - allocated)
        available = free + reusable + len(self.resident_layers) * layer_bytes
        reserve = cache_bytes + max(total // 4, 512 * 1024**2)
        target = min(self.num_layers, max(0, available - reserve) // layer_bytes)
        del self.resident_layers[target:]
        while len(self.resident_layers) < target:
            host = self.host_registry.get_layer(len(self.resident_layers))
            try:
                weights = {name: host[name].to(self.device) for name in self.scratchpad.active_slot.tensors}
                module = self._module_for_weights(weights)
            except torch.cuda.OutOfMemoryError:
                break
            self.resident_layers.append(module)

    def forward_streamed(
        self,
        hidden_states: torch.Tensor,
        layer_forward_fn: Optional[Callable[[nn.Module, torch.Tensor, int], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Executes a full forward pass through all N layers with pipelined prefetching."""
        if not self.double_buffering:
            return self._forward_sequential(hidden_states, layer_forward_fn)

        # -------------------------------------------------------------
        # Double-Buffered Pipelined Forward Pass
        # -------------------------------------------------------------
        start = len(self.resident_layers)
        if start == self.num_layers:
            x = hidden_states
            for index, module in enumerate(self.resident_layers):
                x = layer_forward_fn(module, x, index) if layer_forward_fn else module(x)
            return x

        stream = self.transfer_engine.compute_stream
        caller = torch.cuda.current_stream(self.device)
        stream.wait_stream(caller)
        self.transfer_engine.transfer_stream.wait_stream(caller)
        # Fetch the first nonresident layer while the resident prefix runs.
        self.transfer_engine.async_prefetch(self.scratchpad.active_slot, self.host_registry.get_layer(start))

        x = hidden_states
        with torch.cuda.stream(stream):
            for index, module in enumerate(self.resident_layers):
                x = layer_forward_fn(module, x, index) if layer_forward_fn else module(x)

        self.transfer_engine.sync_compute_with_transfer()

        for layer_idx in range(start, self.num_layers):
            # Asynchronously prefetch Layer (idx + 1) into prefetch_slot
            if layer_idx + 1 < self.num_layers:
                next_weights = self.host_registry.get_layer(layer_idx + 1)
                self.transfer_engine.async_prefetch(
                    target_slot=self.scratchpad.prefetch_slot,
                    host_weights=next_weights,
                )

            with torch.cuda.stream(stream):
                module = self.slot_modules[self.scratchpad.active_slot_idx]
                x = layer_forward_fn(module, x, layer_idx) if layer_forward_fn else module(x)
                self.transfer_engine.record_compute_finished()

            # Synchronize: Ensure next layer DMA is done before next iteration computes on it
            if layer_idx + 1 < self.num_layers:
                self.transfer_engine.sync_compute_with_transfer()
                self.scratchpad.swap_slots()

        self.transfer_engine.synchronize()
        return x

    def _forward_sequential(
        self,
        hidden_states: torch.Tensor,
        layer_forward_fn: Optional[Callable[[nn.Module, torch.Tensor, int], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Naive sequential execution: Load -> Compute -> Next."""
        x = hidden_states
        slot = self.scratchpad.active_slot

        for layer_idx in range(self.num_layers):
            weights = self.host_registry.get_layer(layer_idx)
            stream = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
            slot.copy_from_host(weights, stream=stream)

            if layer_forward_fn:
                x = layer_forward_fn(self.layer_module, x, layer_idx)
            else:
                x = self.layer_module(x)

        return x
