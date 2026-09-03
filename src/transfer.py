"""
Dual CUDA Stream Transfer Engine with Lock-Free Hardware Event Synchronization.
"""

from typing import Dict
import torch
try:
    from .scratchpad import LayerBufferSlot
except (ImportError, ValueError):
    from scratchpad import LayerBufferSlot


class DualStreamTransferEngine:
    """Orchestrates asynchronous PCIe transfers in parallel with GPU computation."""

    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = (device.type == "cuda")

        if self.is_cuda:
            self.compute_stream = torch.cuda.Stream(device=device)
            self.transfer_stream = torch.cuda.Stream(device=device)
            self.transfer_done_event = torch.cuda.Event()
            self.compute_done_event = torch.cuda.Event()
        else:
            self.compute_stream = None
            self.transfer_stream = None
            self.transfer_done_event = None
            self.compute_done_event = None

    def async_prefetch(self, target_slot: LayerBufferSlot, host_weights: Dict[str, torch.Tensor]) -> None:
        """Prefetches weights over PCIe on the background transfer stream."""
        if not self.is_cuda:
            # CPU fallback: direct copy
            for k, v in host_weights.items():
                if k in target_slot.tensors:
                    target_slot.tensors[k].copy_(v)
            return

        # Before writing to target_slot, wait until any ongoing compute using this slot is finished
        self.transfer_stream.wait_event(self.compute_done_event)

        # Launch DMA transfer on transfer stream
        target_slot.copy_from_host(host_weights, stream=self.transfer_stream)

        # Record completion event on transfer stream
        self.transfer_done_event.record(self.transfer_stream)

    def sync_compute_with_transfer(self) -> None:
        """Makes the compute stream wait until the background transfer finishes."""
        if self.is_cuda:
            self.compute_stream.wait_event(self.transfer_done_event)

    def record_compute_finished(self) -> None:
        """Records on the compute stream that execution on the active slot has finished."""
        if self.is_cuda:
            self.compute_done_event.record(self.compute_stream)

    def synchronize(self) -> None:
        """Blocks the host CPU until both compute and transfer streams are completely idle."""
        if self.is_cuda:
            self.compute_stream.synchronize()
            self.transfer_stream.synchronize()
