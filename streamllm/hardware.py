"""
Hardware Profiler and PCIe Bandwidth Meter for StreamLLM.
"""

from dataclasses import dataclass
import time
from typing import Optional
import torch


@dataclass
class HardwareProfile:
    device_name: str
    compute_capability: str
    total_vram_mb: float
    free_vram_mb: float
    total_ram_mb: float
    pcie_h2d_bandwidth_gb_s: float
    recommended_vram_budget_mb: float


class HardwareProfiler:
    """Detects GPU / CPU limits and measures PCIe Host-to-Device bandwidth."""

    def __init__(self, device_idx: int = 0):
        self.device_idx = device_idx
        self.is_cuda = torch.cuda.is_available()

    def profile(self, benchmark_size_mb: int = 128) -> HardwareProfile:
        if not self.is_cuda:
            return HardwareProfile(
                device_name="CPU Only",
                compute_capability="N/A",
                total_vram_mb=0.0,
                free_vram_mb=0.0,
                total_ram_mb=self._get_system_ram_mb(),
                pcie_h2d_bandwidth_gb_s=0.0,
                recommended_vram_budget_mb=0.0,
            )

        device = torch.device(f"cuda:{self.device_idx}")
        props = torch.cuda.get_device_properties(device)
        total_vram_mb = props.total_memory / (1024 * 1024)

        # Query free VRAM
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_vram_mb = free_bytes / (1024 * 1024)

        # Measure PCIe H2D Bandwidth
        h2d_gb_s = self.measure_pcie_bandwidth(device, size_mb=benchmark_size_mb)

        # Calculate safety budget (leave at least 450MB for OS display / PyTorch context)
        safety_headroom_mb = 500.0
        recommended_budget_mb = max(0.0, free_vram_mb - safety_headroom_mb)

        return HardwareProfile(
            device_name=props.name,
            compute_capability=f"{props.major}.{props.minor}",
            total_vram_mb=round(total_vram_mb, 2),
            free_vram_mb=round(free_vram_mb, 2),
            total_ram_mb=round(self._get_system_ram_mb(), 2),
            pcie_h2d_bandwidth_gb_s=round(h2d_gb_s, 2),
            recommended_vram_budget_mb=round(recommended_budget_mb, 2),
        )

    def measure_pcie_bandwidth(self, device: torch.device, size_mb: int = 128, iterations: int = 5) -> float:
        """Measures Host-to-Device transfer speed using pinned memory."""
        try:
            num_elements = (size_mb * 1024 * 1024) // 4  # float32 elements
            # Allocate page-locked host tensor
            host_tensor = torch.empty(num_elements, dtype=torch.float32, pin_memory=True)
            gpu_tensor = torch.empty(num_elements, dtype=torch.float32, device=device)

            # Warmup
            gpu_tensor.copy_(host_tensor, non_blocking=True)
            torch.cuda.synchronize(device)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            times_ms = []
            for _ in range(iterations):
                start_event.record()
                gpu_tensor.copy_(host_tensor, non_blocking=True)
                end_event.record()
                end_event.synchronize()
                times_ms.append(start_event.elapsed_time(end_event))

            del host_tensor
            del gpu_tensor
            torch.cuda.empty_cache()

            avg_time_s = (sum(times_ms) / len(times_ms)) / 1000.0
            bytes_transferred = size_mb * 1024 * 1024
            gb_s = (bytes_transferred / (1024**3)) / avg_time_s
            return gb_s
        except Exception:
            return 0.0

    def _get_system_ram_mb(self) -> float:
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 * 1024)
        except ImportError:
            return 0.0
