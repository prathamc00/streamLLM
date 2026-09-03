"""
Unit tests for hardware telemetry and PCIe benchmarking.
"""

from pathlib import Path
import sys
import unittest

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
from streamllm.hardware import HardwareProfiler


class TestHardware(unittest.TestCase):
    def test_profiler_output(self):
        profiler = HardwareProfiler(device_idx=0)
        profile = profiler.profile(benchmark_size_mb=32)

        self.assertIsNotNone(profile.device_name)
        self.assertGreaterEqual(profile.total_vram_mb, 0)
        self.assertGreaterEqual(profile.free_vram_mb, 0)
        self.assertGreaterEqual(profile.total_ram_mb, 0)
        if torch.cuda.is_available():
            self.assertGreater(profile.pcie_h2d_bandwidth_gb_s, 0.0)


if __name__ == "__main__":
    unittest.main()
