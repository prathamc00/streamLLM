"""
Unit tests for GPU double-buffer scratchpad pool and allocation invariance.
"""

from pathlib import Path
import sys
import unittest

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
import torch.nn as nn
from src.scratchpad import GPUScratchpadPool


class TestScratchpad(unittest.TestCase):
    def test_allocation_invariance(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        meta = {
            "weight": (torch.Size([1024, 1024]), torch.float32),
            "bias": (torch.Size([1024]), torch.float32),
        }

        pool = GPUScratchpadPool(layer_template_meta=meta, device=device, num_slots=2)
        self.assertEqual(len(pool.slots), 2)
        self.assertGreater(pool.total_pool_vram_mb(), 0)

        # In-place copy check
        host_weights = {
            "weight": torch.randn(1024, 1024, dtype=torch.float32),
            "bias": torch.randn(1024, dtype=torch.float32),
        }

        if device.type == "cuda":
            initial_mem = torch.cuda.memory_allocated(device)
            stream = torch.cuda.current_stream()
            pool.active_slot.copy_from_host(host_weights, stream=stream)
            torch.cuda.synchronize(device)
            after_mem = torch.cuda.memory_allocated(device)

            # Strict requirement: Zero new memory allocated during weight copy!
            self.assertEqual(initial_mem, after_mem)

        # Check slot swapping
        first_slot_id = pool.active_slot.slot_id
        pool.swap_slots()
        second_slot_id = pool.active_slot.slot_id
        self.assertNotEqual(first_slot_id, second_slot_id)


if __name__ == "__main__":
    unittest.main()
