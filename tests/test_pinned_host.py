"""
Comprehensive unit tests for Pinned Host Weight Registry.
"""

from pathlib import Path
import sys
import unittest

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
from src.pinned_host import PinnedHostWeightRegistry


class TestPinnedHost(unittest.TestCase):
    def test_registry_pinning_and_retrieval(self):
        registry = PinnedHostWeightRegistry()

        weights_l0 = {
            "weight": torch.randn(512, 512, dtype=torch.float32),
            "bias": torch.randn(512, dtype=torch.float32),
        }
        weights_l1 = {
            "weight": torch.randn(512, 512, dtype=torch.float32),
            "bias": torch.randn(512, dtype=torch.float32),
        }
        embed = torch.randn(1000, 512, dtype=torch.float32)

        registry.register_layer(0, weights_l0, pin=True)
        registry.register_layer(1, weights_l1, pin=True)
        registry.register_non_layer_weight("embed_tokens", embed, pin=True)

        self.assertIn(0, registry.layers)
        self.assertIn(1, registry.layers)
        self.assertIn("embed_tokens", registry.non_layer_weights)

        # Check pinned status if CUDA is active
        if torch.cuda.is_available():
            l0 = registry.get_layer(0)
            self.assertTrue(l0["weight"].is_pinned())
            self.assertTrue(l0["bias"].is_pinned())
            self.assertTrue(registry.non_layer_weights["embed_tokens"].is_pinned())

        # Check total RAM calculation
        ram_mb = registry.total_ram_mb()
        self.assertGreater(ram_mb, 0.0)

        # Check key error on missing layer
        with self.assertRaises(KeyError):
            registry.get_layer(999)


if __name__ == "__main__":
    unittest.main()
