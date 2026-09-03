"""
Comprehensive unit tests for KV-Cache budget tracking and layer cache updates.
"""

from pathlib import Path
import sys
import unittest

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
from src.kv_cache import KVCacheConfig, KVCacheBudgetTracker, LayerKVCache


class TestKVCache(unittest.TestCase):
    def test_budget_calculation(self):
        config = KVCacheConfig(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            max_context_length=2048,
            dtype=torch.float16,
        )
        tracker = KVCacheBudgetTracker(config)

        # Formula: 2 * 32 * 8 * 128 * 2048 * 2 bytes = 268,435,456 bytes = 256.0 MB
        mem_mb = tracker.max_context_memory_mb()
        self.assertEqual(mem_mb, 256.0)

        # Partial sequence length: 512 tokens -> 64.0 MB
        mem_512 = tracker.calculate_memory_mb(512)
        self.assertEqual(mem_512, 64.0)

    def test_layer_kv_cache_update_and_reset(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        kv_cache = LayerKVCache(device=device)

        self.assertIsNone(kv_cache.key)
        self.assertIsNone(kv_cache.value)

        # Step 1: Add first token
        k1 = torch.randn(1, 8, 1, 128, device=device)
        v1 = torch.randn(1, 8, 1, 128, device=device)
        k_out, v_out = kv_cache.update(k1, v1)
        self.assertEqual(k_out.shape, (1, 8, 1, 128))
        self.assertEqual(v_out.shape, (1, 8, 1, 128))

        # Step 2: Append second token
        k2 = torch.randn(1, 8, 1, 128, device=device)
        v2 = torch.randn(1, 8, 1, 128, device=device)
        k_out, v_out = kv_cache.update(k2, v2)
        self.assertEqual(k_out.shape, (1, 8, 2, 128))
        self.assertEqual(v_out.shape, (1, 8, 2, 128))

        # Reset
        kv_cache.reset()
        self.assertIsNone(kv_cache.key)
        self.assertIsNone(kv_cache.value)


if __name__ == "__main__":
    unittest.main()
