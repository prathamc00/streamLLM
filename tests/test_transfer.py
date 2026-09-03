"""
Unit tests for layer streaming engine and execution correctness.
"""

from pathlib import Path
import sys
import unittest

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
import torch.nn as nn
from src.pinned_host import PinnedHostWeightRegistry
from src.engine import StreamLLMEngine


class SimpleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear(x)


class TestTransferEngine(unittest.TestCase):
    def test_forward_streamed_equivalence(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dim = 256
        num_layers = 4
        seq_len = 16

        registry = PinnedHostWeightRegistry()
        for i in range(num_layers):
            weights = {"linear.weight": torch.randn(dim, dim, dtype=torch.float32)}
            registry.register_layer(i, weights, pin=True)

        engine_seq = StreamLLMEngine(
            layer_module_template=SimpleBlock(dim),
            host_registry=registry,
            num_layers=num_layers,
            device=device,
            double_buffering=False,
        )

        engine_double = StreamLLMEngine(
            layer_module_template=SimpleBlock(dim),
            host_registry=registry,
            num_layers=num_layers,
            device=device,
            double_buffering=True,
        )

        input_data = torch.randn(1, seq_len, dim, device=device)

        out_seq = engine_seq.forward_streamed(input_data)
        out_double = engine_double.forward_streamed(input_data)

        # Both sequential and double-buffered modes should produce mathematically identical outputs
        self.assertTrue(torch.allclose(out_seq, out_double, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
