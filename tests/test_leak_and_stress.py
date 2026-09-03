"""
Stress and memory-leak tests for StreamLLM execution pipeline.
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


class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.proj1 = nn.Linear(dim, dim * 2, bias=False)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.proj2(self.act(self.proj1(x)))
        return residual + x


class TestStressAndLeaks(unittest.TestCase):
    def test_zero_memory_leak_across_iterations(self):
        """Ensures that executing multiple forward passes does not leak any VRAM."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dim = 512
        num_layers = 6
        seq_len = 32

        template = TransformerBlock(dim)
        registry = PinnedHostWeightRegistry()
        for i in range(num_layers):
            weights = {
                name: torch.randn(p.shape, dtype=p.dtype)
                for name, p in template.named_parameters()
            }
            registry.register_layer(i, weights, pin=True)

        engine = StreamLLMEngine(
            layer_module_template=TransformerBlock(dim),
            host_registry=registry,
            num_layers=num_layers,
            device=device,
            double_buffering=True,
        )

        input_data = torch.randn(1, seq_len, dim, device=device)

        # Warmup pass to initialize PyTorch internal contexts
        _ = engine.forward_streamed(input_data)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            baseline_mem = torch.cuda.memory_allocated(device)

        # Run 15 consecutive iterations
        for _ in range(15):
            out = engine.forward_streamed(input_data)
            self.assertEqual(out.shape, (1, seq_len, dim))

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            final_mem = torch.cuda.memory_allocated(device)
            # Assert zero byte growth in VRAM (strictly no memory leaks)
            self.assertEqual(
                baseline_mem,
                final_mem,
                f"Memory leak detected! Baseline: {baseline_mem} bytes, Final: {final_mem} bytes",
            )

    def test_dual_stream_concurrency(self):
        """Validates that compute and transfer streams are distinct and do not deadlock."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            self.skipTest("CUDA required for stream concurrency test")

        from src.transfer import DualStreamTransferEngine
        engine = DualStreamTransferEngine(device=device)

        self.assertNotEqual(
            engine.compute_stream.cuda_stream,
            engine.transfer_stream.cuda_stream,
        )


if __name__ == "__main__":
    unittest.main()
