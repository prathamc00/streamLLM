"""
StreamLLM Command-Line Interface.
"""

import argparse
from pathlib import Path
import sys
import time
import torch
import torch.nn as nn

# Ensure both root directory and src directory are in sys.path
_current_dir = Path(__file__).resolve().parent
_root_dir = _current_dir.parent
for p in [str(_root_dir), str(_current_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from .hardware import HardwareProfiler
    from .pinned_host import PinnedHostWeightRegistry
    from .engine import StreamLLMEngine
except (ImportError, ValueError):
    from streamllm.hardware import HardwareProfiler
    from streamllm.pinned_host import PinnedHostWeightRegistry
    from streamllm.engine import StreamLLMEngine


def cmd_hardware(args):
    print("==================================================")
    print(" StreamLLM Hardware & PCIe Diagnostic Profile")
    print("==================================================")
    profiler = HardwareProfiler(device_idx=args.device)
    profile = profiler.profile(benchmark_size_mb=args.bench_mb)

    print(f"Device Name:               {profile.device_name}")
    print(f"Compute Capability:        {profile.compute_capability}")
    print(f"Total GPU VRAM:            {profile.total_vram_mb:.1f} MB")
    print(f"Available Free VRAM:       {profile.free_vram_mb:.1f} MB")
    print(f"Total System RAM:          {profile.total_ram_mb:.1f} MB")
    print(f"PCIe H2D Bandwidth:        {profile.pcie_h2d_bandwidth_gb_s:.2f} GB/s")
    print(f"Recommended VRAM Budget:   {profile.recommended_vram_budget_mb:.1f} MB")
    print("==================================================")


def cmd_bench(args):
    print("==================================================")
    print(" StreamLLM Micro-Benchmark: Streaming vs Overlap")
    print("==================================================")
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    num_layers = args.layers
    hidden_dim = args.hidden_dim
    seq_len = args.seq_len

    print(f"Benchmarking with {num_layers} layers, hidden_dim={hidden_dim}, seq_len={seq_len}")
    print(f"Execution Target: {device}")

    # Build synthetic transformer feed-forward block
    class ToyBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim * 4, bias=False)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(dim * 4, dim, bias=False)

        def forward(self, x):
            return x + self.fc2(self.act(self.fc1(x)))

    template = ToyBlock(hidden_dim)
    param_mb = sum(p.nelement() * p.element_size() for p in template.parameters()) / (1024 * 1024)
    print(f"Per-Layer Parameter Footprint: {param_mb:.2f} MB")
    print(f"Total Model Weight Footprint:  {param_mb * num_layers:.2f} MB")

    # Generate pinned host weights
    registry = PinnedHostWeightRegistry()
    for i in range(num_layers):
        layer_weights = {
            name: torch.randn(param.shape, dtype=param.dtype)
            for name, param in template.named_parameters()
        }
        registry.register_layer(i, layer_weights, pin=True)

    input_tensor = torch.randn(1, seq_len, hidden_dim, device=device)

    # 1. Sequential Mode Benchmark
    engine_seq = StreamLLMEngine(
        layer_module_template=ToyBlock(hidden_dim),
        host_registry=registry,
        num_layers=num_layers,
        device=device,
        double_buffering=False,
    )
    # Warmup
    _ = engine_seq.forward_streamed(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(args.runs):
        _ = engine_seq.forward_streamed(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seq_time = (time.perf_counter() - start) / args.runs

    # 2. Asynchronous Double-Buffered Mode Benchmark
    engine_double = StreamLLMEngine(
        layer_module_template=ToyBlock(hidden_dim),
        host_registry=registry,
        num_layers=num_layers,
        device=device,
        double_buffering=True,
    )
    # Warmup
    _ = engine_double.forward_streamed(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(args.runs):
        _ = engine_double.forward_streamed(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    double_time = (time.perf_counter() - start) / args.runs

    speedup = (seq_time / double_time) if double_time > 0 else 1.0

    print("--------------------------------------------------")
    print(f"Sequential Execution:      {seq_time * 1000:.2f} ms")
    print(f"Double-Buffered Prefetch:  {double_time * 1000:.2f} ms")
    print(f"Prefetch Latency Speedup:  {speedup:.2f}x")
    print("--------------------------------------------------")


def main():
    parser = argparse.ArgumentParser(description="StreamLLM CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # hardware
    p_hw = subparsers.add_parser("hardware", help="Display GPU & PCIe telemetry")
    p_hw.add_argument("--device", type=int, default=0, help="CUDA device index")
    p_hw.add_argument("--bench-mb", type=int, default=128, help="PCIe benchmark buffer size in MB")
    p_hw.set_defaults(func=cmd_hardware)

    # bench
    p_bench = subparsers.add_parser("bench", help="Run streaming micro-benchmark")
    p_bench.add_argument("--device", type=int, default=0, help="CUDA device index")
    p_bench.add_argument("--layers", type=int, default=16, help="Number of layers to stream")
    p_bench.add_argument("--hidden-dim", type=int, default=2048, help="Hidden dimension size")
    p_bench.add_argument("--seq-len", type=int, default=128, help="Sequence length (prefill tokens)")
    p_bench.add_argument("--runs", type=int, default=3, help="Benchmark repeat runs")
    p_bench.set_defaults(func=cmd_bench)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
