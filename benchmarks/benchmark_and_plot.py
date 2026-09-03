"""
StreamLLM Benchmark & Plotting Tool.

Executes comparative benchmarks across multiple layer depths,
measures Sequential vs. Double-Buffered Prefetch latency and VRAM footprint,
and automatically exports a high-resolution benchmark chart to assets/benchmark_results.png.
"""

import gc
from pathlib import Path
import sys
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from streamllm.pinned_host import PinnedHostWeightRegistry
    from streamllm.engine import StreamLLMEngine
    from streamllm.hardware import HardwareProfiler
except ImportError:
    from src.pinned_host import PinnedHostWeightRegistry
    from src.engine import StreamLLMEngine
    from src.hardware import HardwareProfiler


class BenchmarkBlock(nn.Module):
    def __init__(self, dim: int, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.norm = nn.LayerNorm(dim, dtype=dtype)
        self.fc1 = nn.Linear(dim, dim * 4, bias=False, dtype=dtype)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim * 4, dim, bias=False, dtype=dtype)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


def run_benchmark_suite(
    layer_counts=(8, 16, 24, 32),
    hidden_dim=2048,
    seq_len=128,
    runs=5,
    dtype=torch.float16,
    device_str="cuda:0",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[*] Running StreamLLM Benchmark on: {device} (Precision: {dtype})")
    
    # Profile hardware
    profiler = HardwareProfiler()
    hw_profile = profiler.profile(benchmark_size_mb=64)
    print(f"[*] Hardware: {hw_profile.device_name} (Total VRAM: {hw_profile.total_vram_mb:.1f} MB, PCIe: {hw_profile.pcie_h2d_bandwidth_gb_s:.2f} GB/s)")

    results = []

    for num_layers in layer_counts:
        template = BenchmarkBlock(hidden_dim, dtype=dtype)
        param_mb_per_layer = sum(p.nelement() * p.element_size() for p in template.parameters()) / (1024 * 1024)
        total_model_mb = param_mb_per_layer * num_layers

        # Register pinned weights in host RAM
        registry = PinnedHostWeightRegistry()
        for i in range(num_layers):
            layer_weights = {
                name: torch.randn(p.shape, dtype=dtype)
                for name, p in template.named_parameters()
            }
            registry.register_layer(i, layer_weights, pin=True)

        input_tensor = torch.randn(1, seq_len, hidden_dim, dtype=dtype, device=device)

        # 1. Sequential Mode Benchmark
        engine_seq = StreamLLMEngine(
            layer_module_template=BenchmarkBlock(hidden_dim, dtype=dtype),
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
        for _ in range(runs):
            _ = engine_seq.forward_streamed(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seq_ms = ((time.perf_counter() - start) / runs) * 1000

        # 2. Double-Buffered Prefetch Mode Benchmark
        engine_prefetch = StreamLLMEngine(
            layer_module_template=BenchmarkBlock(hidden_dim, dtype=dtype),
            host_registry=registry,
            num_layers=num_layers,
            device=device,
            double_buffering=True,
        )
        # Warmup
        _ = engine_prefetch.forward_streamed(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(runs):
            _ = engine_prefetch.forward_streamed(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prefetch_ms = ((time.perf_counter() - start) / runs) * 1000

        speedup = (seq_ms / prefetch_ms) if prefetch_ms > 0 else 1.0
        vram_used_mb = engine_prefetch.scratchpad.total_pool_vram_mb()

        results.append({
            "layers": num_layers,
            "total_model_mb": total_model_mb,
            "seq_ms": seq_ms,
            "prefetch_ms": prefetch_ms,
            "speedup": speedup,
            "vram_used_mb": vram_used_mb,
        })
        print(f"  > Layers: {num_layers:2d} | Model: {total_model_mb:6.1f} MB | Seq: {seq_ms:6.2f} ms | Prefetch: {prefetch_ms:6.2f} ms | Speedup: {speedup:.2f}x | GPU VRAM: {vram_used_mb:.1f} MB")

        # Free memory between benchmark iterations
        del registry
        del engine_seq
        del engine_prefetch
        del input_tensor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return hw_profile, results


def generate_benchmark_plot(hw_profile, results, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    layers = [r["layers"] for r in results]
    seq_latencies = [r["seq_ms"] for r in results]
    prefetch_latencies = [r["prefetch_ms"] for r in results]
    model_sizes = [r["total_model_mb"] for r in results]

    # Style: Modern dark palette
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=200)

    # Plot 1: Latency Comparison
    bar_width = 0.35
    x = list(range(len(layers)))
    r1 = [i - bar_width / 2 for i in x]
    r2 = [i + bar_width / 2 for i in x]

    ax1.bar(r1, seq_latencies, width=bar_width, label="Sequential Streaming", color="#e74c3c", alpha=0.9, edgecolor="#ffffff", linewidth=0.5)
    ax1.bar(r2, prefetch_latencies, width=bar_width, label="Double-Buffered Prefetch", color="#2ecc71", alpha=0.9, edgecolor="#ffffff", linewidth=0.5)

    ax1.set_title("Inference Latency: Sequential vs. Prefetch", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Transformer Layer Count", fontsize=11, labelpad=8)
    ax1.set_ylabel("Latency (ms)", fontsize=11, labelpad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{l} Layers\n({m:.0f} MB)" for l, m in zip(layers, model_sizes)])
    ax1.legend(loc="upper left", frameon=True, framealpha=0.3)
    ax1.grid(True, linestyle="--", alpha=0.25)

    # Annotate speedup
    for i in range(len(layers)):
        speedup = results[i]["speedup"]
        ax1.text(
            r2[i],
            prefetch_latencies[i] + (max(seq_latencies) * 0.02),
            f"{speedup:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#2ecc71",
        )

    # Plot 2: Memory Efficiency
    vram_scratchpad = [r["vram_used_mb"] for r in results]
    ax2.plot(layers, model_sizes, marker="o", linewidth=2.5, color="#f39c12", label="Full Model Weights (RAM)")
    ax2.plot(layers, vram_scratchpad, marker="s", linewidth=2.5, color="#00bcd4", linestyle="--", label="StreamLLM GPU VRAM Pool")

    ax2.set_title("Memory Efficiency: GPU VRAM Cap vs. Model Size", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Transformer Layer Count", fontsize=11, labelpad=8)
    ax2.set_ylabel("Memory Footprint (MB)", fontsize=11, labelpad=8)
    ax2.set_xticks(layers)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.3)
    ax2.grid(True, linestyle="--", alpha=0.25)

    plt.suptitle(
        f"StreamLLM Benchmark Performance ({hw_profile.device_name} | PCIe {hw_profile.pcie_h2d_bandwidth_gb_s:.1f} GB/s)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[+] Benchmark graph successfully saved to: {output_path}")


def print_markdown_table(hw_profile, results):
    print("\n==================== MARKDOWN BENCHMARK TABLE ====================")
    print(f"### Benchmark Hardware: `{hw_profile.device_name}` (VRAM: {hw_profile.total_vram_mb:.0f} MB, PCIe: {hw_profile.pcie_h2d_bandwidth_gb_s:.2f} GB/s)\n")
    print("| Layers | Model Weight Size | Sequential Latency | Prefetch Latency | Speedup | GPU VRAM Scratchpad |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['layers']} Layers | {r['total_model_mb']:.1f} MB | {r['seq_ms']:.2f} ms | **{r['prefetch_ms']:.2f} ms** | **{r['speedup']:.2f}x** | **{r['vram_used_mb']:.1f} MB** |")
    print("===================================================================\n")


def main():
    chart_path = root_dir / "assets" / "benchmark_results.png"
    hw_profile, results = run_benchmark_suite(
        layer_counts=(8, 16, 24, 32),
        hidden_dim=2048,
        seq_len=128,
        runs=5,
        dtype=torch.float16,
    )
    generate_benchmark_plot(hw_profile, results, chart_path)
    print_markdown_table(hw_profile, results)


if __name__ == "__main__":
    main()
