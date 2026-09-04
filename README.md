# StreamLLM

[![PyPI version](https://img.shields.io/pypi/v/streamllm.svg?color=blue)](https://pypi.org/project/streamllm/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

**StreamLLM** optimizes inference memory usage, allowing large language models (such as 14B, 32B, and 70B models) to run on consumer GPUs with as little as **4GB or 8GB VRAM** without requiring distributed hardware.

Instead of loading the entire model into GPU memory at once, StreamLLM dynamically streams transformer layers between system RAM/disk and GPU VRAM, using asynchronous prefetching and double buffering so computation overlaps with data movement.

---

## Quickstart

### 1. Install package

```bash
pip install streamllm
```

### 2. Discover and install a model

```bash
# View available curated models and installation status
streamllm list

# Download and install a model to your local cache
streamllm install smollm-135m
# Or install mid-sized/larger models:
streamllm install qwen-0.5b
streamllm install qwen-14b
```

### 3. Inference via Python SDK

Run inference with automatic layer streaming:

```python
from streamllm import AutoModel

# Load any installed preset or HuggingFace repo ID
model = AutoModel.from_pretrained("smollm-135m")

tokens = model.tokenizer("Hello, what is layer streaming?", return_tensors="pt")
output = model.generate(tokens["input_ids"], max_new_tokens=40)

print(model.tokenizer.decode(output.sequences[0]))
```

You can pass in any preset alias, HuggingFace model repo ID, or local directory path.

---

## Model Management CLI

StreamLLM provides full model management (like Ollama or Docker) for discovering and downloading open-source weights:

```bash
# List all available presets, parameters, memory specs, and install status
streamllm list

# List only locally installed models
streamllm list --installed

# Download & install a model (preset alias or HuggingFace repo ID)
streamllm install qwen-0.5b
streamllm install HuggingFaceTB/SmolLM2-135M-Instruct
streamllm pull llama3.2-1b

# Remove an installed model from local cache
streamllm models remove qwen-0.5b
```

### Curated Model Presets

| Preset | Model Repo ID | Parameters | Est. Download | StreamLLM Min VRAM | Standard VRAM |
|---|---|---|---|---|---|
| `smollm-135m` | `HuggingFaceTB/SmolLM2-135M-Instruct` | 135M | ~270 MB | **~120 MB** | ~600 MB |
| `smollm-360m` | `HuggingFaceTB/SmolLM2-360M-Instruct` | 360M | ~720 MB | **~220 MB** | ~1.4 GB |
| `smollm-1.7b` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | ~3.4 GB | **~420 MB** | ~4.5 GB |
| `qwen-0.5b` | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | ~980 MB | **~260 MB** | ~2.0 GB |
| `qwen-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ~3.1 GB | **~400 MB** | ~4.8 GB |
| `llama3.2-1b` | `meta-llama/Llama-3.2-1B-Instruct` | 1.2B | ~2.4 GB | **~380 MB** | ~3.6 GB |
| `llama3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` | 3.2B | ~6.4 GB | **~580 MB** | ~8.0 GB |
| `qwen-7b` | `Qwen/Qwen2.5-7B-Instruct` | 7B | ~14.5 GB | **~850 MB** | ~16.0 GB |
| `qwen-14b` | `Qwen/Qwen2.5-14B-Instruct-AWQ` | 14B (AWQ) | ~8.5 GB | **~1200 MB** | ~10.0 GB |
| `llama3-8b` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 8B | ~16.0 GB | **~950 MB** | ~18.0 GB |
| `mistral-7b` | `mistralai/Mistral-7B-Instruct-v0.3` | 7.3B | ~14.5 GB | **~900 MB** | ~16.5 GB |
| `deepseek-1.5b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B | ~3.1 GB | **~420 MB** | ~4.8 GB |
| `deepseek-7b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 7B | ~14.5 GB | **~880 MB** | ~16.5 GB |

---

## Interactive Chat & One-Shot Run

StreamLLM includes an interactive terminal chat REPL and command-line runner with real-time token streaming:

```bash
# 1. Interactive Chat REPL
streamllm chat smollm-135m
streamllm chat qwen-0.5b
streamllm chat qwen-14b

# If you don't provide a model name, StreamLLM automatically selects your installed model!
streamllm chat

# Inside chat, you have built-in slash commands:
# /stats   - Check live GPU VRAM & scratchpad allocation
# /system  - Update the system instruction prompt
# /clear   - Clear conversation history
# /exit    - Quit chat

# 2. One-shot command execution
streamllm run smollm-135m "Explain layer streaming in 2 sentences"
streamllm run qwen-0.5b "Summarize quantum computing" --max-tokens 100

# 3. Pipeable raw output for scripts
streamllm run smollm-135m "Generate JSON" --raw

# 4. Check GPU VRAM and live PCIe bandwidth
streamllm hardware

# 5. Run streaming micro-benchmark
streamllm bench --layers 16 --hidden-dim 2048 --seq-len 128
```

---

## How It Works

During inference, StreamLLM keeps only the currently executing layer on the GPU while asynchronously prefetching upcoming layers over PCIe DMA:

- **Static GPU Scratchpad Pool (`GPUScratchpadPool`):** Pre-allocates two static VRAM buffer slots (Slot A and Slot B). Eliminates `cudaMalloc` and `cudaFree` churn during generation.
- **Dual CUDA Streams:** Dedicated `compute_stream` and `transfer_stream` with lock-free hardware event synchronization (`torch.cuda.Event`).
- **Pinned Host Memory:** Uses page-locked RAM (`torch.Tensor.pin_memory()`) for true non-blocking PCIe transfers.
- **KV-Cache Manager:** Deterministic memory budgeting that ensures context length growth never causes an Out-of-Memory (OOM) crash.

---

## Benchmark Performance

![StreamLLM Benchmark Performance](assets/benchmark_results.png)

### Measured Hardware: `NVIDIA GeForce RTX 3050 Laptop GPU` (VRAM: 4096 MB, PCIe: 8.65 GB/s)

| Layers | Model Weight Size | Sequential Latency | Prefetch Latency | Speedup | GPU VRAM Scratchpad |
|---|---|---|---|---|---|
| **8 Layers** | 512.1 MB | 68.08 ms | **56.70 ms** | **1.20x** | **128.0 MB** |
| **16 Layers** | 1024.1 MB | 136.51 ms | **126.11 ms** | **1.08x** | **128.0 MB** |
| **24 Layers** | 1536.2 MB | 219.05 ms | **170.54 ms** | **1.28x** | **128.0 MB** |
| **32 Layers** | 2048.2 MB | 278.85 ms | **265.03 ms** | **1.05x** | **128.0 MB** |

> **Key Takeaway:** Even as total model weights scale past 2.0 GB, StreamLLM's static scratchpad pool holds physical GPU memory strictly at **128.0 MB** while asynchronous prefetching delivers up to **1.28x faster inference**.
