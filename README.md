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

### 2. Inference

Run inference just like a regular transformer model:

```python
from streamllm import AutoModel

model = AutoModel.from_pretrained("Qwen/Qwen2.5-14B-Instruct-AWQ")

tokens = model.tokenizer("Hello, how are you?", return_tensors="pt")
output = model.generate(tokens["input_ids"].cuda(), max_new_tokens=30)

print(model.tokenizer.decode(output.sequences[0]))
```

You can pass in any HuggingFace model repo ID or local path.

---

## How It Works

During inference, StreamLLM keeps only the currently executing layer on the GPU while asynchronously prefetching upcoming layers over PCIe DMA:

- **Static GPU Scratchpad Pool (`GPUScratchpadPool`):** Pre-allocates two static VRAM buffer slots (Slot A and Slot B). Eliminates `cudaMalloc` and `cudaFree` churn during generation.
- **Dual CUDA Streams:** Dedicated `compute_stream` and `transfer_stream` with lock-free hardware event synchronization (`torch.cuda.Event`).
- **Pinned Host Memory:** Uses page-locked RAM (`torch.Tensor.pin_memory()`) for true non-blocking PCIe transfers.
- **KV-Cache Manager:** Deterministic memory budgeting that ensures context length growth never causes an Out-of-Memory (OOM) crash.

---

## Supported Models

StreamLLM works out of the box with popular open model architectures:

- **Qwen** (Qwen 2 / 2.5 / 3 — dense & AWQ/GPTQ quantized)
- **Llama** (Llama 2 / Llama 3 / 3.1 / 3.2 — 8B, 14B, 70B)
- **Mistral & Mixtral**
- **DeepSeek**
- **Phi**
- **Gemma**

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

---

## CLI & Interactive Commands

StreamLLM includes an interactive terminal chat REPL and command-line runner with real-time token streaming:

```bash
# 1. Interactive Chat REPL (Ollama-style terminal experience)
streamllm chat demo
streamllm chat qwen-14b
streamllm chat llama3-8b

# Inside chat, you have built-in slash commands:
# /stats   - Check live GPU VRAM & scratchpad allocation
# /system  - Update the system instruction prompt
# /clear   - Clear conversation history
# /exit    - Quit chat

# 2. One-shot command execution
streamllm run demo "Explain layer streaming in 2 sentences"
streamllm run qwen-14b "Summarize quantum computing" --max-tokens 100

# 3. Pipeable raw output for scripts
streamllm run demo "Generate JSON" --raw

# 4. Check GPU VRAM and live PCIe bandwidth
streamllm hardware

# 5. Run streaming micro-benchmark
streamllm bench --layers 16 --hidden-dim 2048 --seq-len 128
```
