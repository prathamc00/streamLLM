# StreamLLM

[![PyPI version](https://img.shields.io/pypi/v/streamllm.svg?color=blue)](https://pypi.org/project/streamllm/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

> **Run bigger LLMs on smaller GPUs through intelligent, asynchronous layer streaming.**

StreamLLM is a lightweight, memory-aware LLM inference runtime that breaks the physical VRAM barrier. By dynamically streaming transformer layers between host RAM and GPU VRAM using pre-allocated double-buffer scratchpads, StreamLLM allows running large quantized models on consumer GPUs (such as 4GB/6GB/8GB cards) with minimal transfer overhead.

---

## Key Architectural Highlights

- **Static GPU Scratchpad Pool (`GPUScratchpadPool`):** Pre-allocates two static VRAM buffers (`Slot A` and `Slot B`). Eliminates `cudaMalloc` and `cudaFree` allocation churn during the token generation loop.
- **Dual CUDA Streams:** Dedicated `compute_stream` and `transfer_stream` with zero-CPU-blocking synchronization via `torch.cuda.Event` hardware queues.
- **Pinned Host Memory (`PinnedHostWeightRegistry`):** Uses page-locked RAM (`torch.Tensor.pin_memory()`) for true non-blocking PCIe DMA transfers.
- **KV-Cache Sizing Manager:** Deterministic VRAM memory budgeting ensuring safe context lengths without Out-Of-Memory (OOM) crashes.

---

## Installation

Install directly from [PyPI](https://pypi.org/project/streamllm/):

```bash
pip install streamllm
```

Or install from source in development mode:

```bash
git clone https://github.com/prathamc00/llm-Forge.git
cd llm-Forge
pip install -e .
```

---

## Quickstart & Python Usage

```python
from streamllm import AutoModel

MAX_LENGTH = 128

# 1. Initialize AutoModel (supports HuggingFace repo IDs or local paths)
model = AutoModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct", prefetching=True)

# 2. Tokenize input prompt
input_text = ['What is the capital of the United States?']
input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False
)

# 3. Streamed layer generation
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True
)

# 4. Decode output tokens
output = model.tokenizer.decode(generation_output.sequences[0])
print(output)
```

---

## CLI & Diagnostic Commands

Once installed, the `streamllm` command is available directly in your terminal:

### 1. Check Hardware & PCIe Bandwidth
Measure GPU VRAM, system RAM, and live Host-to-Device (H2D) PCIe throughput:
```bash
streamllm hardware
```

### 2. Run Streaming vs. Prefetch Micro-Benchmark
Benchmark sequential layer execution against double-buffered prefetching:
```bash
streamllm bench --layers 16 --hidden-dim 2048 --seq-len 128
```

### 3. Run Test Suite
```bash
python -m unittest discover -s tests
```
