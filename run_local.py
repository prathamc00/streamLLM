"""
Run StreamLLM locally on your machine (NVIDIA GeForce RTX 3050 Laptop GPU).
"""

import time
import torch
from streamllm import AutoModel
from streamllm.hardware import HardwareProfiler

def main():
    print("=" * 60)
    print(" StreamLLM Local Hardware Diagnostic")
    print("=" * 60)
    profiler = HardwareProfiler()
    profile = profiler.profile(benchmark_size_mb=64)
    print(f"GPU Model:               {profile.device_name}")
    print(f"Total GPU VRAM:          {profile.total_vram_mb:.1f} MB (4 GB)")
    print(f"Available Free VRAM:     {profile.free_vram_mb:.1f} MB")
    print(f"Host-to-Device PCIe:     {profile.pcie_h2d_bandwidth_gb_s:.2f} GB/s")
    print(f"Safe VRAM Budget:        {profile.recommended_vram_budget_mb:.1f} MB")
    print("=" * 60)

    # Choose model:
    # Option 1: Fast local test (Qwen 0.5B - downloads in seconds)
    # Option 2: 14B model: "Qwen/Qwen2.5-14B-Instruct"
    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"\n[*] Initializing {MODEL_ID} with StreamLLM layer streaming...")
    start_time = time.time()

    model = AutoModel.from_pretrained(
        MODEL_ID,
        prefetching=True,          # Overlaps layer compute with PCIe DMA
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    print(f"[+] Model initialized in {time.time() - start_time:.2f} seconds!")

    # Prompt
    prompt = "Explain why GPUs are efficient for parallel computing in 3 bullet points:"
    print(f"\n[*] Input Prompt:\n{prompt}\n")

    input_tokens = model.tokenizer(prompt, return_tensors="pt")
    input_ids = input_tokens["input_ids"]
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    print("[*] Generating response with layer streaming...")
    start_gen = time.time()

    output = model.generate(
        input_ids,
        max_new_tokens=30,
        use_cache=True,
        return_dict_in_generate=True,
    )

    gen_time = time.time() - start_gen
    num_tokens = output.sequences.shape[1] - input_ids.shape[1]

    # Decode response
    response_text = model.tokenizer.decode(output.sequences[0])

    print("\n" + "=" * 60)
    print(f"Result: Generated {num_tokens} tokens in {gen_time:.2f}s ({num_tokens / gen_time:.2f} tokens/sec)")
    print("=" * 60)
    print(response_text)
    print("=" * 60)

if __name__ == "__main__":
    main()
