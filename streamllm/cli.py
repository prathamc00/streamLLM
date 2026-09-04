"""
StreamLLM Command-Line Interface.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Optional
import torch
import torch.nn as nn

# Ensure both root directory and src directory are in sys.path
_current_dir = Path(__file__).resolve().parent
_root_dir = _current_dir.parent
for p in [str(_root_dir), str(_current_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from .hardware import HardwareProfiler
    from .pinned_host import PinnedHostWeightRegistry
    from .engine import StreamLLMEngine
    from .auto_model import AutoModel, MODEL_PRESETS
    from .models import (
        AVAILABLE_MODELS,
        install_model,
        remove_model,
        is_model_installed,
        get_first_installed_model,
        print_models_table,
        resolve_model_name,
    )
except (ImportError, ValueError):
    from streamllm.hardware import HardwareProfiler
    from streamllm.pinned_host import PinnedHostWeightRegistry
    from streamllm.engine import StreamLLMEngine
    from streamllm.auto_model import AutoModel, MODEL_PRESETS
    from streamllm.models import (
        AVAILABLE_MODELS,
        install_model,
        remove_model,
        is_model_installed,
        get_first_installed_model,
        print_models_table,
        resolve_model_name,
    )

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.text import Text
    from rich import box
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


def get_console() -> Optional["Console"]:
    """Returns a Rich console configured for universal safe rendering across all terminals."""
    if not _HAS_RICH:
        return None
    return Console(safe_box=True)


def resolve_device(device_arg) -> torch.device:
    """Helper to parse device arguments (e.g. 'auto', '0', 'cuda:0', 'cpu')."""
    if device_arg is None or device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device_arg, int) or str(device_arg).isdigit():
        return torch.device(f"cuda:{device_arg}")
    return torch.device(str(device_arg))


def resolve_chat_model(requested_model: Optional[str]) -> Optional[str]:
    """Ensures a model is selected and installed before running chat or run."""
    console = get_console()

    if requested_model:
        is_inst, size_str, _ = is_model_installed(requested_model)
        if not is_inst:
            resolved = resolve_model_name(requested_model)
            if console:
                console.print(f"[yellow]Model '{requested_model}' ({resolved}) is not installed locally.[/yellow]")
                try:
                    choice = Prompt.ask(
                        "[bold cyan]Would you like to install it now?[/bold cyan]",
                        choices=["y", "n"],
                        default="y",
                    )
                    if choice.lower() == "y":
                        install_model(requested_model)
                        return requested_model
                    else:
                        console.print("[dim]Aborted. Run 'streamllm list' to view available models.[/dim]")
                        return None
                except (KeyboardInterrupt, EOFError):
                    return None
            else:
                print(f"Model '{requested_model}' ({resolved}) is not installed locally.")
                print(f"Run: streamllm install {requested_model}")
                return None
        return requested_model

    # If no model was passed, look for the first installed model
    first_inst = get_first_installed_model()
    if first_inst:
        if console:
            console.print(f"[dim]No model specified. Using installed model: [bold cyan]{first_inst}[/bold cyan][/dim]")
        else:
            print(f"No model specified. Using installed model: {first_inst}")
        return first_inst

    # No models installed at all: guide the user
    if console:
        console.print("[yellow]No models are installed yet.[/yellow]\n")
        print_models_table()
        try:
            choice = Prompt.ask(
                "[bold cyan]Enter a preset model to install (or press Enter to exit)[/bold cyan]",
                default="smollm-135m",
            )
            if choice:
                install_model(choice)
                return choice
        except (KeyboardInterrupt, EOFError):
            return None
    else:
        print("No models are installed yet.")
        print("Run 'streamllm list' to see available models, or 'streamllm install smollm-135m'.")
    return None


def cmd_hardware(args):
    profiler = HardwareProfiler(device_idx=args.device)
    profile = profiler.profile(benchmark_size_mb=args.bench_mb)

    console = get_console()
    if console:
        table = Table(title="StreamLLM Hardware & PCIe Diagnostic Profile", border_style="cyan")
        table.add_column("Property", style="bold white", no_wrap=True)
        table.add_column("Value", style="green")

        table.add_row("Device Name", profile.device_name)
        table.add_row("Compute Capability", profile.compute_capability)
        table.add_row("Total GPU VRAM", f"{profile.total_vram_mb:.1f} MB")
        table.add_row("Available Free VRAM", f"{profile.free_vram_mb:.1f} MB")
        table.add_row("Total System RAM", f"{profile.total_ram_mb:.1f} MB")
        table.add_row("PCIe H2D Bandwidth", f"{profile.pcie_h2d_bandwidth_gb_s:.2f} GB/s")
        table.add_row("Recommended VRAM Budget", f"{profile.recommended_vram_budget_mb:.1f} MB")
        console.print(table)
    else:
        print("==================================================")
        print(" StreamLLM Hardware & PCIe Diagnostic Profile")
        print("==================================================")
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
        registry.register_layer(i, layer_weights, pin=(device.type == "cuda"))

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


def cmd_run(args):
    """Execute a single one-shot prompt with streaming output."""
    model_name = resolve_chat_model(args.model)
    if not model_name:
        return

    prompt_text = args.prompt or args.prompt_opt
    if not prompt_text:
        if not sys.stdin.isatty():
            prompt_text = sys.stdin.read().strip()
        else:
            console = get_console()
            if console:
                prompt_text = Prompt.ask("[bold green]Enter prompt[/bold green]")
            else:
                prompt_text = input("Enter prompt: ")

    if not prompt_text:
        print("Error: No prompt provided.")
        return

    device = resolve_device(args.device)
    console = get_console() if not args.raw else None

    if not args.raw and console:
        console.print(f"[dim]Loading '{model_name}' on {device}...[/dim]")

    model = AutoModel.from_pretrained(model_name, device=device)

    if hasattr(model.tokenizer, "apply_chat_template") and getattr(model.tokenizer, "chat_template", None):
        try:
            messages = [{"role": "user", "content": prompt_text}]
            input_text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            input_text = prompt_text
    else:
        input_text = prompt_text

    tokens = model.tokenizer(input_text, return_tensors="pt")
    input_ids = tokens["input_ids"]

    if not args.raw and console:
        console.print(f"\n[bold green]User:[/bold green] {prompt_text}")
        console.print("[bold cyan]Assistant:[/bold cyan] ", end="")
    elif not args.raw:
        print(f"\nUser: {prompt_text}")
        print("Assistant: ", end="", flush=True)

    start_time = time.perf_counter()
    token_count = 0

    for chunk in model.stream_generate(input_ids, max_new_tokens=args.max_tokens, temperature=args.temperature):
        token_count += 1
        if args.raw:
            sys.stdout.write(chunk)
            sys.stdout.flush()
        elif console:
            console.print(chunk, end="", highlight=False)
        else:
            sys.stdout.write(chunk)
            sys.stdout.flush()

    elapsed = time.perf_counter() - start_time
    tps = token_count / elapsed if elapsed > 0 else 0.0

    if not args.raw:
        print()
        if console:
            console.print(
                f"\n[dim][Stats] Generated {token_count} tokens in {elapsed:.2f}s "
                f"({tps:.1f} tok/s) | Layer Streaming Scratchpad Active[/dim]\n"
            )
        else:
            print(f"\n[Generated {token_count} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)]\n")


def cmd_chat(args):
    """Start an interactive multi-turn terminal chat REPL."""
    model_name = resolve_chat_model(args.model)
    if not model_name:
        return

    device = resolve_device(args.device)
    console = get_console()
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"

    # Display Welcoming Header
    if console:
        banner_content = (
            f"[bold green]StreamLLM Asynchronous Layer Streaming[/bold green]\n\n"
            f"[bold white]Model:[/bold white]       [cyan]{model_name}[/cyan]\n"
            f"[bold white]Device:[/bold white]      [yellow]{device_name}[/yellow] ({device})\n"
            f"[bold white]Memory Mode:[/bold white] [magenta]Double-Buffered Static Scratchpad Pool[/magenta]\n\n"
            f"[dim]Commands:[/dim] [cyan]/help[/cyan], [cyan]/stats[/cyan], [cyan]/system[/cyan], [cyan]/clear[/cyan], [cyan]/exit[/cyan]"
        )
        console.print(Panel(banner_content, title="[bold cyan]StreamLLM Interactive Chat[/bold cyan]", border_style="cyan"))
        console.print(f"[dim]Initializing model '{model_name}'...[/dim]")
    else:
        print(f"=== StreamLLM Chat: {model_name} on {device_name} ===")

    model = AutoModel.from_pretrained(model_name, device=device)
    system_prompt = args.system

    if console:
        console.print("[bold green][Ready] Type your message or slash command below:[/bold green]\n")
    else:
        print("Ready! Type your message or /exit to quit.\n")

    history = []

    while True:
        try:
            if console:
                user_input = Prompt.ask("[bold cyan]>>> [/bold cyan]").strip()
            else:
                user_input = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            if console:
                console.print("\n[yellow]Session ended. Goodbye![/yellow]")
            else:
                print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Process Slash Commands
        if user_input.startswith("/"):
            parts = user_input.split(" ", 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/exit", "/quit", "/q"):
                if console:
                    console.print("[yellow]Exiting StreamLLM. Goodbye![/yellow]")
                else:
                    print("Goodbye!")
                break
            elif cmd in ("/help", "/h", "/?"):
                if console:
                    table = Table(title="Available Slash Commands", border_style="dim")
                    table.add_column("Command", style="cyan bold")
                    table.add_column("Description", style="white")
                    table.add_row("/stats", "Display live GPU VRAM & scratchpad memory telemetry")
                    table.add_row("/system <prompt>", "Update the system instruction prompt")
                    table.add_row("/clear", "Clear terminal screen and reset conversation")
                    table.add_row("/help", "Show this help guide")
                    table.add_row("/exit, /quit", "Exit the chat session")
                    console.print(table)
                else:
                    print("/stats  - Hardware stats\n/system <prompt> - Update system prompt\n/clear  - Clear screen\n/exit   - Quit")
                continue
            elif cmd == "/clear":
                if console:
                    console.clear()
                    console.print("[dim]Chat history reset.[/dim]\n")
                else:
                    print("Chat history reset.\n")
                history.clear()
                continue
            elif cmd == "/system":
                if arg:
                    system_prompt = arg
                    if console:
                        console.print(f"[green][OK] System prompt updated:[/green] {system_prompt}\n")
                    else:
                        print(f"System prompt updated: {system_prompt}\n")
                else:
                    if console:
                        console.print(f"[dim]Current system prompt:[/dim] {system_prompt}\n")
                    else:
                        print(f"Current system prompt: {system_prompt}\n")
                continue
            elif cmd == "/stats":
                if console:
                    table = Table(title="Live Hardware & VRAM Telemetry", border_style="cyan")
                    table.add_column("Metric", style="bold white")
                    table.add_column("Status", style="green")
                    table.add_row("Device", device_name)
                    if device.type == "cuda":
                        free_b, total_b = torch.cuda.mem_get_info(device)
                        table.add_row("GPU Total VRAM", f"{total_b / (1024*1024):.1f} MB")
                        table.add_row("GPU Free VRAM", f"{free_b / (1024*1024):.1f} MB")
                        table.add_row("VRAM Allocated", f"{torch.cuda.memory_allocated(device) / (1024*1024):.1f} MB")
                    table.add_row("Scratchpad Pool", "Pre-allocated Static Slots A & B")
                    table.add_row("Overlapping Engine", "Dual CUDA Streams (Compute + PCIe Transfer)")
                    console.print(table)
                    console.print()
                else:
                    print(f"Device: {device_name}")
                continue
            else:
                if console:
                    console.print(f"[red]Unknown command: {cmd}. Type /help for options.[/red]")
                else:
                    print(f"Unknown command: {cmd}")
                continue

        # Autoregressive generation
        if hasattr(model.tokenizer, "apply_chat_template") and getattr(model.tokenizer, "chat_template", None):
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                for u, a in history:
                    messages.append({"role": "user", "content": u})
                    messages.append({"role": "assistant", "content": a})
                messages.append({"role": "user", "content": user_input})
                prompt_with_context = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt_with_context = f"System: {system_prompt}\nUser: {user_input}\nAssistant:"
        else:
            prompt_with_context = f"System: {system_prompt}\nUser: {user_input}\nAssistant:"

        tokens = model.tokenizer(prompt_with_context, return_tensors="pt")
        input_ids = tokens["input_ids"]

        if console:
            console.print("\n[bold green]Assistant:[/bold green] ", end="")
        else:
            print("\nAssistant: ", end="", flush=True)

        start_time = time.perf_counter()
        token_count = 0
        response_text = ""

        for chunk in model.stream_generate(input_ids, max_new_tokens=args.max_tokens, temperature=args.temperature):
            token_count += 1
            response_text += chunk
            if console:
                console.print(chunk, end="", highlight=False)
            else:
                sys.stdout.write(chunk)
                sys.stdout.flush()

        elapsed = time.perf_counter() - start_time
        tps = token_count / elapsed if elapsed > 0 else 0.0

        if console:
            console.print(f"\n[dim][Stats] {token_count} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)[/dim]\n")
        else:
            print(f"\n[{token_count} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)]\n")

        history.append((user_input, response_text))


def cmd_models(args):
    """Subcommand dispatcher for model management."""
    action = getattr(args, "action", "list")
    if not action or action == "list":
        print_models_table(installed_only=getattr(args, "installed", False))
    elif action in ("install", "pull"):
        model_target = getattr(args, "model_name", None)
        if not model_target:
            print("Error: Specify model to install. Example: streamllm install qwen-0.5b")
            return
        install_model(model_target, token=getattr(args, "token", None), force=getattr(args, "force", False))
    elif action in ("remove", "rm", "delete"):
        model_target = getattr(args, "model_name", None)
        if not model_target:
            print("Error: Specify model to remove. Example: streamllm models remove qwen-0.5b")
            return
        remove_model(model_target)
    else:
        print(f"Unknown action: '{action}'. Available: list, install, remove")


def cmd_list(args):
    """List available and installed models."""
    print_models_table(installed_only=getattr(args, "installed", False))


def cmd_install(args):
    """Download and install a model."""
    if not args.model:
        print("Error: Specify model to install. Example: streamllm install qwen-0.5b")
        return
    install_model(args.model, token=getattr(args, "token", None), force=getattr(args, "force", False))


def main():
    parser = argparse.ArgumentParser(
        description="StreamLLM CLI: Run large language models on low VRAM GPUs with layer streaming.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # chat
    p_chat = subparsers.add_parser("chat", help="Start an interactive chat session with live streaming")
    p_chat.add_argument("model", nargs="?", default=None, help="Model preset (e.g. qwen-0.5b, smollm-135m, llama3-8b) or HuggingFace repo ID")
    p_chat.add_argument("--system", type=str, default="You are a helpful and concise AI assistant.", help="System prompt")
    p_chat.add_argument("--max-tokens", type=int, default=128, help="Maximum new tokens per turn")
    p_chat.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p_chat.add_argument("--device", type=str, default="auto", help="Execution device (auto, cuda:0, cpu)")
    p_chat.set_defaults(func=cmd_chat)

    # run
    p_run = subparsers.add_parser("run", help="Run a one-shot prompt with streaming output")
    p_run.add_argument("model", nargs="?", default=None, help="Model preset (e.g. qwen-0.5b, smollm-135m, llama3-8b) or HuggingFace repo ID")
    p_run.add_argument("prompt", nargs="?", default=None, help="Prompt text to send to the model")
    p_run.add_argument("--prompt", "-p", dest="prompt_opt", default=None, help="Alternative prompt flag")
    p_run.add_argument("--max-tokens", type=int, default=64, help="Maximum new tokens to generate")
    p_run.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p_run.add_argument("--device", type=str, default="auto", help="Execution device (auto, cuda:0, cpu)")
    p_run.add_argument("--raw", action="store_true", help="Print only raw tokens (useful for piping to scripts)")
    p_run.set_defaults(func=cmd_run)

    # list (convenience alias)
    p_list = subparsers.add_parser("list", help="List available and locally installed models")
    p_list.add_argument("--installed", action="store_true", help="Show only locally installed models")
    p_list.set_defaults(func=cmd_list)

    # install / pull (convenience aliases)
    p_install = subparsers.add_parser("install", help="Download and install a model into local cache")
    p_install.add_argument("model", help="Preset name (e.g. qwen-0.5b, smollm-135m) or HuggingFace repo ID")
    p_install.add_argument("--token", type=str, default=None, help="Hugging Face authorization token")
    p_install.add_argument("--force", action="store_true", help="Force re-download even if already present")
    p_install.set_defaults(func=cmd_install)

    p_pull = subparsers.add_parser("pull", help="Alias for 'install'")
    p_pull.add_argument("model", help="Preset name or HuggingFace repo ID")
    p_pull.add_argument("--token", type=str, default=None, help="Hugging Face authorization token")
    p_pull.add_argument("--force", action="store_true", help="Force re-download even if already present")
    p_pull.set_defaults(func=cmd_install)

    # models (full management group)
    p_models = subparsers.add_parser("models", help="Model management: list, install, remove")
    p_models_sub = p_models.add_subparsers(dest="action", help="Action to perform")

    p_m_list = p_models_sub.add_parser("list", help="List available and locally installed models")
    p_m_list.add_argument("--installed", action="store_true", help="Show only installed models")
    p_m_list.set_defaults(func=cmd_models)

    p_m_inst = p_models_sub.add_parser("install", help="Install a model")
    p_m_inst.add_argument("model_name", help="Preset name or HuggingFace repo ID")
    p_m_inst.add_argument("--token", type=str, default=None, help="Hugging Face token")
    p_m_inst.add_argument("--force", action="store_true", help="Force re-download")
    p_m_inst.set_defaults(func=cmd_models)

    p_m_pull = p_models_sub.add_parser("pull", help="Pull a model (alias for install)")
    p_m_pull.add_argument("model_name", help="Preset name or HuggingFace repo ID")
    p_m_pull.add_argument("--token", type=str, default=None, help="Hugging Face token")
    p_m_pull.add_argument("--force", action="store_true", help="Force re-download")
    p_m_pull.set_defaults(func=cmd_models)

    p_m_rm = p_models_sub.add_parser("remove", help="Remove a model from cache")
    p_m_rm.add_argument("model_name", help="Preset name or HuggingFace repo ID to remove")
    p_m_rm.set_defaults(func=cmd_models)

    p_models.set_defaults(func=cmd_models)

    # hardware
    p_hw = subparsers.add_parser("hardware", help="Display GPU & PCIe diagnostic profile")
    p_hw.add_argument("--device", type=int, default=0, help="CUDA device index")
    p_hw.add_argument("--bench-mb", type=int, default=128, help="PCIe benchmark buffer size in MB")
    p_hw.set_defaults(func=cmd_hardware)

    # bench
    p_bench = subparsers.add_parser("bench", help="Run streaming vs sequential micro-benchmark")
    p_bench.add_argument("--device", type=int, default=0, help="CUDA device index")
    p_bench.add_argument("--layers", type=int, default=16, help="Number of layers to stream")
    p_bench.add_argument("--hidden-dim", type=int, default=2048, help="Hidden dimension size")
    p_bench.add_argument("--seq-len", type=int, default=128, help="Sequence length (prefill tokens)")
    p_bench.add_argument("--runs", type=int, default=3, help="Benchmark repeat runs")
    p_bench.set_defaults(func=cmd_bench)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
