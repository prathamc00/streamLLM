"""
StreamLLM Model Registry and Management.

Provides model preset management, local Hugging Face cache discovery,
download/installation of open-source models, and memory profiling estimates.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

try:
    import huggingface_hub
    from huggingface_hub import scan_cache_dir, snapshot_download
    _HAS_HF_HUB = True
except ImportError:
    _HAS_HF_HUB = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


# Curated presets optimized for StreamLLM layer streaming
AVAILABLE_MODELS: Dict[str, Dict] = {
    "smollm-135m": {
        "repo_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "name": "SmolLM2 135M Instruct",
        "params": "135M",
        "size_str": "~270 MB",
        "vram_streaming_mb": 120,
        "vram_standard_mb": 600,
        "description": "Ultra-lightweight reasoning model; instant download & fast CPU/GPU inference",
    },
    "smollm-360m": {
        "repo_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "name": "SmolLM2 360M Instruct",
        "params": "360M",
        "size_str": "~720 MB",
        "vram_streaming_mb": 220,
        "vram_standard_mb": 1400,
        "description": "Compact model with strong instruction following and math capabilities",
    },
    "smollm-1.7b": {
        "repo_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "name": "SmolLM2 1.7B Instruct",
        "params": "1.7B",
        "size_str": "~3.4 GB",
        "vram_streaming_mb": 420,
        "vram_standard_mb": 4500,
        "description": "Top-performing sub-2B parameter model for complex instruction tasks",
    },
    "qwen-0.5b": {
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "name": "Qwen 2.5 0.5B Instruct",
        "params": "0.5B",
        "size_str": "~980 MB",
        "vram_streaming_mb": 260,
        "vram_standard_mb": 2000,
        "description": "High-efficiency multilingual model, excellent for edge & low-resource devices",
    },
    "qwen-1.5b": {
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "name": "Qwen 2.5 1.5B Instruct",
        "params": "1.5B",
        "size_str": "~3.1 GB",
        "vram_streaming_mb": 400,
        "vram_standard_mb": 4800,
        "description": "Powerful mid-size model for general reasoning, coding, and multi-turn chat",
    },
    "qwen-7b": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "name": "Qwen 2.5 7B Instruct",
        "params": "7B",
        "size_str": "~14.5 GB",
        "vram_streaming_mb": 850,
        "vram_standard_mb": 16000,
        "description": "Leading 7B foundation model for complex coding, mathematics, and agentic workflows",
    },
    "qwen-14b": {
        "repo_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "name": "Qwen 2.5 14B Instruct (AWQ 4-bit)",
        "params": "14B (AWQ)",
        "size_str": "~8.5 GB",
        "vram_streaming_mb": 1200,
        "vram_standard_mb": 10000,
        "description": "High-tier 14B model compressed with 4-bit AWQ, runnable on 4GB VRAM cards",
    },
    "llama3.2-1b": {
        "repo_id": "meta-llama/Llama-3.2-1B-Instruct",
        "name": "Llama 3.2 1B Instruct",
        "params": "1.2B",
        "size_str": "~2.4 GB",
        "vram_streaming_mb": 380,
        "vram_standard_mb": 3600,
        "description": "Meta's highly capable edge model for summarization and agentic tool use",
    },
    "llama3.2-3b": {
        "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
        "name": "Llama 3.2 3B Instruct",
        "params": "3.2B",
        "size_str": "~6.4 GB",
        "vram_streaming_mb": 580,
        "vram_standard_mb": 8000,
        "description": "Meta's compact powerhouse for multilingual reasoning and code synthesis",
    },
    "llama3-8b": {
        "repo_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "name": "Llama 3.1 8B Instruct",
        "params": "8B",
        "size_str": "~16.0 GB",
        "vram_streaming_mb": 950,
        "vram_standard_mb": 18000,
        "description": "Industry benchmark 8B open model with 128k context support",
    },
    "mistral-7b": {
        "repo_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "name": "Mistral 7B Instruct v0.3",
        "params": "7.3B",
        "size_str": "~14.5 GB",
        "vram_streaming_mb": 900,
        "vram_standard_mb": 16500,
        "description": "Fast and versatile 7B reasoning model with native function calling support",
    },
    "deepseek-1.5b": {
        "repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "name": "DeepSeek R1 Distill Qwen 1.5B",
        "params": "1.5B",
        "size_str": "~3.1 GB",
        "vram_streaming_mb": 420,
        "vram_standard_mb": 4800,
        "description": "Distilled reasoning model trained on DeepSeek-R1 chain-of-thought outputs",
    },
    "deepseek-7b": {
        "repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "name": "DeepSeek R1 Distill Qwen 7B",
        "params": "7B",
        "size_str": "~14.5 GB",
        "vram_streaming_mb": 880,
        "vram_standard_mb": 16500,
        "description": "High-accuracy mathematical and code reasoning distilled from R1",
    },
}

# Mapping of aliases to canonical HuggingFace repo IDs
MODEL_PRESETS: Dict[str, str] = {
    alias: meta["repo_id"] for alias, meta in AVAILABLE_MODELS.items()
}


def resolve_model_name(name_or_preset: str) -> str:
    """Resolves preset alias to canonical HuggingFace repo ID or returns input."""
    if not name_or_preset:
        return ""
    key = name_or_preset.strip().lower()
    if key in MODEL_PRESETS:
        return MODEL_PRESETS[key]
    return name_or_preset.strip()


def get_preset_alias(repo_id: str) -> Optional[str]:
    """Returns the preset alias matching a repo ID, if one exists."""
    for alias, meta in AVAILABLE_MODELS.items():
        if meta["repo_id"].lower() == repo_id.lower():
            return alias
    return None


@dataclass
class InstalledModelInfo:
    repo_id: str
    preset_alias: Optional[str]
    size_on_disk_str: str
    size_on_disk_bytes: int
    has_weights: bool
    snapshot_path: Optional[str]


def scan_installed_models() -> Dict[str, InstalledModelInfo]:
    """Scans the local HuggingFace cache to identify installed models and weights."""
    installed: Dict[str, InstalledModelInfo] = {}
    if not _HAS_HF_HUB:
        return installed

    try:
        hf_cache = scan_cache_dir()
    except Exception:
        return installed

    for repo in hf_cache.repos:
        if repo.repo_type != "model":
            continue

        repo_id = repo.repo_id
        # Check if model has actual weight files (safetensors or bin)
        has_weights = False
        snapshot_dir = None

        for rev in repo.revisions:
            if hasattr(rev, "snapshot_path"):
                snapshot_dir = str(rev.snapshot_path)
            for f in rev.files:
                name_lower = f.file_name.lower()
                if name_lower.endswith((".safetensors", ".bin", ".pt", ".pth", ".gguf", ".awq")):
                    has_weights = True
                    break
            if has_weights:
                break

        installed[repo_id.lower()] = InstalledModelInfo(
            repo_id=repo_id,
            preset_alias=get_preset_alias(repo_id),
            size_on_disk_str=repo.size_on_disk_str,
            size_on_disk_bytes=repo.size_on_disk,
            has_weights=has_weights,
            snapshot_path=snapshot_dir,
        )

    return installed


def is_model_installed(name_or_preset: str) -> Tuple[bool, str, Optional[str]]:
    """Checks whether a given model or preset has weights downloaded locally.
    
    Returns:
        (is_installed, size_str, snapshot_path)
    """
    resolved_repo = resolve_model_name(name_or_preset)
    
    # Check if user passed an existing local directory path
    if os.path.isdir(resolved_repo):
        return True, "Local Folder", resolved_repo

    installed_map = scan_installed_models()
    info = installed_map.get(resolved_repo.lower())
    if info and info.has_weights:
        return True, info.size_on_disk_str, info.snapshot_path

    return False, "Not Installed", None


def get_first_installed_model() -> Optional[str]:
    """Returns the first installed preset or model ID, or None if no models are installed."""
    installed_map = scan_installed_models()
    # Check presets first for preferred models
    for preset, meta in AVAILABLE_MODELS.items():
        info = installed_map.get(meta["repo_id"].lower())
        if info and info.has_weights:
            return preset

    # Otherwise return any installed repo ID with weights
    for info in installed_map.values():
        if info.has_weights:
            return info.repo_id

    return None


def install_model(
    name_or_preset: str,
    token: Optional[str] = None,
    force: bool = False,
) -> str:
    """Downloads and installs a model into the local cache.
    
    Args:
        name_or_preset: Preset name (e.g. 'qwen-0.5b') or HuggingFace repo ID.
        token: Optional Hugging Face authorization token for gated models (e.g. LLaMA).
        force: Force re-download even if already present.

    Returns:
        The snapshot path where the model is stored.
    """
    if not _HAS_HF_HUB:
        raise RuntimeError("huggingface_hub is required to install models. Run 'pip install huggingface_hub'.")

    resolved_repo = resolve_model_name(name_or_preset)
    alias = get_preset_alias(resolved_repo)
    display_title = f"{resolved_repo} ({alias})" if alias else resolved_repo

    installed, size_str, snapshot_path = is_model_installed(resolved_repo)
    if installed and not force and snapshot_path:
        print(f"[StreamLLM] Model '{display_title}' is already installed ({size_str}).")
        return snapshot_path

    print(f"[StreamLLM] Installing model '{display_title}'...")
    if alias and alias in AVAILABLE_MODELS:
        meta = AVAILABLE_MODELS[alias]
        print(f"  Parameters:          {meta['params']}")
        print(f"  Est. Download Size:  {meta['size_str']}")
        print(f"  StreamLLM Min VRAM:  ~{meta['vram_streaming_mb']} MB (vs ~{meta['vram_standard_mb']} MB standard)")

    try:
        snapshot_dir = snapshot_download(
            repo_id=resolved_repo,
            repo_type="model",
            token=token,
        )
        print(f"\n[StreamLLM] Successfully installed '{resolved_repo}'!")
        print(f"  Location: {snapshot_dir}")
        print(f"  You can now run: streamllm chat {alias or resolved_repo}")
        return snapshot_dir
    except Exception as e:
        print(f"\n[StreamLLM Error] Failed to install '{resolved_repo}': {e}")
        if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
            print("  Note: This model may be gated. Supply your HF token via --token or 'huggingface-cli login'.")
        raise


def remove_model(name_or_preset: str) -> bool:
    """Removes a model from the local HuggingFace cache."""
    if not _HAS_HF_HUB:
        raise RuntimeError("huggingface_hub is required. Run 'pip install huggingface_hub'.")

    resolved_repo = resolve_model_name(name_or_preset)
    try:
        hf_cache = scan_cache_dir()
    except Exception as e:
        print(f"Error scanning cache: {e}")
        return False

    matching_repos = [r for r in hf_cache.repos if r.repo_id.lower() == resolved_repo.lower()]
    if not matching_repos:
        print(f"Model '{name_or_preset}' ({resolved_repo}) is not currently cached.")
        return False

    commit_hashes = []
    for r in matching_repos:
        for rev in r.revisions:
            commit_hashes.append(rev.commit_hash)

    if not commit_hashes:
        print(f"No revisions found for '{resolved_repo}'.")
        return False

    delete_strategy = hf_cache.delete_revisions(*commit_hashes)
    print(f"Removing '{resolved_repo}' ({delete_strategy.expected_freed_size_str})...")
    delete_strategy.execute()
    print(f"Successfully deleted '{resolved_repo}'.")
    return True


def print_models_table(installed_only: bool = False) -> None:
    """Displays available models and their local installation status."""
    installed_map = scan_installed_models()

    if _HAS_RICH:
        console = Console(safe_box=True)
        title = "Installed Models" if installed_only else "Available StreamLLM Models"
        table = Table(title=title, border_style="cyan", show_header=True, header_style="bold magenta")
        table.add_column("Preset", style="bold cyan", no_wrap=True)
        table.add_column("Model ID", style="white")
        table.add_column("Params", justify="center", style="yellow")
        table.add_column("Download Size", justify="right", style="dim")
        table.add_column("Stream VRAM", justify="right", style="green")
        table.add_column("Status", style="bold")

        displayed_repos = set()

        for preset, meta in AVAILABLE_MODELS.items():
            repo_id = meta["repo_id"]
            displayed_repos.add(repo_id.lower())
            info = installed_map.get(repo_id.lower())
            is_inst = info is not None and info.has_weights

            if installed_only and not is_inst:
                continue

            if is_inst:
                status_text = f"[bold green]Installed ({info.size_on_disk_str})[/bold green]"
            elif info and not info.has_weights:
                status_text = "[yellow]Incomplete (Config only)[/yellow]"
            else:
                status_text = "[dim]Available (Not Installed)[/dim]"

            table.add_row(
                preset,
                repo_id,
                meta["params"],
                meta["size_str"],
                f"~{meta['vram_streaming_mb']} MB",
                status_text,
            )

        # Also show other cached models not in the preset list if any
        other_installed = [
            info for repo_key, info in installed_map.items()
            if repo_key not in displayed_repos and info.has_weights
        ]

        if other_installed:
            for info in other_installed:
                table.add_row(
                    "-",
                    info.repo_id,
                    "Custom",
                    info.size_on_disk_str,
                    "-",
                    f"[bold green]Installed ({info.size_on_disk_str})[/bold green]",
                )

        console.print(table)
        console.print(
            "\n[dim]Tips:\n"
            " - Install a model:   [cyan]streamllm install <preset>[/cyan]  (e.g., [green]streamllm install qwen-0.5b[/green])\n"
            " - Chat with a model: [cyan]streamllm chat <preset>[/cyan]     (e.g., [green]streamllm chat qwen-0.5b[/green])\n"
            " - Run prompt:        [cyan]streamllm run <preset> \"<prompt>\"[/cyan][/dim]\n"
        )
    else:
        print("=" * 85)
        print(f" {'Installed Models' if installed_only else 'Available StreamLLM Models'}")
        print("=" * 85)
        print(f"{'Preset':<15} {'Model ID':<35} {'Params':<8} {'Stream VRAM':<12} {'Status'}")
        print("-" * 85)
        for preset, meta in AVAILABLE_MODELS.items():
            repo_id = meta["repo_id"]
            info = installed_map.get(repo_id.lower())
            is_inst = info is not None and info.has_weights
            if installed_only and not is_inst:
                continue
            status = f"Installed ({info.size_on_disk_str})" if is_inst else "Available"
            print(f"{preset:<15} {repo_id:<35} {meta['params']:<8} ~{meta['vram_streaming_mb']} MB    {status}")
        print("=" * 85)
