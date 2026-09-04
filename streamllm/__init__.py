"""
StreamLLM: A memory-aware LLM inference runtime that streams transformer layers
between host RAM and GPU memory using asynchronous prefetching and static double buffering.
"""

try:
    from .auto_model import AutoModel, StreamLLMModel, GenerationOutput
    from .models import (
        AVAILABLE_MODELS,
        MODEL_PRESETS,
        install_model,
        remove_model,
        is_model_installed,
        scan_installed_models,
        resolve_model_name,
        print_models_table,
    )
except (ImportError, ValueError):
    from auto_model import AutoModel, StreamLLMModel, GenerationOutput
    from models import (
        AVAILABLE_MODELS,
        MODEL_PRESETS,
        install_model,
        remove_model,
        is_model_installed,
        scan_installed_models,
        resolve_model_name,
        print_models_table,
    )

__version__ = "0.1.2"
__all__ = [
    "AutoModel",
    "StreamLLMModel",
    "GenerationOutput",
    "AVAILABLE_MODELS",
    "MODEL_PRESETS",
    "install_model",
    "remove_model",
    "is_model_installed",
    "scan_installed_models",
    "resolve_model_name",
    "print_models_table",
]
