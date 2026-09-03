"""
StreamLLM: A memory-aware LLM inference runtime that streams transformer layers
between host RAM and GPU memory using asynchronous prefetching and static double buffering.
"""

try:
    from .auto_model import AutoModel, StreamLLMModel, GenerationOutput
except (ImportError, ValueError):
    from auto_model import AutoModel, StreamLLMModel, GenerationOutput

__version__ = "0.1.0"
__all__ = ["AutoModel", "StreamLLMModel", "GenerationOutput"]
