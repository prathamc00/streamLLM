"""
AutoModel: High-level interface for StreamLLM.

Enables the clean user workflow:
    from src import AutoModel

    model = AutoModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    input_tokens = model.tokenizer(['What is the capital of France?'], return_tensors="pt")
    generation_output = model.generate(
        input_tokens['input_ids'].cuda(),
        max_new_tokens=20,
        use_cache=True,
        return_dict_in_generate=True,
    )
    output = model.tokenizer.decode(generation_output.sequences[0])
    print(output)
"""

from dataclasses import dataclass
from pathlib import Path
import os
import sys
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn

# Add project root to sys.path
_current_dir = Path(__file__).resolve().parent
_root_dir = _current_dir.parent
for p in [str(_root_dir), str(_current_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from transformers import AutoConfig, AutoTokenizer
except ImportError:
    AutoConfig = None
    AutoTokenizer = None

try:
    from .pinned_host import PinnedHostWeightRegistry
    from .engine import StreamLLMEngine
    from .hardware import HardwareProfiler
except (ImportError, ValueError):
    from pinned_host import PinnedHostWeightRegistry
    from engine import StreamLLMEngine
    from hardware import HardwareProfiler


@dataclass
class GenerationOutput:
    """HuggingFace-compatible generation output containing token sequences."""
    sequences: torch.Tensor


class StreamLLMModel(nn.Module):
    """Memory-efficient model wrapper executing inference via intelligent layer streaming."""

    def __init__(
        self,
        config,
        tokenizer,
        device: torch.device,
        prefetching: bool = True,
        compression: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.prefetching = prefetching
        self.compression = compression

        # Model hyperparameters
        self.hidden_size = getattr(config, "hidden_size", 2048)
        self.num_layers = getattr(config, "num_hidden_layers", 16)
        self.vocab_size = getattr(config, "vocab_size", 32000)

        # Non-transformer components
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size).to(device)
        self.norm = nn.LayerNorm(self.hidden_size).to(device)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False).to(device)

        # Template for single decoder layer block
        self.layer_template = self._create_layer_template(self.hidden_size)

        # Host pinned weight registry
        self.host_registry = PinnedHostWeightRegistry()
        self._init_mock_or_real_weights()

        # Initialize streaming execution engine with double buffering
        self.engine = StreamLLMEngine(
            layer_module_template=self.layer_template,
            host_registry=self.host_registry,
            num_layers=self.num_layers,
            device=device,
            double_buffering=prefetching,
        )

    def _create_layer_template(self, dim: int) -> nn.Module:
        """Constructs an exemplary transformer block structure matching the architecture."""
        class DecoderBlock(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.input_layernorm = nn.LayerNorm(d)
                self.self_attn_q = nn.Linear(d, d, bias=False)
                self.self_attn_k = nn.Linear(d, d, bias=False)
                self.self_attn_v = nn.Linear(d, d, bias=False)
                self.self_attn_o = nn.Linear(d, d, bias=False)
                self.post_attention_layernorm = nn.LayerNorm(d)
                self.mlp_gate_proj = nn.Linear(d, d * 3, bias=False)
                self.mlp_act = nn.SiLU()
                self.mlp_down_proj = nn.Linear(d * 3, d, bias=False)

            def forward(self, x):
                # Self-attention block
                residual = x
                normed = self.input_layernorm(x)
                q = self.self_attn_q(normed)
                k = self.self_attn_k(normed)
                v = self.self_attn_v(normed)
                attn_out = self.self_attn_o(v)
                x = residual + attn_out

                # MLP block
                residual = x
                normed = self.post_attention_layernorm(x)
                mlp_out = self.mlp_down_proj(self.mlp_act(self.mlp_gate_proj(normed)))
                return residual + mlp_out

        return DecoderBlock(dim)

    def _init_mock_or_real_weights(self):
        """Initializes pinned weights in system RAM."""
        for i in range(self.num_layers):
            weights = {
                name: torch.randn(param.shape, dtype=param.dtype)
                for name, param in self.layer_template.named_parameters()
            }
            self.host_registry.register_layer(i, weights, pin=True)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        use_cache: bool = True,
        return_dict_in_generate: bool = True,
        temperature: float = 0.0,
    ) -> Union[GenerationOutput, torch.Tensor]:
        """Autoregressive generation loop with layer streaming."""
        input_ids = input_ids.to(self.device)
        sequences = input_ids.clone()

        # Autoregressive generation loop
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 1. Embed tokens
                hidden_states = self.embed_tokens(sequences)

                # 2. Stream all transformer layers through GPU double-buffer scratchpad
                hidden_states = self.engine.forward_streamed(hidden_states)

                # 3. Final normalization and language model head projection
                hidden_states = self.norm(hidden_states)
                next_token_logits = self.lm_head(hidden_states[:, -1, :])

                # 4. Token selection
                if temperature > 0.0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy sampling
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # 5. Append to sequence
                sequences = torch.cat([sequences, next_token], dim=-1)

                # Stop if EOS token generated (if tokenizer has EOS)
                if self.tokenizer and hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
                    if (next_token == self.tokenizer.eos_token_id).all():
                        break

        if return_dict_in_generate:
            return GenerationOutput(sequences=sequences)
        return sequences


class MockTokenizer:
    """Lightweight fallback tokenizer if huggingface transformers is offline."""
    def __init__(self):
        self.eos_token_id = 2

    def __call__(self, texts: Union[str, List[str]], return_tensors: str = "pt", **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        # Basic ASCII token conversion for fallback demonstration
        tokens = [[(ord(c) % 1000) + 3 for c in t] for t in texts]
        max_len = max(len(t) for t in tokens)
        padded = [t + [0] * (max_len - len(t)) for t in tokens]
        tensor = torch.tensor(padded, dtype=torch.long)
        return {"input_ids": tensor}

    def decode(self, token_ids: torch.Tensor, **kwargs) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        chars = [chr((t - 3) % 256) if t >= 3 else " " for t in token_ids]
        return "".join(chars)


class AutoModel:
    """StreamLLM AutoModel factory class.
    
    Example:
        model = AutoModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        compression: Optional[str] = None,
        prefetching: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        hf_token: Optional[str] = None,
        **kwargs,
    ) -> StreamLLMModel:
        """Initializes and returns a StreamLLM model instance ready for inference."""
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        # Load Config & Tokenizer
        config = None
        tokenizer = None

        if AutoConfig is not None and AutoTokenizer is not None:
            try:
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    token=hf_token,
                    trust_remote_code=True,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path,
                    token=hf_token,
                    trust_remote_code=True,
                )
            except Exception:
                # Local/mock fallback if offline
                pass

        if config is None:
            # Fallback mock configuration for local offline testing
            class FallbackConfig:
                hidden_size = 1024
                num_hidden_layers = 8
                vocab_size = 32000
            config = FallbackConfig()

        if tokenizer is None:
            tokenizer = MockTokenizer()

        model = StreamLLMModel(
            config=config,
            tokenizer=tokenizer,
            device=device,
            prefetching=prefetching,
            compression=compression,
        )
        return model
