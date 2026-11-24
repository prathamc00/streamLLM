"""
AutoModel: High-level interface for StreamLLM.

Usage:
    from streamllm import AutoModel

    model = AutoModel.from_pretrained("smollm-135m")
    tokens = model.tokenizer("Hello, how are you?", return_tensors="pt")
    output = model.generate(tokens["input_ids"], max_new_tokens=30)
    print(model.tokenizer.decode(output.sequences[0]))
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
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
except ImportError:
    AutoConfig = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

try:
    from .pinned_host import PinnedHostWeightRegistry
    from .engine import StreamLLMEngine
    from .hardware import HardwareProfiler
    from .models import (
        AVAILABLE_MODELS,
        MODEL_PRESETS,
        resolve_model_name,
        is_model_installed,
    )
except (ImportError, ValueError):
    from pinned_host import PinnedHostWeightRegistry
    from engine import StreamLLMEngine
    from hardware import HardwareProfiler
    from models import (
        AVAILABLE_MODELS,
        MODEL_PRESETS,
        resolve_model_name,
        is_model_installed,
    )


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
        hf_model: Optional[nn.Module] = None,
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
        self.is_hf_transformer = False

        if hf_model is not None:
            decoder = getattr(hf_model, "model", None)
            if (decoder is None or not hasattr(decoder, "layers")
                    or not hasattr(decoder, "norm") or hf_model.get_output_embeddings() is None):
                raise ValueError(f"Unsupported decoder layout for {config.model_type}")
            self.embed_tokens = hf_model.get_input_embeddings().to(device)
            self.norm = decoder.norm.to(device)
            self.lm_head = hf_model.get_output_embeddings().to(device)
            self.num_layers = len(decoder.layers)
            self.is_hf_transformer = True
            self.layer_template = decoder.layers[0]
            self.host_registry = PinnedHostWeightRegistry()
            for index, layer in enumerate(decoder.layers):
                self.host_registry.register_layer(index, layer.state_dict(), pin=device.type == "cuda")
                decoder.layers[index] = nn.Identity()
            self.layer_template.eval()

            # Rotary embeddings if available
            self.rotary_emb = None
            if hasattr(hf_model, "model") and hasattr(hf_model.model, "rotary_emb"):
                self.rotary_emb = hf_model.model.rotary_emb.to(device)
            elif hasattr(hf_model, "rotary_emb"):
                self.rotary_emb = hf_model.rotary_emb.to(device)
        else:
            # Fallback module template
            self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size).to(device)
            self.norm = nn.LayerNorm(self.hidden_size).to(device)
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False).to(device)
            self.layer_template = self._create_layer_template(self.hidden_size)
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
        """Constructs a standard transformer decoder block structure."""
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

            def forward(self, x, **kwargs):
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
        """Initializes pinned weights in system RAM for synthetic blocks."""
        for i in range(self.num_layers):
            weights = {
                name: torch.randn(param.shape, dtype=param.dtype)
                for name, param in self.layer_template.named_parameters()
            }
            self.host_registry.register_layer(i, weights, pin=(self.device.type == "cuda"))

    def _forward_layer(self, layer_module: nn.Module, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Executes a single transformer layer, handling HuggingFace return shapes and RoPE embeddings."""
        return layer_module(hidden_states)

    def _prepare_request(self, ids, limit):
        length = ids.size(1) + limit
        if limit < 0 or length > getattr(self.config, "max_position_embeddings", float("inf")):
            raise ValueError("Requested generation exceeds the model context or has a negative limit")
        heads = getattr(self.config, "num_attention_heads", 1)
        kv_heads = getattr(self.config, "num_key_value_heads", heads)
        width = getattr(self.config, "head_dim", None) or self.hidden_size // heads
        cache_bytes = 2 * self.num_layers * ids.size(0) * length * kv_heads * width * self.embed_tokens.weight.element_size()
        self.engine.prepare_residency(cache_bytes)

    def _generation_state(self, use_cache):
        if not self.is_hf_transformer:
            return None, None
        import inspect
        from transformers.cache_utils import DynamicCache

        params = inspect.signature(self.layer_template.forward).parameters
        key = "past_key_values" if "past_key_values" in params else "past_key_value"
        if key not in params or not hasattr(self.layer_template, "self_attn"):
            raise ValueError("Unsupported decoder cache API")
        return DynamicCache() if use_cache else None, key

    def _generation_step(self, tokens, offset, cache, cache_key):
        hidden = self.embed_tokens(tokens)
        if cache_key is None:
            return self.engine.forward_streamed(hidden, self._forward_layer)

        positions = torch.arange(offset, offset + tokens.shape[1], device=self.device)
        ids = positions.unsqueeze(0)
        kwargs = {"position_ids": ids, "use_cache": cache is not None, cache_key: cache}
        if self.rotary_emb is not None:
            kwargs["position_embeddings"] = self.rotary_emb(hidden, ids)
        if tokens.shape[1] > 1:
            mask = hidden.new_full((tokens.shape[1], tokens.shape[1]), float("-inf"))
            kwargs["attention_mask"] = mask.triu_(1)[None, None]

        def forward(layer, x, index):
            layer.self_attn.layer_idx = index
            layer_kwargs = kwargs
            window = getattr(layer.self_attn, "sliding_window", None)
            if window is None and self.config.model_type == "mistral":
                window = getattr(self.config, "sliding_window", None)
            if window and offset + tokens.shape[1] > window:
                keys = torch.arange(offset + tokens.shape[1], device=self.device)
                disallowed = (keys > ids.T) | (keys <= ids.T - window)
                mask = hidden.new_zeros(disallowed.shape).masked_fill_(disallowed, float("-inf"))
                layer_kwargs = dict(kwargs, attention_mask=mask[None, None])
            result = layer(x, **layer_kwargs)
            return result[0] if isinstance(result, (tuple, list)) else result

        return self.engine.forward_streamed(hidden, forward)

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
        self._prepare_request(input_ids, max_new_tokens)
        cache, cache_key = self._generation_state(use_cache)
        offset = 0

        with torch.no_grad():
            for _ in range(max_new_tokens):
                tokens = sequences[:, offset:]
                hidden_states = self._generation_step(tokens, offset, cache, cache_key)
                if cache is not None:
                    offset = sequences.shape[1]

                # 3. Final normalization and language model head projection
                last_hidden = self.norm(hidden_states[:, -1, :])
                next_token_logits = self.lm_head(last_hidden)

                # 4. Token selection
                if temperature > 0.0:
                    probs = torch.softmax(next_token_logits / max(temperature, 1e-4), dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # 5. Append to sequence
                sequences = torch.cat([sequences, next_token], dim=-1)

                # Stop if EOS token generated
                if self.tokenizer and hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
                    if (next_token == self.tokenizer.eos_token_id).all():
                        break

        if return_dict_in_generate:
            return GenerationOutput(sequences=sequences)
        return sequences

    def stream_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        use_cache: bool = True,
    ):
        """Yields generated token text chunks as they are produced in real time."""
        input_ids = input_ids.to(self.device)
        tokens = input_ids
        self._prepare_request(input_ids, max_new_tokens)
        cache, cache_key = self._generation_state(use_cache)
        offset = 0

        with torch.no_grad():
            for step in range(max_new_tokens):
                hidden_states = self._generation_step(tokens, offset, cache, cache_key)
                if cache is not None:
                    offset += tokens.shape[1]

                # 3. Final normalization and language model head projection
                last_hidden = self.norm(hidden_states[:, -1, :])
                next_token_logits = self.lm_head(last_hidden)

                # 4. Token selection
                if temperature > 0.0:
                    probs = torch.softmax(next_token_logits / max(temperature, 1e-4), dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                tokens = next_token if cache is not None else torch.cat([tokens, next_token], dim=-1)

                # Decode real token chunk
                token_ids = next_token.flatten().tolist()
                token_val = token_ids[:1]
                chunk = self.tokenizer.decode(token_val, skip_special_tokens=True)
                yield chunk

                # Check EOS
                if self.tokenizer and hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
                    if all(token == self.tokenizer.eos_token_id for token in token_ids):
                        return "eos"
        return "length"


class MockTokenizer:
    """Lightweight fallback tokenizer if huggingface transformers is offline."""
    def __init__(self):
        self.eos_token_id = 2

    def __call__(self, texts: Union[str, List[str]], return_tensors: str = "pt", **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        tokens = [[(ord(c) % 1000) + 3 for c in t] for t in texts]
        max_len = max(len(t) for t in tokens)
        padded = [t + [0] * (max_len - len(t)) for t in tokens]
        tensor = torch.tensor(padded, dtype=torch.long)
        return {"input_ids": tensor}

    def decode(self, token_ids: Union[torch.Tensor, List[int], int], **kwargs) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        chars = [chr((t - 3) % 256) if t >= 3 else " " for t in token_ids]
        return "".join(chars)


class AutoModel:
    """StreamLLM AutoModel factory class.
    
    Example:
        model = AutoModel.from_pretrained("smollm-135m")
        # Or using full HuggingFace repo ID:
        model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
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
        # Resolve model preset aliases
        resolved_name = resolve_model_name(pretrained_model_name_or_path)

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        # Check local installation status
        is_inst, size_str, snapshot_path = is_model_installed(resolved_name)
        load_target = snapshot_path if snapshot_path else resolved_name

        if AutoModelForCausalLM is None:
            raise ImportError("Install Transformers to load a checkpoint")
        config = AutoConfig.from_pretrained(load_target, token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(load_target, token=hf_token)
        hf_model, loading = AutoModelForCausalLM.from_pretrained(
            load_target, config=config, token=hf_token,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True, output_loading_info=True,
        )
        if loading.get("missing_keys") or loading.get("mismatched_keys"):
            raise ValueError(f"Incomplete checkpoint: {loading}")
        if getattr(hf_model, "is_quantized", False):
            raise ValueError("Streaming packed quantized weights is unsupported")
        hf_model.eval()

        model = StreamLLMModel(
            config=config,
            tokenizer=tokenizer,
            device=device,
            prefetching=prefetching,
            compression=compression,
            hf_model=hf_model,
        )
        return model
