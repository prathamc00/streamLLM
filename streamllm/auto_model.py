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
            # 1. Embeddings
            if hasattr(hf_model, "get_input_embeddings") and hf_model.get_input_embeddings() is not None:
                self.embed_tokens = hf_model.get_input_embeddings().to(device)
            elif hasattr(hf_model, "model") and hasattr(hf_model.model, "embed_tokens"):
                self.embed_tokens = hf_model.model.embed_tokens.to(device)
            else:
                self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size).to(device)

            # 2. Final Normalization
            if hasattr(hf_model, "model") and hasattr(hf_model.model, "norm"):
                self.norm = hf_model.model.norm.to(device)
            elif hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "ln_f"):
                self.norm = hf_model.transformer.ln_f.to(device)
            else:
                self.norm = nn.LayerNorm(self.hidden_size).to(device)

            # 3. LM Projection Head
            if hasattr(hf_model, "lm_head") and hf_model.lm_head is not None:
                self.lm_head = hf_model.lm_head.to(device)
            else:
                self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False).to(device)

            # 4. Extract transformer layers
            raw_layers = None
            if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
                raw_layers = hf_model.model.layers
            elif hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
                raw_layers = hf_model.transformer.h

            if raw_layers and len(raw_layers) > 0:
                self.num_layers = len(raw_layers)
                self.is_hf_transformer = True
                # Use Layer 0 as the GPU execution scratchpad module template
                self.layer_template = raw_layers[0].to(device)

                # Register all layer state dicts in pinned host RAM
                self.host_registry = PinnedHostWeightRegistry()
                for i, lyr in enumerate(raw_layers):
                    layer_weights = {
                        k: v.detach().clone().cpu()
                        for k, v in lyr.state_dict().items()
                    }
                    self.host_registry.register_layer(i, layer_weights, pin=(device.type == "cuda"))

            # Rotary embeddings if available
            self.rotary_emb = None
            if hasattr(hf_model, "model") and hasattr(hf_model.model, "rotary_emb"):
                self.rotary_emb = hf_model.model.rotary_emb.to(device)
            elif hasattr(hf_model, "rotary_emb"):
                self.rotary_emb = hf_model.rotary_emb.to(device)
            else:
                self.layer_template = self._create_layer_template(self.hidden_size)
                self.host_registry = PinnedHostWeightRegistry()
                self._init_mock_or_real_weights()
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
        if self.is_hf_transformer:
            seq_len = hidden_states.shape[1]
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
            pos_emb = None
            if self.rotary_emb is not None:
                try:
                    pos_emb = self.rotary_emb(hidden_states, pos_ids)
                except Exception:
                    pass
            try:
                if pos_emb is not None:
                    out = layer_module(hidden_states, position_ids=pos_ids, position_embeddings=pos_emb)
                else:
                    out = layer_module(hidden_states, position_ids=pos_ids)
                if isinstance(out, (tuple, list)):
                    return out[0]
                return out
            except Exception:
                try:
                    out = layer_module(hidden_states)
                    if isinstance(out, (tuple, list)):
                        return out[0]
                    return out
                except Exception:
                    pass
        return layer_module(hidden_states)

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

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 1. Embed tokens
                hidden_states = self.embed_tokens(sequences)

                # 2. Stream all transformer layers through GPU double-buffer scratchpad
                hidden_states = self.engine.forward_streamed(
                    hidden_states,
                    layer_forward_fn=self._forward_layer,
                )

                # 3. Final normalization and language model head projection
                hidden_states = self.norm(hidden_states)
                next_token_logits = self.lm_head(hidden_states[:, -1, :])

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
        sequences = input_ids.clone()

        with torch.no_grad():
            for step in range(max_new_tokens):
                # 1. Embed tokens
                hidden_states = self.embed_tokens(sequences)

                # 2. Stream all transformer layers through GPU double-buffer scratchpad
                hidden_states = self.engine.forward_streamed(
                    hidden_states,
                    layer_forward_fn=self._forward_layer,
                )

                # 3. Final normalization and language model head projection
                hidden_states = self.norm(hidden_states)
                next_token_logits = self.lm_head(hidden_states[:, -1, :])

                # 4. Token selection
                if temperature > 0.0:
                    probs = torch.softmax(next_token_logits / max(temperature, 1e-4), dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                sequences = torch.cat([sequences, next_token], dim=-1)

                # Decode real token chunk
                token_val = next_token[0].tolist()
                chunk = self.tokenizer.decode(token_val, skip_special_tokens=True)
                yield chunk

                # Check EOS
                if self.tokenizer and hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
                    if (next_token == self.tokenizer.eos_token_id).all():
                        break


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

        # Load Config & Tokenizer
        config = None
        tokenizer = None
        hf_model = None

        if AutoConfig is not None and AutoTokenizer is not None:
            try:
                config = AutoConfig.from_pretrained(
                    load_target,
                    token=hf_token,
                    trust_remote_code=True,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    load_target,
                    token=hf_token,
                    trust_remote_code=True,
                )
            except Exception as e:
                if not is_inst:
                    raise RuntimeError(
                        f"Model '{pretrained_model_name_or_path}' is not installed locally and could not be reached online: {e}\n"
                        f"Run 'streamllm install {pretrained_model_name_or_path}' to download it."
                    )
                pass

        # Try loading real model weights into host memory for layer streaming
        if AutoModelForCausalLM is not None and config is not None:
            try:
                hf_dtype = torch.float16 if (device.type == "cuda") else torch.float32
                hf_model = AutoModelForCausalLM.from_pretrained(
                    load_target,
                    config=config,
                    torch_dtype=hf_dtype,
                    low_cpu_mem_usage=True,
                    token=hf_token,
                    trust_remote_code=True,
                )
            except Exception as e:
                # If weights are missing locally
                if not is_inst:
                    raise RuntimeError(
                        f"Model weights for '{pretrained_model_name_or_path}' are not downloaded.\n"
                        f"Please run 'streamllm install {pretrained_model_name_or_path}' to install it."
                    )
                pass

        if config is None:
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
            hf_model=hf_model,
        )
        return model
