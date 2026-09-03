"""
KV-Cache Lifecycle and Memory Budget Tracker.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch


@dataclass
class KVCacheConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_context_length: int
    dtype: torch.dtype = torch.float16


class KVCacheBudgetTracker:
    """Calculates and monitors Key-Value cache memory requirements."""

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.bytes_per_elem = torch.tensor([], dtype=config.dtype).element_size()

    def calculate_memory_mb(self, sequence_length: int) -> float:
        """Calculates exact KV-cache size in megabytes for a given sequence length.
        
        Formula: 2 * num_layers * num_kv_heads * head_dim * sequence_length * bytes_per_elem
        (Factor of 2 accounts for both Keys and Values)
        """
        total_elements = (
            2
            * self.config.num_layers
            * self.config.num_kv_heads
            * self.config.head_dim
            * sequence_length
        )
        return (total_elements * self.bytes_per_elem) / (1024 * 1024)

    def max_context_memory_mb(self) -> float:
        return self.calculate_memory_mb(self.config.max_context_length)


class LayerKVCache:
    """Stores Key and Value activation tensors for one transformer layer."""

    def __init__(self, device: torch.device):
        self.device = device
        self.key: Optional[torch.Tensor] = None
        self.value: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.key is None:
            self.key = new_k
            self.value = new_v
        else:
            self.key = torch.cat([self.key, new_k], dim=-2)
            self.value = torch.cat([self.value, new_v], dim=-2)
        return self.key, self.value

    def reset(self) -> None:
        self.key = None
        self.value = None
