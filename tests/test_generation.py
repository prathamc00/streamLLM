import copy
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from streamllm.auto_model import MockTokenizer, StreamLLMModel


class CachedGenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2)
        self.reference = Qwen2ForCausalLM(config).eval()
        self.model = StreamLLMModel(config, MockTokenizer(), torch.device("cpu"),
                                    hf_model=copy.deepcopy(self.reference))
        self.ids = torch.tensor([[5, 12, 9, 7]])

    @torch.no_grad()
    def test_cached_logits_match_reference(self):
        cache, key = self.model._generation_state(True)
        sequence = self.ids
        offset = 0
        for _ in range(5):
            hidden = self.model._generation_step(sequence[:, offset:], offset, cache, key)
            actual = self.model.lm_head(self.model.norm(hidden))
            expected = self.reference(sequence, use_cache=False).logits[:, offset:]
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
            offset = sequence.size(1)
            sequence = torch.cat([sequence, expected[:, -1].argmax(-1, keepdim=True)], dim=1)

    def test_cached_and_uncached_greedy_generation(self):
        cached = self.model.generate(self.ids, max_new_tokens=5).sequences
        uncached = self.model.generate(self.ids, max_new_tokens=5, use_cache=False).sequences
        self.assertTrue(torch.equal(cached, uncached))
