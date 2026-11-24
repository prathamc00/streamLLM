import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

import torch
from torch import nn

from streamllm.engine import StreamLLMEngine
from streamllm.pinned_host import PinnedHostWeightRegistry
from streamllm.scratchpad import LayerBufferSlot


class LayerExecutionTests(unittest.TestCase):
    def test_resident_and_streamed_layer_order(self):
        registry = PinnedHostWeightRegistry()
        values = [torch.tensor([[1., 2.], [float(i), 1.]]) for i in range(1, 6)]
        for index, value in enumerate(values):
            registry.register_layer(index, {"weight": value}, pin=False)
        engine = StreamLLMEngine(nn.Linear(2, 2, bias=False), registry, 5, torch.device("cpu"))
        second = LayerBufferSlot(1, {"weight": (torch.Size([2, 2]), torch.float32)}, engine.device)
        pool = engine.scratchpad
        pool.slots.append(second)
        pool.num_slots = 2
        pool.prefetch_slot_idx = 1
        engine.slot_modules.append(engine._module_for_weights(second.tensors))
        engine.double_buffering = True
        engine.transfer_engine.compute_stream = Mock()
        engine.transfer_engine.transfer_stream = Mock()
        inputs = torch.tensor([[1., 3.]])
        expected = inputs
        for weight in values:
            expected = expected @ weight.T

        with patch("torch.cuda.current_stream"), patch("torch.cuda.stream", return_value=nullcontext()):
            for count in (0, 2, 5):
                engine.resident_layers = [engine._module_for_weights({"weight": w}) for w in values[:count]]
                for _ in range(2):
                    torch.testing.assert_close(engine.forward_streamed(inputs), expected)
