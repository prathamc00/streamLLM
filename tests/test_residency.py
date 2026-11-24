import unittest
from unittest.mock import Mock, patch

from streamllm.engine import StreamLLMEngine


class ResidencyTests(unittest.TestCase):
    def test_allocator_caching_and_external_pressure(self):
        mib = 1024**2
        engine = Mock(spec=StreamLLMEngine)
        engine.double_buffering = True
        engine.device = "cuda:0"
        engine.num_layers = 28
        engine.resident_layers = [{} for _ in range(11)]
        engine.scratchpad = Mock()
        engine.scratchpad.active_slot.total_bytes.return_value = 90 * mib
        with patch("torch.cuda.mem_get_info") as info, \
             patch("torch.cuda.memory_allocated", return_value=2000 * mib), \
             patch("torch.cuda.memory_reserved") as reserved:
            for cached in (0, 90, 180, 270, 360):
                info.return_value = ((1100 - cached) * mib, 4096 * mib)
                reserved.return_value = (2000 + cached) * mib
                StreamLLMEngine.prepare_residency(engine, 64 * mib)
                self.assertEqual(len(engine.resident_layers), 11)

            # Another process consumes two layers' worth of device memory.
            info.return_value = ((1100 - 360 - 180) * mib, 4096 * mib)
            StreamLLMEngine.prepare_residency(engine, 64 * mib)
            self.assertEqual(len(engine.resident_layers), 9)
