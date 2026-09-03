"""
Unit tests verifying the StreamLLM AutoModel user workflow.
"""

from pathlib import Path
import sys
import unittest

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
try:
    from streamllm import AutoModel
except ImportError:
    from src import AutoModel


class TestAutoModelWorkflow(unittest.TestCase):
    def test_automodel_inference_workflow(self):
        """Validates the StreamLLM AutoModel user workflow:
        
            from src import AutoModel
            model = AutoModel.from_pretrained(...)
            input_tokens = model.tokenizer(...)
            generation_output = model.generate(input_tokens['input_ids'].cuda(), ...)
            output = model.tokenizer.decode(generation_output.sequences[0])
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained(
            "mock-llama-model",
            prefetching=True,
            device=device,
        )

        input_text = ["What is the capital of United States?"]
        input_tokens = model.tokenizer(
            input_text,
            return_tensors="pt",
        )

        self.assertIn("input_ids", input_tokens)
        input_ids = input_tokens["input_ids"]
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        generation_output = model.generate(
            input_ids,
            max_new_tokens=10,
            use_cache=True,
            return_dict_in_generate=True,
        )

        self.assertTrue(hasattr(generation_output, "sequences"))
        self.assertEqual(generation_output.sequences.shape[0], 1)
        self.assertGreater(generation_output.sequences.shape[1], input_ids.shape[1])

        output_text = model.tokenizer.decode(generation_output.sequences[0])
        self.assertIsInstance(output_text, str)
        self.assertGreater(len(output_text), 0)


if __name__ == "__main__":
    unittest.main()
