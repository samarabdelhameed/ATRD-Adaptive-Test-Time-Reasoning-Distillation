"""Unit tests for SFT training execution logic.

Verifies Nemotron formatting templates, dataset preparation, and early stopping
plateau detection logic under mocked dependencies.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Mock out torch and other training dependencies
mock_torch = MagicMock()
sys.modules["torch"] = mock_torch

mock_transformers = MagicMock()
sys.modules["transformers"] = mock_transformers

mock_peft = MagicMock()
sys.modules["peft"] = mock_peft

mock_trl = MagicMock()
sys.modules["trl"] = mock_trl

mock_datasets = MagicMock()
sys.modules["datasets"] = mock_datasets

from src.training.sft_trainer import SFTTrainerWrapper, should_early_stop


class TestSFTTraining(unittest.TestCase):
    """Test suite for SFT training execution components."""

    def test_should_early_stop(self):
        """Verify that early stopping logic correctly flags plateaus."""
        # Not enough history
        self.assertFalse(should_early_stop([1.5, 1.4]))

        # No plateau (decreasing steadily)
        self.assertFalse(should_early_stop([1.5, 1.4, 1.3, 1.1]))

        # Plateau detected (fluctuations within 0.01 tolerance)
        self.assertTrue(should_early_stop([1.5, 1.25, 1.24, 1.25]))
        self.assertTrue(should_early_stop([0.5, 0.40, 0.402, 0.399, 0.401]))

    def test_nemotron_formatting(self):
        """Verify that Nemotron question/thinking/answer layout is correct."""
        wrapper = SFTTrainerWrapper(model=None, tokenizer=None)
        
        # Test synthetic format
        example = {
            "question": "What is 2+2?",
            "thinking_trace": "<<thinking>>\nCalculate 2+2\n</thinking>>",
            "answer": "4"
        }
        formatted = wrapper._format_example(example)
        self.assertIn("<|begin_of_text|>What is 2+2?\n\n<<thinking>>\nCalculate 2+2\n</thinking>>\n\nAnswer: 4<|end_of_text|>", formatted)

        # Test prompt/completion fallback format
        example_fallback = {
            "prompt": "Q: 3+3?",
            "completion": "A: 6"
        }
        formatted_fallback = wrapper._format_example(example_fallback)
        self.assertIn("<|begin_of_text|>Q: 3+3?\n\nA: 6<|end_of_text|>", formatted_fallback)


if __name__ == "__main__":
    unittest.main()
