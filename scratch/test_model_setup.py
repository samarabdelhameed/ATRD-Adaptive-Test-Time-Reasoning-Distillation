"""Unit tests for QLoRA model setup components with mocked dependencies.

Verifies correctness of ModelLoader and create_lora_config logic, including
rank constraints and hardware optimization configurations without requiring
GPU libraries locally.
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# 1. Mock GPU-dependent modules before importing our source code
mock_torch = MagicMock()
mock_torch.bfloat16 = "bfloat16"
mock_torch.cuda.is_available.return_value = True
mock_torch.cuda.current_device.return_value = 0

mock_device_props = MagicMock()
mock_device_props.name = "RTX PRO 6000 Blackwell"
mock_device_props.major = 10
mock_device_props.minor = 0
mock_torch.cuda.get_device_properties.return_value = mock_device_props

sys.modules["torch"] = mock_torch

# Mock transformers
mock_transformers = MagicMock()
mock_bnb_config = MagicMock()
mock_transformers.BitsAndBytesConfig = mock_bnb_config
sys.modules["transformers"] = mock_transformers

# Mock peft
mock_peft = MagicMock()
class MockTaskType:
    CAUSAL_LM = "CAUSAL_LM"
mock_peft.TaskType = MockTaskType

# Simple mock for LoraConfig to behave like a class
class MockLoraConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
mock_peft.LoraConfig = MockLoraConfig

sys.modules["peft"] = mock_peft

# 2. Now import our actual modules under test
from src.models.loader import setup_blackwell_optimizations, ModelLoader
from src.models.lora_config import create_lora_config, validate_lora_config, validate_adapter


class TestModelSetup(unittest.TestCase):
    """Test suite for QLoRA model setup components."""

    def setUp(self):
        """Prepare mock configuration files."""
        self.config_dir = Path("configs")
        self.config_dir.mkdir(exist_ok=True)
        self.params_file = self.config_dir / "competition_params.json"
        
        # Ensure competition params exist without overwriting
        if not self.params_file.exists():
            with open(self.params_file, "w") as f:
                json.dump({
                    "model_name": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
                    "max_lora_rank": 32,
                    "max_tokens": 7680,
                    "max_model_len": 8192,
                    "inference_engine": "vllm",
                    "gpu_memory_utilization": 0.85
                }, f)

        # Base LoRA config file
        self.lora_file = self.config_dir / "base_lora.json"
        with open(self.lora_file, "w") as f:
            json.dump({
                "r": 32,
                "lora_alpha": 64,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            }, f)

    def test_lora_config_validation(self):
        """Verify that invalid LoRA configurations are correctly caught."""
        # Valid config should pass
        valid = {"r": 32, "lora_alpha": 64, "lora_dropout": 0.1}
        try:
            validate_lora_config(valid)
        except AssertionError as e:
            self.fail(f"validate_lora_config failed on a valid config: {e}")

        # Invalid rank (> 32)
        invalid_rank = {"r": 64, "lora_alpha": 64, "lora_dropout": 0.1}
        with self.assertRaises(AssertionError):
            validate_lora_config(invalid_rank)

        # Invalid alpha (< rank)
        invalid_alpha = {"r": 32, "lora_alpha": 16, "lora_dropout": 0.1}
        with self.assertRaises(AssertionError):
            validate_lora_config(invalid_alpha)

        # Invalid dropout (>= 0.5)
        invalid_dropout = {"r": 32, "lora_alpha": 64, "lora_dropout": 0.6}
        with self.assertRaises(AssertionError):
            validate_lora_config(invalid_dropout)

    def test_create_lora_config(self):
        """Verify LoraConfig object creation and overrides."""
        config = create_lora_config(override={"r": 16, "lora_alpha": 32})
        self.assertEqual(config.r, 16)
        self.assertEqual(config.lora_alpha, 32)

        # Rank override exceeding 32 should raise ValueError
        with self.assertRaises(ValueError):
            create_lora_config(override={"r": 64})

    def test_tokenizer_padding(self):
        """Test that pad token settings are correctly configured."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.eos_token_id = 2
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        
        loader = ModelLoader()
        tokenizer = loader.load_tokenizer()
        
        # Verify pad token overrides
        self.assertEqual(tokenizer.pad_token, "</s>")
        self.assertEqual(tokenizer.padding_side, "right")

    def test_blackwell_optimizations(self):
        """Ensure Blackwell optimization function runs without runtime errors."""
        setup_blackwell_optimizations()
        mock_torch.cuda.set_per_process_memory_fraction.assert_called_with(0.85)


if __name__ == "__main__":
    unittest.main()
