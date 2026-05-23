"""
Model Loader

Handles loading the Nemotron-3-Nano-30B base model with
quantization and memory optimization for Kaggle T4 GPUs.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class ModelLoader:
    """Load and configure the base Nemotron model.

    Attributes:
        model_name: HuggingFace model identifier.
        config: Competition parameters.
    """

    def __init__(
        self,
        config_path: str = "configs/competition_params.json",
    ) -> None:
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.model_name = self.config["model_name"]

    def load_model(
        self,
        quantize: bool = True,
        device_map: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
    ) -> Any:
        """Load the base model with optional quantization.

        Args:
            quantize: Whether to apply 4-bit quantization (QLoRA).
            device_map: Device mapping strategy.
            torch_dtype: Override torch dtype (default: bf16).

        Returns:
            Loaded model ready for LoRA attachment.
        """
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": True,
        }

        if quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config

        print(f"Loading model: {self.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        print(f"Model loaded. Parameters: {model.num_parameters():,}")
        return model

    def load_tokenizer(self) -> Any:
        """Load the tokenizer for the base model.

        Returns:
            Configured tokenizer with proper padding settings.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
        return tokenizer

    def get_model_info(self) -> Dict[str, Any]:
        """Return model configuration summary.

        Returns:
            Dict with model name, max tokens, and inference settings.
        """
        return {
            "model_name": self.model_name,
            "max_tokens": self.config["max_tokens"],
            "max_model_len": self.config["max_model_len"],
            "inference_engine": self.config["inference_engine"],
            "gpu_memory_utilization": self.config["gpu_memory_utilization"],
        }
