"""Model loader with 4-bit NF4 quantization for Nemotron-3-Nano-30B.

Handles model loading with BitsAndBytes quantization, GPU memory
management, Blackwell-specific optimizations, and gradient checkpointing.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch
except ImportError:
    torch = None


def load_model_with_cleanup(
    model_name: str,
    quantize: bool = True,
    device_map: str = "auto",
    torch_dtype: "Any" = None,
) -> Any:
    """Load model with memory cleanup and GPU usage reporting.

    Args:
        model_name: HuggingFace model identifier.
        quantize: Whether to apply 4-bit quantization.
        device_map: Device mapping strategy.
        torch_dtype: Override torch dtype (default: bf16).

    Returns:
        Loaded model.
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

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

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )

    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory: {allocated:.2f} GB")
    if allocated > 14:
        print("WARNING: Near memory limit. Reduce batch size.")
    return model


def setup_blackwell_optimizations(memory_fraction: float = 0.85) -> None:
    """Configure CUDA for RTX PRO 6000 Blackwell GPU.

    Enables TF32 matmul if compute capability >= 10.x
    and sets memory fraction per competition params.

    Args:
        memory_fraction: GPU memory fraction (0-1). Reads from
            configs/competition_params.json if available.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping Blackwell optimizations")
        return

    config_path = Path("configs/competition_params.json")
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        memory_fraction = cfg.get("gpu_memory_utilization", memory_fraction)

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    print(f"GPU: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")

    if props.major >= 10:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Blackwell TF32 optimizations enabled")

    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    print(f"Memory fraction set to {memory_fraction:.2f}")


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
        torch_dtype: "Any" = None,
    ) -> Any:
        """Load the base model with optional quantization.

        Args:
            quantize: Whether to apply 4-bit quantization (QLoRA).
            device_map: Device mapping strategy.
            torch_dtype: Override torch dtype (default: bf16).

        Returns:
            Loaded model ready for LoRA attachment.
        """
        model = load_model_with_cleanup(
            self.model_name, quantize, device_map, torch_dtype
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

    def enable_gradient_checkpointing(self, model: Any) -> None:
        """Enable gradient checkpointing to trade compute for memory."""
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled — trading compute for memory")

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


def enable_gradient_checkpointing(model: Any) -> None:
    """Enable gradient checkpointing on a model. Module-level convenience wrapper.

    Args:
        model: The model to enable gradient checkpointing on.
    """
    model.gradient_checkpointing_enable()
