"""
vLLM Inference Engine Wrapper

Provides a unified interface for running inference with
the Nemotron model using vLLM for high-throughput generation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class VLLMEngine:
    """vLLM-based inference engine for Nemotron model.

    Attributes:
        config: Competition parameters.
        engine: vLLM LLM engine instance.
    """

    def __init__(
        self,
        config_path: str = "configs/competition_params.json",
        adapter_path: Optional[str] = None,
    ) -> None:
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.adapter_path = adapter_path
        self.engine = None

    def initialize(self) -> None:
        """Initialize the vLLM engine with competition parameters.

        Loads the base model with optional LoRA adapter.
        """
        from vllm import LLM, SamplingParams

        engine_kwargs: Dict[str, Any] = {
            "model": self.config["model_name"],
            "max_model_len": self.config["max_model_len"],
            "gpu_memory_utilization": self.config["gpu_memory_utilization"],
            "max_num_seqs": self.config["max_num_seqs"],
            "trust_remote_code": True,
            "dtype": "bfloat16",
        }

        if self.adapter_path:
            engine_kwargs["enable_lora"] = True
            engine_kwargs["max_lora_rank"] = self.config["max_lora_rank"]

        print(f"Initializing vLLM engine: {self.config['model_name']}")
        self.engine = LLM(**engine_kwargs)
        print("vLLM engine initialized successfully.")

    def generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> List[str]:
        """Generate completions for a batch of prompts.

        Args:
            prompts: List of input prompts.
            max_tokens: Override max tokens (default from config).
            temperature: Override temperature (default from config).
            top_p: Override top_p (default from config).

        Returns:
            List of generated completion strings.
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens or self.config["max_tokens"],
            temperature=temperature if temperature is not None else self.config["temperature"],
            top_p=top_p or self.config["top_p"],
        )

        outputs = self.engine.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def generate_single(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion for a single prompt.

        Args:
            prompt: Input prompt.
            max_tokens: Override max tokens.

        Returns:
            Generated completion string.
        """
        results = self.generate([prompt], max_tokens=max_tokens)
        return results[0] if results else ""

    def get_engine_info(self) -> Dict[str, Any]:
        """Return engine configuration summary.

        Returns:
            Dict with engine configuration details.
        """
        return {
            "model": self.config["model_name"],
            "max_model_len": self.config["max_model_len"],
            "max_tokens": self.config["max_tokens"],
            "temperature": self.config["temperature"],
            "top_p": self.config["top_p"],
            "gpu_memory_utilization": self.config["gpu_memory_utilization"],
            "adapter_loaded": self.adapter_path is not None,
            "initialized": self.engine is not None,
        }
