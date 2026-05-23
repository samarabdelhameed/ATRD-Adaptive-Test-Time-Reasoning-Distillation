"""
SFT Trainer Wrapper

Supervised Fine-Tuning pipeline using TRL's SFTTrainer
with LoRA adapter on Nemotron-3-Nano-30B.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset


class SFTTrainerWrapper:
    """Wrapper for TRL SFTTrainer with competition-specific defaults.

    Attributes:
        model: Base model with LoRA adapter.
        tokenizer: Model tokenizer.
        config: Training configuration.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        output_dir: str = "checkpoints/sft",
        config_path: str = "configs/competition_params.json",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(config_path, "r") as f:
            self.config = json.load(f)

    def prepare_dataset(
        self,
        examples: List[Dict[str, str]],
        max_length: Optional[int] = None,
    ) -> Dataset:
        """Prepare dataset for SFT training.

        Args:
            examples: List of dicts with 'prompt' and 'completion' keys.
            max_length: Maximum sequence length (default from config).

        Returns:
            HuggingFace Dataset ready for training.
        """
        if max_length is None:
            max_length = self.config.get("max_model_len", 8192)

        formatted = []
        for ex in examples:
            text = self._format_example(ex)
            formatted.append({"text": text})

        dataset = Dataset.from_list(formatted)
        print(f"Prepared {len(dataset)} examples for SFT (max_length={max_length})")
        return dataset

    def _format_example(self, example: Dict[str, str]) -> str:
        """Format a single example into the model's expected format.

        Args:
            example: Dict with 'prompt' and 'completion'.

        Returns:
            Formatted text string.
        """
        prompt = example.get("prompt", "")
        completion = example.get("completion", "")
        return f"<|begin_of_text|>{prompt}\n\n{completion}<|end_of_text|>"

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
    ) -> Any:
        """Run SFT training loop.

        Args:
            train_dataset: Training dataset.
            eval_dataset: Optional evaluation dataset.
            num_epochs: Number of training epochs.
            learning_rate: Learning rate.
            batch_size: Per-device batch size.
            gradient_accumulation_steps: Gradient accumulation steps.

        Returns:
            Training result object.
        """
        from transformers import TrainingArguments
        from trl import SFTTrainer

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            logging_steps=10,
            save_steps=50,
            eval_steps=50 if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            max_seq_length=self.config.get("max_model_len", 8192),
        )

        print("Starting SFT training...")
        result = trainer.train()
        print(f"SFT training complete. Loss: {result.training_loss:.4f}")
        return result

    def save_adapter(self, path: Optional[str] = None) -> Path:
        """Save the LoRA adapter weights.

        Args:
            path: Output path (default: output_dir/final_adapter).

        Returns:
            Path to saved adapter.
        """
        save_path = Path(path) if path else self.output_dir / "final_adapter"
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        print(f"Adapter saved to {save_path}")
        return save_path
