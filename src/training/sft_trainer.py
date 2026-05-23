"""SFT Trainer Wrapper for supervised fine-tuning on Nemotron-3-Nano-30B.

Uses TRL's SFTTrainer with LoRA adapter. Teaches the model
structured reasoning format using the curated Phase 1 dataset.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset


def format_sft_example(example: dict) -> str:
    """Format with Nemotron reasoning parser tokens.

    Args:
        example: Dict with 'question', 'thinking_trace', and 'answer' keys.

    Returns:
        Formatted text string.
    """
    prompt = example["question"]

    if "thinking_trace" in example:
        thinking = example["thinking_trace"]
    else:
        thinking = "<<thinking>>\n[Reason step by step]\n</thinking>>"

    answer = example["answer"]
    return f"{prompt}\n\n{thinking}\n\nAnswer: {answer}"


def should_early_stop(loss_history: List[float], patience: int = 3) -> bool:
    """Stop if validation loss plateaus for `patience` evaluations.

    Args:
        loss_history: List of validation loss values.
        patience: Number of evaluations to check for plateau.

    Returns:
        True if loss has plateaued (max - min < 0.01 over patience window).
    """
    if len(loss_history) < patience + 1:
        return False
    recent = loss_history[-patience:]
    return max(recent) - min(recent) < 0.01


def test_generation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 512,
) -> str:
    """Generate a test completion from a prompt.

    Args:
        model: The model to generate from.
        tokenizer: The model tokenizer.
        prompt: Input prompt text.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Decoded generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


class SFTTrainerWrapper:
    """Wrapper for TRL SFTTrainer with competition-specific defaults.

    Attributes:
        model: Base model with LoRA adapter.
        tokenizer: Model tokenizer.
        config: Training configuration.
        output_dir: Directory for checkpoints and results.
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
        max_length: int = 4096,
    ) -> Dataset:
        """Prepare dataset for SFT training using format_sft_example.

        Args:
            examples: List of dicts with at least 'question' and 'answer' keys.
            max_length: Maximum sequence length (default 4096 per spec).

        Returns:
            HuggingFace Dataset ready for training.
        """
        formatted = []
        for ex in examples:
            text = format_sft_example(ex)
            formatted.append({"text": text})

        dataset = Dataset.from_list(formatted)
        print(f"Prepared {len(dataset)} examples for SFT (max_seq_length={max_length})")
        return dataset

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        max_seq_length: int = 4096,
        warmup_steps: int = 100,
    ) -> Any:
        """Run SFT training loop with competition-specified hyperparameters.

        Args:
            train_dataset: Training dataset.
            eval_dataset: Optional evaluation dataset.
            num_epochs: Number of training epochs (default 3).
            learning_rate: Learning rate (default 2e-4).
            batch_size: Per-device batch size (default 1).
            gradient_accumulation_steps: Gradient accumulation steps (default 8).
            max_seq_length: Max sequence length (default 4096).
            warmup_steps: Learning rate warmup steps (default 100).

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
            gradient_checkpointing=True,
            logging_steps=10,
            save_steps=50,
            eval_steps=50 if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            optim="adamw_torch_fused",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            max_seq_length=max_seq_length,
        )

        print("Starting SFT training...")
        print(
            f"  lr={learning_rate}, epochs={num_epochs}, "
            f"batch={batch_size}, grad_acc={gradient_accumulation_steps}"
        )
        result = trainer.train()
        print(f"SFT training complete. Loss: {result.training_loss:.4f}")

        self._save_results(result)
        return result

    def _save_results(self, result: Any) -> None:
        """Save training results to logs/sft_results.json."""
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "training_loss": float(result.training_loss),
            "output_dir": str(self.output_dir),
        }
        path = logs_dir / "sft_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {path}")

    def save_adapter(self, path: Optional[str] = None) -> Path:
        """Save the LoRA adapter weights.

        Args:
            path: Output path (default: output_dir/final_adapter).

        Returns:
            Path to saved adapter.
        """
        save_path = Path(path) if path else self.output_dir / "final_adapter"
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        print(f"Adapter saved to {save_path}")
        return save_path
