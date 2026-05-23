"""
GRPO Trainer Wrapper

Group Relative Policy Optimization training pipeline
with PRM-guided reward signals for reasoning improvement.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch


class GRPOTrainerWrapper:
    """Wrapper for TRL GRPOTrainer with PRM-guided rewards.

    Attributes:
        model: Base model with LoRA adapter (post-SFT).
        tokenizer: Model tokenizer.
        grpo_config: GRPO-specific training parameters.
        competition_config: Competition parameters.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        grpo_config_path: str = "configs/base_grpo.json",
        competition_config_path: str = "configs/competition_params.json",
        output_dir: str = "checkpoints/grpo",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(grpo_config_path, "r") as f:
            self.grpo_config = json.load(f)

        with open(competition_config_path, "r") as f:
            self.competition_config = json.load(f)

    def create_reward_function(
        self,
        answer_key: str = "answer",
        tolerance: Optional[float] = None,
    ) -> Callable:
        """Create a reward function for GRPO training.

        The reward function scores model outputs based on:
        1. Answer correctness (primary signal)
        2. Reasoning structure quality (secondary signal)
        3. Format compliance (\\boxed{} format)

        Args:
            answer_key: Key for ground truth answer in problem dict.
            tolerance: Numerical tolerance for answer matching.

        Returns:
            Callable reward function.
        """
        if tolerance is None:
            tolerance = self.competition_config.get("numerical_tolerance", 0.01)

        def reward_fn(
            completions: List[str],
            ground_truth: Optional[str] = None,
        ) -> List[float]:
            """Score a batch of completions.

            Args:
                completions: List of model-generated completions.
                ground_truth: Expected answer.

            Returns:
                List of reward scores.
            """
            rewards = []
            for completion in completions:
                score = 0.0

                # Format compliance reward
                if "\\boxed{" in completion:
                    score += 0.2

                # Answer correctness reward
                if ground_truth:
                    extracted = _extract_boxed_answer(completion)
                    if _check_answer(extracted, ground_truth, tolerance):
                        score += 0.8

                rewards.append(score)
            return rewards

        return reward_fn

    def train(
        self,
        train_dataset: Any,
        reward_function: Callable,
        eval_dataset: Optional[Any] = None,
    ) -> Any:
        """Run GRPO training loop.

        Args:
            train_dataset: Training dataset with prompts.
            reward_function: Reward scoring function.
            eval_dataset: Optional evaluation dataset.

        Returns:
            Training result object.
        """
        from trl import GRPOConfig, GRPOTrainer

        config = GRPOConfig(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=self.grpo_config["batch_size"],
            gradient_accumulation_steps=self.grpo_config["gradient_accumulation_steps"],
            learning_rate=self.grpo_config["learning_rate"],
            max_grad_norm=self.grpo_config["max_grad_norm"],
            num_train_epochs=self.grpo_config["num_train_epochs"],
            max_steps=self.grpo_config["max_steps"],
            warmup_ratio=self.grpo_config["warmup_ratio"],
            logging_steps=self.grpo_config["logging_steps"],
            save_steps=self.grpo_config["save_steps"],
            bf16=self.grpo_config["bf16"],
            gradient_checkpointing=self.grpo_config["gradient_checkpointing"],
            report_to="none",
        )

        trainer = GRPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            reward_funcs=reward_function,
        )

        print("Starting GRPO training...")
        result = trainer.train()
        print(f"GRPO training complete.")
        return result

    def save_adapter(self, path: Optional[str] = None) -> Path:
        """Save the LoRA adapter weights after GRPO training.

        Args:
            path: Output path.

        Returns:
            Path to saved adapter.
        """
        save_path = Path(path) if path else self.output_dir / "final_adapter"
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        print(f"GRPO adapter saved to {save_path}")
        return save_path


def _extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} format.

    Args:
        text: Model output text.

    Returns:
        Extracted answer string, or empty string if not found.
    """
    import re

    pattern = r"\\boxed\{([^}]*)\}"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def _check_answer(predicted: str, expected: str, tolerance: float = 0.01) -> bool:
    """Check if predicted answer matches expected within tolerance.

    Args:
        predicted: Predicted answer string.
        expected: Expected answer string.
        tolerance: Numerical tolerance.

    Returns:
        True if answers match.
    """
    try:
        return abs(float(predicted) - float(expected)) <= tolerance
    except (ValueError, TypeError):
        return predicted.strip() == expected.strip()
