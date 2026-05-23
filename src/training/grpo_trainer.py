import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch


class GRPOTrainerWrapper:
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
        use_prm: bool = True,
        use_log_ratio: bool = False,
        ref_model: Optional[Any] = None,
    ) -> Callable:
        if tolerance is None:
            tolerance = self.competition_config.get("numerical_tolerance", 0.01)

        from src.training.prm import compute_prm_guided_reward

        def reward_fn(
            completions: List[str],
            ground_truth: Optional[str] = None,
        ) -> List[float]:
            rewards = []
            for completion in completions:
                if use_prm:
                    score = compute_prm_guided_reward(
                        completion=completion,
                        ground_truth=ground_truth or "",
                        ref_model=ref_model,
                        current_model=self.model,
                        tokenizer=self.tokenizer,
                        use_log_ratio=use_log_ratio,
                    )
                else:
                    score = 0.0
                    if "\\boxed{" in completion:
                        score += 0.2
                    if "<<thinking>>" in completion and "</thinking>>" in completion:
                        score += 0.2

                    if ground_truth:
                        extracted = _extract_boxed_answer(completion)
                        if _check_answer(extracted, ground_truth, tolerance):
                            score += 0.8

                    if _detect_redundancy(completion):
                        score -= 0.3

                    score = max(-1.0, min(1.0, score))
                rewards.append(score)
            return rewards

        return reward_fn

    def train(
        self,
        train_dataset: Any,
        reward_function: Callable,
        eval_dataset: Optional[Any] = None,
    ) -> Any:
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
        print("GRPO training complete.")
        return result

    def save_adapter(self, path: Optional[str] = None) -> Path:
        save_path = Path(path) if path else self.output_dir / "final_adapter"
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        print(f"GRPO adapter saved to {save_path}")
        return save_path


class KLMonitor:
    def __init__(self, ref_model: Any, threshold: float = 0.05):
        self.ref_model = ref_model
        self.threshold = threshold
        self.history: List[float] = []

    def log_kl(self, current_model: Any, batch: Any) -> float:
        kl = _compute_kl(current_model, self.ref_model, batch)
        self.history.append(kl)

        if kl > self.threshold:
            print(f"WARNING: KL={kl:.4f} exceeds threshold={self.threshold}")
        if kl > 0.1:
            raise RuntimeError(f"KL divergence too high ({kl:.4f}). Stopping training.")

        return kl


def _extract_boxed_answer(text: str) -> str:
    import re

    pattern = r"\\boxed\{([^}]*)\}"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def _check_answer(predicted: str, expected: str, tolerance: float = 0.01) -> bool:
    try:
        return abs(float(predicted) - float(expected)) <= tolerance
    except (ValueError, TypeError):
        return predicted.strip() == expected.strip()


def _detect_redundancy(text: str) -> bool:
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    repeat_count = 0
    for i in range(2, len(lines)):
        if lines[i].strip() == lines[i - 2].strip():
            repeat_count += 1
    return repeat_count >= 2


def _compute_kl(current_model: Any, ref_model: Any, batch: Any) -> float:
    with torch.no_grad():
        curr_logits = current_model(**batch).logits
        ref_logits = ref_model(**batch).logits
    curr_probs = torch.nn.functional.log_softmax(curr_logits, dim=-1)
    ref_probs = torch.nn.functional.softmax(ref_logits, dim=-1)
    kl = torch.sum(ref_probs * (torch.log(ref_probs + 1e-10) - curr_probs), dim=-1).mean().item()
    return kl


def verify_monotonic_reward(reward_history: List[float], window: int = 10) -> bool:
    if len(reward_history) < window * 2:
        return True
    recent = reward_history[-window:]
    early = reward_history[-window * 2 : -window]
    return sum(recent) / len(recent) > sum(early) / len(early)
