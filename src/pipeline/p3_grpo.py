"""Phase 3: GRPO reinforcement learning."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.data.dataset_sources import load_training_jsonl
from src.models.loader import ModelLoader
from src.pipeline._config import load_pipeline_config
from src.training.grpo_trainer import GRPOTrainerWrapper

logger = logging.getLogger(__name__)


def run_grpo_training(
    config_path: str = "configs/pipeline.json",
    sft_adapter_path: str = "checkpoints/sft/final_adapter",
    dataset_path: str = "data/final_train_dataset.jsonl",
) -> Path:
    """Run GRPO from SFT checkpoint."""
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("PyTorch required for GRPO") from e

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for GRPO. Run on Kaggle P100 or G4 VM.")

    sft_path = Path(sft_adapter_path)
    if not (sft_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"SFT adapter not found at {sft_path}. Run p2_sft first."
        )

    cfg = load_pipeline_config(config_path)
    examples = load_training_jsonl(dataset_path)

    loader = ModelLoader("configs/competition_params.json")
    tokenizer = loader.load_tokenizer()

    from peft import PeftModel

    base = loader.load_model(quantize=True)
    model = PeftModel.from_pretrained(base, str(sft_path), is_trainable=True)

    out_dir = cfg.get("grpo_output_dir", "checkpoints/grpo")
    wrapper = GRPOTrainerWrapper(model, tokenizer, output_dir=out_dir)

    # GRPO dataset: prompt + ground truth answer
    grpo_rows = [
        {
            "prompt": ex["question"],
            "answer": ex.get("answer", "").replace("\\boxed{", "").replace("}", ""),
        }
        for ex in examples[: min(2000, len(examples))]
    ]

    try:
        from datasets import Dataset
    except ImportError as e:
        raise ImportError("datasets package required") from e

    train_ds = Dataset.from_list(grpo_rows)
    reward_fn = wrapper.create_reward_function()

    wrapper.train(train_dataset=train_ds, reward_function=reward_fn)

    save_path = Path(out_dir) / "final_adapter"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    report = {"status": "completed", "adapter_path": str(save_path)}
    log_path = Path("logs/p3_grpo_eval.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("GRPO adapter saved: %s", save_path)
    return save_path
