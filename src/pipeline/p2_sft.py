"""Phase 2: supervised fine-tuning."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.data.dataset_sources import load_training_jsonl
from src.models.loader import ModelLoader
from src.models.lora_config import create_lora_config
from src.pipeline._config import load_pipeline_config
from src.training.sft_trainer import SFTTrainerWrapper

logger = logging.getLogger(__name__)


def run_sft_training(
    config_path: str = "configs/pipeline.json",
    dataset_path: str = "data/final_train_dataset.jsonl",
    lora_config_path: str = "configs/base_lora.json",
) -> Path:
    """Run QLoRA SFT and save adapter to checkpoints/sft/final_adapter/."""
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("PyTorch required for SFT") from e

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required for SFT on Nemotron-30B. Run on Kaggle P100 or G4 VM."
        )

    cfg = load_pipeline_config(config_path)
    examples = load_training_jsonl(dataset_path)
    if len(examples) < 100:
        raise ValueError(
            f"Training set too small ({len(examples)} rows). "
            "Run `python run_pipeline.py --phase p1_data` first."
        )

    loader = ModelLoader("configs/competition_params.json")
    loader.setup_blackwell_optimizations()
    tokenizer = loader.load_tokenizer()
    model = loader.load_model(quantize=True)
    loader.enable_gradient_checkpointing(model)

    lora_cfg = create_lora_config(lora_config_path)
    model = __import__("peft").get_peft_model(model, lora_cfg)

    out_dir = cfg.get("sft_output_dir", "checkpoints/sft")
    trainer = SFTTrainerWrapper(model, tokenizer, output_dir=out_dir)

    split = int(len(examples) * 0.95)
    train_ex = examples[:split]
    eval_ex = examples[split:] if split < len(examples) else None

    train_ds = trainer.prepare_dataset(train_ex)
    eval_ds = trainer.prepare_dataset(eval_ex) if eval_ex else None

    trainer.train(train_dataset=train_ds, eval_dataset=eval_ds)
    adapter_path = trainer.save_adapter()

    report = {
        "status": "completed",
        "train_examples": len(train_ex),
        "eval_examples": len(eval_ex) if eval_ex else 0,
        "adapter_path": str(adapter_path),
    }
    log_path = Path("logs/p2_sft_eval.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("SFT adapter saved: %s", adapter_path)
    return Path(adapter_path)
