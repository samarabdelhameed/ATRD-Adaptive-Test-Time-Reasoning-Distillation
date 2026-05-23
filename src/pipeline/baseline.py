"""Baseline evaluation on the reasoning benchmark."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation.metric import evaluate_submission, extract_boxed_answer, load_benchmark_problems
from src.pipeline._config import load_pipeline_config

logger = logging.getLogger(__name__)


def _format_prompt(question: str) -> str:
    return (
        f"Question: {question}\n"
        "Provide a complete step-by-step thinking trace inside "
        "<<thinking>>...</thinking>> and the final answer in \\boxed{}.\n"
    )


def run_baseline_evaluation(
    config_path: str = "configs/pipeline.json",
    output_path: str = "data/baseline_results.json",
    max_problems: Optional[int] = None,
) -> Dict[str, Any]:
    """Run Nemotron baseline eval; requires CUDA + loaded model.

    Returns:
        Evaluation report dict. On missing GPU, writes skip manifest and raises.
    """
    cfg = load_pipeline_config(config_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    problems = load_benchmark_problems(
        kaggle_dir=cfg["benchmark_kaggle_path"],
        local_jsonl=cfg["benchmark_local_jsonl"],
    )
    if max_problems:
        problems = problems[:max_problems]

    try:
        import torch
    except ImportError as e:
        raise RuntimeError("PyTorch required for baseline evaluation") from e

    if not torch.cuda.is_available():
        skip = {
            "status": "skipped",
            "reason": "CUDA not available — run on Kaggle/G4 GPU",
            "problem_count": len(problems),
        }
        out.write_text(json.dumps(skip, indent=2), encoding="utf-8")
        raise RuntimeError(
            "CUDA not available. Run baseline on Kaggle (notebook 01 or "
            "`python run_pipeline.py --phase p1_baseline`) with GPU enabled."
        )

    from src.models.loader import ModelLoader

    loader = ModelLoader("configs/competition_params.json")
    tokenizer = loader.load_tokenizer()
    model = loader.load_model(quantize=True)
    loader.enable_gradient_checkpointing(model)

    responses: List[Dict[str, Any]] = []
    max_new = cfg.get("baseline_max_new_tokens", 512)

    for p in problems:
        prompt = _format_prompt(p["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append({
            "problem_id": p.get("id", ""),
            "response": text,
            "answer": extract_boxed_answer(text),
            "reasoning": text,
        })

    report = evaluate_submission(responses, problems)
    payload = {
        "status": "completed",
        "overall_accuracy": report["overall_accuracy"],
        "correct_count": report["correct_count"],
        "total_count": report["total_count"],
        "category_accuracy": report.get("category_accuracy", {}),
        "responses": responses,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logs_out = Path("logs/baseline_results.json")
    logs_out.parent.mkdir(parents=True, exist_ok=True)
    logs_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.info(
        "Baseline accuracy: %.2f%% (%d/%d)",
        report["overall_accuracy"] * 100,
        report["correct_count"],
        report["total_count"],
    )
    return payload
