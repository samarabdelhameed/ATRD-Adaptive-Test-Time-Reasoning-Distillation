"""Phase 4: evaluation, ablation logging, and submission packaging."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation.ablation import AblationRunner, AblationConfig
from src.evaluation.metric import evaluate_submission, extract_boxed_answer, load_benchmark_problems
from src.pipeline._config import load_pipeline_config

logger = logging.getLogger(__name__)


def run_final_evaluation(
    config_path: str = "configs/pipeline.json",
    adapter_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate adapter on public benchmark and write logs."""
    cfg = load_pipeline_config(config_path)
    adapter_path = adapter_path or cfg.get("submission_adapter_path", "checkpoints/grpo/final_adapter")
    adapter = Path(adapter_path)

    problems = load_benchmark_problems(
        kaggle_dir=cfg["benchmark_kaggle_path"],
        local_jsonl=cfg["benchmark_local_jsonl"],
    )

    report: Dict[str, Any] = {
        "status": "pending",
        "adapter_path": str(adapter),
        "problem_count": len(problems),
    }

    if not (adapter / "adapter_config.json").exists():
        report["status"] = "skipped"
        report["reason"] = "No trained adapter — run p2_sft and p3_grpo on GPU"
        _save_eval_logs(report)
        return report

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no cuda")

        from src.models.loader import ModelLoader
        from peft import PeftModel

        loader = ModelLoader("configs/competition_params.json")
        tokenizer = loader.load_tokenizer()
        base = loader.load_model(quantize=True)
        model = PeftModel.from_pretrained(base, str(adapter))
        model.eval()

        responses = []
        for p in problems:
            prompt = f"Question: {p['question']}\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            responses.append({"response": text})

        eval_report = evaluate_submission(responses, problems)
        report = {
            "status": "completed",
            "adapter_path": str(adapter),
            **eval_report,
        }
    except Exception as e:
        report["status"] = "error"
        report["error"] = str(e)
        logger.error("Final eval failed: %s", e)

    _save_eval_logs(report)
    return report


def _save_eval_logs(report: Dict[str, Any]) -> None:
    for path in (Path("logs/p4_final_eval.json"), Path("data/p4_final_eval.json")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_ablation_from_logs() -> Path:
    """Build ablation_results.json from available phase logs (real data only)."""
    runner = AblationRunner()
    stages: List[Dict[str, Any]] = []

    baseline_log = Path("data/baseline_results.json")
    if baseline_log.exists():
        b = json.loads(baseline_log.read_text())
        if b.get("status") == "completed":
            stages.append({
                "name": "baseline",
                "score": b["overall_accuracy"],
                "delta": None,
                "config": {},
                "status": "completed",
            })

    # Placeholders filled when training completes
    for name, log_file, cfg_key in [
        ("sft_only", "logs/sft_results.json", {"lora_rank": 32}),
        ("sft_grpo", "logs/p3_grpo_eval.json", {"group_size": 8}),
    ]:
        p = Path(log_file)
        if p.exists():
            stages.append({
                "name": name,
                "score": None,
                "delta": None,
                "config": cfg_key,
                "status": "logged",
            })

    final = Path("logs/p4_final_eval.json")
    if final.exists():
        f = json.loads(final.read_text())
        if f.get("status") == "completed":
            base_score = stages[0]["score"] if stages else 0.0
            score = f["overall_accuracy"]
            stages.append({
                "name": "full_pipeline",
                "score": score,
                "delta": score - base_score if base_score else None,
                "config": {"budget_forcing": True},
                "status": "completed",
            })

    if len(stages) < 2:
        out = Path("logs/ablation_results.json")
        out.write_text(json.dumps({
            "status": "pending",
            "message": "Run full GPU pipeline to populate ablations",
            "ablations": stages,
        }, indent=2), encoding="utf-8")
        return out

    # Recompute incremental deltas
    prev = None
    for s in stages:
        if s.get("score") is not None and prev is not None:
            s["delta"] = s["score"] - prev
        elif s.get("score") is not None:
            s["delta"] = None
        if s.get("score") is not None:
            prev = s["score"]

    return runner.save_results(stages)
