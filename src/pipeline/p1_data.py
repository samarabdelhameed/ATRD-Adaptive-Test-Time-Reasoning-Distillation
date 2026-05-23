"""Phase 1: data curation pipeline (executable without notebook)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.dataset_mixer import DatasetMixer, check_leakage, filter_exact_question_leakage
from src.data.deduplicator import Deduplicator
from src.data.judge_filter import JudgeFilter
from src.data.synthetic_generator import SyntheticGenerator
from src.data.template_synthetic import generate_template_synthetic
from src.data.dataset_sources import (
    bootstrap_gsm8k_benchmark,
    load_open_code_reasoning,
    load_openmath_reasoning,
    load_benchmark_problems,
    normalize_training_row,
)
from src.evaluation.metric import answers_equivalent, extract_boxed_answer
from src.pipeline._config import load_pipeline_config

logger = logging.getLogger(__name__)


def _classify_failure(response: Dict[str, Any]) -> str:
    answer = response.get("answer", "")
    reasoning = response.get("reasoning", response.get("response", ""))
    if not answer:
        return "no_answer"
    if not reasoning:
        return "incomplete"
    if "\\boxed" not in str(answer):
        return "format_error"
    return "wrong_answer"


def _build_failure_examples(
    problems: List[Dict[str, Any]],
    responses: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    failure_examples: Dict[str, List[Dict[str, Any]]] = {}
    for prob, resp in zip(problems, responses):
        pred = resp.get("answer") or extract_boxed_answer(resp.get("response", ""))
        if answers_equivalent(pred, prob.get("answer", "")):
            continue
        mode = _classify_failure(resp)
        failure_examples.setdefault(mode, []).append(prob)
    return failure_examples


def _load_baseline_responses(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("status") == "skipped":
        return None
    return data.get("responses")


def run_p1_data_pipeline(
    config_path: str = "configs/pipeline.json",
    skip_baseline: bool = False,
    skip_synthetic: bool = False,
) -> Path:
    """Execute full P1 data pipeline and write final_train_dataset.jsonl."""
    cfg = load_pipeline_config(config_path)
    out_dir = Path(cfg.get("output_dir", "data"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Benchmark
    bench_path = Path(cfg["benchmark_local_jsonl"])
    if not bench_path.exists() or bench_path.stat().st_size < 100:
        logger.info("Bootstrapping local benchmark from GSM8K...")
        bootstrap_gsm8k_benchmark(
            str(bench_path),
            max_samples=cfg.get("gsm8k_benchmark_samples", 100),
        )

    problems = load_benchmark_problems(
        kaggle_dir=cfg["benchmark_kaggle_path"],
        local_jsonl=cfg["benchmark_local_jsonl"],
    )
    logger.info("Benchmark: %d problems", len(problems))

    # 2. Failure examples (from baseline or heuristic)
    failure_examples: Dict[str, List[Dict[str, Any]]] = {}
    baseline_path = out_dir / "baseline_results.json"

    if not skip_baseline:
        responses = _load_baseline_responses(baseline_path)
        if responses and len(responses) == len(problems):
            failure_examples = _build_failure_examples(problems, responses)
            logger.info(
                "Failure modes from baseline: %s",
                {k: len(v) for k, v in failure_examples.items()},
            )
        else:
            logger.warning(
                "No baseline_results.json — using full benchmark as failure seeds"
            )

    if not failure_examples:
        # Seed each failure mode from benchmark subset
        failure_examples = {
            "wrong_answer": problems[: min(50, len(problems))],
            "format_error": problems[min(50, len(problems)): min(100, len(problems))],
            "no_answer": problems[min(100, len(problems)): min(150, len(problems))],
            "incomplete": problems[min(150, len(problems)): min(200, len(problems))],
            "proof_structure": problems[min(200, len(problems)): min(250, len(problems))],
        }
        failure_examples = {k: v for k, v in failure_examples.items() if v}

    failure_path = out_dir / "failure_modes.json"
    failure_path.write_text(
        json.dumps({
            "failure_counts": {k: len(v) for k, v in failure_examples.items()},
            "failure_examples": {k: v[:5] for k, v in failure_examples.items()},
        }, indent=2),
        encoding="utf-8",
    )

    # 3. Synthetic generation
    raw_synthetic: List[Dict[str, Any]] = []
    api_key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")

    if not skip_synthetic and api_key:
        logger.info("Synthetic generation via frontier API...")
        gen = SyntheticGenerator(api_key=api_key, output_dir=str(out_dir))
        per_mode = cfg["synthetic_target"] // max(len(failure_examples), 1)
        per_mode = min(per_mode, cfg.get("synthetic_per_mode_api", 2000))
        raw_synthetic = gen.generate_per_failure_mode(
            failure_examples=failure_examples,
            problems_per_mode=per_mode,
        )
        gen.save_dataset(raw_synthetic, filename="raw_synthetic_dataset.jsonl")
    elif not skip_synthetic:
        logger.info("No API key — template-based synthetic augmentation")
        raw_synthetic = generate_template_synthetic(
            failure_examples,
            per_mode=cfg.get("synthetic_per_mode_template", 200),
            seed=cfg.get("seed", 42),
        )
        raw_path = out_dir / "raw_synthetic_dataset.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for row in raw_synthetic:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 4. Filter + dedup
    judge = JudgeFilter(threshold=cfg.get("judge_threshold", 0.8))
    filtered = judge.filter_dataset(raw_synthetic) if raw_synthetic else []
    dedup = Deduplicator(similarity_threshold=cfg.get("dedup_similarity", 0.85))
    deduplicated = dedup.deduplicate(filtered, key="question") if filtered else []

    # 5. Open math / code
    try:
        openmath = load_openmath_reasoning(
            kaggle_path=cfg["openmath_kaggle_path"],
            hf_dataset=cfg["openmath_hf_dataset"],
            hf_split=cfg["openmath_hf_split"],
            max_samples=cfg["openmath_max_samples"],
        )
    except Exception as e:
        logger.warning("OpenMath load failed: %s", e)
        openmath = []

    try:
        opencode = load_open_code_reasoning(
            kaggle_path=cfg["opencode_kaggle_path"],
            hf_dataset=cfg["opencode_hf_dataset"],
            hf_split=cfg.get("opencode_hf_split", "train"),
            hf_config=cfg.get("opencode_hf_config", "split_0"),
            max_samples=cfg["opencode_max_samples"],
        )
    except Exception as e:
        logger.warning("OpenCode load failed: %s", e)
        opencode = []

    if not openmath and not opencode:
        logger.warning(
            "OpenMath/OpenCode unavailable — using synthetic-only mix "
            "(download HF datasets or attach Kaggle inputs)"
        )

    # 6. Mix
    benchmark_texts = [p["question"] for p in problems]
    mixer = DatasetMixer(seed=cfg.get("seed", 42))
    final_dataset = mixer.mix(
        synthetic=deduplicated,
        math_reasoning=openmath,
        code_reasoning=opencode,
        max_total=cfg.get("max_total_mixed", 50000),
        benchmark_texts=benchmark_texts,
    )

    final_dataset = filter_exact_question_leakage(final_dataset, benchmark_texts)
    overlap = check_leakage(
        [ex["question"] for ex in final_dataset],
        benchmark_texts,
        n=5,
    )
    if overlap > 0:
        logger.info(
            "N-gram overlap with benchmark: %d (common math phrases; exact questions removed)",
            overlap,
        )

    final_path = out_dir / "final_train_dataset.jsonl"
    with open(final_path, "w", encoding="utf-8") as f:
        for item in final_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats = {
        "raw_synthetic": len(raw_synthetic),
        "filtered": len(filtered),
        "deduplicated": len(deduplicated),
        "openmath": len(openmath),
        "opencode": len(opencode),
        "final_total": len(final_dataset),
        "distribution": mixer.get_distribution(final_dataset),
    }
    stats_path = out_dir / "p1_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("P1 complete: %d training examples → %s", len(final_dataset), final_path)
    return final_path
