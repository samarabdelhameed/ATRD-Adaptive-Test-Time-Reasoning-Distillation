"""Unified dataset loading for Kaggle inputs, local files, and Hugging Face."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from src.evaluation.metric import load_benchmark_problems

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows → %s", len(rows), path)
    return path


def normalize_training_row(row: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
    """Map heterogeneous rows to ATRD training schema."""
    question = (row.get("question") or row.get("problem") or "").strip()
    if not question:
        return None

    thinking = row.get("thinking_trace") or row.get("generated_solution") or row.get("solution") or ""
    thinking = str(thinking).strip()
    if thinking and "<<thinking>>" not in thinking:
        thinking = f"<<thinking>>\n{thinking}\n</thinking>>"
    elif not thinking:
        thinking = "<<thinking>>\n[Reason step by step]\n</thinking>>"

    answer = row.get("answer") or row.get("expected_answer") or row.get("final_answer") or ""
    answer = str(answer).strip()
    if answer and "\\boxed" not in answer:
        answer = f"\\boxed{{{answer}}}"

    out: Dict[str, Any] = {
        "question": question,
        "thinking_trace": thinking,
        "answer": answer,
        "_source": source,
    }
    if row.get("failure_mode_tag"):
        out["failure_mode_tag"] = row["failure_mode_tag"]
    if row.get("id"):
        out["id"] = row["id"]
    return out


def load_openmath_reasoning(
    kaggle_path: str = "/kaggle/input/open-math-reasoning",
    hf_dataset: str = "nvidia/OpenMathReasoning",
    hf_split: str = "cot",
    max_samples: int = 2500,
    cache_path: str = "data/cache/openmath_reasoning.jsonl",
) -> List[Dict[str, Any]]:
    """Load OpenMathReasoning from Kaggle dir, cache, or Hugging Face stream."""
    cache = Path(cache_path)
    if cache.exists() and cache.stat().st_size > 0:
        return [
            normalize_training_row(r, "open_math_reasoning")
            for r in _read_jsonl(cache, max_samples)
            if normalize_training_row(r, "open_math_reasoning")
        ]

    kaggle = Path(kaggle_path)
    if kaggle.exists():
        rows: List[Dict[str, Any]] = []
        for f_path in sorted(kaggle.rglob("*.jsonl")):
            rows.extend(_read_jsonl(f_path))
            if len(rows) >= max_samples:
                break
        normalized = [
            normalize_training_row(r, "open_math_reasoning")
            for r in rows[:max_samples]
        ]
        normalized = [r for r in normalized if r]
        if normalized:
            _write_jsonl(cache, [dict(r, _source="open_math_reasoning") for r in normalized])
            return normalized

    return _load_hf_split(
        hf_dataset, "default", hf_split, max_samples, "open_math_reasoning", cache
    )


def load_open_code_reasoning(
    kaggle_path: str = "/kaggle/input/open-code-reasoning",
    hf_dataset: str = "nvidia/OpenCodeReasoning",
    hf_split: str = "split_0",
    hf_config: Optional[str] = None,
    max_samples: int = 2500,
    cache_path: str = "data/cache/open_code_reasoning.jsonl",
) -> List[Dict[str, Any]]:
    """Load OpenCodeReasoning from Kaggle dir, cache, or Hugging Face stream."""
    cache = Path(cache_path)
    if cache.exists() and cache.stat().st_size > 0:
        return [
            normalize_training_row(r, "open_code_reasoning")
            for r in _read_jsonl(cache, max_samples)
            if normalize_training_row(r, "open_code_reasoning")
        ]

    kaggle = Path(kaggle_path)
    if kaggle.exists():
        rows: List[Dict[str, Any]] = []
        for f_path in sorted(kaggle.rglob("*.jsonl")):
            rows.extend(_read_jsonl(f_path))
            if len(rows) >= max_samples:
                break
        normalized = [
            normalize_training_row(r, "open_code_reasoning")
            for r in rows[:max_samples]
        ]
        normalized = [r for r in normalized if r]
        if normalized:
            _write_jsonl(cache, [dict(r, _source="open_code_reasoning") for r in normalized])
            return normalized

    config = hf_config or "split_0"
    split = hf_split if hf_split in ("split_0", "split_1") else config
    return _load_hf_split(
        hf_dataset, config, split, max_samples, "open_code_reasoning", cache
    )


def _load_hf_split(
    dataset_name: str,
    config_name: str,
    split: str,
    max_samples: int,
    source: str,
    cache_path: Path,
) -> List[Dict[str, Any]]:
    """Stream samples from Hugging Face with caching."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Install datasets: pip install datasets") from e

    logger.info(
        "Streaming %s config=%s split=%s (max=%d)...",
        dataset_name, config_name, split, max_samples,
    )
    try:
        ds = load_dataset(dataset_name, config_name, split=split, streaming=True)
    except Exception as e:
        logger.warning("HF load failed for %s: %s", dataset_name, e)
        return []

    normalized: List[Dict[str, Any]] = []
    for row in ds:
        item = normalize_training_row(dict(row), source)
        if item:
            normalized.append(item)
        if len(normalized) >= max_samples:
            break

    if normalized:
        _write_jsonl(cache_path, normalized)
    logger.info("Loaded %d examples from Hugging Face %s", len(normalized), dataset_name)
    return normalized


def bootstrap_gsm8k_benchmark(
    output_path: str = "data/public_test.jsonl",
    max_samples: int = 100,
) -> List[Dict[str, Any]]:
    """Download GSM8K test split as a local development benchmark."""
    out = Path(output_path)
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets not installed — skipping GSM8K bootstrap")
        return load_benchmark_problems(local_jsonl=str(out)) if out.exists() else []

    logger.info("Bootstrapping benchmark from openai/gsm8k test (%d)...", max_samples)
    ds = load_dataset("openai/gsm8k", "main", split="test", streaming=True)
    problems: List[Dict[str, Any]] = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        question = row["question"]
        # GSM8K answer format: #### 42
        ans_raw = row["answer"].split("####")[-1].strip()
        problems.append({
            "id": f"gsm8k_{i}",
            "question": question,
            "answer": ans_raw,
            "category": "arithmetic",
        })

    _write_jsonl(out, problems)
    return problems


def load_training_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a training JSONL file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Training file not found: {p}")
    return _read_jsonl(p)
