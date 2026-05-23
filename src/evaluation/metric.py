"""
Evaluation Metrics

Competition-aligned evaluation metrics for measuring
reasoning accuracy with numerical tolerance matching.
"""

import json
import logging
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

_DEFAULT_TOLERANCE: Optional[float] = None

# LaTeX \frac{a}{b} (simple forms)
_FRAC_LATEX = re.compile(
    r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
    re.IGNORECASE,
)
# Plain a/b fractions (not dates like 2024/01)
_PLAIN_FRAC = re.compile(
    r"^([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)$"
)


def _get_tolerance(override: Optional[float] = None) -> float:
    if override is not None:
        return override
    global _DEFAULT_TOLERANCE
    if _DEFAULT_TOLERANCE is not None:
        return _DEFAULT_TOLERANCE
    cfg_path = Path("configs/competition_params.json")
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        _DEFAULT_TOLERANCE = cfg.get("numerical_tolerance", 0.01)
    else:
        _DEFAULT_TOLERANCE = 0.01
    return _DEFAULT_TOLERANCE


def normalize_answer(text: str) -> str:
    """Normalize answer strings for comparison (whitespace, outer parens)."""
    if text is None:
        return ""
    s = str(text).strip()
    # Strip outer \boxed{} if present
    boxed = extract_boxed_answer(s)
    if boxed:
        s = boxed
    # Remove common LaTeX wrappers
    s = s.replace("$", "").strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    return s


def parse_numeric_value(text: str) -> Optional[float]:
    """Parse int, float, plain fraction (1/4), or simple \\frac{a}{b} to float."""
    s = normalize_answer(text)
    if not s:
        return None

    try:
        return float(s)
    except (ValueError, TypeError):
        pass

    m = _PLAIN_FRAC.match(s.replace(" ", ""))
    if m:
        try:
            return float(Fraction(int(float(m.group(1))), int(float(m.group(2)))))
        except (ValueError, ZeroDivisionError):
            pass

    m = _FRAC_LATEX.search(s)
    if m:
        try:
            num = float(Fraction(m.group(1).strip()))
            den = float(Fraction(m.group(2).strip()))
            if den != 0:
                return num / den
        except (ValueError, ZeroDivisionError):
            pass

    return None


def answers_equivalent(
    predicted: str,
    expected: str,
    tolerance: Optional[float] = None,
) -> bool:
    """Check if predicted answer matches expected (competition-aligned).

    Supports exact string match, numeric relative tolerance, and fractions.
    """
    tolerance = _get_tolerance(tolerance)
    pred = normalize_answer(predicted)
    exp = normalize_answer(expected)

    if not pred and not exp:
        return True
    if pred == exp:
        return True

    pred_num = parse_numeric_value(pred)
    exp_num = parse_numeric_value(exp)
    if pred_num is not None and exp_num is not None:
        denom = max(abs(exp_num), 1e-9)
        return abs(pred_num - exp_num) / denom <= tolerance

    return pred.lower() == exp.lower()


def _check_answer(predicted: str, expected: str, tolerance: float = 0.01) -> bool:
    """Internal alias used by compute_accuracy."""
    return answers_equivalent(predicted, expected, tolerance)


def compute_accuracy(
    predictions: List[str],
    ground_truths: List[str],
    tolerance: Optional[float] = None,
) -> Dict[str, Union[float, int, List[Dict[str, Any]]]]:
    """Compute accuracy metrics for model predictions."""
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths"
        )

    tolerance = _get_tolerance(tolerance)

    correct = 0
    results_detail = []

    for pred, gt in zip(predictions, ground_truths):
        is_correct = _check_answer(pred, gt, tolerance)
        if is_correct:
            correct += 1
        results_detail.append({
            "predicted": pred,
            "expected": gt,
            "correct": is_correct,
        })

    total = len(predictions)
    return {
        "accuracy": correct / max(total, 1),
        "correct_count": correct,
        "total_count": total,
        "details": results_detail,
    }


def evaluate_submission(
    responses: List[Dict[str, Any]],
    problems: List[Dict[str, Any]],
    tolerance: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate a full submission against problem set."""
    tolerance = _get_tolerance(tolerance)
    predictions = [extract_boxed_answer(r.get("response", "")) for r in responses]
    ground_truths = [p.get("answer", "") for p in problems]

    accuracy_report = compute_accuracy(predictions, ground_truths, tolerance)

    categories: Dict[str, List[bool]] = {}
    for prob, detail in zip(problems, accuracy_report["details"]):
        cat = prob.get("category", "unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(detail["correct"])

    category_accuracy = {
        cat: sum(results) / max(len(results), 1)
        for cat, results in categories.items()
    }

    return {
        "overall_accuracy": accuracy_report["accuracy"],
        "correct_count": accuracy_report["correct_count"],
        "total_count": accuracy_report["total_count"],
        "category_accuracy": category_accuracy,
    }


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} format in model output."""
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else ""


def load_benchmark_problems(
    kaggle_dir: str = "/kaggle/input/nemotron-benchmark",
    local_jsonl: str = "data/public_test.jsonl",
    local_json: str = "data/benchmark.json",
) -> List[Dict[str, Any]]:
    """Load benchmark problems from Kaggle input or local development paths.

    Raises:
        FileNotFoundError: If no benchmark file exists.
    """
    candidates = [
        Path(kaggle_dir) / "benchmark.json",
        Path(local_jsonl),
        Path(local_json),
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            problems: List[Dict[str, Any]] = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        problems.append(json.loads(line))
            if problems:
                logger.info("Loaded %d problems from %s", len(problems), path)
                return problems
        else:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                logger.info("Loaded %d problems from %s", len(data), path)
                return data
            if isinstance(data, dict) and "problems" in data:
                problems = data["problems"]
                logger.info("Loaded %d problems from %s", len(problems), path)
                return problems

    raise FileNotFoundError(
        "Benchmark not found. Provide one of:\n"
        f"  - {kaggle_dir}/benchmark.json (Kaggle)\n"
        f"  - {local_jsonl}\n"
        f"  - {local_json}"
    )
