"""
Evaluation Metrics

Competition-aligned evaluation metrics for measuring
reasoning accuracy with numerical tolerance matching.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


def compute_accuracy(
    predictions: List[str],
    ground_truths: List[str],
    tolerance: float = 0.01,
) -> Dict[str, float]:
    """Compute accuracy metrics for model predictions.

    Args:
        predictions: List of model-predicted answers.
        ground_truths: List of ground truth answers.
        tolerance: Numerical tolerance for floating-point answers.

    Returns:
        Dict with accuracy, correct_count, total_count.
    """
    assert len(predictions) == len(ground_truths), (
        f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths"
    )

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
    tolerance: float = 0.01,
) -> Dict[str, Any]:
    """Evaluate a full submission against problem set.

    Args:
        responses: List of model response dicts.
        problems: List of problem dicts with ground truth answers.
        tolerance: Numerical tolerance.

    Returns:
        Evaluation report dict.
    """
    predictions = [extract_boxed_answer(r.get("response", "")) for r in responses]
    ground_truths = [p.get("answer", "") for p in problems]

    accuracy_report = compute_accuracy(predictions, ground_truths, tolerance)

    # Compute per-category breakdown if categories available
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
    """Extract answer from \\boxed{} format in model output.

    Args:
        text: Full model output text.

    Returns:
        Extracted answer string, or empty string if not found.
    """
    # Handle nested braces
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else ""


def _check_answer(predicted: str, expected: str, tolerance: float = 0.01) -> bool:
    """Check if predicted answer matches expected within tolerance.

    Args:
        predicted: Predicted answer string.
        expected: Expected answer string.
        tolerance: Numerical tolerance for float comparison.

    Returns:
        True if answers match.
    """
    # Clean up whitespace
    predicted = predicted.strip()
    expected = expected.strip()

    # Try numerical comparison first
    try:
        pred_val = float(predicted)
        exp_val = float(expected)
        return abs(pred_val - exp_val) <= tolerance
    except (ValueError, TypeError):
        pass

    # Fall back to exact string match
    return predicted == expected
