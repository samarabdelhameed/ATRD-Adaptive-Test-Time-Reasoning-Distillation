"""Evaluation metrics and ablation study utilities."""
__version__ = "1.0.0"
__author__ = "Samar Abdelhameed Ahmed"
__license__ = "MIT"

from .metric import (
    answers_equivalent,
    compute_accuracy,
    evaluate_submission,
    extract_boxed_answer,
    load_benchmark_problems,
    normalize_answer,
    parse_numeric_value,
)
from .ablation import AblationRunner

__all__ = [
    "answers_equivalent",
    "compute_accuracy",
    "evaluate_submission",
    "extract_boxed_answer",
    "load_benchmark_problems",
    "normalize_answer",
    "parse_numeric_value",
    "AblationRunner",
]
