"""Data generation, filtering, deduplication, and mixing utilities."""
__version__ = "1.0.0"
__author__ = "Samar Abdelhameed Ahmed"
__license__ = "MIT"

from .synthetic_generator import SyntheticGenerator
from .judge_filter import JudgeFilter
from .deduplicator import Deduplicator
from .dataset_mixer import DatasetMixer

from .budget_forcer import (
    estimate_difficulty,
    allocate_budget,
    difficulty_tier,
    get_budget_stats,
    validate_refinement_improvement,
    check_answer,
    generate,
    set_generate_backend,
    reset_generate_backend,
    refine_hard_problem,
    generate_training_data_with_budget,
)

__all__ = [
    "SyntheticGenerator", "JudgeFilter", "Deduplicator", "DatasetMixer",
    "estimate_difficulty", "allocate_budget", "difficulty_tier",
    "get_budget_stats", "validate_refinement_improvement", "check_answer",
    "generate", "set_generate_backend", "reset_generate_backend",
    "refine_hard_problem",
    "generate_training_data_with_budget",
]
