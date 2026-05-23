"""End-to-end ATRD pipeline orchestration."""

from .p1_data import run_p1_data_pipeline
from .baseline import run_baseline_evaluation
from .p2_sft import run_sft_training
from .p3_grpo import run_grpo_training
from .p4_eval import run_final_evaluation

__all__ = [
    "run_p1_data_pipeline",
    "run_baseline_evaluation",
    "run_sft_training",
    "run_grpo_training",
    "run_final_evaluation",
]
