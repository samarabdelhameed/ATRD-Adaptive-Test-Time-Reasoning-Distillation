"""SFT and GRPO training pipelines."""
__version__ = "1.0.0"
__author__ = "Samar Abdelhameed Ahmed"
__license__ = "MIT"

from .sft_trainer import SFTTrainerWrapper
from .grpo_trainer import GRPOTrainerWrapper
from .prm import segment_thinking_trace, heuristic_step_score, compute_prm_guided_reward

__all__ = [
    "SFTTrainerWrapper",
    "GRPOTrainerWrapper",
    "segment_thinking_trace",
    "heuristic_step_score",
    "compute_prm_guided_reward",
]
