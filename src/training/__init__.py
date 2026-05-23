"""SFT and GRPO training pipelines."""
__version__ = "1.0.0"
__author__ = "Samar Abdelhameed Ahmed"
__license__ = "MIT"

try:
    from .sft_trainer import SFTTrainerWrapper
except ImportError:
    SFTTrainerWrapper = None

try:
    from .grpo_trainer import GRPOTrainerWrapper
except ImportError:
    GRPOTrainerWrapper = None

__all__ = ["SFTTrainerWrapper", "GRPOTrainerWrapper"]
