"""
LoRA Configuration Factory

Creates PEFT LoRA configurations from JSON config files,
enforcing rank <= 32 constraint per competition rules.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from peft import LoraConfig, TaskType
    _HAS_PEFT = True
except ImportError:
    LoraConfig = TaskType = None  # type: ignore
    _HAS_PEFT = False


def validate_lora_config(config: dict):
    """Validate LoRA configuration meets competition constraints.

    Args:
        config: LoRA configuration dict.

    Returns:
        Tuple of (ok: bool, message: str).

    Raises:
        AssertionError: If any constraint is violated.
    """
    if config.get("r", 0) > 32:
        return False, (
            f"LoRA rank {config['r']} exceeds competition maximum of 32. "
            "This would result in disqualification."
        )
    if config.get("lora_alpha", 0) < config.get("r", 0):
        return False, (
            f"Alpha ({config['lora_alpha']}) must be >= rank ({config['r']})"
        )
    if config.get("lora_dropout", 0) >= 0.5:
        return False, (
            f"Dropout ({config['lora_dropout']}) too high for small datasets"
        )
    return True, "LoRA config validated"


def create_lora_config(
    config_path: str = "configs/base_lora.json",
    override: Optional[Dict[str, Any]] = None,
) -> "Any":
    """Create a LoRA configuration from JSON config file.

    Args:
        config_path: Path to LoRA configuration JSON file.
        override: Optional dict of parameters to override.

    Returns:
        PEFT LoraConfig object.

    Raises:
        ValueError: If LoRA rank exceeds competition maximum of 32.
        RuntimeError: If peft is not installed.
    """
    if not _HAS_PEFT:
        raise RuntimeError(
            "peft is required for create_lora_config. Install with: pip install peft"
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    if override:
        config.update(override)

    # CRITICAL: Enforce competition rank constraint
    max_rank = 32
    if config.get("r", 0) > max_rank:
        raise ValueError(
            f"LoRA rank {config['r']} exceeds competition maximum of {max_rank}. "
            f"This would result in disqualification."
        )

    # Map task_type string to PEFT TaskType enum
    task_type = config.pop("task_type", "CAUSAL_LM")
    task_type_enum = getattr(TaskType, task_type, TaskType.CAUSAL_LM)

    # Remove non-LoraConfig keys
    config.pop("_comment", None)

    lora_config = LoraConfig(
        task_type=task_type_enum,
        **config,
    )

    print(f"LoRA config created: rank={lora_config.r}, alpha={lora_config.lora_alpha}")
    print(f"Target modules: {lora_config.target_modules}")
    return lora_config


def validate_adapter(adapter_path: str):
    """Validate that a saved LoRA adapter meets competition constraints.

    Args:
        adapter_path: Path to saved adapter directory.

    Returns:
        Tuple of (ok: bool, message: str).

    Raises:
        FileNotFoundError: If adapter_config.json is missing.
        ValueError: If adapter violates competition constraints.
    """
    config_file = Path(adapter_path) / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

    with open(config_file, "r") as f:
        config = json.load(f)

    rank = config.get("r", 0)
    if rank > 32:
        raise ValueError(f"Adapter rank {rank} exceeds maximum of 32")

    return True, f"Adapter validated: rank={rank}, compliant=True"
