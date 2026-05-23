"""
LoRA Configuration Factory

Creates PEFT LoRA configurations from JSON config files,
enforcing rank <= 32 constraint per competition rules.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from peft import LoraConfig, TaskType


def validate_lora_config(config: dict) -> None:
    """Validate LoRA configuration meets competition constraints.

    Args:
        config: LoRA configuration dict.

    Raises:
        AssertionError: If any constraint is violated.
    """
    assert config.get("r", 0) <= 32, (
        f"LoRA rank {config['r']} exceeds competition maximum of 32. "
        "This would result in disqualification."
    )
    assert config.get("lora_alpha", 0) >= config.get("r", 0), (
        f"Alpha ({config['lora_alpha']}) must be >= rank ({config['r']})"
    )
    assert config.get("lora_dropout", 0) < 0.5, (
        f"Dropout ({config['lora_dropout']}) too high for small datasets"
    )
    print(f"✓ LoRA config validated: rank={config['r']}, alpha={config['lora_alpha']}")


def create_lora_config(
    config_path: str = "configs/base_lora.json",
    override: Optional[Dict[str, Any]] = None,
) -> LoraConfig:
    """Create a LoRA configuration from JSON config file.

    Args:
        config_path: Path to LoRA configuration JSON file.
        override: Optional dict of parameters to override.

    Returns:
        PEFT LoraConfig object.

    Raises:
        ValueError: If LoRA rank exceeds competition maximum of 32.
    """
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


def validate_adapter(adapter_path: str) -> bool:
    """Validate that a saved LoRA adapter meets competition constraints.

    Args:
        adapter_path: Path to saved adapter directory.

    Returns:
        True if adapter is competition-compliant.

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

    print(f"✓ Adapter validated: rank={rank}, compliant=True")
    return True
