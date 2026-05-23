#!/usr/bin/env python3
"""
Package Submission

Creates a competition-compliant submission.zip containing:
- adapter_config.json
- adapter_model.safetensors (or .bin)
- tokenizer files (if modified)

Validates:
1. adapter_config.json schema
2. LoRA rank <= 32
3. File size constraints
4. Required files present
5. vLLM compatibility (loads adapter and generates output)

Usage:
    python scripts/package_submission.py --adapter-path checkpoints/grpo/final_adapter
    python scripts/package_submission.py --adapter-path checkpoints/sft/final_adapter --output submission.zip
"""

import argparse
import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Required files in submission
REQUIRED_FILES = [
    "adapter_config.json",
]

# Optional but expected files
EXPECTED_FILES = [
    "adapter_model.safetensors",
    "adapter_model.bin",
]

# Maximum submission size (100 MB)
MAX_SUBMISSION_SIZE_MB = 100

# Model name for vLLM compatibility test
MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"


def validate_adapter_config(adapter_path: Path) -> Tuple[bool, List[str]]:
    """Validate adapter_config.json schema and constraints.

    Args:
        adapter_path: Path to adapter directory.

    Returns:
        Tuple of (is_valid, error_messages).
    """
    errors = []
    config_path = adapter_path / "adapter_config.json"

    if not config_path.exists():
        return False, ["adapter_config.json not found"]

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]

    # Check LoRA rank
    rank = config.get("r", None)
    if rank is None:
        errors.append("Missing 'r' (LoRA rank) in adapter_config.json")
    elif rank > 32:
        errors.append(f"LoRA rank {rank} exceeds maximum of 32 — DISQUALIFICATION RISK")

    # Check task type
    if config.get("task_type") != "CAUSAL_LM":
        errors.append(f"task_type should be 'CAUSAL_LM', got '{config.get('task_type')}'")

    return len(errors) == 0, errors


def find_model_files(adapter_path: Path) -> List[Path]:
    """Find all files to include in submission.

    Args:
        adapter_path: Path to adapter directory.

    Returns:
        List of file paths to include.
    """
    files = []
    for f in adapter_path.iterdir():
        if f.is_file() and not f.name.startswith("."):
            files.append(f)
    return files


def test_vllm_compatibility(adapter_path: Path) -> bool:
    """Verify adapter loads correctly in vLLM.

    Checks if vLLM is installed; if not, logs a warning and skips the test.
    If vLLM is available, loads the model with the adapter and generates
    a test response to verify compatibility.

    Args:
        adapter_path: Path to adapter directory.

    Returns:
        True if test passes or vLLM is not installed, False on failure.
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        logger.warning("vLLM not installed — skipping compatibility test")
        return True

    logger.info("Testing vLLM compatibility...")
    try:
        engine = LLM(
            model=MODEL_NAME,
            adapter=str(adapter_path),
            max_model_len=8192,
            gpu_memory_utilization=0.85,
            enable_lora=True,
            max_lora_rank=32,
        )
        test_prompt = "Solve for x: 2x = 10"
        outputs = engine.generate(test_prompt, SamplingParams(max_tokens=128))
        if outputs and outputs[0].outputs and outputs[0].outputs[0].text:
            logger.info("vLLM test generation successful")
            return True
        logger.error("vLLM test generated empty response")
        return False
    except Exception as e:
        logger.error("vLLM compatibility test failed: %s", e)
        return False


def package(
    adapter_path: str,
    output_path: str = "submission.zip",
    dry_run: bool = False,
) -> bool:
    """Package adapter into submission.zip.

    Args:
        adapter_path: Path to LoRA adapter directory.
        output_path: Output zip file path.
        dry_run: If True, validate only without creating zip.

    Returns:
        True if packaging was successful.
    """
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        logger.error("Adapter path does not exist: %s", adapter_dir)
        return False

    logger.info("Packaging submission from: %s", adapter_dir)
    logger.info("Output: %s\n", output_path)

    # Validate adapter config
    logger.info("1. Validating adapter_config.json...")
    is_valid, errors = validate_adapter_config(adapter_dir)
    if not is_valid:
        for err in errors:
            logger.error(err)
        return False
    logger.info("  ✓ adapter_config.json is valid")

    # Test vLLM compatibility
    logger.info("Testing vLLM compatibility...")
    vllm_ok = test_vllm_compatibility(adapter_dir)
    if not vllm_ok:
        logger.error("vLLM compatibility test failed — submission may fail at inference")

    # Find files
    logger.info("2. Collecting files...")
    files = find_model_files(adapter_dir)
    if not files:
        logger.error("No files found in adapter directory")
        return False

    total_size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info("  • %s (%.2f MB)", f.name, size_mb)
    logger.info("  Total: %.2f MB", total_size_mb)

    if total_size_mb > MAX_SUBMISSION_SIZE_MB:
        logger.error("Total size exceeds %d MB limit", MAX_SUBMISSION_SIZE_MB)
        return False

    # Check for required files
    logger.info("3. Checking required files...")
    file_names = {f.name for f in files}
    for req in REQUIRED_FILES:
        if req in file_names:
            logger.info("  ✓ %s", req)
        else:
            logger.error("Missing: %s", req)
            return False

    # Check for model weights
    has_weights = any(
        f.name in EXPECTED_FILES
        for f in files
    )
    if not has_weights:
        logger.warning("No model weight file found (%s)", EXPECTED_FILES)

    if dry_run:
        logger.info("[DRY RUN] Validation complete. No zip created.")
        return True

    # Create zip
    logger.info("4. Creating %s...", output_path)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.name)
            logger.info("  + %s", f.name)

    zip_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Submission packaged: %s (%.2f MB)", output_path, zip_size_mb)
    return True


def main() -> int:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Package LoRA adapter for competition submission"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="checkpoints/grpo/final_adapter",
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.zip",
        help="Output zip file path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only, don't create zip",
    )
    args = parser.parse_args()

    success = package(args.adapter_path, args.output, args.dry_run)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
