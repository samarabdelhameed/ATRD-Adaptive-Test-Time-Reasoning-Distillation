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

Usage:
    python scripts/package_submission.py --adapter-path checkpoints/grpo/final_adapter
    python scripts/package_submission.py --adapter-path checkpoints/sft/final_adapter --output submission.zip
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
        print(f"✗ Adapter path does not exist: {adapter_dir}")
        return False

    print(f"Packaging submission from: {adapter_dir}")
    print(f"Output: {output_path}\n")

    # Validate adapter config
    print("1. Validating adapter_config.json...")
    is_valid, errors = validate_adapter_config(adapter_dir)
    if not is_valid:
        for err in errors:
            print(f"  ✗ {err}")
        return False
    print("  ✓ adapter_config.json is valid")

    # Find files
    print("\n2. Collecting files...")
    files = find_model_files(adapter_dir)
    if not files:
        print("  ✗ No files found in adapter directory")
        return False

    total_size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  • {f.name} ({size_mb:.2f} MB)")
    print(f"  Total: {total_size_mb:.2f} MB")

    if total_size_mb > MAX_SUBMISSION_SIZE_MB:
        print(f"  ✗ Total size exceeds {MAX_SUBMISSION_SIZE_MB} MB limit")
        return False

    # Check for required files
    print("\n3. Checking required files...")
    file_names = {f.name for f in files}
    for req in REQUIRED_FILES:
        if req in file_names:
            print(f"  ✓ {req}")
        else:
            print(f"  ✗ Missing: {req}")
            return False

    # Check for model weights
    has_weights = any(
        f.name in EXPECTED_FILES
        for f in files
    )
    if not has_weights:
        print(f"  ⚠ Warning: No model weight file found ({EXPECTED_FILES})")

    if dry_run:
        print("\n[DRY RUN] Validation complete. No zip created.")
        return True

    # Create zip
    print(f"\n4. Creating {output_path}...")
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.name)
            print(f"  + {f.name}")

    zip_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✓ Submission packaged: {output_path} ({zip_size_mb:.2f} MB)")
    return True


def main() -> int:
    """Main entry point."""
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
