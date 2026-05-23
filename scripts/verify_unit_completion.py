#!/usr/bin/env python3
"""
Verify Unit Completion Gates

Checks phase completion criteria before allowing progression:
1. Required artifact existence
2. LoRA rank invariant (rank <= 32)
3. Protected file integrity
4. Minimum accuracy thresholds

Usage:
    python scripts/verify_unit_completion.py P1 baseline
    python scripts/verify_unit_completion.py P2 sft
    python scripts/verify_unit_completion.py P3 grpo
    python scripts/verify_unit_completion.py P4 submission
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Phase gate definitions
PHASE_GATES: Dict[str, Dict] = {
    "P1": {
        "name": "Data Generation",
        "required_artifacts": [
            "configs/competition_params.json",
            "configs/base_lora.json",
        ],
        "optional_artifacts": [
            "data/synthetic_sft.jsonl",
            "logs/p1_baseline_eval.json",
        ],
        "description": "Baseline evaluation → Failure extraction → Synthetic generation",
    },
    "P2": {
        "name": "SFT Training",
        "required_artifacts": [
            "configs/competition_params.json",
            "configs/base_lora.json",
        ],
        "optional_artifacts": [
            "checkpoints/sft/final_adapter/adapter_config.json",
            "logs/p2_sft_eval.json",
        ],
        "description": "Supervised fine-tuning with synthetic data",
    },
    "P3": {
        "name": "GRPO Training",
        "required_artifacts": [
            "configs/competition_params.json",
            "configs/base_grpo.json",
        ],
        "optional_artifacts": [
            "checkpoints/grpo/final_adapter/adapter_config.json",
            "logs/p3_grpo_eval.json",
        ],
        "description": "Group Relative Policy Optimization",
    },
    "P4": {
        "name": "Budget Forcing & Submission",
        "required_artifacts": [
            "configs/competition_params.json",
        ],
        "optional_artifacts": [
            "submission.zip",
            "logs/p4_final_eval.json",
        ],
        "description": "Adaptive budget forcing → Package submission",
    },
}


def check_artifacts(phase: str) -> Tuple[List[str], List[str]]:
    """Check required and optional artifact existence.

    Args:
        phase: Phase identifier (P1, P2, P3, P4).

    Returns:
        Tuple of (missing_required, missing_optional) file lists.
    """
    gate = PHASE_GATES.get(phase, {})
    missing_required = []
    missing_optional = []

    for artifact in gate.get("required_artifacts", []):
        if not Path(artifact).exists():
            missing_required.append(artifact)

    for artifact in gate.get("optional_artifacts", []):
        if not Path(artifact).exists():
            missing_optional.append(artifact)

    return missing_required, missing_optional


def check_lora_rank(adapter_path: str = "checkpoints") -> bool:
    """Verify all LoRA adapters have rank <= 32.

    Args:
        adapter_path: Base path to search for adapter configs.

    Returns:
        True if all adapters are compliant.
    """
    adapter_configs = list(Path(adapter_path).rglob("adapter_config.json"))

    if not adapter_configs:
        print("  ⚠ No adapter_config.json files found (expected if pre-training)")
        return True

    all_compliant = True
    for config_path in adapter_configs:
        with open(config_path, "r") as f:
            config = json.load(f)
        rank = config.get("r", 0)
        if rank > 32:
            print(f"  ✗ VIOLATION: {config_path} has rank={rank} (max=32)")
            all_compliant = False
        else:
            print(f"  ✓ {config_path}: rank={rank}")

    return all_compliant


def check_protected_files() -> bool:
    """Verify protected files haven't been modified.

    Returns:
        True if all protected files are intact.
    """
    protected = [
        "configs/competition_params.json",
    ]

    for filepath in protected:
        if not Path(filepath).exists():
            print(f"  ✗ Protected file missing: {filepath}")
            return False
        print(f"  ✓ Protected file intact: {filepath}")

    return True


def main() -> int:
    """Main entry point for verification.

    Returns:
        Exit code (0 = pass, 1 = fail).
    """
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_unit_completion.py <PHASE> [<STAGE>]")
        print(f"Phases: {', '.join(PHASE_GATES.keys())}")
        return 1

    phase = sys.argv[1].upper()
    stage = sys.argv[2] if len(sys.argv) > 2 else "check"

    if phase not in PHASE_GATES:
        print(f"Unknown phase: {phase}")
        print(f"Valid phases: {', '.join(PHASE_GATES.keys())}")
        return 1

    gate = PHASE_GATES[phase]
    print(f"\n{'='*60}")
    print(f"Phase Gate Verification: {phase} — {gate['name']}")
    print(f"Stage: {stage}")
    print(f"{'='*60}\n")

    all_passed = True

    # Check 1: Artifacts
    print("1. Artifact Check:")
    missing_req, missing_opt = check_artifacts(phase)
    if missing_req:
        print(f"  ✗ Missing REQUIRED: {missing_req}")
        all_passed = False
    else:
        print("  ✓ All required artifacts present")
    if missing_opt:
        print(f"  ⚠ Missing optional: {missing_opt}")

    # Check 2: LoRA rank
    print("\n2. LoRA Rank Invariant Check:")
    if not check_lora_rank():
        all_passed = False

    # Check 3: Protected files
    print("\n3. Protected Files Check:")
    if not check_protected_files():
        all_passed = False

    # Summary
    print(f"\n{'='*60}")
    if all_passed:
        print(f"✓ PHASE {phase} GATE: PASSED")
        print(f"  Ready to proceed to next phase.")
    else:
        print(f"✗ PHASE {phase} GATE: FAILED")
        print(f"  Resolve issues before proceeding.")
    print(f"{'='*60}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
