#!/usr/bin/env python3
"""
Verify Unit Completion Gates

Checks phase completion criteria before allowing progression:
1. Required artifact existence (stage-specific)
2. LoRA rank invariant (rank <= 32)
3. Protected file integrity

Usage:
    python scripts/verify_unit_completion.py P1              # config-only check
    python scripts/verify_unit_completion.py P1 baseline     # baseline eval artifacts
    python scripts/verify_unit_completion.py P1 complete     # full data pipeline
    python scripts/verify_unit_completion.py P2 sft
    python scripts/verify_unit_completion.py P3 grpo
    python scripts/verify_unit_completion.py P4 submission
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PHASE_GATES: Dict[str, Dict] = {
    "P1": {
        "name": "Data Generation",
        "config_artifacts": [
            "configs/competition_params.json",
            "configs/base_lora.json",
        ],
        "description": "Baseline evaluation → Failure extraction → Synthetic generation",
    },
    "P2": {
        "name": "SFT Training",
        "config_artifacts": [
            "configs/competition_params.json",
            "configs/base_lora.json",
        ],
        "description": "Supervised fine-tuning with synthetic data",
    },
    "P3": {
        "name": "GRPO Training",
        "config_artifacts": [
            "configs/competition_params.json",
            "configs/base_grpo.json",
        ],
        "description": "Group Relative Policy Optimization",
    },
    "P4": {
        "name": "Budget Forcing & Submission",
        "config_artifacts": [
            "configs/competition_params.json",
        ],
        "description": "Evaluation → Package submission",
    },
}

# Stage-specific REQUIRED artifacts (must exist to pass)
STAGE_REQUIRED: Dict[Tuple[str, str], List[str]] = {
    ("P1", "baseline"): [
        "data/public_test.jsonl",  # local dev OR produced on Kaggle
    ],
    ("P1", "complete"): [
        "data/final_train_dataset.jsonl",
    ],
    ("P2", "sft"): [
        "checkpoints/sft/final_adapter/adapter_config.json",
        "data/final_train_dataset.jsonl",
    ],
    ("P3", "grpo"): [
        "checkpoints/sft/final_adapter/adapter_config.json",
        "checkpoints/grpo/final_adapter/adapter_config.json",
    ],
    ("P4", "submission"): [
        "checkpoints/grpo/final_adapter/adapter_config.json",
        "submission.zip",
    ],
}

# Any one of these satisfies baseline eval requirement
STAGE_ALTERNATIVES: Dict[Tuple[str, str], List[List[str]]] = {
    ("P1", "baseline"): [
        ["data/baseline_results.json"],
        ["logs/baseline_results.json"],
    ],
}


def _any_group_exists(groups: List[List[str]]) -> Tuple[bool, str]:
    for group in groups:
        if all(Path(p).exists() for p in group):
            return True, ", ".join(group)
    return False, ""


def check_stage_artifacts(phase: str, stage: str) -> Tuple[List[str], List[str]]:
    """Return (missing_required, satisfied_notes)."""
    key = (phase, stage)
    missing: List[str] = []
    notes: List[str] = []

    if key in STAGE_ALTERNATIVES:
        ok, found = _any_group_exists(STAGE_ALTERNATIVES[key])
        if not ok:
            missing.append(
                f"one of: {STAGE_ALTERNATIVES[key]} (baseline evaluation output)"
            )
        else:
            notes.append(f"baseline output: {found}")

    for artifact in STAGE_REQUIRED.get(key, []):
        if not Path(artifact).exists():
            missing.append(artifact)
        else:
            notes.append(f"found {artifact}")

    return missing, notes


def check_config_artifacts(phase: str) -> List[str]:
    gate = PHASE_GATES.get(phase, {})
    return [
        a for a in gate.get("config_artifacts", [])
        if not Path(a).exists()
    ]


def check_lora_rank(adapter_path: str = "checkpoints") -> bool:
    adapter_configs = list(Path(adapter_path).rglob("adapter_config.json"))

    if not adapter_configs:
        print("  ⚠ No adapter_config.json files found (expected if pre-training)")
        return True

    all_compliant = True
    for config_path in adapter_configs:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        rank = config.get("r", 0)
        if rank > 32:
            print(f"  ✗ VIOLATION: {config_path} has rank={rank} (max=32)")
            all_compliant = False
        else:
            print(f"  ✓ {config_path}: rank={rank}")

    return all_compliant


def check_protected_files() -> bool:
    protected = ["configs/competition_params.json"]
    for filepath in protected:
        if not Path(filepath).exists():
            print(f"  ✗ Protected file missing: {filepath}")
            return False
        print(f"  ✓ Protected file intact: {filepath}")
    return True


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_unit_completion.py <PHASE> [<STAGE>]")
        print(f"Phases: {', '.join(PHASE_GATES.keys())}")
        print("Stages: check (default), baseline, complete, sft, grpo, submission")
        return 1

    phase = sys.argv[1].upper()
    stage = sys.argv[2].lower() if len(sys.argv) > 2 else "check"

    if phase not in PHASE_GATES:
        print(f"Unknown phase: {phase}")
        return 1

    gate = PHASE_GATES[phase]
    print(f"\n{'='*60}")
    print(f"Phase Gate Verification: {phase} — {gate['name']}")
    print(f"Stage: {stage}")
    print(f"{'='*60}\n")

    all_passed = True

    print("1. Config Artifact Check:")
    missing_cfg = check_config_artifacts(phase)
    if missing_cfg:
        print(f"  ✗ Missing: {missing_cfg}")
        all_passed = False
    else:
        print("  ✓ All config artifacts present")

    if stage != "check":
        print(f"\n2. Stage Artifact Check ({stage}):")
        missing_stage, notes = check_stage_artifacts(phase, stage)
        if missing_stage:
            print(f"  ✗ Missing REQUIRED:")
            for m in missing_stage:
                print(f"      - {m}")
            all_passed = False
        else:
            print("  ✓ All stage artifacts present")
            for n in notes:
                print(f"      • {n}")

    print("\n3. LoRA Rank Invariant Check:")
    if not check_lora_rank():
        all_passed = False

    print("\n4. Protected Files Check:")
    if not check_protected_files():
        all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print(f"✓ PHASE {phase} GATE ({stage}): PASSED")
    else:
        print(f"✗ PHASE {phase} GATE ({stage}): FAILED")
        print("  Resolve missing artifacts before proceeding.")
    print(f"{'='*60}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
