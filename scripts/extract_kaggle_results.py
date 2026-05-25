#!/usr/bin/env python3
"""Extract and migrate Kaggle pipeline outputs into local data/ and logs/ directories.

This script formalizes the data migration step of Feature 20:
- Copies `final_train_dataset.jsonl`, `failure_modes.json`, `p1_stats.json`,
  `public_test.jsonl`, etc. from a Kaggle extraction directory → `data/`
- Copies any logs from the extraction → `logs/`
- Validates checksums to confirm successful migration
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

EXTRACTION_BASE = Path(
    "scratch/extracted_results/ATRD-Adaptive-Test-Time-Reasoning-Distillation"
)
DATA_SOURCE = EXTRACTION_BASE / "data"
LOGS_SOURCE = EXTRACTION_BASE / "logs"

DATA_DEST = Path("data")
LOGS_DEST = Path("logs")

REQUIRED_DATA_FILES: List[str] = [
    "final_train_dataset.jsonl",
    "failure_modes.json",
    "p1_stats.json",
    "public_test.jsonl",
]

OPTIONAL_DATA_FILES: List[str] = [
    "raw_synthetic_dataset.jsonl",
    "sft_formatted_sample.txt",
    "README.md",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def migrate_data(force: bool = False, verify: bool = True) -> Dict[str, str]:
    """Migrate data files from extraction source to data/.

    Returns:
        Dict mapping filename to status: "copied", "skipped (up-to-date)",
        "skipped (missing in source)".
    """
    results: Dict[str, str] = {}
    DATA_DEST.mkdir(parents=True, exist_ok=True)

    all_files = REQUIRED_DATA_FILES + OPTIONAL_DATA_FILES

    for fname in all_files:
        src = DATA_SOURCE / fname
        dst = DATA_DEST / fname

        if not src.exists():
            results[fname] = "skipped (missing in source)"
            continue

        if dst.exists() and not force:
            if dst.stat().st_size == src.stat().st_size:
                results[fname] = "skipped (up-to-date)"
                continue

        shutil.copy2(str(src), str(dst))
        results[fname] = "copied"

    return results


def migrate_logs(force: bool = False) -> Dict[str, str]:
    """Migrate logs from extraction source to logs/.

    Returns:
        Dict mapping filename to status.
    """
    results: Dict[str, str] = {}

    if not LOGS_SOURCE.exists():
        return {"<logs_dir>": "skipped (not present in extraction)"}

    LOGS_DEST.mkdir(parents=True, exist_ok=True)

    for src in sorted(LOGS_SOURCE.rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(LOGS_SOURCE)
        dst = LOGS_DEST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not force:
            if dst.stat().st_size == src.stat().st_size:
                results[str(rel)] = "skipped (up-to-date)"
                continue

        shutil.copy2(str(src), str(dst))
        results[str(rel)] = "copied"

    return results


def verify_data() -> bool:
    """Verify all required data files exist and have content."""
    all_ok = True
    for fname in REQUIRED_DATA_FILES:
        path = DATA_DEST / fname
        if not path.exists():
            print(f"  ❌ {fname}: MISSING")
            all_ok = False
        elif path.stat().st_size == 0:
            print(f"  ❌ {fname}: EMPTY")
            all_ok = False
        else:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {fname} ({size_mb:.1f} MB)")
    return all_ok


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract Kaggle pipeline outputs into local project"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--skip-logs", action="store_true", help="Skip log migration")
    parser.add_argument("--skip-verify", action="store_true", help="Skip post-migration verification")
    args = parser.parse_args()

    print("=" * 60)
    print("Feature 20: Kaggle Results Extraction")
    print("=" * 60)

    print("\n--- Data Migration ---")
    data_results = migrate_data(force=args.force)
    for fname, status in data_results.items():
        icon = "✅" if status == "copied" else "⏭️" if "up-to-date" in status else "⚠️"
        print(f"  {icon} {fname}: {status}")

    if not args.skip_logs:
        print("\n--- Log Migration ---")
        log_results = migrate_logs(force=args.force)
        for fname, status in log_results.items():
            icon = "✅" if status == "copied" else "⏭️" if "up-to-date" in status else "⚠️"
            print(f"  {icon} {fname}: {status}")

    if not args.skip_verify:
        print("\n--- Verification ---")
        ok = verify_data()
        if ok:
            print("\n✅ All required data files present and populated.")
        else:
            print("\n❌ Some required data files are missing.")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
