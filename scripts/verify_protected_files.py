#!/usr/bin/env python3
"""
Verify Protected Files

Pre-commit hook script that prevents modification of
immutable competition configuration files.

Protected files:
- configs/competition_params.json

Usage:
    Called automatically via .git/hooks/pre-commit
    Or manually: python scripts/verify_protected_files.py
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# Files that must NEVER be modified after initial commit
PROTECTED_FILES = [
    "configs/competition_params.json",
]

# Known-good hashes (updated on initial commit)
KNOWN_HASHES: Dict[str, str] = {}


def get_staged_files() -> List[str]:
    """Get list of files staged for commit.

    Returns:
        List of staged file paths.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
        )
        return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    except Exception:
        return []


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        filepath: Path to file.

    Returns:
        Hex digest of file hash.
    """
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def verify() -> bool:
    """Verify no protected files are being modified.

    Returns:
        True if no protected files are staged for commit.
    """
    staged = get_staged_files()
    violations = []

    for protected in PROTECTED_FILES:
        if protected in staged:
            violations.append(protected)

    if violations:
        print("╔══════════════════════════════════════════════════════╗")
        print("║  ⛔ PROTECTED FILE MODIFICATION DETECTED            ║")
        print("╠══════════════════════════════════════════════════════╣")
        for v in violations:
            print(f"║  • {v:<50} ║")
        print("╠══════════════════════════════════════════════════════╣")
        print("║  These files are IMMUTABLE per competition rules.   ║")
        print("║  Unstage them with: git reset HEAD <file>           ║")
        print("╚══════════════════════════════════════════════════════╝")
        return False

    print("✓ Protected files check passed")
    return True


def main() -> int:
    """Main entry point."""
    return 0 if verify() else 1


if __name__ == "__main__":
    sys.exit(main())
