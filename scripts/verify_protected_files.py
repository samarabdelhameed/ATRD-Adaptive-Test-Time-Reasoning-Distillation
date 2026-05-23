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

import subprocess
import sys
from typing import List

# Files that must NEVER be modified after initial commit
PROTECTED_FILES = [
    "configs/competition_params.json",
]


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
