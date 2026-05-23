#!/usr/bin/env python3
"""Fill [REAL DATA] markers in writeup/METHODOLOGY.md from pipeline logs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.2f}%"


def collect_metrics() -> Dict[str, str]:
    """Gather substitution values from logs (only real completed runs)."""
    m: Dict[str, str] = {}

    baseline = _load_json(Path("data/baseline_results.json")) or _load_json(
        Path("logs/baseline_results.json")
    )
    if baseline and baseline.get("status") == "completed":
        m["baseline_accuracy"] = _pct(baseline.get("overall_accuracy"))
        m["baseline_count"] = str(baseline.get("total_count", "—"))

    p1 = _load_json(Path("data/p1_stats.json"))
    if p1:
        m["synthetic_count"] = str(p1.get("raw_synthetic", "—"))
        m["final_dataset_count"] = str(p1.get("final_total", "—"))
        m["openmath_count"] = str(p1.get("openmath", "—"))
        m["opencode_count"] = str(p1.get("opencode", "—"))

    sft = _load_json(Path("logs/sft_results.json"))
    if sft:
        m["sft_loss"] = f"{sft.get('training_loss', 0):.4f}"

    final = _load_json(Path("logs/p4_final_eval.json"))
    if final and final.get("status") == "completed":
        m["public_accuracy"] = _pct(final.get("overall_accuracy"))
        m["public_correct"] = str(final.get("correct_count", "—"))

    ablation = _load_json(Path("logs/ablation_results.json"))
    if ablation and ablation.get("ablations"):
        for row in ablation["ablations"]:
            name = row.get("name", "")
            score = row.get("score") or row.get("accuracy")
            delta = row.get("delta")
            if score is not None:
                m[f"ablation_{name}"] = _pct(score)
            if delta is not None:
                m[f"ablation_{name}_delta"] = f"+{_pct(delta).replace('—', '0')}"

    gap = (ablation or {}).get("generalization_gap") or {}
    if gap.get("public_test_accuracy") is not None:
        m["private_accuracy"] = _pct(gap.get("private_test_accuracy"))
        m["generalization_gap"] = _pct(gap.get("generalization_gap"))

    return m


def fill_methodology(
    writeup_path: str = "writeup/METHODOLOGY.md",
) -> int:
    """Replace [REAL DATA: ...] markers where metrics exist."""
    path = Path(writeup_path)
    text = path.read_text(encoding="utf-8")
    metrics = collect_metrics()

    replacements = {
        "baseline accuracy": metrics.get("baseline_accuracy", None),
        "final public": metrics.get("public_accuracy", None),
        "public and private accuracy": metrics.get("public_accuracy", None),
        "synthetic": metrics.get("synthetic_count", None),
        "final dataset": metrics.get("final_dataset_count", None),
        "SFT loss": metrics.get("sft_loss", None),
        "accuracy delta": metrics.get("ablation_sft_only_delta", None),
    }

    filled = 0

    def replacer(match: re.Match) -> str:
        nonlocal filled
        inner = match.group(1).lower()
        for key, val in replacements.items():
            if key in inner and val is not None:
                filled += 1
                return val
        return match.group(0)

    new_text = re.sub(r"\[REAL DATA:\s*([^\]]+)\]", replacer, text, flags=re.IGNORECASE)

    # Append metrics table if we have real baseline
    if metrics.get("baseline_accuracy") and "## 10. Results Summary" not in new_text:
        appendix = (
            "\n\n## 10. Results Summary (auto-filled from logs)\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
        )
        for k, v in sorted(metrics.items()):
            appendix += f"| {k} | {v} |\n"
        new_text += appendix

    path.write_text(new_text, encoding="utf-8")
    return filled


if __name__ == "__main__":
    n = fill_methodology()
    print(f"Filled {n} markers")
