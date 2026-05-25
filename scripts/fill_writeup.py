#!/usr/bin/env python3
"""Fill [REAL DATA: ...] markers in writeup/METHODOLOGY.md from pipeline logs.

Replaces each marker with real data when available, or with a descriptive
pending message (not mock data) when the relevant pipeline phase hasn't run.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fill_writeup")


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.2f}%"


def _fmt_mb(path: Path) -> str:
    if not path.exists():
        return "—"
    return f"{path.stat().st_size / (1024 * 1024):.1f} MB"


def collect_metrics() -> Dict[str, str]:
    """Gather substitution values from all available pipeline data."""
    m: Dict[str, str] = {}

    # ── P1: Data stats ──────────────────────────────────────────
    p1 = _load_json(Path("data/p1_stats.json"))
    if p1 is not None:
        m["synthetic_count"] = str(p1.get("raw_synthetic", "—"))
        m["filtered_count"] = str(p1.get("filtered", "—"))
        m["deduplicated_count"] = str(p1.get("deduplicated", "—"))
        m["final_dataset_count"] = str(p1.get("final_total", "—"))
        m["openmath_count"] = str(p1.get("openmath", "—"))
        m["opencode_count"] = str(p1.get("opencode", "—"))
    else:
        m["synthetic_count"] = "—"

    # ── Dataset file sizes ───────────────────────────────────────
    train_path = Path("data/final_train_dataset.jsonl")
    m["dataset_size_mb"] = _fmt_mb(train_path)
    m["train_examples"] = str(_load_jsonl_count(train_path))

    # ── Baseline eval (if available) ─────────────────────────────
    baseline = _load_json(Path("data/baseline_results.json")) or _load_json(
        Path("logs/baseline_results.json")
    )
    if baseline and baseline.get("status") == "completed":
        m["baseline_accuracy"] = _pct(baseline.get("overall_accuracy"))
        m["baseline_count"] = str(baseline.get("total_count", "—"))
    else:
        m["baseline_accuracy"] = "pending (run p1_baseline)"

    # ── SFT results (if available) ───────────────────────────────
    sft = _load_json(Path("logs/sft_results.json"))
    if sft:
        m["sft_loss"] = f"{sft.get('training_loss', 0):.4f}"
        m["sft_accuracy"] = _pct(sft.get("eval_accuracy"))
    else:
        m["sft_loss"] = "pending (run p2_sft)"

    # ── GRPO results (if available) ──────────────────────────────
    grpo = _load_json(Path("logs/grpo_results.json"))
    if grpo:
        m["grpo_final_reward"] = f"{grpo.get('final_reward', 0):.4f}"
        m["grpo_final_kl"] = f"{grpo.get('final_kl', 0):.6f}"
    else:
        m["grpo_final_reward"] = "pending (run p3_grpo)"

    # ── Final eval (if available) ────────────────────────────────
    final = _load_json(Path("logs/p4_final_eval.json"))
    if final and final.get("status") == "completed":
        m["public_accuracy"] = _pct(final.get("overall_accuracy"))
        m["public_correct"] = str(final.get("correct_count", "—"))
    else:
        m["public_accuracy"] = "pending (run p4_eval)"

    # ── Ablation results (if available) ──────────────────────────
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
    else:
        m["ablation_baseline"] = "pending (run full pipeline)"
        m["ablation_sft_only"] = "pending"
        m["ablation_sft_grpo"] = "pending"
        m["ablation_full_pipeline"] = "pending"

    # ── Private / generalization gap ─────────────────────────────
    if ablation:
        gap = ablation.get("generalization_gap") or {}
        if gap.get("public_test_accuracy") is not None:
            m["private_accuracy"] = _pct(gap.get("private_test_accuracy"))
            m["generalization_gap"] = _pct(gap.get("generalization_gap"))
    else:
        m["private_accuracy"] = "pending"
        m["generalization_gap"] = "pending"

    return m


def fill_methodology(
    writeup_path: str = "writeup/METHODOLOGY.md",
    dry_run: bool = False,
) -> int:
    """Replace [REAL DATA: ...] markers in the methodology write-up.

    Args:
        writeup_path: Path to the methodology markdown file.
        dry_run: If True, print replacements without modifying the file.

    Returns:
        Number of markers filled.
    """
    path = Path(writeup_path)
    text = path.read_text(encoding="utf-8")
    metrics = collect_metrics()

    # Map marker keywords → metric keys (case-insensitive prefix match)
    marker_map: Dict[str, str] = {
        "baseline accuracy": metrics.get("baseline_accuracy", "—"),
        "public and private accuracy": metrics.get("public_accuracy", "—"),
        "final public": metrics.get("public_accuracy", "—"),
        "leaderboard position": "pending (run full pipeline + submit)",
        "sft training loss": metrics.get("sft_loss", "—"),
        "accuracy delta": metrics.get("ablation_sft_only_delta", "—"),
        "synthetic": metrics.get("synthetic_count", "—"),
        "final dataset": metrics.get("final_dataset_count", "—"),
        "dataset size": metrics.get("dataset_size_mb", "—"),
        "train examples": metrics.get("train_examples", "—"),
        "grpo reward": metrics.get("grpo_final_reward", "—"),
        "kl divergence": metrics.get("grpo_final_kl", "—"),
        "stratified evaluation": metrics.get("ablation_full_pipeline", "—"),
        "ablation waterfall": metrics.get("ablation_full_pipeline", "—"),
        "openmath": metrics.get("openmath_count", "—"),
        "opencode": metrics.get("opencode_count", "—"),
        "prm correlation": "pending (run p3_grpo + evaluate)",
        "budget statistics": "pending (run p4_eval)",
    }

    filled = 0

    def replacer(match: re.Match) -> str:
        nonlocal filled
        full_match = match.group(0)
        # Bare [REAL DATA] — use generic pending
        if full_match.strip() == "[REAL DATA]":
            filled += 1
            return "— (pending)"
        # [REAL DATA: ...] — try keyword matching
        inner = match.group(1).lower().strip() if match.lastindex >= 1 else ""
        for keyword, replacement in marker_map.items():
            if keyword in inner:
                filled += 1
                val = str(replacement)
                if val.startswith("pending"):
                    logger.info("  ⏳ %s → %s", inner[:60], val)
                else:
                    logger.info("  ✅ %s → %s", inner[:60], val)
                return val
        logger.warning("  ❓ Unrecognized marker: %s", inner[:80])
        # Return the raw marker text unchanged for unrecognized patterns
        return match.group(0)

    # Single pass: replace both [REAL DATA: ...] and bare [REAL DATA]
    new_text = re.sub(
        r"\[REAL DATA(:\s*([^\]]+))?\]",
        replacer,
        text,
        flags=re.IGNORECASE,
    )

    # Section 10: Results Summary (append if baseline or ablation available)
    has_real_results = (
        metrics.get("baseline_accuracy")
        and not metrics["baseline_accuracy"].startswith("pending")
    ) or (
        metrics.get("ablation_baseline")
        and not metrics["ablation_baseline"].startswith("pending")
    )

    if has_real_results and "## 10. Results Summary" not in new_text:
        appendix = (
            "\n\n## 10. Results Summary (auto-filled from logs)\n\n"
            "| Metric | Value |\n|--------|-------|\n"
        )
        for k, v in sorted(metrics.items()):
            if not v.startswith("pending"):
                appendix += f"| {k} | {v} |\n"
        new_text += appendix

    if dry_run:
        print("\n[Dry-run mode — file not modified]")
        return filled

    path.write_text(new_text, encoding="utf-8")
    return filled


if __name__ == "__main__":
    import sys

    dry_run = "--dry-run" in sys.argv
    n = fill_methodology(dry_run=dry_run)

    # Summary
    remaining = (
        len(re.findall(r"\[REAL DATA:[^\]]*\]", Path("writeup/METHODOLOGY.md").read_text()))
        if not dry_run
        else 0
    )
    print(f"\n{'=' * 50}")
    print(f"Filled {n} markers.")

    markers_path = Path("writeup/METHODOLOGY.md")
    if not dry_run and markers_path.exists():
        full_text = markers_path.read_text()
        total_remaining = len(re.findall(r"\[REAL DATA[\]:\s]*\]", full_text))
        if total_remaining == 0:
            print("🎉 All [REAL DATA] markers replaced!")
        else:
            print(
                f"⏳ {total_remaining} markers remain. "
                "Need more pipeline phases to run."
            )
        sys.exit(0 if total_remaining == 0 else 1)
