#!/usr/bin/env python3
"""
ATRD Pipeline CLI — production entry point.

Usage:
    python run_pipeline.py --phase validate
    python run_pipeline.py --phase test
    python run_pipeline.py --phase p1_data
    python run_pipeline.py --phase p1_baseline
    python run_pipeline.py --phase p2_sft
    python run_pipeline.py --phase p3_grpo
    python run_pipeline.py --phase p4_eval
    python run_pipeline.py --phase p4_submit
    python run_pipeline.py --phase fill_writeup
    python run_pipeline.py --phase all
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("atrd")


def phase_p1_data(args: argparse.Namespace) -> bool:
    from src.pipeline.p1_data import run_p1_data_pipeline

    path = run_p1_data_pipeline(
        skip_baseline=args.skip_baseline,
        skip_synthetic=args.skip_synthetic,
    )
    logger.info("✅ P1 data → %s", path)
    return path.exists()


def phase_p1_baseline(args: argparse.Namespace) -> bool:
    from src.pipeline.baseline import run_baseline_evaluation

    try:
        report = run_baseline_evaluation(max_problems=args.max_problems)
        logger.info("✅ Baseline accuracy: %.2f%%", report["overall_accuracy"] * 100)
        return True
    except RuntimeError as e:
        logger.error("Baseline skipped: %s", e)
        return False


def phase_p2_sft(_: argparse.Namespace) -> bool:
    from src.pipeline.p2_sft import run_sft_training

    path = run_sft_training()
    logger.info("✅ SFT adapter → %s", path)
    return (path / "adapter_config.json").exists()


def phase_p3_grpo(_: argparse.Namespace) -> bool:
    from src.pipeline.p3_grpo import run_grpo_training

    path = run_grpo_training()
    logger.info("✅ GRPO adapter → %s", path)
    return (path / "adapter_config.json").exists()


def phase_p4_eval(_: argparse.Namespace) -> bool:
    from src.pipeline.p4_eval import run_final_evaluation, run_ablation_from_logs

    report = run_final_evaluation()
    ablation_path = run_ablation_from_logs()
    logger.info("Eval status: %s | ablation → %s", report.get("status"), ablation_path)
    return report.get("status") in ("completed", "pending", "skipped")


def phase_p4_submit(_: argparse.Namespace) -> bool:
    from scripts.package_submission import package
    from src.pipeline._config import load_pipeline_config

    cfg = load_pipeline_config()
    adapter = cfg.get("submission_adapter_path", "checkpoints/grpo/final_adapter")
    ok = package(adapter_path=adapter, output_path="submission.zip")
    if ok:
        logger.info("✅ submission.zip created")
    else:
        logger.error("❌ Packaging failed — train adapter first (p2_sft → p3_grpo)")
    return ok


def phase_fill_writeup(_: argparse.Namespace) -> bool:
    from scripts.fill_writeup import fill_methodology

    n = fill_methodology()
    logger.info("✅ Filled %d [REAL DATA] markers in METHODOLOGY.md", n)
    return n >= 0


def phase_validate() -> bool:
    checks = [
        ("Config files", _check_configs),
        ("Source syntax", lambda: _check_syntax("src")),
        ("Scripts syntax", lambda: _check_syntax("scripts")),
        ("No mock in notebooks", _check_no_mock),
        ("Tests", _run_tests_quick),
    ]
    ok = True
    for name, fn in checks:
        try:
            result = fn()
            logger.info("  %s %s", "✅" if result else "❌", name)
            ok = ok and result
        except Exception as e:
            logger.error("  ❌ %s: %s", name, e)
            ok = False
    return ok


def phase_test() -> bool:
    r = subprocess.run([sys.executable, "tests/test_all.py"])
    return r.returncode == 0


def phase_all(args: argparse.Namespace) -> int:
    steps = [
        ("validate", lambda: phase_validate()),
        ("p1_data", lambda: phase_p1_data(args)),
        ("p1_baseline", lambda: phase_p1_baseline(args)),
        ("p2_sft", lambda: phase_p2_sft(args)),
        ("p3_grpo", lambda: phase_p3_grpo(args)),
        ("p4_eval", lambda: phase_p4_eval(args)),
        ("p4_submit", lambda: phase_p4_submit(args)),
        ("fill_writeup", lambda: phase_fill_writeup(args)),
    ]
    results = {}
    for name, fn in steps:
        logger.info("=" * 60)
        logger.info("STEP: %s", name)
        logger.info("=" * 60)
        try:
            results[name] = fn()
        except Exception as e:
            logger.exception("Step %s failed: %s", name, e)
            results[name] = False
        if name in ("p2_sft", "p3_grpo", "p4_submit") and not results[name]:
            logger.warning(
                "Stopping after %s — requires GPU. Resume on Kaggle with remaining phases.",
                name,
            )
            break

    logger.info("Pipeline summary: %s", results)
    return 0 if results.get("validate") and results.get("p1_data") else 1


def _check_configs() -> bool:
    return all(
        Path(f"configs/{f}").exists()
        for f in ("competition_params.json", "base_lora.json", "base_grpo.json", "pipeline.json")
    )


def _check_syntax(directory: str) -> bool:
    import py_compile
    for py in Path(directory).rglob("*.py"):
        py_compile.compile(str(py), doraise=True)
    return True


def _check_no_mock() -> bool:
    import json
    for nb in Path("notebooks").glob("*.ipynb"):
        data = json.loads(nb.read_text())
        code = " ".join("".join(c["source"]) for c in data["cells"] if c["cell_type"] == "code")
        if "mock" in code.lower() or "dummy" in code.lower():
            return False
    return True


def _run_tests_quick() -> bool:
    return subprocess.run(
        [sys.executable, "tests/test_all.py"],
        capture_output=True,
    ).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ATRD competition pipeline")
    parser.add_argument(
        "--phase",
        default="validate",
        choices=[
            "validate", "test", "p1_data", "p1_baseline",
            "p2_sft", "p3_grpo", "p4_eval", "p4_submit",
            "fill_writeup", "all",
        ],
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--max-problems", type=int, default=None)
    args = parser.parse_args()

    handlers = {
        "validate": lambda: phase_validate(),
        "test": phase_test,
        "p1_data": lambda: phase_p1_data(args),
        "p1_baseline": lambda: phase_p1_baseline(args),
        "p2_sft": lambda: phase_p2_sft(args),
        "p3_grpo": lambda: phase_p3_grpo(args),
        "p4_eval": lambda: phase_p4_eval(args),
        "p4_submit": lambda: phase_p4_submit(args),
        "fill_writeup": lambda: phase_fill_writeup(args),
        "all": lambda: phase_all(args),
    }

    result = handlers[args.phase]()
    if args.phase == "all":
        return result
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
