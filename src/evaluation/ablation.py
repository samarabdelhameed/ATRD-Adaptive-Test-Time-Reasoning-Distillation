"""
Ablation Study Runner

Systematic ablation framework for comparing different
training configurations and their impact on reasoning accuracy.
"""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class AblationRunner:
    """Run systematic ablation studies across configurations.

    Attributes:
        results_dir: Directory to save ablation results.
        results: List of completed ablation results.
    """

    def __init__(self, results_dir: str = "logs/ablations") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []

    def run_ablation(
        self,
        name: str,
        config: Dict[str, Any],
        train_fn: Callable,
        eval_fn: Callable,
        baseline_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run a single ablation experiment.

        Args:
            name: Experiment name.
            config: Configuration dict for this experiment.
            train_fn: Training function that takes config and returns model.
            eval_fn: Evaluation function that takes model and returns score.
            baseline_score: Optional baseline score for delta computation.

        Returns:
            Result dict with name, config, score, and timing.
        """
        print(f"Running ablation: {name}")
        start_time = time.time()

        try:
            model = train_fn(config)
            score = eval_fn(model)
            status = "completed"
        except Exception as e:
            score = 0.0
            status = f"failed: {str(e)}"
            print(f"  Ablation '{name}' failed: {e}")

        elapsed = time.time() - start_time

        result = {
            "name": name,
            "config": config,
            "score": score,
            "delta": score - baseline_score if baseline_score is not None else None,
            "elapsed_seconds": elapsed,
            "status": status,
        }

        self.results.append(result)
        print(f"  Score: {score:.4f} | Time: {elapsed:.1f}s | Status: {status}")
        return result

    def run_sweep(
        self,
        param_name: str,
        param_values: List[Any],
        base_config: Dict[str, Any],
        train_fn: Callable,
        eval_fn: Callable,
    ) -> List[Dict[str, Any]]:
        """Run a parameter sweep ablation.

        Args:
            param_name: Name of parameter to sweep.
            param_values: List of values to try.
            base_config: Base configuration dict.
            train_fn: Training function.
            eval_fn: Evaluation function.

        Returns:
            List of result dicts for each value.
        """
        sweep_results = []
        for value in param_values:
            config = {**base_config, param_name: value}
            result = self.run_ablation(
                name=f"{param_name}={value}",
                config=config,
                train_fn=train_fn,
                eval_fn=eval_fn,
            )
            sweep_results.append(result)

        return sweep_results

    def save_results(self, filename: str = "ablation_results.json") -> Path:
        """Save all ablation results to JSON file.

        Args:
            filename: Output filename.

        Returns:
            Path to saved results file.
        """
        output_path = self.results_dir / filename
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Saved {len(self.results)} ablation results to {output_path}")
        return output_path

    def get_best_config(self) -> Optional[Dict[str, Any]]:
        """Get the configuration with the highest score.

        Returns:
            Best result dict, or None if no results.
        """
        completed = [r for r in self.results if r["status"] == "completed"]
        if not completed:
            return None
        return max(completed, key=lambda r: r["score"])

    def summary_table(self) -> str:
        """Generate a formatted summary table of results.

        Returns:
            Formatted string table.
        """
        if not self.results:
            return "No ablation results."

        header = f"{'Name':<30} {'Score':<10} {'Delta':<10} {'Time':<10} {'Status':<15}"
        separator = "-" * len(header)
        rows = [header, separator]

        for r in sorted(self.results, key=lambda x: x["score"], reverse=True):
            delta_str = f"{r['delta']:+.4f}" if r["delta"] is not None else "N/A"
            rows.append(
                f"{r['name']:<30} {r['score']:<10.4f} {delta_str:<10} "
                f"{r['elapsed_seconds']:<10.1f} {r['status']:<15}"
            )

        return "\n".join(rows)
