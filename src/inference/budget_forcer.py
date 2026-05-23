"""
Adaptive Budget Forcing (inference wrapper)

Delegates core difficulty/budget logic to ``src.data.budget_forcer`` for a
single source of truth. This class batches problems for data-generation or
local evaluation workflows; competition submission uses fixed vLLM params.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.data import budget_forcer as data_budget


class BudgetForcer:
    """Adaptive reasoning budget controller (batch wrapper)."""

    def __init__(
        self,
        config_path: str = "configs/competition_params.json",
        min_tokens: int = 512,
        max_tokens: Optional[int] = None,
    ) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.max_tokens = max_tokens or self.config.get("max_tokens", 7680)
        self.min_tokens = min_tokens

    def estimate_difficulty(self, problem: str) -> float:
        return data_budget.estimate_difficulty(problem)

    def allocate_budget(self, difficulty: float) -> int:
        return data_budget.allocate_budget(difficulty)

    def force_budget(
        self,
        problems: List[str],
        generate_fn: Callable[..., str],
        adaptive: bool = True,
    ) -> List[Dict[str, Any]]:
        """Apply budget forcing to a batch of problems."""
        data_budget.set_generate_backend(generate_fn)
        results: List[Dict[str, Any]] = []

        for problem in problems:
            if adaptive:
                difficulty = self.estimate_difficulty(problem)
                budget = self.allocate_budget(difficulty)
            else:
                difficulty = 1.0
                budget = self.max_tokens

            response = data_budget.generate(problem, max_tokens=budget)
            results.append({
                "problem": problem,
                "response": response,
                "budget_allocated": budget,
                "difficulty_estimate": difficulty,
                "budget_used": len(response.split()) if isinstance(response, str) else 0,
            })

        return results

    def get_budget_stats(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        return data_budget.get_budget_stats(results)
