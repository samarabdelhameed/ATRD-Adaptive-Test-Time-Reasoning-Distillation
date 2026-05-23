"""
Adaptive Budget Forcing

Controls reasoning computation budget at test time by dynamically
adjusting token allocation based on problem difficulty estimation.
Implements the "think longer on hard problems" strategy.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class BudgetForcer:
    """Adaptive test-time reasoning budget controller.

    Manages token budget allocation for inference, allowing the model
    to spend more tokens on harder problems and fewer on easier ones.

    Attributes:
        max_tokens: Maximum token budget per problem.
        min_tokens: Minimum token budget per problem.
        config: Competition parameters.
    """

    def __init__(
        self,
        config_path: str = "configs/competition_params.json",
        min_tokens: int = 256,
        max_tokens: Optional[int] = None,
    ) -> None:
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.max_tokens = max_tokens or self.config.get("max_tokens", 7680)
        self.min_tokens = min_tokens

    def estimate_difficulty(self, problem: str) -> float:
        """Estimate problem difficulty on a 0-1 scale.

        Uses heuristic features to estimate how much reasoning
        budget a problem requires.

        Args:
            problem: Problem text.

        Returns:
            Difficulty score between 0.0 (easy) and 1.0 (hard).
        """
        score = 0.0

        # Length-based heuristic
        word_count = len(problem.split())
        if word_count > 100:
            score += 0.2
        if word_count > 200:
            score += 0.1

        # Mathematical complexity indicators
        math_indicators = [
            r"\int", r"\sum", r"\prod", r"\lim",
            "prove", "show that", "find all",
            "maximum", "minimum", "optimize",
            "probability", "expected value",
        ]
        indicator_count = sum(
            1 for indicator in math_indicators
            if indicator.lower() in problem.lower()
        )
        score += min(0.4, indicator_count * 0.1)

        # Multi-step indicators
        step_indicators = ["and", "then", "given that", "such that", "where"]
        step_count = sum(
            1 for s in step_indicators
            if s.lower() in problem.lower()
        )
        score += min(0.3, step_count * 0.06)

        return min(1.0, score)

    def allocate_budget(self, difficulty: float) -> int:
        """Allocate token budget based on estimated difficulty.

        Args:
            difficulty: Difficulty score from estimate_difficulty().

        Returns:
            Token budget for this problem.
        """
        # Linear interpolation between min and max tokens
        budget = int(self.min_tokens + difficulty * (self.max_tokens - self.min_tokens))
        return max(self.min_tokens, min(self.max_tokens, budget))

    def force_budget(
        self,
        problems: List[str],
        generate_fn: Any,
        adaptive: bool = True,
    ) -> List[Dict[str, Any]]:
        """Apply budget forcing to a batch of problems.

        Args:
            problems: List of problem texts.
            generate_fn: Callable that takes (prompt, max_tokens) -> response.
            adaptive: If True, use adaptive budgets. If False, use max_tokens for all.

        Returns:
            List of result dicts with 'response', 'budget', 'difficulty'.
        """
        results = []
        for problem in problems:
            if adaptive:
                difficulty = self.estimate_difficulty(problem)
                budget = self.allocate_budget(difficulty)
            else:
                difficulty = 1.0
                budget = self.max_tokens

            response = generate_fn(problem, budget)

            results.append({
                "problem": problem,
                "response": response,
                "budget_allocated": budget,
                "difficulty_estimate": difficulty,
                "budget_used": len(response.split()) if isinstance(response, str) else 0,
            })

        return results

    def get_budget_stats(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute budget allocation statistics.

        Args:
            results: Results from force_budget().

        Returns:
            Stats dict with mean, min, max budgets and token savings.
        """
        budgets = [r["budget_allocated"] for r in results]
        used = [r["budget_used"] for r in results]

        return {
            "mean_budget": sum(budgets) / max(len(budgets), 1),
            "min_budget": min(budgets) if budgets else 0,
            "max_budget": max(budgets) if budgets else 0,
            "mean_used": sum(used) / max(len(used), 1),
            "total_savings_pct": (
                1.0 - sum(budgets) / (self.max_tokens * max(len(budgets), 1))
            ) * 100,
        }
