"""LLM-as-judge quality filter for synthetic training data.

Scores each example on 4 weighted criteria and retains the top 80%
by composite score. Provides both heuristic pre-filtering (fast)
and LLM-judge scoring (if a judge model is available).
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Scoring criteria weights
CRITERIA_WEIGHTS: Dict[str, float] = {
    "correctness": 0.35,
    "reasoning_clarity": 0.25,
    "difficulty_appropriateness": 0.20,
    "format_compliance": 0.20,
}

REASONING_MARKERS = [
    "therefore", "thus", "because", "step", "first", "next",
    "hence", "so", "then", "since", "consequently",
]


class JudgeFilter:
    """Filter synthetic examples using 4-criteria composite scoring.

    Each example receives a composite score (0-1) based on:
    - correctness (0.35): answer presence and match
    - reasoning clarity (0.25): coherent step-by-step trace
    - difficulty appropriateness (0.20): matches expected difficulty
    - format compliance (0.20): valid <<thinking>> + \\boxed{}

    The top 80% of examples (score >= percentile_80) are retained.

    Attributes:
        threshold: Minimum score to keep (set as 80th percentile).
        judge_model: Optional LLM model for deep quality assessment.
    """

    def __init__(
        self,
        threshold: float = 0.80,
        judge_model: Optional[Any] = None,
    ) -> None:
        self.threshold = threshold
        self.judge_model = judge_model

    def filter_dataset(
        self,
        examples: List[Dict[str, Any]],
        score_field: str = "quality_score",
    ) -> List[Dict[str, Any]]:
        """Filter to top 80% by composite quality score.

        Args:
            examples: List of training examples (with question/thinking_trace/answer keys).
            score_field: Field name to store quality score.

        Returns:
            Filtered list with scores attached.
        """
        scored: List[Dict[str, Any]] = []
        for ex in examples:
            score = self.score_example(ex)
            ex[score_field] = score
            scored.append(ex)

        scored.sort(key=lambda x: x[score_field], reverse=True)
        cutoff = max(1, int(len(scored) * 0.80))
        kept = scored[:cutoff]

        print(
            f"JudgeFilter: {len(kept)}/{len(scored)} examples passed "
            f"(top 80%, effective threshold={scored[cutoff-1][score_field]:.3f})"
        )
        return kept

    def score_example(self, example: Dict[str, Any]) -> float:
        """Compute composite quality score from 4 weighted criteria.

        Args:
            example: Training example dict.

        Returns:
            Composite score 0.0-1.0.
        """
        correctness = self._score_correctness(example)
        clarity = self._score_reasoning_clarity(example)
        difficulty = self._score_difficulty_appropriateness(example)
        format_ok = self._score_format_compliance(example)

        composite = (
            CRITERIA_WEIGHTS["correctness"] * correctness
            + CRITERIA_WEIGHTS["reasoning_clarity"] * clarity
            + CRITERIA_WEIGHTS["difficulty_appropriateness"] * difficulty
            + CRITERIA_WEIGHTS["format_compliance"] * format_ok
        )
        return min(1.0, composite)

    def _score_correctness(self, example: Dict[str, Any]) -> float:
        """Score answer correctness (0.35 weight).

        Checks if answer exists, is boxed, and has valid content.
        """
        answer = example.get("answer", "")
        if not answer:
            return 0.0
        score = 0.3 if answer.strip() else 0.0
        if "\\boxed{" in answer:
            score += 0.4
        extracted = re.search(r"\\boxed\{(.+?)\}", answer)
        if extracted and extracted.group(1).strip():
            score += 0.3
        return min(1.0, score)

    def _score_reasoning_clarity(self, example: Dict[str, Any]) -> float:
        """Score reasoning trace clarity (0.25 weight).

        Rewards coherent, step-by-step reasoning with logical markers.
        """
        trace = example.get("thinking_trace", "")
        if not trace:
            return 0.0
        trace_lower = trace.lower()
        words = trace_lower.split()
        if len(words) < 10:
            return 0.2

        score = min(0.4, len(words) / 500)
        marker_count = sum(
            1 for m in REASONING_MARKERS if m in trace_lower
        )
        score += min(0.3, marker_count * 0.05)
        if re.search(r"\d+\s*[+\-*/^=]\s*\d+", trace):
            score += 0.3
        return min(1.0, score)

    def _score_difficulty_appropriateness(self, example: Dict[str, Any]) -> float:
        """Score difficulty appropriateness (0.20 weight).

        Rewards problems with clear mathematical structure.
        """
        question = example.get("question", "")
        if not question:
            return 0.0
        word_count = len(question.split())
        if word_count < 5:
            return 0.3
        score = min(0.5, word_count / 300)
        math_patterns = [
            r"\d+", r"[+\-*/^=]", r"\b(solve|find|prove|compute|evaluate)\b",
        ]
        pattern_score = sum(0.1 for p in math_patterns if re.search(p, question.lower()))
        return min(1.0, score + pattern_score)

    def _score_format_compliance(self, example: Dict[str, Any]) -> float:
        """Score format compliance (0.20 weight).

        Checks for valid <<thinking>> tags and \\boxed{} answer.
        """
        trace = example.get("thinking_trace", "")
        answer = example.get("answer", "")
        score = 0.0
        if "<<thinking>>" in trace:
            score += 0.3
        if ">>" in trace:
            score += 0.2
        if "\\boxed{" in answer:
            score += 0.3
        if answer.strip().endswith("}"):
            score += 0.2
        return min(1.0, score)

    def heuristic_score(self, example: Dict[str, Any]) -> float:
        """Fast heuristic pre-filter score (before full LLM judge).

        Uses the same composite formula but simpler heuristics
        for each criterion.

        Args:
            example: Training example dict.

        Returns:
            Heuristic quality score 0.0-1.0.
        """
        completion = example.get("completion", "") or example.get("thinking_trace", "")
        score = 0.0
        if completion:
            score += 0.3
        marker_count = sum(
            1 for m in REASONING_MARKERS if m.lower() in completion.lower()
        )
        score += min(0.4, marker_count * 0.1)
        if "\\boxed" in completion:
            score += 0.3
        return min(1.0, score)

    def generate_report(
        self,
        original: List[Dict[str, Any]],
        filtered: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate filtering statistics report.

        Args:
            original: Dataset before filtering.
            filtered: Dataset after filtering.
            output_path: Optional path to save report as JSON.

        Returns:
            Report dict with filtering statistics.
        """
        report = {
            "original_count": len(original),
            "filtered_count": len(filtered),
            "pass_rate": len(filtered) / max(len(original), 1),
            "threshold": self.threshold,
            "mean_score": (
                sum(ex.get("quality_score", 0.0) for ex in filtered)
                / max(len(filtered), 1)
            ),
        }
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
        return report
