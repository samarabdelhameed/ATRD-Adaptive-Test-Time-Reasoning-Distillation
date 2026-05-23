"""Budget forcing for adaptive token allocation during data generation.

Provides difficulty estimation, token budget allocation, and refinement
validation for the ATRD pipeline's data generation phase. Designed for
data-generation (not inference-time), since the competition fixes
temperature=0.0 and max_tokens=7680 at inference.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Union

from src.evaluation.metric import answers_equivalent, extract_boxed_answer

_DEFAULT_MIN_TOKENS: int = 512
_DEFAULT_MAX_TOKENS: int = 7680
_HARD_DIFFICULTY_THRESHOLD: float = 0.65

_generate_fn: Optional[Callable[..., str]] = None


def set_generate_backend(generate_fn: Optional[Callable[..., str]] = None) -> None:
    """Set or clear the generation function used by generate()."""
    global _generate_fn
    _generate_fn = generate_fn


def reset_generate_backend() -> None:
    """Clear the generation backend (for tests and session resets)."""
    set_generate_backend(None)


def generate(prompt: str, max_tokens: int = 512, **kwargs: Any) -> str:
    """Generate text using the configured backend."""
    if _generate_fn is None:
        raise RuntimeError(
            "No generation backend configured. Call set_generate_backend() first."
        )
    return _generate_fn(prompt, max_tokens=max_tokens, **kwargs)


def estimate_difficulty(problem: str) -> float:
    """Estimate problem difficulty on a 0-1 scale using heuristic features."""
    score = 0.0
    word_count = len(problem.split())

    if word_count > 100:
        score += 0.2
    if word_count > 200:
        score += 0.1

    math_indicators = [
        r"\int", r"\sum", r"\prod", r"\lim",
        "integral", "derivative", "gradient",
        "prove", "show", "find", "solve", "compute", "evaluate", "calculate",
        "maximum", "minimum", "optimize",
        "probability", "expected value",
    ]
    indicator_count = sum(
        1 for indicator in math_indicators
        if indicator.lower() in problem.lower()
    )
    score += min(0.4, indicator_count * 0.1)

    step_indicators = ["and", "then", "given that", "such that", "where"]
    step_count = sum(1 for s in step_indicators if s.lower() in problem.lower())
    score += min(0.3, step_count * 0.06)

    return min(1.0, score)


def allocate_budget(difficulty: float) -> int:
    """Allocate token budget based on estimated difficulty (512–7680)."""
    budget = int(
        _DEFAULT_MIN_TOKENS + difficulty * (_DEFAULT_MAX_TOKENS - _DEFAULT_MIN_TOKENS)
    )
    return max(_DEFAULT_MIN_TOKENS, min(_DEFAULT_MAX_TOKENS, budget))


def difficulty_tier(difficulty: float) -> str:
    """Map difficulty score to easy / medium / hard tier."""
    if difficulty < 0.3:
        return "easy"
    if difficulty < _HARD_DIFFICULTY_THRESHOLD:
        return "medium"
    return "hard"


def check_answer(response: str, expected: str) -> bool:
    """Check if a response contains the correct boxed answer."""
    extracted = extract_boxed_answer(response)
    if not extracted:
        match = re.search(r"\\boxed\{(.+?)\}", response)
        extracted = match.group(1).strip() if match else response.strip()
    return answers_equivalent(extracted, expected)


def refine_hard_problem(
    problem: str,
    initial_completion: str,
    ground_truth: str,
    max_attempts: int = 3,
) -> str:
    """Regenerate hard problem solutions when the initial answer is wrong."""
    completion = initial_completion
    for _ in range(max_attempts):
        if check_answer(completion, ground_truth):
            return completion

        correction_prompt = (
            f"{problem}\n\n"
            "The previous solution was incorrect. Double-check each step carefully. "
            "Break down the problem and verify your reasoning. "
            "Provide a complete thinking trace inside <<thinking>>...</thinking>> "
            "and the final answer in \\boxed{}."
        )
        completion = generate(correction_prompt, max_tokens=_DEFAULT_MAX_TOKENS)

    return completion


def generate_training_data_with_budget(
    problem: Union[str, Dict[str, Any]],
    ground_truth: Optional[str] = None,
    difficulty: Optional[float] = None,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Generate one training example with difficulty-aware budget allocation.

    Accepts either:
      - (problem: str, ground_truth: str, difficulty: float)
      - a list of dicts with 'question' and 'answer' keys
      - a single dict with 'question' and 'answer' keys
    """
    if isinstance(problem, list):
        return [
            generate_training_data_with_budget(item)  # type: ignore[arg-type]
            for item in problem
        ]

    if isinstance(problem, dict):
        question = problem.get("question", problem.get("problem", ""))
        answer = problem.get("answer", ground_truth or "")
        diff = problem.get("difficulty", difficulty)
        if diff is None:
            diff = estimate_difficulty(question)
        return _generate_single_example(str(question), str(answer), float(diff))

    if ground_truth is None:
        raise ValueError("ground_truth is required when problem is a string")

    if difficulty is None:
        difficulty = estimate_difficulty(str(problem))

    return _generate_single_example(str(problem), str(ground_truth), float(difficulty))


def _generate_single_example(
    question: str,
    ground_truth: str,
    difficulty: float,
) -> Dict[str, Any]:
    """Internal: build one budget-forced training record."""
    budget = allocate_budget(difficulty)
    completion = generate(question, max_tokens=budget)
    correct = check_answer(completion, ground_truth)
    refined = False

    if difficulty > _HARD_DIFFICULTY_THRESHOLD and not correct:
        completion = refine_hard_problem(question, completion, ground_truth)
        correct = check_answer(completion, ground_truth)
        refined = True

    return {
        "question": question,
        "answer": ground_truth,
        "completion": completion,
        "difficulty": difficulty,
        "budget_allocated": budget,
        "correct": correct,
        "refined": refined,
        "difficulty_tier": difficulty_tier(difficulty),
    }


def get_budget_stats(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute budget allocation statistics from a list of results."""
    if not results:
        return {"min_budget": 0.0, "max_budget": 0.0, "mean_budget": 0.0}

    budgets = [r["budget_allocated"] for r in results]
    mean_budget = sum(budgets) / len(budgets)
    max_tokens = _DEFAULT_MAX_TOKENS
    savings_pct = (1.0 - mean_budget / max_tokens) * 100 if max_tokens else 0.0

    return {
        "min_budget": float(min(budgets)),
        "max_budget": float(max(budgets)),
        "mean_budget": mean_budget,
        "total_savings_pct": savings_pct,
    }


def validate_refinement_improvement(refinements: List[Dict[str, Any]]) -> Dict[str, float]:
    """Validate that budget-forced refinements improve correctness."""
    if not refinements:
        return {"improvement": 0.0, "total": 0, "improved": 0}

    improved_count = sum(1 for ref in refinements if ref.get("correct", False))
    total = len(refinements)

    return {
        "improvement": improved_count / max(total, 1),
        "total": total,
        "improved": improved_count,
    }
