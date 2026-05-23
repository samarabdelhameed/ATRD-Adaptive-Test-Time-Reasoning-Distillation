# 14 — Budget Forcing Specification

## ATRD: Adaptive Test-Time Reasoning Distillation

### 1. Purpose and Setup Order
This specification defines the budget forcing mechanism for **training data quality enhancement**. Since the competition evaluates only the LoRA adapter with fixed vLLM params (`temperature=0.0`, `max_tokens=7680`), budget forcing cannot be applied at submission inference time. Instead, budget forcing improves **training data quality** by allocating compute budget adaptively during data generation: hard problems get more reasoning tokens, easy problems get fewer, and hard problem failures trigger multi-stage refinement.

> [!IMPORTANT]
> Implement alongside `05-synthetic-data-generation.md`. Used during data generation, not submission.

---

## 2. Technical Components

### 2.1 Difficulty Estimator (`src/data/budget_forcer.py`)

#### 2.1.1 Heuristic-Based Estimation
```python
def estimate_difficulty(problem: str) -> float:
    """Estimate problem difficulty on 0–1 scale for token budget allocation."""
    score = 0.0

    # Length-based heuristic
    word_count = len(problem.split())
    if word_count > 100: score += 0.2
    if word_count > 200: score += 0.1

    # Mathematical complexity indicators
    math_indicators = [
        r"\int", r"\sum", r"\prod", r"\lim",
        "prove", "show that", "find all",
        "maximum", "minimum", "optimize",
        "probability", "expected value",
        "matrix", "determinant", "eigenvalue",
        "integral", "derivative", "gradient",
    ]
    score += min(0.4, sum(
        0.1 for ind in math_indicators
        if ind.lower() in problem.lower()
    ))

    # Multi-step indicators
    step_indicators = ["and", "then", "given that", "such that", "where", "therefore"]
    score += min(0.3, sum(
        0.06 for s in step_indicators
        if s.lower() in problem.lower()
    ))

    return min(1.0, score)
```

#### 2.1.2 Difficulty Tiers
| Difficulty | Score Range | Token Budget | Data Gen Strategy |
|-----------|-------------|--------------|-------------------|
| Easy | 0.00–0.30 | 512–1024 | Keep 1 generation |
| Medium | 0.30–0.65 | 2048–4096 | Keep 1 generation |
| Hard | 0.65–1.00 | 4096–7680 | Generate → evaluate → refine if wrong |

### 2.2 Multi-Stage Data Refinement

#### 2.2.1 Refinement Pipeline
For hard problems where the initial generation produces an incorrect answer, re-generate with stronger reasoning:

```python
def refine_hard_problem(
    problem: str,
    initial_completion: str,
    ground_truth: str,
    max_attempts: int = 3,
) -> str:
    """Regenerate hard problem solutions when answer is wrong."""
    for attempt in range(max_attempts):
        if check_answer(initial_completion, ground_truth):
            return initial_completion

        # Re-prompt with focus on error correction
        corrected = generate_with_prompt(
            problem=problem,
            system_instruction=(
                "The previous solution was incorrect. "
                "Double-check each step carefully. "
                "Break down the problem and verify your reasoning."
            ),
            max_tokens=7680,
        )
        initial_completion = corrected

    return initial_completion
```

#### 2.2.2 Data Generation with Budget Forcing
```python
def generate_training_data_with_budget(
    problem: str,
    ground_truth: str,
    difficulty: float,
) -> Dict[str, Any]:
    """Generate training example with difficulty-aware budget allocation."""
    budget = allocate_budget(difficulty)
    completion = generate(problem, max_tokens=budget)
    correct = check_answer(completion, ground_truth)

    # Hard problems: attempt refinement if wrong
    if difficulty > 0.65 and not correct:
        completion = refine_hard_problem(problem, completion, ground_truth)
        correct = check_answer(completion, ground_truth)

    return {
        "question": problem,
        "answer": ground_truth,
        "completion": completion,
        "difficulty": difficulty,
        "budget_allocated": budget,
        "correct": correct,
        "refined": difficulty > 0.65,
        "difficulty_tier": (
            "easy" if difficulty < 0.3
            else "medium" if difficulty < 0.65
            else "hard"
        ),
    }
```

### 2.3 Budget Allocation
```python
def allocate_budget(difficulty: float, min_tokens: int = 512, max_tokens: int = 7680) -> int:
    """Linear interpolation between min and max tokens based on difficulty."""
    budget = int(min_tokens + difficulty * (max_tokens - min_tokens))
    return max(min_tokens, min(max_tokens, budget))
```

### 2.4 Integration with Data Generation Pipeline

Budget forcing operates during the **data generation phase** (spec 05), not at inference time:

```python
# During synthetic data generation:
from src.data.budget_forcer import estimate_difficulty, generate_training_data_with_budget

all_problems = [...]  # List of (question, answer) pairs
dataset = []
for question, answer in all_problems:
    difficulty = estimate_difficulty(question)
    example = generate_training_data_with_budget(question, answer, difficulty)
    dataset.append(example)

# Statistics
hard_correct = sum(1 for d in dataset if d["difficulty_tier"] == "hard" and d["correct"])
hard_total = sum(1 for d in dataset if d["difficulty_tier"] == "hard")
print(f"Hard problem accuracy after refinement: {hard_correct}/{hard_total}")
```

---

## 3. Validation

### 3.1 Refinement Quality Check
```python
def validate_refinement_improvement(results: List[Dict]) -> Dict:
    """Check that refinement increases hard-problem accuracy."""
    hard_problems = [r for r in results if r["difficulty"] > 0.65]
    initial_correct = sum(
        1 for r in hard_problems
        if check_answer(r["completion_before_refinement"], r["answer"])
    )
    final_correct = sum(1 for r in hard_problems if r["correct"])
    return {
        "initial_accuracy": initial_correct / max(len(hard_problems), 1),
        "final_accuracy": final_correct / max(len(hard_problems), 1),
        "improvement": (final_correct - initial_correct) / max(len(hard_problems), 1),
    }
```

### 3.2 Budget Statistics
```python
def get_budget_stats(results: List[Dict]) -> Dict:
    budgets = [r["budget_allocated"] for r in results]
    return {
        "mean_budget": sum(budgets) / len(budgets),
        "min_budget": min(budgets),
        "max_budget": max(budgets),
        "total_savings_pct": (
            1.0 - sum(budgets) / (7680 * len(budgets))
        ) * 100,
    }
```

---

## 4. Exit Quality Gate
- [ ] Difficulty estimator correctly classifies easy/medium/hard problems
- [ ] Hard-problem refinement improves accuracy ≥ 10% over initial generation
- [ ] Easy problems use ≤ 1024 tokens (compute saved for hard problems)
- [ ] budget forcing integrated into data generation pipeline (spec 05)
- [ ] Token budget statistics logged to `logs/budget_stats.json`
- [ ] Training dataset includes `difficulty` and `refined` metadata columns
- [ ] No impact on submission (budget forcing is data-gen-only, not inference-time)
