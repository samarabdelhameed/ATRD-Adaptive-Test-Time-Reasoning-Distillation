# 14 — Budget Forcing Specification

## ATRD: Adaptive Test-Time Reasoning Distillation

### 1. Purpose and Setup Order
This specification defines the budget forcing mechanism for test-time compute scaling. The system monitors the generated token stream and dynamically adjusts reasoning depth based on problem difficulty.

> [!IMPORTANT]
> Read `12-grpo-training-loop.md` before implementing. GRPO checkpoint must exist.

---

## 2. Technical Components

### 2.1 Difficulty Estimator (`src/inference/budget_forcer.py`)

#### 2.1.1 Heuristic-Based Estimation
```python
def estimate_difficulty(problem: str) -> float:
    """Estimate problem difficulty on 0–1 scale."""
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
| Difficulty | Score Range | Token Budget | Strategy |
|-----------|-------------|--------------|----------|
| Easy | 0.00–0.30 | 256–512 | Force termination early |
| Medium | 0.30–0.65 | 1024–4096 | Normal inference |
| Hard | 0.65–1.00 | 4096–7680 | "Wait" injection to extend |

### 2.2 Token-Level Budget Forcer

#### 2.2.1 Core Algorithm
```python
def force_budget(
    problem: str,
    generate_fn: Callable,
    max_tokens: int = 7680,
    min_tokens: int = 256,
) -> str:
    """
    Generate with budget forcing.
    
    1. Estimate difficulty
    2. Allocate token budget
    3. Generate tokens one by one
    4. Monitor for </thinking> token
    5. If hard and budget remaining: inject "Wait" to extend
    6. If easy or budget exceeded: force </thinking> and final answer
    """
    difficulty = estimate_difficulty(problem)
    budget = allocate_budget(difficulty, min_tokens, max_tokens)
    
    if difficulty < 0.3:
        # Easy: force short thinking
        return generate_with_limit(problem, max_tokens=budget)
    elif difficulty > 0.65:
        # Hard: enable Wait injection
        return generate_with_wait_injection(problem, budget)
    else:
        # Medium: normal generation
        return generate_fn(problem, max_tokens=budget)
```

#### 2.2.2 Wait Token Injection
```python
def generate_with_wait_injection(
    prompt: str,
    budget: int,
    max_wait_injections: int = 3,
) -> str:
    """Generate with up to 3 'Wait' injections when </thinking> detected early."""
    
    thinking_tokens = 0
    wait_injections = 0
    completed = ""

    while thinking_tokens < budget:
        # Generate next token
        token = generate_next_token(completed)
        
        # Check for end-of-thinking
        if token == "</thinking>" and thinking_tokens < budget * 0.8:
            if wait_injections < max_wait_injections:
                # Inject "Wait" to extend reasoning
                token = "Wait, let me verify this step...\\n"
                wait_injections += 1
                print(f"  Budget forcing: Wait injection #{wait_injections}")
            else:
                break  # Max injections reached

        completed += token
        thinking_tokens += 1

    return completed
```

### 2.3 Budget Allocation
```python
def allocate_budget(difficulty: float, min_tokens: int, max_tokens: int) -> int:
    """Linear interpolation between min and max tokens."""
    budget = int(min_tokens + difficulty * (max_tokens - min_tokens))
    return max(min_tokens, min(max_tokens, budget))
```

### 2.4 vLLM Integration (`src/inference/vllm_engine.py`)

```python
from vllm import LLM, SamplingParams

engine = LLM(
    model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
    max_model_len=8192,
    gpu_memory_utilization=0.85,
    enable_lora=True,
    max_lora_rank=32,
    trust_remote_code=True,
)

sampling_params = SamplingParams(
    temperature=0.0,      # Competition-mandated
    max_tokens=7680,       # Competition-mandated
    top_p=1.0,             # Competition-mandated
)
```

---

## 3. Validation

### 3.1 Stratified Evaluation
| Difficulty Bin | Without Budget Forcing | With Budget Forcing | Expected Delta |
|---------------|----------------------|-------------------|----------------|
| Easy (bottom 20%) | Accuracy_baseline | Accuracy_forced | Degradation ≤ 2% |
| Hard (top 20%) | Accuracy_baseline | Accuracy_forced | Improvement ≥ 5% |
| Overall | Accuracy_baseline | Accuracy_forced | Net ≥ +3% |

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
- [ ] Wait injection triggers on hard problems when `</thinking>` appears early
- [ ] Easy problems terminate early (≤ 512 tokens)
- [ ] Hard problems get extended reasoning (Wait up to 3x)
- [ ] Easy problem accuracy degrades ≤ 2% (no regression)
- [ ] Hard problem accuracy improves ≥ 5%
- [ ] Overall net accuracy improvement ≥ 3%
- [ ] vLLM compatible — runs with `temperature=0.0`, `max_tokens=7680`
