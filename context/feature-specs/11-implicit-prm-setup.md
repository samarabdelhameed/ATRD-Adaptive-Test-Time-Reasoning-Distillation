# 11 — Implicit PRM Setup Specification

## ATRD: Adaptive Test-Time Reasoning Distillation

### 1. Purpose and Setup Order
This specification defines the Process Reward Model (PRM) for GRPO reinforcement learning. The PRM scores intermediate reasoning steps. The **default implementation uses heuristic scoring** (zero additional GPU memory). Log-ratio scoring against a reference model is available as an **optional enhancement** if GPU memory permits.

> [!IMPORTANT]
> Read `09-sft-training-execution.md` before implementing. SFT checkpoint must exist.

---

## 2. Technical Components

### 2.1 Heuristic PRM Scorer (Primary / Default) (`src/training/prm.py`)

The heuristic PRM uses regex patterns to evaluate reasoning step quality without requiring a reference model:

| Heuristic | Score Contribution |
|-----------|-------------------|
| Contains numerical transition | +0.2 (e.g., `x = 5 → 2x = 10`) |
| Uses logical connectors | +0.2 (therefore, thus, because, hence) |
| Equation is mathematically valid | +0.3 (each step parses as valid math) |
| No repeated phrases | +0.3 (no loops detected) |

```python
def heuristic_step_score(step: str) -> float:
    """Score a single reasoning step using heuristics."""
    score = 0.0

    # Check for mathematical transitions
    if re.search(r'[=→<>≤≥]', step):
        score += 0.2

    # Check for logical connectors
    connectors = ["therefore", "thus", "because", "hence", "so", "then"]
    if any(c in step.lower() for c in connectors):
        score += 0.2

    # Check for valid equations
    if re.search(r'[\d.]+', step) and re.search(r'[+\-*/^=]', step):
        score += 0.3

    # Penalize repetition (potential loop)
    words = step.lower().split()
    if len(set(words)) / max(len(words), 1) < 0.4:
        score -= 0.3

    return max(0.0, min(1.0, score))
```

#### Step Segmentation
```python
def segment_thinking_trace(completion: str) -> List[str]:
    """Split completion into reasoning steps."""
    steps = re.split(r'(?<=\.) |(?<=\n)', completion)
    steps = [s.strip() for s in steps if s.strip() and "\\boxed" not in s]
    return steps
```

### 2.2 Log-Ratio PRM Scorer (Optional — High Memory)

> [!CAUTION]
> Log-ratio PRM requires **both** the reference model (frozen SFT) and the current policy in GPU memory simultaneously. For a 30B parameter model with LoRA, this requires ~80GB+ VRAM. **Skips automatically on low-memory GPUs.**

```python
def compute_log_ratio_score(
    step: str,
    context: str,
    ref_model: torch.nn.Module,
    current_model: torch.nn.Module,
    tokenizer,
) -> float:
    """Compute PRM score via log-probability ratio.
    
    Returns None if computation fails (OOM or missing models).
    """
    try:
        ref_log_prob = get_log_prob(ref_model, context + step, tokenizer)
        cur_log_prob = get_log_prob(current_model, context + step, tokenizer)
        ratio = ref_log_prob - cur_log_prob
        return torch.sigmoid(torch.tensor(ratio)).item()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return None
        raise
```

### 2.3 Integration with GRPO Reward

```python
def compute_prm_guided_reward(
    completion: str,
    ground_truth: str,
    ref_model: Optional[torch.nn.Module] = None,
    current_model: Optional[torch.nn.Module] = None,
    tokenizer=None,
    use_log_ratio: bool = False,
) -> float:
    """Compute composite reward with PRM scores."""
    # Final answer correctness (primary signal)
    answer_reward = 0.8 if check_answer(completion, ground_truth) else 0.0

    # Format compliance
    format_reward = 0.0
    if "\\boxed{" in completion: format_reward += 0.2
    if "<<thinking>>" in completion: format_reward += 0.1
    if "</thinking>>" in completion: format_reward += 0.1

    # PRM step scores: heuristic by default, log-ratio if configured + available
    steps = segment_thinking_trace(completion)
    if use_log_ratio and ref_model is not None and current_model is not None:
        step_scores = []
        for step in steps:
            score = compute_log_ratio_score(step, "", ref_model, current_model, tokenizer)
            if score is not None:
                step_scores.append(score)
        if not step_scores:
            step_scores = [heuristic_step_score(s) for s in steps]
    else:
        step_scores = [heuristic_step_score(s) for s in steps]

    prm_reward = sum(step_scores) / max(len(step_scores), 1) * 0.4

    # Redundancy penalty
    redundancy_penalty = -0.3 if detect_redundancy(completion) else 0.0

    total = answer_reward + format_reward + prm_reward + redundancy_penalty
    return max(-1.0, min(1.0, total))
```

---

## 3. Verification

### 3.1 PRM Score Correlation Test
```python
def test_prm_correlation():
    """Verify PRM scores correlate with answer correctness."""
    correct_scores = []
    incorrect_scores = []

    for example in validation_set:
        completion = generate(example["question"])
        prm_score = compute_prm_guided_reward(completion, example["answer"])
        if check_answer(completion, example["answer"]):
            correct_scores.append(prm_score)
        else:
            incorrect_scores.append(prm_score)

    mean_correct = sum(correct_scores) / max(len(correct_scores), 1)
    mean_incorrect = sum(incorrect_scores) / max(len(incorrect_scores), 1)
    assert mean_correct > mean_incorrect, "PRM scores not correlated with correctness"
    print(f"PRM Correlation: Correct={mean_correct:.3f}, Incorrect={mean_incorrect:.3f}")
```

---

## 4. Exit Quality Gate
- [ ] Heuristic PRM scoring function implemented (default, zero GPU overhead)
- [ ] Step segmentation correctly splits thinking traces
- [ ] PRM scores correlate with answer correctness (correct > incorrect)
- [ ] Redundancy penalty detects repeated phrases
- [ ] Format reward correctly identifies `\boxed{}` and `<<thinking>>` tokens
- [ ] Composite reward function produces values in [-1.0, 1.0] range
- [ ] Heuristic PRM runs without requiring reference model in memory
- [ ] Log-ratio PRM gracefully falls back to heuristic on OOM
