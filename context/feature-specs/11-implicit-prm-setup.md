# 11 — Implicit PRM Setup Specification

## ATRD: Adaptive Test-Time Reasoning Distillation

### 1. Purpose and Setup Order
This specification defines the implicit Process Reward Model (PRM) setup for GRPO reinforcement learning. The PRM scores intermediate reasoning steps without requiring a separate trained model.

> [!IMPORTANT]
> Read `09-sft-training-execution.md` before implementing. SFT checkpoint must exist.

---

## 2. Technical Components

### 2.1 Log-Ratio PRM Scorer (`src/training/grpo_trainer.py`)

#### 2.1.1 Core Mechanism
The implicit PRM computes step-level quality scores by comparing log-probabilities:

```
PRM_score(step_i) = sigmoid(log P_ref(step_i | context) - log P_θ(step_i | context))
```

Where:
- `P_ref` = reference policy (SFT checkpoint, frozen)
- `P_θ` = current policy (being trained by GRPO)
- Higher score → current policy produces more confident reasoning than reference

#### 2.1.2 Step Segmentation
```python
def segment_thinking_trace(completion: str) -> List[str]:
    """Split completion into reasoning steps."""
    # Split on sentence boundaries, equation lines, or step markers
    steps = re.split(r'(?<=\.) |(?<=\n)', completion)

    # Filter out empty steps and the final answer box
    steps = [s.strip() for s in steps if s.strip() and "\\boxed" not in s]
    return steps
```

### 2.2 Regex-Based PRM (Fallback)

If log-ratio computation introduces excessive memory overhead, use heuristic PRM:

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

### 2.3 Integration with GRPO Reward

```python
def compute_prm_guided_reward(
    completion: str,
    ground_truth: str,
    ref_log_probs: Optional[List[float]] = None,
    current_log_probs: Optional[List[float]] = None,
) -> float:
    """Compute composite reward with PRM scores."""
    # Final answer correctness (primary signal)
    answer_reward = 0.8 if check_answer(completion, ground_truth) else 0.0

    # Format compliance
    format_reward = 0.0
    if "\\boxed{" in completion: format_reward += 0.2
    if "<<thinking>>" in completion: format_reward += 0.1
    if "</thinking>>" in completion: format_reward += 0.1

    # PRM step scores (if log-probs available)
    prm_reward = 0.0
    if ref_log_probs and current_log_probs:
        steps = segment_thinking_trace(completion)
        step_scores = []
        for i, step in enumerate(steps):
            if i < len(ref_log_probs) and i < len(current_log_probs):
                ratio = ref_log_probs[i] - current_log_probs[i]
                step_score = torch.sigmoid(torch.tensor(ratio)).item()
                step_scores.append(step_score)
        prm_reward = sum(step_scores) / max(len(step_scores), 1) * 0.4
    else:
        # Fallback: heuristic PRM
        steps = segment_thinking_trace(completion)
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
- [ ] PRM scoring function implemented (log-ratio or heuristic fallback)
- [ ] Step segmentation correctly splits thinking traces
- [ ] PRM scores correlate with answer correctness (correct > incorrect)
- [ ] Redundancy penalty detects repeated phrases
- [ ] Format reward correctly identifies `\boxed{}` and `<<thinking>>` tokens
- [ ] Composite reward function produces values in [-1.0, 1.0] range
