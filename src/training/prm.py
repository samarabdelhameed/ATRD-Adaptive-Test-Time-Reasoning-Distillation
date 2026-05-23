"""Implicit Process Reward Model (PRM) scoring for reasoning steps.

Provides heuristic-based (zero additional memory) and log-ratio based (optional)
process reward signals to score intermediate reasoning traces during GRPO.
"""

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch


def segment_thinking_trace(completion: str) -> List[str]:
    """Split completion into distinct reasoning steps.

    Args:
        completion: The full generated completion text.

    Returns:
        List of segmented reasoning steps.
    """
    # Split by period followed by space, or newline
    steps = re.split(r"(?<=\.) |(?<=\n)", completion)
    steps = [s.strip() for s in steps if s.strip() and "\\boxed" not in s]
    return steps


def detect_redundancy(completion: str) -> bool:
    """Check for repetitive sentences or phrases (loops) in the completion.

    Args:
        completion: The full completion text.

    Returns:
        True if excessive redundancy or looping is detected.
    """
    lines = [line.strip().lower() for line in completion.split("\n") if line.strip()]
    # Check for exact 3-line repetition loop
    for i in range(len(lines) - 2):
        if lines[i] == lines[i+1] == lines[i+2]:
            return True
            
    # Check for word-level loops (e.g. repeated n-grams)
    words = [w.lower() for w in re.findall(r"\b\w+\b", completion)]
    if len(words) > 15:
        # Check windowed redundancy
        for size in [3, 4, 5]:
            for i in range(len(words) - size * 3):
                chunk1 = words[i : i + size]
                chunk2 = words[i + size : i + size * 2]
                chunk3 = words[i + size * 2 : i + size * 3]
                if chunk1 == chunk2 == chunk3:
                    return True
    return False


def heuristic_step_score(step: str) -> float:
    """Score a single reasoning step using heuristics.

    Evaluates step quality based on numerical transition indicators,
    logical connectors, valid equations, and step repetition.

    Args:
        step: A single segmented reasoning step.

    Returns:
        Score between 0.0 and 1.0.
    """
    score = 0.0

    # 1. Check for mathematical transitions (e.g., = or → or relations)
    if re.search(r"[=→<>≤≥]", step):
        score += 0.2

    # 2. Check for logical connectors
    connectors = ["therefore", "thus", "because", "hence", "so", "then", "since", "implies"]
    if any(c in step.lower() for c in connectors):
        score += 0.2

    # 3. Check for valid equations or arithmetic expressions
    if re.search(r"[\d.]+", step) and re.search(r"[+\-*/^=]", step):
        score += 0.3

    # 4. Penalize word repetition (low lexical diversity in the step indicates looping)
    words = [w.lower() for w in re.findall(r"\b\w+\b", step)]
    if len(words) > 4:
        diversity = len(set(words)) / len(words)
        if diversity < 0.4:
            score -= 0.3

    return max(0.0, min(1.0, score))


def get_log_prob(model: Any, text: str, tokenizer: Any) -> float:
    """Helper to compute model log probability for a given text."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        # Cross entropy loss is negative log likelihood
        loss = outputs.loss.item()
        return -loss * inputs["input_ids"].shape[1]


def compute_log_ratio_score(
    step: str,
    context: str,
    ref_model: Any,
    current_model: Any,
    tokenizer: Any,
) -> Optional[float]:
    """Compute PRM score via log-probability ratio.

    Returns None if computation fails (e.g. OOM).

    Args:
        step: The reasoning step to score.
        context: Prior context prefix.
        ref_model: Frozen reference model.
        current_model: Current policy model.
        tokenizer: Tokenizer.

    Returns:
        Sigmoid probability score or None if error.
    """
    try:
        ref_log_prob = get_log_prob(ref_model, context + step, tokenizer)
        cur_log_prob = get_log_prob(current_model, context + step, tokenizer)
        ratio = ref_log_prob - cur_log_prob
        
        # Calculate sigmoid probability
        sigmoid_val = 1.0 / (1.0 + torch.exp(torch.tensor(-ratio)).item())
        return sigmoid_val
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
        raise


def check_answer(completion: str, ground_truth: str) -> bool:
    """Check if predicted answer matches expected within tolerance."""
    # Extractor
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    match = re.search(pattern, completion)
    predicted = match.group(1).strip() if match else ""
    
    try:
        return abs(float(predicted) - float(ground_truth)) <= 0.01
    except (ValueError, TypeError):
        return predicted.strip() == ground_truth.strip()


def compute_prm_guided_reward(
    completion: str,
    ground_truth: str,
    ref_model: Optional[Any] = None,
    current_model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    use_log_ratio: bool = False,
) -> float:
    """Compute composite reward with PRM scores.

    Composite components:
    - Answer correctness: 0.8
    - Format compliance: +0.2 for \boxed{}, +0.1 for <<thinking>>, +0.1 for </thinking>>
    - PRM reasoning quality: 0.4 * average step score
    - Redundancy penalty: -0.3

    Total clamped to [-1.0, 1.0].

    Args:
        completion: The model completion string.
        ground_truth: The target ground truth answer.
        ref_model: Reference model for log-ratio.
        current_model: Current policy model.
        tokenizer: Tokenizer.
        use_log_ratio: Whether to attempt log-ratio scoring.

    Returns:
        Reward score clamped to [-1.0, 1.0].
    """
    # 1. Correctness reward
    answer_reward = 0.8 if check_answer(completion, ground_truth) else 0.0

    # 2. Format compliance reward
    format_reward = 0.0
    if "\\boxed{" in completion:
        format_reward += 0.2
    if "<<thinking>>" in completion:
        format_reward += 0.1
    if "</thinking>>" in completion:
        format_reward += 0.1

    # 3. Process Reward scoring
    steps = segment_thinking_trace(completion)
    step_scores = []
    
    if use_log_ratio and ref_model is not None and current_model is not None and tokenizer is not None:
        for step in steps:
            score = compute_log_ratio_score(step, "", ref_model, current_model, tokenizer)
            if score is not None:
                step_scores.append(score)
                
    # Fallback to heuristic scoring if list is empty or log-ratio skipped
    if not step_scores:
        step_scores = [heuristic_step_score(s) for s in steps]

    prm_reward = (sum(step_scores) / max(len(step_scores), 1)) * 0.4

    # 4. Redundancy penalty
    redundancy_penalty = -0.3 if detect_redundancy(completion) else 0.0

    total = answer_reward + format_reward + prm_reward + redundancy_penalty
    return max(-1.0, min(1.0, total))
