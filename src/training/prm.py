import re
from pathlib import Path
from typing import List, Optional

try:
    import torch
except ImportError:
    torch = None


def heuristic_step_score(step: str) -> float:
    score = 0.0
    if re.search(r'[=→<>≤≥]', step):
        score += 0.2
    connectors = ["therefore", "thus", "because", "hence", "so", "then"]
    if any(c in step.lower() for c in connectors):
        score += 0.2
    if re.search(r'[\d.]+', step) and re.search(r'[+\-*/^=]', step):
        score += 0.3
    words = step.lower().split()
    if len(set(words)) / max(len(words), 1) < 0.4:
        score -= 0.3
    return max(0.0, min(1.0, score))


def segment_thinking_trace(completion: str) -> List[str]:
    steps = re.split(r'(?<=\.) |(?<=\n)', completion)
    steps = [s.strip() for s in steps if s.strip() and "\\boxed" not in s]
    return steps


if torch is not None:

    def get_log_prob(model: torch.nn.Module, text: str, tokenizer) -> float:
        inputs = tokenizer(text, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        if logits.size(1) <= 1:
            return 0.0
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.mean().item()

    def compute_log_ratio_score(
        step: str,
        context: str,
        ref_model: torch.nn.Module,
        current_model: torch.nn.Module,
        tokenizer,
    ) -> Optional[float]:
        try:
            ref_log_prob = get_log_prob(ref_model, context + step, tokenizer)
            cur_log_prob = get_log_prob(current_model, context + step, tokenizer)
            ratio = ref_log_prob - cur_log_prob
            return torch.sigmoid(torch.tensor(ratio)).item()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return None
            raise

else:

    def get_log_prob(*args, **kwargs) -> float:
        raise NotImplementedError("torch is required for get_log_prob")

    def compute_log_ratio_score(*args, **kwargs) -> Optional[float]:
        return None


def _extract_boxed_answer(text: str) -> str:
    pattern = r"\\boxed\{([^}]*)\}"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def check_answer(completion: str, ground_truth: str) -> bool:
    extracted = _extract_boxed_answer(completion)
    tolerance = 0.01
    try:
        return abs(float(extracted) - float(ground_truth)) <= tolerance
    except (ValueError, TypeError):
        return extracted.strip() == ground_truth.strip()


def detect_redundancy(completion: str) -> bool:
    lines = [l for l in completion.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    repeat_count = 0
    for i in range(2, len(lines)):
        if lines[i].strip() == lines[i - 2].strip():
            repeat_count += 1
    return repeat_count >= 2


def compute_prm_guided_reward(
    completion: str,
    ground_truth: str,
    ref_model=None,
    current_model=None,
    tokenizer=None,
    use_log_ratio: bool = False,
) -> float:
    answer_reward = 0.8 if check_answer(completion, ground_truth) else 0.0
    format_reward = 0.0
    if "\\boxed{" in completion:
        format_reward += 0.2
    if "<<thinking>>" in completion:
        format_reward += 0.1
    if "</thinking>>" in completion:
        format_reward += 0.1
    steps = segment_thinking_trace(completion)
    if not steps:
        prm_reward = 0.0
    else:
        if use_log_ratio and ref_model is not None and current_model is not None and tokenizer is not None:
            step_scores = []
            for step in steps:
                score = compute_log_ratio_score(step, "", ref_model, current_model, tokenizer)
                if score is not None:
                    step_scores.append(score)
            if not step_scores:
                step_scores = [heuristic_step_score(s) for s in steps]
        else:
            step_scores = [heuristic_step_score(s) for s in steps]
        prm_reward = (sum(step_scores) / max(len(step_scores), 1)) * 0.4
    redundancy_penalty = -0.3 if detect_redundancy(completion) else 0.0
    total = answer_reward + format_reward + prm_reward + redundancy_penalty
    return max(-1.0, min(1.0, total))


def test_prm_correlation(
    validation_set_path: str = "data/validation_set.jsonl",
    generate_fn=None,
) -> None:
    path = Path(validation_set_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Validation set not found at {path}. "
            "Provide real benchmark data to test PRM correlation."
        )
    import json
    validation_set = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                validation_set.append(json.loads(line))
    if generate_fn is None:
        raise ValueError(
            "generate_fn must be provided (e.g., vllm_engine.generate_single). "
            "Cannot compute PRM correlation without a real model."
        )
    correct_scores = []
    incorrect_scores = []
    for example in validation_set:
        question = example.get("question", "")
        answer = example.get("answer", "")
        if not question or not answer:
            continue
        completion = generate_fn(question)
        prm_score = compute_prm_guided_reward(completion, answer)
        if check_answer(completion, answer):
            correct_scores.append(prm_score)
        else:
            incorrect_scores.append(prm_score)
    mean_correct = sum(correct_scores) / max(len(correct_scores), 1)
    mean_incorrect = sum(incorrect_scores) / max(len(incorrect_scores), 1)
    assert mean_correct > mean_incorrect, "PRM scores not correlated with correctness"
    print(f"PRM Correlation: Correct={mean_correct:.3f}, Incorrect={mean_incorrect:.3f}")
