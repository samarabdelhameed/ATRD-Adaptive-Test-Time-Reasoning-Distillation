"""Failure-grounded synthetic data generation for SFT training.

Generates synthetic training examples targeting specific failure modes
identified during baseline evaluation. Uses a teacher model API
(DeepSeek-R1 / Qwen3-235B) to produce corrected reasoning traces.
"""

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests


SYSTEM_PROMPT = """You are an expert mathematics tutor. Your task is to generate
variations of reasoning problems that target specific failure modes.

The student model (Nemotron-3-Nano-30B) struggles with: {failure_description}

Generate {num_problems} problems that test this specific weakness.
Each problem must include:
1. A clear question
2. A complete step-by-step thinking trace inside <<thinking>>...</thinking>>
3. A final answer in \\boxed{{}} format
"""

FAILURE_GROUNDED_PROMPT = """You are an expert math tutor. The student model failed on this problem:
{problem_example}

The failure reason was: {failure_category}: {failure_description}.

Generate {batch_size} similar problems of comparable difficulty that test
the same reasoning weakness. For each problem:
- Write a clear question
- Provide a complete step-by-step solution inside <<thinking>>...</thinking>>
- End with the answer in \\boxed{{}}

IMPORTANT: The thinking trace must be detailed and show complete reasoning.
"""


@dataclass
class GeneratorConfig:
    """Configuration for synthetic data generation API calls."""
    batch_size: int = 10
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.95
    timeout: int = 120
    retry_base_delay: float = 2.0


FAILURE_MODE_DESCRIPTIONS: Dict[str, str] = {
    # Baseline taxonomy (notebook 01 / METHODOLOGY)
    "no_answer": (
        "The model fails to produce a final answer in \\boxed{} format."
    ),
    "incomplete": (
        "The model stops before completing the required reasoning trace."
    ),
    "format_error": (
        "The model produces malformed \\boxed{} notation or missing "
        "<<thinking>> tags."
    ),
    "wrong_answer": (
        "The model produces a final answer that is mathematically incorrect."
    ),
    "proof_structure": (
        "The reasoning lacks logical step-by-step structure or skips key steps."
    ),
    # Legacy / API aliases (backward compatible)
    "reasoning_loop": (
        "The model gets stuck repeating assertions without progressing."
    ),
    "format_violation": (
        "The completion lacks the required \\boxed{} answer block "
        "or <<thinking>> tags."
    ),
    "early_termination": (
        "The model halts generation before arriving at a final value."
    ),
    "calculation_error": (
        "The model makes arithmetic errors in intermediate reasoning."
    ),
    "misinterpretation": (
        "The model misreads the problem and solves a different question."
    ),
}

# Map notebook tags to canonical keys for description lookup
FAILURE_MODE_ALIASES: Dict[str, str] = {
    "format_violation": "format_error",
    "early_termination": "incomplete",
    "calculation_error": "wrong_answer",
    "reasoning_loop": "proof_structure",
}


class SyntheticGenerator:
    """Generate failure-grounded synthetic data for SFT training.

    Attributes:
        seed: Random seed for reproducibility.
        config: Competition parameters loaded from configs/.
        gen_config: Generator API call configuration.
        output_dir: Directory to save generated datasets.
        api_key: API key for teacher model provider.
        api_base: Base URL for the API provider.
        primary_model: Primary teacher model name.
        fallback_model: Fallback teacher model name.
    """

    def __init__(
        self,
        config_path: str = "configs/competition_params.json",
        gen_config: Optional[GeneratorConfig] = None,
        output_dir: str = "data/synthetic",
        seed: int = 42,
        api_key: Optional[str] = None,
        api_base: str = "https://api.together.xyz/v1",
        primary_model: str = "deepseek-ai/DeepSeek-R1",
        fallback_model: str = "Qwen/Qwen3-235B-A22B",
    ) -> None:
        self.seed = seed
        self.gen_config = gen_config or GeneratorConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key
        self.api_base = api_base
        self.primary_model = primary_model
        self.fallback_model = fallback_model

        random.seed(seed)
        np.random.seed(seed)

        with open(config_path, "r") as f:
            self.config = json.load(f)

    def generate_per_failure_mode(
        self,
        failure_examples: Dict[str, List[Dict[str, Any]]],
        problems_per_mode: int = 2000,
    ) -> List[Dict[str, Any]]:
        """Generate synthetic problems for each failure mode.

        Args:
            failure_examples: Dict mapping failure_mode_tag to list of
                example problems that exhibited that failure.
            problems_per_mode: Number of problems to generate per mode.

        Returns:
            List of synthetic problem dicts matching the output schema.
        """
        all_problems: List[Dict[str, Any]] = []
        for mode, examples in failure_examples.items():
            canonical = FAILURE_MODE_ALIASES.get(mode, mode)
            description = FAILURE_MODE_DESCRIPTIONS.get(
                canonical,
                FAILURE_MODE_DESCRIPTIONS.get(
                    mode, "General reasoning difficulty."
                ),
            )
            n = min(problems_per_mode, len(examples) * 50)
            batch = self._generate_for_mode(mode, description, examples, n)
            all_problems.extend(batch)
            print(f"  [{mode}] Generated {len(batch)} problems")
        return all_problems

    def _generate_for_mode(
        self,
        mode: str,
        description: str,
        examples: List[Dict[str, Any]],
        target_count: int,
    ) -> List[Dict[str, Any]]:
        """Generate problems for a single failure mode."""
        problems: List[Dict[str, Any]] = []
        attempts = 0
        max_attempts = (target_count // self.gen_config.batch_size) * 2 + 5

        while len(problems) < target_count and attempts < max_attempts:
            attempts += 1
            example = random.choice(examples)
            prompt = FAILURE_GROUNDED_PROMPT.format(
                problem_example=example.get("question", ""),
                failure_category=mode,
                failure_description=description,
                batch_size=self.gen_config.batch_size,
            )
            try:
                batch = self._call_teacher_model(prompt)
                parsed = self._parse_batch_response(batch, mode)
                problems.extend(parsed)
                remaining = target_count - len(problems)
                print(
                    f"  Generated {len(parsed)} / {remaining} remaining "
                    f"for {mode}"
                )
            except Exception as e:
                print(f"  Attempt {attempts} failed for {mode}: {e}")
                time.sleep(self.gen_config.retry_base_delay * attempts)

        return problems[:target_count]

    def _call_teacher_model(self, prompt: str) -> str:
        """Call the teacher model API with retry logic.

        Args:
            prompt: The formatted prompt string.

        Returns:
            Raw text response from the model.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        last_error: Optional[Exception] = None
        models = [self.primary_model, self.fallback_model]

        for model in models:
            for attempt in range(1, self.gen_config.max_retries + 1):
                try:
                    return self._api_request(model, prompt)
                except requests.exceptions.HTTPError as e:
                    status = e.response.status_code if e.response is not None else 0
                    if status == 429:
                        delay = (
                            self.gen_config.retry_base_delay
                            * (2 ** (attempt - 1))
                        )
                        print(
                            f"  Rate limited ({model}), "
                            f"retrying in {delay:.1f}s "
                            f"(attempt {attempt}/{self.gen_config.max_retries})"
                        )
                        time.sleep(delay)
                        last_error = e
                    else:
                        last_error = e
                        if model == self.fallback_model:
                            break
                except Exception as e:
                    last_error = e
                    if model == self.fallback_model:
                        break

        raise RuntimeError(
            f"Teacher model call failed after all retries: {last_error}"
        )

    def _api_request(self, model: str, prompt: str) -> str:
        """Execute a single API request to the teacher model."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.gen_config.temperature,
            "max_tokens": self.gen_config.max_tokens,
            "top_p": self.gen_config.top_p,
        }
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.gen_config.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _parse_batch_response(
        self,
        raw: str,
        failure_mode_tag: str,
    ) -> List[Dict[str, Any]]:
        """Parse the model response into individual problem dicts."""
        problems: List[Dict[str, Any]] = []
        
        # Split raw text by "Question:" to separate multiple generated problems
        parts = raw.split("Question:")
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Parse within this problem part
            question_text = ""
            thinking_text = ""
            answer_text = ""
            
            # Find "Thinking:" or "<<thinking>>"
            thinking_idx = part.find("Thinking:")
            if thinking_idx == -1:
                thinking_idx = part.find("<<thinking>>")
                
            answer_idx = part.find("Answer:")
            if answer_idx == -1:
                answer_idx = part.find("\\boxed")
                
            # Determine question text (everything before thinking or answer)
            end_q_idx = len(part)
            if thinking_idx != -1:
                end_q_idx = min(end_q_idx, thinking_idx)
            if answer_idx != -1:
                end_q_idx = min(end_q_idx, answer_idx)
                
            question_text = part[:end_q_idx].strip()
            
            # Determine thinking text
            if thinking_idx != -1:
                # Find where thinking ends
                start_t = thinking_idx + (9 if part[thinking_idx:].startswith("Thinking:") else 12)
                end_t = len(part)
                if answer_idx != -1 and answer_idx > thinking_idx:
                    end_t = answer_idx
                thinking_content = part[start_t:end_t].strip()
                # Clean up any trailing tags
                if thinking_content.endswith("</thinking>"):
                    thinking_content = thinking_content[:-11].strip()
                elif thinking_content.endswith("</thinking>>"):
                    thinking_content = thinking_content[:-12].strip()
                thinking_text = f"<<thinking>>\n{thinking_content}\n</thinking>>"
                
            # Determine answer text
            if answer_idx != -1:
                start_a = answer_idx + (7 if part[answer_idx:].startswith("Answer:") else 0)
                answer_content = part[start_a:].strip()
                if "\\boxed" not in answer_content:
                    answer_text = f"\\boxed{{{answer_content}}}"
                else:
                    answer_text = answer_content
            
            current = {
                "question": question_text,
                "thinking_trace": thinking_text,
                "answer": answer_text
            }
            
            finalized = self._finalize_problem(current, failure_mode_tag)
            if finalized:
                problems.append(finalized)
                
        return problems

    def _finalize_problem(
        self,
        problem: Dict[str, str],
        failure_mode_tag: str,
    ) -> Optional[Dict[str, Any]]:
        """Add metadata and validate a parsed problem."""
        if not problem.get("thinking_trace") or not problem.get("answer"):
            return None

        difficulty = self._estimate_difficulty(problem.get("question", ""))
        return {
            "question": problem["question"],
            "thinking_trace": problem["thinking_trace"],
            "answer": problem["answer"],
            "failure_mode_tag": failure_mode_tag,
            "difficulty_estimate": difficulty,
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_model": self.primary_model,
        }

    def _estimate_difficulty(self, question: str) -> float:
        """Heuristic difficulty estimate on a 0-1 scale."""
        score = 0.0
        word_count = len(question.split())
        if word_count > 100:
            score += 0.2
        if word_count > 200:
            score += 0.1

        math_indicators = [
            r"\int", "prove", "show that", "find all",
            "maximum", "minimum", "optimize",
            "probability", "expected value",
            "matrix", "determinant", "eigenvalue",
            "integral", "derivative",
        ]
        score += min(0.4, sum(
            0.1 for ind in math_indicators
            if ind.lower() in question.lower()
        ))

        step_indicators = [
            "and", "then", "given that", "such that", "therefore",
        ]
        score += min(0.3, sum(
            0.06 for s in step_indicators
            if s.lower() in question.lower()
        ))

        return min(1.0, score)

    def _check_answer(self, expected: str, predicted: str) -> bool:
        """Check if predicted answer matches expected within tolerance."""
        from src.evaluation.metric import answers_equivalent

        return answers_equivalent(predicted, expected)

    def _classify_failure(self, response: Dict[str, Any]) -> str:
        """Classify the type of failure in a model response."""
        answer = response.get("answer", "")
        reasoning = response.get("reasoning", "")
        if not answer:
            return "no_answer"
        if not reasoning:
            return "incomplete"
        if "\\boxed" not in answer:
            return "format_error"
        return "wrong_answer"

    def save_dataset(
        self,
        data: List[Dict[str, Any]],
        filename: str = "raw_synthetic_dataset.jsonl",
    ) -> Path:
        """Save generated dataset to JSONL file.

        Args:
            data: List of synthetic problem dicts matching the output schema.
            filename: Output filename (default: raw_synthetic_dataset.jsonl).

        Returns:
            Path to saved file.
        """
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")
        print(f"Saved {len(data)} examples to {output_path}")
        return output_path

    def generate_synthetic_batch(self, mode: str, count: int) -> List[Dict[str, Any]]:
        """Generate a batch of synthetic problems for a given failure mode.

        Args:
            mode: Failure mode tag.
            count: Number of problems to generate.

        Returns:
            List of synthetic problem dicts.
        """
        canonical = FAILURE_MODE_ALIASES.get(mode, mode)
        description = FAILURE_MODE_DESCRIPTIONS.get(
            canonical,
            FAILURE_MODE_DESCRIPTIONS.get(mode, "General reasoning difficulty."),
        )
        prompt = SYSTEM_PROMPT.format(
            failure_description=description,
            num_problems=count,
        )
        try:
            raw = self._call_teacher_model(prompt)
            return self._parse_batch_response(raw, mode)
        except Exception as e:
            print(f"generate_synthetic_batch failed for {mode}: {e}")
            return []

    def dataset_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Compute per-mode statistics for a generated dataset.

        Args:
            data: List of synthetic problem dicts.

        Returns:
            Dict mapping failure_mode_tag to count.
        """
        counts: Dict[str, int] = {}
        for example in data:
            tag = example.get("failure_mode_tag", "unknown")
            counts[tag] = counts.get(tag, 0) + 1
        return counts
