"""
Failure-Grounded Synthetic Data Generator

Generates synthetic training examples by:
1. Running baseline model on competition problems
2. Extracting failure modes (wrong answers, incomplete reasoning)
3. Using teacher model to generate corrected reasoning traces
4. Formatting into SFT-ready (prompt, completion) pairs
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SyntheticGenerator:
    """Generate failure-grounded synthetic data for SFT training.

    Attributes:
        seed: Random seed for reproducibility.
        config: Competition parameters loaded from configs/.
        output_dir: Directory to save generated datasets.
    """

    def __init__(
        self,
        config_path: str = "configs/competition_params.json",
        output_dir: str = "data/synthetic",
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        random.seed(seed)
        np.random.seed(seed)

        with open(config_path, "r") as f:
            self.config = json.load(f)

    def generate_from_failures(
        self,
        problems: List[Dict[str, Any]],
        baseline_responses: List[Dict[str, Any]],
        teacher_model: Optional[Any] = None,
    ) -> List[Dict[str, str]]:
        """Generate synthetic examples from baseline failure cases.

        Args:
            problems: List of problem dicts with 'question' and 'answer' keys.
            baseline_responses: Model responses to classify as pass/fail.
            teacher_model: Optional teacher model for generating corrections.

        Returns:
            List of (prompt, completion) dicts ready for SFT.
        """
        failures = self._extract_failures(problems, baseline_responses)
        synthetic_data = []

        for failure in failures:
            example = self._generate_corrected_trace(failure, teacher_model)
            if example is not None:
                synthetic_data.append(example)

        return synthetic_data

    def _extract_failures(
        self,
        problems: List[Dict[str, Any]],
        responses: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Identify problems where the baseline model failed.

        Args:
            problems: Original problem set.
            responses: Baseline model responses.

        Returns:
            List of failure cases with problem + wrong response.
        """
        failures = []
        for prob, resp in zip(problems, responses):
            if not self._check_answer(prob.get("answer", ""), resp.get("answer", "")):
                failures.append({
                    "problem": prob,
                    "wrong_response": resp,
                    "failure_type": self._classify_failure(resp),
                })
        return failures

    def _generate_corrected_trace(
        self,
        failure: Dict[str, Any],
        teacher_model: Optional[Any] = None,
    ) -> Optional[Dict[str, str]]:
        """Generate a corrected reasoning trace for a failure case.

        Args:
            failure: Dict containing problem, wrong_response, failure_type.
            teacher_model: Teacher model for generating corrections.

        Returns:
            Dict with 'prompt' and 'completion' keys, or None if generation fails.
        """
        # TODO: Implement teacher-model-based correction generation
        # Placeholder structure for Phase 1
        problem = failure["problem"]
        return {
            "prompt": problem.get("question", ""),
            "completion": "",  # To be filled by teacher model
            "metadata": {
                "failure_type": failure["failure_type"],
                "source": "synthetic_correction",
            },
        }

    def _check_answer(self, expected: str, predicted: str) -> bool:
        """Check if predicted answer matches expected within tolerance.

        Args:
            expected: Ground truth answer string.
            predicted: Model's predicted answer string.

        Returns:
            True if answers match within numerical_tolerance.
        """
        tolerance = self.config.get("numerical_tolerance", 0.01)
        try:
            return abs(float(expected) - float(predicted)) <= tolerance
        except (ValueError, TypeError):
            return expected.strip() == predicted.strip()

    def _classify_failure(self, response: Dict[str, Any]) -> str:
        """Classify the type of failure in a model response.

        Args:
            response: Model response dict.

        Returns:
            Failure type string: 'wrong_answer', 'incomplete', 'no_answer', or 'format_error'.
        """
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
        data: List[Dict[str, str]],
        filename: str = "synthetic_sft.jsonl",
    ) -> Path:
        """Save generated dataset to JSONL file.

        Args:
            data: List of training examples.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")
        print(f"Saved {len(data)} examples to {output_path}")
        return output_path
