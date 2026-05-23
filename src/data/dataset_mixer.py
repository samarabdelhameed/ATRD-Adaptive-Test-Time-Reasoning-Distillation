"""Dataset mixer with stratified sampling and leakage checking.

Combines synthetic (filtered), OpenMathReasoning, and
OpenCodeReasoning datasets at a 50/25/25 ratio.
Preserves failure mode distribution during mixing.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REASONING_SOURCES: Dict[str, float] = {
    "synthetic_filtered": 0.50,
    "open_math_reasoning": 0.25,
    "open_code_reasoning": 0.25,
}

REASONING_RATIO_LOWER: float = 0.70
REASONING_RATIO_UPPER: float = 0.80


class DatasetMixer:
    """Mix multiple datasets with 50/25/25 synthesis-to-open ratios.

    Preserves failure mode distribution from baseline evaluation
    and verifies reasoning ratio is within 70-80%.

    Attributes:
        seed: Random seed for reproducible mixing.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        random.seed(seed)

    def mix(
        self,
        synthetic: List[Dict[str, Any]],
        math_reasoning: List[Dict[str, Any]],
        code_reasoning: List[Dict[str, Any]],
        max_total: Optional[int] = None,
        failure_mode_ratios: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Mix datasets preserving stratified failure mode distribution.

        Args:
            synthetic: Filtered synthetic dataset with failure_mode_tag.
            math_reasoning: OpenMathReasoning dataset.
            code_reasoning: OpenCodeReasoning dataset.
            max_total: Maximum total examples in mixed dataset.
            failure_mode_ratios: Dict mapping failure_mode_tag to target
                ratio within the synthetic portion. If None, uses equal ratios.

        Returns:
            Mixed and shuffled dataset with _source metadata.
        """
        datasets = {
            "synthetic_filtered": synthetic,
            "open_math_reasoning": math_reasoning,
            "open_code_reasoning": code_reasoning,
        }

        mixed: List[Dict[str, Any]] = []
        for source_name, examples in datasets.items():
            ratio = REASONING_SOURCES[source_name]
            if max_total:
                n_target = int(max_total * ratio)
            else:
                n_target = int(len(examples) * ratio)

            n_available = len(examples)
            n_samples = min(n_target, n_available)

            if source_name == "synthetic_filtered" and failure_mode_ratios:
                sampled = self._stratified_sample(
                    examples, n_samples, failure_mode_ratios
                )
            else:
                sampled = random.sample(examples, n_samples)

            for ex in sampled:
                ex["_source"] = source_name
            mixed.extend(sampled)

        random.shuffle(mixed)

        reasoning_count = sum(
            1 for ex in mixed
            if ex.get("_source") in ("synthetic_filtered", "open_math_reasoning")
        )
        total = len(mixed)
        actual_ratio = reasoning_count / max(total, 1)

        print(f"DatasetMixer: {total} examples from {len(datasets)} sources")
        print(f"  Reasoning ratio: {actual_ratio:.2f} (target 0.70-0.80)")

        if actual_ratio < REASONING_RATIO_LOWER or actual_ratio > REASONING_RATIO_UPPER:
            print(
                f"  WARNING: Reasoning ratio {actual_ratio:.2f} "
                f"outside target range [{REASONING_RATIO_LOWER}, {REASONING_RATIO_UPPER}]"
            )

        return mixed

    def _stratified_sample(
        self,
        examples: List[Dict[str, Any]],
        n_samples: int,
        mode_ratios: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Sample from synthetic examples preserving failure mode ratios.

        Args:
            examples: Synthetic examples with failure_mode_tag.
            n_samples: Total number of samples to draw.
            mode_ratios: Target ratio per mode (sums to 1.0).

        Returns:
            Stratified sample of examples.
        """
        mode_groups: Dict[str, List[Dict[str, Any]]] = {}
        for ex in examples:
            mode = ex.get("failure_mode_tag", "unknown")
            mode_groups.setdefault(mode, []).append(ex)

        sampled: List[Dict[str, Any]] = []
        for mode, target_ratio in mode_ratios.items():
            n_from_mode = int(n_samples * target_ratio)
            available = mode_groups.get(mode, [])
            if not available:
                continue
            n_actual = min(n_from_mode, len(available))
            sampled.extend(random.sample(available, n_actual))

        remainder = n_samples - len(sampled)
        if remainder > 0:
            remaining = [
                ex for ex in examples if ex not in sampled
            ]
            if remaining:
                sampled.extend(random.sample(remaining, min(remainder, len(remaining))))

        return sampled

    def save_mixed(
        self,
        data: List[Dict[str, Any]],
        output_path: str = "data/final_train_dataset.jsonl",
    ) -> Path:
        """Save mixed dataset to JSONL file.

        Args:
            data: Mixed dataset.
            output_path: Output file path (default: data/final_train_dataset.jsonl).

        Returns:
            Path to saved file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")
        print(f"Saved mixed dataset ({len(data)} examples) to {path}")
        return path

    def get_distribution(
        self,
        data: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """Get source distribution of mixed dataset.

        Args:
            data: Mixed dataset with '_source' metadata.

        Returns:
            Dict mapping source name to count.
        """
        dist: Dict[str, int] = {}
        for example in data:
            source = example.get("_source", "unknown")
            dist[source] = dist.get(source, 0) + 1
        return dist


def check_leakage(
    train_dataset: List[str],
    test_set: List[str],
    n: int = 5,
) -> int:
    """Check n-gram overlap between training and test sets.

    Args:
        train_dataset: List of training text strings.
        test_set: List of test text strings.
        n: N-gram size (default 5).

    Returns:
        Number of overlapping n-grams.
    """
    train_ngrams: Set[str] = set()
    for text in train_dataset:
        for i in range(len(text) - n + 1):
            train_ngrams.add(text[i:i + n])

    test_ngrams: Set[str] = set()
    for text in test_set:
        for i in range(len(text) - n + 1):
            test_ngrams.add(text[i:i + n])

    overlap = len(train_ngrams & test_ngrams)
    if overlap > 0:
        print(f"  LEAKAGE DETECTED: {overlap} n-gram overlaps")
    else:
        print(f"  No leakage: 0 n-gram overlaps")
    return overlap
