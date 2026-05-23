"""
Dataset Mixer

Combines multiple data sources (synthetic, curated, augmented)
into a single training dataset with configurable mixing ratios.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class DatasetMixer:
    """Mix multiple datasets with configurable ratios and stratification.

    Attributes:
        seed: Random seed for reproducible mixing.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        random.seed(seed)

    def mix(
        self,
        datasets: Dict[str, List[Dict[str, Any]]],
        ratios: Optional[Dict[str, float]] = None,
        max_total: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Mix multiple datasets according to specified ratios.

        Args:
            datasets: Dict mapping source name to list of examples.
            ratios: Dict mapping source name to mixing ratio (sums to 1.0).
                    If None, uses equal ratios.
            max_total: Maximum total examples in mixed dataset.

        Returns:
            Mixed and shuffled dataset.
        """
        if ratios is None:
            ratios = {k: 1.0 / len(datasets) for k in datasets}

        # Normalize ratios
        total_ratio = sum(ratios.values())
        ratios = {k: v / total_ratio for k, v in ratios.items()}

        mixed = []
        for source_name, examples in datasets.items():
            ratio = ratios.get(source_name, 0.0)
            if max_total:
                n_samples = int(max_total * ratio)
            else:
                n_samples = int(len(examples) * ratio)

            sampled = random.sample(examples, min(n_samples, len(examples)))
            for ex in sampled:
                ex["_source"] = source_name
            mixed.extend(sampled)

        random.shuffle(mixed)
        print(f"DatasetMixer: {len(mixed)} total examples from {len(datasets)} sources")
        return mixed

    def save_mixed(
        self,
        data: List[Dict[str, Any]],
        output_path: str = "data/mixed_train.jsonl",
    ) -> Path:
        """Save mixed dataset to JSONL file.

        Args:
            data: Mixed dataset.
            output_path: Output file path.

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
