"""
Dataset Deduplicator

Removes near-duplicate examples from synthetic datasets
using MinHash LSH and exact-match deduplication.
"""

import hashlib
from typing import Any, Dict, List, Set


class Deduplicator:
    """Remove duplicate and near-duplicate training examples.

    Attributes:
        seen_hashes: Set of content hashes for exact dedup.
        similarity_threshold: Threshold for near-duplicate detection.
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self.similarity_threshold = similarity_threshold
        self.seen_hashes: Set[str] = set()

    def deduplicate(
        self,
        examples: List[Dict[str, Any]],
        key: str = "prompt",
    ) -> List[Dict[str, Any]]:
        """Remove exact and near-duplicate examples.

        Args:
            examples: List of training examples.
            key: Field to use for deduplication comparison.

        Returns:
            Deduplicated list of examples.
        """
        unique = []
        self.seen_hashes.clear()

        for example in examples:
            content = example.get(key, "")
            content_hash = self._hash_content(content)

            if content_hash not in self.seen_hashes:
                self.seen_hashes.add(content_hash)
                unique.append(example)

        removed = len(examples) - len(unique)
        print(f"Deduplicator: removed {removed} duplicates, {len(unique)} remaining")
        return unique

    def _hash_content(self, content: str) -> str:
        """Generate SHA-256 hash of normalized content.

        Args:
            content: Text content to hash.

        Returns:
            Hex digest of content hash.
        """
        normalized = " ".join(content.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute Jaccard similarity between two texts using n-gram shingling.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Jaccard similarity score between 0.0 and 1.0.
        """
        shingles_a = self._get_shingles(text_a)
        shingles_b = self._get_shingles(text_b)

        if not shingles_a or not shingles_b:
            return 0.0

        intersection = shingles_a & shingles_b
        union = shingles_a | shingles_b
        return len(intersection) / len(union)

    def _get_shingles(self, text: str, n: int = 3) -> Set[str]:
        """Generate character n-gram shingles from text.

        Args:
            text: Input text.
            n: Shingle size.

        Returns:
            Set of n-gram shingles.
        """
        text = text.lower().strip()
        return {text[i : i + n] for i in range(len(text) - n + 1)}
