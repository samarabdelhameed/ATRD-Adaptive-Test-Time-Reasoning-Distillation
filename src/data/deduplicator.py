"""Dataset deduplicator with MinHash LSH near-duplicate detection.

Removes exact duplicates (SHA-256) and near-duplicates
(Jaccard > 0.85 via MinHash with 128 permutations).
"""

import hashlib
import random
from typing import Any, Dict, List, Set, Tuple

N_GRAM_SIZE: int = 5
NUM_PERMUTATIONS: int = 128
SIMILARITY_THRESHOLD: float = 0.85


def _hash_content(content: str) -> str:
    """Generate SHA-256 hash of normalized content.

    Args:
        content: Text content to hash.

    Returns:
        Hex digest of content hash.
    """
    normalized = " ".join(content.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def _get_shingles(text: str, n: int = N_GRAM_SIZE) -> Set[str]:
    """Generate character n-gram shingles from text.

    Args:
        text: Input text.
        n: Shingle size (default 5).

    Returns:
        Set of n-gram shingles.
    """
    text = text.lower().strip()
    return {text[i: i + n] for i in range(len(text) - n + 1)}


class MinHash:
    """MinHash signature generator for Jaccard similarity estimation.

    Generates a signature of NUM_PERMUTATIONS hash values
    to approximate the Jaccard similarity between two sets.
    """

    def __init__(self, num_perm: int = NUM_PERMUTATIONS, seed: int = 42) -> None:
        self.num_perm = num_perm
        self.seed = seed
        rng = random.Random(seed)
        self.a: List[int] = [rng.randint(1, 2**31 - 1) for _ in range(num_perm)]
        self.b: List[int] = [rng.randint(0, 2**31 - 1) for _ in range(num_perm)]
        self.p: int = 2**61 - 1

    def signature(self, shingles: Set[str]) -> List[int]:
        """Compute MinHash signature for a set of shingles.

        Args:
            shingles: Set of n-gram shingles.

        Returns:
            List of NUM_PERMUTATIONS hash values.
        """
        sig: List[int] = [2**31 - 1] * self.num_perm
        for shingle in shingles:
            h = hash(shingle) & 0xFFFFFFFF
            for i in range(self.num_perm):
                val = (self.a[i] * h + self.b[i]) % self.p
                if val < sig[i]:
                    sig[i] = val
        return sig

    @staticmethod
    def jaccard(sig_a: List[int], sig_b: List[int]) -> float:
        """Estimate Jaccard similarity from two MinHash signatures.

        Args:
            sig_a: Signature of first set.
            sig_b: Signature of second set.

        Returns:
            Estimated Jaccard similarity 0.0-1.0.
        """
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / max(len(sig_a), 1)


class LSH:
    """Locality-Sensitive Hashing for candidate pair bucketing.

    Bands MinHash signatures into buckets so similar items
    are likely to collide.
    """

    def __init__(self, num_bands: int = 16, rows_per_band: int = 8) -> None:
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.buckets: Dict[int, List[int]] = {}

    def hash_band(self, band: Tuple[int, ...]) -> int:
        """Hash a band of signature values into a bucket key."""
        return hash(band) & 0xFFFFFFFF

    def get_candidates(self, signature: List[int], doc_id: int) -> Set[int]:
        """Find candidate near-duplicates for a document.

        Args:
            signature: MinHash signature of the document.
            doc_id: Document index.

        Returns:
            Set of document indices that are candidates for near-duplicates.
        """
        candidates: Set[int] = set()
        for b in range(self.num_bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            band = tuple(signature[start:end])
            bucket_key = self.hash_band(band)
            if bucket_key not in self.buckets:
                self.buckets[bucket_key] = []
            self.buckets[bucket_key].append(doc_id)
            for other_id in self.buckets[bucket_key]:
                if other_id != doc_id:
                    candidates.add(other_id)
        return candidates


class Deduplicator:
    """Remove duplicate and near-duplicate training examples.

    Two-phase deduplication:
    1. Exact: SHA-256 hash of normalized content
    2. Near-duplicate: MinHash + LSH with Jaccard > 0.85

    Attributes:
        similarity_threshold: Jaccard threshold for near-duplicate detection.
        minhash: MinHash signature generator.
        lsh: Locality-Sensitive Hashing index.
    """

    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD) -> None:
        self.similarity_threshold = similarity_threshold
        self.minhash = MinHash()
        self.lsh = LSH()
        self.seen_hashes: Set[str] = set()

    def deduplicate(
        self,
        examples: List[Dict[str, Any]],
        key: str = "question",
    ) -> List[Dict[str, Any]]:
        """Remove exact and near-duplicate examples.

        Args:
            examples: List of training examples.
            key: Field to use for deduplication comparison.

        Returns:
            Deduplicated list of examples.
        """
        self.seen_hashes.clear()
        signatures: List[Tuple[int, List[int]]] = []
        unique: List[Dict[str, Any]] = []

        for idx, example in enumerate(examples):
            content = example.get(key, "")
            content_hash = _hash_content(content)

            if content_hash in self.seen_hashes:
                continue

            shingles = _get_shingles(content)
            sig = self.minhash.signature(shingles)
            signatures.append((idx, sig))
            unique.append(example)

        self.seen_hashes.clear()

        # LSH near-duplicate removal
        lsh = LSH()
        near_dup_ids: Set[int] = set()
        for sig_idx, (orig_idx, sig) in enumerate(signatures):
            candidates = lsh.get_candidates(sig, sig_idx)
            for cand_idx in candidates:
                if cand_idx >= len(signatures):
                    continue
                cand_sig = signatures[cand_idx][1]
                sim = MinHash.jaccard(sig, cand_sig)
                if sim > self.similarity_threshold:
                    near_dup_ids.add(sig_idx)
                    near_dup_ids.add(cand_idx)

        final: List[Dict[str, Any]] = [
            ex for i, ex in enumerate(unique) if i not in near_dup_ids
        ]

        removed = len(examples) - len(final)
        print(
            f"Deduplicator: removed {removed} duplicates "
            f"({len(final)} remaining)"
        )
        return final

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute Jaccard similarity between two texts.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Jaccard similarity score 0.0-1.0.
        """
        shingles_a = _get_shingles(text_a)
        shingles_b = _get_shingles(text_b)
        if not shingles_a or not shingles_b:
            return 0.0
        intersection = shingles_a & shingles_b
        union = shingles_a | shingles_b
        return len(intersection) / len(union)
