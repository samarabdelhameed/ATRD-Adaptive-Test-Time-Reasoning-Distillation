# 06 — Data Filtering & Deduplication Specification

## ATRD: Adaptive Test-Time Reasoning Distillation

### 1. Purpose and Setup Order
This specification defines the quality filtering and deduplication pipeline for synthetic data. Ensures only high-quality, diverse, non-leaking problems enter the training corpus.

> [!IMPORTANT]
> Read `05-synthetic-data-generation.md` before implementing. Raw synthetic data must exist.

---

## 2. Technical Components

### 2.1 LLM-as-Judge Quality Filter (`src/data/judge_filter.py`)

#### 2.1.1 Scoring Criteria
| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Correctness** | 0.35 | Answer matches verified computation |
| **Reasoning Clarity** | 0.25 | Thinking trace is coherent, step-by-step |
| **Difficulty Appropriateness** | 0.20 | Matches expected difficulty for failure mode |
| **Format Compliance** | 0.20 | Contains valid `<<thinking>>` + `\boxed{}` |

#### 2.1.2 Composite Score
```
final_score = 0.35 * correctness + 0.25 * clarity + 0.20 * difficulty + 0.20 * format
```

#### 2.1.3 Heuristic Pre-Filter (Before LLM Judge)
```python
def heuristic_score(example: dict) -> float:
    score = 0.0

    # Non-empty completion
    if example.get("completion", ""):
        score += 0.3

    # Reasoning markers present
    markers = ["therefore", "thus", "because", "step", "first", "next"]
    marker_count = sum(1 for m in markers if m.lower() in completion.lower())
    score += min(0.4, marker_count * 0.1)

    # Boxed answer present
    if "\\boxed" in completion:
        score += 0.3

    return min(1.0, score)
```

#### 2.1.4 Filter Threshold
- **Keep**: Top 80% by composite score (score ≥ 0.80)
- **Discard**: Bottom 20% — low-quality, incorrect, or malformed

### 2.2 MinHash Deduplication (`src/data/deduplicator.py`)

#### 2.2.1 Algorithm
1. Convert each problem text into character n-gram shingles (n=5)
2. Generate MinHash signatures (128 permutations)
3. Compare Jaccard similarity via LSH (Locality-Sensitive Hashing)
4. Remove pairs with Jaccard similarity > 0.85

```python
def compute_similarity(text_a: str, text_b: str) -> float:
    shingles_a = set(text_a.lower().split())
    shingles_b = set(text_b.lower().split())
    intersection = shingles_a & shingles_b
    union = shingles_a | shingles_b
    return len(intersection) / len(union) if union else 0.0
```

#### 2.2.2 Exact Dedup (SHA-256)
```python
def _hash_content(content: str) -> str:
    normalized = " ".join(content.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()
```

### 2.3 Dataset Mixing (`src/data/dataset_mixer.py`)

#### 2.3.1 Mixing Ratio
| Source | Ratio | Purpose |
|--------|-------|---------|
| Synthetic (filtered) | 50% | Failure-grounded reasoning |
| OpenMathReasoning | 25% | General math reasoning |
| OpenCodeReasoning | 25% | Code reasoning (non-reasoning split) |
| **Total Reasoning** | **75%** | Maintains reasoning focus |
| **Total Non-Reasoning** | **25%** | Prevents catastrophic forgetting |

#### 2.3.2 Stratification Rules
- Preserve failure mode distribution from baseline
- Shuffle within each stratum
- Verify final ratio: 0.70 ≤ reasoning_ratio ≤ 0.80

### 2.4 Leakage Check
```python
def check_leakage(train_dataset: List[str], test_set: List[str], n: int = 5) -> int:
    """Check n-gram overlap between training and test sets."""
    train_ngrams = set()
    for text in train_dataset:
        for i in range(len(text) - n + 1):
            train_ngrams.add(text[i:i+n])

    test_ngrams = set()
    for text in test_set:
        for i in range(len(text) - n + 1):
            test_ngrams.add(text[i:i+n])

    return len(train_ngrams & test_ngrams)
```
- **Requirement**: Zero 5-gram matches between training corpus and test set.

---

## 3. Implementation in Notebook

### Cell Structure
| Cell | Content |
|------|---------|
| 1 | Imports + seed fixing |
| 2 | Configuration (thresholds, paths, mixing ratios) |
| 3 | Load raw synthetic dataset from `raw_synthetic_dataset.jsonl` |
| 4 | Run LLM-as-judge scoring (heuristic) |
| 5 | Filter: keep top 80%, save `filtered_synthetic_dataset.jsonl` |
| 6 | Load OpenMathReasoning + OpenCodeReasoning datasets |
| 7 | Apply MinHash deduplication across all sources |
| 8 | Mix datasets at 75/25 ratio |
| 9 | Run leakage check against public test set |
| 10 | Save `final_train_dataset.jsonl` |
| 11 | Upload to Kaggle Datasets with version note |
| 12 | Statistics report (sizes, ratios, dedup rates, leakage) |

---

## 4. Exit Quality Gate
- [ ] `filtered_synthetic_dataset.jsonl` — top 80% quality score
- [ ] MinHash dedup removes near-duplicates (< 5% removal = good diversity signal)
- [ ] Final dataset: 75% reasoning ± 5%, 25% non-reasoning
- [ ] Zero n-gram overlap with test set (leakage check passes)
- [ ] Dataset versioned on Kaggle Datasets with descriptive version note
- [ ] `final_train_dataset.jsonl` validated (≥ 10k examples)
