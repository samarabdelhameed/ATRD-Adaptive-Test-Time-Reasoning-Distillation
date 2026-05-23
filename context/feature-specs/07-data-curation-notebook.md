# 07 — Data Curation Notebook Specification

## ATRD: Adaptive Test-Time Reasoning Distillation

### 1. Purpose and Setup Order
This specification defines the complete `01_data_generation.ipynb` notebook structure. This notebook executes Phase 1 end-to-end: baseline evaluation → failure extraction → synthetic generation → filtering → deduplication → mixing.

> [!IMPORTANT]
> Read specs `04-baseline-evaluation.md`, `05-synthetic-data-generation.md`, and `06-data-filtering-deduplication.md` before implementing this notebook.

---

## 2. Notebook Structure

### Cell 1: Imports and Reproducibility Setup
```python
import random
import numpy as np
import torch
import os, sys, json, re, hashlib
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
```

### Cell 2: Configuration
```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Phase1Config:
    BASE_MODEL: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
    MAX_TOKENS: int = 7680
    TEMPERATURE: float = 0.0
    BENCHMARK_PATH: str = "/kaggle/input/nemotron-benchmark"
    OUTPUT_DIR: Path = Path("/kaggle/working")
    RAW_SYNTHETIC_PATH: str = "/kaggle/working/raw_synthetic_dataset.jsonl"
    FILTERED_PATH: str = "/kaggle/working/filtered_synthetic_dataset.jsonl"
    FINAL_PATH: str = "/kaggle/working/final_train_dataset.jsonl"
    SYNTHETIC_TARGET: int = 10000
    NUM_FAILURE_MODES: int = 5
    API_MODEL: str = "deepseek-r1"
    API_TEMPERATURE: float = 0.7
```

### Cell 3: Helper Functions
- `format_prompt(question: str) -> str` — Format with `<<thinking>>` tokens
- `extract_boxed_answer(text: str) -> str` — Regex extraction with nested braces
- `check_answer(predicted: str, expected: str, tolerance: float) -> bool` — Numerical + string match
- `classify_failure(response: dict) -> str` — Map to 5 failure mode categories

### Cell 4: Load Base Model
```python
from src.models.loader import ModelLoader

loader = ModelLoader()
model = loader.load_model(quantize=True)
tokenizer = loader.load_tokenizer()
print(f"Model loaded: {model.num_parameters():,} params")
```

### Cell 5: Baseline Evaluation
- Load public benchmark from `/kaggle/input/`
- Run inference with `temperature=0.0`
- Extract answers from `\boxed{}`
- Compare against ground truth
- Classify failures
- Save `baseline_results.json`

### Cell 6: Failure Mode Analysis
- Aggregate failure counts per category
- Generate failure distribution chart
- Save `failure_modes.json` with ≥20 examples per mode
- Print summary table

### Cell 7: Synthetic Data Generation
- Configure API connection
- For each failure mode, generate `batch_size=10` problems
- Loop until target reached (10k–50k)
- Save `raw_synthetic_dataset.jsonl` with progress tracking

### Cell 8: Quality Filtering
```python
from src.data.judge_filter import JudgeFilter

judge = JudgeFilter(threshold=0.80)
filtered = judge.filter_dataset(raw_data)
report = judge.generate_report(raw_data, filtered)
```

### Cell 9: Deduplication
```python
from src.data.deduplicator import Deduplicator

dedup = Deduplicator(similarity_threshold=0.85)
deduplicated = dedup.deduplicate(filtered, key="question")
```

### Cell 10: Dataset Mixing
```python
from src.data.dataset_mixer import DatasetMixer

mixer = DatasetMixer(seed=42)
final_dataset = mixer.mix(
    datasets={"synthetic": deduplicated, "openmath": openmath_data},
    ratios={"synthetic": 0.5, "openmath": 0.25, "opencode": 0.25},
    max_total=50000
)
```

### Cell 11: Leakage Check
- Compute 5-gram overlap with public test set
- Assert zero matches
- Print verification report

### Cell 12: Save & Upload
- Save `final_train_dataset.jsonl`
- Version on Kaggle Datasets via `kaggle datasets create`

### Cell 13: Cleanup
```python
import gc
del model
torch.cuda.empty_cache()
gc.collect()
```

---

## 3. Kaggle-Specific Constraints

| Constraint | Strategy |
|-----------|----------|
| 4-hour session limit | Save intermediate artifacts every 30 min |
| Internet access (Phase 1 only) | All API calls in first half; offline processing in second |
| GPU memory | `torch.cuda.empty_cache()` between model and generation cells |
| Read-only `/kaggle/input/` | Outputs to `/kaggle/working/`, version to Kaggle Datasets |

---

## 4. Exit Quality Gate
- [ ] All cells execute sequentially without errors in fresh Kaggle session
- [ ] `final_train_dataset.jsonl` ≥ 10,000 problems
- [ ] `baseline_results.json` with accuracy and failure modes
- [ ] `logs/p1_baseline_eval.json` saved
- [ ] Dataset versioned on Kaggle Datasets
- [ ] GPU memory cleared in final cell
