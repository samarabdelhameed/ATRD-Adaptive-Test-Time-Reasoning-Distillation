# 01 — Design System Specification

## ATRD: Adaptive Test-Time Reasoning Distillation

### Purpose

This document defines the design system and architectural patterns governing all ATRD components — from data pipelines to training loops to inference engines.

---

## 1. Module Architecture

### 1.1 Core Modules

| Module | Responsibility | Key Classes |
|--------|---------------|-------------|
| `src/data/` | Synthetic data generation, filtering, dedup, mixing | `SyntheticGenerator`, `JudgeFilter`, `Deduplicator`, `DatasetMixer` |
| `src/models/` | Model loading, LoRA configuration | `ModelLoader`, `create_lora_config` |
| `src/training/` | SFT and GRPO training pipelines | `SFTTrainerWrapper`, `GRPOTrainerWrapper` |
| `src/inference/` | Budget forcing, vLLM engine | `BudgetForcer`, `VLLMEngine` |
| `src/evaluation/` | Metrics, ablation studies | `compute_accuracy`, `AblationRunner` |

### 1.2 Dependency Flow

```
Data Generation (P1) → SFT Training (P2) → GRPO Training (P3) → Budget Forcing (P4)
       ↓                      ↓                    ↓                     ↓
  synthetic_sft.jsonl    sft_adapter/         grpo_adapter/        submission.zip
```

---

## 2. Configuration System

### 2.1 Immutable Configs (NEVER modify)

- `configs/competition_params.json` — Official competition parameters
  - Protected by pre-commit hook
  - Model name, max LoRA rank, token limits, inference engine

### 2.2 Tunable Configs

- `configs/base_lora.json` — Default LoRA hyperparameters
- `configs/custom_lora.json` — Experimental LoRA variants
- `configs/base_grpo.json` — GRPO training hyperparameters

### 2.3 Config Loading Pattern

```python
import json
from pathlib import Path

def load_config(name: str) -> dict:
    """Load config from configs/ directory."""
    path = Path(f"configs/{name}.json")
    with open(path, "r") as f:
        return json.load(f)
```

---

## 3. Data Pipeline Design

### 3.1 Flow

```
Problems → Baseline Inference → Failure Extraction → Teacher Correction
    → Judge Filtering → Deduplication → Dataset Mixing → Final SFT Data
```

### 3.2 Data Format (JSONL)

```json
{
  "prompt": "Solve: What is the integral of x^2 dx?",
  "completion": "<think>\nStep 1: Apply the power rule...\n</think>\n\\boxed{\\frac{x^3}{3} + C}",
  "metadata": {
    "source": "synthetic_correction",
    "failure_type": "wrong_answer",
    "quality_score": 0.85
  }
}
```

### 3.3 Answer Format

All answers MUST use `\boxed{}` format per competition rules:
- Numerical: `\boxed{42}`
- Expression: `\boxed{\frac{x^3}{3} + C}`
- Tolerance: `0.01` for numerical comparison

---

## 4. Training Design

### 4.1 LoRA Constraints

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank (`r`) | 32 | Competition maximum |
| Alpha | 64 | 2× rank for stable training |
| Target modules | All 7 projections | Maximum expressiveness |
| Dropout | 0.05 | Light regularization |
| Quantization | 4-bit NF4 | Fits T4 GPU (16GB) |

### 4.2 Training Phases

**Phase 2 — SFT:**
- Input: Synthetic (prompt, completion) pairs
- Loss: Cross-entropy on completion tokens
- Expected: Teach structured reasoning patterns

**Phase 3 — GRPO:**
- Input: Prompts with ground-truth answers
- Reward: 0.8 × correctness + 0.2 × format compliance
- Expected: Improve answer accuracy via exploration

---

## 5. Inference Design

### 5.1 Budget Forcing Strategy

```
Easy problems   → min_tokens (256)   → Fast inference
Medium problems → mid_tokens (~4000) → Balanced
Hard problems   → max_tokens (7680)  → Full reasoning
```

### 5.2 Difficulty Estimation Heuristics

- Problem length (word count)
- Mathematical complexity indicators (integrals, proofs, optimization)
- Multi-step reasoning markers

### 5.3 vLLM Engine Settings

- `temperature: 0.0` — Deterministic generation
- `top_p: 1.0` — No nucleus sampling
- `gpu_memory_utilization: 0.85` — Safe GPU memory limit

---

## 6. Quality Gates

Each phase must pass its gate before proceeding:

| Gate | Required | Checked By |
|------|----------|------------|
| P1 → P2 | Config files exist | `verify_unit_completion.py P1` |
| P2 → P3 | SFT adapter saved, rank ≤ 32 | `verify_unit_completion.py P2` |
| P3 → P4 | GRPO adapter saved, rank ≤ 32 | `verify_unit_completion.py P3` |
| P4 → Submit | `submission.zip` valid | `package_submission.py --dry-run` |

---

## 7. Code Standards

### 7.1 Python Style

- Type hints on all function signatures
- Docstrings (Google style) on all public methods
- Specific exception handling (no bare `except:`)
- Constants in UPPER_CASE

### 7.2 Reproducibility

- `SEED = 42` fixed in every notebook Cell #1
- `random.seed()`, `np.random.seed()`, `torch.manual_seed()` all set
- `torch.backends.cudnn.deterministic = True`

### 7.3 Naming Conventions

- Modules: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case()`
- Config keys: `snake_case`

---

## 8. File Protection

### Protected (immutable after initial commit):
- `configs/competition_params.json`

### Safe to modify:
- `configs/custom_lora.json`
- `configs/base_grpo.json` (with caution)
- All `src/` modules
- All notebooks

### Git-ignored (never committed):
- `checkpoints/`
- `logs/`
- `*.safetensors`, `*.bin`, `*.pt`
- `submission.zip`
