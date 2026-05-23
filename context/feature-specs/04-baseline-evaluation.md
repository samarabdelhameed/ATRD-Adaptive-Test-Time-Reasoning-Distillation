# 04 — Baseline Evaluation Specification

## ATRD: Adaptive Test-Time Reasoning Distillation

### 1. Purpose and Setup Order
This specification defines the implementation details for evaluating the baseline Nemotron-3-Nano-30B model on the public benchmark. The results drive failure mode extraction and targeted synthetic data generation.

> [!IMPORTANT]
> Read `context/project-overview.md` and `context/architecture.md` before implementing.

---

## 2. Technical Components

### 2.1 Baseline Evaluator (`src/evaluation/baseline.py`)
- **Input**: Base model `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` + public validation benchmark.
- **Action**: Run zero-shot inference with competition-mandated parameters:
  - `temperature = 0.0`
  - `max_tokens = 7680`
  - `top_p = 1.0`
- **Prompt Format**:
  ```text
  {question}

  <<thinking>>
  [Think step by step]
  </thinking>>

  Answer: \boxed{}
  ```

### 2.2 Output Schema (`baseline_results.json`)
```json
{
  "baseline_results": [
    {
      "question_id": "string",
      "question": "string",
      "predicted_answer": "string",
      "extracted_answer": "string",
      "ground_truth": "string",
      "is_correct": false,
      "failure_mode": "arithmetic_error | reasoning_loop | algebraic_error | format_violation | early_termination | null",
      "reasoning_trace": "string"
    }
  ],
  "summary": {
    "total_questions": 0,
    "correct": 0,
    "accuracy": 0.0,
    "accuracy_threshold": 0.60
  }
}
```

### 2.3 Failure Mode Taxonomy
| # | Failure Mode | Detection Heuristic | Example |
|---|-------------|---------------------|---------|
| 1 | **Arithmetic/Calculation Error** | Step-by-step arithmetic mistakes in reasoning | `2 + 2 = 5` |
| 2 | **Reasoning Loop** | Repeated phrases or equations without progress | `x = 5, x = 5, x = 5...` |
| 3 | **Algebraic Manipulation Error** | Incorrect equation rearrangement or factorization | `x + 2 = 5 → x = 7` |
| 4 | **Format Violation** | Missing or malformed `\boxed{}` answer block | `Answer: 42` (no box) |
| 5 | **Early Termination** | Model stops before reaching conclusion | Reasoning cut off mid-step |

### 2.4 Accuracy Metric (`src/evaluation/metric.py`)
- **Exact string match** for algebraic/formula answers (e.g., `\frac{\pi}{2}`)
- **Numerical tolerance**: `|predicted - expected| <= 1e-5` for numeric answers
- **Regex extraction**: Parse `\boxed{...}` with nested brace support

```python
def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} with nested brace support."""
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else ""
```

### 2.5 Decision Gate
If baseline accuracy > 70%: Re-evaluate benchmark difficulty assumptions. Expected target: < 60% to justify improvement effort.

---

## 3. Implementation in Notebook

### Cell Structure for `01_data_generation.ipynb`
| Cell | Content |
|------|---------|
| 1 | Imports + seed fixing |
| 2 | Configuration dataclass |
| 3 | Helper functions (format prompt, extract answer, classify failure) |
| 4 | Load base model via `ModelLoader` |
| 5 | Run baseline inference on benchmark |
| 6 | Extract answers and compute accuracy |
| 7 | Classify failures and save `failure_modes.json` |
| 8 | GPU memory cleanup |

---

## 4. Exit Quality Gate
- [ ] `baseline_results.json` generated with accuracy score
- [ ] At least 5 distinct failure modes identified
- [ ] Each failure mode has ≥ 20 example failures documented
- [ ] GPU memory cleared after evaluation (`torch.cuda.empty_cache()`)
- [ ] Results reproducible with seed 42
