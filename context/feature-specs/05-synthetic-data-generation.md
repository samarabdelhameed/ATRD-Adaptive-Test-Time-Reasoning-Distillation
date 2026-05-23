# 05 — Synthetic Data Generation Specification

## ATRD: Adaptive Test-Time Reasoning Distillation

### 1. Purpose and Setup Order
This specification defines the implementation details for generating synthetic training data targeting the specific failure modes identified in the baseline evaluation.

> [!IMPORTANT]
> Read `04-baseline-evaluation.md` before implementing. Phase 1 baseline evaluation must be complete.

---

## 2. Technical Components

### 2.1 Prompt Template Engine (`src/data/synthetic_generator.py`)
```python
SYSTEM_PROMPT = """You are an expert mathematics tutor. Your task is to generate
variations of reasoning problems that target specific failure modes.

The student model (Nemotron-3-Nano-30B) struggles with: {failure_description}

Generate {num_problems} problems that test this specific weakness.
Each problem must include:
1. A clear question
2. A complete step-by-step thinking trace inside <<thinking>>...</thinking>>
3. A final answer in \boxed{} format
"""
```

### 2.2 Failure-Grounded Prompt Template
```text
You are an expert math tutor. The student model failed on this problem:
{problem_example}

The failure reason was: {failure_category}: {failure_description}.

Generate {batch_size} similar problems of comparable difficulty that test
the same reasoning weakness. For each problem:
- Write a clear question
- Provide a complete step-by-step solution inside <<thinking>>...</thinking>>
- End with the answer in \boxed{}

IMPORTANT: The thinking trace must be detailed and show complete reasoning.
```

### 2.3 Generator Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `batch_size` | 10 | Problems per API call |
| `max_retries` | 3 | With exponential backoff |
| `temperature` | 0.7 | For diversity |
| `max_tokens` | 4096 | Per generation |
| `top_p` | 0.95 | Nucleus sampling |
| `timeout` | 120s | API call timeout |

### 2.4 API Integration
- **Primary**: DeepSeek-R1 via OpenRouter / Together AI
- **Fallback**: Qwen3-235B via same API providers
- **Rate Limiting**: Max 50 problems per batch request; exponential backoff on 429 errors
- **Authentication**: API keys via Kaggle Secrets or environment variables

### 2.5 Output Schema (`raw_synthetic_dataset.jsonl`)
```json
{
  "question": "Solve the integral of x^2 from 0 to 3.",
  "thinking_trace": "<<thinking>>\nWe need to compute ∫₀³ x² dx...\n</thinking>>",
  "answer": "\\boxed{9}",
  "failure_mode_tag": "arithmetic_error",
  "difficulty_estimate": 0.65,
  "generation_timestamp": "2026-05-23T12:00:00Z",
  "source_model": "deepseek-r1"
}
```

### 2.6 Target Volume
- Minimum: 10,000 synthetic problems
- Target: 20,000–50,000 problems
- Per failure mode: 2,000–10,000 problems depending on mode severity

---

## 3. Implementation in Notebook

### Cell Structure
| Cell | Content |
|------|---------|
| 1 | Imports + seed fixing |
| 2 | Configuration (API keys, model names, batch sizes) |
| 3 | `generate_synthetic_batch()` function |
| 4 | API connection test (ping DeepSeek-R1 with 1 problem) |
| 5 | Generate per failure mode (loop over 5 modes) |
| 6 | Save `raw_synthetic_dataset.jsonl` |
| 7 | Dataset statistics report (count per failure mode) |

---

## 4. Exit Quality Gate
- [ ] ≥ 10,000 raw synthetic problems generated
- [ ] Each problem has `question`, `thinking_trace`, `answer`, `failure_mode_tag`
- [ ] Raw dataset saved before any filtering (preserve reproducibility)
- [ ] API rate limit handling works (exponential backoff logged)
- [ ] Fallback model tested if primary API unavailable
