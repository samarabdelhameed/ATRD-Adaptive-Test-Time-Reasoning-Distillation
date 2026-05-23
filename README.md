# ATRD — Adaptive Test-Time Reasoning Distillation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**LoRA-based fine-tuning pipeline for NVIDIA Nemotron-3-Nano-30B** — combining failure-grounded synthetic data, PRM-guided GRPO reinforcement learning, and difficulty-aware budget forcing for structured reasoning.

> 🏆 **NVIDIA Nemotron Model Reasoning Challenge**  
> Base model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`  
> Submission: LoRA rank-32 adapter (`submission.zip`)  
> Inference: vLLM `temperature=0.0`, `max_tokens=7680`

---

## Pipeline

```
[Base Model] → [Baseline Eval] → [Failure Collection] → [Synthetic Gen]
                                                              ↓
[LLM Judge Filter] → [Dataset Mix] → [SFT Training] → [GRPO Training]
                                                              ↓
[Budget Forcing] → [Final Eval] → [submission.zip] → [Kaggle]
```

### Four Phases

| Phase | Notebook | Output | Duration |
|-------|----------|--------|----------|
| P1: Data Curation | `01_data_generation.ipynb` | `final_train_dataset.jsonl` | ~2-4 hrs |
| P2: SFT Training | `02_sft_training.ipynb` | `checkpoints/sft/final_adapter/` | ~2-4 hrs |
| P3: GRPO RL | `03_grpo_training.ipynb` | `checkpoints/grpo/final_adapter/` | ~2-4 hrs |
| P4: Eval + Submit | `04_budget_forcing.ipynb` | `submission.zip` | ~1-2 hrs |

---

## Quick Start

### 1. Setup

```bash
git clone <repo> && cd atrd
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 2. Run Pipeline (CLI)

```bash
# Validate + tests
python run_pipeline.py --phase validate
python run_pipeline.py --phase test

# Data (CPU OK — streams OpenMath from Hugging Face)
python run_pipeline.py --phase p1_data

# GPU required (Kaggle P100 / G4)
python run_pipeline.py --phase p1_baseline
python run_pipeline.py --phase p2_sft
python run_pipeline.py --phase p3_grpo
python run_pipeline.py --phase p4_eval
python run_pipeline.py --phase p4_submit

# Fill write-up from logs
python run_pipeline.py --phase fill_writeup
```

See [docs/KAGGLE_RUNBOOK.md](docs/KAGGLE_RUNBOOK.md) for full competition workflow.

### 3. Run on Kaggle

Open the sequential notebooks in order:

| Notebook | Kaggle GPU | Est. Time |
|----------|-----------|-----------|
| `notebooks/01_data_generation.ipynb` | T4 × 2 | 2 hr |
| `notebooks/02_sft_training.ipynb` | P100 | 3 hr |
| `notebooks/03_grpo_training.ipynb` | P100 | 3 hr |
| `notebooks/04_budget_forcing.ipynb` | T4 × 2 | 1 hr |

### 4. Validate & Package

```bash
python scripts/verify_unit_completion.py P1 baseline
python scripts/verify_unit_completion.py P2 sft
python scripts/verify_unit_completion.py P3 grpo
python scripts/package_submission.py
```

---

## Project Structure

```
atrd/
├── configs/                 # Immutable configs
│   ├── competition_params.json
│   ├── base_lora.json
│   └── base_grpo.json
├── src/
│   ├── data/                # Data pipeline
│   │   ├── synthetic_generator.py
│   │   ├── judge_filter.py
│   │   ├── deduplicator.py
│   │   ├── dataset_mixer.py
│   │   └── budget_forcer.py
│   ├── models/              # Model loading + LoRA
│   │   ├── loader.py
│   │   └── lora_config.py
│   ├── training/            # SFT + GRPO + PRM
│   │   ├── sft_trainer.py
│   │   ├── grpo_trainer.py
│   │   └── prm.py
│   ├── inference/           # vLLM engine
│   │   └── vllm_engine.py
│   └── evaluation/          # Metrics + ablation
│       ├── metric.py
│       └── ablation.py
├── notebooks/               # Kaggle notebooks
│   ├── 01_data_generation.ipynb
│   ├── 02_sft_training.ipynb
│   ├── 03_grpo_training.ipynb
│   ├── 04_budget_forcing.ipynb
│   └── 05_public_kaggle.ipynb
├── scripts/                 # Automation
│   ├── package_submission.py
│   ├── verify_unit_completion.py
│   ├── verify_protected_files.py
│   └── sync_to_hub.py
├── tests/                   # Test suite
│   ├── test_data/
│   ├── test_models/
│   ├── test_training/
│   └── test_evaluation/
├── writeup/
│   └── METHODOLOGY.md       # Competition write-up
├── logs/                    # Training logs + metrics
├── checkpoints/             # LoRA adapter checkpoints
├── run_pipeline.py           # CLI entry point
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Key Techniques

### 1. Failure-Grounded Synthetic Data
- Baseline evaluation identifies 5 failure categories
- Frontier model (DeepSeek-R1 / Qwen3-235B) generates targeted examples
- LLM-as-judge filters top 80% (4 criteria composite score)
- MinHash (128-perm) + LSH deduplication at Jaccard > 0.85
- 50/25/25 stratified mix (synthetic / OpenMathReasoning / OpenCodeReasoning)

### 2. QLoRA Supervised Fine-Tuning
- 4-bit NF4 quantization with double quant + bfloat16
- LoRA rank-32, alpha=64, 7 target modules
- LR 2e-4, cosine schedule, warmup 100 steps, adamw_torch_fused
- Early stopping via plateau detection

### 3. GRPO + Implicit PRM
- Group size G=8, KL penalty 0.001, LR 5e-6
- Heuristic PRM (zero GPU): regex-based step scoring
- Optional log-ratio PRM: reference/policy log-prob ratio
- `KLMonitor` with hard stop at KL > 0.1

### 4. Budget Forcing (Data Quality)
- Heuristic difficulty estimation (0-1 scale)
- Linear token allocation: easy=512, medium=2048-4096, hard=4096-7680
- Hard problems: multi-stage refinement (max 3 attempts)
- Data-gen-only (not inference-time — competition fixes params)

---

## Configuration

All parameters are centralized in `configs/`:

- **`competition_params.json`** — IMMUTABLE. Inference engine settings.
- **`base_lora.json`** — LoRA rank, alpha, target modules, dropout.
- **`base_grpo.json`** — GRPO group size, KL penalty, LR, steps.

Never hardcode values. Always read from configs.

---

## Reproducibility

- Seeds: `random(42)`, `numpy(42)`, `torch(42)`, `cuda(42)`
- Packages pinned in `requirements.txt`
- Kaggle notebooks with sequential cell execution
- Phase gates via `verify_unit_completion.py`
- Logs saved to `logs/` with timestamps

---

## Evaluation Metric

Competition metric (from `src/evaluation/metric.py`):
1. Extract `\boxed{answer}` from completion
2. Fallback: heuristic patterns → last numeric value
3. Correct if: exact string match OR within 0.01 relative tolerance

---

## License

MIT — see [LICENSE](LICENSE)

**Note:** This project is for the NVIDIA Nemotron Model Reasoning Challenge. Base model and competition data are subject to NVIDIA's terms.
