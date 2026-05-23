# ATRD: Adaptive Test-Time Reasoning Distillation

**NVIDIA Nemotron Model Reasoning Challenge 2026**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Competition](https://img.shields.io/badge/Kaggle-NVIDIA%20Nemotron-20BEFF.svg)](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)

## Overview

ATRD improves structured reasoning on **Nemotron-3-Nano-30B** through a three-stage pipeline:

1. **Failure-Grounded Synthetic Data Generation** — Extract failure modes from baseline model, generate corrected reasoning traces
2. **PRM-Guided GRPO** — Group Relative Policy Optimization with process reward signals
3. **Adaptive Budget Forcing** — Dynamic token budget allocation based on problem difficulty

**Deliverable:** Rank-32 LoRA adapter packaged as `submission.zip`

## Repository Structure

```
ATRD/
├── context/          # Specification layer (project docs)
├── notebooks/        # 4 Kaggle notebooks (P1-P4)
│   ├── 01_data_generation.ipynb
│   ├── 02_sft_training.ipynb
│   ├── 03_grpo_training.ipynb
│   └── 04_budget_forcing.ipynb
├── src/              # Python modules
│   ├── data/         # Synthetic generation, filtering, dedup, mixing
│   ├── models/       # Model loading, LoRA configuration
│   ├── training/     # SFT and GRPO training pipelines
│   ├── inference/    # Budget forcing, vLLM engine
│   └── evaluation/   # Metrics, ablation studies
├── configs/          # Immutable configuration files
├── scripts/          # Automation utilities
├── writeup/          # Prize documentation
├── logs/             # Structured telemetry (gitignored)
└── checkpoints/      # LoRA checkpoints (gitignored)
```

## Quick Start

```bash
# Clone repository
git clone https://github.com/samarabdelhameed/ATRD-Adaptive-Test-Time-Reasoning-Distillation.git
cd ATRD-Adaptive-Test-Time-Reasoning-Distillation

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"

# Run phase gate verification
python scripts/verify_unit_completion.py P1 baseline
```

## Kaggle Notebooks

| Phase | Notebook | Description |
|-------|----------|-------------|
| P1 | `01_data_generation.ipynb` | Baseline eval → Failure extraction → Synthetic generation |
| P2 | `02_sft_training.ipynb` | Supervised fine-tuning with LoRA |
| P3 | `03_grpo_training.ipynb` | GRPO with PRM-guided rewards |
| P4 | `04_budget_forcing.ipynb` | Adaptive budget forcing → Submission packaging |

## Competition Constraints

| Parameter | Value |
|-----------|-------|
| Base Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` |
| Max LoRA Rank | 32 |
| Max Tokens | 7680 |
| Inference Engine | vLLM |
| Answer Format | `\boxed{}` |
| Numerical Tolerance | 0.01 |

## Author

**Samar Abdelhameed Ahmed**

## License

MIT License — see [LICENSE](../LICENSE) for details.
