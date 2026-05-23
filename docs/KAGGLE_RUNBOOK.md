# ATRD Kaggle Runbook

Professional execution guide for the NVIDIA Nemotron Model Reasoning Challenge.

## Prerequisites

| Item | Action |
|------|--------|
| Kaggle account | Join competition |
| GPU notebook | **P100** (SFT/GRPO) or **T4×2** (data/eval) |
| Secret `TOGETHER_API_KEY` | For frontier synthetic generation (optional) |
| Dataset `nemotron-benchmark` | Competition data |
| Dataset `open-math-reasoning` | Or HF stream fallback |
| Dataset `open-code-reasoning` | Or HF stream fallback |

## Local / CLI (before Kaggle)

```bash
pip install -e .
python run_pipeline.py --phase validate
python run_pipeline.py --phase test
python run_pipeline.py --phase p1_data        # builds data/final_train_dataset.jsonl
python run_pipeline.py --phase fill_writeup   # after logs exist
```

## Kaggle execution order

1. **Upload** this repo as Kaggle Dataset or clone from GitHub.
2. **Notebook `01_data_generation.ipynb`** — GPU, ~2h  
   - Set `TOGETHER_API_KEY` in Secrets (optional).  
   - Outputs: `final_train_dataset.jsonl`, `baseline_results.json`, `failure_modes.json`.
3. **Notebook `02_sft_training.ipynb`** — P100, ~3h  
   - Output: `checkpoints/sft/final_adapter/`
4. **Notebook `03_grpo_training.ipynb`** — P100, ~3h  
   - Output: `checkpoints/grpo/final_adapter/`
5. **Notebook `04_budget_forcing.ipynb` or `05_final_evaluation_ablation.ipynb`** — eval + ablation.
6. **Package submission:**
   ```bash
   python scripts/package_submission.py --adapter-path checkpoints/grpo/final_adapter
   ```
7. **Publish** `notebooks/05_public_kaggle.ipynb` as public Kaggle notebook.
8. **Fill write-up:**
   ```bash
   python run_pipeline.py --phase fill_writeup
   ```

## Phase gate verification

```bash
python scripts/verify_unit_completion.py P1 complete
python scripts/verify_unit_completion.py P2 sft
python scripts/verify_unit_completion.py P3 grpo
python scripts/verify_unit_completion.py P4 submission
```

## Artifact checklist

| File | Phase |
|------|-------|
| `data/final_train_dataset.jsonl` | P1 |
| `data/baseline_results.json` | P1 baseline |
| `checkpoints/sft/final_adapter/adapter_config.json` | P2 |
| `checkpoints/grpo/final_adapter/adapter_model.safetensors` | P3 |
| `submission.zip` | P4 |
| `logs/ablation_results.json` | P4 (real scores, not mock) |

## GPU-only steps

These **cannot** run on CPU-only machines:

- `p1_baseline`, `p2_sft`, `p3_grpo`, `p4_submit` (full weights)

Run them on Kaggle or NVIDIA G4 VM.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| OOM on 30B | Use 4-bit QLoRA; reduce `max_seq_length` to 2048 |
| No OpenMath locally | `p1_data` streams from Hugging Face (cached in `data/cache/`) |
| No API key | Template synthetic augmentation runs automatically |
| `submission.zip` empty | Complete GRPO checkpoint first |
