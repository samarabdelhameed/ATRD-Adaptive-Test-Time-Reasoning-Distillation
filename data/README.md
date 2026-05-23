# Data Artifacts

| File | Description | Produced by |
|------|-------------|-------------|
| `public_test.jsonl` | Local dev benchmark (GSM8K bootstrap) | `run_pipeline.py --phase p1_data` |
| `baseline_results.json` | Nemotron baseline eval | `p1_baseline` (GPU) |
| `failure_modes.json` | Failure taxonomy | P1 |
| `raw_synthetic_dataset.jsonl` | Synthetic before filter | P1 |
| `final_train_dataset.jsonl` | **Training corpus** | P1 |
| `p1_stats.json` | P1 pipeline statistics | P1 |
| `cache/` | HF streamed subsets | P1 (auto) |

Do not commit large cache files to git. Add `data/cache/` to `.gitignore` if needed.
