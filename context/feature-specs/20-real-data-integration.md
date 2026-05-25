# Feature 20: Real Data Integration & Frontend Population

## Overview
This feature specification details the integration of the real-world artifacts generated from the Kaggle Supervised Fine-Tuning (SFT) and data generation pipelines into the local project structure. The goal is to replace all mock data and placeholder values within the Next.js frontend and the methodology write-up with the actual, verifiable data produced by the NVIDIA Nemotron-3-Nano-30B model.

## Core Objectives
1. **Data Migration**: Securely transfer the Kaggle outputs (`results.zip`) into the local `data/` and `logs/` directories.
2. **Automated Documentation**: Trigger the `fill_writeup.py` script to parse the real logs and inject the actual metrics (accuracy, loss, etc.) into `writeup/METHODOLOGY.md`.
3. **Frontend Hydration**: Ensure the Next.js dashboard reads directly from the newly integrated `data/` and `logs/` paths to render the Failure Heatmaps, Budget Gauges, and Telemetry correctly.

## Implementation Steps

### 1. Data Extrication and Injection
The pipeline generated an extensive set of `.json` and `.jsonl` files during its 12-hour Kaggle execution. These must be moved to the primary project data sink.

**Source**: `scratch/extracted_results/ATRD-Adaptive-Test-Time-Reasoning-Distillation/data/*`
**Destination**: `data/*`

*Required Files to Move:*
- `final_train_dataset.jsonl` (The actual 57MB curated training set used for the SFT phase)
- `failure_modes.json` (The structured taxonomy of weaknesses extracted during Phase 1)
- `p1_stats.json` (Phase 1 generation statistics)
- `public_test.jsonl` (If evaluated)

### 2. Log Synchronization
Any logging artifacts generated during the Kaggle run (e.g., SFT results, ablation results) must be synchronized to the `logs/` directory so the Next.js `Telemetry` components can parse them.

**Source**: `scratch/extracted_results/ATRD-Adaptive-Test-Time-Reasoning-Distillation/logs/*` (If available)
**Destination**: `logs/*`

### 3. Automated Write-Up Hydration
The `writeup/METHODOLOGY.md` contains multiple `[REAL DATA]` markers. We must execute the `fill_writeup.py` orchestrator script.
- The script uses Regex to locate `[REAL DATA: ...]` markers.
- It maps these markers to the `logs/` and `data/` payloads.
- It overwrites the Markdown file with production-ready metrics, preparing it for immediate submission.

### 4. Next.js Dashboard Execution
With the data injected, the React/Next.js frontend must be built and served.
- Execute `npm install` to ensure all `@radix-ui` and `framer-motion` dependencies are present.
- Execute `npm run dev` to launch the dashboard.
- **Verification Gate**: The dashboard should no longer display "Simulated" data, but should accurately reflect the 57MB dataset size and the actual failure categories from `failure_modes.json`.

## Success Criteria
- [ ] `final_train_dataset.jsonl` exists in the local `/data` directory.
- [ ] `python scripts/fill_writeup.py` executes with exit code 0.
- [ ] `METHODOLOGY.md` contains 0 instances of the string `[REAL DATA]`.
- [ ] `npm run dev` successfully renders the dashboard with the Kaggle telemetry.
