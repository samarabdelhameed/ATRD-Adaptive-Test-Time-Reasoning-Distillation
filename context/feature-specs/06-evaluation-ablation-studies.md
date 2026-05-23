# 06 — Evaluation & Ablation Studies Specification

## Final Phase: Evaluation, Verification, and Documentation

### 1. Purpose and Setup Order
This specification defines the final evaluation phase, which validates the independent contribution of each pipeline component (baseline, SFT, GRPO, Budget Forcing) and generates the required documentation and notebooks for competition submission.

---

## 2. Technical Components to Implement

### 2.1 Ablation Study Framework (`src/evaluation/ablation.py`)
- **Action**: Automate sequential evaluations across four configurations:
  1. *Baseline (Base Model)*: Zero-shot performance.
  2. *SFT-only*: Performance after Phase 2 training.
  3. *SFT + GRPO*: Performance after Phase 3 reinforcement learning.
  4. *Full (SFT + GRPO + Budget Forcing)*: Final optimized pipeline performance.
- **Log**: Save scores, sample generations, and token count stats to `logs/ablation_results.json`.

### 2.2 Numerical Accuracy Metric Scorer (`src/evaluation/metric.py`)
- **Action**: Compare parsed model answers against target values with mathematical tolerance.
- **Handling Rules**:
  - Exact string match for algebraic terms (e.g., `\frac{\pi}{2}`).
  - Floating-point comparison with a default tolerance of $10^{-5}$ for numerical outputs.
  - Equivalence reduction for equations (simplifying terms before check).

### 2.3 Verification Scripts (`scripts/verify_unit_completion.py`)
- **Action**: Check that all phase gate checkpoints (`sft_checkpoint/`, `grpo_checkpoint/`) and files are present, valid, and fully compliant.
- **Run Command**:
  ```bash
  python scripts/verify_unit_completion.py P1
  python scripts/verify_unit_completion.py P2
  python scripts/verify_unit_completion.py P3
  python scripts/verify_unit_completion.py P4
  ```

### 2.4 Notebook Verification (`notebooks/*`)
- **Action**: Ensure all four notebooks execute sequentially, seed-locked, without errors, under the 4-hour Kaggle GPU session limit.

### 2.5 Publication Files
- **Notebooks**: Prepare clean, commented notebook templates for public Kaggle dataset.
- **README / Write-Up**: Write a detailed markdown summary ($\ge 2000$ words) describing:
  - Technical methodology (synthetic failure generation).
  - GRPO + PRM details.
  - Ablation results proving the accuracy delta at each step.
  - Award application submission details.

---

## 3. Exit Quality Gate
Before closing the project:
- [ ] All verification scripts run and return successfully.
- [ ] `submission.zip` is created, verified, and uploaded.
- [ ] The public Kaggle notebook is published and verified.
