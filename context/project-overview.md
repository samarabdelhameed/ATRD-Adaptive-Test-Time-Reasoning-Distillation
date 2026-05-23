# Adaptive Test-Time Reasoning Distillation (ATRD)

## Project Overview: Adaptive Test-Time Reasoning Distillation (ATRD)

### Mission Statement
Develop a LoRA-based fine-tuning pipeline for NVIDIA Nemotron-3-Nano-30B that achieves state-of-the-art reasoning accuracy on the NVIDIA Research reasoning benchmark through a novel combination of failure-grounded synthetic data generation, Process Reward Model-guided GRPO reinforcement learning, and dynamic test-time compute scaling via budget forcing.

### Problem Statement
Current reasoning benchmarks reveal a critical gap: language models struggle with structured multi-step reasoning tasks (mathematics, logic, code) despite strong performance on simpler tasks. Independent research efforts use inconsistent datasets, prompts, and evaluation setups, making reproducible improvement difficult. The NVIDIA Nemotron Model Reasoning Challenge provides a shared baseline (Nemotron-3-Nano-30B) and novel benchmark, but participants lack a unified methodology that combines:
1. Targeted synthetic data addressing model-specific failure modes
2. RL training that rewards correct intermediate reasoning steps, not just final answers
3. Inference-time compute allocation that adapts to problem difficulty

### Target Outcome
A rank-32 LoRA adapter for Nemotron-3-Nano-30B that:
- Achieves top-1% leaderboard accuracy on the private test set
- Generalizes better to private test than public test (anti-overfitting signal)
- Qualifies for all three Open Contribution Awards (Data, RL, Fine-tuning)
- Provides fully reproducible Kaggle notebooks with ablation studies

### Core Innovation
The "Adaptive Test-Time Reasoning Distillation" (ATRD) framework — the first known integration of:
- Failure-grounded synthetic data (AAAI 2026 methodology)
- PRM-guided GRPO with group size G=8
- Budget forcing for dynamic inference-time reasoning depth

### Success Metrics
| Metric | Target |
|--------|--------|
| Public Test Accuracy | >85% |
| Private Test Accuracy | >Public Test (generalization signal) |
| LoRA Rank | ≤32 (max allowed) |
| Inference Latency | Compatible with vLLM max_tokens=7680 |
| Documentation | Public Kaggle notebook + write-up |

### Scope: In-Scope
- Synthetic data generation pipeline targeting Nemotron-3-Nano-30B failure modes
- LoRA fine-tuning (SFT phase) with QLoRA 4-bit quantization
- GRPO reinforcement learning with verifiable rewards
- Implicit PRM via log-ratio scoring of reasoning steps
- Budget forcing implementation for test-time compute scaling
- Kaggle notebook with full reproducibility
- Ablation studies for each component

### Scope: Out-of-Scope
- Full model fine-tuning (prohibited by competition rules)
- Cloud deployment infrastructure (competition uses vLLM)
- UI/frontend development (not required for submission)
- Multi-modal reasoning (text-only benchmark)
- Ensemble methods or model merging

### Constraints & Rules
- Base model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16
- LoRA rank: ≤32
- Submission format: submission.zip with adapter_config.json
- Inference engine: vLLM with reasoning parser plugin
- Temperature: 0.0 (fixed by competition)
- Max tokens: 7680 (fixed by competition)
- GPU memory utilization: 0.85 (fixed by competition)
- Compute: Google Cloud G4 VMs (RTX PRO 6000 Blackwell)

### Timeline
| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Failure Analysis & Data Curation | Days 1-7 | 10-50k synthetic problems + curated mix |
| Phase 2: SFT + LoRA Training | Days 8-14 | Trained LoRA checkpoint |
| Phase 3: GRPO + PRM RL | Days 15-21 | RL-optimized adapter |
| Phase 4: Budget Forcing & Evaluation | Days 22-25 | Final submission + write-up |

### Risk Mitigation
| Risk | Mitigation |
|------|------------|
| Overfitting to public test | 75/25 reasoning/non-reasoning mix; diversity in synthetic data |
| Compute limitations | QLoRA 4-bit; gradient checkpointing; efficient batching |
| PRM training instability | Implicit PRM (log-ratios) instead of separate model |
| Budget forcing incompatibility | Test with vLLM reasoning parser early |
| Synthetic data quality | LLM-as-judge filtering; keep top 80% |

### Technical Stack
| Layer | Technology |
|-------|------------|
| Base Model | NVIDIA Nemotron-3-Nano-30B-A3B-Base-BF16 |
| Fine-Tuning | PEFT (LoRA), QLoRA (bitsandbytes) |
| Training Framework | Hugging Face TRL, Unsloth, or Axolotl |
| RL | TRL GRPO implementation |
| Data Processing | Python, pandas, datasets (Hugging Face) |
| Evaluation | vLLM, custom metric matching competition |
| Compute | Kaggle Notebooks (T4x2/P100) + Google Cloud G4 VMs |
| Version Control | Git + Kaggle Dataset versioning |
| Documentation | Kaggle Notebook + Markdown write-up |

### Deliverables
1. `submission.zip` — LoRA adapter (rank 32)
2. `01_data_generation.ipynb` — Failure-grounded synthetic data pipeline
3. `02_sft_training.ipynb` — Supervised fine-tuning with LoRA
4. `03_grpo_training.ipynb` — GRPO + PRM reinforcement learning
5. `04_budget_forcing.ipynb` — Test-time compute scaling
6. `05_evaluation.ipynb` — Final evaluation and ablation studies
7. `README.md` — Complete methodology write-up

### Quality Gates
Before proceeding to next phase, verify:
- [ ] Phase 1: ≥10k synthetic problems generated; failure modes documented; LLM-as-judge filter applied
- [ ] Phase 2: LoRA adapter trains without OOM; loss converges; sample outputs show structured reasoning
- [ ] Phase 3: GRPO reward increases monotonically; KL divergence < threshold; PRM scores correlate with correctness
- [ ] Phase 4: Budget forcing improves accuracy on hard problems without degrading easy problems; vLLM compatible
- [ ] Final: Submission.zip passes competition validation; public notebook published; write-up complete

## Goals

### Primary Goals (Must Achieve)

1. **Achieve Top-1% Leaderboard Accuracy**
   - Target: >85% accuracy on public test set of the NVIDIA Nemotron Reasoning Benchmark
   - Target: Private test accuracy > public test accuracy (generalization signal, anti-overfitting)
   - Metric: Proportion of correctly answered questions within \boxed{} format
   - Measurement: Kaggle leaderboard position relative to all submissions

2. **Deliver Competition-Compliant LoRA Adapter**
   - Constraint: Rank ≤32 (maximum allowed)
   - Format: submission.zip containing adapter_config.json + adapter weights
   - Compatibility: Verified functional with vLLM inference engine (temperature=0.0, max_tokens=7680)
   - Base Model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16
   - Validation: Passes official competition submission checker

3. **Publish Reproducible Public Documentation**
   - Deliverable: Public Kaggle notebook documenting complete methodology
   - Deliverable: Markdown write-up explaining techniques, datasets, and ablation studies
   - Requirement: Mandatory for prize eligibility (per competition rules)
   - Standard: Enables independent reproduction of results by other researchers

### Secondary Goals (Competitive Differentiation)

4. **Win Open Contribution Awards (Top 10% Qualifier)**
   - Target: Best Data/Synthetic Data Method — for failure-grounded synthetic pipeline
   - Target: Best RL Method — for GRPO + PRM implementation
   - Target: Best Fine-tuning Method — for adaptive budget forcing LoRA
   - Strategy: Submit separate entries for each category with clear identification

5. **Demonstrate Novel Research Contribution**
   - Innovation: First known integration of failure-grounded data + GRPO-PRM + budget forcing for Nemotron
   - Validation: Ablation studies proving each component's independent contribution
   - Impact: Advancing open reasoning workflows reproducible by community

6. **Optimize Resource Efficiency**
   - Compute: Train within Google Cloud G4 VM constraints (RTX PRO 6000 Blackwell)
   - Memory: QLoRA 4-bit quantization to fit 30B model on available GPU memory
   - Time: Complete full pipeline within 25-day competition window (June 15 deadline)
   - Cost: Minimize unnecessary compute through efficient batching and checkpointing

### Tertiary Goals (Long-term Value)

7. **Build Portfolio Asset for Master's Scholarship Applications**
   - Demonstration: End-to-end ML engineering project with production-grade documentation
   - Evidence: Public Kaggle profile + published notebook + competition ranking
   - Alignment: Computer Science / Financial Engineering / AI specialization

8. **Establish Reproducible Research Template**
   - Output: Modular pipeline reusable for future LLM fine-tuning competitions
   - Documentation: Clear separation of data, training, RL, and inference components
   - Open Source: MIT-licensed code enabling community extension

## Core Pipeline Flow

```
[Base Model] → [Baseline Eval] → [Failure Collection]
                                              ↓
[Synthetic Gen] ← [Frontier LLM] ← [Failure Analysis]
        ↓
[LLM Judge Filter] → [Dataset Mix] → [Final Dataset]
                                              ↓
[QLoRA Config] → [SFT Training] → [SFT Checkpoint]
                                              ↓
[PRM Setup] → [GRPO Training] → [GRPO Checkpoint]
                                              ↓
[Budget Forcing] → [Final Eval] → [submission.zip]
                                              ↓
[Kaggle Notebook] + [Write-up] + [Award Forms]
```

### Phase 1: Failure Analysis & Data Curation (Days 1-7)

**Step 1: Baseline Evaluation**
- Input: Nemotron-3-Nano-30B base model (untrained) + public benchmark test cases
- Action: Run inference on 100% of public test set using vLLM with default parameters
- Output: `baseline_results.json` — predicted answers vs. ground truth for each problem
- Decision Gate: If baseline accuracy < 60%, proceed to failure collection. If > 70%, re-evaluate benchmark difficulty assumption.

**Step 2: Failure Mode Extraction**
- Input: `baseline_results.json`
- Action: Filter incorrect predictions; categorize by error type (arithmetic, logic, code, algebraic manipulation, proof structure)
- Output: `failure_modes.json` — structured taxonomy of weaknesses with example problems
- Quality Check: ≥5 distinct failure modes identified; each mode has ≥20 example failures

**Step 3: Targeted Synthetic Data Generation**
- Input: `failure_modes.json` + frontier model API (DeepSeek-R1 or Qwen3-235B)
- Action: For each failure mode, prompt frontier model to generate 100-500 similar problems with detailed thinking traces and \boxed{} answers
- Prompt Template: "The Nemotron model failed this [problem type] because [failure reason]. Generate 10 variations that test the same reasoning weakness with step-by-step solutions."
- Output: `raw_synthetic_dataset.jsonl` — 10,000-50,000 synthetic problems

**Step 4: LLM-as-Judge Quality Filtering**
- Input: `raw_synthetic_dataset.jsonl`
- Action: Use secondary LLM judge to score each synthetic problem on: correctness of answer, clarity of thinking trace, difficulty appropriateness, novelty vs. training data overlap
- Filter: Keep top 80% by composite score; discard bottom 20%
- Output: `filtered_synthetic_dataset.jsonl` — 8,000-40,000 high-quality problems

**Step 5: Dataset Mixing & Deduplication**
- Input: `filtered_synthetic_dataset.jsonl` + NVIDIA OpenMathReasoning + OpenCodeReasoning
- Action: Merge datasets; apply MinHash deduplication to remove near-duplicates; stratify split 75% reasoning / 25% non-reasoning
- Output: `final_train_dataset.jsonl` — balanced, deduplicated training corpus
- Validation: Verify no test set leakage via n-gram overlap analysis

---

### Phase 2: Supervised Fine-Tuning (Days 8-14)

**Step 6: LoRA Configuration & Model Loading**
- Input: `final_train_dataset.jsonl` + base model checkpoint
- Action: Load Nemotron-3-Nano-30B in 4-bit (NF4) via bitsandbytes; attach LoRA adapters (rank=32, alpha=64) to all linear layers
- Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Output: QLoRA-wrapped model ready for training

**Step 7: SFT Training Execution**
- Input: QLoRA model + formatted training data
- Action: Train with causal LM objective; format prompts with Nemotron reasoning parser tokens (`<<thinking>...</thinking>`)
- Hyperparameters: Learning rate 2e-4, batch size 1 (gradient accumulation 8), max sequence length 4096, warmup 100 steps
- Monitoring: Log training loss, validation loss, sample generations every 500 steps
- Output: `sft_checkpoint/` — LoRA weights after convergence
- Decision Gate: If validation loss plateaus > 3 epochs, stop early. If diverges, reduce LR by 50%.

**Step 8: SFT Evaluation**
- Input: `sft_checkpoint/` + public benchmark subset
- Action: Run inference; compare accuracy vs. baseline
- Output: `sft_results.json` — accuracy improvement metric
- Threshold: Must show ≥10% absolute improvement over baseline to proceed to RL

---

### Phase 3: GRPO Reinforcement Learning (Days 15-21)

**Step 9: Implicit PRM Setup**
- Input: `sft_checkpoint/`
- Action: Initialize log-ratio scorer: for each reasoning step, compute log-probability under SFT policy vs. current policy; use ratio as proxy for step quality
- Alternative: Train lightweight separate PRM (1-2 layers) on step-level correctness if compute allows
- Output: PRM scoring function callable on generated reasoning traces

**Step 10: GRPO Training Loop**
- Input: `sft_checkpoint/` + PRM scorer + training problems
- Action: For each problem, generate G=8 candidate solutions; score each with PRM + final answer correctness; compute relative advantage; update policy
- Reward: Binary (1 if \boxed{} answer correct, 0 otherwise) + PRM step bonuses
- KL Penalty: 1e-3 to prevent deviation from SFT checkpoint
- Training: 30-500 steps depending on reward convergence
- Output: `grpo_checkpoint/` — RL-optimized LoRA weights

**Step 11: RL Validation**
- Input: `grpo_checkpoint/` + validation set
- Action: Evaluate accuracy; check for reward hacking (correct answers but nonsensical reasoning)
- Output: `grpo_results.json`
- Decision Gate: If accuracy < SFT accuracy, reduce KL penalty or increase PRM weight. If reasoning quality degraded, stop RL.

---

### Phase 4: Test-Time Adaptation & Submission (Days 22-25)

**Step 12: Budget Forcing Implementation**
- Input: `grpo_checkpoint/` + inference engine (vLLM)
- Action: Build reasoning parser that monitors end-of-thinking token (`</thinking>`)
- Logic:
  - Easy problems (detected by problem length/complexity heuristics): Force termination at 512 thinking tokens
  - Hard problems: Append "Wait" token up to 3 times when end-of-thinking detected, extending reasoning
- Validation: Test on held-out problems; verify hard problems improve, easy problems don't degrade

**Step 13: Final Evaluation**
- Input: Budget-forcing inference pipeline + full public benchmark
- Action: Run complete evaluation with competition parameters (temperature=0.0, max_tokens=7680, top_p=1.0)
- Output: `final_accuracy_score` + `submission_predictions.json`

**Step 14: Submission Packaging**
- Input: `grpo_checkpoint/` LoRA weights
- Action: Package into `submission.zip` with `adapter_config.json`; verify with competition submission demo
- Output: `submission.zip` ready for upload

**Step 15: Documentation & Publication**
- Input: All notebooks + results + write-up
- Action: Publish Kaggle notebook; submit write-up; fill Open Contribution Award forms
- Output: Public documentation + award submissions

## Features

### Data Pipeline Features

- **Failure-Grounded Synthetic Data Generation**
  - Baseline evaluation of Nemotron-3-Nano-30B on public benchmark to identify failure modes
  - Automated extraction and categorization of incorrect predictions by error type (arithmetic, logic, code, algebraic manipulation, proof structure)
  - Targeted synthetic problem generation using frontier models (DeepSeek-R1, Qwen3-235B) with structured prompts based on failure analysis
  - LLM-as-judge quality filtering with composite scoring (correctness, clarity, difficulty, novelty) retaining top 80%

- **Dataset Curation & Deduplication**
  - MinHash-based near-duplicate detection across synthetic and existing datasets
  - Stratified mixing: 75% reasoning + 25% non-reasoning to preserve general capabilities
  - N-gram overlap analysis to prevent test set leakage
  - Integration with NVIDIA OpenMathReasoning and OpenCodeReasoning datasets

### Model Training Features

- **QLoRA Supervised Fine-Tuning (SFT)**
  - 4-bit quantization (NF4) via bitsandbytes for memory-efficient loading of 30B parameter model
  - LoRA rank-32 configuration targeting all linear projection layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
  - Nemotron reasoning parser token integration (`<<thinking>...</thinking>`) for structured thinking trace formatting
  - Early stopping based on validation loss plateau detection

- **GRPO Reinforcement Learning**
  - Group Relative Policy Optimization with group size G=8 for stable training
  - Verifiable reward function: binary correctness of \boxed{} final answer
  - Implicit Process Reward Model (PRM) via log-ratio scoring of intermediate reasoning steps
  - KL divergence penalty (1e-3) to prevent catastrophic forgetting of SFT checkpoint

### Test-Time Optimization Features

- **Dynamic Budget Forcing**
  - Real-time monitoring of reasoning token stream for end-of-thinking token detection (`</thinking>`)
  - Difficulty-based adaptive compute allocation:
    - Easy problems: forced termination after 512 thinking tokens
    - Hard problems: "Wait" token injection up to 3 times to extend reasoning depth
  - vLLM-compatible implementation with reasoning parser plugin integration

- **Competition Inference Pipeline**
  - vLLM engine configuration matching competition parameters (temperature=0.0, max_tokens=7680, top_p=1.0)
  - Automated \boxed{} answer extraction with fallback heuristic patterns
  - Batch inference optimization for leaderboard iteration efficiency

### Evaluation & Validation Features

- **Ablation Study Framework**
  - Modular component isolation: baseline, SFT-only, SFT+GRPO, SFT+GRPO+BudgetForcing
  - Per-component accuracy contribution measurement
  - Generalization validation: private test accuracy vs. public test accuracy comparison

- **Submission Packaging**
  - Automated `submission.zip` generation with `adapter_config.json` validation
  - Competition submission demo compatibility verification
  - LoRA rank constraint enforcement (≤32)

### Documentation Features

- **Reproducible Kaggle Notebook**
  - Complete end-to-end pipeline in sequential executable cells
  - Hyperparameter configuration tables with rationale
  - Training loss curves, sample generations, and intermediate checkpoint evaluation

- **Methodology Write-Up**
  - Technical explanation of failure-grounded data approach with literature references
  - GRPO+PRM implementation details with pseudocode
  - Budget forcing algorithm with complexity analysis
  - Ablation study results with statistical significance testing

## Scope

### In Scope

- **Failure-Grounded Synthetic Data Pipeline**
  - Baseline evaluation of Nemotron-3-Nano-30B on public benchmark
  - Automated failure mode extraction and categorization
  - Targeted synthetic problem generation using frontier LLMs (DeepSeek-R1, Qwen3-235B)
  - LLM-as-judge quality filtering with composite scoring
  - Dataset mixing, deduplication, and leakage prevention

- **QLoRA Supervised Fine-Tuning**
  - 4-bit quantization and LoRA rank-32 adapter training
  - Nemotron reasoning parser token formatting (`<<thinking>...</thinking>`)
  - Training on curated dataset (synthetic + OpenMathReasoning + OpenCodeReasoning)
  - Validation loss monitoring and early stopping

- **GRPO Reinforcement Learning**
  - Group Relative Policy Optimization with group size G=8
  - Binary reward based on \boxed{} answer correctness
  - Implicit PRM via log-ratio scoring of reasoning steps
  - KL penalty to preserve SFT checkpoint knowledge

- **Budget Forcing Test-Time Adaptation**
  - Dynamic reasoning depth control based on problem difficulty
  - End-of-thinking token monitoring and "Wait" token injection
  - vLLM-compatible inference pipeline with reasoning parser plugin
  - Evaluation on public benchmark with competition parameters

- **Competition Submission Packaging**
  - `submission.zip` generation with `adapter_config.json`
  - Rank-32 LoRA adapter validation
  - Public Kaggle notebook publication
  - Methodology write-up with ablation studies

### Out of Scope

- **Full Model Fine-Tuning**
  - Prohibited by competition rules; only LoRA adapters (rank ≤32) permitted
  - Any technique requiring modification of base model weights directly

- **Multi-Modal Reasoning**
  - Image, audio, or video understanding tasks
  - Vision-language model components or cross-modal fusion

- **Frontend / UI Development**
  - Web application, dashboard, or interactive interface
  - React, Next.js, Streamlit, or any UI framework
  - Real-time user-facing demo or SaaS deployment

- **Cloud Deployment Infrastructure**
  - Production API endpoints, load balancing, or auto-scaling
  - Docker containers, Kubernetes, or serverless architecture
  - Cost monitoring or billing systems beyond competition compute allocation

- **Ensemble Methods & Model Merging**
  - Combining multiple LoRA adapters or base models
  - Weight averaging, task arithmetic, or model soup techniques
  - Multi-model voting or mixture-of-experts approaches

- **External Benchmark Evaluation**
  - Testing on MATH, GSM8K, HumanEval, or other public benchmarks
  - Comparison against GPT-4, Claude, or other commercial models
  - Academic paper publication or peer review process

- **Post-Competition Extensions**
  - Larger model variants (70B, 405B) or different base architectures
  - Real-world deployment for enterprise clients
  - Integration with Electronic Health Records (EHR) or other domain systems
  - Commercial licensing or productization of the methodology

### Scope Matrix

| Feature Request                       | In Scope | Out of Scope | Rationale                  |
| ------------------------------------- | -------- | ------------ | -------------------------- |
| LoRA rank-32 training                 | ✅        |              | Core requirement           |
| Full fine-tuning all layers           |          | ✅            | Competition rules prohibit |
| Budget forcing inference              | ✅        |              | Competitive advantage      |
| Streamlit dashboard for visualization |          | ✅            | UI not required            |
| GRPO with G=8                         | ✅        |              | State-of-the-art RL        |
| Model ensemble (3 adapters)           |          | ✅            | Out of scope               |
| Ablation studies in notebook          | ✅        |              | Required for prizes        |
| Deploy API on AWS                     |          | ✅            | Infrastructure not needed  |
| Synthetic data from failures          | ✅        |              | Novel data method          |
| Test on GSM8K benchmark               |          | ✅            | External evaluation        |

## Success Criteria

### Success Criteria Dashboard

| #  | Criterion          | Target     | Current | Status | Evidence           |
| -- | ------------------ | ---------- | ------- | ------ | ------------------ |
| 1  | Public Accuracy    | ≥ 85%      | —       | ⬜      | Kaggle leaderboard |
| 2  | Generalization Gap | > 0%       | —       | ⬜      | Final leaderboard  |
| 3  | Submission Valid   | Pass       | —       | ⬜      | Validation tool    |
| 4  | Synthetic Data     | ≥ 10k      | —       | ⬜      | Dataset report     |
| 5  | Training Converge  | Stable     | —       | ⬜      | Loss curves        |
| 6  | Budget Forcing     | +3% net    | —       | ⬜      | Stratified eval    |
| 7  | Notebook Published | Yes        | —       | ⬜      | Kaggle URL         |
| 8  | Write-Up Complete  | ≥ 2k words | —       | ⬜      | Markdown file      |
| 9  | Top 10% Rank       | Yes        | —       | ⬜      | Leaderboard        |
| 10 | On Schedule        | All phases | —       | ⬜      | Gantt chart        |
| 11 | Compute Budget     | ≤ $500     | —       | ⬜      | Cloud billing      |

### Competition Performance Criteria

1. **Leaderboard Accuracy Threshold**
   - Condition: Public test set accuracy ≥ 85%
   - Verification: Kaggle public leaderboard position in top 1% of all submissions
   - Metric: Proportion of correctly answered questions with exact string match or relative numerical tolerance
   - Deadline: June 15, 2026 (final submission)

2. **Generalization Validation**
   - Condition: Private test accuracy > Public test accuracy
   - Verification: Competition final leaderboard comparison showing positive generalization gap
   - Metric: (Private_Accuracy - Public_Accuracy) > 0%
   - Significance: Demonstrates anti-overfitting; model learned reasoning not memorization

3. **Submission Compliance**
   - Condition: `submission.zip` passes official competition validation
   - Verification: Competition submission checker confirms:
     - `adapter_config.json` present and valid
     - LoRA rank ≤ 32 (verified via config inspection)
     - Compatible with vLLM inference engine
     - File size within competition limits
   - Metric: Binary pass/fail with error log if failed

### Technical Implementation Criteria

4. **Synthetic Data Quality**
   - Condition: Generated dataset passes quality gates
   - Verification:
     - ≥ 10,000 synthetic problems generated from failure analysis
     - LLM-as-judge composite score ≥ 0.80 for retained problems
     - MinHash deduplication removes < 5% near-duplicates (indicates diversity)
     - Zero test set leakage confirmed via n-gram overlap analysis
   - Metric: Dataset statistics report with histograms

5. **Training Convergence**
   - Condition: SFT and GRPO training complete without divergence
   - Verification:
     - SFT: Validation loss decreases monotonically for ≥ 3 epochs before plateau
     - GRPO: Mean reward increases monotonically for ≥ 30 steps
     - KL divergence < 0.05 throughout GRPO (prevents catastrophic forgetting)
   - Metric: Training logs with loss curves and reward plots

6. **Budget Forcing Efficacy**
   - Condition: Test-time adaptation improves hard problems without degrading easy problems
   - Verification:
     - Hard problem subset (top 20% by length/complexity): Accuracy improvement ≥ 5% with budget forcing vs. without
     - Easy problem subset (bottom 20%): Accuracy degradation ≤ 2% with budget forcing vs. without
     - Overall: Net accuracy improvement ≥ 3%
   - Metric: Stratified evaluation report with per-difficulty-bin accuracy

### Documentation & Reproducibility Criteria

7. **Public Notebook Publication**
   - Condition: Kaggle notebook published and executable
   - Verification:
     - Notebook runs end-to-end without errors on Kaggle T4x2 GPU
     - All cells execute within 4-hour Kaggle session limit
     - Clear cell outputs showing intermediate results (sample generations, metrics)
   - Metric: Notebook "Fork" count + community comments

8. **Methodology Write-Up Completeness**
   - Condition: Write-up covers all mandatory sections for prize eligibility
   - Verification:
     - Synthetic data generation pipeline explained with code snippets
     - GRPO training configuration documented with hyperparameter rationale
     - Budget forcing algorithm described with pseudocode
     - Ablation studies present for each component (baseline, SFT-only, SFT+GRPO, full pipeline)
     - Statistical significance testing (p-values) for accuracy differences
   - Metric: Write-up word count ≥ 2,000 words with ≥ 5 figures/tables

9. **Open Contribution Award Qualification**
   - Condition: Submission ranks in top 10% of final leaderboard
   - Verification: Final leaderboard position ≤ 10th percentile
   - Metric: Eligibility for Best Data Method, Best RL Method, or Best Fine-tuning Method awards

### Project Management Criteria

10. **Timeline Adherence**
    - Condition: All phases complete within 25-day competition window
    - Verification:
      - Phase 1 (Data): Complete by Day 7
      - Phase 2 (SFT): Complete by Day 14
      - Phase 3 (GRPO): Complete by Day 21
      - Phase 4 (Submission): Complete by Day 25 (June 15 deadline)
    - Metric: Gantt chart with actual vs. planned completion dates

11. **Compute Budget Compliance**
    - Condition: Total training cost within Google Cloud G4 VM allocation
    - Verification:
      - GPU hours logged per phase
      - No out-of-memory (OOM) errors in training logs
      - Gradient checkpointing and efficient batching utilized
    - Metric: Total compute cost ≤ $500 USD (estimated G4 VM cost for 25 days)
