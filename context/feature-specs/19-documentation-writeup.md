# 19 — Documentation & Write-Up Specification

## ATRD: Adaptive Test-Time Reasoning Distillation

### 1. Purpose and Setup Order
This specification defines the documentation requirements for competition submission: public Kaggle notebook, methodology write-up, and Open Contribution Award applications.

> [!IMPORTANT]
> All phases must be complete before writing final documentation. Results from all ablations must be available.

---

## 2. Public Kaggle Notebook

### 2.1 Notebook Requirements
- **Platform**: Kaggle Notebooks (T4x2 or P100 GPU)
- **Session Limit**: Must run end-to-end within 4 hours
- **Reproducibility**: All seeds fixed (42), packages pinned
- **Outputs**: Clear cell outputs showing intermediate results

### 2.2 Notebook Structure
| Section | Cells | Content |
|---------|-------|---------|
| 1 | 1 | Title, author, competition info |
| 2 | 1 | Imports + seed fixing |
| 3 | 1 | Configuration display |
| 4 | 2 | Baseline evaluation + failure mode visualization |
| 5 | 2 | Synthetic data generation (or load pre-generated) |
| 6 | 2 | Quality filtering + deduplication summary |
| 7 | 3 | QLoRA model loading + LoRA config |
| 8 | 3 | SFT training + loss curves |
| 9 | 3 | GRPO training + reward curves |
| 10 | 2 | Budget forcing implementation + demonstration |
| 11 | 3 | Final evaluation + ablation studies |
| 12 | 1 | Submission packaging + validation |
| 13 | 1 | Conclusion + award applications |

### 2.3 Required Visualizations
| Visualization | Location | Description |
|--------------|----------|-------------|
| Failure Mode Distribution | Section 4 | Bar chart of 5 failure categories |
| Training Loss Curve | Section 8 | Step vs loss with convergence annotation |
| Reward Progression | Section 9 | Step vs mean reward with KL overlay |
| Ablation Waterfall | Section 11 | Component contribution breakdown |
| Budget Forcing Impact | Section 11 | Before/after comparison by difficulty bin |
| GPU Memory Timeline | Section 7 | Memory usage during model loading |

---

## 3. Methodology Write-Up (`writeup/METHODOLOGY.md`)

### 3.1 Required Sections
| Section | Word Count | Content |
|---------|-----------|---------|
| 1. Abstract | 200 | Problem, approach, key results |
| 2. Introduction | 300 | Competition background, baseline challenges |
| 3. Failure-Grounded Data Generation | 400 | Process, prompt engineering, quality filtering |
| 4. LoRA Fine-Tuning (SFT) | 300 | QLoRA setup, hyperparameters, convergence |
| 5. GRPO + PRM Reinforcement Learning | 400 | Reward design, group size rationale, PRM mechanism |
| 6. Budget Forcing at Test Time | 300 | Difficulty estimation, Wait injection, results |
| 7. Ablation Studies | 300 | Component contributions, statistical significance |
| 8. Results & Analysis | 200 | Final accuracy, generalization gap |
| 9. Open Contribution Awards | 200 | Data, RL, Fine-tuning method descriptions |
| **Total** | **≥ 2,600** | **Exceeds 2,000 minimum** |

### 3.2 Write-Up Format
```markdown
# ATRD: Adaptive Test-Time Reasoning Distillation

## 1. Abstract

[Problem statement, approach overview, key results with numbers]

## 2. Introduction

[Competition background, baseline model performance, challenges]

## 3. Failure-Grounded Data Generation

### 3.1 Baseline Evaluation
[Process, results, failure taxonomy table]

### 3.2 Synthetic Generation
[Prompt template, frontier model choice, quality control]

### 3.3 Filtering & Deduplication
[LLM-as-judge criteria, MinHash parameters, mixing ratio]

## 4. LoRA Fine-Tuning (SFT)

### 4.1 QLoRA Configuration
[4-bit NF4 quantization, rank-32, target modules]

### 4.2 Training Setup
[Hyperparameters table, data formatting, monitoring]

### 4.3 Results
[Loss curves, accuracy improvement, sample generations]

## 5. GRPO + PRM Reinforcement Learning

### 5.1 Reward Design
[Component breakdown: format + correctness + PRM + penalty]

### 5.2 Training Dynamics
[Group size G=8, KL penalty, convergence behavior]

### 5.3 PRM Correlation
[Step-level scoring validation]

## 6. Budget Forcing

### 6.1 Difficulty Estimation
[Heuristic features, tier mapping]

### 6.2 Wait Injection Mechanism
[Token-level control, max 3 injections]

### 6.3 Impact Analysis
[Stratified results: easy vs hard problems]

## 7. Ablation Studies

| Component | Accuracy | Delta | p-value |
|-----------|----------|-------|---------|
| Baseline | XX% | — | — |
| +SFT | XX% | +XX% | < 0.05 |
| +GRPO | XX% | +XX% | < 0.05 |
| +Budget Forcing | XX% | +XX% | < 0.05 |

## 8. Results

[Final public and private accuracy, generalization gap]

## 9. Open Contribution Awards

### Best Data Method
[Failure-grounded synthetic generation innovation]

### Best RL Method
[PRM-guided GRPO with implicit scoring]

### Best Fine-tuning Method
[Adaptive budget forcing LoRA]
```

---

## 4. Open Contribution Award Applications

### 4.1 Best Data/Synthetic Data Method
- **Title**: "Failure-Grounded Synthetic Data for Targeted Reasoning Improvement"
- **Key Innovation**: Generating training data from model-specific failure modes
- **Evidence**: Ablation showing +11% from SFT data

### 4.2 Best RL Method
- **Title**: "Implicit PRM-Guided GRPO for Structured Reasoning"
- **Key Innovation**: Log-ratio step scoring without separate PRM model
- **Evidence**: Ablation showing +5% from GRPO

### 4.3 Best Fine-Tuning Method
- **Title**: "Adaptive Budget Forcing for Test-Time Compute Scaling"
- **Key Innovation**: Dynamic token allocation with Wait injection
- **Evidence**: Ablation showing +3% from budget forcing, +17% on hard problems

---

## 5. Exit Quality Gate
- [ ] Kaggle notebook runs end-to-end without errors in fresh session
- [ ] All cells execute within 4-hour Kaggle limit
- [ ] Methodology write-up ≥ 2,000 words with ≥ 5 figures/tables
- [ ] Ablation studies present for all 4 configurations
- [ ] Statistical significance (p-values) reported
- [ ] Open Contribution Award applications written
- [ ] Notebook published publicly on Kaggle
- [ ] Write-up saved to `writeup/METHODOLOGY.md`
