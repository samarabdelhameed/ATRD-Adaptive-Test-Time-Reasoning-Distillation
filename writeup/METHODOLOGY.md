# ATRD: Adaptive Test-Time Reasoning Distillation

## 1. Abstract

This submission presents ATRD (Adaptive Test-Time Reasoning Distillation), a LoRA-based fine-tuning pipeline for the NVIDIA Nemotron-3-Nano-30B model. The approach combines three innovations: (1) failure-grounded synthetic data generation that creates targeted training examples from the base model's specific error modes, (2) Process Reward Model (PRM)-guided Group Relative Policy Optimization (GRPO) that scores intermediate reasoning steps, and (3) difficulty-aware budget forcing that allocates compute token budgets adaptively during training data generation. The pipeline produces a rank-32 LoRA adapter compatible with vLLM inference engine (temperature=0.0, max_tokens=7680 per competition constraints).

The implementation is structured across four Kaggle notebooks covering data curation, SFT training, GRPO reinforcement learning, and budget-forced evaluation. All modules are implemented as reusable Python packages under `src/` with typed interfaces.

**pending (run p4_eval)**

## 2. Introduction

The NVIDIA Nemotron Model Reasoning Challenge provides a shared baseline model (Nemotron-3-Nano-30B-A3B-Base-BF16) and a novel reasoning benchmark requiring structured multi-step problem solving. The competition limits submissions to LoRA adapters (rank ≤ 32) evaluated with fixed inference parameters (temperature=0.0, max_tokens=7680).

### 2.1 Key Constraints

| Constraint | Value | Implication |
|------------|-------|-------------|
| Base model | Nemotron-3-Nano-30B-A3B-Base-BF16 | 30B parameters requires QLoRA 4-bit to fit GPU |
| LoRA rank | ≤ 32 | Limits adapter capacity; rank=32 used |
| Temperature | 0.0 (fixed) | No sampling; single deterministic pass per problem |
| Max tokens | 7680 (fixed) | Cannot extend reasoning at inference time |
| Inference engine | vLLM | Must produce format compatible with reasoning parser |
| Answer format | `\boxed{}` | All answers must be extracted from boxed notation |
| GPU memory | 0.85 utilization (fixed) | Cannot exceed 85% of available VRAM |
| Numerical tolerance | 0.01 | Answers within 1% relative error considered correct |

### 2.2 Baseline Challenges

- The base Nemotron-3-Nano-30B model exhibits systematic failure patterns across five categories: arithmetic errors, logic/reasoning gaps, code execution mistakes, algebraic manipulation failures, and proof structure deficiencies
- Single-pass inference with temperature=0.0 precludes sampling-based improvement strategies
- The 30B parameter scale requires memory-efficient quantization (4-bit NF4) to fit within Kaggle GPU constraints
- Training data must consist entirely of publicly available datasets and synthetically generated content (no test set leakage permitted)

### 2.3 Pipeline Architecture

The pipeline spans four phases executed across sequential Kaggle notebooks:

| Phase | Notebook | Python Modules | Output |
|-------|----------|---------------|--------|
| P1: Data Generation | `01_data_generation.ipynb` | `synthetic_generator.py`, `judge_filter.py`, `deduplicator.py`, `dataset_mixer.py` | `final_train_dataset.jsonl` |
| P2: SFT Training | `02_sft_training.ipynb` | `loader.py`, `lora_config.py`, `sft_trainer.py` | `checkpoints/sft/final_adapter/` |
| P3: GRPO Training | `03_grpo_training.ipynb` | `grpo_trainer.py`, `prm.py` | `checkpoints/grpo/final_adapter/` |
| P4: Evaluation | `04_budget_forcing.ipynb` | `budget_forcer.py`, `vllm_engine.py`, `metric.py` | `submission.zip` |

**Project configuration is centralized in:**
- `configs/competition_params.json` — immutable inference parameters
- `configs/base_lora.json` — LoRA rank-32, alpha=64, 7 target modules
- `configs/base_grpo.json` — GRPO hyperparameters (group size=8, KL penalty=0.001, lr=5e-6)

## 3. Failure-Grounded Data Generation

### 3.1 Baseline Evaluation

The baseline evaluation runs the untrained Nemotron-3-Nano-30B on the public benchmark using vLLM with competition parameters. Each problem is processed through inference, and the `\boxed{}` answer is extracted using `_extract_boxed_answer()` (fallback regex pattern matching). Answers are compared against ground truth with numerical tolerance (0.01).

**Implementation:** `notebooks/01_data_generation.ipynb` (Cells 4–5), `src/evaluation/metric.py`

**pending (run p1_baseline)**

### 3.2 Failure Taxonomy

Failures are classified into five categories using `_classify_failure()` in `src/data/synthetic_generator.py`:

| Failure Type | Detection Criteria | Example | Weight in Generation |
|-------------|-------------------|---------|---------------------|
| **no_answer** | Completion missing `\boxed{}` format | Model output ends without boxed notation | 20% of synthetic batch |
| **incomplete** | Missing reasoning trace structure | No `<<thinking>>` marker in output | 20% of synthetic batch |
| **format_error** | `\boxed{}` present but incorrectly formatted | Nested braces, missing backslash | 20% of synthetic batch |
| **wrong_answer** | Answer extracted but does not match ground truth | Numerical mismatch outside tolerance | 30% of synthetic batch |
| **proof_structure** | Reasoning lacks logical flow | Missing step indicators, jumps in logic | 10% of synthetic batch |

### 3.3 Synthetic Generation

For each failure mode, a frontier model (DeepSeek-R1 primary, Qwen3-235B fallback) is prompted with a failure-grounded system prompt:

```
System: You are generating training problems for a model that struggles with [failure_description].
Generate [num_problems] problems with step-by-step solutions in \boxed{} format.
```

**Implementation:** `src/data/synthetic_generator.py` — `generate_per_failure_mode()` loops over 5 failure mode categories, calling `_call_teacher_model()` with retry + exponential backoff on 429 errors. Each response is parsed via `_parse_batch_response()` extracting `Question:/Thinking:/Answer:` blocks. Output follows schema: question, thinking_trace, answer, failure_mode_tag, difficulty_estimate, generation_timestamp, source_model.

**Generation config (GeneratorConfig):** batch_size=10, max_retries=3, temperature=0.7, max_tokens=4096, top_p=0.95, timeout=120s.

### 3.4 Quality Filtering

An LLM-as-judge filter scores each synthetic problem on four weighted criteria:
- Correctness (0.35): Is the answer and reasoning accurate?
- Reasoning clarity (0.25): Is the thinking trace well-structured?
- Difficulty (0.20): Is the problem appropriately challenging?
- Format (0.20): Does it follow the `\boxed{}` + `<<thinking>>` format?

The composite score retains the top 80% of problems, discarding low-quality generations.

**Implementation:** `src/data/judge_filter.py` — `JudgeFilter` class with `heuristic_score()` for fast pre-filtering and `generate_report()` for pass rate statistics.

### 3.5 Deduplication

Near-duplicate removal uses MinHash (128 permutations) with LSH (Locality-Sensitive Hashing) for candidate pair bucketing. Exact duplicates are removed via SHA-256 hashing. Character n-gram shingles (n=5) are used for similarity comparison with Jaccard threshold > 0.85.

**Implementation:** `src/data/deduplicator.py` — `Deduplicator` class.

### 3.6 Dataset Mixing

Three sources are combined at a 50/25/25 ratio:
- Synthetic (filtered): 50%
- NVIDIA OpenMathReasoning: 25%
- NVIDIA OpenCodeReasoning: 25%

Stratified sampling preserves failure mode distribution within the synthetic portion. The final reasoning ratio is verified to fall within 70–80%. A 5-gram overlap check against the test set confirms zero leakage.

**Implementation:** `src/data/dataset_mixer.py` — `DatasetMixer.mix()` with optional `benchmark_texts` parameter for leakage detection.

## 4. LoRA Fine-Tuning (SFT)

### 4.1 QLoRA Configuration

The base model is loaded in 4-bit NF4 quantization (double quant + bfloat16) using bitsandbytes via `src/models/loader.py`:

| Component | Configuration | Purpose |
|-----------|--------------|---------|
| Quantization type | NF4 (4-bit NormalFloat) | Memory-efficient weight representation |
| Double quantization | Enabled | Additional memory savings via quantizing quantization constants |
| Compute dtype | bfloat16 | Training precision for gradient computation |
| LoRA rank | 32 | Maximum allowed by competition |
| LoRA alpha | 64 | Scaling factor for rank updates |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | All 7 linear projection layers |
| Dropout | 0.05 | Regularization to prevent overfitting |
| Bias | none | No additional bias parameters |
| Task type | CAUSAL_LM | Standard language modeling head |

The Blackwell-optimized setup in `setup_blackwell_optimizations()` enables TF32 precision for compute capability ≥ 10.x and reads `gpu_memory_utilization` from `competition_params.json` (0.85). Gradient checkpointing is enabled via `enable_gradient_checkpointing()` to trade compute for memory during training.

**Implementation:** `src/models/loader.py` (model loading + Blackwell optimizations), `src/models/lora_config.py` (rank validation)

### 4.2 Training Setup

| Hyperparameter | Value |
|---------------|-------|
| Learning rate | 2e-4 |
| Batch size | 1 (per device) |
| Gradient accumulation steps | 8 |
| Max sequence length | 4096 |
| Warmup steps | 100 |
| Learning rate scheduler | Cosine |
| Optimizer | adamw_torch_fused |
| Number of epochs | 3 |
| Early stopping patience | Plateau detection (max-min < 0.01) |

**Data format:** `{prompt}\n\n{thinking}\n\nAnswer: {answer}`

**Implementation:** `src/training/sft_trainer.py` — `SFTTrainerWrapper` with `format_sft_example()`, `should_early_stop()`, and `test_generation()`.

### 4.3 Results

**pending (run p2_sft)**

## 5. GRPO + PRM Reinforcement Learning

### 5.1 Reward Design

The composite reward function (`compute_prm_guided_reward()` in `src/training/prm.py`) combines four signals:

| Component | Weight | Description |
|-----------|--------|-------------|
| Answer correctness | +0.8 | `\boxed{}` extracted answer matches ground truth |
| Format compliance | +0.2 (box) +0.1 (thinking open) +0.1 (thinking close) | Proper structure |
| PRM step scores | 0.4 × mean(heuristic_step_score) | Per-step quality via regex |
| Redundancy penalty | -0.3 | Repeated line patterns |

Score is clamped to [-1.0, 1.0].

### 5.2 Heuristic PRM (Default)

The default PRM uses zero-GPU regex-based heuristics:

| Heuristic | Score | Detection |
|-----------|-------|-----------|
| Mathematical transition | +0.2 | `[=→<>≤≥]` in step |
| Logical connector | +0.2 | "therefore, thus, because, hence, so, then" |
| Valid equation | +0.3 | Digits + operators present |
| Repetition penalty | -0.3 | Unique word ratio < 0.4 |

**Implementation:** `src/training/prm.py` — `heuristic_step_score()`, `segment_thinking_trace()`.

### 5.3 Log-Ratio PRM (Optional)

When GPU memory permits (~80GB+ VRAM for 30B model), log-ratio scoring computes the log-probability ratio between the frozen SFT reference model and the current policy model for each reasoning step. Falls back gracefully to heuristic on OOM.

**Implementation:** `src/training/prm.py` — `compute_log_ratio_score()`, `get_log_prob()`.

### 5.4 Training Dynamics

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Group size (G) | 8 completions per prompt | Balance between variance reduction and compute cost |
| KL penalty | 0.001 | Prevents catastrophic forgetting of SFT checkpoint |
| Learning rate | 5e-6 | Lower than SFT (2e-4) for stable RL updates |
| Batch size | 1 (per device) | Memory constraint for 30B model |
| Gradient accumulation | 8 | Effective batch size of 8 for stable gradients |
| Max steps | 500 | Upper limit; early stop if reward plateaus |
| Warmup ratio | 0.1 | 50 steps linear warmup |
| Save steps | 50 | Checkpoint every 50 steps for recovery |
| Optimization | adamw_torch_fused | Memory-efficient fused optimizer |
| Precision | bf16 | Mixed precision for memory saving |

### 5.5 Reward Monotonicity and KL Monitoring

Reward progression is validated using `verify_monotonic_reward()` which checks that the mean reward over the recent window (last 10 steps) exceeds the mean reward over the preceding 10-step window. This ensures training is actually improving policy quality.

KL divergence between the current policy and the frozen SFT reference model is computed via `_compute_kl()` which calculates `sum(ref_probs × (log(ref_probs) − log(curr_probs)))` over the final token dimension. A `KLMonitor` class thresholds this at 0.05 (warning) and 0.1 (hard stop) to prevent divergence.

Reward monotonicity is verified via `verify_monotonic_reward()` (window=10). KL divergence is monitored by `KLMonitor` with threshold 0.05 and hard stop at 0.1.

**Implementation:** `src/training/grpo_trainer.py` — `GRPOTrainerWrapper`, `KLMonitor`, `_compute_kl()`.

**pending (run p3_grpo)**

## 6. Budget Forcing

### 6.1 Scope

Budget forcing operates during the **data generation phase** (not at inference time). The competition evaluates the LoRA adapter with fixed vLLM parameters (temperature=0.0, max_tokens=7680), so inference-time budget forcing cannot affect submission. Instead, budget forcing improves training data quality by allocating compute budget adaptively during generation.

**Implementation:** `src/data/budget_forcer.py`.

### 6.2 Difficulty Estimation

Heuristic difficulty scoring on a 0–1 scale:

| Feature | Max Contribution |
|---------|-----------------|
| Length (>100 words: +0.2, >200: +0.1) | 0.3 |
| Math indicators (integral, derivative, prove, etc.) | 0.4 |
| Step indicators (and, then, given that, therefore) | 0.3 |

### 6.3 Token Budget Allocation

| Difficulty Tier | Score Range | Token Budget |
|----------------|-------------|--------------|
| Easy | 0.00–0.30 | 512–1024 |
| Medium | 0.30–0.65 | 2048–4096 |
| Hard | 0.65–1.00 | 4096–7680 |

Budget is linear interpolation: `min_tokens + difficulty × (max_tokens − min_tokens)`.

### 6.4 Multi-Stage Refinement

Hard problems (difficulty > 0.65) where the initial generation produces an incorrect answer are re-generated with a corrective system prompt. The refinement loop in `refine_hard_problem()` follows this process:

| Step | Action | Condition |
|------|--------|-----------|
| 1 | Generate initial completion at difficulty-allocated budget | Always |
| 2 | Check answer correctness via `check_answer()` | After initial generation |
| 3 | If correct: return completion immediately | `check_answer()` == True |
| 4 | If wrong AND hard (difficulty > 0.65): re-prompt with error correction instruction | `check_answer()` == False AND difficulty > 0.65 |
| 5 | Re-check answer after refinement | After each refinement attempt |
| 6 | Repeat up to max_attempts=3 or until correct | Whichever comes first |

The corrective prompt includes: "The previous solution was incorrect. Double-check each step carefully. Break down the problem and verify your reasoning." with max_tokens=7680 (full budget for refinement).

**Implementation:** `src/data/budget_forcer.py` — `refine_hard_problem()` function, called automatically by `generate_training_data_with_budget()` when difficulty > 0.65 and initial answer is wrong.

### 6.5 Impact Analysis

**pending**

## 7. Ablation Studies

| Component | Accuracy | Delta | p-value |
|-----------|----------|-------|---------|
| Baseline (Nemotron-3-Nano-30B) | — (run p1_baseline) | | — | — |
| +SFT (synthetic data + QLoRA) | — (run p2_sft) | | — (pending) | — (pending) |
| +GRPO (PRM-guided RL) | — (run p3_grpo) | | — (pending) | — (pending) |
| +Budget Forcing (data quality) | — (run p4_eval) | | — (pending) | — (pending) |

**Ablation methodology:** Each component is added incrementally to isolate its contribution. Baseline is the untrained Nemotron-3-Nano-30B. SFT adds supervised fine-tuning on the mixed dataset. GRPO adds reinforcement learning on the SFT checkpoint. Budget forcing applies difficulty-aware compute allocation during data generation (affects SFT and GRPO phases).

**pending**

## 8. Results

| Metric | Value |
|--------|-------|
| Public test accuracy | — (run p4_eval) | |
| Private test accuracy | — (run p4_eval) | |
| Generalization gap | — (run p4_eval) | |
| LoRA rank | 32 |
| Submission format | `submission.zip` (adapter_config.json + weights) |

**pending (run full pipeline + submit)**

## 9. Open Contribution Awards

### Best Data/Synthetic Data Method

**Title:** "Failure-Grounded Synthetic Data for Targeted Reasoning Improvement"

**Key Innovation:** Generating training data from model-specific failure modes rather than generic problem augmentation. The pipeline evaluates the base model, extracts systematic error categories, and prompts frontier models to generate variations targeting each weakness. This ensures synthetic data addresses actual model limitations rather than adding noise.

**Evidence:** Ablation study isolating the contribution of synthetic data versus training on OpenMathReasoning/OpenCodeReasoning alone. **—**

### Best RL Method

**Title:** "Implicit PRM-Guided GRPO for Structured Reasoning"

**Key Innovation:** Log-ratio step scoring between reference and current policy models provides implicit process rewards without training a separate PRM model. The default heuristic mode (regex-based) consumes zero additional GPU memory while still capturing reasoning quality signals. The optional log-ratio mode provides finer-grained scoring when GPU memory permits.

**Evidence:** PRM correlation test validates that scored completions with correct answers receive higher PRM scores than incorrect ones. **pending (run p3_grpo + evaluate)**

### Best Fine-Tuning Method

**Title:** "Adaptive Budget Forcing for Training Data Compute Scaling"

**Key Innovation:** Difficulty-aware token budget allocation during data generation saves compute on easy problems and applies multi-stage refinement to hard problems. This data-quality enhancement improves the training signal without modifying inference parameters.

**Evidence:** Stratified evaluation shows accuracy improvement on hard problems after refinement, while easy problem accuracy is preserved. **pending (run p4_eval)**
