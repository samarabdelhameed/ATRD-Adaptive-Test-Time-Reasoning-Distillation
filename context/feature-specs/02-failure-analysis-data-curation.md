# 02 — Failure Analysis & Data Curation Specification

## Phase 1: Data Pipeline and Corpus Curation

### 1. Purpose and Setup Order
This specification defines the implementation details for the first phase of the ATRD pipeline: assessing the base model, extracting specific failure modes, generating targeted synthetic training problems, and preparing a balanced training dataset.

---

## 2. Technical Components to Implement

### 2.1 Baseline Evaluator (`src/data/dataset_mixer.py` & `src/evaluation/metric.py`)
- **Input**: Nemotron-3-Nano-30B base model (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`) + public validation benchmark.
- **Action**: Run zero-shot inference with a fixed prompt format on the evaluation set.
- **Output**: `baseline_results.json` containing `question`, `predicted_answer`, `extracted_answer`, `ground_truth`, and `is_correct`.

### 2.2 Failure Mode Extractor (`src/data/judge_filter.py` or separate scripts)
- **Input**: `baseline_results.json`.
- **Action**: Identify all incorrect answers, analyze the reasoning paths, and categorize them into five core failure modes:
  1. *Arithmetic/Calculation Errors*: Basic arithmetic slip-ups in steps.
  2. *Reasoning Loop*: Model repeating statements without arriving at a final answer.
  3. *Algebraic Manipulation*: Failure to properly isolate or solve math equations.
  4. *Format Violations*: Missing `\boxed{}` formatting around the final answer.
  5. *Early Termination*: Ending reasoning before conclusion.
- **Output**: `failure_modes.json` with a structured list of failure examples per category.

### 2.3 Targeted Synthetic Generator (`src/data/synthetic_generator.py`)
- **Input**: `failure_modes.json` + API key for a frontier model (DeepSeek-R1 / Qwen3-235B).
- **Action**: Prompt the frontier model to generate variations targeting the weaknesses.
- **Prompt Template**:
  ```text
  You are an expert math tutor. The student model failed on this problem:
  [Insert Problem Example]
  The failure reason was: [Insert Failure Category Description].
  Generate 10 similar problems of similar difficulty, each with a complete step-by-step thinking trace inside <<thinking>>...</thinking> and a final numerical answer in \boxed{}.
  ```
- **Output**: `raw_synthetic_dataset.jsonl` (Target: 10,000–50,000 samples).

### 2.4 LLM-as-Judge Quality Filter (`src/data/judge_filter.py`)
- **Input**: `raw_synthetic_dataset.jsonl`.
- **Action**: Use a judge LLM to score each generated sample on correctness of answer, reasoning clarity, and format compliance.
- **Rules**: Discard all problems scoring below 0.80 or failing the `\boxed{}` regex structure.
- **Output**: `filtered_synthetic_dataset.jsonl`.

### 2.5 MinHash Deduplicator (`src/data/deduplicator.py`)
- **Input**: `filtered_synthetic_dataset.jsonl` + existing datasets (OpenMathReasoning/OpenCodeReasoning).
- **Action**: Extract MinHash signatures (using shingling) and filter out near-duplicates (Jaccard similarity threshold > 0.70).
- **Output**: Deduplicated dataset lines.

### 2.6 Dataset Mixer (`src/data/dataset_mixer.py`)
- **Input**: Deduplicated synthetic dataset + OpenMath/OpenCode datasets.
- **Action**: Create a stratified training mix: 75% reasoning (with thinking steps) + 25% non-reasoning data (standard QA) to prevent general degradation.
- **Output**: `final_train_dataset.jsonl` uploaded to Kaggle Datasets.

---

## 3. Exit Quality Gate
Before proceeding to Phase 2, verify:
- [ ] At least 10,000 synthetic questions exist.
- [ ] Leakage check: 0 matches in 5-gram overlap analysis against the public test set.
- [ ] `verify_unit_completion.py P1` returns success.
