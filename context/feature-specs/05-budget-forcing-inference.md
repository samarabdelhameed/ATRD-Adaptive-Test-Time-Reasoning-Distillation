# 05 — Budget Forcing & Inference Specification

## Phase 4: Test-Time Adaptation & Submission Packaging

### 1. Purpose and Setup Order
This specification defines the implementation details for dynamic compute scaling at inference time (Test-Time Adaptation) and packaging the final LoRA adapter config and weights into a compliant competition submission.

---

## 2. Technical Components to Implement

### 2.1 Difficulty Estimator (`src/inference/budget_forcer.py`)
- **Input**: User prompt/question.
- **Action**: Estimate problem complexity based on text length, keywords (e.g., "integral", "matrix", "theorem"), and prompt characteristics.
- **Output**: Complexity rating (float $[0.0, 1.0]$) mapping to token budget:
  - *Easy*: 256–512 tokens.
  - *Medium*: 1024–4096 tokens.
  - *Hard*: 7680 tokens.

### 2.2 Token-Level Budget Forcer (`src/inference/budget_forcer.py`)
- **Action**: Monitor the generated token stream during vLLM decoding.
- **Logic**:
  - Intercept the generation when the model attempts to emit the end-of-thinking token `</thinking>`.
  - If the problem is classified as *Hard* and the number of generated tokens is below the allocated budget threshold:
    - Replace the token with a mathematical extension trigger: `"Wait, "` or `"Let me double check... "`.
    - Force the model to continue generation to expand its thinking path.
  - If the problem is *Easy* and the token count exceeds 512:
    - Inject the `</thinking>` token and force final answer generation.

### 2.3 vLLM Engine Wrapper (`src/inference/vllm_engine.py`)
- **Action**: Initialize and configure the vLLM engine matching the competition rules.
- **Configuration**:
  - `temperature = 0.0`
  - `max_model_len = 8192`
  - `max_tokens = 7680`
  - `gpu_memory_utilization = 0.85`
- **Output**: Standardized inference interface.

### 2.4 Answer Extractor (`src/evaluation/metric.py` / `src/inference/vllm_engine.py`)
- **Input**: Model completion string.
- **Action**: Extract final answers from mathematical `\boxed{}` formats using regex.
- **Fallback**: If `\boxed{}` is missing, implement fallback heuristics (e.g., extracting final isolated numbers or strings).

### 2.5 Submission Packager (`scripts/package_submission.py`)
- **Action**: Automate the creation of the final `submission.zip`.
- **Steps**:
  1. Retrieve active LoRA adapter weights (`adapter_model.safetensors` or `adapter_model.bin`) and `adapter_config.json` from `checkpoints/grpo_checkpoint/`.
  2. Validate that `adapter_config.json` specifies a rank $r \le 32$.
  3. Validate that the weights load successfully on a dummy instance.
  4. Zip the files into `submission.zip` in the root workspace.

---

## 3. Exit Quality Gate
Before submitting:
- [ ] Overall evaluation accuracy achieves $>85\%$.
- [ ] `submission.zip` package size and config schema matches competition rules.
- [ ] `verify_unit_completion.py P4` returns success.
