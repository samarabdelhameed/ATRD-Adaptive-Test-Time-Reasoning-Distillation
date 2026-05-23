# 03 — Supervised Fine-Tuning Specification

## Phase 2: QLoRA SFT Fine-Tuning

### 1. Purpose and Setup Order
This specification defines the implementation details for training the initial LoRA adapter on the curated dataset `final_train_dataset.jsonl` using Supervised Fine-Tuning (SFT). This establishes the structured reasoning token layout for Nemotron.

---

## 2. Technical Components to Implement

### 2.1 Model Loader (`src/models/loader.py`)
- **Input**: Base model name `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`.
- **Action**: Load model in 4-bit (NF4) quantization using `bitsandbytes` to stay within RTX PRO 6000 Blackwell GPU memory limits (16 GB).
- **Settings**:
  - `load_in_4bit = True`
  - `bnb_4bit_quant_type = "nf4"`
  - `bnb_4bit_compute_dtype = torch.bfloat16`
  - `bnb_4bit_use_double_quant = True`
- **Output**: Quantized model instance ready for PEFT wrapping.

### 2.2 LoRA Configuration Validator (`src/models/lora_config.py`)
- **Action**: Instantiate and validate the LoRA adapter configuration.
- **Parameters**:
  - `r = 32` (Maximum allowed constraint)
  - `lora_alpha = 64`
  - `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
  - `lora_dropout = 0.05`
  - `bias = "none"`
  - `task_type = "CAUSAL_LM"`
- **Constraint Enforcement**: Raise an `AssertionError` if `r > 32` is configured.

### 2.3 SFT Trainer (`src/training/sft_trainer.py`)
- **Input**: Quantized base model + LoRA config + `final_train_dataset.jsonl`.
- **Action**: Execute causal language modeling fine-tuning.
- **Formatting**:
  - Encapsulate reasoning steps in custom special tokens: `<<thinking>... </thinking>`.
  - Train only on the loss of the output completions, ignoring the prompts (using `DataCollatorForCompletionOnlyLM`).
- **Hyperparameters**:
  - Learning Rate: `2e-4`
  - Warmup Steps: `100`
  - Batch Size: `1` (Gradient accumulation `8`)
  - Max Sequence Length: `4096`
  - Optim: `adamw_torch_fused` or `paged_adamw_8bit`
  - Gradient Checkpointing: `True`
- **Output**: Saved adapter weights under `checkpoints/sft_checkpoint/`.

### 2.4 SFT Evaluation (`src/evaluation/metric.py` / `02_sft_training.ipynb`)
- **Input**: Trained SFT adapter checkpoint.
- **Action**: Run validation on the public benchmark set using `vLLM` with the adapter applied.
- **Comparison**: Compare validation accuracy against `baseline_results.json`.
- **Output**: `sft_results.json` containing accuracy, latency, and sample reasoning completions.

---

## 3. Exit Quality Gate
Before proceeding to Phase 3, verify:
- [ ] Training loss decreased monotonically and converged.
- [ ] No GPU Out-Of-Memory (OOM) failures occurred.
- [ ] Validation accuracy shows a $\ge 10\%$ absolute improvement over the baseline.
- [ ] `verify_unit_completion.py P2` returns success.
