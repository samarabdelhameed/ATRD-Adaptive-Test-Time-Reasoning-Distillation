# 18 — Configuration & Scripts Specification

## ATRD: Adaptive Test-Time Reasoning Distillation

### 1. Purpose and Setup Order
This specification defines the configuration files and automation scripts that support the entire pipeline. Configs centralize hyperparameters; scripts automate verification, packaging, and synchronization.

> [!IMPORTANT]
> This is a cross-cutting specification. Configs and scripts should be implemented alongside their corresponding phases.

---

## 2. Configuration Files (`configs/`)

### 2.1 File Inventory

| File | Purpose | Mutability |
|------|---------|------------|
| `competition_params.json` | Official competition inference parameters | **IMMUTABLE** |
| `base_lora.json` | Default LoRA config from NVIDIA recipes | Read-only (copy to custom) |
| `base_grpo.json` | Default GRPO config from TRL | Read-only (copy to custom) |
| `custom_lora.json` | Editable LoRA config for experiments | Editable |

### 2.2 competition_params.json Schema
```json
{
  "model_name": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
  "max_lora_rank": 32,
  "max_tokens": 7680,
  "top_p": 1.0,
  "temperature": 0.0,
  "max_num_seqs": 64,
  "gpu_memory_utilization": 0.85,
  "max_model_len": 8192,
  "answer_format": "\\boxed{}",
  "numerical_tolerance": 0.01,
  "inference_engine": "vllm",
  "reasoning_parser": "nvidia-reasoning-parser"
}
```

### 2.3 base_lora.json Schema
```json
{
  "r": 32,
  "lora_alpha": 64,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "use_rslora": false,
  "init_lora_weights": "gaussian"
}
```

### 2.4 base_grpo.json Schema
```json
{
  "group_size": 8,
  "kl_penalty": 0.001,
  "learning_rate": 5e-6,
  "batch_size": 1,
  "gradient_accumulation_steps": 8,
  "max_grad_norm": 1.0,
  "num_train_epochs": 1,
  "max_steps": 500,
  "warmup_ratio": 0.1,
  "logging_steps": 10,
  "save_steps": 50,
  "eval_steps": 50,
  "bf16": true,
  "gradient_checkpointing": true,
  "use_vllm": false
}
```

---

## 3. Automation Scripts (`scripts/`)

### 3.1 verify_unit_completion.py
**Purpose**: Phase gate verification — checks artifacts, LoRA rank, and protected files before allowing phase progression.

```bash
# Usage
python scripts/verify_unit_completion.py P1    # Phase 1 gate
python scripts/verify_unit_completion.py P2    # Phase 2 gate
python scripts/verify_unit_completion.py P3    # Phase 3 gate
python scripts/verify_unit_completion.py P4    # Phase 4 gate
```

**Checks per phase**:
| Phase | Required Artifacts | Invariants |
|-------|-------------------|------------|
| P1 | competition_params.json, base_lora.json | All present |
| P2 | Competition params + LoRA config | Rank ≤ 32 |
| P3 | Competition params + GRPO config | Rank ≤ 32 |
| P4 | competition_params.json | Submission valid |

### 3.2 package_submission.py
**Purpose**: Creates `submission.zip` with validated adapter config and weights.

```bash
# Package adapter
python scripts/package_submission.py \
    --adapter-path checkpoints/grpo/final_adapter \
    --output submission.zip

# Validate only (dry run)
python scripts/package_submission.py \
    --adapter-path checkpoints/grpo/final_adapter \
    --dry-run
```

**Validation pipeline**:
1. Schema check: `adapter_config.json` has valid `r`, `task_type`, `target_modules`
2. Rank check: `r ≤ 32` (disqualification risk if violated)
3. File size: Total < 100 MB
4. Required files: `adapter_config.json` must be present
5. vLLM compatibility test (optional dry-run mode)

### 3.3 verify_protected_files.py
**Purpose**: Pre-commit hook that prevents modification of immutable competition files.

```bash
# Run manually or via git pre-commit hook
python scripts/verify_protected_files.py
```

**Protected files**:
- `configs/competition_params.json`

**Behavior**: Checks `git diff --cached` for protected files. If any are staged, blocks the commit with a clear error message.

### 3.4 sync_to_hub.py
**Purpose**: Uploads LoRA adapter checkpoints to Hugging Face Hub for backup and sharing.

```bash
# Sync SFT adapter
python scripts/sync_to_hub.py \
    --adapter-path checkpoints/sft/final_adapter \
    --repo-id samar/atrd-nemotron-sft-r32 \
    --message "SFT Phase 2: rank-32 adapter after 3 epochs"

# Sync GRPO adapter
python scripts/sync_to_hub.py \
    --adapter-path checkpoints/grpo/final_adapter \
    --repo-id samar/atrd-nemotron-grpo-r32 \
    --message "GRPO Phase 3: RL-optimized with G=8"

# Make public (default: private)
python scripts/sync_to_hub.py --adapter-path ... --repo-id ... --public
```

---

## 4. Requirements File (`requirements.txt`)

### 4.1 Pinned Dependencies
```txt
torch==2.3.1
transformers==4.40.2
peft==0.11.1
trl==0.9.6
accelerate==0.30.1
bitsandbytes==0.43.1
vllm==0.5.0
datasets==2.19.1
pandas==2.2.2
numpy==1.26.4
scipy==1.13.1
python-dotenv==1.0.1
tqdm==4.66.4
huggingface-hub==0.23.2
matplotlib==3.9.0
seaborn==0.13.2
ipython==8.24.0
```

---

## 5. Exit Quality Gate
- [ ] All 4 config files exist with correct schemas
- [ ] All 4 scripts execute without errors
- [ ] competition_params.json marked as protected (pre-commit hook blocks edits)
- [ ] `verify_unit_completion.py` correctly validates each phase
- [ ] `package_submission.py` creates valid zip with dry-run mode
- [ ] `sync_to_hub.py` connects to Hugging Face Hub
- [ ] `requirements.txt` has all pinned versions
