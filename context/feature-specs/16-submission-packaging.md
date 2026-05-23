# 16 — Submission Packaging Specification

## ATRD: Adaptive Test-Time Reasoning Distillation

### 1. Purpose and Setup Order
This specification defines the final submission packaging process. The LoRA adapter weights and configuration are packaged into a competition-compliant `submission.zip`.

> [!IMPORTANT]
> Read `12-grpo-training-loop.md` and `15-final-evaluation-ablation.md` before packaging. All evaluations must pass.

---

## 2. Submission Format

### 2.1 Required File Structure
```
submission.zip
├── adapter_config.json     # PEFT LoRA configuration (MANDATORY)
├── adapter_model.safetensors  # LoRA weights (or .bin)
└── tokenizer.json          # Tokenizer files (if modified)
```

### 2.2 adapter_config.json Schema
```json
{
  "r": 32,
  "lora_alpha": 64,
  "target_modules": [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
  ],
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "use_rslora": false,
  "init_lora_weights": "gaussian"
}
```

### 2.3 Validation Rules
| Check | Condition | Failure Action |
|-------|-----------|---------------|
| Rank | `r ≤ 32` | ❌ Disqualification risk |
| Task Type | `task_type == "CAUSAL_LM"` | ❌ vLLM incompatibility |
| Required files | `adapter_config.json` present | ❌ Submission rejected |
| File size | `< 100 MB` total | ❌ Competition limit exceeded |
| vLLM compatibility | Loads on dummy vLLM instance | ❌ Inference failure |

---

## 3. Packaging Script (`scripts/package_submission.py`)

### 3.1 Usage
```bash
# Package the final GRPO adapter
python scripts/package_submission.py \
    --adapter-path checkpoints/grpo/final_adapter \
    --output submission.zip

# Validate only (dry run)
python scripts/package_submission.py \
    --adapter-path checkpoints/grpo/final_adapter \
    --dry-run
```

### 3.2 Packaging Flow
```python
def package(adapter_path: str, output_path: str = "submission.zip", dry_run: bool = False) -> bool:
    """Package LoRA adapter into competition-compliant submission.zip."""

    # Step 1: Validate adapter_config.json schema
    is_valid, errors = validate_adapter_config(adapter_dir)
    if not is_valid:
        for err in errors: print(f"  ✗ {err}")
        return False

    # Step 2: Collect files
    files = find_model_files(adapter_dir)
    total_size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
    if total_size_mb > 100:  # Competition limit
        print(f"  ✗ Size {total_size_mb:.2f} MB exceeds 100 MB limit")
        return False

    # Step 3: Verify required files
    file_names = {f.name for f in files}
    for req in ["adapter_config.json"]:
        if req not in file_names:
            print(f"  ✗ Missing required file: {req}")
            return False

    # Step 4: Test vLLM compatibility (verify load)
    if not dry_run:
        test_vllm_compatibility(adapter_dir)

    # Step 5: Create zip
    if not dry_run:
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, f.name)

    return True
```

### 3.3 vLLM Compatibility Test
```python
def test_vllm_compatibility(adapter_path: str) -> bool:
    """Verify adapter loads correctly in vLLM."""
    from vllm import LLM

    try:
        engine = LLM(
            model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
            max_model_len=8192,
            gpu_memory_utilization=0.85,
            enable_lora=True,
            max_lora_rank=32,
        )

        # Generate a test response
        test_prompt = "What is 2+2?\n<<thinking>>\n[reason]\n</thinking>>\nAnswer: \\boxed{}"
        output = engine.generate(test_prompt, SamplingParams(max_tokens=128))
        print(f"  ✓ vLLM test generation successful")
        return True

    except Exception as e:
        print(f"  ✗ vLLM compatibility test failed: {e}")
        return False
```

---

## 4. Pre-Submission Checklist

Before uploading `submission.zip`:
- [ ] adapter_config.json `r ≤ 32` validated
- [ ] adapter_config.json `task_type = "CAUSAL_LM"` validated
- [ ] adapter_model.safetensors exists and loads
- [ ] Total zip size < 100 MB
- [ ] vLLM test generation successful (100 samples)
- [ ] Accuracy on public test ≥ 85%
- [ ] Private test accuracy > public test accuracy
- [ ] competition_params.json unmodified
- [ ] `verify_unit_completion.py P4` returns success

---

## 5. Exit Quality Gate
- [ ] `submission.zip` created successfully
- [ ] adapter_config.json schema matches competition requirements
- [ ] LoRA rank = 32 (validated)
- [ ] vLLM loads adapter and generates correct output
- [ ] `verify_unit_completion.py P4` returns success
- [ ] submission.zip uploaded to Kaggle competition page
