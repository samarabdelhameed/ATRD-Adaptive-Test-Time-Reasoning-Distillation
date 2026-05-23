# 09 — SFT Training Execution Specification

## ATRD: Adaptive Test-Time Reasoning Distillation

### 1. Purpose and Setup Order
This specification defines the supervised fine-tuning loop execution. The SFT phase teaches the model structured reasoning format using the curated dataset from Phase 1.

> [!IMPORTANT]
> Read `08-qlora-model-setup.md` before implementing. Model must be loaded with LoRA adapter.

---

## 2. Training Configuration

### 2.1 Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | `2e-4` | Standard for LoRA fine-tuning |
| Batch Size (per device) | `1` | Memory constraint (30B model) |
| Gradient Accumulation Steps | `8` | Effective batch size = 8 |
| Max Sequence Length | `4096` | Balances context and memory |
| Warmup Steps | `100` | Stable training start |
| Optimizer | `adamw_torch_fused` | Memory-efficient fused AdamW |
| Learning Rate Scheduler | `cosine` | Smooth convergence |
| Number of Epochs | `3` | Prevents overfitting (small synthetic dataset) |
| Max Grad Norm | `1.0` | Gradient clipping for stability |
| Logging Steps | `10` | Sufficient for loss monitoring |
| Save Steps | `50` | Every 50 steps (~30 min at 1 batch/8 GA) |

### 2.2 Training Arguments (`src/training/sft_trainer.py`)
```python
training_args = TrainingArguments(
    output_dir="checkpoints/sft",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=50,
    eval_steps=50,
    evaluation_strategy="steps",
    save_total_limit=3,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    report_to="none",  # No W&B to save Kaggle session time
)
```

### 2.3 Data Formatting
```python
def format_sft_example(example: dict) -> str:
    """Format with Nemotron reasoning parser tokens."""
    prompt = example["question"]

    if "thinking_trace" in example:
        thinking = example["thinking_trace"]
    else:
        thinking = "<<thinking>>\n[Reason step by step]\n</thinking>>"

    answer = example["answer"]
    return f"{prompt}\n\n{thinking}\n\nAnswer: {answer}"
```

### 2.4 Monitoring & Early Stopping
```python
def should_early_stop(loss_history: list, patience: int = 3) -> bool:
    """Stop if validation loss plateaus for `patience` evaluations."""
    if len(loss_history) < patience + 1:
        return False
    recent = loss_history[-patience:]
    return max(recent) - min(recent) < 0.01  # Plateau threshold
```

---

## 3. Training Loop

### 3.1 TRL SFTTrainer Integration
```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    max_seq_length=4096,
)
result = trainer.train()
print(f"Training complete. Final loss: {result.training_loss:.4f}")
```

### 3.2 Checkpointing Strategy
| Condition | Action |
|-----------|--------|
| Every 50 training steps | Save `checkpoint-{step}/` with adapter weights |
| Kaggle 3-hour mark | Force save to prevent session timeout loss |
| Training complete | Save `final_adapter/` with `adapter_config.json` + `adapter_model.safetensors` |

---

## 4. Evaluation After SFT

### 4.1 Metrics to Track
| Metric | Target | When |
|--------|--------|------|
| Training Loss | Decrease monotonically | Every 10 steps |
| Validation Loss | Plateau after 3 epochs | Every 50 steps |
| Sample Quality | Structured reasoning with `\boxed{}` | Every 500 steps |
| Accuracy | ≥ 10% absolute improvement over baseline | End of training |

### 4.2 Sample Generation Test
```python
def test_generation(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## 5. Exit Quality Gate
- [ ] Training loss decreases monotonically over 3 epochs
- [ ] Validation loss converges (plateau detected)
- [ ] No GPU OOM errors during training
- [ ] Sample generation shows structured `<<thinking>>` + `\boxed{}` format
- [ ] Validation accuracy ≥ 10% absolute improvement over baseline
- [ ] `sft_checkpoint/final_adapter/` saved with `adapter_config.json`
- [ ] adapter_config.json validated: rank = 32
- [ ] `sft_results.json` saved to `logs/`
