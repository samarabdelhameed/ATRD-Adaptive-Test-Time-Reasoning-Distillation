# 13 — GRPO Training Notebook Specification

## ATRD: Adaptive Test-Time Reasoning Distillation

### 1. Purpose and Setup Order
This specification defines the complete `03_grpo_training.ipynb` notebook structure. This notebook executes Phase 3 end-to-end: SFT checkpoint loading → PRM setup → GRPO training → evaluation → checkpoint export.

> [!IMPORTANT]
> Read specs `11-implicit-prm-setup.md` and `12-grpo-training-loop.md` before implementing. Phase 2 must be complete.

---

## 2. Notebook Structure

### Cell 1: Imports and Reproducibility Setup
```python
import random, numpy as np, torch, os, sys, json, re
from pathlib import Path
from typing import List, Dict, Optional, Callable

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
```

### Cell 2: Configuration
```python
@dataclass(frozen=True)
class Phase3Config:
    BASE_MODEL: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
    SFT_CHECKPOINT: str = "/kaggle/input/atrd-sft-adapter"
    GRPO_CONFIG: str = "configs/base_grpo.json"
    GROUP_SIZE: int = 8
    KL_PENALTY: float = 0.001
    LEARNING_RATE: float = 5e-6
    MAX_STEPS: int = 500
    OUTPUT_DIR: Path = Path("/kaggle/working/checkpoints/grpo")
    LOG_DIR: Path = Path("/kaggle/working/logs")
```

### Cell 3: Load SFT Model + LoRA
```python
from src.models.loader import ModelLoader
from peft import PeftModel

loader = ModelLoader()
base_model = loader.load_model(quantize=True)
tokenizer = loader.load_tokenizer()

# Load SFT adapter
model = PeftModel.from_pretrained(base_model, config.SFT_CHECKPOINT)
model.print_trainable_parameters()
```

### Cell 4: Setup PRM Scorer
```python
from src.training.grpo_trainer import GRPOTrainerWrapper

trainer = GRPOTrainerWrapper(
    model=model,
    tokenizer=tokenizer,
    output_dir=str(config.OUTPUT_DIR),
)

reward_fn = trainer.create_reward_function(tolerance=0.01)
print("Reward function created with format + correctness + redundancy components")
```

### Cell 5: Load GRPO Training Data
```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="/kaggle/input/final_train_dataset.jsonl")["train"]
grpo_train = dataset.select(range(min(2000, len(dataset))))  # Subset for GRPO
print(f"GRPO training set: {len(grpo_train)} problems")
```

### Cell 6: Train GRPO
```python
print("Starting GRPO training...")
print(f"Group size G={config.GROUP_SIZE}, KL penalty={config.KL_PENALTY}")
print(f"Max steps: {config.MAX_STEPS}")

result = trainer.train(
    train_dataset=grpo_train,
    reward_function=reward_fn,
)

trainer.save_adapter("checkpoints/grpo/final_adapter")
```

### Cell 7: Reward & KL Monitoring
- Load `trainer_state.json` from checkpoint
- Plot reward trajectory (step vs mean reward)
- Plot KL divergence trajectory
- Verify monotonic reward increase
- Verify KL < 0.05

### Cell 8: Evaluation
- Run inference on public benchmark subset
- Compare accuracy: baseline vs SFT vs GRPO
- Generate 10 sample reasoning traces for manual inspection
- Check for reward hacking (garbage text with correct answer)

### Cell 9: Export to Hugging Face Hub
```python
from scripts.sync_to_hub import sync_adapter

sync_adapter(
    adapter_path="checkpoints/grpo/final_adapter",
    repo_id="samar/atrd-nemotron-grpo-r32",
    commit_message="GRPO Phase 3: RL-optimized policy after 500 steps with G=8",
    private=True,
)
```

### Cell 10: Cleanup
```python
import gc
del model, base_model, trainer
torch.cuda.empty_cache()
gc.collect()
```

---

## 3. Monitoring Dashboard (In-Cell Outputs)

Every 10 steps, print:
```
Step 100/500 | Reward: 0.742 | KL: 0.012 | LR: 5.0e-6
Step 110/500 | Reward: 0.758 | KL: 0.011 | LR: 4.9e-6
...
```

Every 50 steps, save checkpoint + log full metrics to `logs/grpo_rewards.json`.

---

## 4. Exit Quality Gate
- [ ] All cells execute sequentially without errors
- [ ] Mean reward increases monotonically over training
- [ ] KL divergence < 0.05 throughout training
- [ ] Final accuracy > SFT accuracy
- [ ] No reward hacking in sample outputs (manual inspection)
- [ ] `grpo_checkpoint/final_adapter/` saved with adapter_config.json
- [ ] Adapter synced to Hugging Face Hub (private repo)
- [ ] `logs/p3_grpo_eval.json` with reward curves and accuracy comparison
- [ ] `verify_unit_completion.py P3` returns success
