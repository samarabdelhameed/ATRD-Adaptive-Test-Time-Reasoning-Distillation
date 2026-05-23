# 10 — SFT Training Notebook Specification

## ATRD: Adaptive Test-Time Reasoning Distillation

### 1. Purpose and Setup Order
This specification defines the complete `02_sft_training.ipynb` notebook structure. This notebook executes Phase 2 end-to-end: model loading → LoRA setup → SFT training → evaluation → checkpoint saving.

> [!IMPORTANT]
> Read specs `08-qlora-model-setup.md` and `09-sft-training-execution.md` before implementing. Phase 1 must be complete (`final_train_dataset.jsonl` must exist).

---

## 2. Notebook Structure

### Cell 1: Imports and Reproducibility Setup
```python
import random, numpy as np, torch, os, sys, json
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
```

### Cell 2: Configuration
```python
@dataclass(frozen=True)
class Phase2Config:
    BASE_MODEL: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
    LORA_RANK: int = 32
    LORA_ALPHA: int = 64
    LEARNING_RATE: float = 2e-4
    BATCH_SIZE: int = 1
    GRADIENT_ACCUMULATION_STEPS: int = 8
    MAX_SEQ_LENGTH: int = 4096
    NUM_EPOCHS: int = 3
    DATASET_PATH: str = "/kaggle/input/final_train_dataset.jsonl"
    OUTPUT_DIR: Path = Path("/kaggle/working/checkpoints/sft")
    LOG_DIR: Path = Path("/kaggle/working/logs")
```

### Cell 3: Load Model + LoRA
```python
from src.models.loader import ModelLoader
from src.models.lora_config import create_lora_config
from peft import get_peft_model

loader = ModelLoader()
base_model = loader.load_model(quantize=True)
tokenizer = loader.load_tokenizer()

lora_config = create_lora_config("configs/base_lora.json")
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
```

### Cell 4: Prepare Dataset
```python
from datasets import load_dataset
from src.training.sft_trainer import SFTTrainerWrapper

dataset = load_dataset("json", data_files=config.DATASET_PATH)["train"]
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_data = train_test_split["train"]
eval_data = train_test_split["test"]
```

### Cell 5: Train
```python
trainer = SFTTrainerWrapper(model=model, tokenizer=tokenizer)
train_dataset = trainer.prepare_dataset(train_data)
eval_dataset = trainer.prepare_dataset(eval_data)
result = trainer.train(train_dataset, eval_dataset)
trainer.save_adapter("checkpoints/sft/final_adapter")
```

### Cell 6: Evaluation
- Run inference on public benchmark subset
- Compare accuracy vs baseline
- Generate sample reasoning traces (5 examples)
- Save `sft_results.json` to `logs/`

### Cell 7: Visualization
- Plot training loss curve (step vs loss)
- Plot validation loss curve
- Display sample generations in formatted markdown

### Cell 8: Sync to Hugging Face Hub
```python
from scripts.sync_to_hub import sync_adapter

sync_adapter(
    adapter_path="checkpoints/sft/final_adapter",
    repo_id="samar/atrd-nemotron-sft-r32",
    commit_message="SFT Phase 2: LoRA rank-32 after 3 epochs on synthetic data",
    private=True,
)
```

### Cell 9: Cleanup
```python
import gc
del model, base_model, trainer
torch.cuda.empty_cache()
gc.collect()
```

---

## 3. Kaggle-Specific Considerations

| Constraint | Strategy |
|-----------|----------|
| 4-hour session | Checkpoint every 50 steps (≈30 min) |
| GPU Memory | QLoRA 4-bit + gradient checkpointing |
| No internet after setup | Download model + data in first cell |
| Save frequency | `save_steps=50`, `save_total_limit=3` |

---

## 4. Exit Quality Gate
- [ ] All cells execute sequentially without errors
- [ ] Training loss decreases monotonically and converges
- [ ] No GPU OOM errors
- [ ] Validation accuracy ≥ 10% improvement over baseline
- [ ] `sft_checkpoint/final_adapter/` saved with adapter_config.json
- [ ] Adapter synced to Hugging Face Hub (private repo)
- [ ] `logs/p2_sft_eval.json` with accuracy metrics
- [ ] Sample generations show structured reasoning with `\boxed{}`
