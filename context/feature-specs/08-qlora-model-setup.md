# 08 — QLoRA Model Setup Specification

## ATRD: Adaptive Test-Time Reasoning Distillation

### 1. Purpose and Setup Order
This specification defines the model loading and LoRA configuration for Phase 2 supervised fine-tuning. The model must load in 4-bit quantization within RTX PRO 6000 Blackwell memory constraints.

> [!IMPORTANT]
> Read `context/code-standards.md` and `configs/competition_params.json` before implementing.

---

## 2. Technical Components

### 2.1 4-Bit NF4 Quantization (`src/models/loader.py`)

#### 2.1.1 BitsAndBytes Configuration
```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NormalFloat4 optimal for normally distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16,  # Blackwell native support
    bnb_4bit_use_double_quant=True,    # Double quantization for additional memory savings
)
```

#### 2.1.2 Model Loading
```python
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
```

### 2.2 GPU Memory Management
```python
def load_model_with_cleanup(model_name: str) -> PreTrainedModel:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(model_name, ...)

    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory: {allocated:.2f} GB / 16 GB")
    if allocated > 14:
        print("WARNING: Near memory limit. Reduce batch size.")
    return model
```

### 2.3 LoRA Configuration (`src/models/lora_config.py`)

#### 2.3.1 Competition-Compliant Configuration
| Parameter | Value | Constraint |
|-----------|-------|------------|
| `r` | 32 | Maximum allowed (rank ≤ 32) |
| `lora_alpha` | 64 | ≥ rank |
| `target_modules` | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | All linear layers |
| `lora_dropout` | 0.05 | Prevents overfitting on small synthetic dataset |
| `bias` | "none" | Standard LoRA practice |
| `task_type` | "CAUSAL_LM" | Nemotron is a causal language model |

#### 2.3.2 Validation
```python
def validate_lora_config(config: dict) -> None:
    assert config["r"] <= 32, "LoRA rank exceeds competition maximum of 32"
    assert config["lora_alpha"] >= config["r"], "Alpha must be ≥ rank"
    assert config["lora_dropout"] < 0.5, "Dropout too high for small datasets"
```

### 2.4 Tokenizer Setup
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"  # For autoregressive generation
```

### 2.5 Gradient Checkpointing
```python
model.gradient_checkpointing_enable()
print("Gradient checkpointing enabled — trading compute for memory")
```

---

## 3. Blackwell-Specific Optimizations

```python
def setup_blackwell_optimizations():
    """Configure CUDA for RTX PRO 6000 Blackwell."""
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    print(f"GPU: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")

    # Blackwell is compute capability 10.x
    if props.major >= 10:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Blackwell TF32 optimizations enabled")

    # Match competition parameter
    torch.cuda.set_per_process_memory_fraction(0.85)
    print("Memory fraction set to 85%")
```

---

## 4. Exit Quality Gate
- [ ] Model loads in 4-bit NF4 without OOM
- [ ] GPU memory < 14 GB after loading (leaves headroom for training)
- [ ] LoRA config validated: rank = 32, alpha = 64, 7 target modules
- [ ] Tokenizer has pad_token set to eos_token
- [ ] Gradient checkpointing enabled
- [ ] Blackwell TF32 optimizations applied if compute capability ≥ 10.x
