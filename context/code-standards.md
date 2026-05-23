# Code Standards

### General Principles

- **Keep functions single-purpose and under 50 lines**
  - One function = one logical operation (load model, generate batch, compute reward)
  - If a function needs comments to explain its multiple steps, it should be split
  - Exception: Configuration dictionaries and dataclass definitions

- **Fix root causes, do not layer workarounds**
  - OOM error? Reduce batch size or enable gradient checkpointing — don't just restart kernel
  - API rate limit? Implement exponential backoff with jitter — don't just sleep(60)
  - Loss divergence? Reduce learning rate or check data quality — don't just clip gradients blindly
  - Bad synthetic data? Fix the prompt template — don't just filter more aggressively

- **Do not mix unrelated concerns in one notebook cell**
  - Cell for imports and seeds only
  - Cell for configuration only
  - Cell for helper functions only
  - Cell for data loading only
  - Cell for model loading only
  - Cell for training loop only
  - Cell for evaluation only
  - Cell for visualization only

- **Prefer explicit over implicit**
  - `learning_rate=2e-4` not `lr=2e-4` (abbreviations confuse)
  - `max_sequence_length=4096` not `max_len=4096`
  - `num_train_epochs=3` not `epochs=3`
  - `gradient_accumulation_steps=8` not `grad_acc=8`

- **Fail fast with descriptive errors**
  - No bare `except:` or `except Exception:`
  - Catch specific exceptions: `except torch.cuda.OutOfMemoryError`, `except requests.HTTPError`
  - Error messages must include: what failed, why it failed, suggested fix

- **Never suppress warnings without documentation**
  - `warnings.filterwarnings("ignore")` must be accompanied by comment explaining why
  - Better: fix the root cause causing the warning

---

### Python & ML-Specific Standards

#### Type Hints (Mandatory)

```python
# ✅ CORRECT
from typing import Optional, List, Dict, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

def generate_synthetic_problems(
    failure_mode: str,
    num_problems: int,
    frontier_model: str,
    api_key: str,
    temperature: float = 0.7,
    max_tokens: int = 2048
) -> List[Dict[str, str]]:
    """
    Generate synthetic problems targeting a specific failure mode.
    
    Args:
        failure_mode: Category of reasoning failure (e.g., "algebraic_manipulation")
        num_problems: Number of problems to generate (1-100 per batch)
        frontier_model: Model identifier for API call
        api_key: Authentication key for frontier model API
        temperature: Sampling temperature for generation
        max_tokens: Maximum tokens per generation
        
    Returns:
        List of problem dictionaries with keys: question, thinking_trace, answer, failure_mode_tag
        
    Raises:
        requests.HTTPError: If API call fails after retries
        ValueError: If num_problems > 100 (rate limit protection)
    """
    ...

# ❌ INCORRECT
def gen_probs(failure, n, model, key):
    # No types, no docstring, unclear parameter names
    ...
```

#### Data Structures (Mandatory)

```python
from dataclasses import dataclass
from typing import TypedDict

# ✅ CORRECT — TypedDict for JSON-like structures
class SyntheticProblem(TypedDict):
    question: str
    thinking_trace: str
    answer: str
    failure_mode_tag: str
    difficulty_estimate: float
    generation_timestamp: str

# ✅ CORRECT — Dataclass for configuration
@dataclass(frozen=True)
class LoRAConfig:
    r: int = 32  # LoRA rank
    lora_alpha: int = 64
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    )
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    def validate(self) -> None:
        """Validate configuration invariants."""
        assert self.r <= 32, "LoRA rank must be ≤ 32 per competition rules"
        assert self.lora_alpha >= self.r, "Alpha must be ≥ rank"
        assert self.lora_dropout < 0.5, "Dropout too high for small datasets"

# ❌ INCORRECT — Dictionary with magic strings
config = {
    "r": 32,
    "alpha": 64,
    "modules": ["q_proj", ...],  # No validation, no type safety
}
```

#### GPU Memory Management (Mandatory)

```python
import torch

# ✅ CORRECT — Explicit memory management
def load_model_with_cleanup(
    model_name: str,
    device_map: str = "auto"
) -> PreTrainedModel:
    """Load model and verify GPU memory state."""
    
    # Clear cache before loading
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16,  # Explicit dtype
        quantization_config=qlora_config  # Explicit quantization
    )
    
    # Verify load succeeded
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded. GPU memory: {allocated:.2f} GB")
    
    if allocated > 20:  # Alert threshold for RTX PRO 6000
        print("WARNING: High memory usage. Consider smaller batch size.")
    
    return model

# ✅ CORRECT — Cleanup after use
def cleanup_model(model: PreTrainedModel) -> None:
    """Safely unload model and free GPU memory."""
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"Model unloaded. GPU memory: {allocated:.2f} GB")

# ❌ INCORRECT — No memory management
model = AutoModelForCausalLM.from_pretrained("...")  # Load
# ... use model ...
# No cleanup, next cell may OOM
```

#### Error Handling (Mandatory)

```python
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# ✅ CORRECT — Specific exceptions, retry logic, logging
logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((requests.HTTPError, requests.ConnectionError))
)
def call_frontier_model_api(
    prompt: str,
    model: str,
    api_key: str
) -> Dict[str, Any]:
    """
    Call frontier model API with retry and exponential backoff.
    
    Retries on: HTTP errors, connection errors
    Does NOT retry on: 4xx client errors (bad request), invalid API key
    """
    try:
        response = requests.post(
            f"https://api.together.xyz/v1/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": 2048,
                "temperature": 0.7
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
        
    except requests.HTTPError as e:
        if e.response.status_code == 429:
            logger.warning(f"Rate limit hit. Retrying with backoff...")
            raise  # Let @retry handle this
        elif e.response.status_code >= 500:
            logger.error(f"Server error: {e}. Retrying...")
            raise  # Let @retry handle this
        else:
            logger.error(f"Client error: {e}. Not retrying.")
            raise  # Don't retry 4xx errors
        
    except requests.Timeout:
        logger.error(f"API timeout after 30s.")
        raise
        
    except Exception as e:
        logger.critical(f"Unexpected error: {type(e).__name__}: {e}")
        raise

# ❌ INCORRECT — Bare except, no retry, no logging
def bad_api_call(prompt, model, key):
    try:
        response = requests.post("...", json={"prompt": prompt})
        return response.json()
    except:  # Catches KeyboardInterrupt, SystemExit, everything!
        return None  # Silent failure, caller has no idea what happened
```

---

### Notebook Structure Standards

Every Kaggle notebook must follow this exact cell structure:

| Cell # | Content | Purpose |
|--------|---------|---------|
| 1 | Imports + seed fixing | Reproducibility foundation |
| 2 | Configuration (dataclass or YAML) | Centralized hyperparameters |
| 3 | Helper functions (with docstrings) | Reusable utilities |
| 4 | Data loading / model loading | Resource-intensive setup |
| 5+ | Main execution | Training, generation, evaluation |
| Final-2 | Evaluation metrics | Quantitative results |
| Final-1 | Visualization | Loss curves, sample outputs |
| Final | Cleanup + save | `torch.cuda.empty_cache()`, save artifacts |

**Cell 1 Template:**
```python
# Cell 1: Imports and Reproducibility Setup

import random
import numpy as np
import torch
from pathlib import Path

# Fixed seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Deterministic behavior (may slow down slightly)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Seeds fixed to {SEED}. Reproducibility mode enabled.")
```

**Cell 2 Template:**
```python
# Cell 2: Configuration

from dataclasses import dataclass

@dataclass(frozen=True)
class ProjectConfig:
    """Centralized configuration for this notebook."""
    
    # Model
    BASE_MODEL: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
    LORA_RANK: int = 32
    LORA_ALPHA: int = 64
    
    # Training
    LEARNING_RATE: float = 2e-4
    BATCH_SIZE: int = 1
    GRADIENT_ACCUMULATION_STEPS: int = 8
    MAX_SEQ_LENGTH: int = 4096
    NUM_EPOCHS: int = 3
    
    # Paths
    DATA_DIR: Path = Path("/kaggle/input/nemotron-data")
    OUTPUT_DIR: Path = Path("/kaggle/working/checkpoints")
    LOG_DIR: Path = Path("/kaggle/working/logs")
    
    def validate(self) -> None:
        """Verify configuration invariants."""
        assert self.LORA_RANK <= 32, "Competition constraint violated"
        assert self.LEARNING_RATE < 1e-3, "LR too high for stability"
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

# Instantiate and validate
config = ProjectConfig()
config.validate()
print(f"Configuration validated: {config}")
```

---

### Naming Conventions

| Category | Pattern | Example |
|----------|---------|---------|
| Functions | `verb_noun` with specific verbs | `generate_synthetic_batch`, `compute_grpo_reward`, `evaluate_budget_forcing` |
| Classes | `PascalCase`, descriptive | `LoRAConfig`, `GRPOTrainer`, `BudgetForcer` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_LORA_RANK`, `KL_PENALTY`, `SEED` |
| Variables | `snake_case`, no abbreviations | `gradient_accumulation_steps` not `grad_acc` |
| Notebooks | `##_descriptive_name.ipynb` | `01_data_generation.ipynb`, `02_sft_training.ipynb` |
| Checkpoints | `phase_unit_timestamp/` | `p2_sft_20260522_1430/` |
| Logs | `phase_metric_timestamp.json` | `p2_training_loss_20260522.json` |

---

### Documentation Standards

#### Docstring Format (Google Style)

```python
def train_lora_sft(
    model: PreTrainedModel,
    dataset: Dataset,
    config: LoRAConfig,
    output_dir: Path
) -> Path:
    """
    Train LoRA adapter using supervised fine-tuning.
    
    Args:
        model: Base model loaded with QLoRA quantization
        dataset: Training dataset with formatted prompts
        config: LoRA configuration (rank, alpha, target modules)
        output_dir: Directory to save checkpoints and logs
        
    Returns:
        Path to final checkpoint directory
        
    Raises:
        RuntimeError: If training diverges (loss > 2x initial after 100 steps)
        ValueError: If dataset is empty or incompatible format
        
    Example:
        >>> config = LoRAConfig(r=32, lora_alpha=64)
        >>> checkpoint = train_lora_sft(model, dataset, config, Path("./checkpoints"))
        >>> print(f"Checkpoint saved to: {checkpoint}")
    """
    ...
```

#### Inline Comments

```python
# ✅ CORRECT — Explain WHY, not WHAT
# Use bfloat16 on RTX PRO 6000 Blackwell for native hardware support
# and 2x throughput vs float16 on this architecture
torch_dtype = torch.bfloat16

# ✅ CORRECT — Explain non-obvious decision
# Group size G=8 balances variance reduction (larger = better) 
# with memory constraints (larger = more GPU RAM)
grpo_group_size = 8

# ❌ INCORRECT — States the obvious
x = 5  # Set x to 5
```

---

### Testing Standards

#### Unit Tests (Per-Cell Verification)

```python
# After every helper function cell, add verification:

def test_generate_synthetic_batch():
    """Verify synthetic generation produces valid output."""
    problems = generate_synthetic_batch(
        failure_mode="algebraic_manipulation",
        num_problems=5,
        frontier_model="deepseek-r1",
        api_key="test-key"
    )
    
    assert len(problems) == 5, "Wrong batch size"
    assert all("question" in p for p in problems), "Missing question field"
    assert all("thinking_trace" in p for p in problems), "Missing thinking trace"
    assert all("answer" in p for p in problems), "Missing answer"
    assert all(p["answer"].startswith("\\boxed{") for p in problems), "Answer not in boxed format"
    
    print("✅ All tests passed")

# Run immediately
test_generate_synthetic_batch()
```

#### Integration Tests (Per-Phase Verification)

```python
# At end of Phase 1 notebook:

def test_phase1_data_pipeline():
    """Verify complete data pipeline output."""
    dataset = load_dataset("final_train_dataset.jsonl")
    
    assert len(dataset) >= 10000, "Insufficient data"
    assert "reasoning" in dataset.features, "Missing reasoning column"
    assert "non_reasoning" in dataset.features, "Missing non_reasoning column"
    
    # Verify 75/25 split
    reasoning_pct = len(dataset.filter(lambda x: x["type"] == "reasoning")) / len(dataset)
    assert 0.70 <= reasoning_pct <= 0.80, f"Wrong split: {reasoning_pct:.2%}"
    
    # Verify no test leakage
    overlap = check_ngram_overlap(dataset, test_set)
    assert overlap == 0, f"Test leakage detected: {overlap} overlapping n-grams"
    
    print("✅ Phase 1 integration tests passed")

test_phase1_data_pipeline()
```

---

### Version Control Standards

#### Git Commit Messages

```
Format: <type>(<scope>): <description>

Types:
- feat: New feature or capability
- fix: Bug fix
- docs: Documentation update
- refactor: Code restructuring without behavior change
- test: Adding or updating tests
- chore: Maintenance, dependencies, cleanup

Examples:
feat(data): implement failure-grounded synthetic generation
fix(training): resolve OOM in GRPO with G=8 by reducing batch size
docs(notebook): add ablation study section to 03_grpo_training
refactor(inference): extract budget forcing logic into BudgetForcer class
```

#### Kaggle Dataset Versioning

```python
# After generating dataset, version it:

from kaggle.api.kaggle_api_extended import KaggleApi

def version_dataset(
    local_path: Path,
    dataset_slug: str,
    version_notes: str
) -> None:
    """Create new version of Kaggle dataset."""
    api = KaggleApi()
    api.authenticate()
    
    # Create metadata
    metadata = {
        "title": f"Nemotron-ATRD-{dataset_slug}",
        "id": f"your-username/{dataset_slug}",
        "licenses": [{"name": "CC0-1.0"}]
    }
    
    # Save metadata
    with open(local_path / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f)
    
    # Create version
    api.dataset_create_version(
        str(local_path),
        version_notes=version_notes,
        delete_old_versions=False  # Keep history
    )
    
    print(f"✅ Dataset versioned: {dataset_slug} — {version_notes}")

# Usage
version_dataset(
    Path("/kaggle/working/synthetic_data"),
    "nemotron-synthetic-v1",
    "Phase 1: 12,000 failure-grounded problems, filtered top 80%"
)
```

---

### NVIDIA-Specific Standards

#### CUDA & GPU Optimization

```python
# ✅ CORRECT — Verify GPU and optimize for architecture
def setup_cuda():
    """Configure CUDA for RTX PRO 6000 Blackwell."""
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This project requires GPU.")
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    
    print(f"GPU: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"Total Memory: {props.total_memory / 1e9:.2f} GB")
    print(f"Multi-Processor Count: {props.multi_processor_count}")
    
    # Blackwell-specific optimizations
    if props.major >= 10:  # Blackwell is compute capability 10.x
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ Blackwell optimizations enabled (TF32)")
    
    # Memory settings
    torch.cuda.set_per_process_memory_fraction(0.85)  # Match competition parameter
    
    return device

# ✅ CORRECT — Gradient checkpointing for large models
def enable_gradient_checkpointing(model: PreTrainedModel) -> None:
    """Enable gradient checkpointing to trade compute for memory."""
    model.gradient_checkpointing_enable()
    print("✅ Gradient checkpointing enabled")
```

#### Nemotron Tokenizer Handling

```python
# ✅ CORRECT — Preserve reasoning parser tokens
def format_prompt_with_thinking(
    question: str,
    thinking_trace: Optional[str] = None
) -> str:
    """
    Format prompt with Nemotron reasoning parser tokens.
    
    Nemotron uses <<thinking>> and </thinking>> to delimit reasoning.
    These tokens must be preserved exactly for vLLM parser plugin.
    """
    if thinking_trace:
        return f"""{question}

<<thinking>>
{thinking_trace}
</thinking>>

Answer: \\boxed{{...}}"""
    else:
        return f"""{question}

<<thinking>>
[Think step by step]
</thinking>>

Answer: \\boxed{{...}}"""

# ❌ INCORRECT — Manual token IDs or different format
def bad_format(question, trace):
    return f"{question}\n[THINK]{trace}[/THINK]\nAnswer: ..."  # Wrong tokens!
```

---

### Prohibited Patterns

| Pattern | Why Forbidden | Correct Alternative |
|---------|-------------|---------------------|
| `globals()` or `locals()` | Breaks type checking, makes code unmaintainable | Explicit parameter passing |
| `eval()` or `exec()` | Security risk, hides bugs | `ast.literal_eval()` for safe parsing |
| `pickle` for model checkpoints | Not portable, security risk | `safetensors` or `torch.save()` with `weights_only=True` |
| `print()` for logging | No timestamps, no levels, not persistent | `logging` module with file handlers |
| `time.sleep()` for rate limits | Inefficient, not robust | `tenacity` with exponential backoff |
| Hard-coded paths | Breaks on different environments | `Path` objects with config-driven paths |
| Magic numbers | No context, hard to change | Named constants in config |

---

### Verification Checklist

Before committing any code:

- [ ] All functions have type hints and docstrings
- [ ] All variables use descriptive names (no abbreviations)
- [ ] GPU memory is explicitly managed (clear cache, track usage)
- [ ] Error handling is specific (no bare `except:`)
- [ ] Seeds are fixed for reproducibility
- [ ] Notebook cells follow numbered structure
- [ ] Unit tests pass for all helper functions
- [ ] No magic numbers or hard-coded paths
- [ ] No `print()` statements (use `logging`)
- [ ] Git commit message follows `<type>(<scope>): <description>` format


## Python & ML Framework Standards

### Python Language Rules

- **Strict type checking with mypy**
  - All functions must have complete type hints
  - No `Any` type except in legacy API wrappers with `# type: ignore` justification
  - Use `typing` module: `Optional`, `Union`, `List`, `Dict`, `Tuple`, `Callable`
  - Run `mypy --strict` before every commit

- **Validate all external input at system boundaries**
  - API responses: Validate JSON schema with `pydantic` before processing
  - Dataset files: Verify format and checksum before loading
  - Model checkpoints: Verify `adapter_config.json` schema before inference
  - User-provided paths: Resolve with `Path()` and verify existence

- **Explicit error handling over silent failures**
  - No bare `except:` or `except Exception:`
  - Catch specific exceptions with context-rich error messages
  - All file operations wrapped in `try/except` with cleanup in `finally`

- **Immutable data structures where possible**
  - Use `dataclass(frozen=True)` for configurations
  - Use `TypedDict` for JSON-like structures
  - Avoid mutating function arguments

### ML Framework: PyTorch + Transformers

- **Default to explicit device management**
  - Always specify `device` parameter
  - Use `torch.cuda.current_device()` not implicit GPU 0
  - Verify tensor device before operations

- **Add gradient checkpointing only when memory requires it**
  - Monitor `torch.cuda.memory_allocated()` before enabling
  - Document memory savings vs. compute overhead trade-off
  - Disable if training speed becomes bottleneck

- **Keep training loops focused on single responsibility**
  - Data loading: separate function/class
  - Model forward pass: separate
  - Loss computation: separate
  - Optimization step: separate
  - Logging: separate callback

### Hugging Face Ecosystem

- **Use `AutoModel` / `AutoTokenizer` for portability**
  - Never hard-code model class names
  - Verify model type with `model.config.model_type` after loading

- **Save adapters with `save_pretrained()` not manual file copy**
  - Ensures `adapter_config.json` is valid and complete
  - Compatible with PEFT loading conventions

- **Use `Dataset` objects for data, not raw lists**
  - Enables streaming, batching, and memory mapping
  - Compatible with `Trainer` API

### PEFT (LoRA) Conventions

- **LoRA rank ≤ 32 (competition constraint)**
  - Validate in config dataclass: `assert config.r <= 32`
  - Log actual rank at training start

- **Target all linear projection layers**
  - Default: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
  - Document if any layer excluded and why

- **Use `LoraConfig` from PEFT, not manual initialization**
  - Ensures compatibility with `PeftModel` and `merge_and_unload()`

### TRL (GRPO) Conventions

- **Group size G=8 for GRPO (research-validated)**
  - Document if changed due to memory constraints
  - Log group size and effective batch size

- **KL penalty = 1e-3 (Nemotron Ultra default)**
  - Validate KL divergence stays < 0.05 during training
  - Stop if KL > 0.1 (policy diverging too far)

- **Reward function must be verifiable**
  - Binary correctness: exact match or numerical tolerance
  - No subjective or learned reward functions
  - Log reward distribution per batch

### vLLM Integration

- **Use `LLM` class for inference, not raw model**
  - Enables tensor parallelism and continuous batching
  - Required for competition evaluation environment

- **LoRA adapter loaded via `llm.generate(lora_request=...)`**
  - Never modify base model weights directly
  - Verify adapter is active with test generation before evaluation

- **Reasoning parser plugin enabled for Nemotron**
  - Verify `<<thinking>>` and `</thinking>>` tokens are preserved
  - Test parser with sample prompt before full evaluation

### Kaggle Platform Conventions

- **Default to Kaggle Datasets for input, `/kaggle/working/` for output**
  - Input data: Read-only, versioned
  - Output: Writable, ephemeral (save to Kaggle Datasets for persistence)

- **Save checkpoints every 30 minutes or every epoch**
  - Kaggle sessions timeout after 4 hours
  - Use `Trainer` callbacks for automatic checkpointing

- **Internet access disabled after data download**
  - Download all models and datasets in first cell
  - Subsequent cells must work offline

- **Use `kaggle_secrets` for API keys, never hard-code**
  - Add keys via Kaggle UI, access via `UserSecretsClient()`
  - Rotate keys after competition

### Data Processing

- **Use `datasets` library for all data operations**
  - Streaming for large datasets
  - `map()` for transformations
  - `filter()` for quality control

- **MinHash for deduplication, not exact match**
  - Near-duplicate detection catches paraphrases
  - Configurable Jaccard threshold (default 0.85)

- **Stratified splits for train/validation**
  - Preserve failure mode distribution
  - Ensure no data leakage between splits

### Experiment Tracking

- **Use `logging` module, not `print()`**
  - Structured logs with timestamps
  - File handler for persistence
  - Console handler for real-time monitoring

- **Log all hyperparameters at training start**
  - Config dataclass serialized to JSON
  - Git commit hash for reproducibility

- **Save metrics every step, not just at end**
  - Enables early stopping and debugging
  - Use `Trainer` callbacks or manual `jsonlines` logging

### GPU Memory Management

- **Monitor memory before and after every model operation**
  - `torch.cuda.memory_allocated()` / `memory_reserved()`
  - Log peak memory with `torch.cuda.max_memory_allocated()`

- **Clear cache between unrelated operations**
  - `torch.cuda.empty_cache()` after model loading
  - `gc.collect()` after dataset processing

- **Use `accelerate` for multi-GPU if needed**
  - `device_map="auto"` for model parallelism
  - `DeepSpeed` integration for ZeRO optimization

### Testing

- **Unit test every helper function**
  - Mock API calls for data generation
  - Mock model forward for training logic
  - Use `pytest` with `tmp_path` fixture

- **Integration test per phase**
  - End-to-end notebook execution
  - Artifact validation (size, format, checksum)
  - Metric sanity checks (accuracy between 0 and 1)

- **Regression test before submission**
  - Baseline accuracy must not decrease
  - New feature must improve target metric
  - Ablation study confirms contribution

### Version Control

- **Git commit after every completed unit**
  - Descriptive message: `feat(data): generate synthetic batch for algebraic failures`
  - Include phase and unit in message

- **Tag important checkpoints**
  - `baseline-eval`: Initial model evaluation
  - `sft-complete`: Supervised fine-tuning finished
  - `grpo-stable`: RL training converged
  - `submission-ready`: Final adapter validated

- **`.gitignore` for generated artifacts**
  - Checkpoints (large binary files)
  - Logs (frequent changes)
  - API response caches (sensitive data)

### Documentation

- **Docstring for every public function**
  - Google style: Args, Returns, Raises, Example
  - Type information in signature, not docstring

- **README per notebook**
  - Purpose, inputs, outputs, expected runtime
  - Link to upstream and downstream notebooks

- **Architecture Decision Records (ADRs)**
  - `docs/adr/001-qlora-nf4.md`
  - `docs/adr/002-implicit-prm.md`
  - Date, context, decision, consequences

### Security

- **No secrets in code**
  - API keys via environment variables or Kaggle secrets
  - `.env` file in `.gitignore`

- **No PII in datasets**
  - Verify synthetic data contains no real names, emails, etc.
  - Filter if using web-scraped data

- **Model outputs audited before publication**
  - Review sample generations for harmful content
  - Document known limitations and biases


## Visualization & Notebook Styling Standards

### Notebook Aesthetics

- **Use matplotlib/seaborn style templates — no hardcoded colors**
  - Default: `plt.style.use('seaborn-v0_8-whitegrid')` or `plt.style.use('ggplot')`
  - Dark mode: `plt.style.use('dark_background')` for Kaggle dark theme
  - Custom palette defined in config, never inline hex codes

- **Follow the color scale defined in visualization config**
  - Primary: NVIDIA Green `#76B900` for main metrics
  - Secondary: `#1F77B4` (blue) for baselines
  - Success: `#2CA02C` (green) for improvements
  - Warning: `#FF7F0E` (orange) for cautions
  - Error: `#D62728` (red) for failures/divergence
  - Neutral: `#7F7F7F` (gray) for secondary elements

### Matplotlib Configuration

```python
# ✅ CORRECT — Centralized style configuration
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Modern professional style
plt.style.use('seaborn-v0_8-whitegrid')

# Custom color palette (NVIDIA-inspired)
COLORS = {
    'primary': '#76B900',      # NVIDIA Green
    'secondary': '#1F77B4',    # Blue
    'success': '#2CA02C',      # Green
    'warning': '#FF7F0E',      # Orange
    'error': '#D62728',        # Red
    'neutral': '#7F7F7F',      # Gray
    'background': '#FAFAFA',   # Light gray
    'text': '#333333',         # Dark gray
    'grid': '#E0E0E0'          # Light grid
}

# Font configuration for readability
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
rcParams['font.size'] = 11
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 12
rcParams['figure.titlesize'] = 16
rcParams['figure.dpi'] = 150  # High resolution for publications

# Animation-ready backend
rcParams['animation.html'] = 'jshtml'  # For interactive animations in notebooks

def apply_nvidia_style():
    """Apply NVIDIA-inspired professional styling to all plots."""
    rcParams['axes.prop_cycle'] = plt.cycler(color=[
        COLORS['primary'], COLORS['secondary'], 
        COLORS['warning'], COLORS['error'], COLORS['neutral']
    ])
    rcParams['axes.facecolor'] = COLORS['background']
    rcParams['figure.facecolor'] = 'white'
    rcParams['grid.color'] = COLORS['grid']
    rcParams['grid.alpha'] = 0.7
    rcParams['axes.edgecolor'] = COLORS['neutral']
    rcParams['axes.labelcolor'] = COLORS['text']
    rcParams['text.color'] = COLORS['text']
    
apply_nvidia_style()
```

### Chart Types by Purpose

| Purpose | Chart Type | Style Rules |
|---------|-----------|-------------|
| **Training Loss** | Line plot with smoothing | Primary color, shaded confidence interval, annotated convergence point |
| **Reward Curves** | Line plot with dual axis | Primary for reward, secondary for KL divergence |
| **Accuracy Comparison** | Grouped bar chart | Baseline in neutral, improvements in success, degradations in error |
| **Failure Mode Distribution** | Horizontal bar chart | Sorted by frequency, color-coded by severity |
| **Budget Forcing Impact** | Before/after grouped bars | Easy problems (neutral), hard problems (primary) |
| **Ablation Study** | Waterfall chart | Baseline at 0, each component's contribution stacked |
| **GPU Memory Usage** | Area chart with fill | Warning threshold line at 85%, error at 95% |

### Animation Standards

```python
# ✅ CORRECT — Animated training progress
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def animate_training_progress(losses: List[float], rewards: List[float]):
    """
    Create animated visualization of training convergence.
    Modern, professional animation for notebook presentation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('white')
    
    # Setup axes with modern styling
    ax1.set_facecolor(COLORS['background'])
    ax2.set_facecolor(COLORS['background'])
    
    line1, = ax1.plot([], [], color=COLORS['primary'], linewidth=2.5, label='Loss')
    line2, = ax2.plot([], [], color=COLORS['success'], linewidth=2.5, label='Reward')
    
    # Fill under curves for modern look
    fill1 = ax1.fill_between([], [], alpha=0.3, color=COLORS['primary'])
    fill2 = ax2.fill_between([], [], alpha=0.3, color=COLORS['success'])
    
    # Annotations
    step_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12, 
                         fontweight='bold', color=COLORS['text'])
    
    def init():
        ax1.set_xlim(0, len(losses))
        ax1.set_ylim(0, max(losses) * 1.1)
        ax1.set_title('Training Loss Convergence', fontweight='bold', pad=15)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlim(0, len(rewards))
        ax2.set_ylim(min(rewards) * 0.9, max(rewards) * 1.1)
        ax2.set_title('Reward Progression', fontweight='bold', pad=15)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward')
        ax2.grid(True, alpha=0.3)
        
        return line1, line2, fill1, fill2, step_text
    
    def update(frame):
        # Update loss plot with smooth animation
        x = range(frame + 1)
        line1.set_data(x, losses[:frame+1])
        
        # Update fill
        global fill1
        fill1.remove()
        fill1 = ax1.fill_between(x, 0, losses[:frame+1], 
                                  alpha=0.2, color=COLORS['primary'])
        
        # Update reward plot
        line2.set_data(x, rewards[:frame+1])
        global fill2
        fill2.remove()
        fill2 = ax2.fill_between(x, min(rewards)*0.9, rewards[:frame+1], 
                                  alpha=0.2, color=COLORS['success'])
        
        # Update step counter with modern styling
        step_text.set_text(f'Step {frame+1} / {len(losses)}')
        
        # Add value annotations on final frame
        if frame == len(losses) - 1:
            ax1.annotate(f'Final: {losses[-1]:.4f}', 
                        xy=(frame, losses[-1]), 
                        xytext=(frame*0.7, losses[-1]*1.05),
                        arrowprops=dict(arrowstyle='->', color=COLORS['primary']),
                        fontsize=10, fontweight='bold', color=COLORS['primary'])
            
            ax2.annotate(f'Final: {rewards[-1]:.4f}', 
                        xy=(frame, rewards[-1]), 
                        xytext=(frame*0.7, rewards[-1]*1.02),
                        arrowprops=dict(arrowstyle='->', color=COLORS['success']),
                        fontsize=10, fontweight='bold', color=COLORS['success'])
        
        return line1, line2, fill1, fill2, step_text
    
    anim = FuncAnimation(fig, update, init_func=init, 
                         frames=len(losses), interval=100, blit=False)
    
    plt.tight_layout()
    plt.close()  # Prevent double display
    
    return HTML(anim.to_jshtml())

# Usage in notebook
# animate_training_progress(loss_history, reward_history)
```

### Interactive Visualizations

```python
# ✅ CORRECT — Interactive plotly charts for exploration
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_interactive_ablation_chart(ablation_results: Dict[str, float]):
    """
    Modern interactive chart for ablation study presentation.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Component Contribution', 'Cumulative Improvement'),
        specs=[[{"type": "bar"}, {"type": "waterfall"}]]
    )
    
    # Component contribution bars with modern styling
    components = list(ablation_results.keys())
    values = list(ablation_results.values())
    colors = [COLORS['success'] if v > 0 else COLORS['error'] for v in values]
    
    fig.add_trace(
        go.Bar(
            x=components,
            y=values,
            marker_color=colors,
            text=[f'{v:+.2%}' for v in values],
            textposition='outside',
            textfont=dict(size=11, color=COLORS['text']),
            hovertemplate='<b>%{x}</b><br>Improvement: %{y:+.2%}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Waterfall chart for cumulative effect
    cumulative = [0]
    for v in values:
        cumulative.append(cumulative[-1] + v)
    
    fig.add_trace(
        go.Waterfall(
            x=['Baseline'] + components,
            y=[0] + values,
            measure=['absolute'] + ['relative'] * len(values),
            text=[f'{c:.2%}' for c in cumulative],
            textposition='outside',
            connector=dict(line=dict(color=COLORS['neutral'], width=2)),
            decreasing=dict(marker=dict(color=COLORS['error'])),
            increasing=dict(marker=dict(color=COLORS['success'])),
            totals=dict(marker=dict(color=COLORS['primary']))
        ),
        row=1, col=2
    )
    
    # Modern layout
    fig.update_layout(
        title=dict(
            text='Ablation Study: Component Contribution Analysis',
            font=dict(size=18, color=COLORS['text'], family='Arial Black'),
            x=0.5
        ),
        showlegend=False,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor='white',
        font=dict(family='Arial', size=11, color=COLORS['text']),
        height=500,
        margin=dict(t=80, b=60, l=60, r=40),
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family='Arial'
        )
    )
    
    # Grid styling
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=COLORS['grid'])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=COLORS['grid'])
    
    return fig

# Usage: fig.show() in notebook
```

### Progress Bars & Status Indicators

```python
# ✅ CORRECT — Modern progress indicators for long operations
from tqdm.notebook import tqdm
import time

def train_with_progress_bar(model, dataset, num_epochs: int):
    """
    Professional progress tracking with modern styling.
    """
    # Modern tqdm styling
    epoch_bar = tqdm(
        range(num_epochs),
        desc='Training Progress',
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        ncols=800,
        colour='#76B900'  # NVIDIA Green
    )
    
    for epoch in epoch_bar:
        # Batch progress with nested bar
        batch_bar = tqdm(
            dataset,
            desc=f'Epoch {epoch+1}',
            leave=False,
            colour='#1F77B4'
        )
        
        epoch_losses = []
        for batch in batch_bar:
            loss = train_step(model, batch)
            epoch_losses.append(loss)
            
            # Dynamic description update
            batch_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'avg': f'{sum(epoch_losses)/len(epoch_losses):.4f}'
            })
        
        # Epoch summary with modern formatting
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_bar.set_postfix({
            'epoch_loss': f'{avg_loss:.4f}',
            'best': f'{min(epoch_losses):.4f}'
        })
        
        # Visual separator between epochs
        print(f"\n{'─' * 60}")
        print(f"✅ Epoch {epoch+1}/{num_epochs} Complete | Loss: {avg_loss:.4f}")
        print(f"{'─' * 60}\n")
        
        time.sleep(0.1)  # Brief pause for visual clarity

# Usage in notebook
# train_with_progress_bar(model, dataset, num_epochs=3)
```

### Table Styling (Pandas)

```python
# ✅ CORRECT — Modern styled tables for results presentation
import pandas as pd

def style_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply modern professional styling to results DataFrame.
    """
    return df.style \
        .set_properties(**{
            'text-align': 'center',
            'font-size': '12px',
            'font-family': 'Arial'
        }) \
        .set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#76B900'),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('padding', '12px'),
                ('border', '1px solid #5A8F00')
            ]},
            {'selector': 'td', 'props': [
                ('padding', '10px'),
                ('border', '1px solid #E0E0E0')
            ]},
            {'selector': 'tr:nth-child(even)', 'props': [
                ('background-color', '#F5F5F5')
            ]},
            {'selector': 'tr:hover', 'props': [
                ('background-color', '#E8F5E9')
            ]}
        ]) \
        .highlight_max(subset=['Accuracy', 'Reward'], color='#C8E6C9') \
        .highlight_min(subset=['Loss', 'KL_Divergence'], color='#C8E6C9') \
        .format({
            'Accuracy': '{:.2%}',
            'Loss': '{:.4f}',
            'Reward': '{:.4f}',
            'KL_Divergence': '{:.6f}',
            'Training_Time': '{:.1f} min'
        }) \
        .set_caption('📊 ATRD Pipeline Results — Phase Summary')

# Usage
# results_df = pd.DataFrame({...})
# style_results_table(results_df)
```

### Badge & Status Indicators

```python
# ✅ CORRECT — Modern status badges for notebook cells
from IPython.display import HTML, display

def status_badge(status: str, message: str) -> None:
    """
    Display modern status badge in notebook.
    """
    colors = {
        'success': ('#2CA02C', '#E8F5E9', '✅'),
        'warning': ('#FF7F0E', '#FFF3E0', '⚠️'),
        'error': ('#D62728', '#FFEBEE', '❌'),
        'info': ('#1F77B4', '#E3F2FD', 'ℹ️'),
        'running': ('#76B900', '#F1F8E9', '🔄')
    }
    
    border_color, bg_color, icon = colors.get(status, colors['info'])
    
    html = f'''
    <div style="
        display: inline-block;
        padding: 8px 16px;
        background-color: {bg_color};
        border-left: 4px solid {border_color};
        border-radius: 4px;
        font-family: Arial, sans-serif;
        font-size: 13px;
        color: #333;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <span style="font-size: 16px; margin-right: 8px;">{icon}</span>
        <strong style="color: {border_color};">{status.upper()}</strong>
        <span style="margin-left: 8px;">{message}</span>
    </div>
    '''
    
    display(HTML(html))

# Usage in notebook cells
# status_badge('success', 'Phase 1 Complete: 12,000 problems generated')
# status_badge('running', 'Phase 2: SFT Training in progress...')
# status_badge('warning', 'GPU memory at 82% — consider clearing cache')
```

### Prohibited Styling Patterns

| Pattern | Why Forbidden | Correct Alternative |
|---------|-------------|---------------------|
| `plt.plot(x, y, 'r-')` | Hardcoded color, no legend | Use `COLORS` dict with labeled lines |
| `print(f"Loss: {loss}")` | Plain text, no visual hierarchy | Use `status_badge()` or `tqdm` |
| `df.head()` | Unstyled raw output | Use `.style` with professional formatting |
| `plt.figure(figsize=(8,6))` | Default size, no DPI | Use `rcParams` with high DPI and appropriate size |
| Static PNG for training curves | No interactivity, large file size | Use `FuncAnimation` or `plotly` |
| Rainbow color schemes | Unprofessional, accessibility issues | Use `COLORS` palette with semantic meaning |

### Export Standards

```python
# ✅ CORRECT — High-quality export for documentation
def save_publication_figure(fig, filename: str, dpi: int = 300):
    """
    Save figure in multiple formats for documentation.
    """
    # High-res PNG for reports
    fig.savefig(f'{filename}.png', dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # SVG for web/scaling
    fig.savefig(f'{filename}.svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    # PDF for publications
    fig.savefig(f'{filename}.pdf', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"✅ Figure saved: {filename}.{{png,svg,pdf}}")
```

## API Routes (Pipeline Interfaces)

### Definition
In the ATRD project, "API Routes" refer to **contractual interfaces between pipeline phases** — not HTTP endpoints. Each phase exposes a predictable input/output contract that the next phase consumes. These contracts enforce data validation, authorship tracking, and consistent response shapes across the 4-phase pipeline.

---

### Phase Interface Contracts

#### P1 → P2: Data Curation → SFT Training
```python
# Input Contract (P1 Output)
class DataCurationOutput(TypedDict):
    dataset_path: str          # Path to final_train_dataset.jsonl
    num_problems: int          # Total problems after filtering
    reasoning_ratio: float     # Must be 0.75 ± 0.05
    failure_modes: List[str]   # Documented failure categories
    quality_score: float       # LLM-as-judge mean score (≥ 0.80)
    dedup_report: Dict         # MinHash deduplication statistics
    leakage_check: bool        # True if zero test overlap confirmed

# Validation Rule
- [Rule — Validate dataset schema before any training logic runs]
- [Rule — Enforce 75/25 reasoning/non-reasoning ratio before model ingestion]
- [Rule — Return consistent artifact manifest with checksums]
```

#### P2 → P3: SFT Training → GRPO RL
```python
# Input Contract (P2 Output)
class SFTCheckpointOutput(TypedDict):
    checkpoint_path: str       # Path to sft_checkpoint/ directory
    adapter_config: Dict       # Validated adapter_config.json contents
    lora_rank: int             # Must be ≤ 32 (competition invariant)
    validation_loss: float     # Final validation loss
    sample_generations: List[str]  # 10 sample reasoning traces
    convergence_status: str    # "converged" | "plateaued" | "diverged"

# Validation Rule
- [Rule — Validate LoRA rank ≤ 32 before any RL initialization]
- [Rule — Enforce checkpoint integrity (hash verification) before policy mutation]
- [Rule — Return consistent training log shape with step-level metrics]
```

#### P3 → P4: GRPO RL → Budget Forcing
```python
# Input Contract (P3 Output)
class GRPOCheckpointOutput(TypedDict):
    checkpoint_path: str       # Path to grpo_checkpoint/ directory
    mean_reward: float         # Final mean reward across validation
    kl_divergence: float       # Must be < 0.05 (stability invariant)
    reward_trend: List[float]  # Monotonically increasing (verified)
    prm_scores: Dict           # Per-step PRM score distribution
    ablation_sft_vs_grpo: float  # Accuracy delta vs SFT baseline

# Validation Rule
- [Rule — Validate KL divergence < 0.05 before any inference adaptation]
- [Rule — Enforce reward monotonicity before test-time optimization]
- [Rule — Return consistent evaluation manifest with per-difficulty metrics]
```

#### P4 → Submission: Budget Forcing → Competition Export
```python
# Input Contract (P4 Output)
class SubmissionOutput(TypedDict):
    submission_zip: str        # Path to submission.zip
    adapter_validated: bool    # True if vLLM compatibility confirmed
    public_accuracy: float       # Accuracy on public test set
    stratified_results: Dict   # Per-difficulty accuracy breakdown
    budget_force_stats: Dict   # Wait-token injection statistics
    notebook_url: str          # Public Kaggle notebook URL
    writeup_path: str          # Path to methodology markdown

# Validation Rule
- [Rule — Validate submission.zip schema (adapter_config.json present) before upload]
- [Rule — Enforce vLLM inference test on 100 samples before any leaderboard submission]
- [Rule — Return consistent submission manifest with reproducibility checklist]
```

---

### Cross-Cutting Interface Rules

| Rule # | Principle | Enforcement Point |
|--------|-----------|-------------------|
| 1 | **Validate and parse phase input before any logic runs** | Every phase entry cell (Cell #4+) |
| 2 | **Enforce authorship and provenance before any artifact mutation** | Git commit + Kaggle Dataset versioning |
| 3 | **Return consistent, predictable artifact shapes** | TypedDict validation at phase boundaries |
| 4 | **Fail fast on schema mismatch — no silent coercion** | `assert` + `pydantic` validation |
| 5 | **Every output must include reproducibility metadata** | Git hash, timestamp, seed, config snapshot |

---

### Interface Anti-Patterns (Forbidden)

| Anti-Pattern | Why Forbidden | Correct Approach |
|-------------|-------------|----------------|
| Passing raw lists between phases | No schema validation, silent failures | Use `TypedDict` or `dataclass` contracts |
| Hard-coded paths in phase transitions | Breaks on Kaggle session restart | Use `Path` objects from config dataclass |
| Skipping validation "to save time" | Data corruption propagates downstream | Mandatory `validate_*()` function per phase |
| Modifying upstream artifacts in-place | Destroys reproducibility | Write to new path, version with timestamp |
| Implicit dependency on global variables | Race conditions in Kaggle sessions | Explicit parameter passing only |

---

### Validation Functions Template

```python
# ✅ CORRECT — Explicit phase boundary validation
from pathlib import Path
from typing import TypedDict, List, Dict
import json

def validate_p1_output(output: DataCurationOutput) -> None:
    """
    Validate Phase 1 output before allowing Phase 2 to begin.
    
    Raises:
        ValueError: If any invariant is violated
        FileNotFoundError: If expected artifacts missing
    """
    # Rule 1: Schema validation
    required_keys = DataCurationOutput.__required_keys__
    missing = required_keys - set(output.keys())
    if missing:
        raise ValueError(f"P1 output missing keys: {missing}")
    
    # Rule 2: Invariant enforcement
    if not (0.70 <= output["reasoning_ratio"] <= 0.80):
        raise ValueError(
            f"Reasoning ratio {output['reasoning_ratio']} outside 75% ± 5% tolerance"
        )
    
    if output["quality_score"] < 0.80:
        raise ValueError(
            f"Quality score {output['quality_score']} below 0.80 threshold. "
            f"Re-run LLM-as-judge with stricter criteria."
        )
    
    if not output["leakage_check"]:
        raise ValueError("Test set leakage detected. Halting pipeline.")
    
    # Rule 3: Artifact existence
    dataset_path = Path(output["dataset_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    if dataset_path.stat().st_size < 1024 * 1024:  # < 1MB suspicious
        raise ValueError(f"Dataset suspiciously small: {dataset_path.stat().st_size} bytes")
    
    # Rule 4: Consistent response shape
    manifest = {
        "phase": "P1",
        "status": "validated",
        "artifact_count": 1,
        "artifacts": [str(dataset_path)],
        "invariants_passed": 4,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    print(f"✅ P1 output validated: {json.dumps(manifest, indent=2)}")

# Usage at P2 Cell #4 entry
# validate_p1_output(p1_output)
```

---

### Kaggle-Specific Interface Rules

| Constraint | Rule |
|-----------|------|
| **Session Timeout (4h)** | Every phase must save artifacts every 30 min; validation must run < 10 min |
| **Internet Access** | Phase 1 only (API calls); Phases 2-4 must work offline with validated inputs |
| **GPU Memory** | Phase boundary validation must include `torch.cuda.empty_cache()` call |
| **Read-only /input** | Phase inputs read from `/kaggle/input/`; outputs write to `/kaggle/working/` |
| **Dataset Versioning** | Every validated output must be versioned as Kaggle Dataset with semantic note |

---

### Human Checkpoint Gates (Phase Transitions)

The agent must pause and request human approval at these interface boundaries:

| Transition | Gate Condition | Human Action Required |
|-----------|---------------|----------------------|
| **P1 → P2** | `validate_p1_output()` passes | Approve dataset quality, confirm failure modes documented |
| **P2 → P3** | `validate_p2_output()` passes | Approve SFT convergence, confirm LoRA rank ≤ 32 |
| **P3 → P4** | `validate_p3_output()` passes | Approve RL stability, confirm KL < 0.05 |
| **P4 → Submit** | `validate_p4_output()` passes | Approve public accuracy, confirm notebook published |


## Data and Storage

### Storage Architecture Overview

The ATRD pipeline operates across two distinct storage tiers optimized for Kaggle's ephemeral compute environment and the competition's artifact requirements. All storage decisions are driven by data size, access frequency, and persistence requirements.

| Tier | Location | Purpose | Persistence |
|------|----------|---------|-------------|
| **Hot** | `/kaggle/working/` | Active training artifacts, checkpoints, logs | Session-only (ephemeral) |
| **Warm** | Kaggle Datasets | Versioned datasets, stable checkpoints | Persistent, versioned |
| **Cold** | Hugging Face Hub | Final model weights, reproducibility snapshots | Permanent, public/private |
| **Competition** | `submission.zip` | Competition-mandated delivery format | Immutable once submitted |

---

### Artifact Storage Rules

#### Rule 1: Metadata and small structured data belong in JSON/JSONL files
- **Applies to:** Configuration files, evaluation metrics, training logs, dataset manifests, ablation results
- **Format:** `json` for single objects, `jsonl` for line-oriented records
- **Size threshold:** ≤ 10 MB per file
- **Rationale:** Human-readable, diff-friendly in git, fast random access for analysis
- **Example:**
  ```python
  # ✅ CORRECT — Structured metadata in JSON
  {
    "phase": "P2",
    "checkpoint_id": "sft_20260522_1430",
    "lora_rank": 32,
    "lora_alpha": 64,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "validation_loss": 1.247,
    "convergence_epoch": 2.3,
    "git_commit": "a1b2c3d",
    "seed": 42
  }
  ```

#### Rule 2: Large generated content belongs in file storage with streaming access
- **Applies to:** Synthetic datasets (10k-50k problems), training corpora, tokenized datasets, raw API responses
- **Format:** `jsonl` for text records, `parquet` for tabular data, `safetensors` for model weights
- **Size threshold:** > 10 MB per file
- **Rationale:** Memory-mapped I/O, columnar compression for parquet, zero-copy loading for safetensors
- **Example:**
  ```python
  # ✅ CORRECT — Streaming dataset with memory mapping
  from datasets import load_dataset
  
  dataset = load_dataset(
      "json",
      data_files="/kaggle/input/nemotron-synthetic-v1/filtered_problems.jsonl",
      streaming=False  # Enable memory mapping for large files
  )
  ```

#### Rule 3: Do not store large content directly in notebook variables or git
- **Forbidden:** Pickled Python objects, raw tensor dumps, unversioned binary blobs in git
- **Rationale:** Git corruption, notebook bloat, OOM on reload, irreproducible state
- **Exception:** Small demonstration tensors (< 1 MB) for unit test verification only

#### Rule 4: Model checkpoints must use `safetensors` format exclusively
- **Applies to:** All LoRA adapter weights, base model references
- **Forbidden:** `pytorch_model.bin`, `model.ckpt`, custom formats
- **Rationale:** `safetensors` is competition-mandated, enables zero-copy loading, prevents arbitrary code execution
- **Enforcement:**
  ```python
  # ✅ CORRECT — PEFT native safetensors save
  model.save_pretrained("/kaggle/working/checkpoints/sft_latest/")
  # Automatically produces: adapter_config.json + adapter_model.safetensors
  
  # ❌ INCORRECT — Manual torch save
  torch.save(model.state_dict(), "lora_weights.pt")  # Breaks vLLM compatibility
  ```

#### Rule 5: Raw API responses must be preserved before filtering
- **Applies to:** DeepSeek-R1, Qwen3-235B synthetic generation outputs
- **Format:** `jsonl` with full response metadata (timestamp, model, prompt hash, raw completion)
- **Rationale:** Reproducibility audit trail, cost tracking, prompt engineering iteration
- **Retention:** Until competition end date, then archive to cold storage

---

### Directory Structure and Storage Mapping

```
/kaggle/
├── input/                          # READ-ONLY — Competition + uploaded datasets
│   ├── nvidia-nemotron-reasoning/  # Competition-provided benchmark data
│   ├── nemotron-synthetic-v1/       # Versioned synthetic dataset (Kaggle Dataset)
│   ├── nemotron-checkpoints/        # Versioned model checkpoints (Kaggle Dataset)
│   └── openmath-opencode-mix/       # NVIDIA OpenMathReasoning + OpenCodeReasoning
│
└── working/                        # WRITEABLE — Session ephemeral output
    ├── checkpoints/                # Active training checkpoints (sync to Kaggle Datasets)
    │   ├── sft_20260522_1430/      # SFT checkpoint with adapter_config.json
    │   ├── grpo_20260523_0900/     # GRPO checkpoint with reward logs
    │   └── latest -> symlink to most recent
    │
    ├── logs/                       # Structured training logs
    │   ├── p1_data_generation.json
    │   ├── p2_sft_training.jsonl
    │   ├── p3_grpo_rewards.jsonl
    │   └── p4_evaluation.json
    │
    ├── synthetic_data/             # Raw + filtered synthetic generation
    │   ├── raw_deepseek_r1.jsonl   # Unfiltered API responses
    │   ├── raw_qwen3_235b.jsonl
    │   ├── filtered_problems.jsonl # Post-LLM-judge
    │   └── dedup_report.json
    │
    ├── evaluation/                   # Inference outputs and metrics
    │   ├── baseline_results.json
    │   ├── sft_results.json
    │   ├── grpo_results.json
    │   ├── budget_force_ablation.json
    │   └── predictions.json        # Competition submission format
    │
    ├── submission.zip              # Final competition artifact
    └── notebooks/                  # Executed notebook copies for versioning
        ├── 01_data_generation.ipynb
        ├── 02_sft_training.ipynb
        ├── 03_grpo_training.ipynb
        └── 04_budget_forcing.ipynb
```

---

### Data Lifecycle Rules

| Stage | Location | Action | Trigger |
|-------|----------|--------|---------|
| **Generation** | `/kaggle/working/synthetic_data/raw_*.jsonl` | Write raw API responses | Per-batch completion |
| **Filtering** | `/kaggle/working/synthetic_data/filtered_*.jsonl` | LLM-as-judge scoring | Phase 1 quality gate |
| **Deduplication** | `/kaggle/working/synthetic_data/dedup_report.json` | MinHash + n-gram analysis | Pre-dataset mixing |
| **Versioning** | Kaggle Dataset `nemotron-synthetic-vX` | `kaggle datasets create-version` | Phase 1 completion |
| **Training** | `/kaggle/working/checkpoints/` | SFT/GRPO checkpoint saves | Every 30 min / epoch end |
| **Checkpoint Sync** | Kaggle Dataset `nemotron-checkpoints` | Upload after session | Pre-session timeout |
| **Evaluation** | `/kaggle/working/evaluation/` | Inference + metric computation | Phase 3/4 validation |
| **Submission** | `submission.zip` + Kaggle Competition | Final upload | Human approval gate |

---

### Storage Constraints and Optimization

| Constraint | Rule | Rationale |
|-----------|------|-----------|
| Kaggle session disk: 20 GB | Compress datasets with `gzip` or `zstd`; use `streaming=True` for large datasets | Prevents disk-full crashes mid-training |
| Kaggle session memory: 16-32 GB | Use `datasets` library with `mmap` mode; avoid `pandas` for > 1M rows | Prevents OOM during data loading |
| Kaggle Dataset upload: 100 GB max | Split large datasets into shards; version incrementally | Enables incremental dataset updates |
| Hugging Face Hub: No size limit for public | Push final checkpoints as private until submission; public after competition | Protects competitive advantage |
| Git repo: 100 MB max per file | `.gitignore` all checkpoints, datasets, logs; commit code + configs only | Keeps repository cloneable |

---

### Persistence and Recovery Rules

#### Rule 6: Every 30 minutes or epoch end, sync active checkpoints to Kaggle Datasets
```python
# ✅ CORRECT — Automated checkpoint persistence
from kaggle.api.kaggle_api_extended import KaggleApi

def emergency_sync(checkpoint_dir: Path, dataset_slug: str) -> None:
    """
    Sync checkpoint to Kaggle Dataset before session timeout.
    Call from Trainer callback or signal handler.
    """
    api = KaggleApi()
    api.authenticate()
    
    # Create incremental version
    api.dataset_create_version(
        str(checkpoint_dir.parent),
        version_notes=f"Emergency sync: {checkpoint_dir.name}",
        delete_old_versions=False
    )
    print(f"✅ Checkpoint synced: {dataset_slug}")
```

#### Rule 7: Never rely on `/kaggle/working/` persistence across sessions
- All paths in `/kaggle/working/` are wiped on session restart
- **Mandatory:** Download versioned Kaggle Datasets at notebook start
- **Mandatory:** Re-upload outputs before session end or timeout

#### Rule 8: Use deterministic file naming with timestamps and git hashes
```python
# ✅ CORRECT — Traceable artifact naming
from datetime import datetime
import subprocess

def artifact_name(phase: str, description: str) -> str:
    """Generate traceable artifact filename."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    return f"{phase}_{description}_{timestamp}_{git_hash}"
```

---

### Competition-Specific Storage Rules

| Competition Requirement | Storage Implementation |
|------------------------|------------------------|
| `submission.zip` with `adapter_config.json` | Package from `/kaggle/working/checkpoints/grpo_latest/`; validate schema before zip |
| vLLM compatibility | Ensure `adapter_model.safetensors` + `adapter_config.json` present; test load in isolation cell |
| Public Kaggle notebook | Commit executed notebook to git; upload as competition notebook; link in write-up |
| Reproducibility proof | Include `requirements.txt`, `config.yaml`, and dataset version IDs in notebook preamble |

---

### Storage Anti-Patterns (Forbidden)

| Anti-Pattern | Why Forbidden | Correct Approach |
|-------------|-------------|----------------|
| `pickle.dump(model, f)` | Security risk, not portable, breaks across Python versions | `model.save_pretrained()` with `safetensors` |
| Storing tensors in notebook output cells | Bloats `.ipynb` file, causes git conflicts, OOM on open | Save to `.safetensors`, reference path in output |
| `pd.read_csv()` for 1M+ row datasets | Loads entire dataset into RAM; crashes on Kaggle | `datasets.load_dataset(streaming=True)` |
| Hard-coding `/kaggle/working/` paths in functions | Breaks when running locally or on G4 VM | `Path(config.OUTPUT_DIR)` from centralized config |
| Keeping only latest checkpoint | Kaggle session death loses all progress | Version every checkpoint to Kaggle Dataset |
| Mixing raw and filtered data in same file | Impossible to reproduce filtering decisions | Separate `raw_*.jsonl` and `filtered_*.jsonl` with manifest |

## File Organization

```
atrd-nemotron-reasoning/
│
├── .git/                           # Version control — code + configs only
├── .gitignore                      # Excludes: checkpoints, datasets, logs, *.zip
│
├── context/                        # PROJECT SPECIFICATIONS (Single Source of Truth)
│   ├── agents.md                   # Entry point for AI agents
│   ├── project_overview.md         # Mission, scope, success criteria
│   ├── architecture.md             # Tech stack, storage model, invariants
│   ├── code_standards.md           # Python/ML conventions (this file)
│   ├── ai_workflow_rules.md        # Agent behavior, phase gates, scoping
│   ├── progress_tracker.md         # Current phase, blockers, decisions
│   └── feature_specs/              # Per-phase detailed specs
│       ├── p1_data_curation_spec.md
│       ├── p2_sft_training_spec.md
│       ├── p3_grpo_training_spec.md
│       └── p4_budget_forcing_spec.md
│
├── configs/                        # CONFIGURATION TEMPLATES (Read-only baselines)
│   ├── base_lora.json              # NVIDIA default LoRA config
│   ├── base_grpo.json              # TRL default GRPO config
│   ├── competition_params.json     # Official inference params (IMMUTABLE)
│   ├── custom_lora.json            # Project-specific LoRA overrides
│   ├── custom_grpo.json            # Project-specific GRPO overrides
│   └── visualization.json          # Matplotlib/Plotly style config
│
├── notebooks/                      # KAGGLE NOTEBOOKS (Primary deliverables)
│   ├── 01_data_generation.ipynb    # Phase 1: Failure analysis + synthetic data
│   ├── 02_sft_training.ipynb       # Phase 2: QLoRA supervised fine-tuning
│   ├── 03_grpo_training.ipynb      # Phase 3: GRPO + PRM reinforcement learning
│   ├── 04_budget_forcing.ipynb     # Phase 4: Test-time adaptation + submission
│   └── 05_evaluation.ipynb         # Phase 5: Ablation studies + final metrics
│
├── src/                            # PYTHON MODULES (Reusable library code)
│   ├── __init__.py
│   ├── data/                       # Data pipeline modules
│   │   ├── __init__.py
│   │   ├── baseline_eval.py        # Base model failure collection
│   │   ├── synthetic_generator.py  # Frontier model API wrapper
│   │   ├── llm_judge.py            # Quality filtering scorer
│   │   ├── deduplicator.py         # MinHash near-duplicate detection
│   │   └── dataset_mixer.py        # Stratified mixing + leakage check
│   ├── training/                   # Training modules
│   │   ├── __init__.py
│   │   ├── lora_config.py          # LoRAConfig dataclass + validation
│   │   ├── sft_trainer.py          # SFT training loop wrapper
│   │   ├── grpo_trainer.py         # GRPO training loop wrapper
│   │   ├── prm_scorer.py           # Implicit PRM log-ratio scorer
│   │   └── checkpoint_manager.py   # Save/load with Kaggle Dataset sync
│   ├── inference/                  # Inference modules
│   │   ├── __init__.py
│   │   ├── vllm_engine.py          # vLLM initialization + LoRA loading
│   │   ├── budget_forcer.py        # Dynamic token injection / termination
│   │   ├── answer_extractor.py     # \boxed{} extraction + fallback heuristics
│   │   └── difficulty_classifier.py# Easy/medium/hard problem classifier
│   ├── evaluation/                 # Evaluation modules
│   │   ├── __init__.py
│   │   ├── competition_metric.py   # Official accuracy metric wrapper
│   │   ├── ablation_framework.py   # Component isolation tests
│   │   └── stratified_eval.py      # Per-difficulty-bin accuracy
│   └── utils/                      # Shared utilities
│       ├── __init__.py
│       ├── config_loader.py          # YAML/JSON config parsing
│       ├── gpu_utils.py            # CUDA setup, memory monitoring, OOM recovery
│       ├── logging_utils.py          # Structured JSON logging
│       ├── seed_utils.py           # Reproducibility seed fixing
│       └── kaggle_uploader.py      # Dataset versioning + emergency sync
│
├── scripts/                        # STANDALONE SCRIPTS (Automation + verification)
│   ├── verify_unit_completion.py   # Phase transition gate checker
│   ├── verify_protected_files.py   # Protected file integrity checker
│   ├── package_submission.py       # submission.zip builder + validator
│   ├── run_baseline.sh             # One-command baseline evaluation
│   └── run_full_pipeline.sh        # End-to-end pipeline orchestrator
│
├── tests/                          # TEST SUITE (Per-unit + integration)
│   ├── __init__.py
│   ├── unit/                       # Per-function tests
│   │   ├── test_synthetic_generator.py
│   │   ├── test_llm_judge.py
│   │   ├── test_budget_forcer.py
│   │   └── test_answer_extractor.py
│   └── integration/                # Per-phase tests
│       ├── test_p1_pipeline.py
│       ├── test_p2_sft.py
│       ├── test_p3_grpo.py
│       └── test_p4_inference.py
│
├── docs/                           # DOCUMENTATION (Prize eligibility requirement)
│   ├── README.md                   # Project overview + quickstart
│   ├── methodology.md              # Full write-up (≥ 2,000 words)
│   ├── adr/                          # Architecture Decision Records
│   │   ├── 001-qlora-nf4.md
│   │   ├── 002-implicit-prm.md
│   │   └── 003-budget-forcing-heuristic.md
│   └── figures/                    # Exported charts for write-up
│       ├── loss_curves.png
│       ├── ablation_waterfall.svg
│       └── budget_force_impact.pdf
│
├── data/                           # LOCAL DATA CACHE (Git-ignored, Kaggle-populated)
│   ├── raw/                        # Downloaded / input datasets
│   │   ├── nvidia_benchmark/       # Competition-provided test cases
│   │   ├── openmath_reasoning/     # NVIDIA OpenMathReasoning dataset
│   │   └── opencode_reasoning/     # NVIDIA OpenCodeReasoning dataset
│   ├── synthetic/                  # Generated synthetic data
│   │   ├── raw_deepseek_r1.jsonl
│   │   ├── raw_qwen3_235b.jsonl
│   │   ├── filtered_problems.jsonl
│   │   └── dedup_report.json
│   └── processed/                  # Final training corpora
│       └── final_train_dataset.jsonl
│
├── checkpoints/                    # MODEL CHECKPOINTS (Git-ignored, Kaggle Dataset-backed)
│   ├── sft_20260522_1430/          # SFT checkpoint with adapter_config.json
│   ├── grpo_20260523_0900/         # GRPO checkpoint with reward logs
│   └── latest -> grpo_20260523_0900/  # Symlink to current active checkpoint
│
├── logs/                           # TRAINING LOGS (Git-ignored, ephemeral)
│   ├── p1_data_generation.json
│   ├── p2_sft_training.jsonl
│   ├── p3_grpo_rewards.jsonl
│   ├── p4_evaluation.json
│   └── error_logs/                 # Segregated error traces
│       └── oom_recovery_20260522.log
│
├── evaluation/                     # INFERENCE OUTPUTS (Git-ignored)
│   ├── baseline_results.json
│   ├── sft_results.json
│   ├── grpo_results.json
│   ├── budget_force_ablation.json
│   └── predictions.json            # Competition submission format
│
├── requirements.txt                # PINNED dependencies (reproducibility)
├── requirements-dev.txt            # Testing + linting dependencies
├── setup.py                        # Package install: pip install -e .
├── LICENSE                         # MIT (community reuse for scholarship)
│
└── submission.zip                  # FINAL COMPETITION ARTIFACT (Generated, not committed)

---

### Folder Purpose Reference

| Folder | What Belongs Here | What Does NOT Belong Here |
|--------|-------------------|---------------------------|
| `context/` | Markdown specs, progress tracker, feature requirements | Code, data, notebooks, generated artifacts |
| `configs/` | JSON/YAML configuration templates, style definitions | Secrets, API keys, runtime-generated configs |
| `notebooks/` | Kaggle `.ipynb` files (01-05), one per phase | Python modules, large data files, checkpoints |
| `src/` | Reusable `.py` modules with type hints and docstrings | Notebook cells, hard-coded paths, API keys |
| `scripts/` | Automation scripts, verification tools, packaging | Training logic, data processing, model code |
| `tests/` | `pytest` unit + integration tests | Training data, model weights, logs |
| `docs/` | Markdown documentation, ADRs, exported figures | Raw code, notebooks, unprocessed data |
| `data/` | Downloaded and generated datasets (git-ignored) | Model checkpoints, training logs, submission zip |
| `checkpoints/` | LoRA adapter weights + adapter_config.json (git-ignored) | Raw data, notebooks, base model weights |
| `logs/` | Structured JSON/JSONL training logs (git-ignored) | Checkpoints, data, source code |
| `evaluation/` | Inference predictions, ablation metrics (git-ignored) | Training code, raw API responses, datasets |

---

### Critical File Rules

| File | Rule | Rationale |
|------|------|-----------|
| `requirements.txt` | Pin exact versions (`==`) with hashes | Kaggle session reproducibility; prevents dependency drift |
| `.gitignore` | Ignore: `data/`, `checkpoints/`, `logs/`, `evaluation/`, `*.zip`, `__pycache__/` | Repository stays lightweight and cloneable; binaries in Kaggle Datasets |
| `adapter_config.json` | Generated by PEFT only; never hand-edited | Competition validator checks schema; manual edits break vLLM loading |
| `submission.zip` | Generated by `scripts/package_submission.py` only | Ensures consistent packaging; never commit to git |
| `progress_tracker.md` | Updated after every work session | Human + agent alignment on current state |
| `configs/competition_params.json` | **ABSOLUTELY IMMUTABLE** | temperature=0.0, max_tokens=7680, etc. are competition-mandated |

---

### Kaggle Path Mapping

When running on Kaggle, the local structure maps to Kaggle's filesystem as follows:

| Local Path | Kaggle Equivalent | Purpose |
|-----------|-------------------|---------|
| `data/raw/nvidia_benchmark/` | `/kaggle/input/nvidia-nemotron-reasoning/` | Competition data |
| `data/processed/` | `/kaggle/input/nemotron-synthetic-v1/` | Versioned training dataset |
| `checkpoints/` | `/kaggle/working/checkpoints/` + Kaggle Dataset sync | Active + persisted checkpoints |
| `notebooks/` | `/kaggle/working/notebooks/` | Executed copies for versioning |
| `logs/` | `/kaggle/working/logs/` | Session training logs |
| `evaluation/` | `/kaggle/working/evaluation/` | Inference outputs |
| `submission.zip` | `/kaggle/working/submission.zip` | Final upload artifact |

---

### Git Commit Scope Rules

| Change Type | Files to Commit | Files to NEVER Commit |
|-------------|---------------|----------------------|
| Code change | `src/`, `notebooks/`, `scripts/`, `tests/` | `data/`, `checkpoints/`, `logs/` |
| Config change | `configs/*.json` (except `competition_params.json`) | Runtime-generated configs |
| Documentation | `docs/`, `context/*.md` | Exported figures (> 10MB) |
| Dependency | `requirements.txt`, `setup.py` | `venv/`, `__pycache__/` |
| Notebook output | Clear all outputs before commit | Raw cell outputs, embedded images |
