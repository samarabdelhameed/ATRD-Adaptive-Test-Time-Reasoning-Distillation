# 17 — Reusable Python Modules Specification

## ATRD: Adaptive Test-Time Reasoning Distillation

### 1. Purpose and Setup Order
This specification defines the complete Python module architecture under `src/`. These modules contain the core ML logic extracted from notebooks for testability, reuse, and clean abstraction.

> [!IMPORTANT]
> This is a cross-cutting specification. Modules should be implemented alongside their corresponding phase specs.

---

## 2. Package Structure

```
src/
├── __init__.py              # Package metadata (version, author, license)
├── data/
│   ├── __init__.py           # Exports: SyntheticGenerator, JudgeFilter, Deduplicator, DatasetMixer
│   ├── synthetic_generator.py # Failure-grounded synthetic data generation
│   ├── judge_filter.py        # LLM-as-judge quality scoring and filtering
│   ├── deduplicator.py        # MinHash near-duplicate detection and removal
│   └── dataset_mixer.py       # Multi-source dataset mixing with ratios
├── models/
│   ├── __init__.py           # Exports: ModelLoader, create_lora_config
│   ├── loader.py              # QLoRA 4-bit model loading with GPU memory management
│   └── lora_config.py         # LoRA configuration factory with competition validation
├── training/
│   ├── __init__.py           # Exports: SFTTrainerWrapper, GRPOTrainerWrapper
│   ├── sft_trainer.py         # TRL SFTTrainer wrapper with competition defaults
│   └── grpo_trainer.py        # TRL GRPOTrainer wrapper with PRM-guided rewards
├── inference/
│   ├── __init__.py           # Exports: BudgetForcer, VLLMEngine
│   ├── budget_forcer.py       # Adaptive test-time compute scaling
│   └── vllm_engine.py         # vLLM inference engine wrapper
└── evaluation/
    ├── __init__.py           # Exports: compute_accuracy, evaluate_submission, AblationRunner
    ├── metric.py              # Competition-grade accuracy evaluation
    └── ablation.py            # Systematic ablation study framework
```

---

## 3. Module Standards

### 3.1 All Modules Must Have
- [ ] Type hints on every function signature
- [ ] Google-style docstrings (Args, Returns, Raises)
- [ ] Named exports in `__init__.py`
- [ ] No hardcoded paths (use config-driven `Path` objects)
- [ ] All random seeds fixed (42)
- [ ] GPU memory explicitly managed (load/unload + `empty_cache()`)

### 3.2 Package Init Pattern
```python
"""Module description."""
__version__ = "1.0.0"
__author__ = "Samar Abdelhameed Ahmed"
__license__ = "MIT"

from .module_name import ClassName
__all__ = ["ClassName"]
```

### 3.3 Class Pattern
```python
class ClassName:
    """Short description.

    Attributes:
        attr1: Description.
        attr2: Description.
    """

    def __init__(self, param1: str, param2: int = 42) -> None:
        self.param1 = param1
        self.param2 = param2

    def method(self, input_data: List[str]) -> Dict[str, Any]:
        """Method description.

        Args:
            input_data: Description of input.

        Returns:
            Description of return value.

        Raises:
            ValueError: If input is invalid.
        """
        ...
```

---

## 4. Cross-Module Contracts

| Module | Consumed By | Contract (TypedDict) |
|--------|-------------|---------------------|
| `data.synthetic_generator` | `training.sft_trainer` | `SyntheticProblem: {question, thinking_trace, answer, failure_mode_tag}` |
| `data.dataset_mixer` | `training.sft_trainer` | `DataSet: List[Dict]` with `_source` metadata |
| `models.loader` | `training.sft_trainer`, `training.grpo_trainer` | `PreTrainedModel` + `PreTrainedTokenizer` |
| `models.lora_config` | `training.sft_trainer` | `LoraConfig` (PEFT) with `validate()` method |
| `training.sft_trainer` | `inference.vllm_engine` | `sft_checkpoint/` directory with adapter weights |
| `training.grpo_trainer` | `inference.budget_forcer` | `grpo_checkpoint/` directory with adapter weights |
| `inference.vllm_engine` | `evaluation.metric` | `List[str]` of model completions |
| `evaluation.metric` | `evaluation.ablation` | `Dict` with accuracy, category breakdown |

---

## 5. Testing Requirements

Each module must have:
- [ ] Unit test for every public method
- [ ] Integration test verifying module interactions
- [ ] Mock API calls for data generation tests
- [ ] Mock model forward passes for training tests

---

## 6. Exit Quality Gate
- [ ] All 13 source files exist in correct directories
- [ ] All `__init__.py` files export correct symbols
- [ ] All public functions have type hints and docstrings
- [ ] No circular imports between modules
- [ ] `python3 -m py_compile` passes for all files
- [ ] `import src` works without errors
