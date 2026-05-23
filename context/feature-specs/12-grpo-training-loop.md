# 12 — GRPO Training Loop Specification

## ATRD: Adaptive Test-Time Reasoning Distillation

### 1. Purpose and Setup Order
This specification defines the GRPO (Group Relative Policy Optimization) training loop. This phase uses reinforcement learning to optimize the policy for structured reasoning with verifiable rewards.

> [!IMPORTANT]
> Read `11-implicit-prm-setup.md` before implementing. PRM scorer must be ready.

---

## 2. GRPO Algorithm

### 2.1 Core Mechanism
For each prompt, GRPO generates G=8 completions, scores them, computes relative advantages, and updates the policy:

```
For each prompt x:
  1. Generate G=8 completions {y₁, ..., y₈} ~ π_θ(·|x)
  2. Score each completion: r_i = Reward(y_i, ground_truth)
  3. Compute advantages: A_i = (r_i - mean(r)) / std(r)
  4. Update policy: θ ← θ + α * ∇_θ J_GRPO(θ)
```

### 2.2 Group Size Configuration
```python
@dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 8  # G=8 — validated for stability
    kl_penalty: float = 0.001  # KL divergence coefficient
    learning_rate: float = 5e-6  # Lower than SFT for stability
    batch_size: int = 1  # Per device
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    num_train_epochs: int = 1
    max_steps: int = 500  # Upper limit
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_steps: int = 50
```

### 2.3 Reward Function (`src/training/grpo_trainer.py`)

```python
def create_reward_function(tolerance: float = 0.01):
    def reward_fn(completions: List[str], ground_truth: str) -> List[float]:
        rewards = []
        for completion in completions:
            score = 0.0

            # Format reward (+0.2 for box, +0.2 for thinking tags)
            if "\\boxed{" in completion:
                score += 0.2
            if "<<thinking>>" in completion and "</thinking>>" in completion:
                score += 0.2

            # Correctness reward (+0.8 if correct)
            extracted = _extract_boxed_answer(completion)
            if _check_answer(extracted, ground_truth, tolerance):
                score += 0.8

            # Redundancy penalty (-0.3 if looping)
            if _detect_redundancy(completion):
                score -= 0.3

            rewards.append(max(-1.0, min(1.0, score)))
        return rewards
    return reward_fn
```

### 2.4 TRL GRPOTrainer Integration

```python
from trl import GRPOTrainer, GRPOConfig

grpo_config = GRPOConfig(
    output_dir="checkpoints/grpo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    max_grad_norm=1.0,
    num_train_epochs=1,
    max_steps=500,
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=50,
    bf16=True,
    gradient_checkpointing=True,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=grpo_config,
    train_dataset=train_dataset,
    reward_funcs=reward_fn,
)
result = trainer.train()
```

### 2.5 KL Divergence Monitoring

```python
class KLMonitor:
    """Monitor KL divergence between current and reference policy."""

    def __init__(self, ref_model, threshold: float = 0.05):
        self.ref_model = ref_model
        self.threshold = threshold
        self.history = []

    def log_kl(self, current_model, batch):
        """Compute and log KL divergence."""
        kl = compute_kl(current_model, self.ref_model, batch)
        self.history.append(kl)

        if kl > self.threshold:
            print(f"WARNING: KL={kl:.4f} exceeds threshold={self.threshold}")
        if kl > 0.1:
            raise RuntimeError(f"KL divergence too high ({kl:.4f}). Stopping training.")

        return kl
```

---

## 3. Training Flow

### 3.1 Step-by-Step Loop
| Step | Action | Duration |
|------|--------|----------|
| 1 | Load SFT checkpoint | ~2 min |
| 2 | Initialize GRPO config | ~1 min |
| 3 | For each training step: | 30–500 steps |
| 3a | Generate G=8 completions per prompt | ~30s per prompt |
| 3b | Score completions with reward function | ~1s |
| 3c | Compute advantages (group normalization) | ~0.5s |
| 3d | Update policy (PPO-clip objective) | ~5s |
| 3e | Log metrics (reward, KL, advantage stats) | ~0.5s |
| 4 | Save checkpoint | ~2 min |

### 3.2 Monotonic Reward Check
```python
def verify_monotonic_reward(reward_history: List[float], window: int = 10) -> bool:
    """Verify reward increases monotonically over recent window."""
    if len(reward_history) < window * 2:
        return True  # Not enough data

    recent = reward_history[-window:]
    early = reward_history[-window*2:-window]
    return sum(recent) / len(recent) > sum(early) / len(early)
```

---

## 4. Exit Quality Gate
- [ ] Mean reward increases monotonically over training steps
- [ ] KL divergence stays below 0.05 throughout training
- [ ] Final accuracy improves over SFT checkpoint
- [ ] No reward hacking detected (check sample output quality)
- [ ] `grpo_checkpoint/final_adapter/` saved with validated adapter_config.json
- [ ] `logs/grpo_rewards.json` with step-level reward and KL trajectories
- [ ] `verify_unit_completion.py P3` returns success
