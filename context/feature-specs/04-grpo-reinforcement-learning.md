# 04 — GRPO Reinforcement Learning Specification

## Phase 3: GRPO RL Alignment

### 1. Purpose and Setup Order
This specification defines the implementation details for aligning the model using Group Relative Policy Optimization (GRPO) reinforcement learning. The objective is to reward structured step-by-step thinking traces and format compliance, avoiding reward hacking.

---

## 2. Technical Components to Implement

### 2.1 Implicit Process Reward Model (PRM) Scorer (`src/training/grpo_trainer.py`)
- **Action**: Evaluate intermediate reasoning steps within the generated traces.
- **Mechanism**: Compute the log-ratio probability under the baseline SFT policy vs the active training policy for each token in the `<<thinking>...</thinking>` block.
- **Alternate (Fallback)**: If log-ratio calculations introduce high memory overhead, use regex checkers to verify intermediate equations, calculations, or mathematical transitions.
- **Output**: Numeric step rewards (value range $[0.0, 1.0]$).

### 2.2 Verifiable Reward Functions (`src/training/grpo_trainer.py`)
Implement a composite reward function comprised of:
1. **Format Reward**:
   - Return `+0.2` if output exactly contains `\boxed{}`.
   - Return `+0.2` if output utilizes `<<thinking>...</thinking>` correctly.
   - Return `-0.5` if format rules are violated.
2. **Answer Correctness Reward**:
   - Compare the parsed value inside the `\boxed{}` with the ground-truth value.
   - Return `+0.8` if correct, else `0.0`.
3. **Redundancy/Looping Penalty**:
   - Detect repeated phrases or equations inside the thinking traces.
   - Return `-0.3` penalty if repeating loops are found.

### 2.3 GRPO Trainer (`src/training/grpo_trainer.py`)
- **Input**: Model loaded from `checkpoints/sft_checkpoint/`.
- **Orchestration**: Use `trl.GRPOTrainer`.
- **Loop Parameters**:
  - Group Size $G = 8$ (Generates 8 completions per prompt).
  - Compute relative advantages of the 8 candidates using their reward scores:
    $$A_i = \frac{R_i - \text{mean}(R)}{\text{std}(R)}$$
  - Apply standard PPO-clip objective on the advantage values.
- **Hyperparameters**:
  - Learning Rate: `5e-6`
  - KL Divergence Coefficient (Beta): `0.001` (Keeps policy close to SFT baseline).
  - Steps: `100` to `500`.
- **Output**: Checkpoints saved under `checkpoints/grpo_checkpoint/`.

### 2.4 RL Validation (`03_grpo_training.ipynb`)
- **Action**: Evaluate model completions on validation subsets.
- **Check**: Verify that the model is not "reward hacking" (generating garbage text containing correct answers or fake boxes).
- **Log**: Export reward trajectories to `logs/grpo_rewards.json`.

---

## 3. Exit Quality Gate
Before proceeding to Phase 4, verify:
- [ ] Average reward score increases monotonically over training steps.
- [ ] KL divergence remains below 0.05.
- [ ] Final accuracy improves over SFT checkpoint.
- [ ] `verify_unit_completion.py P3` returns success.
