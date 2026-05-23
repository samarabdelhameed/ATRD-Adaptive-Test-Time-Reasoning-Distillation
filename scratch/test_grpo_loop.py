import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock PyTorch and model packages for CPU/headless test runs
class MockTorch(MagicMock):
    class nn:
        class Module:
            pass
    @staticmethod
    def sigmoid(val):
        import math
        class SigmoidTensor:
            def __init__(self, v):
                self.v = float(v)
            def item(self):
                return 1.0 / (1.0 + math.exp(-self.v))
        return SigmoidTensor(val)
    @staticmethod
    def tensor(val):
        return val

sys.modules['torch'] = MockTorch
sys.modules['transformers'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['trl'] = MagicMock()
sys.modules['datasets'] = MagicMock()

# Add root folder to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.training.grpo_trainer import GRPOTrainerWrapper, KLMonitor, verify_monotonic_reward

def test_reward_function_with_list_ground_truth():
    print("Testing GRPO reward_fn with list ground_truth...")
    
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    
    # Initialize wrapper
    wrapper = GRPOTrainerWrapper(
        model=mock_model,
        tokenizer=mock_tokenizer,
        grpo_config_path="configs/base_grpo.json",
        competition_config_path="configs/competition_params.json",
    )
    
    # Create reward function with PRM
    reward_fn_prm = wrapper.create_reward_function(use_prm=True)
    
    # 8 completions generated for 1 prompt (hence ground_truth list has length 8)
    completions = [
        "<<thinking>>\nStep 1: Simplify. Step 2: Answer x = 5.\n</thinking>>\n\\boxed{5}",  # Correct + formatting
        "<<thinking>>\nStep 1: Simplify. Step 2: Answer x = 99.\n</thinking>>\n\\boxed{99}",  # Incorrect + formatting
        "\\boxed{5}",  # Correct + no formatting
        "loop\nloop\nloop\nloop",  # Redundant
        "\\boxed{5}",
        "\\boxed{5}",
        "\\boxed{5}",
        "\\boxed{5}",
    ]
    ground_truth = ["5"] * 8
    
    rewards = reward_fn_prm(completions, ground_truth)
    print("Computed Rewards:", rewards)
    
    assert len(rewards) == 8
    assert rewards[0] > rewards[1], "Correct completion should have higher reward than incorrect completion"
    assert rewards[2] > rewards[3], "Correct completion without formatting should have higher reward than redundancy loop"
    assert all(-1.0 <= r <= 1.0 for r in rewards), "All rewards must be clamped between -1.0 and 1.0"
    print("Reward function list test passed.")

def test_kl_monitor():
    print("Testing KLMonitor...")
    
    ref_model = MagicMock()
    current_model = MagicMock()
    batch = MagicMock()
    
    monitor = KLMonitor(ref_model=ref_model, threshold=0.05)
    
    # Mock _compute_kl to return a safe value
    import src.training.grpo_trainer
    original_compute_kl = src.training.grpo_trainer._compute_kl
    src.training.grpo_trainer._compute_kl = lambda c, r, b: 0.02
    
    try:
        kl1 = monitor.log_kl(current_model, batch)
        assert kl1 == 0.02
        assert len(monitor.history) == 1
        
        # Test exceeding warning threshold
        src.training.grpo_trainer._compute_kl = lambda c, r, b: 0.06
        kl2 = monitor.log_kl(current_model, batch)
        assert kl2 == 0.06
        
        # Test exceeding critical error threshold
        src.training.grpo_trainer._compute_kl = lambda c, r, b: 0.12
        try:
            monitor.log_kl(current_model, batch)
            assert False, "Should raise RuntimeError when KL divergence > 0.1"
        except RuntimeError as e:
            print("Successfully caught high KL error:", e)
            
    finally:
        # Revert mock
        src.training.grpo_trainer._compute_kl = original_compute_kl
        
    print("KLMonitor test passed.")

def test_monotonic_reward_verification():
    print("Testing monotonic reward verification...")
    
    # History with increasing rewards
    increasing_history = [0.1, 0.15, 0.2, 0.22, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.52, 0.55, 0.6, 0.65, 0.7, 0.72, 0.75, 0.8, 0.82, 0.85]
    assert verify_monotonic_reward(increasing_history, window=5) == True, "Should be True for increasing trend"
    
    # History with decreasing/stagnating rewards
    decreasing_history = [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.05, 0.02, 0.01, 0.0]
    assert verify_monotonic_reward(decreasing_history, window=5) == False, "Should be False for decreasing trend"
    
    # Not enough data
    short_history = [0.5, 0.6]
    assert verify_monotonic_reward(short_history, window=5) == True, "Should be True if history is shorter than window * 2"
    
    print("Monotonic reward verification test passed.")

def test_save_training_log():
    print("Testing save_training_log...")
    import tempfile, json, os

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    wrapper = GRPOTrainerWrapper(
        model=mock_model,
        tokenizer=mock_tokenizer,
        grpo_config_path="configs/base_grpo.json",
        competition_config_path="configs/competition_params.json",
    )

    # Use a temp dir inside workspace
    log_dir = "logs"
    reward_history = [0.2, 0.3, 0.35, 0.4, 0.5]
    kl_history = [0.01, 0.02, 0.015, 0.03, 0.025]

    output_file = wrapper.save_training_log(reward_history, kl_history, log_dir=log_dir)

    assert output_file.exists(), "grpo_rewards.json should exist"

    with open(output_file) as f:
        data = json.load(f)

    assert data["total_steps"] == 5
    assert len(data["reward_trajectory"]) == 5
    assert len(data["kl_trajectory"]) == 5
    assert "mean_reward" in data
    assert "monotonic_improvement" in data
    print(f"Log saved: {output_file} | mean_reward={data['mean_reward']:.3f}")
    print("save_training_log test passed.")

def test_grpo_config_matches_spec():
    print("Testing GRPO config matches spec...")
    import json

    with open("configs/base_grpo.json") as f:
        cfg = json.load(f)

    assert cfg["group_size"] == 8, f"group_size should be 8, got {cfg['group_size']}"
    assert cfg["kl_penalty"] == 0.001, f"kl_penalty should be 0.001"
    assert cfg["learning_rate"] == 5e-6, f"learning_rate should be 5e-6"
    assert cfg["batch_size"] == 1, f"batch_size should be 1"
    assert cfg["gradient_accumulation_steps"] == 8
    assert cfg["max_grad_norm"] == 1.0
    assert cfg["num_train_epochs"] == 1
    assert cfg["max_steps"] == 500
    assert cfg["warmup_ratio"] == 0.1
    assert cfg["logging_steps"] == 10
    assert cfg["save_steps"] == 50
    assert cfg["bf16"] == True
    assert cfg["gradient_checkpointing"] == True
    print("GRPO config matches spec 12 exactly.")

if __name__ == "__main__":
    print("=== STARTING GRPO LOOP UNIT TESTS ===")
    test_reward_function_with_list_ground_truth()
    print("-" * 40)
    test_kl_monitor()
    print("-" * 40)
    test_monotonic_reward_verification()
    print("-" * 40)
    test_save_training_log()
    print("-" * 40)
    test_grpo_config_matches_spec()
    print("=== ALL GRPO LOOP TESTS COMPLETED SUCCESSFULLY ===")
