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

# Import target components
from src.training.prm import (
    heuristic_step_score,
    segment_thinking_trace,
    compute_prm_guided_reward,
    test_prm_correlation,
    compute_log_ratio_score,
    check_answer,
    detect_redundancy
)

def test_step_segmentation():
    print("Testing Step Segmentation...")
    completion = "<<thinking>>\nStep 1: Simplify expression. Step 2: Solve x = 5.\n</thinking>>\n\\boxed{5}"
    steps = segment_thinking_trace(completion)
    print("Segmented steps:", steps)
    assert len(steps) >= 2, "Should segment into at least two steps"
    assert all("\\boxed" not in s for s in steps), "Should exclude boxed answer"
    print("Step Segmentation Passed.")

def test_heuristic_scoring():
    print("Testing Heuristic Scoring...")
    good_step = "therefore x = 5 hence 2*x = 10"
    bad_step = "loop loop loop loop loop loop"
    
    good_score = heuristic_step_score(good_step)
    bad_score = heuristic_step_score(bad_step)
    
    print(f"Good Step Score: {good_score:.3f}")
    print(f"Bad Step Score: {bad_score:.3f}")
    
    assert good_score > bad_score, "Good step should have higher heuristic score than loop step"
    print("Heuristic Scoring Passed.")

def test_composite_reward():
    print("Testing Composite Reward...")
    correct_trace = "<<thinking>>\nStep 1: x + 3 = 7. therefore x = 4.\n</thinking>>\n\\boxed{4}"
    incorrect_trace = "<<thinking>>\nStep 1: x + 3 = 7. therefore x = 99.\n</thinking>>\n\\boxed{99}"
    
    correct_reward = compute_prm_guided_reward(correct_trace, "4")
    incorrect_reward = compute_prm_guided_reward(incorrect_trace, "4")
    
    print(f"Correct Trace Reward: {correct_reward:.3f}")
    print(f"Incorrect Trace Reward: {incorrect_reward:.3f}")
    
    assert correct_reward > incorrect_reward, "Correct trace should yield higher reward than incorrect trace"
    assert -1.0 <= correct_reward <= 1.0, "Reward must be in range [-1.0, 1.0]"
    assert -1.0 <= incorrect_reward <= 1.0, "Reward must be in range [-1.0, 1.0]"
    print("Composite Reward Passed.")

def test_redundancy_and_formatting():
    print("Testing Redundancy & Formatting penalties...")
    redundant_trace = "x = 5\nx = 5\nx = 5\nx = 5\n"
    assert detect_redundancy(redundant_trace), "Should detect redundancy loop"
    
    no_thinking = "\\boxed{99}"
    reward_no_thinking = compute_prm_guided_reward(no_thinking, "5")
    # without <<thinking>> and </thinking>>, formatting reward should be lower
    trace_with_thinking = "<<thinking>>\n2x = 10 therefore x = 99\n</thinking>>\n\\boxed{99}"
    reward_with_thinking = compute_prm_guided_reward(trace_with_thinking, "5")
    
    print(f"No thinking reward (incorrect ans): {reward_no_thinking:.3f}")
    print(f"With thinking reward (incorrect ans): {reward_with_thinking:.3f}")
    assert reward_with_thinking > reward_no_thinking, "Trace with <<thinking>> format tokens should receive formatting bonus"
    print("Redundancy & Formatting Passed.")

def test_mock_prm_correlation():
    print("Testing Mock PRM Correlation...")
    validation_set = [
        {"question": "Solve 2x = 10", "answer": "5"},
        {"question": "Find x: x + 3 = 7", "answer": "4"},
    ]
    
    def mock_generate(q):
        if "2x = 10" in q:
            return "<<thinking>>\n2x = 10 therefore x = 5\n</thinking>>\n\\boxed{5}"
        else:
            return "<<thinking>>\nx + 3 = 7 hence x = 99\n</thinking>>\n\\boxed{99}"

    correct_scores = []
    incorrect_scores = []

    for example in validation_set:
        completion = mock_generate(example["question"])
        prm_score = compute_prm_guided_reward(completion, example["answer"])
        if check_answer(completion, example["answer"]):
            correct_scores.append(prm_score)
        else:
            incorrect_scores.append(prm_score)

    mean_correct = sum(correct_scores) / max(len(correct_scores), 1)
    mean_incorrect = sum(incorrect_scores) / max(len(incorrect_scores), 1)
    
    assert mean_correct > mean_incorrect, f"PRM scores not correlated with correctness (Correct: {mean_correct:.3f}, Incorrect: {mean_incorrect:.3f})"
    print(f"PRM Correlation Test Passed: Correct={mean_correct:.3f}, Incorrect={mean_incorrect:.3f}")

if __name__ == "__main__":
    print("=== STARTING PRM UNIT TESTS ===")
    test_step_segmentation()
    print("-" * 40)
    test_heuristic_scoring()
    print("-" * 40)
    test_composite_reward()
    print("-" * 40)
    test_redundancy_and_formatting()
    print("-" * 40)
    test_mock_prm_correlation()
    print("=== ALL PRM TESTS COMPLETED SUCCESSFULLY ===")
