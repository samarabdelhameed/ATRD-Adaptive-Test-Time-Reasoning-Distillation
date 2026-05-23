"""Unit tests for Process Reward Model (PRM) scoring system.

Verifies segmenting, heuristic scoring, redundancy detection, composite reward
aggregation, OOM fallback logic, and correctness correlation.
"""

import sys
import unittest
from unittest.mock import MagicMock

# Mock torch and training dependencies for local environment execution
mock_torch = MagicMock()
sys.modules["torch"] = mock_torch
sys.modules["datasets"] = MagicMock()
sys.modules["trl"] = MagicMock()

from src.training.prm import (
    segment_thinking_trace,
    detect_redundancy,
    heuristic_step_score,
    compute_log_ratio_score,
    compute_prm_guided_reward,
)


class TestPRMScoring(unittest.TestCase):
    """Test suite for the PRM scoring logic."""

    def test_segment_thinking_trace(self):
        """Verify thinking trace splitting by period or newline."""
        completion = (
            "<<thinking>>\n"
            "First, let x = 5. Then we calculate y.\n"
            "Finally, y = 10.\n"
            "</thinking>>\n"
            "The answer is \\boxed{10}."
        )
        steps = segment_thinking_trace(completion)
        
        # Verify segment count and exclusion of the boxed answer
        self.assertTrue(len(steps) >= 3)
        for step in steps:
            self.assertNotIn("\\boxed", step)

    def test_detect_redundancy(self):
        """Verify redundancy/looping detection heuristics."""
        # Clean completion (no redundancy)
        clean = (
            "We add 3 to both sides.\n"
            "This gives x = 8.\n"
            "Thus, the answer is 8."
        )
        self.assertFalse(detect_redundancy(clean))

        # Redundancy: 3 repeating identical lines
        looping_lines = (
            "We divide by 2.\n"
            "We divide by 2.\n"
            "We divide by 2."
        )
        self.assertTrue(detect_redundancy(looping_lines))

        # Redundancy: word-level loop
        looping_words = "We divide by 2 We divide by 2 We divide by 2 We divide by 2 We divide by 2 We divide by 2"
        self.assertTrue(detect_redundancy(looping_words))

    def test_heuristic_step_score(self):
        """Verify step quality scoring based on key features."""
        # Top-tier step: has transition, connectors, and math formula
        step_best = "Therefore, we have x = 5 + 3 = 8."
        score_best = heuristic_step_score(step_best)
        self.assertGreater(score_best, 0.5)

        # Repetitive step: penalization
        step_looping = "loop loop loop loop loop loop loop loop"
        score_looping = heuristic_step_score(step_looping)
        self.assertEqual(score_looping, 0.0)

    def test_composite_reward_bounds_and_components(self):
        """Verify composite reward is in [-1.0, 1.0] and format rewards apply."""
        # Clean correct completion with proper format
        completion = (
            "<<thinking>>\n"
            "Therefore, 2 + 2 = 4.\n"
            "</thinking>>\n"
            "Answer: \\boxed{4}"
        )
        
        reward = compute_prm_guided_reward(completion, "4")
        self.assertGreater(reward, 0.8)  # High score for correctness and formatting
        self.assertTrue(-1.0 <= reward <= 1.0)

        # Repeating, incorrect completion
        completion_bad = (
            "Therefore, 2 + 2 = 5.\n"
            "Therefore, 2 + 2 = 5.\n"
            "Therefore, 2 + 2 = 5."
        )
        reward_bad = compute_prm_guided_reward(completion_bad, "4")
        self.assertLess(reward_bad, 0.2)  # Low/negative score for incorrectness & repeating
        self.assertTrue(-1.0 <= reward_bad <= 1.0)

    def test_log_ratio_oom_fallback(self):
        """Test that log ratio scoring gracefully handles OOM and falls back to heuristics."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Mock RuntimeError("out of memory") on forward call
        mock_model.side_effect = RuntimeError("CUDA out of memory error")
        
        # Test compute_log_ratio_score directly returns None under OOM
        score = compute_log_ratio_score("x = 5", "context", mock_model, mock_model, mock_tokenizer)
        self.assertIsNone(score)

        # Test composite reward falls back to heuristic score under log-ratio OOM
        reward = compute_prm_guided_reward(
            completion="<<thinking>>\nTherefore, x = 5.\n</thinking>>\n\\boxed{5}",
            ground_truth="5",
            ref_model=mock_model,
            current_model=mock_model,
            tokenizer=mock_tokenizer,
            use_log_ratio=True
        )
        self.assertGreater(reward, 0.8) # Heuristic fallback is successful

    def test_prm_correlation(self):
        """Test correlation exit gate: correct completions score higher than incorrect ones."""
        correct_completions = [
            "<<thinking>>\nTherefore, we multiply 3 by 4 to get 12.\n</thinking>>\n\\boxed{12}",
            "<<thinking>>\nSince it is a square, the area is 4^2 = 16.\n</thinking>>\n\\boxed{16}"
        ]
        
        incorrect_completions = [
            "<<thinking>>\nWe multiply 3 by 4 to get 15.\n</thinking>>\n\\boxed{15}",
            "We loop y = 1. We loop y = 1. We loop y = 1. \\boxed{16}"
        ]
        
        ground_truths = ["12", "16"]
        
        correct_rewards = [
            compute_prm_guided_reward(comp, gt)
            for comp, gt in zip(correct_completions, ground_truths)
        ]
        incorrect_rewards = [
            compute_prm_guided_reward(comp, gt)
            for comp, gt in zip(incorrect_completions, ground_truths)
        ]
        
        mean_correct = sum(correct_rewards) / len(correct_rewards)
        mean_incorrect = sum(incorrect_rewards) / len(incorrect_rewards)
        
        self.assertGreater(mean_correct, mean_incorrect)


if __name__ == "__main__":
    unittest.main()
