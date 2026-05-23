import unittest
from unittest.mock import patch, MagicMock
import json
from pathlib import Path
from src.data.synthetic_generator import SyntheticGenerator, GeneratorConfig


class TestSyntheticGenerator(unittest.TestCase):
    def setUp(self):
        # Create generator instance with dummy configs
        self.generator = SyntheticGenerator(
            config_path="configs/competition_params.json",
            output_dir="data/test_synthetic",
            api_key="dummy_api_key",
            seed=42
        )

    def test_difficulty_estimation(self):
        """Test that difficulty estimator returns float scores within [0.0, 1.0]."""
        easy_q = "What is 2 + 2?"
        hard_q = (
            "Prove that the integral of x^2 from 0 to 3 is 9, and verify the maximum "
            "and minimum values using the optimize derivative theorem for a matrix "
            "determinant eigenvalue calculation such that the expected value holds."
        )

        easy_score = self.generator._estimate_difficulty(easy_q)
        hard_score = self.generator._estimate_difficulty(hard_q)

        print(f"\n[Difficulty Estimate Tests]")
        print(f"Easy Q Score: {easy_score:.2f} (Expected: low)")
        print(f"Hard Q Score: {hard_score:.2f} (Expected: high)")

        self.assertTrue(0.0 <= easy_score <= 1.0)
        self.assertTrue(0.0 <= hard_score <= 1.0)
        self.assertTrue(hard_score > easy_score)

    def test_robust_response_parser(self):
        """Test that our parser handles different formatting styles of LLM response."""
        # Style 1: Separated by double newlines, starts with Question/Thinking/Answer
        style_1 = (
            "Question: Solve 3x = 9.\n\n"
            "Thinking:\n"
            "We divide both sides by 3 to isolate x.\n"
            "So x = 3.\n\n"
            "Answer: 3"
        )

        # Style 2: Consecutively written, no double newlines, includes thinking tags
        style_2 = (
            "Question: Integrate x from 0 to 2.\n"
            "Thinking: <<thinking>>We compute x^2/2 evaluated from 0 to 2. That is 2 - 0 = 2.</thinking>\n"
            "Answer: \\boxed{2}"
        )

        # Style 3: Multiple questions in one raw response string
        style_3 = (
            "Question: Find 5 + 3.\n"
            "Thinking: Basic addition yields 8.\n"
            "Answer: 8\n\n"
            "Question: Find 10 - 4.\n"
            "Thinking: Basic subtraction yields 6.\n"
            "Answer: 6"
        )

        parsed_1 = self.generator._parse_batch_response(style_1, "calculation_error")
        parsed_2 = self.generator._parse_batch_response(style_2, "reasoning_loop")
        parsed_3 = self.generator._parse_batch_response(style_3, "early_termination")

        print(f"\n[Parser Robustness Tests]")
        print(f"Parsed Style 1 Count: {len(parsed_1)}")
        print(f"Parsed Style 2 Count: {len(parsed_2)}")
        print(f"Parsed Style 3 Count: {len(parsed_3)}")

        # Verification style 1
        self.assertEqual(len(parsed_1), 1)
        self.assertEqual(parsed_1[0]["question"], "Solve 3x = 9.")
        self.assertIn("<<thinking>>", parsed_1[0]["thinking_trace"])
        self.assertEqual(parsed_1[0]["answer"], "\\boxed{3}")

        # Verification style 2
        self.assertEqual(len(parsed_2), 1)
        self.assertEqual(parsed_2[0]["question"], "Integrate x from 0 to 2.")
        self.assertIn("We compute x^2/2", parsed_2[0]["thinking_trace"])
        self.assertEqual(parsed_2[0]["answer"], "\\boxed{2}")

        # Verification style 3
        self.assertEqual(len(parsed_3), 2)
        self.assertEqual(parsed_3[0]["question"], "Find 5 + 3.")
        self.assertEqual(parsed_3[0]["answer"], "\\boxed{8}")
        self.assertEqual(parsed_3[1]["question"], "Find 10 - 4.")
        self.assertEqual(parsed_3[1]["answer"], "\\boxed{6}")

    @patch("src.data.synthetic_generator.requests.post")
    def test_mock_generation_loop(self, mock_post):
        """Test the full pipeline generation loop under mocked API returns."""
        # Setup mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": (
                        "Question: What is 12 / 4?\n"
                        "Thinking: 12 divided by 4 is 3.\n"
                        "Answer: 3"
                    )
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Setup dummy failure examples
        dummy_failures = {
            "calculation_error": [{"question": "What is 10/2?"}],
            "reasoning_loop": [{"question": "Simplify 2x + 2x."}]
        }

        # Run generator
        results = self.generator.generate_per_failure_mode(
            failure_examples=dummy_failures,
            problems_per_mode=2
        )

        print(f"\n[Mock Generation Loop Tests]")
        print(f"Total problems generated: {len(results)}")

        self.assertEqual(len(results), 4) # 2 modes * 2 per mode
        for r in results:
            self.assertEqual(r["answer"], "\\boxed{3}")
            self.assertIn("calculation_error", ["calculation_error", "reasoning_loop"])

        # Test saving
        output_file = self.generator.save_dataset(results, "raw_synthetic_dataset.jsonl")
        print(f"Dataset successfully saved to: {output_file}")
        self.assertTrue(output_file.exists())

        # Cleanup
        if output_file.exists():
            output_file.unlink()
        if output_file.parent.exists():
            output_file.parent.rmdir()


if __name__ == "__main__":
    unittest.main()
