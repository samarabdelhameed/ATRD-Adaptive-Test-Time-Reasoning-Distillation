import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.baseline import BaselineEvaluator

if __name__ == "__main__":
    print("Testing BaselineEvaluator...")
    evaluator = BaselineEvaluator()
    dataset = [
        {"question": "What is 5 + 3?", "answer": "8", "question_id": "q1"},
        {"question": "Solve for x: x - 2 = 10", "answer": "12", "question_id": "q2"},
        {"question": "What is 7 * 6?", "answer": "42", "question_id": "q3"},
    ]
    evaluator.evaluate(dataset)
    print("Test passed successfully.")
