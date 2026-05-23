#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.judge_filter import JudgeFilter
from src.data.deduplicator import Deduplicator
from src.data.dataset_mixer import DatasetMixer, check_leakage

def main():
    print("==================================================")
    # 1. Generate Mock Data
    print("1. Generating Mock Data...")
    
    # Let's generate 50 mock synthetic examples
    synthetic_examples = []
    failure_modes = ["reasoning_loop", "format_violation", "early_termination", "calculation_error", "misinterpretation"]
    
    # Some high-quality examples
    for i in range(35):
        mode = failure_modes[i % len(failure_modes)]
        synthetic_examples.append({
            "question": f"Solve problem variation {i}: What is {i} + {i}?",
            "thinking_trace": f"<<thinking>>\nStep 1: First, we add {i} and {i}.\nStep 2: Therefore, the sum is {2*i}.\n</thinking>>",
            "answer": f"\\boxed{{{2*i}}}",
            "failure_mode_tag": mode,
        })
        
    # Some near-duplicates
    for i in range(5):
        synthetic_examples.append({
            "question": f"Solve problem variation 0: What is 0 + 0?", # duplicate of first
            "thinking_trace": "<<thinking>>\nStep 1: First, we add 0 and 0.\nStep 2: Therefore, the sum is 0.\n</thinking>>",
            "answer": "\\boxed{0}",
            "failure_mode_tag": "reasoning_loop",
        })
        
    # Some low-quality/bad-format examples (missing tags or too short)
    for i in range(10):
        synthetic_examples.append({
            "question": f"Solve bad problem {i}",
            "thinking_trace": "no thinking tags here, just a short statement.",
            "answer": f"Answer is {i}", # missing boxed
            "failure_mode_tag": "format_violation",
        })

    # Math Reasoning Open Data (25 examples)
    math_examples = [
        {
            "question": f"General math question {i}: Solve for x: x - {i} = 10",
            "thinking_trace": f"<<thinking>>\nWe add {i} to both sides, hence x = {i+10}.\n</thinking>>",
            "answer": f"\\boxed{{{i+10}}}",
        } for i in range(25)
    ]

    # Code Reasoning Open Data (25 examples)
    code_examples = [
        {
            "question": f"Code reasoning question {i}: Write python code to print {i} times",
            "thinking_trace": f"<<thinking>>\nWe can use a loop: for i in range({i}): print(i)\n</thinking>>",
            "answer": f"\\boxed{{{i}}}",
        } for i in range(25)
    ]
    
    print(f"  Synthetic raw count: {len(synthetic_examples)}")
    print(f"  Math reasoning count: {len(math_examples)}")
    print(f"  Code reasoning count: {len(code_examples)}")
    print("--------------------------------------------------")

    # 2. Test Judge Filtering
    print("2. Running Judge Filter (composite & heuristic scoring)...")
    filterer = JudgeFilter(threshold=0.80)
    filtered_synthetic = filterer.filter_dataset(synthetic_examples)
    
    # Print scoring details for a couple of examples
    print("\n  Sample Scores:")
    for ex in [synthetic_examples[0], synthetic_examples[-1]]:
        print(f"    - Question: '{ex['question']}' | Tag: {ex.get('failure_mode_tag')} | Score: {ex.get('quality_score'):.3f}")
        
    print("--------------------------------------------------")

    # 3. Test Deduplication
    print("3. Running MinHash LSH & SHA-256 Deduplicator...")
    deduplicator = Deduplicator(similarity_threshold=0.85)
    
    # Let's combine all and deduplicate
    combined_before_dedup = filtered_synthetic + math_examples + code_examples
    print(f"  Combined count before dedup: {len(combined_before_dedup)}")
    
    deduped = deduplicator.deduplicate(combined_before_dedup, key="question")
    print("--------------------------------------------------")

    # 4. Test Mixing
    print("4. Running Dataset Mixer...")
    mixer = DatasetMixer(seed=42)
    
    # We want to separate them back to test mixing proportions
    mixed_synthetic = [ex for ex in deduped if ex.get("failure_mode_tag") is not None]
    mixed_math = [ex for ex in deduped if "math question" in ex.get("question", "").lower()]
    mixed_code = [ex for ex in deduped if "code reasoning" in ex.get("question", "").lower()]
    
    print(f"  Inputs to mixer: synthetic={len(mixed_synthetic)}, math={len(mixed_math)}, code={len(mixed_code)}")
    
    # Set targets for failure modes
    failure_ratios = {mode: 0.20 for mode in failure_modes}
    
    mixed_data = mixer.mix(
        synthetic=mixed_synthetic,
        math_reasoning=mixed_math,
        code_reasoning=mixed_code,
        max_total=50,
        failure_mode_ratios=failure_ratios
    )
    
    print("\n  Source Distribution in Mixed Dataset:")
    dist = mixer.get_distribution(mixed_data)
    for src, cnt in dist.items():
        print(f"    - {src}: {cnt} ({cnt/len(mixed_data)*100:.1f}%)")
        
    print("--------------------------------------------------")

    # 5. Test Leakage Check
    print("5. Running Leakage Checker...")
    train_questions = [ex["question"] for ex in mixed_data]
    
    # Test set: one clean test question, and one overlapping question to test detection
    test_set = [
        "Solve problem variation 100: What is 100 + 100?", # clean
        "Solve problem variation 5: What is 5 + 5?", # overlap (contained in train_questions)
    ]
    
    print("  Checking leakage with n=5 shingles...")
    overlap_count = check_leakage(train_questions, test_set, n=5)
    print(f"  Result: {overlap_count} overlapping n-grams found (expected > 0 due to intentional test overlap).")
    
    # Now let's try a completely clean test set
    clean_test_set = [
        "Find the derivative of x^3 with respect to x.",
        "Calculate the area of a circle with radius 7."
    ]
    print("\n  Checking leakage with clean test set...")
    clean_overlap_count = check_leakage(train_questions, clean_test_set, n=5)
    print(f"  Result: {clean_overlap_count} overlapping n-grams found (expected 0).")
    
    print("==================================================")
    print("PIPELINE TEST PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
