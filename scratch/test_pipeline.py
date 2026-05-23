import sys
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.judge_filter import JudgeFilter
from src.data.deduplicator import Deduplicator
from src.data.dataset_mixer import DatasetMixer, check_leakage

def run_test():
    print("=== ATRD Data Curation Pipeline Verification ===")
    
    # 1. Create simulated raw synthetic data
    # Some good, some bad, some duplicates
    raw_synthetic = [
        # Good examples
        {
            "question": "Solve x + 3 = 5.",
            "thinking_trace": "<<thinking>> We need to subtract 3 from both sides. x = 5 - 3. Therefore x = 2. </thinking>>",
            "answer": "\\boxed{2}",
            "failure_mode_tag": "calculation_error"
        },
        {
            "question": "Solve 2x = 8.",
            "thinking_trace": "<<thinking>> Dividing both sides by 2 gives x = 4. Hence x = 4. </thinking>>",
            "answer": "\\boxed{4}",
            "failure_mode_tag": "reasoning_loop"
        },
        # Duplicate of the second one (near-duplicate)
        {
            "question": "Solve 2x = 8.",
            "thinking_trace": "<<thinking>> Dividing both sides by 2 gives x = 4. So x = 4. </thinking>>",
            "answer": "\\boxed{4}",
            "failure_mode_tag": "reasoning_loop"
        },
        # Exact duplicate of the first one
        {
            "question": "Solve x + 3 = 5.",
            "thinking_trace": "<<thinking>> We need to subtract 3 from both sides. x = 5 - 3. Therefore x = 2. </thinking>>",
            "answer": "\\boxed{2}",
            "failure_mode_tag": "calculation_error"
        },
        # Low quality example (no boxed answer)
        {
            "question": "Solve x - 1 = 0.",
            "thinking_trace": "We get x = 1.",
            "answer": "1",
            "failure_mode_tag": "format_violation"
        },
        # Another good example
        {
            "question": "Find the maximum of f(x) = -x^2 + 4.",
            "thinking_trace": "<<thinking>> The derivative is f'(x) = -2x. Setting f'(x) = 0 gives x = 0. Therefore maximum value is f(0) = 4. </thinking>>",
            "answer": "\\boxed{4}",
            "failure_mode_tag": "calculation_error"
        },
    ]

    print(f"Initial raw examples: {len(raw_synthetic)}")

    # 2. Test JudgeFilter
    print("\n--- Testing JudgeFilter ---")
    judge = JudgeFilter(threshold=0.80)
    filtered = judge.filter_dataset(raw_synthetic)
    # Cutoff at 80% should retain 4 out of 6 examples (cutoff = max(1, int(6 * 0.80)) = 4)
    print(f"Filtered examples: {len(filtered)}")
    for ex in filtered:
        print(f"  Score: {ex['quality_score']:.3f} | Question: {ex['question']}")
    
    assert len(filtered) == 4, f"Expected 4 examples, got {len(filtered)}"

    # 3. Test Deduplicator
    print("\n--- Testing Deduplicator ---")
    dedup = Deduplicator(similarity_threshold=0.85)
    deduplicated = dedup.deduplicate(filtered, key="question")
    # Duplicate checking should remove the duplicates but preserve exactly one copy
    print(f"Deduplicated examples: {len(deduplicated)}")
    for ex in deduplicated:
        print(f"  Question: {ex['question']}")
        
    # We should have removed the duplicates but kept the originals.
    # The filtered set of 4 has:
    # 1. "Find the maximum of f(x)..." (unique)
    # 2. "Solve x + 3 = 5" (duplicate present in raw, filtered should contain the first one)
    # 3. "Solve 2x = 8" (two copies in raw, both good so they pass filter, but they are duplicates)
    # So deduplication should keep exactly one "Solve x + 3 = 5" and one "Solve 2x = 8", plus the "Find the maximum...".
    # Therefore, final count should be 3.
    print(f"Deduplicated count: {len(deduplicated)} (Expected: 3)")
    assert len(deduplicated) == 3, f"Expected 3 deduplicated examples, got {len(deduplicated)}"

    # 4. Test DatasetMixer
    print("\n--- Testing DatasetMixer ---")
    mixer = DatasetMixer(seed=42)
    math_data = [{"question": f"Math Q{i}", "answer": f"{i}"} for i in range(10)]
    code_data = [{"question": f"Code Q{i}", "answer": f"{i}"} for i in range(10)]
    
    # Let's mix them with max_total = 10 (5 synthetic, 2.5 math, 2.5 code)
    mixed = mixer.mix(
        synthetic=deduplicated,
        math_reasoning=math_data,
        code_reasoning=code_data,
        max_total=10
    )
    print(f"Mixed dataset size: {len(mixed)}")
    dist = mixer.get_distribution(mixed)
    print(f"Distribution: {dist}")
    
    # 5. Test Leakage Check
    print("\n--- Testing Leakage Checker ---")
    train_texts = [ex["question"] for ex in mixed]
    test_texts = ["Solve 2x = 8.", "Solve x + 3 = 5."]
    
    overlap = check_leakage(train_texts, test_texts, n=5)
    print(f"Leakage overlap count: {overlap}")

    print("\n=== All Tests Passed Successfully! ===")

if __name__ == "__main__":
    run_test()
