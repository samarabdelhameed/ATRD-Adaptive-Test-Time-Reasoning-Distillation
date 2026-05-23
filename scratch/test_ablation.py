import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.ablation import AblationRunner, AblationConfig

def run_tests():
    print("Testing AblationRunner...")
    runner = AblationRunner(output_dir="logs")
    
    # 1. Ablation configurations
    ablation_configs = [
        AblationConfig(name="baseline", components_active=["base"], config={}),
        AblationConfig(name="sft_only", components_active=["base", "sft"], config={"lora_rank": 32}),
        AblationConfig(name="sft_grpo", components_active=["base", "sft", "grpo"], config={"group_size": 8}),
        AblationConfig(name="full_pipeline", components_active=["base", "sft", "grpo", "budget"], config={"budget_forcing": True}),
    ]

    # Mock models and eval function for testing the logic
    # We'll simulate progressive improvement
    def mock_train(config):
        return config

    def mock_eval(model):
        # Simulate accuracies
        if "lora_rank" in model and "group_size" not in model:
            return 0.73
        elif "group_size" in model and "budget_forcing" not in model:
            return 0.78
        elif "budget_forcing" in model:
            return 0.81
        else:
            return 0.62

    # 2. Run all ablations
    results = runner.run_all_ablations(ablation_configs, mock_train, mock_eval)
    
    assert len(results) == 4, "Should have 4 results"
    assert results[0]["score"] == 0.62
    assert results[1]["score"] == 0.73
    assert abs(results[1]["delta"] - 0.11) < 1e-5
    assert results[2]["score"] == 0.78
    assert abs(results[2]["delta"] - 0.05) < 1e-5
    assert results[3]["score"] == 0.81
    assert abs(results[3]["delta"] - 0.03) < 1e-5
    print("✓ run_all_ablations logic passed")

    # 3. Statistical significance
    baseline_scores = [0.6, 0.6, 0.6, 0.6, 0.6]
    treatment_scores = [0.7, 0.7, 0.8, 0.7, 0.8]
    sig = runner.compute_significance(baseline_scores, treatment_scores)
    p_value = sig["p_value"] if isinstance(sig, dict) else sig
    assert p_value < 0.05, "P-value should show significance"
    print("✓ compute_significance passed")

    # 4. Stratified evaluation
    def mock_eval_stratified(bin_name):
        return {
            "easy": 0.94 if bin_name == "easy" else 0.0,
            "medium": 0.74 if bin_name == "medium" else 0.0,
            "hard": 0.42 if bin_name == "hard" else 0.0
        }[bin_name]
    
    strat_results = runner.stratified_evaluation(results, mock_eval_stratified)
    assert "easy" in strat_results
    assert strat_results["easy"] == 0.94
    print("✓ stratified_evaluation passed")

    # 5. Generalization gap
    gap_info = runner.check_generalization_gap(0.83, 0.85)
    assert gap_info["is_positive"] is True
    assert gap_info["generalization_gap"] > 0
    print("✓ check_generalization_gap passed")

    # 6. Save results
    output_path = Path("logs/ablation_results.mock.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "ablations": results,
            "stratified_evaluation": strat_results,
            "generalization_gap": gap_info,
            "note": "MOCK — for unit tests only; do not use in write-up",
        }, f, indent=2)
    assert Path(output_path).exists()
    
    with open(output_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
        assert len(saved_data["ablations"]) == 4
        assert saved_data["ablations"][0]["score"] == 0.62
    print("✓ mock results file passed")

    # 7. Waterfall data
    waterfall = runner.generate_waterfall_data(results)
    assert len(waterfall["stages"]) == 3
    assert waterfall["baseline"] == 0.62
    print("✓ generate_waterfall_data passed")

    # 8. Verify Quality Gates
    gates = runner.verify_exit_quality_gate(results, strat_results, gap_info)
    assert gates["all_gates_passed"] is True
    print("✓ verify_exit_quality_gate passed")

    print("\nALL ABLATION TESTS PASSED")

if __name__ == "__main__":
    run_tests()
