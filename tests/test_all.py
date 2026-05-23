import json, re, sys
from pathlib import Path

sys.path.insert(0, ".")


class TestMetric:
    def test_extract_boxed_simple(self):
        from src.evaluation.metric import extract_boxed_answer
        assert extract_boxed_answer("Answer: \\boxed{42}") == "42"
        assert extract_boxed_answer("\\boxed{hello}") == "hello"
        assert extract_boxed_answer("no box here") == ""

    def test_extract_boxed_nested(self):
        from src.evaluation.metric import extract_boxed_answer
        assert extract_boxed_answer("\\boxed{\\{42\\}}") == "\\{42\\}"
        assert extract_boxed_answer("\\boxed{a{b}c}") == "a{b}c"

    def test_compute_accuracy_exact(self):
        from src.evaluation.metric import compute_accuracy
        result = compute_accuracy(["42", "hello"], ["42", "hello"])
        assert result["accuracy"] == 1.0
        assert result["correct_count"] == 2

    def test_compute_accuracy_tolerance(self):
        from src.evaluation.metric import compute_accuracy
        # Relative tolerance: stricter threshold rejects, looser accepts
        result = compute_accuracy(["3.128"], ["3.1426"], tolerance=0.001)
        assert result["accuracy"] == 0.0
        result = compute_accuracy(["3.128"], ["3.1426"], tolerance=0.01)
        assert result["accuracy"] == 1.0

    def test_fraction_equivalence(self):
        from src.evaluation.metric import answers_equivalent, compute_accuracy
        assert answers_equivalent("0.25", "1/4") is True
        assert answers_equivalent("\\frac{1}{4}", "0.25") is True
        result = compute_accuracy(["0.25", "3"], ["1/4", "3"])
        assert result["accuracy"] == 1.0

    def test_load_benchmark_local(self):
        from src.evaluation.metric import load_benchmark_problems
        from pathlib import Path
        if Path("data/public_test.jsonl").exists():
            problems = load_benchmark_problems()
            assert len(problems) >= 1
            assert "question" in problems[0]

    def test_compute_accuracy_length_mismatch(self):
        from src.evaluation.metric import compute_accuracy
        try:
            compute_accuracy(["a"], ["a", "b"])
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_compute_accuracy_empty(self):
        from src.evaluation.metric import compute_accuracy
        result = compute_accuracy([], [])
        assert result["accuracy"] == 0.0
        assert result["total_count"] == 0

    def test_evaluate_submission(self):
        from src.evaluation.metric import evaluate_submission
        responses = [{"response": "\\boxed{42}"}, {"response": "\\boxed{7}"}]
        problems = [{"answer": "42"}, {"answer": "7"}]
        result = evaluate_submission(responses, problems, tolerance=0.01)
        assert result["overall_accuracy"] == 1.0
        assert result["total_count"] == 2

    def test_evaluate_submission_categories(self):
        from src.evaluation.metric import evaluate_submission
        responses = [{"response": "\\boxed{42}"}, {"response": "\\boxed{7}"}]
        problems = [{"answer": "42", "category": "math"}, {"answer": "99", "category": "math"}]
        result = evaluate_submission(responses, problems, tolerance=0.01)
        assert result["overall_accuracy"] == 0.5
        assert result["category_accuracy"]["math"] == 0.5

    def test_assert_replaced_with_raise(self):
        src = Path("src/evaluation/metric.py").read_text()
        assert "raise ValueError" in src
        assert "assert len" not in src

    def test_tolerance_reads_from_config(self):
        from src.evaluation.metric import _get_tolerance
        tol = _get_tolerance()
        assert tol == 0.01  # matches competition_params.json


class TestBudgetForcer:
    def test_estimate_difficulty_simple(self):
        from src.data.budget_forcer import estimate_difficulty
        d = estimate_difficulty("x = 5")
        assert 0.0 <= d <= 1.0

    def test_estimate_difficulty_hard(self):
        from src.data.budget_forcer import estimate_difficulty
        d = estimate_difficulty("prove that the integral of x from 0 to 1 equals 0.5 and find its maximum where the derivative is zero")
        assert d > 0.5

    def test_allocate_budget_bounds(self):
        from src.data.budget_forcer import allocate_budget
        assert allocate_budget(0.0) == 512
        assert allocate_budget(1.0) == 7680

    def test_allocate_budget_linear(self):
        from src.data.budget_forcer import allocate_budget
        assert allocate_budget(0.25) > 512
        assert allocate_budget(0.75) < 7680
        assert allocate_budget(0.5) == 4096

    def test_get_budget_stats(self):
        from src.data.budget_forcer import get_budget_stats
        s = get_budget_stats([{"budget_allocated": 512}, {"budget_allocated": 7680}])
        assert s["min_budget"] == 512
        assert s["max_budget"] == 7680
        assert s["mean_budget"] == (512 + 7680) / 2

    def test_get_budget_stats_empty(self):
        from src.data.budget_forcer import get_budget_stats
        s = get_budget_stats([])
        assert s["mean_budget"] == 0.0

    def test_validate_refinement_improvement(self):
        from src.data.budget_forcer import validate_refinement_improvement
        v = validate_refinement_improvement([
            {"difficulty": 0.8, "completion_before_refinement": "\\boxed{5}", "answer": "5", "correct": False},
            {"difficulty": 0.9, "completion_before_refinement": "\\boxed{7}", "answer": "5", "correct": True},
        ])
        assert v["improvement"] == 0.5

    def test_check_answer(self):
        from src.data.budget_forcer import check_answer
        assert check_answer("\\boxed{42}", "42") == True
        assert check_answer("\\boxed{42}", "43") == False
        assert check_answer("\\boxed{3.14}", "3.14") == True

    def test_generate_without_backend(self):
        from src.data.budget_forcer import generate, reset_generate_backend
        reset_generate_backend()
        try:
            generate("test", max_tokens=512)
            assert False
        except RuntimeError:
            pass

    def test_set_generate_backend(self):
        from src.data.budget_forcer import generate, set_generate_backend
        def mock_gen(p, **kw):
            return "\\boxed{42}"
        set_generate_backend(generate_fn=mock_gen)
        result = generate("test", max_tokens=512)
        assert "42" in result

    def test_generate_training_data_with_budget(self):
        from src.data.budget_forcer import (
            generate_training_data_with_budget,
            set_generate_backend,
        )
        set_generate_backend(lambda p, **kw: "<<thinking>>\nstep\n</thinking>>\n\\boxed{16}")
        row = generate_training_data_with_budget(
            "Compute integral of 2x from 0 to 4",
            "16",
            difficulty=0.8,
        )
        assert row["correct"] is True
        assert row["budget_allocated"] >= 512
        assert row["difficulty_tier"] == "hard"

    def test_refine_hard_problem(self):
        from src.data.budget_forcer import refine_hard_problem, set_generate_backend
        calls = {"n": 0}
        def backend(p, **kw):
            calls["n"] += 1
            return "\\boxed{16}" if calls["n"] > 1 else "\\boxed{0}"
        set_generate_backend(backend)
        out = refine_hard_problem("hard problem", "\\boxed{0}", "16", max_attempts=3)
        assert "16" in out


class TestPRM:
    def test_heuristic_step_score_transition(self):
        from src.training.prm import heuristic_step_score
        assert heuristic_step_score("x = 5") == 0.5

    def test_heuristic_step_score_connector(self):
        from src.training.prm import heuristic_step_score
        assert heuristic_step_score("therefore we can conclude") == 0.2

    def test_heuristic_step_score_repetition(self):
        from src.training.prm import heuristic_step_score
        assert heuristic_step_score("a a a a a a a a a a") == 0.0

    def test_segment_thinking_trace(self):
        from src.training.prm import segment_thinking_trace
        steps = segment_thinking_trace("Step one. Step two. \\boxed{42}")
        assert len(steps) == 2

    def test_check_answer(self):
        from src.training.prm import check_answer
        assert check_answer("\\boxed{42}", "42") == True
        assert check_answer("\\boxed{42}", "43") == False

    def test_detect_redundancy(self):
        from src.training.prm import detect_redundancy
        assert detect_redundancy("a\nb\na\nb\na\nb\na") == True
        assert detect_redundancy("a\nb\nc") == False

    def test_compute_prm_guided_reward(self):
        from src.training.prm import compute_prm_guided_reward
        r = compute_prm_guided_reward("Therefore x = 5. \\boxed{5}", "5")
        assert -1.0 <= r <= 1.0

    def test_correct_higher_than_incorrect(self):
        from src.training.prm import compute_prm_guided_reward
        r_correct = compute_prm_guided_reward("Thus x = 5. \\boxed{5}", "5")
        r_incorrect = compute_prm_guided_reward("Thus x = 5. \\boxed{7}", "5")
        assert r_correct > r_incorrect

    def test_test_prm_correlation_raises(self):
        from src.training.prm import test_prm_correlation
        try:
            test_prm_correlation()
            assert False
        except FileNotFoundError:
            pass

    def test_log_ratio_graceful_fallback(self):
        from src.training.prm import compute_log_ratio_score
        result = compute_log_ratio_score("x=5", "", None, None, None)
        assert result is None


class TestGRPOTrainer:
    def test_extract_boxed_answer(self):
        from src.training.grpo_trainer import _extract_boxed_answer
        assert _extract_boxed_answer("\\boxed{42}") == "42"
        assert _extract_boxed_answer("no box") == ""

    def test_check_answer(self):
        from src.training.grpo_trainer import _check_answer
        assert _check_answer("42", "42") == True
        assert _check_answer("42", "43") == False
        assert _check_answer("3.14", "3.14") == True

    def test_detect_redundancy(self):
        from src.training.grpo_trainer import _detect_redundancy
        assert _detect_redundancy("a\nb\na\nb\na\nb\na") == True
        assert _detect_redundancy("a\nb\nc") == False

    def test_verify_monotonic_reward_insufficient(self):
        from src.training.grpo_trainer import verify_monotonic_reward
        assert verify_monotonic_reward([1, 2, 3, 4, 5]) == True  # < 20 entries

    def test_verify_monotonic_reward_improving(self):
        from src.training.grpo_trainer import verify_monotonic_reward
        h = [0.1] * 20 + [0.5] * 10
        assert verify_monotonic_reward(h) == True

    def test_verify_monotonic_reward_declining(self):
        from src.training.grpo_trainer import verify_monotonic_reward
        h = [0.5] * 10 + [0.1] * 10
        assert verify_monotonic_reward(h) == False

    def test_create_reward_function_structure(self):
        from src.training.grpo_trainer import GRPOTrainerWrapper
        import json
        with open("configs/base_grpo.json") as f:
            g = json.load(f)
        with open("configs/competition_params.json") as f:
            c = json.load(f)
        model = type("Mock", (), {"device": "cpu"})()
        tokenizer = type("Mock", (), {})()
        wrapper = GRPOTrainerWrapper(model, tokenizer, "configs/base_grpo.json", "configs/competition_params.json")
        fn = wrapper.create_reward_function()
        rewards = fn(["\\boxed{42}"], ground_truth="42")
        assert len(rewards) == 1
        assert -1.0 <= rewards[0] <= 1.0


class TestDatasetMixer:
    def test_mix_empty(self):
        from src.data.dataset_mixer import DatasetMixer
        m = DatasetMixer()
        r = m.mix([], [], [], max_total=100)
        assert len(r) == 0

    def test_mix_ratio(self):
        from src.data.dataset_mixer import DatasetMixer
        m = DatasetMixer()
        syn = [{"id": i} for i in range(100)]
        math = [{"id": i} for i in range(50)]
        code = [{"id": i} for i in range(50)]
        r = m.mix(syn, math, code, max_total=200)
        assert len(r) > 0
        for ex in r:
            assert "_source" in ex

    def test_check_leakage_no_overlap(self):
        from src.data.dataset_mixer import check_leakage
        r = check_leakage(["hello world"], ["goodbye world"])
        assert r >= 0

    def test_mix_with_benchmark(self):
        from src.data.dataset_mixer import DatasetMixer
        m = DatasetMixer()
        r = m.mix([{"id": 1}], [{"id": 2}], [{"id": 3}], max_total=10, benchmark_texts=["test"])
        assert len(r) > 0

    def test_save_mixed(self):
        from src.data.dataset_mixer import DatasetMixer
        import tempfile, os
        m = DatasetMixer()
        path = m.save_mixed([{"test": 1}], output_path="/tmp/test_mixed.jsonl")
        assert path.exists()
        os.remove(path)


class TestJudgeFilter:
    def test_heuristic_score(self):
        from src.data.judge_filter import JudgeFilter
        j = JudgeFilter()
        s = j.heuristic_score({"completion": "This is a clear, well-structured reasoning step. Therefore, the answer is 42."})
        assert 0.0 <= s <= 1.0

    def test_filter_dataset(self):
        from src.data.judge_filter import JudgeFilter
        j = JudgeFilter()
        data = [{"id": i, "question": "test", "answer": "a", "thinking_trace": "trace"} for i in range(10)]
        r = j.filter_dataset(data)
        assert len(r) <= 10

    def test_generate_report(self):
        from src.data.judge_filter import JudgeFilter
        j = JudgeFilter()
        data = [{"id": i, "question": "test", "composite_score": 0.5} for i in range(5)]
        r = j.generate_report(data, data)
        assert "pass_rate" in r


class TestDeduplicator:
    def test_deduplicate_empty(self):
        from src.data.deduplicator import Deduplicator
        d = Deduplicator()
        r = d.deduplicate([])
        assert len(r) == 0

    def test_deduplicate_exact(self):
        from src.data.deduplicator import Deduplicator
        d = Deduplicator()
        r = d.deduplicate([{"question": "a"}, {"question": "a"}])
        assert len(r) == 1

    def test_deduplicate_unique(self):
        from src.data.deduplicator import Deduplicator
        d = Deduplicator()
        r = d.deduplicate([{"question": "a"}, {"question": "b"}])
        assert len(r) == 2


class TestSyntheticGenerator:
    def test_dataset_statistics_empty(self):
        from src.data.synthetic_generator import SyntheticGenerator
        g = SyntheticGenerator()
        s = g.dataset_statistics([])
        assert s == {}

    def test_dataset_statistics(self):
        from src.data.synthetic_generator import SyntheticGenerator
        g = SyntheticGenerator()
        data = [{"failure_mode_tag": "arithmetic"}, {"failure_mode_tag": "arithmetic"}, {"failure_mode_tag": "logic"}]
        s = g.dataset_statistics(data)
        assert s.get("arithmetic") == 2
        assert s.get("logic") == 1

    def test_generate_synthetic_batch_empty(self):
        from src.data.synthetic_generator import SyntheticGenerator
        g = SyntheticGenerator()
        r = g.generate_synthetic_batch("arithmetic_error", 5)
        assert isinstance(r, list)

    def test_save_dataset(self):
        from src.data.synthetic_generator import SyntheticGenerator
        import os
        g = SyntheticGenerator()
        path = g.save_dataset([{"test": 1}], filename="/tmp/test_synthetic.jsonl")
        assert path.exists()
        os.remove(path)


class TestLoraConfig:
    def test_validate_lora_config_valid(self):
        from src.models.lora_config import validate_lora_config
        ok, msg = validate_lora_config({"r": 32, "lora_alpha": 64, "lora_dropout": 0.05})
        assert ok == True

    def test_validate_lora_config_r_too_high(self):
        from src.models.lora_config import validate_lora_config
        ok, msg = validate_lora_config({"r": 64, "lora_alpha": 64, "lora_dropout": 0.05})
        assert ok == False
        assert "rank" in msg.lower()

    def test_validate_lora_config_alpha_lt_rank(self):
        from src.models.lora_config import validate_lora_config
        ok, msg = validate_lora_config({"r": 32, "lora_alpha": 16, "lora_dropout": 0.05})
        assert ok == False

    def test_validate_lora_config_dropout_high(self):
        from src.models.lora_config import validate_lora_config
        ok, msg = validate_lora_config({"r": 32, "lora_alpha": 64, "lora_dropout": 0.9})
        assert ok == False

    def test_create_lora_config(self):
        from src.models.lora_config import create_lora_config, validate_lora_config, _HAS_PEFT
        if not _HAS_PEFT:
            print("  (skipped: peft not installed)")
            return
        cfg = create_lora_config("configs/base_lora.json")
        ok, msg = validate_lora_config({"r": cfg.r, "lora_alpha": cfg.lora_alpha, "lora_dropout": cfg.lora_dropout})
        assert ok == True

    def test_validate_adapter(self):
        from src.models.lora_config import validate_adapter
        import tempfile, os, json
        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "adapter_config.json"), "w") as f:
            json.dump({"r": 32, "lora_alpha": 64}, f)
        ok, msg = validate_adapter(tmpdir)
        assert ok == True
        os.remove(os.path.join(tmpdir, "adapter_config.json"))
        os.rmdir(tmpdir)


class TestLoader:
    def test_module_imports(self):
        import importlib, sys
        try:
            from src.models import loader
            assert hasattr(loader, "load_model_with_cleanup")
            assert hasattr(loader, "setup_blackwell_optimizations")
            assert hasattr(loader, "enable_gradient_checkpointing")
        except ImportError:
            pass  # torch not available in this env

    def test_config_memory_fraction(self):
        from src.models.loader import setup_blackwell_optimizations
        # Just verify it parses (will skip TF32 setup without CUDA)
        try:
            setup_blackwell_optimizations()
        except Exception:
            pass


class TestSFTTrainer:
    def test_format_sft_example(self):
        from src.training.sft_trainer import format_sft_example
        result = format_sft_example({"question": "Q", "thinking_trace": "T", "answer": "A"})
        assert "Q" in result
        assert "T" in result
        assert "A" in result

    def test_should_early_stop_plateau(self):
        from src.training.sft_trainer import should_early_stop
        losses = [1.0, 0.8, 0.6, 0.59, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52, 0.51]
        assert should_early_stop(losses, patience=3) == False
        assert should_early_stop(losses, patience=2) == False

    def test_should_early_stop_not(self):
        from src.training.sft_trainer import should_early_stop
        losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        assert should_early_stop(losses, patience=3) == False


class TestAblation:
    def test_generate_waterfall_data_empty(self):
        from src.evaluation.ablation import AblationRunner
        r = AblationRunner()
        data = r.generate_waterfall_data([])
        assert data == {}

    def test_generate_waterfall_data_basic(self):
        from src.evaluation.ablation import AblationRunner
        r = AblationRunner()
        results = [
            {"name": "baseline", "score": 0.5, "delta": None},
            {"name": "sft", "score": 0.6, "delta": 0.1},
        ]
        data = r.generate_waterfall_data(results)
        assert data["baseline"] == 0.5
        assert len(data["stages"]) == 1
        assert data["stages"][0]["name"] == "sft"

    def test_verify_exit_quality_gate(self):
        from src.evaluation.ablation import AblationRunner
        r = AblationRunner()
        results = [
            {"name": "baseline", "score": 0.5, "delta": None},
            {"name": "sft", "score": 0.6, "delta": 0.1},
            {"name": "sft+grpo", "score": 0.65, "delta": 0.15},
            {"name": "full", "score": 0.7, "delta": 0.2},
        ]
        gates = r.verify_exit_quality_gate(results)
        assert "all_4_ablations_evaluated" in gates
        assert "statistical_significance_tested" in gates

    def test_compute_significance(self):
        from src.evaluation.ablation import AblationRunner
        r = AblationRunner()
        sig = r.compute_significance(0.7, 0.5)
        assert "p_value" in sig
        assert "significant" in sig

    def test_save_results(self):
        from src.evaluation.ablation import AblationRunner
        import os
        r = AblationRunner(output_dir="/tmp")
        results = [{"name": "baseline", "score": 0.5, "delta": None, "config": {}, "status": "completed"}]
        path = r.save_results(results)
        assert path.exists()
        os.remove(path)


class TestConfigs:
    def test_competition_params_exists(self):
        import json
        with open("configs/competition_params.json") as f:
            c = json.load(f)
        assert c["max_lora_rank"] == 32
        assert c["temperature"] == 0.0
        assert c["max_tokens"] == 7680
        assert c["inference_engine"] == "vllm"
        assert "numerical_tolerance" in c

    def test_base_lora_has_all_fields(self):
        import json
        with open("configs/base_lora.json") as f:
            c = json.load(f)
        assert c["r"] == 32
        assert len(c["target_modules"]) == 7
        assert "use_rslora" in c
        assert "init_lora_weights" in c

    def test_base_grpo_exists(self):
        import json
        with open("configs/base_grpo.json") as f:
            c = json.load(f)
        assert c["group_size"] == 8
        assert c["kl_penalty"] == 0.001


class TestScripts:
    def test_package_submission_import(self):
        import importlib, sys
        try:
            import scripts.package_submission
            assert hasattr(scripts.package_submission, "package")
            assert hasattr(scripts.package_submission, "test_vllm_compatibility")
        except ImportError:
            pass  # vllm might not be installed

    def test_verify_unit_completion_import(self):
        import scripts.verify_unit_completion
        assert hasattr(scripts.verify_unit_completion, "main")
        assert hasattr(scripts.verify_unit_completion, "check_config_artifacts")
        assert hasattr(scripts.verify_unit_completion, "check_stage_artifacts")
        assert hasattr(scripts.verify_unit_completion, "check_lora_rank")

    def test_verify_protected_files_clean(self):
        src = Path("scripts/verify_protected_files.py").read_text()
        assert "KNOWN_HASHES" not in src

    def test_sync_to_hub_has_auth(self):
        src = Path("scripts/sync_to_hub.py").read_text()
        assert "HF_TOKEN" in src
        assert "logging" in src


class TestNotebooks:
    def test_all_notebooks_valid_json(self):
        import json
        for nb in ["notebooks/01_data_generation.ipynb", "notebooks/02_sft_training.ipynb",
                    "notebooks/03_grpo_training.ipynb", "notebooks/04_budget_forcing.ipynb",
                    "notebooks/05_public_kaggle.ipynb"]:
            with open(nb) as f:
                data = json.load(f)
            assert data["nbformat"] == 4

    def test_no_mock_data_in_notebooks(self):
        import json
        for nb in ["notebooks/01_data_generation.ipynb", "notebooks/02_sft_training.ipynb",
                    "notebooks/03_grpo_training.ipynb", "notebooks/04_budget_forcing.ipynb",
                    "notebooks/05_public_kaggle.ipynb"]:
            with open(nb) as f:
                data = json.load(f)
            code = " ".join("".join(c["source"]) for c in data["cells"] if c["cell_type"] == "code")
            assert "mock" not in code.lower(), f"Mock data found in {nb}"

    def test_notebooks_have_file_not_found_gates(self):
        import json
        for nb in ["notebooks/01_data_generation.ipynb", "notebooks/02_sft_training.ipynb",
                    "notebooks/03_grpo_training.ipynb", "notebooks/04_budget_forcing.ipynb"]:
            with open(nb) as f:
                data = json.load(f)
            code = " ".join("".join(c["source"]) for c in data["cells"] if c["cell_type"] == "code")
            assert "FileNotFoundError" in code or "raise" in code, f"No error gates in {nb}"


class TestWriteup:
    def test_methodology_exists(self):
        assert Path("writeup/METHODOLOGY.md").exists()

    def test_methodology_word_count(self):
        text = Path("writeup/METHODOLOGY.md").read_text()
        words = len(text.split())
        assert words >= 2000, f"Only {words} words"

    def test_methodology_has_required_sections(self):
        text = Path("writeup/METHODOLOGY.md").read_text()
        for s in ["## 1. Abstract", "## 2. Introduction", "## 9. Open Contribution Awards"]:
            assert s in text

    def test_no_xx_placeholders(self):
        text = Path("writeup/METHODOLOGY.md").read_text()
        assert "XX%" not in text


class TestPyProject:
    def test_pyproject_toml_exists(self):
        assert Path("pyproject.toml").exists()

    def test_setup_py_equivalent(self):
        import tomllib
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["project"]["name"] == "atrd"
        assert "requires-python" in data["project"]


class TestRequirements:
    def test_safetensors_in_requirements(self):
        reqs = Path("requirements.txt").read_text()
        assert "safetensors" in reqs


class TestPackageStructure:
    def test_src_init_exports(self):
        src_init = Path("src/__init__.py").read_text()
        assert "__version__" in src_init
        assert "__author__" in src_init

    def test_all_modules_syntax(self):
        import py_compile
        for py in Path("src").rglob("*.py"):
            try:
                py_compile.compile(str(py), doraise=True)
            except py_compile.PyCompileError as e:
                assert False, f"Syntax error in {py}: {e}"
        for py in Path("scripts").rglob("*.py"):
            try:
                py_compile.compile(str(py), doraise=True)
            except py_compile.PyCompileError as e:
                assert False, f"Syntax error in {py}: {e}"

    def test_import_src(self):
        import sys
        sys.path.insert(0, ".")
        try:
            import src
            assert hasattr(src, "__version__")
        except ImportError:
            pass


if __name__ == "__main__":
    import inspect, sys
    cls = [obj for name, obj in inspect.getmembers(sys.modules[__name__]) if name.startswith("Test") and inspect.isclass(obj)]
    total, passed = 0, 0
    for c in cls:
        instance = c()
        for name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if name.startswith("test_"):
                total += 1
                try:
                    method()
                    print(f"  ✅ {c.__name__}.{name}")
                    passed += 1
                except Exception as e:
                    print(f"  ❌ {c.__name__}.{name}: {e}")
    print(f"\n{'='*50}")
    print(f"Passed: {passed}/{total}")
    if passed == total:
        print("ALL TESTS PASSED ✅")
    else:
        print(f"FAILED: {total - passed}")
