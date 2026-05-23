# 15 — Final Evaluation & Ablation Studies Specification

## ATRD: Adaptive Test-Time Reasoning Distillation

### 1. Purpose and Setup Order
This specification defines the final evaluation phase, which validates the independent contribution of each pipeline component through systematic ablation studies.

> [!IMPORTANT]
> Read `14-budget-forcing.md` before implementing. All training phases must be complete.

---

## 2. Ablation Study Framework (`src/evaluation/ablation.py`)

### 2.1 Component Configurations

| # | Configuration | Components Active | Expected Accuracy |
|---|--------------|------------------|-------------------|
| 1 | **Baseline** | Base Nemotron (no LoRA) | Baseline score |
| 2 | **SFT-only** | SFT LoRA adapter | +10% over baseline |
| 3 | **SFT + GRPO** | SFT + GRPO adapter | +3–5% over SFT |
| 4 | **Full Pipeline** | SFT + GRPO + Budget Forcing | +3% net over GRPO |

### 2.2 Ablation Runner
```python
class AblationRunner:
    """Run systematic ablation studies across configurations."""

    def run_ablation(
        self,
        name: str,
        config: Dict[str, Any],
        train_fn: Callable,
        eval_fn: Callable,
        baseline_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        try:
            model = train_fn(config)
            score = eval_fn(model)
            status = "completed"
        except Exception as e:
            score = 0.0
            status = f"failed: {str(e)}"

        return {
            "name": name,
            "score": score,
            "delta": score - baseline_score if baseline_score else None,
            "elapsed_seconds": time.time() - start_time,
            "status": status,
        }
```

### 2.3 Output Schema (`logs/ablation_results.json`)
```json
{
  "ablations": [
    {
      "name": "baseline",
      "accuracy": 0.62,
      "delta": null,
      "config": {},
      "status": "completed"
    },
    {
      "name": "sft_only",
      "accuracy": 0.73,
      "delta": 0.11,
      "config": {"lora_rank": 32, "epochs": 3},
      "status": "completed"
    },
    {
      "name": "sft_grpo",
      "accuracy": 0.78,
      "delta": 0.05,
      "config": {"group_size": 8, "kl_penalty": 0.001},
      "status": "completed"
    },
    {
      "name": "full_pipeline",
      "accuracy": 0.81,
      "delta": 0.03,
      "config": {"budget_forcing": true, "wait_injections": 3},
      "status": "completed"
    }
  ],
  "summary": {
    "baseline": 0.62,
    "total_improvement": 0.19,
    "sft_contribution": 0.11,
    "grpo_contribution": 0.05,
    "budget_forcing_contribution": 0.03
  }
}
```

### 2.4 Statistical Significance
```python
from scipy import stats

def compute_significance(baseline_scores: List[float], treatment_scores: List[float]) -> float:
    """Compute p-value using paired t-test."""
    t_stat, p_value = stats.ttest_rel(treatment_scores, baseline_scores)
    return p_value  # p < 0.05 = statistically significant
```

---

## 3. Stratified Evaluation

### 3.1 Per-Difficulty-Bin Analysis
| Bin | Definition | Baseline | Full Pipeline | Delta |
|-----|-----------|----------|---------------|-------|
| Easy | Bottom 20% by length | 0.95 | 0.94 | -0.01 (≤ 2% OK) |
| Medium | Middle 60% | 0.65 | 0.74 | +0.09 |
| Hard | Top 20% by length | 0.25 | 0.42 | +0.17 (≥ 5% OK) |

---

## 4. Generalization Gap Check
```
Public Test Accuracy: 0.83
Private Test Accuracy: 0.85
Generalization Gap: +0.02 (POSITIVE ✓)
```
- **Target**: Private test accuracy > Public test accuracy
- **Signal**: Model generalizes beyond public test, no overfitting

---

## 5. Visualization Requirements

### 5.1 Ablation Waterfall Chart
```
Accuracy Gain Waterfall:
Baseline:        0.62  |████████████████████████████████████████
+SFT:           +0.11  |██████                                   
+GRPO:          +0.05  |███                                      
+BudgetForcing: +0.03  |██                                       
────────────────────────────────────────────────────
Total:           0.81  |████████████████████████████████████████████████
```

### 5.2 Per-Component Accuracy Bar Chart
Grouped bars showing accuracy for each configuration across easy/medium/hard bins.

---

## 6. Exit Quality Gate
- [ ] All 4 ablation configurations evaluated
- [ ] Each component shows positive contribution (delta > 0)
- [ ] Statistical significance tested (p < 0.05 for each delta)
- [ ] Stratified evaluation shows budget forcing helps hard problems, doesn't hurt easy
- [ ] Generalization gap: private > public (anti-overfitting signal)
- [ ] `logs/ablation_results.json` saved with full report
- [ ] Ablation waterfall chart generated in notebook
