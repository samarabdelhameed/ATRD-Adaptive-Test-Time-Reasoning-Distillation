# ATRD: Methodology

## Adaptive Test-Time Reasoning Distillation

### 1. Problem Analysis

The NVIDIA Nemotron Model Reasoning Challenge requires improving the structured
reasoning capabilities of Nemotron-3-Nano-30B through a rank-32 LoRA adapter.
The evaluation uses numerical tolerance matching with `\boxed{}` answer format.

### 2. Approach: Three-Stage Pipeline

#### Stage 1: Failure-Grounded Synthetic Data Generation

**Motivation:** Instead of generating arbitrary training data, we focus on the
specific failure modes of the baseline model.

**Process:**
1. Run baseline model on competition-representative problems
2. Extract failure cases and classify failure types
3. Generate corrected reasoning traces using teacher model
4. Filter with LLM-as-judge quality scoring
5. Deduplicate to prevent memorization artifacts

#### Stage 2: PRM-Guided GRPO

**Motivation:** Standard SFT teaches the model *what* to output but not *how*
to explore solution paths. GRPO enables learning from multiple attempts.

**Process:**
1. Initialize with SFT-trained LoRA adapter
2. Define reward function combining:
   - Answer correctness (0.8 weight)
   - Format compliance (0.2 weight)
3. Generate group of completions per problem
4. Optimize policy using relative rewards within each group

#### Stage 3: Adaptive Budget Forcing

**Motivation:** Not all problems require equal reasoning effort. Allocating
more tokens to harder problems improves overall accuracy.

**Process:**
1. Estimate problem difficulty using heuristic features
2. Allocate token budget proportional to difficulty
3. Apply at inference time within competition constraints

### 3. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Rank-32 LoRA | Maximum allowed by competition |
| All projection modules | Maximizes adapter expressiveness |
| 4-bit quantization | Fits in Kaggle T4 GPU memory |
| BF16 training | Better numerical stability for reasoning |
| Seed fixing | Reproducibility across runs |

### 4. Ablation Studies

(To be filled after experiments)

### 5. Results

(To be filled after evaluation)

### 6. Conclusion

(To be filled after competition)
