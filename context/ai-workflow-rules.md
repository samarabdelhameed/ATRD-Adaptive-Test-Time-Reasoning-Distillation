## AI Workflow Rules

### Development Approach

Build this project incrementally using a **spec-driven, phase-gated workflow**. The project is organized into four distinct phases (Data Curation → SFT → GRPO → Test-Time Adaptation), each with defined inputs, outputs, and quality gates. No phase may begin until the preceding phase passes its exit criteria. All implementation is driven by structured specification files stored in the `context/` directory; agents must read these specs before executing any work and must not infer, invent, or extrapolate behavior beyond what is explicitly specified.

### Context File Hierarchy

The following files define the project state and must be read in order before any implementation:

| Order | File | Purpose |
|-------|------|---------|
| 1 | `agents.md` | Entry point; instructs agent to read all context files in sequence |
| 2 | `project_overview.md` | Mission, problem statement, success metrics, constraints |
| 3 | `architecture.md` | Tech stack, system boundaries, storage model, invariants |
| 4 | `code_standards.md` | Python/ML conventions, notebook structure, naming conventions |
| 5 | `ai_workflow_rules.md` | This file — how the agent behaves, scopes work, handles decisions |
| 6 | `progress_tracker.md` | Current phase, completed units, in-progress items, blockers |

### Work Unit Discipline

1. **One Phase at a Time**
   - Never combine work across phase boundaries in a single session.
   - Example: Do not begin GRPO training code while Phase 1 (Data Curation) remains incomplete.

2. **One Notebook at a Time**
   - Each phase maps to exactly one Kaggle notebook:
     - `01_data_generation.ipynb`
     - `02_sft_training.ipynb`
     - `03_grpo_training.ipynb`
     - `04_budget_forcing.ipynb`
   - Finish one notebook completely before opening the next.

3. **Spec-Driven Execution**
   - Before writing any code, the agent must read the phase's specification file from `context/feature_specs/`.
   - The spec defines: goal, design decisions, implementation details, dependencies, and verification checklist.
   - The agent implements exactly as specified — no additional features, no inferred behavior.

4. **No Scope Creep**
   - If a requirement is ambiguous, the agent must:
     - Flag the ambiguity in `progress_tracker.md`
     - Pause execution
     - Wait for human clarification
   - The agent must not "guess" or "fill in" missing requirements.

### Decision Handling

| Scenario | Agent Action |
|----------|-------------|
| Spec is clear | Execute exactly as written |
| Spec is ambiguous | Flag in progress tracker, pause, request clarification |
| Spec contradicts architecture | Flag as invariant violation, pause, request resolution |
| New idea emerges during work | Log in progress tracker as "future consideration", do not implement |
| Error occurs during execution | Analyze root cause, propose fix in progress tracker, wait for approval |

### Code Quality Rules

1. **Type Safety**
   - All Python functions must have type hints (`def func(x: int) -> str:`)
   - All data structures must use `TypedDict`, `dataclass`, or Pydantic models where appropriate

2. **Error Handling**
   - No bare `except:` clauses
   - All file I/O, API calls, and GPU operations wrapped in `try/except` with specific exception types
   - Error messages must include context (variable values, line numbers, stack traces logged)

3. **Reproducibility**
   - All random seeds fixed (`random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`)
   - All hyperparameters centralized in `config.yaml` or notebook cell #1
   - All package versions pinned in `requirements.txt`

4. **Notebook Structure**
   - Cell #1: Imports + seed fixing
   - Cell #2: Configuration (hyperparameters, paths, flags)
   - Cell #3: Helper functions (with docstrings)
   - Cell #4+: Main execution blocks
   - Final cells: Evaluation + visualization

5. **GPU Memory Management**
   - Explicit `torch.cuda.empty_cache()` after model loading
   - Gradient checkpointing enabled for models > 10B parameters
   - Batch size reduced automatically on OOM (catch `RuntimeError`, halve batch, retry)

### Progress Tracking

After every work session, the agent must update `progress_tracker.md` with:

```markdown
## Session [Date] — [Phase] — [Notebook]

### Completed
- [x] Item one
- [x] Item two

### In Progress
- [ ] Item three (ETA: 2 hours)

### Blockers
- None / [Description]

### Decisions Made
- [Decision and rationale]

### Next Session
- [What to start next]
```

### Verification Before Commit

Before marking any phase complete, verify:
- [ ] All notebook cells execute sequentially without error
- [ ] Output artifacts exist (model checkpoints, JSON logs, evaluation metrics)
- [ ] At least one ablation metric recorded (baseline comparison)
- [ ] progress_tracker.md updated with completion status
- [ ] Git commit made with descriptive message (feat: complete phase 1 data curation)

### Prohibited Actions

The agent must NEVER:
- ❌ Invent new features not in spec
- ❌ Skip quality gates to "save time"
- ❌ Hard-code paths, API keys, or credentials
- ❌ Leave GPU tensors in memory between cells (always .cpu() or del)
- ❌ Use !pip install without version pinning in notebooks
- ❌ Commit large binary files (> 50MB) to git (use Kaggle Datasets or cloud storage)
- ❌ Modify architecture.md or project_overview.md without explicit human approval

### Human Checkpoint Gates

The agent must pause and request explicit human approval before:
- Starting a new phase (after previous phase completion)
- Changing hyperparameters from spec defaults
- Using cloud compute beyond Kaggle free tier
- Submitting to competition leaderboard
- Publishing Kaggle notebook publicly

### Tool Usage Rules

| Tool                | Usage Rule                                                                       |
| ------------------- | -------------------------------------------------------------------------------- |
| Kaggle Notebooks    | Primary development environment; all phases must run here                        |
| Google Cloud G4 VMs | Secondary compute for long training runs; must be explicitly approved            |
| Hugging Face Hub    | Model checkpoint storage; use `push_to_hub` with private repos until submission  |
| Weights & Biases    | Optional experiment tracking; disable if it conflicts with Kaggle session limits |
| Git                 | All code versioned; commits after every completed unit                           |

### Communication Protocol

When reporting to human:
- **Status updates**: Concise bullet points, metrics first, narrative second
- **Errors**: Full traceback + last 5 lines of log + attempted fix
- **Decisions needed**: Options presented with pros/cons, recommendation flagged
- **Completion**: Checklist of verification items + link to artifact (notebook URL, checkpoint path)

## Scoping Rules

### Core Principle

Work on one phase at a time. Prefer small, verifiable increments over large speculative changes. Do not combine unrelated system boundaries (data, training, inference) in a single implementation step.

---

### Phase Boundaries (Hard Gates)

| Phase | System Boundary | Entry Criteria | Exit Criteria |
|-------|-----------------|----------------|---------------|
| **P1** | Data Curation | Kaggle account ready, base model accessible | `final_train_dataset.jsonl` generated, deduplicated, validated |
| **P2** | SFT Training | P1 exit criteria met, GPU quota available | `sft_checkpoint/` saved, validation loss converged |
| **P3** | GRPO RL | P2 exit criteria met, SFT checkpoint loads | `grpo_checkpoint/` saved, reward monotonically increased |
| **P4** | Budget Forcing + Submission | P3 exit criteria met, inference pipeline functional | `submission.zip` validated, Kaggle notebook published |

**Rule:** No phase may begin before the preceding phase passes all exit criteria. No exceptions.

---

### Feature Unit Definition

A "feature unit" in this project equals one notebook cell block with a single, verifiable purpose:

**Valid Units:**
- `load_base_model()` — load Nemotron-3-Nano-30B in 4-bit
- `run_baseline_evaluation()` — inference on public test, save results
- `extract_failure_modes()` — categorize errors from baseline
- `generate_synthetic_batch()` — create 100 problems for one failure mode
- `filter_with_llm_judge()` — score and filter one batch
- `train_lora_sft()` — one training epoch with logging
- `setup_grpo_trainer()` — initialize GRPO with config
- `run_grpo_step()` — one RL update step
- `implement_budget_forcing()` — inference with dynamic token control
- `package_submission()` — zip adapter with config validation

**Invalid Units (Too Large):**
- ❌ "Do all data work" (spans P1 entirely)
- ❌ "Train the model" (spans P2 + P3)
- ❌ "Build inference pipeline" (spans P4 + evaluation)

---

### Integration Constraints

#### NVIDIA Nemotron Integration Rules

| Component | Integration Point | Scoping Rule |
|-----------|-------------------|--------------|
| **Base Model** | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` | Load in one unit; verify with `model.config` print |
| **LoRA Config** | PEFT library, rank ≤ 32 | Configure in one unit; validate `adapter_config.json` schema |
| **Reasoning Parser** | Nemotron native `<<thinking>>` tokens | Format prompts in one helper function; test on 5 samples |
| **vLLM Engine** | Competition inference backend | Test adapter compatibility in isolated cell before submission |

#### Kaggle Platform Integration Rules

| Tool | Usage | Scoping Rule |
|------|-------|--------------|
| **Kaggle Notebooks** | Primary development environment | One notebook per phase; no cross-notebook dependencies |
| **Kaggle Datasets** | Storage for generated data, checkpoints | Save after each completed unit; version with timestamp |
| **Kaggle GPU (T4x2)** | Training compute | Monitor session timer (4-hour limit); save checkpoint every 30 minutes |
| **Kaggle Competitions** | Submission and leaderboard | Submit only from `04_budget_forcing.ipynb`; validate zip before upload |

#### Google Cloud G4 VM Integration Rules

| Resource | Purpose | Scoping Rule |
|----------|---------|--------------|
| **RTX PRO 6000 Blackwell** | Long training runs | Request explicitly before use; sync checkpoints to Kaggle Datasets after |
| **vLLM on G4** | Final inference validation | Run only after Kaggle-local validation passes |

#### Hugging Face Integration Rules

| Component | Purpose | Scoping Rule |
|-----------|---------|--------------|
| **Transformers** | Model loading, tokenization | Pin version in `requirements.txt`; test `AutoTokenizer` in isolation |
| **PEFT** | LoRA training | Test `LoraConfig` instantiation before attaching to model |
| **TRL** | GRPO implementation | Verify `GRPOTrainer` import and initialization in one cell |
| **Datasets** | Data loading, streaming | Test `load_dataset()` with sample file before full corpus |

#### External API Integration Rules

| API | Purpose | Scoping Rule |
|-----|---------|--------------|
| **DeepSeek-R1 / Qwen3** | Synthetic data generation | Batch requests (max 50 per call); save raw output before filtering |
| **OpenRouter / Together AI** | Fallback frontier model | Use only if primary API fails; log all API errors |

---

### Increment Verification Checklist

Before marking any unit complete, verify:

- [ ] **Execution:** Cell runs without error in fresh Kaggle session
- [ ] **Output:** Artifact exists (file, model, metric, or visualization)
- [ ] **Validation:** Output matches expected format (schema, shape, type)
- [ ] **Logging:** Key metrics printed or saved to `logs/`
- [ ] **Memory:** GPU memory cleared (`torch.cuda.empty_cache()`); no OOM residual
- [ ] **Reproducibility:** Same seed produces same output on re-run

---

### Anti-Patterns (Prohibited)

| Anti-Pattern | Why Forbidden | Correct Approach |
|--------------|-------------|----------------|
| **"I'll just train everything in one notebook"** | Violates phase gates; impossible to debug | One notebook per phase; save/load checkpoints between |
| **"Let me generate data and train simultaneously"** | Data quality unverified before training | Complete P1 exit criteria; inspect samples manually |
| **"I'll add budget forcing during training"** | Mixes training and inference concerns | Complete P3; implement budget forcing in separate P4 notebook |
| **"I'll use the full dataset for quick testing"** | Wastes compute on broken code | Use 100-sample subset for unit tests; scale only after validation |
| **"I'll fix the model later if data is bad"** | Training on bad data = wasted GPU hours | Validate data with LLM-as-judge before any training |
| **"I'll submit without testing on vLLM"** | Submission may fail validation | Test `submission.zip` with competition demo script before upload |

---

### Decision Escalation Matrix

| Scenario | Action | Escalation Path |
|----------|--------|-----------------|
| Unit test passes | Mark complete, update progress tracker | None |
| Unit test fails after 3 retries | Log error, pause, propose fix | Update `progress_tracker.md`, request human review |
| Phase exit criteria partially met | Do not proceed; identify blocker | Flag in progress tracker; human decides: extend phase or descope |
| New tool/library needed | Evaluate compatibility in isolation cell | Human approval before adding to `requirements.txt` |
| Compute budget exceeded | Pause all training; sync checkpoints | Human decides: request more quota or optimize |
| Leaderboard score drops vs. baseline | Stop; analyze before continuing | Human review of training logs and data quality |

---

### Session Boundaries

Each work session must:

1. **Start with:** Read `progress_tracker.md` + relevant phase spec
2. **Execute:** One feature unit only
3. **End with:** Update `progress_tracker.md` + save artifacts + git commit

**Maximum session scope:** One notebook cell block or one helper function implementation. No "end-to-end" sessions.

---

### Tool-Specific Scoping

| Tool | Scoped Usage |
|------|-------------|
| **Jupyter/Kaggle Cells** | One logical operation per cell; cells numbered sequentially |
| **Git** | Commit after every completed unit; message format: `phase/unit: description` |
| **Weights & Biases** | Optional; if used, one run per phase only |
| **MLflow** | Track only final phase metrics; disable during development to save session time |
| **TensorBoard** | Not used; prefer inline matplotlib in notebooks |
| **Docker** | Out of scope; Kaggle environment is fixed |

## When to Split Work

Split an implementation step if it combines:

### Data & Training Concerns

- **Data generation logic and training code in the same notebook cell**
  - Example: A cell that both generates synthetic problems AND starts LoRA training
  - Why split: Data quality must be verified before any GPU training begins; mixing them makes debugging impossible
  
- **Multiple failure modes in one generation batch**
  - Example: Generating problems for "algebraic manipulation" AND "geometric proofs" in the same API call
  - Why split: Each failure mode requires different prompt engineering; batching them reduces quality control

- **Raw generation and LLM-as-judge filtering in the same step**
  - Example: Calling DeepSeek-R1 and immediately scoring output without saving raw data
  - Why split: If filtering logic has bugs, raw data is lost; always preserve intermediate artifacts

### Model Architecture Concerns

- **LoRA configuration and model loading in the same cell**
  - Example: `LoraConfig()` + `get_peft_model()` + `AutoModelForCausalLM.from_pretrained()` in one block
  - Why split: Model loading (30B parameters) takes 5-10 minutes; config errors should be caught before GPU memory is allocated

- **SFT training and checkpoint saving in the same loop**
  - Example: Training for 3 epochs without intermediate saves
  - Why split: Kaggle sessions die after 4 hours; checkpoints must be saved every 30 minutes regardless of training completion

- **GRPO group generation and policy update in the same function**
  - Example: Generating G=8 responses AND computing advantages AND updating weights in one call
  - Why split: Group generation is memory-intensive; policy update requires gradient computation; separating them enables OOM recovery

### Inference & Evaluation Concerns

- **Budget forcing logic and accuracy evaluation in the same script**
  - Example: Implementing "Wait" token injection AND computing final leaderboard score together
  - Why split: Budget forcing must be validated on a small subset before full evaluation; errors in token manipulation corrupt all metrics

- **vLLM compatibility testing and competition submission packaging**
  - Example: First time testing adapter with vLLM AND creating submission.zip simultaneously
  - Why split: vLLM errors require debugging the adapter config; packaging should only happen after inference is verified

### Cross-Phase Concerns

- **Any work spanning two phases**
  - Example: "I'll generate data and start SFT in the same session because I have GPU time left"
  - Why split: Phase gates exist to enforce quality; skipping them guarantees technical debt

- **Checkpoint conversion between formats**
  - Example: Converting Hugging Face LoRA to vLLM-compatible format AND testing inference
  - Why split: Format errors are common; conversion must be verified with `adapter_config.json` inspection before inference

### NVIDIA-Specific Concerns

- **Nemotron reasoning parser tokens and standard prompt formatting in the same template**
  - Example: Mixing `<<thinking>>` tokens with raw text without explicit separation
  - Why split: Parser token placement is critical; incorrect formatting breaks vLLM reasoning parser plugin

- **QLoRA 4-bit loading and gradient checkpointing in the same configuration**
  - Example: `bitsandbytes` config + `torch.utils.checkpoint` in one dictionary
  - Why split: These are independent memory optimization strategies; enabling both simultaneously can cause silent OOM on RTX PRO 6000 Blackwell

- **NVIDIA NIM API calls and fallback mock implementations in the same function**
  - Example: `if api_key: call_nim() else: mock_response()` without explicit environment detection
  - Why split: Mock logic must be isolated in `utils/mock_nim.py`; production code should never contain fallback paths

### Verification Concerns

- **Behavior not clearly defined in the context files**
  - Example: "The spec says 'generate synthetic data' but doesn't specify batch size or API rate limits"
  - Why split: Ambiguous requirements must be clarified in `progress_tracker.md` before implementation; guessing leads to rework

- **Changes that cannot be verified end-to-end within 10 minutes**
  - Example: "Train for 500 steps and check accuracy" — takes 2+ hours
  - Why split: Break into "train 50 steps, verify loss decreases" + "continue to 500 if valid"

- **Metrics that require manual inspection to validate**
  - Example: "Check if generated problems are high quality" — requires human reading
  - Why split: Implement automated LLM-as-judge scoring first; manual inspection is a separate review step

### Kaggle Platform Concerns

- **Internet-dependent API calls and offline computation in the same cell**
  - Example: Calling DeepSeek-R1 API AND computing embeddings while internet is required for both
  - Why split: Kaggle internet can be intermittent; download API responses first, process offline second

- **Large file I/O (>1GB) and GPU computation in the same cell**
  - Example: Loading 30B model AND writing dataset to disk simultaneously
  - Why split: I/O blocks GPU utilization; separate them for optimal RTX PRO 6000 Blackwell throughput

- **Kaggle Dataset upload and notebook execution in the same session**
  - Example: Saving checkpoint to `/kaggle/working/` AND calling `kaggle datasets create` immediately
  - Why split: Uploads are slow and can timeout; save locally first, upload in dedicated session

---

## Splitting Heuristics

If any of the following is true, the scope is too broad — split it:

| Indicator | Action |
|-----------|--------|
| Cell execution time > 10 minutes without intermediate output | Break into sub-cells with progress logging |
| Function has > 50 lines or > 3 levels of indentation | Extract helper functions |
| Variable names become ambiguous (e.g., `data`, `result`, `output`) | Split into named steps with typed returns |
| Error traceback spans > 5 files | Isolate the failing component |
| You need to scroll to see the full cell | It's too long — split it |
| "And then" appears in your description more than once | Multiple steps disguised as one |

---

## NVIDIA Best Practice: Kernel Fusion Principle

On NVIDIA GPUs (especially Blackwell), **kernel fusion** improves performance by combining small operations into single CUDA kernels. Apply the same principle to work units:

**Fuse when:** Operations share the same data and can be verified together (e.g., load model + verify config)

**Split when:** Operations have different failure modes or verification requirements (e.g., train model + evaluate accuracy)

---

## Verification Checklist Before Splitting

Before deciding to split, confirm:

- [ ] Can I verify this unit completes successfully in < 10 minutes?
- [ ] If it fails, can I identify which sub-component caused the failure in < 2 minutes?
- [ ] Does this unit modify only one system boundary (data, model, inference)?
- [ ] Can I checkpoint/save intermediate state and resume without re-running everything?
- [ ] Does this unit require only one type of resource (CPU, GPU, network, disk)?

If any answer is "No" — **split the work**.

---

## Handling Missing Requirements

### Core Principle

Do not invent model behavior, training procedures, or inference logic not explicitly defined in the context files. The context files (`project_overview.md`, `architecture.md`, `code_standards.md`, `ai_workflow_rules.md`, `progress_tracker.md`) are the single source of truth. If a requirement is ambiguous or missing, it must be resolved before any implementation begins.

---

### Requirement Classification Matrix

| State | Definition | Action |
|-------|-----------|--------|
| **Explicit** | Clearly defined in context files with parameters, constraints, and expected output | Implement as specified |
| **Ambiguous** | Mentioned in context files but with unclear parameters, thresholds, or edge cases | Flag in `progress_tracker.md`, resolve before implementation |
| **Missing** | Not mentioned in any context file but implied by project goals | Add as open question in `progress_tracker.md`, request human clarification |
| **Contradictory** | Defined differently in multiple context files | Flag as invariant violation, escalate to human immediately |

**Invariant Escalation Matrix**

| Invariant                            | Violation Example                                   | Escalation Path                                            |
| ------------------------------------ | --------------------------------------------------- | ---------------------------------------------------------- |
| "LoRA rank ≤ 32"                     | Spec suggests rank 64 for better accuracy           | Flag in `progress_tracker.md` + stop work + human decision |
| "All processing on Kaggle or G4 VMs" | Spec suggests using personal GPU                    | Flag as scope creep + human decision                       |
| "No test set leakage"                | Data generation spec suggests using full benchmark  | Flag as competition rules violation + immediate stop       |
| "Temperature = 0.0 for inference"    | Budget forcing spec suggests temperature scheduling | Flag as competition parameter conflict + human decision    |

---

### Ambiguous Requirements — Resolution Protocol

**Step 1: Identify Ambiguity**

Common ambiguity patterns in this project:

| Context File | Ambiguous Phrase | What's Missing |
|--------------|-----------------|----------------|
| `project_overview.md` | "generate 10-50k synthetic problems" | Exact target count? Budget constraint on API calls? |
| `architecture.md` | "QLoRA 4-bit quantization" | Specific bitsandbytes config (NF4 vs FP4)? |
| `code_standards.md` | "thinking traces formatted using Nemotron's reasoning parser tokens" | Exact token sequence? Placement rules? |
| `ai_workflow_rules.md` | "save checkpoint every 30 minutes" | On Kaggle session timeout? On epoch boundary? Both? |
| `success_criteria.md` | "accuracy ≥ 85%" | On full test set or validation subset? |

**Typical Gaps Matrix**

| Area              | Typical Gap                                                               | Why Critical                                   |
| ----------------- | ------------------------------------------------------------------------- | ---------------------------------------------- |
| **Data Pipeline** | "How to handle API rate limits from DeepSeek-R1?"                         | Blocks synthetic data generation if exceeded   |
| **Training**      | "What learning rate scheduler for GRPO?"                                  | Default may diverge or converge too slowly     |
| **Inference**     | "How to detect 'easy' vs 'hard' problems for budget forcing?"             | Core feature cannot function without heuristic |
| **Evaluation**    | "What is the exact tolerance for numerical answers?"                      | Competition metric depends on this             |
| **Submission**    | "How to handle tokenizer mismatches between training and vLLM inference?" | Submission may fail validation                 |

**Technical Gaps Matrix**

| Component                           | Missing Detail                                          | Why Critical                                                                    |
| ----------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Nemotron Reasoning Parser**       | Exact token IDs for `<<thinking>>` and `</thinking>`    | vLLM parser plugin requires specific token sequences; wrong IDs break inference |
| **QLoRA on RTX PRO 6000 Blackwell** | Optimal `bnb_4bit_compute_dtype` (float16 vs bfloat16)  | Blackwell has native bfloat16 support; float16 may underutilize hardware        |
| **vLLM Integration**                | Whether to use `vllm.LLM` or `vllm.entrypoints.llm` API | Different APIs have different LoRA loading paths                                |
| **GRPO on 30B Model**               | Whether gradient checkpointing is compatible with QLoRA | Memory constraints may force trade-offs between checkpointing and batch size    |

**Platform Gaps Matrix**

| Platform Constraint    | Missing Detail                                     | Why Critical                                                         |
| ---------------------- | -------------------------------------------------- | -------------------------------------------------------------------- |
| **Session Timeout**    | Whether to save on wall-clock time or step count   | 4-hour limit is hard; wrong checkpoint strategy loses hours of work  |
| **Internet Access**    | Whether DeepSeek-R1 API calls work in offline mode | Kaggle internet is intermittent; may need to cache all API responses |
| **Dataset Versioning** | Whether to version by timestamp or by git commit   | Kaggle Datasets overwrite; wrong strategy loses reproducibility      |
| **GPU Memory**         | Whether T4x2 means 2x16GB or shared 16GB           | Affects batch size calculations and OOM prevention                   |

**Step 2: Document in `progress_tracker.md`**

*For Ambiguous Requirements:*

```markdown
## Open Question [ID] — [Date]

**Location:** [Context file + section]
**Ambiguity:** [Exact phrase that is unclear]
**Impact:** [What cannot proceed without resolution]
**Options Considered:**
- Option A: [Description + pros/cons]
- Option B: [Description + pros/cons]
**Recommendation:** [Suggested resolution]
```

*For Missing Requirements:*

```markdown
## Open Question [ID] — [Date]

**Category:** Missing Requirement
**Area:** [Data / Training / Inference / Evaluation / Submission]
**Question:** [What needs to be defined]
**Blocking:** [Which phase/unit cannot proceed]
**Proposed Default:** [If you were forced to guess — document it as guess, not decision]
**Status:** ⏳ Awaiting human input
```

**Step 3: Pause Implementation**

- Do not proceed with any work dependent on the ambiguous requirement
- Switch to independent work units if available
- If no independent work exists, pause entirely until resolved

Report the blocker to the human using the following template:

```markdown
**What I cannot do:** [Specific implementation step]
**Why:** [Missing/ambiguous requirement]
**What I need:** [Exact decision or information]
**Options (if known):** [A, B, C with brief pros/cons]
**Impact if delayed:** [What else is blocked]
**Suggested default (if any):** [With explicit "this is a guess" warning]
```

**Step 4: Human Resolution**

- Human provides explicit decision
- Update context file with clarified requirement
- Close open question in `progress_tracker.md` with resolution and date

Format for documenting resolution in `progress_tracker.md`:

*For Ambiguous Requirements:*

```markdown
## Open Question [ID] — RESOLVED [Date]

**Original Ambiguity:** [Quote from progress tracker]
**Resolution:** [Exact decision made]
**Updated Context File:** [File + section modified]
**Implementation Impact:** [What can now proceed]
**Verified By:** [Human name or self-verification]
```

*For Missing Requirements:*

```markdown
## Open Question [ID] — RESOLVED [Date]

**Original Question:** [Quote from progress tracker]
**Decision:** [Exact requirement added]
**Source:** [Human input / NVIDIA documentation / competition rules / empirical test]
**Added To:** [Context file + section]
**Implementation Status:** Ready to proceed
```

*For Deferred Questions:*

```markdown
## Open Question [ID] — DEFERRED [Date]

**Original Question:** [Quote from progress tracker]
**Reason for Deferral:** [Why not critical now]
**Deferred To:** [Phase or date]
**Workaround:** [What to do in meantime]
**Revisit Trigger:** [What event will reopen this question]
```

**Requirement Verification Checklist**

Before marking any requirement as "resolved," verify:

- [ ] Resolution is documented in the relevant context file (not just progress tracker)
- [ ] Resolution does not contradict any other context file
- [ ] Resolution is specific enough to implement (has numbers, thresholds, or explicit choices)
- [ ] Resolution is reproducible (another developer could read it and implement identically)
- [ ] Progress tracker updated with resolution and date

### Requirement Anti-Patterns

| Temptation                                          | Why Forbidden                                   | Correct Action                                       |
| --------------------------------------------------- | ----------------------------------------------- | ---------------------------------------------------- |
| "I'll use learning rate 2e-4, that's standard"      | May be wrong for Nemotron-3-Nano-30B at rank 32 | Check NVIDIA recipes; if not specified, ask          |
| "I'll assume numerical tolerance is 1e-4"           | Competition metric may use different threshold  | Read competition metric implementation               |
| "I'll use 'problem length' as difficulty heuristic" | May not correlate with actual difficulty        | Test on validation set; document if unverified       |
| "I'll generate 50k problems to be safe"             | API costs and time may exceed budget            | Clarify target with human considering compute limits |

**Prohibited Shortcuts**

The agent must NEVER:

- ❌ "I'll just try both and see what works" — wastes compute, violates scoping rules
- ❌ "The default should be fine" — defaults are not requirements
- ❌ "I'll copy what another Kaggle notebook did" — may not match this project's constraints
- ❌ "I'll assume it's the same as [other competition]" — each competition has unique rules
- ❌ "I'll implement a placeholder and fix it later" — technical debt in 25-day timeline is fatal

## Protected Files

Do not modify the following unless explicitly instructed by human. These files are either external dependencies, competition-mandated formats, or validated artifacts that must remain stable.

---

### Competition-Mandated Files (Immutable)

| File | Source | Why Protected |
|------|--------|---------------|
| `adapter_config.json` | Generated by PEFT after LoRA training | Competition submission validator checks schema; manual edits break compatibility |
| `submission.zip` | Packaged by `package_submission()` | Must match competition demo script output exactly; manual repackaging risks validation failure |
| `competition_metric.py` | NVIDIA official implementation | Defines exact grading logic; any modification invalidates local evaluation |

### External Dependencies (Read-Only)

| File/Package | Version | Why Protected |
|-------------|---------|---------------|
| `nvidia/Nemotron-3-Nano-30B-A3B-Base-BF16` | Latest from Hugging Face | Base model weights; modifying them violates competition rules (LoRA only) |
| `vllm` | `>=0.5.0` | Competition inference engine; version mismatch breaks reasoning parser plugin |
| `transformers` | `>=4.40.0` | Required for Nemotron tokenizer and model loading |
| `peft` | `>=0.11.0` | LoRA implementation; manual edits to internal adapters risk corruption |
| `trl` | `>=0.9.0` | GRPO trainer; internal loss computation must not be modified |
| `bitsandbytes` | `>=0.43.0` | 4-bit quantization kernels; CUDA compatibility depends on exact version |

### Generated Artifacts (Do Not Hand-Edit)

| Artifact | Generated By | Why Protected |
|----------|--------------|---------------|
| `synthetic_data/raw_*.jsonl` | `generate_synthetic_batch()` | API responses from DeepSeek-R1/Qwen3; hand-editing loses reproducibility |
| `checkpoints/sft_*/` | `train_lora_sft()` | Training state; manual modification corrupts optimizer states |
| `checkpoints/grpo_*/` | `run_grpo_step()` | RL policy checkpoint; manual edits break reward baseline |
| `logs/training_*.json` | Training callbacks | Structured logs for ablation studies; hand-editing invalidates metrics |
| `evaluation/predictions_*.json` | Inference pipeline | Competition submission predictions; must match exact format |

### Configuration Templates (Read-Only Baselines)

| File | Purpose | Modification Rule |
|------|---------|-----------------|
| `configs/base_lora.json` | Default LoRA config from NVIDIA recipes | Copy to `configs/custom_lora.json` before editing; never modify base |
| `configs/base_grpo.json` | Default GRPO config from TRL | Copy to `configs/custom_grpo.json` before editing; never modify base |
| `configs/competition_params.json` | Official competition inference parameters | **ABSOLUTELY IMMUTABLE** — temperature=0.0, max_tokens=7680, etc. |

### NVIDIA-Specific Protected Components

| Component | Why Protected |
|-----------|---------------|
| Nemotron reasoning parser token IDs (`<<thinking>>`, `</thinking>`) | Hardcoded in vLLM plugin; wrong IDs break inference pipeline |
| `nvidia-llama-nemotron-endpoint` SageMaker endpoint name | Used in dual-hackathon architecture; renaming breaks deployment scripts |
| `ml.g5.xlarge` instance type for NIM deployment | Cost and compatibility validated; changing risks budget overrun or CUDA errors |

### Kaggle Platform Files (Environment-Managed)

| File/Path | Why Protected |
|-----------|---------------|
| `/kaggle/input/` | Read-only input datasets; modifications lost on session restart |
| `/kaggle/working/` session artifacts | Managed by Kaggle; manual deletion risks losing unsubmitted work |
| `kaggle.json` credentials | Authentication file; corruption blocks dataset downloads and submissions |

### Version-Controlled Snapshots (Git-Protected)

| Commit Tag | Protected Until |
|-----------|-----------------|
| `baseline-eval` | End of Phase 1 (provides failure analysis foundation) |
| `sft-complete` | End of Phase 3 (provides rollback point if GRPO diverges) |
| `grpo-stable` | End of Phase 4 (provides checkpoint for budget forcing experiments) |

---

### Modification Exception Process

If a protected file **must** be modified:

1. **Flag in `progress_tracker.md`:**
   ```markdown
   ## Protected File Modification Request [ID]

   **File:** [Path]
   **Current Status:** Protected
   **Proposed Change:** [Exact modification]
   **Justification:** [Why necessary]
   **Risk:** [What breaks if done wrong]
   **Rollback Plan:** [How to revert]
   ```

2. **Await Human Approval:** Pause implementation of this change until human signs off on the request.

   **Human Approval Rules:**
   - No exceptions for competition-mandated files
   - Limited exceptions for generated artifacts (e.g., fixing corrupted checkpoint)

3. **Create Backup Before Modification:**
   ```bash
   cp adapter_config.json adapter_config.json.backup.[timestamp]
   ```

4. **Verify After Modification:**
   - **For `adapter_config.json`:** Run competition validation script
   - **For checkpoints:** Load and verify inference produces identical output
   - **For configs:** Diff against baseline and document changes

### Protected File Violation Matrix

| Action                                      | Why Forbidden                      | Consequence                  |
| ------------------------------------------- | ---------------------------------- | ---------------------------- |
| Editing `adapter_config.json` manually      | Schema mismatch with vLLM          | Submission rejected          |
| Modifying base model weights directly       | Violates competition rules         | Disqualification             |
| Changing competition inference parameters   | Invalidates leaderboard comparison | Submission rejected          |
| Hand-editing training logs                  | Invalidates ablation studies       | Prize eligibility revoked    |
| Deleting raw API responses before filtering | Loses reproducibility evidence     | Documentation incomplete     |
| Modifying `competition_metric.py`           | Grading logic must match official  | Local evaluation meaningless |

### Safe vs. Unsafe Modification Approaches

| Need                                 | Safe Approach                                              | Unsafe Approach                 |
| ------------------------------------ | ---------------------------------------------------------- | ------------------------------- |
| Custom LoRA config                   | Copy `configs/base_lora.json` → `configs/custom_lora.json` | Edit base directly              |
| Experiment with GRPO hyperparameters | Create `configs/grpo_experiment_*.json`                    | Modify `configs/base_grpo.json` |
| Fix corrupted checkpoint             | Load checkpoint, verify, save to new path                  | Edit binary weights directly    |
| Adjust budget forcing thresholds     | Parameterize in `inference_config.yaml`                    | Hardcode in inference script    |

**Verification Command**

Before any commit, run:

```bash
python scripts/verify_protected_files.py
```

Expected output:

```plain
✓ adapter_config.json — unmodified since training
✓ competition_params.json — matches official
✓ base model weights — read-only access confirmed
✓ All checkpoints — integrity hashes valid
✓ No hand-edited logs detected
```

## Keeping Docs in Sync

Update the relevant context file whenever implementation changes. The context files are the single source of truth for the project; stale documentation causes agent drift, human confusion, and reproducibility failures.

---

### Sync Triggers

Update context files immediately when any of the following changes:

#### System Architecture or Boundaries

| Change Type | File to Update | What to Document |
|-------------|---------------|----------------|
| New model component added (e.g., explicit PRM model) | `architecture.md` | Component role, inputs, outputs, dependencies |
| Compute backend changed (Kaggle → G4 VM or vice versa) | `architecture.md` | Instance type, GPU specs, cost implications |
| Inference engine modified (vLLM version upgrade) | `architecture.md` | Version, compatibility notes, parser plugin changes |
| Data storage path changed | `architecture.md` | New path, access permissions, backup strategy |
| API integration added (new frontier model provider) | `architecture.md` | Endpoint, rate limits, fallback behavior |

#### Storage Model Decisions

| Change Type | File to Update | What to Document |
|-------------|---------------|----------------|
| Dataset format changed (JSON → JSONL → Parquet) | `architecture.md` | Format rationale, schema, conversion script |
| Checkpoint save frequency changed | `architecture.md` | New interval, trigger conditions, retention policy |
| New artifact type introduced (e.g., PRM scores log) | `architecture.md` | File format, size estimate, lifecycle |
| Data versioning strategy changed | `code_standards.md` | Naming convention, metadata, reproducibility proof |

#### Code Conventions or Standards

| Change Type | File to Update | What to Document |
|-------------|---------------|----------------|
| New Python package introduced | `code_standards.md` | Version, purpose, where used, pin in `requirements.txt` |
| Type hint convention changed (e.g., `Optional` → `\|`) | `code_standards.md` | New style, migration plan, examples |
| Error handling pattern changed | `code_standards.md` | New pattern, before/after examples, rationale |
| Notebook structure modified | `code_standards.md` | New cell order, mandatory cells, output format |
| Logging format changed | `code_standards.md` | New format, fields, example output |

#### Feature Scope

| Change Type | File to Update | What to Document |
|-------------|---------------|----------------|
| New technique added (e.g., DAPO instead of GRPO) | `project_overview.md`, `scope.md` | Rationale, research basis, impact on timeline |
| Feature descoped (e.g., explicit PRM → implicit PRM) | `scope.md`, `features.md` | Reason, workaround, impact on success criteria |
| New dataset source added | `features.md`, `architecture.md` | Source, license, preprocessing, mixing ratio |
| Evaluation metric modified | `success_criteria.md` | New threshold, measurement method, verification |

---

### Sync Workflow

#### Step 1: Implement Change
- Write code
- Test in isolation
- Verify output matches expectation

#### Step 2: Identify Affected Context Files
Use this decision tree:

```
Did I add/modify a model component?
  → YES → Update `architecture.md`

Did I change how data is stored, loaded, or versioned?
  → YES → Update `architecture.md` + `code_standards.md`

Did I introduce a new package, pattern, or convention?
  → YES → Update `code_standards.md`

Did I add, remove, or modify a feature?
  → YES → Update `features.md` + `scope.md` (if scope changed)

Did I change success criteria or evaluation method?
  → YES → Update `success_criteria.md`

Did I learn something that invalidates previous assumptions?
  → YES → Update `project_overview.md` (assumptions section)
```

#### Step 3: Update Context Files

**Template for context file update:**

```markdown
## Update Log — [Date] — [Author]

**Trigger:** [What implementation change caused this update]
**Files Modified:** [List of context files updated]
**Summary:** [One-line description of change]

### `architecture.md` Changes
- **Section:** [Which section]
- **Before:** [Old text or reference]
- **After:** [New text]
- **Rationale:** [Why this changed]

### `code_standards.md` Changes
- **Section:** [Which section]
- **Before:** [Old convention]
- **After:** [New convention]
- **Migration:** [How existing code should adapt]

### Impact Assessment
- **Phases Affected:** [Which phases need re-verification]
- **Backwards Compatibility:** [Yes / No — if No, migration plan]
- **Reproducibility Risk:** [Low / Medium / High]
- **Action Required:** [What humans/agents must do]
```

#### Step 4: Update `progress_tracker.md`

```markdown
## Context Sync — [Date]

- [x] `architecture.md` updated for [change]
- [x] `code_standards.md` updated for [change]
- [x] `features.md` updated for [change]
- [ ] `success_criteria.md` — N/A
- [ ] `project_overview.md` — N/A

**Verification:** Context files read by agent in fresh session; no confusion detected.
```

#### Step 5: Commit with Descriptive Message

```bash
git commit -m "docs: sync architecture.md for QLoRA config change

- Updated quantization section with NF4 vs FP4 decision
- Added RTX PRO 6000 Blackwell compute dtype note
- Verified agent reads updated context correctly

Refs: progress_tracker.md #2026-05-22"
```

---

### Sync Responsibility Matrix

| Role | Responsibility |
|------|---------------|
| **Human (You)** | Approve architecture changes, resolve scope conflicts, validate success criteria updates |
| **Agent (Me)** | Update technical details (code standards, architecture specifics), flag inconsistencies, propose syncs |
| **Both** | Verify context files are read and understood before next session |

---

### Anti-Patterns (Forbidden)

| Anti-Pattern | Why Forbidden | Correct Approach |
|-------------|-------------|----------------|
| "I'll update docs later when I have time" | Later never comes; agent uses stale context | Update immediately after implementation |
| "The code is self-documenting" | Code shows what, context shows why | Both are necessary; context explains rationale |
| "I'll just tell the agent in chat" | Chat history is lost; context files persist | All decisions in context files |
| "Minor change, no need to document" | Minor changes compound into major drift | If it affects agent behavior, document it |
| "Copy-paste old context for new phase" | Phases have different constraints | Review and adapt context per phase |

---

### Verification: Context Freshness Check

Before every new session, the agent must verify:

```python
def verify_context_freshness():
    """Check if context files match current implementation state."""
    
    checks = {
        "architecture.md": {
            "last_modified": get_git_timestamp("architecture.md"),
            "last_commit_message": get_git_message("architecture.md"),
            "references_valid": check_all_paths_exist(),
        },
        "code_standards.md": {
            "last_modified": get_git_timestamp("code_standards.md"),
            "packages_match_requirements": compare_with_requirements_txt(),
        },
        "progress_tracker.md": {
            "last_entry_date": get_last_entry_date(),
            "no_stale_entries": check_entries_within_24h(),
        }
    }
    
    for file, status in checks.items():
        if status["last_modified"] < get_last_implementation_change():
            raise ContextStaleError(f"{file} may be stale; review before proceeding")
    
    return "All context files verified fresh"
```

---

### Example Sync Scenarios

#### Scenario 1: QLoRA Config Change

**Implementation:** Switched from FP4 to NF4 quantization for better accuracy.

**Sync Actions:**
1. Update `architecture.md` — quantization section with NF4 rationale
2. Update `code_standards.md` — default `bnb_4bit_quant_type` convention
3. Update `progress_tracker.md` — context sync log
4. Commit: `docs: sync NF4 quantization decision`

#### Scenario 2: Budget Forcing Heuristic Change

**Implementation:** Replaced "problem length" with "token count of first reasoning step" as difficulty proxy.

**Sync Actions:**
1. Update `features.md` — budget forcing feature description
2. Update `architecture.md` — inference pipeline flow
3. Update `success_criteria.md` — verification method for hard/easy classification
4. Commit: `docs: sync budget forcing heuristic`

#### Scenario 3: New Frontier Model Added

**Implementation:** Added Qwen3-235B as backup to DeepSeek-R1 for synthetic generation.

**Sync Actions:**
1. Update `architecture.md` — API integrations section
2. Update `code_standards.md` — API call pattern, rate limit handling
3. Update `features.md` — data generation feature description
4. Commit: `docs: sync Qwen3 integration`

---

### Emergency Sync Protocol

If context files are discovered to be severely stale mid-session:

1. **Stop all implementation work immediately**
2. **Flag in `progress_tracker.md`:**
   ```markdown
   ## EMERGENCY — Context Stale Detected

   **File:** [Which context file]
   **Stale Since:** [Date]
   **Impact:** [What could go wrong if we proceed]
   **Action:** Pause implementation; sync context first
   ```
3. **Human decides:** Sync now, or rollback implementation to last known-good state
4. **No work resumes until context is verified fresh**

---

### Sync Checklist (Before Every Commit)

- [ ] Implementation change is complete and tested
- [ ] Affected context files identified using decision tree
- [ ] Context files updated with before/after documentation
- [ ] `progress_tracker.md` updated with sync log
- [ ] Git commit includes `docs:` prefix and references implementation change
- [ ] Agent reads updated context in fresh session and confirms understanding


## Before Moving to the Next Unit

### Unit Completion Gates

Before proceeding to the next feature unit, phase, or notebook, verify ALL of the following:

---

### 1. The Current Unit Works End-to-End Within Its Defined Scope

| Verification | Method | Evidence |
|-------------|--------|----------|
| Unit executes without error | Run cell(s) in fresh Kaggle session | Screenshot or log of successful execution |
| Output artifact exists | Check file system or variable state | `ls -la` output or object inspection |
| Output matches expected format | Schema validation or type check | Assert statements or manual inspection |
| No side effects outside scope | Review imports and file writes | No modifications to protected files |

**Specific to this project:**

- **Data Generation Unit:** `raw_synthetic_batch.jsonl` created with ≥100 problems; each problem has `question`, `thinking_trace`, `answer`, `failure_mode_tag`
- **SFT Training Unit:** `sft_checkpoint/` directory created; `trainer_state.json` shows decreasing loss; sample generation produces structured reasoning
- **GRPO Training Unit:** `grpo_checkpoint/` directory created; `rewards_log.json` shows monotonic increase; KL divergence < 0.05
- **Budget Forcing Unit:** Inference script runs on 10 sample problems; hard problems show extended reasoning; easy problems show early termination

---

### 2. No Invariant Defined in `architecture.md` Was Violated

| Invariant | Verification Method | Pass/Fail |
|-----------|---------------------|-----------|
| LoRA rank ≤ 32 | `adapter_config.json` inspection: `"r": 32` | ⬜ |
| Base model unmodified | `git diff` on model weights directory (should be empty) | ⬜ |
| Temperature = 0.0 for inference | Config file check: `"temperature": 0.0` | ⬜ |
| All processing on Kaggle or G4 VMs | No local paths, no personal GPU references | ⬜ |
| No test set leakage | MinHash deduplication report: 0 overlap | ⬜ |
| 75/25 reasoning/non-reasoning mix | Dataset statistics: `reasoning_pct >= 0.75` | ⬜ |
| Nemotron reasoning parser tokens preserved | Sample output contains `<<thinking>>` and `</thinking>>` | ⬜ |
| GPU memory cleared between sessions | `torch.cuda.empty_cache()` called; `nvidia-smi` shows no residual | ⬜ |

**If ANY invariant is violated:**
- STOP immediately
- Document violation in `progress_tracker.md`
- Do NOT proceed until resolved and re-verified

---

### 3. `progress_tracker.md` Reflects the Completed Work

| Required Entry | Content |
|---------------|---------|
| **Unit ID** | `[Phase]-[Unit]-[Date]` e.g., `P1-DataGen-2026-05-22` |
| **Status** | `✅ COMPLETED` |
| **Scope Summary** | One-line description of what was implemented |
| **Verification Evidence** | Links to artifacts, metrics, or screenshots |
| **Decisions Made** | Any choices made during implementation with rationale |
| **Blockers Resolved** | Any issues encountered and how they were fixed |
| **Next Unit** | Clear identification of what comes next |

**Template:**

```markdown
## Unit: P1-DataGen-2026-05-22 — ✅ COMPLETED

**Scope:** Generated 5,000 synthetic problems targeting "algebraic manipulation" failure mode
**Verification:**
- [x] `raw_synthetic_batch_001.jsonl` created (5,012 problems)
- [x] Sample inspection: 10 problems reviewed, all have thinking traces
- [x] API cost: $12.40 (within budget)
**Decisions:**
- Used Qwen3-235B instead of DeepSeek-R1 (rate limit exceeded)
- Batch size: 50 problems per API call (optimal for cost/speed)
**Blockers:**
- Initial prompt produced too easy problems → Added difficulty constraint
**Next Unit:** P1-DataGen-2026-05-23 (geometric proofs failure mode)
```

---

### 4. Notebook Execution Passes (Kaggle-Specific Build Check)

Since this project uses **Kaggle Notebooks** (not npm/Node.js), replace `npm run build` with:

| Phase | Build Check | Command/Method | Expected Result |
|-------|-------------|----------------|-----------------|
| **P1: Data** | Notebook runs end-to-end | "Run All" in fresh Kaggle session | All cells execute; `final_train_dataset.jsonl` created |
| **P2: SFT** | Training completes without OOM | Monitor GPU memory via `nvidia-smi` | Loss decreases; checkpoint saved |
| **P3: GRPO** | RL training stable | Monitor reward and KL divergence | Reward increases; KL < 0.05 |
| **P4: Inference** | Budget forcing functional | Test on 100-problem subset | Accuracy improvement on hard subset |

**Generic Verification Script (run before every unit transition):**

```python
# scripts/verify_unit_completion.py

import json
import os
import sys
from pathlib import Path

def verify_unit_completion(phase: str, unit: str) -> dict:
    """Verify all completion gates before proceeding."""
    
    results = {
        "phase": phase,
        "unit": unit,
        "checks": {},
        "can_proceed": True
    }
    
    # Check 1: Artifacts exist
    artifact_paths = {
        "P1": ["synthetic_data/raw_*.jsonl", "synthetic_data/filtered_*.jsonl"],
        "P2": ["checkpoints/sft_*/", "logs/sft_training.json"],
        "P3": ["checkpoints/grpo_*/", "logs/grpo_rewards.json"],
        "P4": ["submission.zip", "evaluation/predictions.json"]
    }
    
    for pattern in artifact_paths.get(phase, []):
        matches = list(Path(".").glob(pattern))
        results["checks"][f"artifact_{pattern}"] = len(matches) > 0
    
    # Check 2: Invariants preserved
    if phase in ["P2", "P3", "P4"]:
        # Verify LoRA rank
        config_path = Path("checkpoints") / f"{phase.lower()}_latest" / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            results["checks"]["lora_rank_≤32"] = config.get("r", 999) <= 32
    
    # Check 3: Progress tracker updated
    tracker_path = Path("context/progress_tracker.md")
    if tracker_path.exists():
        content = tracker_path.read_text()
        results["checks"]["progress_tracker_updated"] = unit in content
    
    # Check 4: No OOM in logs
    log_path = Path(f"logs/{phase.lower()}_*.json")
    if log_path.exists():
        # Parse logs for OOM errors
        results["checks"]["no_oom"] = "OutOfMemoryError" not in log_path.read_text()
    
    # Final decision
    results["can_proceed"] = all(results["checks"].values())
    
    return results

if __name__ == "__main__":
    phase = sys.argv[1]  # P1, P2, P3, P4
    unit = sys.argv[2]   # Unit identifier
    
    results = verify_unit_completion(phase, unit)
    
    print(f"\n{'='*60}")
    print(f"Unit Completion Verification: {phase} — {unit}")
    print(f"{'='*60}\n")
    
    for check, passed in results["checks"].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {check}")
    
    print(f"\n{'='*60}")
    if results["can_proceed"]:
        print("✅ ALL CHECKS PASSED — Proceed to next unit")
        sys.exit(0)
    else:
        print("❌ CHECKS FAILED — Do not proceed. Fix issues first.")
        sys.exit(1)
```

**Usage:**
```bash
python scripts/verify_unit_completion.py P1 DataGen-2026-05-22
```

---

### Phase Transition Gates (Hard Gates)

Before moving between phases (P1 → P2 → P3 → P4), additional verification required:

| Transition | Additional Check | Evidence |
|-----------|-----------------|----------|
| **P1 → P2** | Dataset quality verified by LLM-as-judge; ≥10k problems retained | `filtering_report.json` with score distribution |
| **P2 → P3** | SFT checkpoint loads and generates coherent reasoning traces | Sample outputs from 10 problems |
| **P3 → P4** | GRPO checkpoint improves over SFT on validation set | `ablation_sft_vs_grpo.json` with accuracy comparison |
| **P4 → Submission** | Budget forcing improves hard subset without degrading easy subset | `stratified_evaluation.json` with per-bin accuracy |

---

### Emergency Stop Conditions

Do NOT proceed to next unit if:

- ❌ `nvidia-smi` shows GPU memory not freed after unit completion
- ❌ Any cell in notebook shows `WARNING` or `ERROR` in output
- ❌ Artifact file size is unexpectedly small (corruption indicator)
- ❌ Loss curve shows NaN, Inf, or sudden spike
- ❌ Progress tracker not updated for > 24 hours
- ❌ Git working directory has uncommitted changes

---

### Sign-Off Template

```markdown
## Unit Completion Sign-Off

**Unit:** [ID]
**Completed By:** [Name/Agent]
**Date:** [Timestamp]
**Verified By:** [Human review if applicable]

### Checklist
- [ ] Unit works end-to-end within scope
- [ ] No architecture invariants violated
- [ ] `progress_tracker.md` updated
- [ ] Notebook/execution passes verification
- [ ] Artifacts committed to version control
- [ ] Next unit clearly identified

### Signature
`[Name]` — `Date`
```

---

### Automation

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Pre-commit hook: Verify unit completion before allowing commit

python scripts/verify_unit_completion.py --check-staged-files

if [ $? -ne 0 ]; then
    echo "❌ Unit completion checks failed. Fix before committing."
    exit 1
fi

echo "✅ Unit completion verified. Proceeding with commit."
exit 0
```

