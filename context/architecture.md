# Architecture Context — Adaptive Test-Time Reasoning Distillation (ATRD)

## Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **Frontend Framework** | Next.js 15 (App Router) + TypeScript | Documentation dashboard, competition write-up site, live demo interface for reasoning traces and budget forcing visualization |
| **UI System** | Tailwind CSS v4 + shadcn/ui | Glassmorphism design system, neural-themed components, responsive layout across all breakpoints |
| **ML Training Runtime** | Python 3.11 + PyTorch 2.3 + CUDA 12.1 | Core training loop, QLoRA fine-tuning, GRPO reinforcement learning on GPU |
| **Model Ecosystem** | Hugging Face Transformers + PEFT + TRL | Nemotron-3-Nano-30B loading, LoRA adapter management, GRPO trainer orchestration |
| **Inference Engine** | vLLM 0.5.0+ | Competition-mandated inference backend with reasoning parser plugin for `\boxed{}` extraction |
| **Quantization** | bitsandbytes 0.43.0+ | 4-bit NF4 QLoRA to fit 30B parameters within RTX PRO 6000 Blackwell memory constraints |
| **Data Pipeline** | Hugging Face Datasets + pandas | Streaming, filtering, deduplication (MinHash), and mixing of synthetic + OpenMathReasoning corpora |
| **Model Registry** | Hugging Face Hub | Private checkpoint storage for SFT and GRPO adapters with versioned repository structure |
| **Compute Environment** | Kaggle Notebooks (T4x2/P100) + Google Cloud G4 VMs (RTX PRO 6000 Blackwell) | Primary development on Kaggle; long-running training and final validation on G4 VMs |
| **Experiment Tracking** | JSON-line logs + inline matplotlib | Lightweight metric logging (no W&B/MLflow dependency to avoid Kaggle session overhead) |
| **Synthetic Generation** | OpenRouter / Together AI API (optional) | Frontier model access (DeepSeek-R1, Qwen3-235B) for failure-grounded synthetic data generation |
| **Version Control** | Git + Kaggle Dataset versioning | Source code versioning; dataset artifact versioning via Kaggle Datasets API |

---

## System Boundaries

### `context/` — Specification Layer
**Owns:** All project context files (`project_overview.md`, `ui_context.md`, `code_standards.md`, `ai_workflow_rules.md`, `progress_tracker.md`).  
**Responsibility:** Single source of truth for agent behavior, human decisions, and project state. No implementation code lives here. Files are read-only during execution and updated only via explicit human-approved sync workflows.

### `notebooks/` — Competition Execution Layer
**Owns:** Four sequential Kaggle notebooks (`01_data_generation.ipynb`, `02_sft_training.ipynb`, `03_grpo_training.ipynb`, `04_budget_forcing.ipynb`).  
**Responsibility:** End-to-end executable pipelines, one per phase. Each notebook is self-contained (imports, config, helpers, execution, evaluation) and must run independently within a 4-hour Kaggle GPU session. No cross-notebook runtime dependencies.

### `src/` — Reusable Python Modules
**Owns:** Modular Python packages (`data/`, `models/`, `training/`, `inference/`, `evaluation/`).  
**Responsibility:** Core business logic abstracted out of notebooks for testability and reuse. Imported via `%run` or `sys.path` in notebooks. Contains:
- `data/synthetic_generator.py` — Failure-mode targeted synthetic problem generation
- `data/judge_filter.py` — LLM-as-judge composite scoring and filtering
- `data/deduplicator.py` — MinHash near-duplicate detection
- `models/loader.py` — QLoRA-wrapped Nemotron loader with memory optimization
- `models/lora_config.py` — Rank-32 LoRA configuration validator
- `training/sft_trainer.py` — Supervised fine-tuning loop with checkpointing
- `training/grpo_trainer.py` — GRPO training with implicit PRM scoring
- `inference/budget_forcer.py` — Dynamic reasoning depth control (token monitoring + "Wait" injection)
- `inference/vllm_engine.py` — vLLM compatibility wrapper with reasoning parser integration
- `evaluation/metric.py` — Competition-grade accuracy evaluator (exact match + numerical tolerance)
- `evaluation/ablation.py` — Component isolation and per-phase accuracy delta measurement

### `components/` — UI Component Layer (Next.js)
**Owns:** React components for the documentation/demo site (`components/ui/` for shadcn primitives, `components/atrd/` for custom neural-themed components).  
**Responsibility:** Visualizing the ATRD pipeline — reasoning traces, budget gauges, failure heatmaps, phase steppers, leaderboard badges. Consumes static JSON exports from `notebooks/` (e.g., `evaluation/predictions.json`, `logs/training_*.json`).

### `configs/` — Configuration Layer
**Owns:** YAML and JSON configuration files (`base_lora.json`, `custom_lora.json`, `base_grpo.json`, `competition_params.json`).  
**Responsibility:** Immutable baseline configs and derived experiment configs. Hyperparameters are never hardcoded in notebooks; they are loaded from here. `competition_params.json` is absolutely immutable (temperature=0.0, max_tokens=7680, etc.).

### `scripts/` — Automation Layer
**Owns:** Shell and Python utility scripts (`verify_unit_completion.py`, `package_submission.py`, `sync_to_hub.py`).  
**Responsibility:** Pre-commit validation, submission packaging (`submission.zip` with `adapter_config.json` verification), checkpoint synchronization to Hugging Face Hub, and protected file integrity checks.

### `logs/` — Telemetry Layer
**Owns:** Structured JSON-line logs (`baseline_results.json`, `sft_training.json`, `grpo_rewards.json`, `stratified_evaluation.json`).  
**Responsibility:** Reproducibility evidence. Every training run, evaluation, and ablation study writes timestamped, schema-validated logs. Never hand-edited.

### `checkpoints/` — Artifact Storage Layer
**Owns:** LoRA checkpoint directories (`sft_checkpoint/`, `grpo_checkpoint/`).  
**Responsibility:** PEFT adapter weights, optimizer states, and scheduler states. Saved every 30 minutes during training to survive Kaggle session timeouts. Auto-resume capability from latest checkpoint.

---

## Storage Model

### Hugging Face Hub (Primary Model Registry)
**What lives here:**
- Private repositories for SFT and GRPO LoRA adapters (`atrd-nemotron-sft-r32`, `atrd-nemotron-grpo-r32`)
- Base model reference: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` (read-only, external)
- Versioned adapter checkpoints with commit messages describing training phase and hyperparameters
- `adapter_config.json` artifacts validated against vLLM schema before push

**Lifecycle:** Adapters remain private until competition end date. Post-competition, repositories may be made public for community reproducibility.

### Kaggle Datasets (Data Artifact Storage)
**What lives here:**
- `final_train_dataset.jsonl` — Curated, deduplicated, stratified training corpus
- `synthetic_data/raw_*.jsonl` — Raw frontier model outputs before filtering
- `synthetic_data/filtered_*.jsonl` — LLM-as-judge filtered synthetic problems
- `evaluation/predictions_*.jsonl` — Inference outputs per phase for ablation studies
- `submission.zip` — Final competition submission (LoRA adapter + config)

**Lifecycle:** Datasets are versioned by timestamp. Each phase completion triggers a dataset version update. Internet-enabled Kaggle notebooks read/write via `/kaggle/input/` and `/kaggle/working/`.

### Local Filesystem (Transient Compute Storage)
**What lives here:**
- Temporary training checkpoints during active Kaggle/G4 VM sessions
- Cached tokenized datasets to avoid reprocessing
- vLLM engine KV-cache and model weights during inference

**Lifecycle:** Ephemeral. All valuable artifacts are synced to Hugging Face Hub or Kaggle Datasets before session termination. `torch.cuda.empty_cache()` enforced after every model unload.

### Git Repository (Source Code Only)
**What lives here:**
- All `.py`, `.md`, `.json`, `.yaml`, `.ipynb` files
- `requirements.txt` with pinned versions
- `README.md` methodology write-up

**Invariant:** No binary files >50MB committed. Checkpoints and datasets are excluded via `.gitignore` and stored in external registries.

---

## Auth and Access Model

### Kaggle Authentication
- **Mechanism:** Kaggle API token (`kaggle.json`) mounted in notebook environment.
- **Scope:** Competition dataset downloads, dataset publishing, leaderboard submissions.
- **Ownership:** Single-user (Samar Abdelhameed Ahmed). No team collaboration tokens.

### Hugging Face Authentication
- **Mechanism:** `HF_TOKEN` environment variable with write access to private model repositories.
- **Scope:** Push/pull LoRA checkpoints, download base model weights, access gated datasets (OpenMathReasoning).
- **Ownership:** Personal Hugging Face account. Repositories are private until explicit public release.

### Frontier Model API Access (Optional)
- **Mechanism:** OpenRouter or Together AI API keys stored as Kaggle Secrets (not hardcoded).
- **Scope:** Synthetic data generation calls to DeepSeek-R1 / Qwen3-235B.
- **Rate Limiting:** Max 50 problems per batch request. Exponential backoff on 429 errors.
- **Fallback:** If API quota exceeded, switch to local Qwen2.5-7B-Instruct for synthetic generation and LLM-as-judge.

### Documentation Site Access
- **Mechanism:** No authentication required. Static Next.js site deployed on Vercel (or GitHub Pages).
- **Scope:** Public read-only access to methodology, ablation studies, and interactive demo.
- **Data Exposure:** Only aggregated metrics and anonymized failure mode distributions exposed. No raw competition test data or API keys exposed.

---

## Invariants

### 1. LoRA Rank Constraint
**Rule:** The LoRA adapter rank must never exceed 32.  
**Enforcement:** `LoraConfig` instantiation validates `r <= 32` via assertion. `adapter_config.json` is schema-checked by `scripts/verify_unit_completion.py` before any submission packaging.  
**Consequence of violation:** Disqualification from competition. Submission rejected by validation script.

### 2. Base Model Immutability
**Rule:** Base model weights (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`) are never modified, merged, or fine-tuned directly. Only LoRA adapter weights are trained.  
**Enforcement:** All training uses `peft.get_peft_model()` with `AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config)`. Direct weight assignment to base parameters is prohibited.  
**Consequence of violation:** Competition rules violation; submission incompatible with vLLM inference engine.

### 3. Zero Test Set Leakage
**Rule:** No test set data (public or private) may be used in training, data generation, or hyperparameter tuning.  
**Enforcement:** MinHash deduplication runs against test set before any synthetic data is added to training corpus. N-gram overlap analysis (n=5) must return 0 matches.  
**Consequence of violation:** Disqualification; prize eligibility revoked.

### 4. Competition Inference Parameter Lock
**Rule:** Inference parameters `temperature=0.0`, `max_tokens=7680`, `top_p=1.0`, `max_model_len=8192` are immutable. No temperature scheduling, dynamic top_p, or token limit manipulation permitted during evaluation.  
**Enforcement:** `configs/competition_params.json` is read-only. Inference engine initialization loads from this file exclusively.  
**Exception:** Budget forcing operates on token content ("Wait" injection), not generation parameters. This is explicitly allowed as it does not modify sampler hyperparameters.

### 5. No Long-Lived Background Work in Request Handlers
**Rule:** The documentation site (Next.js) serves static content and lightweight visualizations only. No API routes perform model inference, training, or data generation.  
**Enforcement:** All ML computation occurs in Kaggle notebooks or G4 VMs, not in the web frontend. The site consumes pre-generated JSON artifacts only.  
**Consequence of violation:** Vercel/G4 function timeout; security risk from exposed API keys; architectural boundary violation.

### 6. GPU Memory Hygiene
**Rule:** Every notebook cell that loads a model must explicitly unload it (`del model`, `torch.cuda.empty_cache()`) before the cell completes. No residual tensors between cells.  
**Enforcement:** Pre-commit hook runs `nvidia-smi` memory check. OOM errors in logs trigger automatic unit failure.  
**Consequence of violation:** Kaggle session crash; hours of training lost; inability to resume from checkpoint.

### 7. Reproducibility Seal
**Rule:** All random seeds are fixed (`random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`). All package versions are pinned in `requirements.txt`.  
**Enforcement:** Cell #1 of every notebook sets seeds. `requirements.txt` is validated against installed versions at runtime.  
**Consequence of violation:** Non-reproducible results; invalid ablation studies; documentation ineligible for prizes.

### 8. Phase Gate Integrity
**Rule:** No phase may begin before the preceding phase passes all exit criteria. No cross-phase work in a single session.  
**Enforcement:** `progress_tracker.md` must show `✅ COMPLETED` for previous phase before next phase spec is read. `scripts/verify_unit_completion.py` enforces artifact existence and metric thresholds.  
**Consequence of violation:** Unverified data quality corrupts training; unstable GRPO from weak SFT baseline; technical debt in 23-day timeline.

### 9. Documentation Completeness for Prize Eligibility
**Rule:** Public Kaggle notebook and methodology write-up are mandatory for any prize. The notebook must run end-to-end without errors.  
**Enforcement:** Notebook is executed in a fresh Kaggle session before final submission. Write-up word count ≥ 2,000 with ablation studies.  
**Consequence of violation:** Submission deemed ineligible for all prizes regardless of leaderboard position.

### 10. Protected File Immutability
**Rule:** `adapter_config.json`, `competition_metric.py`, `competition_params.json`, and base model weights are never modified by hand or by agent.  
**Enforcement:** `scripts/verify_protected_files.py` runs before every commit. Git pre-commit hook blocks commits with modified protected files.  
**Consequence of violation:** Submission schema mismatch with vLLM; invalid local evaluation; potential disqualification.

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL SOURCES                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ Kaggle       │  │ Hugging Face │  │ OpenRouter   │  │ NVIDIA      │  │
│  │ Competition  │  │ Datasets     │  │ / Together   │  │ Base Model  │  │
│  │ (train.csv)  │  │ (OpenMath)   │  │ (DeepSeek)   │  │ (Nemotron)  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘  │
└─────────┼─────────────────┼─────────────────┼─────────────────┼──────────┘
          │                 │                 │                 │
          ▼                 ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: DATA CURATION                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌────────┐  │
│  │ Baseline    │───▶│ Failure     │───▶│ Synthetic   │───▶│ Filter │  │
│  │ Evaluator   │    │ Extractor   │    │ Generator   │    │ (Judge)│  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └───┬────┘  │
│                                                                │       │
│  ┌───────────────────────────────────────────────────────────────┘       │
│  │                                                                      │
│  ▼                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐  │
│  │ Deduplicate │───▶│ Mix (75/25) │───▶│ final_train_dataset.jsonl   │  │
│  │ (MinHash)   │    │ Stratify    │    │ ──▶ Kaggle Datasets         │  │
│  └─────────────┘    └─────────────┘    └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PHASE 2: SFT TRAINING                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌────────┐  │
│  │ QLoRA       │───▶│ SFT Trainer │───▶│ Checkpoint  │───▶│ Eval   │  │
│  │ Loader      │    │ (TRL)       │    │ Save (HF)   │    │ Metric │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └───┬────┘  │
│                                                                 │       │
│  ┌───────────────────────────────────────────────────────────────┘       │
│  │                                                                      │
│  ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ sft_checkpoint/ ──▶ Hugging Face Hub (private)                  │   │
│  │ sft_results.json ──▶ logs/                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PHASE 3: GRPO RL (Optional)                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌────────┐  │
│  │ SFT         │───▶│ Implicit    │───▶│ GRPO        │───▶│ Eval   │  │
│  │ Checkpoint  │    │ PRM Setup   │    │ Trainer     │    │ Metric │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └───┬────┘  │
│                                                                 │       │
│  ┌───────────────────────────────────────────────────────────────┘       │
│  │                                                                      │
│  ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ grpo_checkpoint/ ──▶ Hugging Face Hub (private)                 │   │
│  │ grpo_results.json ──▶ logs/                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PHASE 4: TEST-TIME + SUBMISSION                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌────────┐  │
│  │ vLLM        │───▶│ Budget      │───▶│ Extract     │───▶│ Grade  │  │
│  │ Engine      │    │ Forcer      │    │ \boxed{}     │    │ Metric │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └───┬────┘  │
│                                                                 │       │
│  ┌───────────────────────────────────────────────────────────────┘       │
│  │                                                                      │
│  ▼                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐  │
│  │ Package ZIP │───▶│ Validate    │───▶│ Kaggle Submission           │  │
│  │ (adapter)   │    │ (vLLM test) │    │ + Public Notebook           │  │
│  └─────────────┘    └─────────────┘    └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Compute Architecture

### Kaggle Notebooks (Development & Iteration)
| Resource | Spec | Usage |
|----------|------|-------|
| GPU | T4 x2 or P100 | Baseline evaluation, data generation, SFT training (QLoRA 4-bit) |
| RAM | 16–32 GB | Dataset loading, preprocessing |
| Disk | 20 GB persistent | Checkpoints, logs, temporary datasets |
| Session | 4-hour wall-clock limit | All work must checkpoint every 30 minutes |
| Internet | Enabled | API calls for synthetic generation, Hugging Face Hub push/pull |

### Google Cloud G4 VMs (Training & Validation)
| Resource | Spec | Usage |
|----------|------|-------|
| GPU | NVIDIA RTX PRO 6000 Blackwell (1x) | Long-running SFT, GRPO training, vLLM inference validation |
| vCPUs | 12+ | Data preprocessing, evaluation |
| RAM | 64+ GB | Model loading, batch processing |
| Disk | 500 GB SSD | Base model weights, checkpoints, datasets |
| Cost | ~$3–5/hour | Budgeted for 25 days; sync checkpoints to Kaggle Datasets after use |

### Local / Vercel (Documentation Site)
| Resource | Spec | Usage |
|----------|------|-------|
| Runtime | Node.js 20 | Next.js static generation and SSR for dashboard |
| CDN | Vercel Edge | Global distribution of methodology write-up and demo |
| Compute | Serverless Functions | Lightweight API routes for static JSON artifact serving only |

---

## Security & Compliance

### API Key Management
- All API keys (Kaggle, Hugging Face, OpenRouter) stored as **Kaggle Secrets** or **environment variables**, never committed to Git.
- `.env.example` documents required variables without values.
- Git pre-commit hook scans for key patterns and blocks commits with potential leaks.

### Data Privacy
- Competition test data is never logged, cached, or transmitted outside Kaggle/G4 environments.
- Synthetic data generation prompts do not include raw test set problems — only failure mode categories and example structures.
- Documentation site exposes only aggregated, anonymized metrics.

### Competition Rules Compliance
- LoRA rank ≤ 32 (enforced via config validation).
- submission.zip format matches official demo exactly.
- Public notebook published before deadline (mandatory for prizes).
- No ensemble methods or model merging (explicitly out of scope per project rules).

---

*Document version: 1.0 — Aligned with ATRD Project Specification*
*Last updated: 2026-05-23*
