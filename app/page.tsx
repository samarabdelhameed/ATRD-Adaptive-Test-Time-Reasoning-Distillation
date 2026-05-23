"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Brain,
  FileCode,
  Folder,
  Terminal,
  Activity,
  Cpu,
  Database,
  TrendingUp,
  Clock,
  CheckCircle2,
  XCircle,
  UploadCloud,
  ArrowRight,
  ArrowLeft,
  RefreshCw,
  Play,
  BarChart2,
  Eye,
  ShieldAlert,
  Sparkles,
  Plus,
  Trophy,
  Zap,
  CheckCircle,
  Circle,
  Loader2,
  Copy,
  Check,
  ChevronDown,
  ChevronRight
} from "lucide-react";
import { cn } from "@/lib/utils";
import { NeuralPulse } from "@/components/atrd/neural-pulse";
import { MetricCard } from "@/components/atrd/metric-card";
import { PhaseStepper, Phase } from "@/components/atrd/phase-stepper";
import { BudgetGauge } from "@/components/atrd/budget-gauge";
import { ReasoningTrace, ReasoningStep } from "@/components/atrd/reasoning-trace";
import { FailureHeatmap, FailureCategory } from "@/components/atrd/failure-heatmap";
import { LeaderboardBadge } from "@/components/atrd/leaderboard-badge";
import { CodeBlock } from "@/components/atrd/code-block";
import { Button } from "@/components/ui/button";

export default function Home() {
  // Navigation Screens: 'welcome' | 'workspace' | 'submission'
  const [currentScreen, setCurrentScreen] = useState<"welcome" | "workspace" | "submission">("welcome");
  
  // Workspace Active Phase
  const [activePhaseId, setActivePhaseId] = useState<string>("p1");
  const [activeFile, setActiveFile] = useState<string>("src/inference/budget_forcer.py");
  
  // Phase 1 Data Curation sub-tabs: 'failure' | 'generator' | 'judge' | 'mixer'
  const [p1Tab, setP1Tab] = useState<"failure" | "generator" | "judge" | "mixer">("failure");
  
  // Slide Budget value
  const [budgetValue, setBudgetValue] = useState<number>(4096);
  
  // Telemetry Console Logs state
  const [logLogs, setLogLogs] = useState<string[]>([
    "SYSTEM: Base model nvidia/NVIDIA-Nemotron-3-Nano-30B initialized.",
    "CUDA: Blackwell TF32 optimizations applied to RTX PRO 6000 memory bounds.",
    "SYSTEM: Data pipelines loaded. Waiting for user action...",
  ]);
  const logEndRef = useRef<HTMLDivElement>(null);

  // Phase 1 Generator Sub-state
  const [genCategory, setGenCategory] = useState<string>("calculation_error");
  const [genCount, setGenCount] = useState<number>(500);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [generatedSamples, setGeneratedSamples] = useState<any[]>([]);

  // Phase 1 Judge Sub-state
  const [isFiltering, setIsFiltering] = useState<boolean>(false);
  const [filterProgress, setFilterProgress] = useState<number>(0);
  const [judgeDataset, setJudgeDataset] = useState<any[]>([]);

  // Phase 1 Mixing Sub-state
  const [mixerStats, setMixerStats] = useState({
    synthetic: 50,
    math: 25,
    code: 25,
    checkedLeakage: false,
    leakageResult: ""
  });

  // Phase 2 SFT Training Simulation Sub-state
  const [sftStatus, setSftStatus] = useState<"idle" | "training" | "completed">("idle");
  const [sftStep, setSftStep] = useState<number>(0);
  const [sftLoss, setSftLoss] = useState<number>(2.4);
  const [sftLossHistory, setSftLossHistory] = useState<{ step: number; loss: number }[]>([]);

  // Phase 3 GRPO Training Simulation Sub-state
  const [grpoStatus, setGrpoStatus] = useState<"idle" | "training" | "completed">("idle");
  const [grpoStep, setGrpoStep] = useState<number>(0);
  const [grpoReward, setGrpoReward] = useState<number>(0.2);
  const [grpoRewardHistory, setGrpoRewardHistory] = useState<{ step: number; value: number }[]>([]);
  const [grpoKL, setGrpoKL] = useState<number>(0.0);
  const [grpoKLHistory, setGrpoKLHistory] = useState<{ step: number; value: number }[]>([]);

  // Phase 4 Interactive Solver Sub-state
  const [customPrompt, setCustomPrompt] = useState<string>("Solve the integral of x*e^{-x} for x >= 0");
  const [solverStatus, setSolverStatus] = useState<"idle" | "solving" | "done">("idle");
  const [solverSteps, setSolverSteps] = useState<ReasoningStep[]>([]);
  const [solverLatency, setSolverLatency] = useState<number>(0);
  const [solverTokens, setSolverTokens] = useState<number>(0);

  // Phase 4 Submission packaging animation
  const [submissionProgress, setSubmissionProgress] = useState<number>(0);
  const [submissionStatus, setSubmissionStatus] = useState<"packaging" | "validating" | "success">("packaging");

  // Project checklist for phase steppers
  const [phases, setPhases] = useState<Phase[]>([
    { id: "p1", name: "Data Curation", description: "Baseline evaluation & synthetic correction generation.", status: "active" },
    { id: "p2", name: "SFT Fine-Tuning", description: "Instruction SFT to establish reasoning formats.", status: "pending" },
    { id: "p3", name: "GRPO RL Training", description: "Reinforcement learning driven by math & format rewards.", status: "pending" },
    { id: "p4", name: "Budget Forcing & TTA", description: "Adaptive compute scaling at test-time.", status: "pending" }
  ]);

  // Log update ticker
  useEffect(() => {
    if (currentScreen !== "workspace") return;

    const interval = setInterval(() => {
      const liveUpdates = [
        "TELEMETRY: Memory usage stable at 82% of RTX PRO 6000 bounds.",
        "MONITOR: GPU Blackwell core temperature is nominal (64°C).",
        `BUDGET: Tokenizer limit mapped to active budget: ${budgetValue} max.`,
        "SYSTEM: Verifying file integrity checks... all modules normal.",
      ];
      const randomMsg = liveUpdates[Math.floor(Math.random() * liveUpdates.length)];
      setLogLogs((prev) => [...prev.slice(-6), `${new Date().toLocaleTimeString()} - ${randomMsg}`]);
    }, 10000);

    return () => clearInterval(interval);
  }, [currentScreen, budgetValue]);

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logLogs]);

  // Helper to append console logs
  const addLog = (msg: string) => {
    setLogLogs((prev) => [...prev.slice(-8), `${new Date().toLocaleTimeString()} - ${msg}`]);
  };

  // Phase 1: Simulator for Synthetic Generation
  const handleStartGeneration = () => {
    if (isGenerating) return;
    setIsGenerating(true);
    addLog(`PIPELINE: Initiating targeted synthetic data generation for category [${genCategory}]...`);
    
    let generatedCount = 0;
    const interval = setInterval(() => {
      generatedCount += 1;
      const sampleQuestion = [
        `Evaluate ∫ x * cos(x) dx variations #${generatedCount}`,
        `Find the local extrema of f(x) = x^3 - 3x + 2 variation #${generatedCount}`,
        `Solve differential equation dy/dx - y = e^x variant #${generatedCount}`,
      ][generatedCount % 3];

      setGeneratedSamples((prev) => [
        {
          id: generatedCount,
          question: sampleQuestion,
          thinking: `<<thinking>> Let u = x, dv = cos(x) dx. Then du = dx, v = sin(x). Integration by parts: x*sin(x) - ∫ sin(x) dx = x*sin(x) + cos(x) + C. </thinking>>`,
          answer: `\\boxed{x \\sin(x) + \\cos(x) + C}`,
          difficulty: 0.45 + (generatedCount * 0.05) % 0.4
        },
        ...prev
      ]);

      addLog(`TEACHER API: Generated problem #${generatedCount} from DeepSeek-R1. Difficulty: ${(0.45 + (generatedCount * 0.05) % 0.4).toFixed(2)}`);

      if (generatedCount >= 5) {
        clearInterval(interval);
        setIsGenerating(false);
        addLog(`PIPELINE: Phase 1 Generation complete. Generated 5 high-quality traces.`);
        // Mark Phase 1 progress step
      }
    }, 1200);
  };

  // Phase 1: Filter Simulator
  const handleRunFilter = () => {
    if (isFiltering) return;
    setIsFiltering(true);
    setFilterProgress(0);
    addLog("JUDGE: Running composite quality score evaluation (weights: 0.35 Correctness, 0.25 Clarity, 0.20 Difficulty, 0.20 Format)...");

    let step = 0;
    const total = judgeDataset.length;

    const interval = setInterval(() => {
      if (step < total) {
        const currentItem = judgeDataset[step];
        const newStatus = currentItem.score >= 0.65 ? "passed" : "rejected";

        setJudgeDataset((prev) => {
          return prev.map((item, idx) => {
            if (idx === step) {
              return { ...item, status: newStatus };
            }
            return item;
          });
        });

        addLog(`JUDGE EVAL: Problem #${currentItem.id} - Score: ${currentItem.score.toFixed(2)} | Status: ${newStatus.toUpperCase()}`);

        step += 1;
        setFilterProgress(Math.round((step / total) * 100));
      } else {
        clearInterval(interval);
        setIsFiltering(false);
        addLog("JUDGE: Dataset filtering completed. Retained 4/6 problems (top 80% selection policy).");
      }
    }, 800);
  };

  // Phase 1: Mixer & Leakage check simulator
  const handleMixLeakageCheck = () => {
    addLog("MIXER: Enforcing 50/25/25 stratified ratio (Synthetic / OpenMath / OpenCode)...");
    addLog("MIXER: Scanning for 5-gram overlap leakage against benchmark evaluation suite...");
    
    setMixerStats((prev) => ({
      ...prev,
      checkedLeakage: true,
      leakageResult: "loading"
    }));

    setTimeout(() => {
      setMixerStats((prev) => ({
        ...prev,
        leakageResult: "clean"
      }));
      addLog("MIXER: Leakage analysis complete: 0 overlapping 5-grams detected. Dataset is clean!");
      
      // Complete Phase 1 and Unlock Phase 2
      setPhases((prev) => {
        const next = [...prev];
        next[0].status = "complete";
        next[1].status = "active";
        return next;
      });
      addLog("PIPELINE: Phase 1 (Data Curation) verified and locked. Phase 2 (SFT Fine-Tuning) is now active.");
    }, 1500);
  };

  // Phase 2: SFT training simulator
  const handleStartSFT = () => {
    if (sftStatus === "training") return;
    setSftStatus("training");
    setSftStep(0);
    setSftLoss(2.4);
    setSftLossHistory([]);
    addLog("TRAINING: Loading Nemotron-3-Nano-30B base weights with 4-bit NF4 quantization...");
    addLog("TRAINING: Attaching LoRA adapter configurations (rank 32, alpha 64)...");

    setTimeout(() => {
      addLog("TRAINING: Initiating AdamW optimizer loop. Starting Epoch 1 of 3...");
      
      let step = 0;
      let currentLoss = 2.45;
      const totalSteps = 300;

      const interval = setInterval(() => {
        step += 15;
        // Exponential decay model with random perturbation
        currentLoss = 0.4 + 2.05 * Math.exp(-step / 100) + (Math.random() - 0.5) * 0.08;
        if (currentLoss < 0.35) currentLoss = 0.35 + Math.random() * 0.05;

        setSftStep(step);
        setSftLoss(currentLoss);
        setSftLossHistory((prev) => [...prev, { step, loss: currentLoss }]);
        
        addLog(`SFT STEP: [Epoch ${Math.ceil(step / 100)}/3] Step ${step}/${totalSteps} | Training Loss: ${currentLoss.toFixed(4)} | LR: 2e-4`);

        if (step >= totalSteps) {
          clearInterval(interval);
          setSftStatus("completed");
          addLog("TRAINING: Supervised Fine-Tuning complete. Convergence criteria met.");
          addLog("TRAINING: Checkpoint weights archived to 'checkpoints/sft/final_adapter/'.");
          
          // Complete Phase 2 and Unlock Phase 3
          setPhases((prev) => {
            const next = [...prev];
            next[1].status = "complete";
            next[2].status = "active";
            return next;
          });
          addLog("PIPELINE: Phase 2 SFT adapter saved. Phase 3 (GRPO Reinforcement Learning) is now active.");
        }
      }, 300);
    }, 1000);
  };

  // Phase 3: GRPO RL simulator
  const handleStartGRPO = () => {
    if (grpoStatus === "training") return;
    setGrpoStatus("training");
    setGrpoStep(0);
    setGrpoReward(0.18);
    setGrpoRewardHistory([]);
    setGrpoKL(0.005);
    setGrpoKLHistory([]);
    addLog("GRPO: Initializing GRPO policy rollout (Group size G=8)...");
    addLog("GRPO: Setting reference policy SFT checkpoint to compute KL-divergence...");

    setTimeout(() => {
      addLog("GRPO: Running reward evaluation loops. Verifying mathematical and format accuracy scores...");

      let step = 0;
      let reward = 0.20;
      let kl = 0.005;
      const totalSteps = 100;

      const interval = setInterval(() => {
        step += 5;
        reward = 0.20 + 0.65 * (1 - Math.exp(-step / 40)) + (Math.random() - 0.5) * 0.04;
        if (reward > 0.94) reward = 0.94 + Math.random() * 0.01;

        kl = 0.005 + 0.025 * (1 - Math.exp(-step / 60)) + (Math.random() - 0.5) * 0.003;

        setGrpoStep(step);
        setGrpoReward(reward);
        setGrpoRewardHistory((prev) => [...prev, { step, value: reward }]);
        setGrpoKL(kl);
        setGrpoKLHistory((prev) => [...prev, { step, value: kl }]);

        addLog(`GRPO STEP: Iteration ${step}/${totalSteps} | Average Advantage: ${(reward * 1.2).toFixed(4)} | KL Penalty: ${kl.toFixed(5)}`);

        if (step >= totalSteps) {
          clearInterval(interval);
          setGrpoStatus("completed");
          addLog("GRPO: Policy alignment converged. Reward target threshold exceeded.");
          addLog("GRPO: Optimized LoRA weights merged. Saving checkpoint to 'checkpoints/grpo/'.");

          // Complete Phase 3 and Unlock Phase 4
          setPhases((prev) => {
            const next = [...prev];
            next[2].status = "complete";
            next[3].status = "active";
            return next;
          });
          addLog("PIPELINE: Phase 3 GRPO adapter verified. Phase 4 (Budget Forcing & TTA) is now active.");
        }
      }, 300);
    }, 1000);
  };

  // Phase 4: Custom Solver Simulation with streaming steps
  const handleSolvePrompt = () => {
    if (!customPrompt.trim() || solverStatus === "solving") return;
    setSolverStatus("solving");
    setSolverSteps([]);
    setSolverLatency(0);
    setSolverTokens(0);
    
    addLog(`INFERENCE: Running test inference on adaptive model for custom question: "${customPrompt}"`);
    addLog(`BUDGET FORCING: Checking problem complexity constraints. Active compute budget is ${budgetValue} tokens.`);

    // Determine problem complexity based on prompt content
    const lowerPrompt = customPrompt.toLowerCase();
    const isHard = lowerPrompt.includes("xe^{-x}") || lowerPrompt.includes("improper") || lowerPrompt.includes("parts") || budgetValue > 5000;
    const isMedium = lowerPrompt.includes("cos^2") || lowerPrompt.includes("integral") || lowerPrompt.includes("trig") || (budgetValue > 1000 && budgetValue <= 5000);

    const stepsToPlay: ReasoningStep[] = isHard ? [
      { id: "cs1", title: "Improper Integral Formulation", content: "Given: Find area under y = xe^{-x} for x >= 0.\nDifficulty classified as: Hard.\nEngaged max token budget: 7,680 tokens. Commencing structural parsing.", type: "thinking", durationMs: 1100, tokenCount: 120 },
      { id: "cs2", title: "Integration by Parts setup", content: "Formula: ∫ u dv = uv - ∫ v du.\nLet u = x => du = dx.\nLet dv = e^{-x} dx => v = -e^{-x}.", type: "assertion", durationMs: 900, tokenCount: 95 },
      { id: "cs3", title: "Executing parts expansion", content: "∫ xe^{-x} dx = -xe^{-x} - ∫ -e^{-x} dx\n= -xe^{-x} - e^{-x} + C\n= -e^{-x}(x + 1) + C.", type: "assertion", durationMs: 1400, tokenCount: 180 },
      { id: "cs4", title: "Intermediate Heuristic Verify", content: "Verification check:\nd/dx [-e^{-x}(x+1)] = e^{-x}(x+1) - e^{-x} = xe^{-x}.\nTrace correct. Moving to boundary limits.", type: "correction", durationMs: 800, tokenCount: 90 },
      { id: "cs5", title: "Improper bounds resolution", content: "Limit at x->∞: lim_{x->∞} -(x+1)/e^x = 0 (via L'Hopital's rule).\nLimit at x=0: -e^0(0+1) = -1.\nArea: F(∞) - F(0) = 0 - (-1) = 1.", type: "conclusion", durationMs: 1200, tokenCount: 150 },
      { id: "cs6", title: "Formatting answer environment", content: "Format rule verified. Wrapping result inside final boxed environment.\nResult: \\boxed{1}", type: "conclusion", durationMs: 400, tokenCount: 45 }
    ] : isMedium ? [
      { id: "cs1", title: "Identify Trigonometric Constraints", content: "Given: ∫ cos^2(x) dx from 0 to π.\nDifficulty classified as: Medium.\nCompute budget set to 4,096 tokens.", type: "thinking", durationMs: 800, tokenCount: 85 },
      { id: "cs2", title: "Trig Identity Reduction", content: "Rewrite cos^2(x) as (1 + cos(2x)) / 2.\nIntegrand becomes: 1/2 + (1/2)*cos(2x).", type: "assertion", durationMs: 700, tokenCount: 70 },
      { id: "cs3", title: "Integrating Terms", content: "∫ (1/2) dx = x/2\n∫ (1/2)*cos(2x) dx = (1/4)*sin(2x)\nAntiderivative F(x) = x/2 + sin(2x)/4.", type: "assertion", durationMs: 1100, tokenCount: 125 },
      { id: "cs4", title: "Boundary Calculation", content: "F(π) - F(0) = [π/2 + sin(2π)/4] - [0 + sin(0)/4] = π/2 - 0 = π/2.", type: "conclusion", durationMs: 900, tokenCount: 110 },
      { id: "cs5", title: "Formatting answer", content: "Boxed tags mapped.\nResult: \\boxed{\\frac{\\pi}{2}}", type: "conclusion", durationMs: 300, tokenCount: 30 }
    ] : [
      { id: "cs1", title: "Input Parsing", content: "Given: Solve 5x - 7 = 8.\nDifficulty: Easy. Compute budget set to 256 tokens.", type: "thinking", durationMs: 300, tokenCount: 40 },
      { id: "cs2", title: "Algebraic Simplification", content: "Add 7 to both sides: 5x = 15.\nDivide both sides by 5: x = 3.", type: "assertion", durationMs: 400, tokenCount: 50 },
      { id: "cs3", title: "Verification & Answer", content: "5(3) - 7 = 15 - 7 = 8. Match.\nResult: \\boxed{3}", type: "conclusion", durationMs: 200, tokenCount: 25 }
    ];

    let stepIdx = 0;
    const playNextStep = () => {
      if (stepIdx < stepsToPlay.length) {
        const nextStep = stepsToPlay[stepIdx];
        setSolverSteps((prev) => [...prev, nextStep]);
        setSolverLatency((prev) => prev + (nextStep.durationMs || 0));
        setSolverTokens((prev) => prev + (nextStep.tokenCount || 0));
        
        addLog(`INFERENCE: RENDERED Step ${stepIdx + 1}: ${nextStep.title} (${nextStep.tokenCount} tokens)`);
        
        stepIdx += 1;
        setTimeout(playNextStep, 800);
      } else {
        setSolverStatus("done");
        addLog("INFERENCE: Solver task completed successfully. Boxed answer extracted.");
      }
    };

    setTimeout(playNextStep, 600);
  };

  // Submit / Packaging simulation
  const handleLaunchSubmission = () => {
    setCurrentScreen("submission");
    setSubmissionStatus("packaging");
    setSubmissionProgress(0);
    addLog("SUBMITTER: Initiating submission archiving protocol...");

    let progress = 0;
    const interval = setInterval(() => {
      progress += 10;
      setSubmissionProgress(progress);

      if (progress === 40) {
        setSubmissionStatus("validating");
        addLog("SUBMITTER: Archive complete. Packaging 'submission.zip' containing weights...");
        addLog("VALIDATOR: Initiating rank verification check (limit: <= 32)...");
      } else if (progress === 70) {
        addLog("VALIDATOR: Rank verified. LoRA adapter rank = 32 (Passed ✅)");
        addLog("VALIDATOR: Verifying vLLM config rules and model templates...");
      } else if (progress >= 100) {
        clearInterval(interval);
        setSubmissionStatus("success");
        addLog("SUBMITTER: Adapter successfully validated. Generated target accuracy: 94.2% ( leaderboard #32 )!");
      }
    }, 400);
  };

  // Dynamic simulation values based on the slider budget
  const getDynamicLatency = () => {
    return ((budgetValue / 7680) * 14.5 + 2.1).toFixed(1);
  };

  const getDynamicTokens = () => {
    return Math.round(budgetValue * 0.96);
  };

  // Preset file configurations mapping
  const fileCodes: Record<string, string> = {
    "src/inference/budget_forcer.py": `class BudgetForcer:
    def __init__(self, min_tokens=256, max_tokens=7680):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.difficulty_threshold = 0.65

    def estimate_difficulty(self, problem: str) -> float:
        # Heuristics for token size
        indicators = ["integral", "proof", "optim", "matrix", "vector"]
        score = sum(0.2 for ind in indicators if ind in problem.lower())
        return min(1.0, score + (len(problem) / 500))

    def get_token_budget(self, problem: str) -> int:
        difficulty = self.estimate_difficulty(problem)
        if difficulty < 0.3:
            return self.min_tokens
        elif difficulty < self.difficulty_threshold:
            return 4096
        return self.max_tokens`,
    "src/training/grpo_trainer.py": `class GRPOTrainerWrapper:
    def __init__(self, model, lora_config, grpo_config):
        self.model = model
        self.lora_config = lora_config
        self.config = grpo_config

    def compute_rewards(self, completions, answers):
        # Format reward rules
        format_rewards = [0.2 if "\\\\boxed{" in c else 0.0 for c in completions]
        # Accuracy reward rules
        accuracy_rewards = [0.8 if self.eval_answer(c, a) else 0.0 
                            for c, a in zip(completions, answers)]
        
        return [f + a for f, a in zip(format_rewards, accuracy_rewards)]`,
    "configs/base_lora.json": `{
  "r": 32,
  "lora_alpha": 64,
  "target_modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
  ],
  "lora_dropout": 0.05,
  "bias": "none"
}`
  };

  return (
    <div className="flex-1 flex flex-col min-h-screen bg-void text-text-primary overflow-x-hidden">
      
      {/* ========================================================================= */}
      {/* ONBOARDING / WELCOME GATE */}
      {/* ========================================================================= */}
      {currentScreen === "welcome" && (
        <div className="flex-1 flex flex-col items-center justify-center px-4 relative">
          
          {/* Neon mesh background grid */}
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(118,185,0,0.12),transparent)] pointer-events-none" />
          <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:4rem_4rem] pointer-events-none" />
          
          <div className="max-w-4xl text-center flex flex-col items-center gap-8 relative z-10 animate-fade-up">
            
            {/* Brain/NVIDIA animated pulse header */}
            <div className="relative">
              <div className="absolute -inset-1 rounded-full bg-nvidia/20 blur-md animate-pulse" />
              <div className="h-16 w-16 rounded-2xl bg-nvidia/10 border border-nvidia/30 text-nvidia flex items-center justify-center shadow-[0_0_30px_rgba(118,185,0,0.3)]">
                <Brain className="h-9 w-9 animate-bounce" />
              </div>
            </div>

            <div className="flex flex-col gap-3">
              <h1 className="font-display text-4xl md:text-6xl font-bold tracking-tight text-white">
                ATRD <span className="text-nvidia">Pipeline Workspace</span>
              </h1>
              <p className="font-sans text-sm md:text-base text-text-secondary max-w-2xl leading-relaxed">
                Adaptive Test-Time Reasoning Distillation workspace. Fine-tuning <strong className="text-white">Nemotron-3-Nano-30B</strong> on 
                failure-grounded datasets with PRM-guided GRPO and adaptive budget-forcing.
              </p>
            </div>

            {/* Pipeline Overview cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 w-full text-left my-4">
              {[
                { step: "01", title: "Curation", desc: "Identify failure modes and generate synthetic corrections.", icon: <Database className="h-4 w-4 text-cyan" /> },
                { step: "02", title: "SFT Training", desc: "Instruction tune on Nemotron formatting tags.", icon: <Cpu className="h-4 w-4 text-nvidia" /> },
                { step: "03", title: "GRPO RL", desc: "Align steps via Process Reward Model policy updates.", icon: <TrendingUp className="h-4 w-4 text-purple" /> },
                { step: "04", title: "Budget Scaling", desc: "Extend search tokens dynamically for hard problems.", icon: <Clock className="h-4 w-4 text-amber" /> }
              ].map((item, idx) => (
                <div key={idx} className="glass-panel p-4 rounded-lg flex flex-col gap-2 relative bg-surface/40 hover:border-nvidia/30 hover:-translate-y-0.5 transition-all duration-300">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs text-nvidia font-semibold">PHASE {item.step}</span>
                    {item.icon}
                  </div>
                  <h3 className="font-sans text-sm font-semibold text-text-primary">{item.title}</h3>
                  <p className="font-sans text-[11px] text-text-secondary leading-normal">{item.desc}</p>
                </div>
              ))}
            </div>

            {/* Launch CTA */}
            <div className="flex flex-col sm:flex-row gap-4 items-center w-full justify-center">
              <Button
                onClick={() => {
                  setCurrentScreen("workspace");
                  addLog("WORKSPACE: Entered active project workspace.");
                }}
                className="bg-nvidia text-text-inverse font-display text-sm font-bold h-12 px-8 rounded-lg shadow-[0_0_30px_rgba(118,185,0,0.4)] hover:brightness-110 flex items-center gap-2 border-0"
              >
                Enter Research Workspace <ArrowRight className="h-4 w-4" />
              </Button>
              
              <a 
                href="https://github.com/samarabdelhameed/ATRD-Adaptive-Test-Time-Reasoning-Distillation" 
                target="_blank" 
                rel="noreferrer"
                className="flex items-center gap-2 px-6 h-12 rounded-lg border border-default text-text-secondary hover:text-white hover:bg-surface/40 transition-all font-display text-sm font-semibold"
              >
                <FileCode className="h-4 w-4" /> View Git Source
              </a>
            </div>

            {/* Specs Footer */}
            <div className="flex flex-wrap justify-center gap-x-8 gap-y-2 text-[11px] font-mono text-text-muted mt-6 border-t border-default/20 pt-6 w-full">
              <span>BASE: Nemotron-3-Nano-30B</span>
              <span>•</span>
              <span>ACCELERATION: TF32 matmul</span>
              <span>•</span>
              <span>MAX RANK: LoRA Rank 32</span>
              <span>•</span>
              <span>HARDWARE: RTX PRO 6000 Blackwell</span>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* MAIN RESEARCH WORKSPACE */}
      {/* ========================================================================= */}
      {currentScreen === "workspace" && (
        <div className="flex-1 flex flex-col">
          
          {/* Header */}
          <header className="sticky top-0 z-20 h-16 w-full glass-panel border-b border-default flex items-center justify-between px-6 bg-void/80 backdrop-blur-md">
            <div className="flex items-center gap-3">
              <div 
                onClick={() => setCurrentScreen("welcome")}
                className="flex h-8 w-8 items-center justify-center rounded-lg bg-nvidia/10 border border-nvidia/30 text-nvidia hover:scale-105 cursor-pointer glow-nvidia transition-all"
              >
                <Brain className="h-5 w-5" />
              </div>
              <div className="flex flex-col">
                <span className="font-display text-sm font-bold tracking-tight text-text-primary">
                  ATRD Dashboard
                </span>
                <span className="font-sans text-[10px] text-text-secondary uppercase tracking-wider">
                  Adaptive Test-Time Reasoning Distillation
                </span>
              </div>
            </div>

            <div className="flex items-center gap-6">
              <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full bg-surface border border-default text-xs">
                <NeuralPulse status={phases.find((p) => p.id === activePhaseId)?.status === "active" ? "active" : "idle"} />
                <span className="font-mono text-text-secondary text-[11px]">
                  Pipeline State: <strong className="text-cyan uppercase">{phases.find((p) => p.id === activePhaseId)?.name}</strong>
                </span>
              </div>

              <Button
                onClick={handleLaunchSubmission}
                className="bg-gradient-to-r from-nvidia to-nvidia/80 hover:brightness-110 text-text-inverse font-display text-xs font-semibold px-4 h-9 shadow-[0_0_15px_rgba(118,185,0,0.3)] border-0"
              >
                Submit Adapter <UploadCloud className="h-4 w-4 ml-1.5" />
              </Button>
            </div>
          </header>

          {/* Core Panel Grid */}
          <div className="flex-1 grid grid-cols-1 lg:grid-cols-[280px_1fr_320px] divide-x divide-default">
            
            {/* LEFT SIDEBAR: Steppers + File navigation */}
            <aside className="p-4 flex flex-col gap-6 bg-surface/20">
              <div className="flex flex-col gap-2">
                <span className="font-display text-xs font-semibold text-text-secondary uppercase tracking-wider pl-1">
                  Pipeline Stages
                </span>
                <PhaseStepper
                  phases={phases}
                  activePhaseId={activePhaseId}
                  onPhaseSelect={(id) => {
                    setActivePhaseId(id);
                    addLog(`WORKSPACE: Switched focus tab to Phase [${id}]`);
                  }}
                />
              </div>

              <div className="flex flex-col gap-2 flex-1">
                <span className="font-display text-xs font-semibold text-text-secondary uppercase tracking-wider pl-1">
                  Workspace Files
                </span>
                <div className="glass-panel p-3 rounded-lg flex flex-col gap-2.5 text-xs font-mono bg-surface/30">
                  <div className="flex items-center gap-2 text-text-secondary hover:text-text-primary cursor-pointer transition-colors">
                    <Folder className="h-4 w-4 text-nvidia shrink-0" />
                    <span className="font-bold">configs/</span>
                  </div>
                  <div 
                    onClick={() => {
                      setActiveFile("configs/base_lora.json");
                      addLog("FILE: Loaded lora configuration configs/base_lora.json");
                    }}
                    className={cn(
                      "flex items-center gap-2 pl-4 cursor-pointer hover:text-text-primary transition-colors",
                      activeFile === "configs/base_lora.json" ? "text-cyan font-bold" : "text-text-muted"
                    )}
                  >
                    <FileCode className="h-3.5 w-3.5" />
                    <span className="truncate">base_lora.json</span>
                  </div>

                  <div className="flex items-center gap-2 text-text-secondary hover:text-text-primary cursor-pointer transition-colors mt-2">
                    <Folder className="h-4 w-4 text-nvidia shrink-0" />
                    <span className="font-bold">src/</span>
                  </div>
                  <div 
                    onClick={() => {
                      setActiveFile("src/inference/budget_forcer.py");
                      addLog("FILE: Loaded python inference module budget_forcer.py");
                    }}
                    className={cn(
                      "flex items-center gap-2 pl-4 cursor-pointer hover:text-text-primary transition-colors",
                      activeFile === "src/inference/budget_forcer.py" ? "text-cyan font-bold" : "text-text-muted"
                    )}
                  >
                    <FileCode className="h-3.5 w-3.5" />
                    <span className="truncate">budget_forcer.py</span>
                  </div>
                  <div 
                    onClick={() => {
                      setActiveFile("src/training/grpo_trainer.py");
                      addLog("FILE: Loaded python training module grpo_trainer.py");
                    }}
                    className={cn(
                      "flex items-center gap-2 pl-4 cursor-pointer hover:text-text-primary transition-colors",
                      activeFile === "src/training/grpo_trainer.py" ? "text-cyan font-bold" : "text-text-muted"
                    )}
                  >
                    <FileCode className="h-3.5 w-3.5" />
                    <span className="truncate">grpo_trainer.py</span>
                  </div>
                </div>
              </div>

              {/* Back to welcome link */}
              <button 
                onClick={() => setCurrentScreen("welcome")}
                className="flex items-center gap-2 text-xs text-text-muted hover:text-nvidia font-display font-semibold transition-colors mt-auto pt-4 border-t border-default/40"
              >
                <ArrowLeft className="h-4 w-4" /> Exit to Introduction
              </button>
            </aside>

            {/* CENTER CANVAS: Phase workflow views */}
            <main className="p-6 flex flex-col gap-6 overflow-y-auto max-h-[calc(100vh-64px)] scrollbar-thin">
              
              {/* Phase Header */}
              <div className="flex items-center justify-between border-b border-default pb-4">
                <div className="flex flex-col gap-1">
                  <span className="font-mono text-[10px] text-nvidia font-bold tracking-widest uppercase">
                    SYSTEM WORKSPACE
                  </span>
                  <h1 className="font-display text-2xl font-bold tracking-tight text-text-primary">
                    {phases.find((p) => p.id === activePhaseId)?.name}
                  </h1>
                  <p className="font-sans text-xs text-text-secondary">
                    {phases.find((p) => p.id === activePhaseId)?.description}
                  </p>
                </div>
                
                <div className="flex items-center gap-1.5 px-3 py-1 rounded bg-elevated border border-default text-[10px] text-text-secondary font-mono">
                  <Activity className="h-3.5 w-3.5 text-cyan shrink-0 animate-pulse" />
                  <span>ACTIVE PHASE ID: 0{activePhaseId.slice(1)}</span>
                </div>
              </div>

              {/* ========================================== */}
              {/* PHASE 1: DATA CURATION VIEW */}
              {/* ========================================== */}
              {activePhaseId === "p1" && (
                <div className="flex flex-col gap-6">
                  
                  {/* Phase 1 sub-tab navigation */}
                  <div className="flex border-b border-default gap-4 text-xs font-display font-semibold uppercase">
                    {[
                      { id: "failure", label: "Failure Analysis" },
                      { id: "generator", label: "Synthetic Generator" },
                      { id: "judge", label: "Judge Filter" },
                      { id: "mixer", label: "Mixing & Leakage" }
                    ].map((tab) => (
                      <button
                        key={tab.id}
                        onClick={() => setP1Tab(tab.id as any)}
                        className={cn(
                          "pb-2 border-b-2 px-1 transition-all",
                          p1Tab === tab.id
                            ? "border-nvidia text-nvidia"
                            : "border-transparent text-text-muted hover:text-text-secondary"
                        )}
                      >
                        {tab.label}
                      </button>
                    ))}
                  </div>

                  {/* SUBTAB CONTENT */}
                  {p1Tab === "failure" && (
                    <div className="flex flex-col gap-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="glass-panel p-5 rounded-lg flex flex-col gap-4 bg-surface/30">
                          <h3 className="font-display text-xs font-semibold tracking-wider text-text-secondary uppercase">
                            Nemotron Failure Mode Breakdown
                          </h3>
                          <p className="text-xs text-text-secondary leading-relaxed">
                            Evaluating Nemotron-3-Nano-30B on the public reasoning benchmark shows major failure clusters. 
                            We target these failure modes with synthetic correction traces generated via teacher models.
                          </p>
                          <div className="flex flex-col gap-2 pt-2">
                            {[
                              { mode: "Calculation Error", rate: "44%", count: 521, color: "bg-rose" },
                              { mode: "Format Violation", rate: "18%", count: 210, color: "bg-amber" },
                              { mode: "Reasoning Loop", rate: "12%", count: 142, color: "bg-nvidia" },
                              { mode: "Early Termination", rate: "6%", count: 72, color: "bg-cyan" }
                            ].map((item, idx) => (
                              <div key={idx} className="flex items-center justify-between text-xs font-mono bg-void/50 p-2.5 rounded border border-default">
                                <div className="flex items-center gap-2">
                                  <span className={cn("h-2.5 w-2.5 rounded-full shrink-0", item.color)} />
                                  <span className="text-text-primary">{item.mode}</span>
                                </div>
                                <span className="text-text-secondary">{item.count} items ({item.rate})</span>
                              </div>
                            ))}
                          </div>
                        </div>

                        <div className="glass-panel p-5 rounded-lg flex flex-col gap-4 bg-surface/30 justify-between">
                          <div className="flex flex-col gap-2">
                            <h3 className="font-display text-xs font-semibold tracking-wider text-text-secondary uppercase">
                              Baseline Evaluation Metrics
                            </h3>
                            <p className="text-xs text-text-secondary leading-relaxed">
                              Pre-training baseline evaluation results on representative math problems.
                            </p>
                          </div>
                          
                          <div className="grid grid-cols-2 gap-3 pt-4">
                            <div className="bg-void/50 p-4 rounded border border-default text-center">
                              <span className="font-sans text-[10px] text-text-muted uppercase block">Baseline Acc</span>
                              <span className="font-mono text-xl font-bold text-rose">59.8%</span>
                            </div>
                            <div className="bg-void/50 p-4 rounded border border-default text-center">
                              <span className="font-sans text-[10px] text-text-muted uppercase block">Boxed Format Compliance</span>
                              <span className="font-mono text-xl font-bold text-amber">68.2%</span>
                            </div>
                          </div>
                          
                          <Button 
                            onClick={() => setP1Tab("generator")}
                            className="bg-nvidia text-text-inverse font-display text-xs font-semibold mt-4 h-9 border-0 w-full"
                          >
                            Proceed to Synthetic Generation <ArrowRight className="h-4 w-4 ml-1" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  )}

                  {p1Tab === "generator" && (
                    <div className="glass-panel p-5 rounded-lg flex flex-col gap-4 bg-surface/30">
                      <div className="flex flex-col gap-1">
                        <h3 className="font-display text-xs font-semibold tracking-wider text-text-secondary uppercase">
                          Targeted Synthetic Generator
                        </h3>
                        <p className="text-xs text-text-secondary leading-relaxed">
                          For each failure mode, we query a frontier teacher model (DeepSeek-R1) to generate variations 
                          with correct reasoning traces wrapped in Nemotron tokens.
                        </p>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 bg-void/50 p-4 rounded border border-default">
                        <div className="flex flex-col gap-1.5">
                          <label className="text-[10px] font-sans text-text-secondary uppercase">Failure Category</label>
                          <select
                            value={genCategory}
                            onChange={(e) => setGenCategory(e.target.value)}
                            className="bg-surface border border-default p-2 rounded text-xs text-text-primary focus:outline-none focus:ring-1 focus:ring-nvidia"
                          >
                            <option value="calculation_error">Calculation Error</option>
                            <option value="format_violation">Format Violation</option>
                            <option value="reasoning_loop">Reasoning Loop</option>
                            <option value="early_termination">Early Termination</option>
                          </select>
                        </div>

                        <div className="flex flex-col gap-1.5">
                          <label className="text-[10px] font-sans text-text-secondary uppercase">Target Problems</label>
                          <input
                            type="number"
                            value={genCount}
                            onChange={(e) => setGenCount(parseInt(e.target.value) || 100)}
                            className="bg-surface border border-default p-2 rounded text-xs text-text-primary focus:outline-none focus:ring-1 focus:ring-nvidia"
                          />
                        </div>

                        <div className="flex items-end">
                          <Button
                            onClick={handleStartGeneration}
                            disabled={isGenerating}
                            className="bg-nvidia text-text-inverse font-display text-xs font-semibold h-9.5 w-full border-0 shadow-[0_0_15px_rgba(118,185,0,0.2)]"
                          >
                            {isGenerating ? (
                              <span className="flex items-center justify-center gap-2">
                                <Loader2 className="h-4 w-4 animate-spin" /> Generating...
                              </span>
                            ) : (
                              <span className="flex items-center justify-center gap-1">
                                <Play className="h-4 w-4" /> Start Generation Loop
                              </span>
                            )}
                          </Button>
                        </div>
                      </div>

                      {/* Display generated samples */}
                      {generatedSamples.length > 0 && (
                        <div className="flex flex-col gap-3 pt-2">
                          <span className="font-mono text-[10px] text-text-muted uppercase">Generated Samples stream:</span>
                          <div className="flex flex-col gap-2 max-h-[220px] overflow-y-auto scrollbar-thin">
                            {generatedSamples.map((sample) => (
                              <div key={sample.id} className="p-3 bg-void/30 rounded border border-default flex flex-col gap-1.5 text-xs font-mono">
                                <div className="flex items-center justify-between text-[10px] text-text-muted">
                                  <span className="text-cyan">API RESPONSE MATCH #{sample.id}</span>
                                  <span>Diff Estimate: {sample.difficulty.toFixed(2)}</span>
                                </div>
                                <span className="text-text-primary font-semibold">Q: {sample.question}</span>
                                <span className="text-text-secondary select-all">{sample.thinking}</span>
                                <span className="text-nvidia">A: {sample.answer}</span>
                              </div>
                            ))}
                          </div>
                          
                          <Button
                            onClick={() => setP1Tab("judge")}
                            className="bg-cyan hover:brightness-110 text-text-inverse font-display text-xs font-semibold self-end h-8 px-4 border-0 mt-2"
                          >
                            Move to Judge Filter Scorer <ArrowRight className="h-3.5 w-3.5 ml-1" />
                          </Button>
                        </div>
                      )}

                    </div>
                  )}

                  {p1Tab === "judge" && (
                    <div className="glass-panel p-5 rounded-lg flex flex-col gap-4 bg-surface/30">
                      <div className="flex flex-col gap-1">
                        <h3 className="font-display text-xs font-semibold tracking-wider text-text-secondary uppercase">
                          LLM-As-Judge Filter & Scorer
                        </h3>
                        <p className="text-xs text-text-secondary leading-relaxed">
                          Retain only the top 80% percentile of generated data based on correctness (35%), clarity (25%), difficulty (20%), and format (20%).
                        </p>
                      </div>

                      <Button
                        onClick={handleRunFilter}
                        disabled={isFiltering}
                        className="bg-nvidia text-text-inverse font-display text-xs font-semibold h-9.5 self-start px-5 border-0 shadow-[0_0_15px_rgba(118,185,0,0.2)]"
                      >
                        {isFiltering ? `Scoring Traces (${filterProgress}%)` : "Launch Judge Filter Engine"}
                      </Button>

                      {/* Score Table */}
                      <div className="overflow-x-auto border border-default rounded">
                        <table className="w-full text-xs text-left font-mono">
                          <thead className="bg-void/50 text-[10px] text-text-secondary uppercase border-b border-default">
                            <tr>
                              <th className="p-3">Problem ID</th>
                              <th className="p-3">Core Question</th>
                              <th className="p-3 text-center">Score</th>
                              <th className="p-3 text-center">Status</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-default bg-void/10">
                            {judgeDataset.map((row) => (
                              <tr key={row.id} className="hover:bg-void/40 transition-colors">
                                <td className="p-3 text-text-muted">#00{row.id}</td>
                                <td className="p-3 text-text-primary max-w-[240px] truncate">{row.question}</td>
                                <td className="p-3 text-center font-bold">{row.score.toFixed(2)}</td>
                                <td className="p-3 text-center">
                                  <span className={cn(
                                    "px-2 py-0.5 rounded-full text-[9px] uppercase border font-semibold",
                                    row.status === "passed" && "bg-nvidia/10 text-nvidia border-nvidia/30",
                                    row.status === "rejected" && "bg-rose/10 text-rose border-rose/30",
                                    row.status === "pending" && "bg-elevated text-text-muted border-default"
                                  )}>
                                    {row.status}
                                  </span>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>

                      <Button
                        onClick={() => setP1Tab("mixer")}
                        className="bg-cyan hover:brightness-110 text-text-inverse font-display text-xs font-semibold self-end h-8 px-4 border-0"
                      >
                        Move to Dataset Mixer <ArrowRight className="h-3.5 w-3.5 ml-1" />
                      </Button>
                    </div>
                  )}

                  {p1Tab === "mixer" && (
                    <div className="glass-panel p-5 rounded-lg flex flex-col gap-4 bg-surface/30">
                      <div className="flex flex-col gap-1">
                        <h3 className="font-display text-xs font-semibold tracking-wider text-text-secondary uppercase">
                          Mixing Ratios & Leakage Guard
                        </h3>
                        <p className="text-xs text-text-secondary leading-relaxed">
                          Merge synthetic traces with open resources using a strict 50% Synthetic / 25% Math / 25% Code ratio. 
                          Then verify zero 5-gram overlap leakage against test sets.
                        </p>
                      </div>

                      {/* Mixing ratio visualization bar */}
                      <div className="flex flex-col gap-2">
                        <div className="flex items-center justify-between text-xs font-mono text-text-secondary">
                          <span>Stratified Ratios Mix:</span>
                          <span className="text-white">50% / 25% / 25%</span>
                        </div>
                        <div className="h-6 w-full rounded overflow-hidden flex font-mono text-[9px] font-bold text-center text-text-inverse border border-default shadow-inner">
                          <div className="bg-nvidia flex items-center justify-center transition-all duration-500" style={{ width: `${mixerStats.synthetic}%` }}>
                            SYNTHETIC (50%)
                          </div>
                          <div className="bg-cyan flex items-center justify-center transition-all duration-500" style={{ width: `${mixerStats.math}%` }}>
                            MATH (25%)
                          </div>
                          <div className="bg-purple flex items-center justify-center transition-all duration-500" style={{ width: `${mixerStats.code}%` }}>
                            CODE (25%)
                          </div>
                        </div>
                      </div>

                      {/* Leakage checking control */}
                      <div className="bg-void/50 p-4 rounded border border-default flex flex-col gap-3 mt-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-sans text-text-secondary">Leakage check status</span>
                          {mixerStats.checkedLeakage && (
                            <span className={cn(
                              "text-xs font-mono font-bold uppercase",
                              mixerStats.leakageResult === "clean" ? "text-nvidia" : "text-amber"
                            )}>
                              {mixerStats.leakageResult === "clean" ? "Passed (Clean)" : "Running..."}
                            </span>
                          )}
                        </div>

                        <Button
                          onClick={handleMixLeakageCheck}
                          disabled={mixerStats.checkedLeakage && mixerStats.leakageResult === "clean"}
                          className="bg-nvidia text-text-inverse font-display text-xs font-semibold h-9.5 self-start px-5 border-0"
                        >
                          Run 5-Gram Leakage Scan
                        </Button>

                        {mixerStats.checkedLeakage && mixerStats.leakageResult === "clean" && (
                          <div className="p-3 bg-nvidia/10 border border-nvidia/30 text-nvidia text-xs rounded font-mono">
                            ✓ No leakage detected. Final training dataset structured successfully as 
                            <strong> final_train_dataset.jsonl</strong>. Phase 1 complete.
                          </div>
                        )}
                      </div>

                    </div>
                  )}

                </div>
              )}

              {/* ========================================== */}
              {/* PHASE 2: SFT TRAINING VIEW */}
              {/* ========================================== */}
              {activePhaseId === "p2" && (
                <div className="flex flex-col gap-6">
                  
                  {/* SFT Panel */}
                  <div className="glass-panel p-5 rounded-lg flex flex-col gap-4 bg-surface/30">
                    <div className="flex flex-col gap-1">
                      <h3 className="font-display text-xs font-semibold tracking-wider text-text-secondary uppercase">
                        Supervised Fine-Tuning execution
                      </h3>
                      <p className="text-xs text-text-secondary leading-relaxed">
                        Attach a rank-32 adapter and train on format templates. Uses QLoRA 4-bit NF4 weights loading 
                        to fit RTX PRO 6000 memory specifications.
                      </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 bg-void/50 p-4 rounded border border-default">
                      <div className="flex flex-col gap-1">
                        <span className="font-sans text-[10px] text-text-secondary uppercase">Learning Rate</span>
                        <span className="font-mono text-xs font-semibold text-text-primary">2e-4 (Cosine scheduler)</span>
                      </div>
                      <div className="flex flex-col gap-1">
                        <span className="font-sans text-[10px] text-text-secondary uppercase">Target Epochs</span>
                        <span className="font-mono text-xs font-semibold text-text-primary">3 epochs (total 300 steps)</span>
                      </div>
                      <div className="flex flex-col gap-1">
                        <span className="font-sans text-[10px] text-text-secondary uppercase">Grad Accumulation</span>
                        <span className="font-mono text-xs font-semibold text-text-primary">8 steps (batch size = 1)</span>
                      </div>
                    </div>

                    <Button
                      onClick={handleStartSFT}
                      disabled={sftStatus === "training"}
                      className="bg-nvidia text-text-inverse font-display text-xs font-semibold h-9.5 self-start px-5 border-0"
                    >
                      {sftStatus === "training" ? (
                        <span className="flex items-center gap-2">
                          <Loader2 className="h-4 w-4 animate-spin" /> Training SFT Adapter...
                        </span>
                      ) : sftStatus === "completed" ? (
                        "Restart SFT Training"
                      ) : (
                        "Launch Supervised Fine-Tuning"
                      )}
                    </Button>
                  </div>

                  {/* SFT Telemetry logs & charts */}
                  {sftStatus !== "idle" && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      
                      {/* Interactive Training stats */}
                      <div className="glass-panel p-5 rounded-lg flex flex-col gap-3 bg-surface/30 justify-between">
                        <div className="flex flex-col gap-1">
                          <span className="font-display text-xs font-semibold tracking-wider text-text-secondary uppercase">
                            SFT Metrics Monitor
                          </span>
                        </div>
                        
                        <div className="flex flex-col gap-2 pt-2">
                          <div className="flex items-center justify-between text-xs font-mono bg-void/50 p-2.5 rounded border border-default">
                            <span className="text-text-secondary">Current Step</span>
                            <span className="text-text-primary font-bold">{sftStep} / 300</span>
                          </div>
                          <div className="flex items-center justify-between text-xs font-mono bg-void/50 p-2.5 rounded border border-default">
                            <span className="text-text-secondary">Training Loss</span>
                            <span className="text-rose font-bold">{sftLoss.toFixed(4)}</span>
                          </div>
                        </div>

                        {sftStatus === "completed" && (
                          <div className="p-3 bg-nvidia/10 border border-nvidia/30 text-nvidia text-xs rounded font-mono mt-4">
                            ✓ Supervised Fine-Tuning complete. Training loss converged at {sftLoss.toFixed(4)}. Checkpoints saved.
                          </div>
                        )}
                      </div>

                      {/* Simulated Loss Curve SVG graph */}
                      <div className="glass-panel p-5 rounded-lg flex flex-col gap-3 bg-surface/30 min-h-[220px]">
                        <span className="font-display text-xs font-semibold tracking-wider text-text-secondary uppercase">
                          SFT Training Loss Curve
                        </span>
                        
                        <div className="flex-1 w-full bg-void/60 rounded border border-default relative flex items-center justify-center p-4">
                          {sftLossHistory.length > 1 ? (
                            <svg className="w-full h-full min-h-[120px]" viewBox="0 0 300 100" preserveAspectRatio="none">
                              {/* Draw SVG Grid Lines */}
                              <line x1="0" y1="20" x2="300" y2="20" stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
                              <line x1="0" y1="50" x2="300" y2="50" stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
                              <line x1="0" y1="80" x2="300" y2="80" stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
                              
                              {/* Loss curve path */}
                              <path
                                d={`M ${sftLossHistory.map((pt) => `${pt.step} ${Math.max(10, 90 - (pt.loss / 2.5) * 80)}`).join(" L ")}`}
                                fill="none"
                                stroke="#FF4D6D"
                                strokeWidth="2"
                                className="transition-all duration-300"
                              />
                            </svg>
                          ) : (
                            <span className="font-mono text-xs text-text-muted">Waiting for training steps...</span>
                          )}
                        </div>
                      </div>

                    </div>
                  )}

                </div>
              )}

              {/* ========================================== */}
              {/* PHASE 3: GRPO REINFORCEMENT LEARNING VIEW */}
              {/* ========================================== */}
              {activePhaseId === "p3" && (
                <div className="flex flex-col gap-6">
                  
                  {/* GRPO config */}
                  <div className="glass-panel p-5 rounded-lg flex flex-col gap-4 bg-surface/30">
                    <div className="flex flex-col gap-1">
                      <h3 className="font-display text-xs font-semibold tracking-wider text-text-secondary uppercase">
                        Group Relative Policy Optimization (GRPO)
                      </h3>
                      <p className="text-xs text-text-secondary leading-relaxed">
                        Tune the SFT adapter using GRPO with group size G=8. Rewarded for correct intermediate thinking 
                        trace steps (monitored by PRM) and correct final boxed answers.
                      </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 bg-void/50 p-4 rounded border border-default">
                      <div className="flex flex-col gap-1">
                        <span className="font-sans text-[10px] text-text-secondary uppercase">Group Size (G)</span>
                        <span className="font-mono text-xs font-semibold text-text-primary">8 rollouts per problem</span>
                      </div>
                      <div className="flex flex-col gap-1">
                        <span className="font-sans text-[10px] text-text-secondary uppercase">KL Divergence Target</span>
                        <span className="font-mono text-xs font-semibold text-text-primary">1e-3 penalty weight</span>
                      </div>
                      <div className="flex flex-col gap-1">
                        <span className="font-sans text-[10px] text-text-secondary uppercase">PRM reward Scorer</span>
                        <span className="font-mono text-xs font-semibold text-text-primary">Heuristic step scoring</span>
                      </div>
                    </div>

                    <Button
                      onClick={handleStartGRPO}
                      disabled={grpoStatus === "training"}
                      className="bg-nvidia text-text-inverse font-display text-xs font-semibold h-9.5 self-start px-5 border-0"
                    >
                      {grpoStatus === "training" ? (
                        <span className="flex items-center gap-2">
                          <Loader2 className="h-4 w-4 animate-spin" /> Running GRPO Optimizer...
                        </span>
                      ) : grpoStatus === "completed" ? (
                        "Restart GRPO Training"
                      ) : (
                        "Run GRPO Reinforcement Loop"
                      )}
                    </Button>
                  </div>

                  {/* GRPO Graphs and stats */}
                  {grpoStatus !== "idle" && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      
                      {/* Metric cards */}
                      <div className="glass-panel p-5 rounded-lg flex flex-col gap-3 bg-surface/30 justify-between">
                        <span className="font-display text-xs font-semibold tracking-wider text-text-secondary uppercase">
                          GRPO Reinforcement Metrics
                        </span>
                        
                        <div className="flex flex-col gap-2 pt-2">
                          <div className="flex items-center justify-between text-xs font-mono bg-void/50 p-2.5 rounded border border-default">
                            <span className="text-text-secondary">GRPO Iteration</span>
                            <span className="text-text-primary font-bold">{grpoStep} / 100</span>
                          </div>
                          <div className="flex items-center justify-between text-xs font-mono bg-void/50 p-2.5 rounded border border-default">
                            <span className="text-text-secondary">Average Reward</span>
                            <span className="text-nvidia font-bold">{grpoReward.toFixed(4)}</span>
                          </div>
                          <div className="flex items-center justify-between text-xs font-mono bg-void/50 p-2.5 rounded border border-default">
                            <span className="text-text-secondary">KL Divergence</span>
                            <span className="text-cyan font-bold">{grpoKL.toFixed(6)}</span>
                          </div>
                        </div>

                        {grpoStatus === "completed" && (
                          <div className="p-3 bg-nvidia/10 border border-nvidia/30 text-nvidia text-xs rounded font-mono mt-4">
                            ✓ GRPO Policy updates complete. Average reward increased monotonically to {grpoReward.toFixed(4)}. KL penalty stable.
                          </div>
                        )}
                      </div>

                      {/* Reward Curve SVG Graph */}
                      <div className="glass-panel p-5 rounded-lg flex flex-col gap-3 bg-surface/30 min-h-[220px]">
                        <span className="font-display text-xs font-semibold tracking-wider text-text-secondary uppercase">
                          Average Reward Convergence Curve
                        </span>
                        
                        <div className="flex-1 w-full bg-void/60 rounded border border-default relative flex items-center justify-center p-4">
                          {grpoRewardHistory.length > 1 ? (
                            <svg className="w-full h-full min-h-[120px]" viewBox="0 0 100 100" preserveAspectRatio="none">
                              <line x1="0" y1="20" x2="100" y2="20" stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
                              <line x1="0" y1="50" x2="100" y2="50" stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
                              <line x1="0" y1="80" x2="100" y2="80" stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
                              
                              <path
                                d={`M ${grpoRewardHistory.map((pt) => `${pt.step} ${90 - pt.value * 80}`).join(" L ")}`}
                                fill="none"
                                stroke="#76B900"
                                strokeWidth="2"
                                className="transition-all duration-300"
                              />
                            </svg>
                          ) : (
                            <span className="font-mono text-xs text-text-muted">Waiting for GRPO rollouts...</span>
                          )}
                        </div>
                      </div>

                    </div>
                  )}

                </div>
              )}

              {/* ========================================== */}
              {/* PHASE 4: BUDGET FORCING & TEST SOLVER VIEW */}
              {/* ========================================== */}
              {activePhaseId === "p4" && (
                <div className="flex flex-col gap-6 animate-fade-up">
                  
                  {/* Gauge & Heatmap metrics grid */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <BudgetGauge value={budgetValue} onChange={setBudgetValue} />
                    <FailureHeatmap categories={[
                      { id: "err1", name: "Reasoning Loop", count: 142, rate: 0.12, description: "Stuck repeating statements." },
                      { id: "err2", name: "Format Violation", rate: 0.18, count: 210, description: "Missing boxed block." },
                      { id: "err3", name: "Early Termination", rate: 0.06, count: 72, description: "Halted generation early." },
                      { id: "err4", name: "Calculation Error", rate: 0.44, count: 521, description: "Arithmetic error in traces." }
                    ]} />
                  </div>

                  {/* Adaptive Math Solver Console */}
                  <div className="glass-panel p-5 rounded-lg flex flex-col gap-4 bg-surface/30">
                    <div className="flex flex-col gap-1">
                      <h3 className="font-display text-xs font-semibold tracking-wider text-text-secondary uppercase">
                        ATRD Interactive Solver Console
                      </h3>
                      <p className="text-xs text-text-secondary leading-relaxed">
                        Enter a math problem or select a preset. The solver will simulate computing a step-by-step reasoning 
                        trace limited by the budget gauge value.
                      </p>
                    </div>

                    <div className="flex flex-col gap-3">
                      
                      {/* Presets */}
                      <div className="flex flex-wrap gap-2 text-xs">
                        <button
                          onClick={() => {
                            setCustomPrompt("Solve 5x - 7 = 8");
                            addLog("CONSOLE: Preset selected (Easy algebra problem).");
                          }}
                          className="px-2.5 py-1 rounded bg-elevated hover:bg-surface border border-default text-text-secondary hover:text-white transition-colors"
                        >
                          Algebra: 5x - 7 = 8
                        </button>
                        <button
                          onClick={() => {
                            setCustomPrompt("Compute the integral of cos^2(x) from 0 to pi");
                            addLog("CONSOLE: Preset selected (Medium integration problem).");
                          }}
                          className="px-2.5 py-1 rounded bg-elevated hover:bg-surface border border-default text-text-secondary hover:text-white transition-colors"
                        >
                          Trig: ∫ cos^2(x) dx
                        </button>
                        <button
                          onClick={() => {
                            setCustomPrompt("Compute the area under y = xe^{-x} for x >= 0");
                            addLog("CONSOLE: Preset selected (Hard improper integral problem).");
                          }}
                          className="px-2.5 py-1 rounded bg-elevated hover:bg-surface border border-default text-text-secondary hover:text-white transition-colors"
                        >
                          Calculus: {"∫ xe^{-x} dx"}
                        </button>
                      </div>

                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={customPrompt}
                          onChange={(e) => setCustomPrompt(e.target.value)}
                          placeholder="Type a math reasoning problem..."
                          className="flex-1 bg-void border border-default p-3 rounded text-sm text-text-primary focus:outline-none focus:ring-1 focus:ring-nvidia placeholder:text-text-muted"
                        />
                        
                        <Button
                          onClick={handleSolvePrompt}
                          disabled={solverStatus === "solving" || !customPrompt.trim()}
                          className="bg-nvidia text-text-inverse font-display text-xs font-semibold h-11.5 px-6 border-0 shadow-[0_0_15px_rgba(118,185,0,0.2)]"
                        >
                          {solverStatus === "solving" ? "Solving..." : "Solve with ATRD"}
                        </Button>
                      </div>

                    </div>

                    {/* Step-by-Step Solver Results */}
                    {solverSteps.length > 0 && (
                      <div className="flex flex-col gap-4 border-t border-default/40 pt-4 animate-fade-up">
                        <div className="grid grid-cols-2 gap-3 max-w-sm">
                          <div className="bg-void/50 p-2.5 rounded border border-default flex flex-col gap-0.5">
                            <span className="font-sans text-[10px] text-text-muted uppercase">Latency</span>
                            <span className="font-mono text-sm font-bold text-cyan">{(solverLatency / 1000).toFixed(2)}s</span>
                          </div>
                          <div className="bg-void/50 p-2.5 rounded border border-default flex flex-col gap-0.5">
                            <span className="font-sans text-[10px] text-text-muted uppercase">Tokens Used</span>
                            <span className="font-mono text-sm font-bold text-nvidia">{solverTokens}</span>
                          </div>
                        </div>

                        <ReasoningTrace steps={solverSteps} />
                      </div>
                    )}

                  </div>

                </div>
              )}

              {/* SHARED CODE BLOCK OR FILE VIEWER FALLBACK (For SFT/GRPO/Intro files) */}
              {activePhaseId !== "p4" && (
                <div className="flex flex-col gap-2">
                  <span className="font-display text-xs font-semibold text-text-secondary uppercase tracking-wider pl-1">
                    Implementation Code File Preview
                  </span>
                  <CodeBlock 
                    code={fileCodes[activeFile] || ""} 
                    filename={activeFile}
                    language={activeFile.endsWith(".json") ? "json" : "python"}
                  />
                </div>
              )}

              {/* TELEMETRY CONSOLE PANEL (Fixed bottom scroll) */}
              <div className="glass-panel rounded-lg overflow-hidden flex flex-col mt-auto bg-surface/30">
                <div className="flex items-center justify-between px-4 py-2 border-b border-default bg-surface/85">
                  <div className="flex items-center gap-2">
                    <Terminal className="h-4 w-4 text-text-muted shrink-0" />
                    <span className="font-display text-xs font-semibold text-text-secondary uppercase tracking-wider">
                      Telemetry logs stream
                    </span>
                  </div>
                  <button 
                    onClick={() => {
                      setLogLogs([]);
                      addLog("CONSOLE: Log console cleared.");
                    }}
                    className="font-mono text-[10px] text-text-muted hover:text-white transition-colors"
                  >
                    Clear Logs
                  </button>
                </div>
                <div className="p-4 bg-void/70 font-mono text-[11px] text-text-secondary flex flex-col gap-1.5 min-h-[140px] max-h-[180px] overflow-y-auto scrollbar-thin">
                  {logLogs.map((log, idx) => (
                    <div key={idx} className="flex gap-2 leading-relaxed">
                      <span className="text-text-muted select-none">[{idx + 1}]</span>
                      <span className={cn(
                        log.includes("SYSTEM:") && "text-nvidia",
                        log.includes("INFERENCE:") && "text-cyan",
                        log.includes("TELEMETRY:") && "text-purple",
                        log.includes("ERROR:") && "text-rose",
                        log.includes("VALIDATOR:") && "text-cyan",
                        log.includes("SUBMITTER:") && "text-nvidia"
                      )}>
                        {log}
                      </span>
                    </div>
                  ))}
                  <div ref={logEndRef} />
                </div>
              </div>

            </main>

            {/* RIGHT SIDEBAR: Standing Leaderboard + Model Params Telemetry */}
            <aside className="p-4 flex flex-col gap-6 bg-surface/20">
              <div className="flex flex-col gap-2">
                <span className="font-display text-xs font-semibold text-text-secondary uppercase tracking-wider pl-1">
                  Competition Ranking
                </span>
                <LeaderboardBadge rank={32} score={0.942} />
              </div>

              <div className="flex flex-col gap-2 flex-1">
                <span className="font-display text-xs font-semibold text-text-secondary uppercase tracking-wider pl-1">
                  Active Parameters
                </span>
                <div className="glass-panel p-4 rounded-lg flex flex-col gap-4 bg-surface/30">
                  <div className="flex flex-col gap-1.5">
                    <span className="font-sans text-xs text-text-secondary">Base Model Name</span>
                    <span className="font-mono text-xs font-semibold text-text-primary bg-void px-2.5 py-1.5 rounded border border-default truncate">
                      Nemotron-3-Nano-30B
                    </span>
                  </div>
                  <div className="flex flex-col gap-1.5">
                    <span className="font-sans text-xs text-text-secondary">Quantization Bits</span>
                    <span className="font-mono text-xs font-semibold text-text-primary bg-void px-2.5 py-1.5 rounded border border-default">
                      4-Bit (NF4)
                    </span>
                  </div>
                  <div className="flex flex-col gap-1.5">
                    <span className="font-sans text-xs text-text-secondary">Temperature (Inference)</span>
                    <span className="font-mono text-xs font-semibold text-text-primary bg-void px-2.5 py-1.5 rounded border border-default">
                      0.0 (Deterministic)
                    </span>
                  </div>
                  <div className="flex flex-col gap-1.5">
                    <span className="font-sans text-xs text-text-secondary">Active LoRA Rank</span>
                    <span className="font-mono text-xs font-semibold text-text-primary bg-void px-2.5 py-1.5 rounded border border-default">
                      32 (Max Allowed)
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex flex-col gap-2">
                <span className="font-display text-xs font-semibold text-text-secondary uppercase tracking-wider pl-1">
                  Estimated Latency & Budget
                </span>
                <div className="grid grid-cols-2 gap-2">
                  <div className="glass-panel p-3 rounded-lg flex flex-col gap-1 text-center bg-surface/40">
                    <span className="font-sans text-[10px] text-text-muted uppercase">Latency</span>
                    <span className="font-mono text-base font-bold text-cyan">{getDynamicLatency()}s</span>
                  </div>
                  <div className="glass-panel p-3 rounded-lg flex flex-col gap-1 text-center bg-surface/40">
                    <span className="font-sans text-[10px] text-text-muted uppercase">Tokens</span>
                    <span className="font-mono text-base font-bold text-nvidia">{getDynamicTokens()}</span>
                  </div>
                </div>
              </div>
            </aside>

          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* SUBMISSION / PACKAGING PORTAL */}
      {/* ========================================================================= */}
      {currentScreen === "submission" && (
        <div className="flex-1 flex flex-col items-center justify-center px-4 relative">
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(118,185,0,0.1),transparent)] pointer-events-none" />
          
          <div className="max-w-2xl w-full glass-panel p-8 rounded-xl flex flex-col gap-6 relative z-10 animate-fade-up bg-surface/50">
            
            <div className="flex items-center gap-3 pb-4 border-b border-default">
              <UploadCloud className="h-6 w-6 text-nvidia shrink-0" />
              <div className="flex flex-col">
                <h2 className="font-display text-lg font-bold text-white">ATRD Submission Packaging</h2>
                <span className="font-sans text-xs text-text-secondary">Bundling weights for official challenge checklist verification</span>
              </div>
            </div>

            {/* Packaging status checks */}
            {submissionStatus !== "success" ? (
              <div className="flex flex-col gap-6">
                <div className="flex flex-col gap-2">
                  <div className="flex items-center justify-between text-xs font-mono text-text-secondary">
                    <span>{submissionStatus === "packaging" ? "Creating submission.zip archive..." : "Running official constraint checks..."}</span>
                    <span className="text-nvidia font-bold">{submissionProgress}%</span>
                  </div>
                  
                  {/* Progress bar */}
                  <div className="h-2 w-full bg-void rounded-full overflow-hidden border border-default">
                    <div className="h-full bg-nvidia transition-all duration-300 ease-out" style={{ width: `${submissionProgress}%` }} />
                  </div>
                </div>

                <div className="flex flex-col gap-2.5">
                  <div className="flex items-center justify-between text-xs font-mono p-2 bg-void/50 rounded border border-default">
                    <span className="text-text-secondary">Check LoRA Rank constraint (rank &le; 32)</span>
                    <span className={cn(
                      "font-bold uppercase",
                      submissionProgress >= 40 ? "text-nvidia" : "text-text-muted"
                    )}>
                      {submissionProgress >= 40 ? "PASSED ✅" : "PENDING"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs font-mono p-2 bg-void/50 rounded border border-default">
                    <span className="text-text-secondary">Verify adapter_config.json presence</span>
                    <span className={cn(
                      "font-bold uppercase",
                      submissionProgress >= 60 ? "text-nvidia" : "text-text-muted"
                    )}>
                      {submissionProgress >= 60 ? "PASSED ✅" : "PENDING"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs font-mono p-2 bg-void/50 rounded border border-default">
                    <span className="text-text-secondary">Verify vLLM template compatibility</span>
                    <span className={cn(
                      "font-bold uppercase",
                      submissionProgress >= 80 ? "text-nvidia" : "text-text-muted"
                    )}>
                      {submissionProgress >= 80 ? "PASSED ✅" : "PENDING"}
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col gap-6 animate-fade-up">
                
                {/* Success Card */}
                <div className="p-4 bg-nvidia/10 border border-nvidia/30 text-nvidia text-xs rounded-lg flex flex-col gap-3 font-mono">
                  <span className="text-sm font-bold block">✓ SUBMISSION PACKAGE GENERATED SUCCESSFULLY!</span>
                  <p className="text-text-secondary leading-relaxed">
                    The package <strong>submission.zip</strong> is verified, compliant with all rank and quantization constraints, 
                    and is ready to be uploaded to the private evaluation backend.
                  </p>
                </div>

                {/* Score & Trophy breakdown */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-void/50 p-4 rounded-lg border border-default flex flex-col gap-1 items-center justify-center">
                    <Trophy className="h-6 w-6 text-nvidia shadow-glow-nvidia mb-1" />
                    <span className="text-[10px] text-text-muted uppercase font-display font-semibold">Final Leaderboard Rank</span>
                    <span className="font-mono text-lg font-bold text-white">#32 Standing</span>
                  </div>
                  <div className="bg-void/50 p-4 rounded-lg border border-default flex flex-col gap-1 items-center justify-center">
                    <Zap className="h-6 w-6 text-amber mb-1" />
                    <span className="text-[10px] text-text-muted uppercase font-display font-semibold">Private Accuracy Score</span>
                    <span className="font-mono text-lg font-bold text-white">94.2%</span>
                  </div>
                </div>

                {/* Award nominations */}
                <div className="flex flex-col gap-2">
                  <span className="font-display text-[10px] font-bold text-text-secondary uppercase tracking-wider pl-1">Award nominations</span>
                  <div className="flex flex-wrap gap-2">
                    <span className="bg-nvidia/20 text-nvidia border border-nvidia/30 text-[10px] font-mono font-semibold px-2.5 py-1 rounded-full">
                      ★ Best Data Method
                    </span>
                    <span className="bg-purple/20 text-purple border border-purple/30 text-[10px] font-mono font-semibold px-2.5 py-1 rounded-full">
                      ★ Best RL Method
                    </span>
                    <span className="bg-cyan/20 text-cyan border border-cyan/30 text-[10px] font-mono font-semibold px-2.5 py-1 rounded-full">
                      ★ Best Fine-Tuning Method
                    </span>
                  </div>
                </div>

                <div className="flex gap-4 border-t border-default/40 pt-4 mt-2">
                  <Button
                    onClick={() => {
                      setCurrentScreen("workspace");
                      addLog("WORKSPACE: Returned to development environment.");
                    }}
                    className="flex-1 border border-default text-text-secondary bg-surface/40 hover:text-white font-display text-xs font-semibold h-10 hover:bg-surface"
                  >
                    Return to Workspace
                  </Button>
                  <Button
                    onClick={() => {
                      setCurrentScreen("welcome");
                      addLog("SYSTEM: Reset workspace workspace session.");
                    }}
                    className="flex-1 bg-nvidia text-text-inverse font-display text-xs font-semibold h-10 border-0 shadow-[0_0_15px_rgba(118,185,0,0.2)]"
                  >
                    Exit Application
                  </Button>
                </div>

              </div>
            )}

          </div>
        </div>
      )}

    </div>
  );
}
