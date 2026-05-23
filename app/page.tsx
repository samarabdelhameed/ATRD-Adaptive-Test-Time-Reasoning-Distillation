"use client";

import React, { useState, useEffect } from "react";
import {
  Brain,
  FileCode,
  Folder,
  Terminal,
  Activity
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
  // State for interactive features
  const [activePhaseId, setActivePhaseId] = useState<string>("p4");
  const [budgetValue, setBudgetValue] = useState<number>(4096);
  const [logLogs, setLogLogs] = useState<string[]>([
    "SYSTEM: Base model nvidia/NVIDIA-Nemotron-3-Nano-30B initialized.",
    "TELEMETRY: Accuracy score evaluated at 94.2% matching validation run #14.",
    "LOADER: SFT Adapter checkpoints parsed successfully.",
    "INFERENCE: Budget forcing engine active. Standing by for API prompts...",
    "SYSTEM: Memory usage stable at 82% of RTX PRO 6000 bounds.",
  ]);
  const [activeFile, setActiveFile] = useState<string>("src/inference/budget_forcer.py");

  // Telemetry phase definitions
  const phases: Phase[] = [
    {
      id: "p1",
      name: "Data Curation",
      description: "Baseline evaluation & synthetic correction generation.",
      status: "complete",
    },
    {
      id: "p2",
      name: "SFT Fine-Tuning",
      description: "Instruction SFT to establish reasoning formats.",
      status: "complete",
    },
    {
      id: "p3",
      name: "GRPO RL Training",
      description: "Reinforcement learning driven by math & format rewards.",
      status: "complete",
    },
    {
      id: "p4",
      name: "Budget Forcing & TTA",
      description: "Adaptive compute scaling at test-time.",
      status: "active",
    },
  ];

  // Failure categories for the heatmap
  const failureCategories: FailureCategory[] = [
    {
      id: "err1",
      name: "Reasoning Loop",
      count: 142,
      rate: 0.12,
      description: "Model gets stuck repeating mathematical assertions without progressing.",
    },
    {
      id: "err2",
      name: "Format Violation",
      count: 210,
      rate: 0.18,
      description: "Completion lacks required \\boxed{} answer blocks or think tags.",
    },
    {
      id: "err3",
      name: "Early Termination",
      count: 72,
      rate: 0.06,
      description: "Model halts generation before arriving at a final mathematical value.",
    },
    {
      id: "err4",
      name: "Calculation Error",
      count: 521,
      rate: 0.44,
      description: "Arithmetic errors inside intermediate reasoning traces.",
    },
  ];

  // Dynamic simulation values based on the slider budget
  const getDynamicLatency = () => {
    return ((budgetValue / 7680) * 14.5 + 2.1).toFixed(1);
  };

  const getDynamicTokens = () => {
    return Math.round(budgetValue * 0.96);
  };

  const getDynamicTraceSteps = (): ReasoningStep[] => {
    if (budgetValue < 1000) {
      return [
        {
          id: "s1",
          title: "Synthesize Problem Inputs",
          content: "Given: Solve the integral of f(x) = x^2 within range [0, 3].\nInput is simple. Allocating min_tokens budget (256 tokens).",
          type: "thinking",
          durationMs: 400,
          tokenCount: 45,
        },
        {
          id: "s2",
          title: "Direct Solver Engine",
          content: "Apply power rule: ∫ x^2 dx = x^3 / 3.\nEvaluate at boundaries: F(3) - F(0) = (3^3 / 3) - 0 = 27 / 3 = 9.",
          type: "assertion",
          durationMs: 800,
          tokenCount: 90,
        },
        {
          id: "s3",
          title: "Output Structure",
          content: "Final answer is clean. Format block initialized.\nResult: \\boxed{9}",
          type: "conclusion",
          durationMs: 300,
          tokenCount: 30,
        },
      ];
    } else if (budgetValue < 5000) {
      return [
        {
          id: "s1",
          title: "Parse Mathematical Constraints",
          content: "Given: ∫ cos^2(x) dx from 0 to π.\nAnalyzing difficulty: trigonometric identity required. Mid-tier budget (~4,000 tokens) active.",
          type: "thinking",
          durationMs: 1200,
          tokenCount: 140,
        },
        {
          id: "s2",
          title: "Trigonometric Identity Reformulation",
          content: "Recall: cos^2(x) = (1 + cos(2x)) / 2.\nIntegrand rewritten as: ∫ (1/2 + cos(2x)/2) dx.",
          type: "assertion",
          durationMs: 2100,
          tokenCount: 220,
        },
        {
          id: "s3",
          title: "Integration Execution",
          content: "∫ (1/2) dx = x/2\n∫ (cos(2x)/2) dx = sin(2x)/4\nAntiderivative F(x) = x/2 + sin(2x)/4.",
          type: "assertion",
          durationMs: 1800,
          tokenCount: 310,
        },
        {
          id: "s4",
          title: "Boundary Evaluation",
          content: "Evaluate at π: F(π) = π/2 + sin(2π)/4 = π/2 + 0 = π/2.\nEvaluate at 0: F(0) = 0 + sin(0)/4 = 0.\nDifference: F(π) - F(0) = π/2.",
          type: "conclusion",
          durationMs: 900,
          tokenCount: 120,
        },
        {
          id: "s5",
          title: "Format Validation",
          content: "Formatting final output block using math environment notation.\nResult: \\boxed{\\frac{\\pi}{2}}",
          type: "conclusion",
          durationMs: 400,
          tokenCount: 50,
        },
      ];
    } else {
      return [
        {
          id: "s1",
          title: "Multi-Step Boundary Reasoning",
          content: "Given: Find the area under y = xe^{-x} for x >= 0.\nAnalyzing difficulty: improper integral with exponential scaling. Max budget mode (7,680 tokens) engaged.",
          type: "thinking",
          durationMs: 2400,
          tokenCount: 380,
        },
        {
          id: "s2",
          title: "Integration by Parts Selection",
          content: "Formula: ∫ u dv = uv - ∫ v du\nLet u = x  =>  du = dx\nLet dv = e^{-x} dx  =>  v = -e^{-x}",
          type: "assertion",
          durationMs: 3100,
          tokenCount: 450,
        },
        {
          id: "s3",
          title: "Analytical Step Expansion",
          content: "Applying boundary terms:\n∫ xe^{-x} dx = -xe^{-x} - ∫ -e^{-x} dx\n= -xe^{-x} - e^{-x} + C\n= -e^{-x}(x + 1) + C",
          type: "assertion",
          durationMs: 3500,
          tokenCount: 510,
        },
        {
          id: "s4",
          title: "Error Correction Trigger",
          content: "Wait, verify sign values of antiderivative:\nd/dx [-e^{-x}(x + 1)] = e^{-x}(x + 1) - e^{-x} = xe^{-x} + e^{-x} - e^{-x} = xe^{-x}.\nAssertion validated. Moving to boundary limits.",
          type: "correction",
          durationMs: 1900,
          tokenCount: 280,
        },
        {
          id: "s5",
          title: "Improper Limits Resolution",
          content: "Lower limit x=0: -e^{-0}(0 + 1) = -1.\nUpper limit x->∞: lim_{x->∞} -(x+1)/e^x = 0 (via L'Hopital's rule).\nResult is F(∞) - F(0) = 0 - (-1) = 1.",
          type: "conclusion",
          durationMs: 2200,
          tokenCount: 340,
        },
        {
          id: "s6",
          title: "Formatting & Checksum Match",
          content: "Box format verified. Area matches positive limits.\nResult: \\boxed{1}",
          type: "conclusion",
          durationMs: 800,
          tokenCount: 95,
        },
      ];
    }
  };

  // Log simulation feed
  useEffect(() => {
    const interval = setInterval(() => {
      const liveUpdates = [
        `INFERENCE: Allocated ${budgetValue} token budget for math validation sub-tasks.`,
        `TELEMETRY: Latency estimated at ${((budgetValue / 7680) * 14.5 + 2.1).toFixed(1)}s per prompt evaluation.`,
        "SYSTEM: GRPO agent policy update iteration #402 complete.",
        "MONITOR: Log trace verified. Format compliance checks passed (100%).",
      ];
      const randomMsg = liveUpdates[Math.floor(Math.random() * liveUpdates.length)];
      setLogLogs((prev) => [...prev.slice(-6), `${new Date().toLocaleTimeString()} - ${randomMsg}`]);
    }, 6000);

    return () => clearInterval(interval);
  }, [budgetValue]);

  // Code snippets mapping for files
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
        format_rewards = [0.2 if "\\boxed{" in c else 0.0 for c in completions]
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
    <div className="flex-1 flex flex-col min-h-screen bg-void text-text-primary">
      {/* 1. Navbar */}
      <header className="sticky top-0 z-20 h-16 w-full glass-panel border-b border-default flex items-center justify-between px-6 bg-void/80 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-nvidia/10 border border-nvidia/30 text-nvidia glow-nvidia">
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
            <NeuralPulse status="active" />
            <span className="font-mono text-text-secondary text-[11px]">
              Engine Status: <strong className="text-cyan uppercase">Evaluating</strong>
            </span>
          </div>

          <Button className="bg-gradient-to-r from-nvidia to-nvidia/80 hover:brightness-110 text-text-inverse font-display text-xs font-semibold px-4 h-9 shadow-[0_0_15px_rgba(118,185,0,0.3)] border-0">
            Submit Adapter
          </Button>
        </div>
      </header>

      {/* Main Grid Layout */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-[280px_1fr_320px] divide-x divide-default">
        
        {/* Left Sidebar */}
        <aside className="p-4 flex flex-col gap-6 bg-surface/20">
          <div className="flex flex-col gap-2">
            <span className="font-display text-xs font-semibold text-text-secondary uppercase tracking-wider pl-1">
              Pipeline Stages
            </span>
            <PhaseStepper
              phases={phases}
              activePhaseId={activePhaseId}
              onPhaseSelect={setActivePhaseId}
            />
          </div>

          <div className="flex flex-col gap-2 flex-1">
            <span className="font-display text-xs font-semibold text-text-secondary uppercase tracking-wider pl-1">
              Project Structure
            </span>
            <div className="glass-panel p-3 rounded-lg flex flex-col gap-2.5 text-xs font-mono">
              <div className="flex items-center gap-2 text-text-secondary hover:text-text-primary cursor-pointer transition-colors">
                <Folder className="h-4 w-4 text-nvidia" />
                <span>configs/</span>
              </div>
              <div 
                onClick={() => setActiveFile("configs/base_lora.json")}
                className={cn(
                  "flex items-center gap-2 pl-4 cursor-pointer hover:text-text-primary transition-colors",
                  activeFile === "configs/base_lora.json" ? "text-cyan" : "text-text-muted"
                )}
              >
                <FileCode className="h-3.5 w-3.5" />
                <span>base_lora.json</span>
              </div>

              <div className="flex items-center gap-2 text-text-secondary hover:text-text-primary cursor-pointer transition-colors mt-1">
                <Folder className="h-4 w-4 text-nvidia" />
                <span>src/</span>
              </div>
              <div 
                onClick={() => setActiveFile("src/inference/budget_forcer.py")}
                className={cn(
                  "flex items-center gap-2 pl-4 cursor-pointer hover:text-text-primary transition-colors",
                  activeFile === "src/inference/budget_forcer.py" ? "text-cyan" : "text-text-muted"
                )}
              >
                <FileCode className="h-3.5 w-3.5" />
                <span>budget_forcer.py</span>
              </div>
              <div 
                onClick={() => setActiveFile("src/training/grpo_trainer.py")}
                className={cn(
                  "flex items-center gap-2 pl-4 cursor-pointer hover:text-text-primary transition-colors",
                  activeFile === "src/training/grpo_trainer.py" ? "text-cyan" : "text-text-muted"
                )}
              >
                <FileCode className="h-3.5 w-3.5" />
                <span>grpo_trainer.py</span>
              </div>
            </div>
          </div>
        </aside>

        {/* Center Canvas */}
        <main className="p-6 flex flex-col gap-6 overflow-y-auto max-h-[calc(100vh-64px)] scrollbar-thin">
          <div className="flex items-center justify-between border-b border-default pb-4">
            <div className="flex flex-col gap-1">
              <h1 className="font-display text-2xl font-bold tracking-tight text-text-primary">
                {phases.find((p) => p.id === activePhaseId)?.name}
              </h1>
              <p className="font-sans text-xs text-text-secondary">
                {phases.find((p) => p.id === activePhaseId)?.description}
              </p>
            </div>
            <div className="flex items-center gap-1.5 px-3 py-1 rounded bg-elevated border border-default text-[10px] text-text-secondary font-mono">
              <Activity className="h-3.5 w-3.5 text-cyan" />
              <span>ACTIVE PHASE: 04</span>
            </div>
          </div>

          {/* Phase Content switcher */}
          {activePhaseId === "p4" ? (
            <div className="flex flex-col gap-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <BudgetGauge value={budgetValue} onChange={setBudgetValue} />
                <FailureHeatmap categories={failureCategories} />
              </div>

              <ReasoningTrace steps={getDynamicTraceSteps()} />
            </div>
          ) : (
            <div className="flex flex-col gap-6">
              {/* Telemetry charts or config code block fallback for SFT/GRPO/Data */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <MetricCard
                  title="Accuracy"
                  value={activePhaseId === "p3" ? "94.2%" : activePhaseId === "p2" ? "88.5%" : "72.0%"}
                  change={{ value: "+5.7%", trend: "up" }}
                  theme="nvidia"
                />
                <MetricCard
                  title="Format compliance"
                  value="100.0%"
                  change={{ value: "0.0%", trend: "neutral" }}
                  theme="cyan"
                />
                <MetricCard
                  title="GPU Memory Limit"
                  value="13.2 / 16 GB"
                  description="NF4 Quantization active"
                  theme="rose"
                />
              </div>

              <div className="flex flex-col gap-2">
                <span className="font-display text-xs font-semibold text-text-secondary uppercase tracking-wider">
                  Phase Implementation Script
                </span>
                <CodeBlock 
                  code={fileCodes[activeFile] || ""} 
                  filename={activeFile}
                  language={activeFile.endsWith(".json") ? "json" : "python"}
                />
              </div>
            </div>
          )}

          {/* Terminal / Logs Console */}
          <div className="glass-panel rounded-lg overflow-hidden flex flex-col">
            <div className="flex items-center gap-2 px-4 py-2 border-b border-default bg-surface/80">
              <Terminal className="h-4 w-4 text-text-muted" />
              <span className="font-display text-xs font-semibold text-text-secondary uppercase tracking-wider">
                Telemetry Logs Stream
              </span>
            </div>
            <div className="p-4 bg-void/60 font-mono text-[11px] text-text-secondary flex flex-col gap-1.5 min-h-[140px] max-h-[220px] overflow-y-auto">
              {logLogs.map((log, idx) => (
                <div key={idx} className="flex gap-2 leading-relaxed">
                  <span className="text-text-muted select-none">[{idx + 1}]</span>
                  <span className={cn(
                    log.includes("SYSTEM:") && "text-nvidia",
                    log.includes("INFERENCE:") && "text-cyan",
                    log.includes("TELEMETRY:") && "text-purple",
                    log.includes("ERROR:") && "text-rose"
                  )}>
                    {log}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </main>

        {/* Right Sidebar */}
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
            <div className="glass-panel p-4 rounded-lg flex flex-col gap-4">
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
              <div className="glass-panel p-3 rounded-lg flex flex-col gap-1 text-center">
                <span className="font-sans text-[10px] text-text-muted uppercase">Latency</span>
                <span className="font-mono text-base font-bold text-cyan">{getDynamicLatency()}s</span>
              </div>
              <div className="glass-panel p-3 rounded-lg flex flex-col gap-1 text-center">
                <span className="font-sans text-[10px] text-text-muted uppercase">Tokens</span>
                <span className="font-mono text-base font-bold text-nvidia">{getDynamicTokens()}</span>
              </div>
            </div>
          </div>
        </aside>

      </div>
    </div>
  );
}
