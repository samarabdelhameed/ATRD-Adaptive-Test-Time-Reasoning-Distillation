import React, { useState } from "react";
import { cn } from "@/lib/utils";
import { ChevronDown, ChevronRight, Brain, CheckCircle, Info } from "lucide-react";

export interface ReasoningStep {
  id: string;
  title: string;
  content: string;
  type: "thinking" | "assertion" | "correction" | "conclusion";
  durationMs?: number;
  tokenCount?: number;
}

interface ReasoningTraceProps {
  steps: ReasoningStep[];
  className?: string;
}

export function ReasoningTrace({ steps, className }: ReasoningTraceProps) {
  const [expandedSteps, setExpandedSteps] = useState<Record<string, boolean>>(() => {
    if (steps && steps.length > 0) {
      return { [steps[0].id]: true };
    }
    return {};
  });

  const toggleStep = (id: string) => {
    setExpandedSteps((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  return (
    <div className={cn("flex flex-col gap-4 w-full", className)}>
      <div className="flex items-center gap-2 pb-2 border-b border-default">
        <Brain className="h-5 w-5 text-cyan" />
        <h3 className="font-display text-sm font-semibold tracking-wide uppercase text-text-primary">
          Reasoning Trace Evaluation
        </h3>
      </div>

      <div className="relative border-l border-default pl-4 ml-2 flex flex-col gap-6">
        {steps.map((step, idx) => {
          const isExpanded = !!expandedSteps[step.id];
          
          const iconColors = {
            thinking: "bg-cyan/10 border-cyan text-cyan",
            assertion: "bg-purple/10 border-purple text-purple",
            correction: "bg-amber/10 border-amber text-amber",
            conclusion: "bg-nvidia/10 border-nvidia text-nvidia",
          }[step.type];

          const lineColors = {
            thinking: "border-l-cyan",
            assertion: "border-l-purple",
            correction: "border-l-amber",
            conclusion: "border-l-nvidia",
          }[step.type];

          const Icon = {
            thinking: <Brain className="h-3.5 w-3.5" />,
            assertion: <Info className="h-3.5 w-3.5" />,
            correction: <Info className="h-3.5 w-3.5 animate-pulse" />,
            conclusion: <CheckCircle className="h-3.5 w-3.5" />,
          }[step.type];

          return (
            <div key={step.id} className="relative group">
              {/* Step indicator node */}
              <div
                className={cn(
                  "absolute -left-[25px] top-1 h-5 w-5 rounded-full border flex items-center justify-center transition-all duration-300",
                  iconColors
                )}
              >
                {Icon}
              </div>

              {/* Step body */}
              <div
                className={cn(
                  "border-l-2 pl-3 bg-surface/30 rounded-r-lg border border-default border-l-transparent transition-all duration-300",
                  isExpanded && lineColors + " bg-surface/50 border-default/40"
                )}
              >
                <button
                  onClick={() => toggleStep(step.id)}
                  className="w-full flex items-center justify-between p-3 focus:outline-none"
                >
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs text-text-muted">Step {idx + 1}</span>
                    <span className="font-sans text-sm font-semibold text-text-primary text-left">
                      {step.title}
                    </span>
                  </div>

                  <div className="flex items-center gap-3">
                    {step.tokenCount && (
                      <span className="font-mono text-[10px] text-text-muted bg-void border border-default px-1.5 py-0.5 rounded">
                        {step.tokenCount} tokens
                      </span>
                    )}
                    {isExpanded ? (
                      <ChevronDown className="h-4 w-4 text-text-muted" />
                    ) : (
                      <ChevronRight className="h-4 w-4 text-text-muted" />
                    )}
                  </div>
                </button>

                {isExpanded && (
                  <div className="p-3 pt-0 border-t border-default/20">
                    <pre className="font-mono text-xs text-text-secondary whitespace-pre-wrap leading-relaxed py-2 overflow-x-auto">
                      {step.content}
                    </pre>
                    {step.durationMs && (
                      <div className="flex justify-end pt-2 text-[10px] text-text-muted font-mono">
                        Latency: {step.durationMs} ms
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
