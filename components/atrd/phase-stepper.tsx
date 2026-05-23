import React from "react";
import { cn } from "@/lib/utils";
import { CheckCircle2, Circle, Loader2 } from "lucide-react";

export interface Phase {
  id: string;
  name: string;
  description: string;
  status: "complete" | "active" | "pending";
}

interface PhaseStepperProps {
  phases: Phase[];
  activePhaseId?: string;
  onPhaseSelect?: (phaseId: string) => void;
  className?: string;
}

export function PhaseStepper({
  phases,
  activePhaseId,
  onPhaseSelect,
  className,
}: PhaseStepperProps) {
  return (
    <div className={cn("flex flex-col gap-3", className)}>
      {phases.map((phase, idx) => {
        const isSelected = activePhaseId === phase.id;
        const statusColors = {
          complete: "border-l-nvidia text-nvidia",
          active: "border-l-cyan text-cyan",
          pending: "border-l-text-muted text-text-secondary",
        };

        const Icon = {
          complete: <CheckCircle2 className="h-4 w-4 text-nvidia shrink-0" />,
          active: <Loader2 className="h-4 w-4 text-cyan animate-spin shrink-0" />,
          pending: <Circle className="h-4 w-4 text-text-muted shrink-0" />,
        }[phase.status];

        return (
          <button
            key={phase.id}
            onClick={() => onPhaseSelect?.(phase.id)}
            className={cn(
              "text-left p-4 rounded-r-lg border-l-2 bg-surface/50 border-y border-r border-default transition-all duration-300",
              statusColors[phase.status],
              isSelected
                ? "bg-elevated border-default/20 shadow-[0_4px_12px_rgba(0,240,255,0.05)] translate-x-1"
                : "hover:bg-elevated/40 hover:translate-x-0.5",
              "group"
            )}
          >
            <div className="flex items-start gap-3">
              <div className="mt-0.5">{Icon}</div>
              <div className="flex flex-col gap-0.5">
                <span className="font-display text-xs font-semibold tracking-wider uppercase text-text-muted">
                  PHASE 0{idx + 1}
                </span>
                <span className="font-sans text-sm font-semibold text-text-primary group-hover:text-white transition-colors duration-200">
                  {phase.name}
                </span>
                <span className="font-sans text-xs text-text-secondary">
                  {phase.description}
                </span>
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}
