import React from "react";
import { cn } from "@/lib/utils";
import { AlertTriangle } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

export interface FailureCategory {
  id: string;
  name: string;
  count: number;
  rate: number; // 0 to 1
  description: string;
}

interface FailureHeatmapProps {
  categories: FailureCategory[];
  className?: string;
}

export function FailureHeatmap({ categories, className }: FailureHeatmapProps) {
  // Find max rate for dynamic scaling
  const maxRate = Math.max(...categories.map((c) => c.rate), 0.1);

  return (
    <div className={cn("glass-panel p-5 rounded-lg flex flex-col gap-4", className)}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-rose" />
          <span className="font-display text-xs font-semibold tracking-wider uppercase text-text-secondary">
            Failure Mode Distribution
          </span>
        </div>
        <span className="font-mono text-xs text-rose font-bold">
          Live Telemetry
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {categories.map((cat) => {
          // Heat level mapping for background colors
          const heatPercent = (cat.rate / maxRate) * 100;
          let heatBg = "bg-rose/10 border-rose/20 text-rose";
          let rateBadge = "bg-rose/20 text-rose border-rose/30";

          if (heatPercent < 30) {
            heatBg = "bg-nvidia/10 border-nvidia/20 text-nvidia";
            rateBadge = "bg-nvidia/20 text-nvidia border-nvidia/30";
          } else if (heatPercent < 70) {
            heatBg = "bg-amber/10 border-amber/20 text-amber";
            rateBadge = "bg-amber/20 text-amber border-amber/30";
          }

          return (
            <Tooltip key={cat.id}>
              <TooltipTrigger>
                <div
                  className={cn(
                    "flex flex-col gap-2 p-3 rounded-lg border bg-surface/40 hover:bg-surface/70 cursor-help transition-all duration-300 text-left w-full",
                    heatBg
                  )}
                >
                  <div className="flex items-center justify-between w-full">
                    <span className="font-sans text-xs font-semibold truncate max-w-[120px]">
                      {cat.name}
                    </span>
                    <span className={cn("font-mono text-[10px] px-1.5 py-0.5 rounded border font-semibold", rateBadge)}>
                      {cat.count}
                    </span>
                  </div>
                  
                  <div className="flex items-baseline gap-1">
                    <span className="font-mono text-lg font-bold">
                      {(cat.rate * 100).toFixed(1)}%
                    </span>
                    <span className="font-sans text-[10px] text-text-muted">rate</span>
                  </div>

                  {/* Tiny heat bar */}
                  <div className="w-full h-1 bg-void/50 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-current transition-all duration-500"
                      style={{ width: `${heatPercent}%`, opacity: 0.8 }}
                    />
                  </div>
                </div>
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-[220px] bg-elevated border border-default p-2 text-xs text-text-secondary">
                <p className="font-semibold text-text-primary mb-1">{cat.name}</p>
                <p>{cat.description}</p>
              </TooltipContent>
            </Tooltip>
          );
        })}
      </div>
    </div>
  );
}
