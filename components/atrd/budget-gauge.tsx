import React from "react";
import { cn } from "@/lib/utils";
import { Clock } from "lucide-react";

interface BudgetGaugeProps {
  value: number; // 256 to 7680
  onChange?: (value: number) => void;
  className?: string;
}

export function BudgetGauge({ value, onChange, className }: BudgetGaugeProps) {
  // Calculate difficulty percentage based on value range [256, 7680]
  const percentage = Math.min(100, Math.max(0, ((value - 256) / (7680 - 256)) * 100));

  // Determine active difficulty tier
  let tier: "easy" | "medium" | "hard" = "easy";
  let color = "text-nvidia";
  let barBg = "bg-nvidia";

  if (value > 4500) {
    tier = "hard";
    color = "text-rose";
    barBg = "bg-rose";
  } else if (value > 1000) {
    tier = "medium";
    color = "text-amber";
    barBg = "bg-amber";
  }

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange?.(parseInt(e.target.value, 10));
  };

  return (
    <div className={cn("glass-panel p-5 rounded-lg flex flex-col gap-4", className)}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Clock className="h-4 w-4 text-cyan" />
          <span className="font-display text-xs font-semibold tracking-wider uppercase text-text-secondary">
            Compute Allocation Budget
          </span>
        </div>
        <span className={cn("font-display text-xs font-bold uppercase transition-colors duration-300", color)}>
          {tier} Mode
        </span>
      </div>

      <div className="flex flex-col gap-2 relative">
        <div className="flex justify-between items-baseline">
          <span className="font-mono text-xs text-text-muted">Max Tokens</span>
          <span className="font-mono text-xl font-bold text-text-primary">
            {value.toLocaleString()} <span className="text-xs font-normal text-text-muted">tokens</span>
          </span>
        </div>

        <div className="relative w-full py-1">
          <div className="relative w-full h-2 bg-void rounded-full overflow-hidden border border-default">
            <div
              className={cn("h-full transition-all duration-300 ease-out", barBg)}
              style={{ width: `${percentage}%` }}
            />
          </div>

          <input
            type="range"
            min="256"
            max="7680"
            step="128"
            value={value}
            onChange={handleSliderChange}
            className="w-full h-full bg-transparent appearance-none cursor-pointer focus:outline-none focus:ring-0 opacity-0 absolute top-0 left-0"
            style={{ pointerEvents: onChange ? "auto" : "none" }}
          />
        </div>
      </div>

      <div className="grid grid-cols-3 gap-1 font-mono text-[10px] text-text-muted text-center pt-1">
        <button
          onClick={() => onChange?.(256)}
          className={cn("hover:text-text-primary transition-colors", value === 256 && "text-nvidia font-bold")}
        >
          256 (Easy)
        </button>
        <button
          onClick={() => onChange?.(4096)}
          className={cn("hover:text-text-primary transition-colors", value === 4096 && "text-amber font-bold")}
        >
          4,096 (Medium)
        </button>
        <button
          onClick={() => onChange?.(7680)}
          className={cn("hover:text-text-primary transition-colors", value === 7680 && "text-rose font-bold")}
        >
          7,680 (Hard)
        </button>
      </div>
    </div>
  );
}
