import React from "react";
import { cn } from "@/lib/utils";
import { Trophy, Zap } from "lucide-react";

interface LeaderboardBadgeProps {
  rank: number;
  score: number; // e.g. 0.942
  className?: string;
}

export function LeaderboardBadge({ rank, score, className }: LeaderboardBadgeProps) {
  return (
    <div
      className={cn(
        "glass-panel px-4 py-3 rounded-lg flex items-center justify-between border border-nvidia/30 shadow-[0_0_20px_rgba(118,185,0,0.15)] bg-gradient-to-r from-nvidia/5 to-transparent",
        className
      )}
    >
      <div className="flex items-center gap-3">
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-nvidia/20 border border-nvidia/40 text-nvidia shadow-[0_0_10px_rgba(118,185,0,0.3)]">
          <Trophy className="h-4 w-4" />
        </div>
        <div className="flex flex-col">
          <span className="font-display text-[10px] font-semibold uppercase text-text-muted tracking-wider">
            Current Rank
          </span>
          <span className="font-mono text-sm font-bold text-text-primary">
            #{rank} <span className="text-[10px] font-normal text-nvidia">Standing</span>
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2 border-l border-default pl-4">
        <Zap className="h-4 w-4 text-amber" />
        <div className="flex flex-col">
          <span className="font-display text-[10px] font-semibold uppercase text-text-muted tracking-wider">
            Accuracy Score
          </span>
          <span className="font-mono text-sm font-bold text-text-primary">
            {(score * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
}
