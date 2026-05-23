import React from "react";
import { cn } from "@/lib/utils";

interface NeuralPulseProps {
  status?: "active" | "success" | "warning" | "error" | "idle";
  className?: string;
}

export function NeuralPulse({ status = "idle", className }: NeuralPulseProps) {
  const colorMap = {
    active: "bg-cyan shadow-[0_0_12px_rgba(0,240,255,0.6)] animate-pulse",
    success: "bg-nvidia shadow-[0_0_12px_rgba(118,185,0,0.6)]",
    warning: "bg-amber shadow-[0_0_12px_rgba(255,184,0,0.6)] animate-pulse",
    error: "bg-rose shadow-[0_0_12px_rgba(255,77,109,0.6)]",
    idle: "bg-text-muted",
  };

  return (
    <div className={cn("relative flex items-center justify-center h-3 w-3", className)}>
      {status === "active" && (
        <span className="absolute inline-flex h-full w-full rounded-full bg-cyan opacity-75 animate-ping" />
      )}
      {status === "warning" && (
        <span className="absolute inline-flex h-full w-full rounded-full bg-amber opacity-75 animate-ping" />
      )}
      <span className={cn("relative inline-flex rounded-full h-2.5 w-2.5 transition-colors duration-300", colorMap[status])} />
    </div>
  );
}
