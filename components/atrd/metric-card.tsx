import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { ArrowUpRight, ArrowDownRight } from "lucide-react";

interface MetricCardProps {
  title: string;
  value: string | number;
  description?: string;
  change?: {
    value: string;
    trend: "up" | "down" | "neutral";
  };
  icon?: React.ReactNode;
  theme?: "nvidia" | "cyan" | "purple" | "amber" | "rose";
  className?: string;
}

export function MetricCard({
  title,
  value,
  description,
  change,
  icon,
  theme = "nvidia",
  className,
}: MetricCardProps) {
  const borderColors = {
    nvidia: "border-t-nvidia hover:border-nvidia/30",
    cyan: "border-t-cyan hover:border-cyan/30",
    purple: "border-t-purple hover:border-purple/30",
    amber: "border-t-amber hover:border-amber/30",
    rose: "border-t-rose hover:border-rose/30",
  };

  const textColors = {
    nvidia: "text-nvidia",
    cyan: "text-cyan",
    purple: "text-purple",
    amber: "text-amber",
    rose: "text-rose",
  };

  return (
    <Card
      className={cn(
        "glass-panel border-t-2 transition-all duration-300 hover:translate-y-[-2px] hover:shadow-[0_8px_32px_rgba(0,0,0,0.5)]",
        borderColors[theme],
        className
      )}
    >
      <CardContent className="p-5 flex flex-col justify-between h-full gap-2">
        <div className="flex items-center justify-between">
          <span className="font-display text-xs font-medium tracking-wider text-text-secondary uppercase">
            {title}
          </span>
          {icon && <div className={cn("opacity-70", textColors[theme])}>{icon}</div>}
        </div>

        <div className="flex flex-col gap-1">
          <span className="font-mono text-2xl font-semibold tracking-tight text-text-primary">
            {value}
          </span>

          {(change || description) && (
            <div className="flex items-center gap-1.5 text-xs">
              {change && (
                <span
                  className={cn(
                    "flex items-center font-medium",
                    change.trend === "up"
                      ? "text-nvidia"
                      : change.trend === "down"
                      ? "text-rose"
                      : "text-text-muted"
                  )}
                >
                  {change.trend === "up" && <ArrowUpRight className="h-3 w-3 mr-0.5" />}
                  {change.trend === "down" && <ArrowDownRight className="h-3 w-3 mr-0.5" />}
                  {change.value}
                </span>
              )}
              {description && (
                <span className="text-text-muted font-sans">{description}</span>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
