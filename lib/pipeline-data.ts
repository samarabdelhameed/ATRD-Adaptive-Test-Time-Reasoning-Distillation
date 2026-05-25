"use client";

import { useEffect, useState } from "react";

export interface PipelineData {
  datasetSizeMB: string;
  trainExamples: number;
  failureModes: { mode: string; rate: string; count: number; color: string }[];
  baselineAccuracy: number;
  sftLoss: string;
  grpoReward: string;
}

const FAILURE_COLORS = [
  "bg-rose",
  "bg-amber",
  "bg-nvidia",
  "bg-cyan",
  "bg-purple",
  "bg-orange",
];

const RAW_SYNTHETIC = 400;
const FILTERED = 320;
const DEDUPLICATED = 72;
const OPENMATH = 2500;
const FINAL_TOTAL = 2572;

const FAILURE_MODE_DATA = [
  { mode: "wrong_answer", count: 50, color: "bg-rose" },
  { mode: "format_error", count: 50, color: "bg-amber" },
  { mode: "reasoning_loop", count: 142, color: "bg-nvidia" },
  { mode: "early_termination", count: 72, color: "bg-cyan" },
];

const DEFAULT_DATA: PipelineData = {
  datasetSizeMB: (57428812 / (1024 * 1024)).toFixed(1),
  trainExamples: FINAL_TOTAL,
  failureModes: FAILURE_MODE_DATA.map((f) => {
    const total = FAILURE_MODE_DATA.reduce((s, x) => s + x.count, 0);
    return {
      mode: f.mode.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
      rate: `${((f.count / total) * 100).toFixed(0)}%`,
      count: f.count,
      color: f.color,
    };
  }),
  baselineAccuracy: 0,
  sftLoss: "pending",
  grpoReward: "pending",
};

export function usePipelineData(): PipelineData {
  const [data] = useState<PipelineData>(DEFAULT_DATA);
  return data;
}
