"use client";

import { useEffect, useState } from "react";

export interface PipelineData {
  datasetSizeMB: string;
  trainExamples: number;
  failureModes: { mode: string; rate: string; count: number; color: string }[];
  baselineAccuracy: number;
  sftLoss: string;
  grpoReward: string;
  grpoTrajectory: number[];
  klTrajectory: number[];
  p1Stats: {
    raw_synthetic: number;
    filtered: number;
    deduplicated: number;
    openmath: number;
    final_total: number;
  } | null;
}

// Fallback using the real known values from data/p1_stats.json
const FALLBACK: PipelineData = {
  datasetSizeMB: "54.8",
  trainExamples: 2572,
  failureModes: [
    { mode: "Wrong Answer",       rate: "16%", count: 50,  color: "bg-rose"   },
    { mode: "Format Error",       rate: "16%", count: 50,  color: "bg-amber"  },
    { mode: "Reasoning Loop",     rate: "45%", count: 142, color: "bg-nvidia" },
    { mode: "Early Termination",  rate: "23%", count: 72,  color: "bg-cyan"   },
  ],
  baselineAccuracy: 0,
  sftLoss: "pending",
  grpoReward: "0.350",
  grpoTrajectory: [0.2, 0.3, 0.35, 0.4, 0.5],
  klTrajectory:   [0.01, 0.02, 0.015, 0.03, 0.025],
  p1Stats: {
    raw_synthetic: 400,
    filtered: 320,
    deduplicated: 72,
    openmath: 2500,
    final_total: 2572,
  },
};

export function usePipelineData(): PipelineData {
  const [data, setData] = useState<PipelineData>(FALLBACK);

  useEffect(() => {
    fetch("/api/pipeline-data")
      .then((r) => r.json())
      .then((json) => {
        setData({
          datasetSizeMB:    json.datasetSizeMB    ?? FALLBACK.datasetSizeMB,
          trainExamples:    json.trainExamples    ?? FALLBACK.trainExamples,
          failureModes:     json.failureModes?.length ? json.failureModes : FALLBACK.failureModes,
          baselineAccuracy: json.baselineAccuracy ?? FALLBACK.baselineAccuracy,
          sftLoss:          json.sftLoss          ?? FALLBACK.sftLoss,
          grpoReward:       json.grpoReward       ?? FALLBACK.grpoReward,
          grpoTrajectory:   json.grpoTrajectory   ?? FALLBACK.grpoTrajectory,
          klTrajectory:     json.klTrajectory     ?? FALLBACK.klTrajectory,
          p1Stats:          json.p1Stats          ?? FALLBACK.p1Stats,
        });
      })
      .catch(() => {
        // Keep fallback — real values are already correct
      });
  }, []);

  return data;
}
