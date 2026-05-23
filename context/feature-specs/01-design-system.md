# 01 — Design System and UI Component Specification

## ATRD: Adaptive Test-Time Reasoning Distillation Dashboard

### 1. Purpose and Setup Order
This document defines the implementation steps and design system architecture for the ATRD Web Dashboard. All styling, component structure, and animations must align with `context/ui-context.md`.

> [!IMPORTANT]
> Read `context/ai-workflow-rules.md` and `context/ui-context.md` before starting any frontend implementation.

---

## 2. Environment Initialization

### 2.1 Shadcn/UI Installation
Install and configure `shadcn/ui` on top of Tailwind CSS v4 in the workspace:
- Initialize shadcn CLI.
- Do not modify any generated files in `components/ui/*` after installation.

### 2.2 Reusable Tailwind CSS Class Merger
Create `lib/utils.ts` to expose the `cn()` helper:
```typescript
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

### 2.3 Icon Library
Install `lucide-react` to handle telemetry icons.
- Ensure all icons use thin stroke outlines (`strokeWidth={1.5}`) as specified in `ui-context.md`.

---

## 3. UI Primitive Components (shadcn/ui)

Add these base primitives to `components/ui/`:
- `button` (Primary with NVIDIA green gradient, ghost with border, and rose danger variants)
- `card` (Glassmorphic cards with blur and inner borders)
- `dialog` (Backdrop-blurred modular overlays)
- `dropdown-menu` (Context menus for difficulty and model parameters)
- `input` (Form and search telemetry fields)
- `progress` (Budget forcing compute and token allocations)
- `scroll-area` (Code highlights and logs container)
- `separator` (Grid dividers using neural node SVG accents)
- `sheet` (Mobile drawers and details panels)
- `skeleton` (Loading screens for telemetry metrics)
- `tabs` (Phase navigation P1–P4 switcher)
- `tooltip` (Explanations for metrics and reward scores)

---

## 4. Theme System & Global Styles (`globals.css`)

Ensure all components match the dark technical workspace theme defined in `globals.css`:

```css
@theme {
  --color-void: #030303;
  --color-surface: #0A0A0F;
  --color-elevated: #111118;
  --color-nvidia: #76B900;
  --color-cyan: #00F0FF;
  --color-purple: #B829DD;
  --color-amber: #FFB800;
  --color-rose: #FF4D6D;
  
  --font-display: "Space Grotesk", sans-serif;
  --font-sans: "Inter", sans-serif;
  --font-mono: "JetBrains Mono", monospace;
}

/* Custom telemetry effects */
.glass-panel {
  background: rgba(17, 17, 24, 0.72);
  backdrop-filter: blur(20px) saturate(140%);
  border: 1px solid rgba(255, 255, 255, 0.06);
}

.glow-nvidia {
  box-shadow: 0 0 20px rgba(118, 185, 0, 0.25);
}
```

---

## 5. Custom Telemetry Components (`components/atrd/`)

Implement specialized visualization widgets under `components/atrd/`:

1. **`ReasoningTrace`** (`components/atrd/reasoning-trace.tsx`):
   - Displays vertical timeline of step-by-step thinking traces.
   - Highlights parser-delimited `<<thinking>>` structures.
2. **`BudgetGauge`** (`components/atrd/budget-gauge.tsx`):
   - Visual slider representing compute bounds (easy/medium/hard).
3. **`FailureHeatmap`** (`components/atrd/failure-heatmap.tsx`):
   - Grid monitoring baseline error types and frequencies.
4. **`PhaseStepper`** (`components/atrd/phase-stepper.tsx`):
   - Visual flow tracking the 4-phase pipeline (Data → SFT → GRPO → Budget Forcing).
5. **`MetricCard`** (`components/atrd/metric-card.tsx`):
   - High-fidelity numeric indicator displaying accuracy gains or loss curves.
6. **`NeuralPulse`** (`components/atrd/neural-pulse.tsx`):
   - Glowing orb depicting current engine execution states.

---

## 6. Dashboard Layout Blueprint

The workspace page layout utilizes a full-viewport split configuration:
- **Navbar (64px)**: Top navigation header displaying connection statuses and main actions.
- **Left Sidebar (280px)**: Collapsible navigator displaying current pipeline phases and files.
- **Center Canvas**: Scrollable container displaying active telemetry graphs, code editors, and trace evaluations.
- **Right Sidebar (320px)**: Dedicated parameters panel for real-time RL budget and temperature adjustments.
