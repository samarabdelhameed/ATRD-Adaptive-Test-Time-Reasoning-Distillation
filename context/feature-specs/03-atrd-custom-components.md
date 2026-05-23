# 03 — ATRD Custom Telemetry Components Specification

## ATRD: Adaptive Test-Time Reasoning Distillation Dashboard

### 1. Purpose and Setup Order
This document defines the implementation details for all custom ATRD visualization widgets under `components/atrd/`. These components render telemetry data, reasoning traces, and pipeline state.

> [!IMPORTANT]
> Read `01-design-system.md` and `02-dashboard-layout.md` before implementing components.

---

## 2. Component: ReasoningTrace (`components/atrd/reasoning-trace.tsx`)

### 2.1 Interface
```typescript
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
```

### 2.2 Visual Structure
- **Header**: `Brain` icon (cyan) + "Reasoning Trace Evaluation" label
- **Timeline**: Vertical left border line with step indicator nodes
- **Step types** color-coded:
  - `thinking`: Cyan (`bg-cyan/10`, `border-cyan`, `text-cyan`)
  - `assertion`: Purple (`bg-purple/10`, `border-purple`, `text-purple`)
  - `correction`: Amber (`bg-amber/10`, `border-amber`, `text-amber`) with pulse animation
  - `conclusion`: NVIDIA Green (`bg-nvidia/10`, `border-nvidia`, `text-nvidia`)

### 2.3 Interactive Features
- Collapsible steps with `ChevronDown`/`ChevronRight` icons
- First step expanded by default
- Token count badge next to each step
- Duration displayed at bottom of expanded content
- Hover highlight on step rows

### 2.4 Animation
- Transition on expand/collapse: `transition-all duration-300`
- Correction steps: `animate-pulse` on the info icon
- Node indicator: `transition-all duration-300` for color shifts

---

## 3. Component: BudgetGauge (`components/atrd/budget-gauge.tsx`)

### 3.1 Interface
```typescript
interface BudgetGaugeProps {
  value: number; // 256 to 7680
  onChange?: (value: number) => void;
  className?: string;
}
```

### 3.2 Visual Structure
- **Header**: `Clock` icon (cyan) + "Compute Allocation Budget" label
- **Difficulty tier badge**: Dynamic color-coded label
  - Easy (≤ 1000): `text-nvidia`
  - Medium (1001–4500): `text-amber`
  - Hard (> 4500): `text-rose`
- **Token display**: Large monospace number with "tokens" label
- **Progress bar**: Gradient fill with dynamic width and color
- **Slider**: Hidden range input (256–7680, step 128) with opacity-0 overlay
- **Quick-select buttons**: 256 (Easy), 4096 (Medium), 7680 (Hard)

### 3.3 Behavior
- Percentage calculated: `((value - 256) / (7680 - 256)) * 100`
- Bar color transitions with tier changes
- If `onChange` not provided, slider interaction disabled

---

## 4. Component: FailureHeatmap (`components/atrd/failure-heatmap.tsx`)

### 4.1 Interface
```typescript
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
```

### 4.2 Visual Structure
- **Header**: `AlertTriangle` icon (rose) + "Failure Mode Distribution" label
- **Live Telemetry badge**: Rose-colored monospace badge
- **2-column grid** of failure category cards
- Each card shows:
  - Category name (truncated)
  - Count badge (rose/amber/nvidia depending on heat level)
  - Rate percentage (large font)
  - Heat bar (proportional fill)

### 4.3 Heat Level Mapping
| Heat % | Color | Usage |
|--------|-------|-------|
| < 30% | NVIDIA Green | Low frequency |
| 30–70% | Amber | Medium frequency |
| > 70% | Rose | High frequency (critical) |

### 4.4 Interactive Features
- Each card is a `TooltipTrigger` showing full description on hover
- `TooltipContent` with dark elevated background
- Hover background highlight: `bg-surface/70`

---

## 5. Component: PhaseStepper (`components/atrd/phase-stepper.tsx`)

### 5.1 Interface
```typescript
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
```

### 5.2 Visual Structure
- Vertical button list with 2px left border color-coded by status:
  - `complete`: NVIDIA Green border
  - `active`: Cyan border
  - `pending`: Muted text color border
- Icons per status:
  - Complete: `CheckCircle2` (nvidia)
  - Active: `Loader2` with spin animation (cyan)
  - Pending: `Circle` (muted)
- Phase number label: "PHASE 01" format in uppercase
- Phase name + description below

### 5.3 Interactive Features
- Clickable: `onPhaseSelect` callback
- Selected state: elevated background + cyan shadow + `translate-x-1`
- Hover: `hover:bg-elevated/40 hover:translate-x-0.5`

---

## 6. Component: MetricCard (`components/atrd/metric-card.tsx`)

### 6.1 Interface
```typescript
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
```

### 6.2 Visual Structure
- Glass panel card with 2px top accent bar colored by theme
- **Header row**: Title (uppercase, secondary) + optional icon
- **Value**: Large monospace font (`font-mono text-2xl font-semibold`)
- **Change indicator**: 
  - Up trend: `ArrowUpRight` + NVIDIA green text
  - Down trend: `ArrowDownRight` + rose text
  - Neutral: muted text
- **Description**: Muted text alongside change

### 6.3 Hover State
- `translate-y-[-2px]` lift effect
- Enhanced shadow
- Border color shift to theme color at 30% opacity

---

## 7. Component: NeuralPulse (`components/atrd/neural-pulse.tsx`)

### 7.1 Interface
```typescript
interface NeuralPulseProps {
  status?: "active" | "success" | "warning" | "error" | "idle";
  className?: string;
}
```

### 7.2 Visual States
| Status | Color | Effect |
|--------|-------|--------|
| `active` | Cyan (`bg-cyan`) | `animate-pulse` + ping ring + glow shadow |
| `success` | NVIDIA Green | Static glow |
| `warning` | Amber | `animate-pulse` + ping ring |
| `error` | Rose | Static glow |
| `idle` | Muted | No effects |

### 7.3 Structure
- Container: `h-3 w-3` relative positioned
- Ping ring: absolute positioned, full size, `animate-ping` (active/warning only)
- Core dot: `h-2.5 w-2.5` rounded-full with color and glow

---

## 8. Component: LeaderboardBadge (`components/atrd/leaderboard-badge.tsx`)

### 8.1 Interface
```typescript
interface LeaderboardBadgeProps {
  rank: number;
  score: number; // e.g. 0.942
  className?: string;
}
```

### 8.2 Visual Structure
- Glass panel with NVIDIA green border and glow shadow
- Gradient background: `from-nvidia/5 to-transparent`
- **Left side**: Trophy icon in circular container + "Current Rank" label + `#{rank} Standing`
- **Right side**: Vertical divider + `Zap` icon (amber) + "Accuracy Score" label + `{score * 100}%`

---

## 9. Component: CodeBlock (`components/atrd/code-block.tsx`)

### 9.1 Interface
```typescript
interface CodeBlockProps {
  code: string;
  language?: string;
  filename?: string;
  className?: string;
}
```

### 9.2 Visual Structure
- Glass panel with border
- **Header**: `FileCode` icon + filename + language badge (`bg-void rounded border font-mono text-[10px] uppercase`)
- **Copy button**: `Copy` icon → `Check` + NVIDIA green on success (2s timeout)
- **Code area**: Split into line numbers (right-aligned, muted, `border-r border-default/20`) + code content

### 9.3 Behavior
- Copy-to-clipboard via `navigator.clipboard.writeText()`
- 2-second visual confirmation after copy
- Horizontal scroll for long lines with scrollbar styling

---

## 10. Export Conventions

All components use named exports:
```typescript
export function ComponentName({ ... }: ComponentNameProps) { ... }
```

All components import `cn` from `@/lib/utils` and lucide-react icons.

---

## 11. Exit Quality Gate
Before moving to the next feature spec, verify:
- [ ] All 8 components render without TypeScript errors
- [ ] All props interfaces are exported
- [ ] Color tokens use CSS custom properties (no hardcoded hex in logic)
- [ ] Animations respect `prefers-reduced-motion` via Tailwind
- [ ] Components accept and apply `className` via `cn()`
- [ ] Tooltip usage in FailureHeatmap imports correctly from `@/components/ui/tooltip`
