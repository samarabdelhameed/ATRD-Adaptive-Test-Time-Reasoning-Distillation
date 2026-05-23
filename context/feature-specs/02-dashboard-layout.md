# 02 — Dashboard Layout Specification

## ATRD: Adaptive Test-Time Reasoning Distillation Dashboard

### 1. Purpose and Setup Order
This document defines the main workspace layout structure for the ATRD Web Dashboard. The layout uses a full-viewport split configuration with three columns and a fixed top navigation bar.

> [!IMPORTANT]
> Read `01-design-system.md` and `context/ui-context.md` before implementing this layout.

---

## 2. Layout Architecture

### 2.1 Viewport Structure
The root layout occupies the full viewport height (`min-h-screen`) with a vertical flex column:

```
┌─────────────────────────────────────────────────────────────┐
│  Navbar (fixed, 64px, glass-panel, z-20)                   │
├──────────┬──────────────────────────────┬──────────────────┤
│          │                              │                  │
│  Left    │       Center Canvas          │   Right          │
│  Sidebar │       (scrollable content)   │   Sidebar        │
│  (280px) │                              │   (320px)        │
│          │                              │                  │
│  │ Phase  │  │ Reasoning Trace │        │  │ Metrics       │
│  │ Nav    │  │ / Code / Viz    │        │  │ Panel         │
│  │ +      │  │                 │        │  │ +             │
│  │ Files  │  │                 │        │  │ Budget        │
│  └────────┘  └─────────────────┘        │  │ Control       │
└──────────┴──────────────────────────────┴──────────────────┘
```

### 2.2 Grid Definition
The main content area uses a CSS grid with three responsive columns:

```typescript
// Layout grid classes
<div className="flex-1 grid grid-cols-1 lg:grid-cols-[280px_1fr_320px] divide-x divide-default">
```

- **Mobile (< 1024px)**: Single column, sidebars collapse to sheets/drawers.
- **Desktop (≥ 1024px)**: Three-column split with fixed-width sidebars.

---

## 3. Navbar (Fixed, 64px)

### 3.1 Structure
- **Height**: `64px` (`h-16`)
- **Position**: `sticky top-0 z-20`
- **Background**: `glass-panel` with `bg-void/80 backdrop-blur-md`
- **Bottom Border**: `border-b border-default`

### 3.2 Left Section
- Logo mark: `Brain` icon from lucide-react in NVIDIA green container
- Wordmark: "ATRD Dashboard" in `font-display text-sm font-bold`
- Subtitle: "Adaptive Test-Time Reasoning Distillation" in `font-sans text-[10px] text-text-secondary uppercase tracking-wider`

### 3.3 Center Section
- Engine status badge: `NeuralPulse` orb with status text
- Background: `bg-surface border-default rounded-full`
- Text: `font-mono text-[11px] text-text-secondary`

### 3.4 Right Section
- "Submit Adapter" primary button: `bg-gradient-to-r from-nvidia to-nvidia/80` with glow shadow

---

## 4. Left Sidebar (280px)

### 4.1 Properties
- **Width**: `280px` (`lg:grid-cols-[280px_1fr_320px]`)
- **Background**: `bg-surface/20`
- **Padding**: `p-4`
- **Border Right**: via grid `divide-x divide-default`

### 4.2 Pipeline Stages Section
- Header label: uppercase `font-display text-xs font-semibold text-text-secondary` with tracking-wider
- `PhaseStepper` component showing P1–P4 status
- Clickable phases navigate the center canvas content

### 4.3 Project Structure Section
- Header label: uppercase
- File tree with `Folder` and `FileCode` icons
- Clickable files update the center canvas `CodeBlock`
- Selected file highlighted with `text-cyan`

---

## 5. Center Canvas

### 5.1 Properties
- **Padding**: `p-6`
- **Overflow**: `overflow-y-auto max-h-[calc(100vh-64px)]`
- **Scrollbar**: Custom thin scrollbar

### 5.2 Phase Header
- Phase name: `font-display text-2xl font-bold`
- Phase description: `font-sans text-xs text-text-secondary`
- Active phase badge: `bg-elevated border border-default font-mono`
- Top border separator: `border-b border-default pb-4`

### 5.3 Phase Content (Phase-Dependent)
- **P4 (Budget Forcing)**: `BudgetGauge` + `FailureHeatmap` grid + `ReasoningTrace`
- **P1–P3**: `MetricCard` grid (Accuracy, Format Compliance, GPU Memory) + `CodeBlock` for active file

### 5.4 Telemetry Logs Console
- Terminal-styled panel at the bottom
- Header with `Terminal` icon and "Telemetry Logs Stream" label
- Scrolling log output with color-coded entries:
  - `SYSTEM:` → `text-nvidia`
  - `INFERENCE:` → `text-cyan`
  - `TELEMETRY:` → `text-purple`
  - `ERROR:` → `text-rose`
- Live log feed simulation via `useEffect` interval

---

## 6. Right Sidebar (320px)

### 6.1 Properties
- **Width**: `320px`
- **Background**: `bg-surface/20`
- **Padding**: `p-4`

### 6.2 Competition Ranking Section
- `LeaderboardBadge` component with rank number and accuracy score
- NVIDIA green glow border and gradient background

### 6.3 Active Parameters Section
- Glass panel with 4 parameter rows:
  - Base Model Name (Nemotron-3-Nano-30B)
  - Quantization Bits (4-Bit NF4)
  - Temperature (0.0 Deterministic)
  - Active LoRA Rank (32 Max Allowed)
- Each row: label `font-sans text-xs text-text-secondary`, value `font-mono text-xs font-semibold` in `bg-void border border-default rounded`

### 6.4 Latency & Budget Section
- 2-column grid with latency (seconds) and token count
- Values dynamically computed from budget slider

---

## 7. Responsive Behavior

| Breakpoint | Width | Layout |
|-----------|-------|--------|
| `xs` | < 480px | Single column, sidebars as bottom sheets |
| `sm` | 480–767px | Single column, compact padding |
| `md` | 768–1023px | Sidebars collapse to overlay drawers |
| `lg` | 1024–1279px | Full 3-column layout |
| `xl` | ≥ 1280px | Full layout with max-width constraints |

---

## 8. Exit Quality Gate
Before moving to the next feature spec, verify:
- [ ] Layout renders without layout shift or overflow
- [ ] Sidebars have correct fixed widths (280px / 320px)
- [ ] Navbar is sticky and backdrop blur works
- [ ] Center canvas scrolls independently
- [ ] Grid collapses to single column on mobile viewport
- [ ] Log console auto-scrolls with new entries
