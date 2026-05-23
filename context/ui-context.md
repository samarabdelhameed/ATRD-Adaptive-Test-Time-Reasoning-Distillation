# UI Context — Adaptive Test-Time Reasoning Distillation (ATRD)

## Theme

**Dark only. No light mode.**

The design language is a **dark technical workspace** — near-black backgrounds, layered glass surfaces, and vivid accent colors for interactive elements. The aesthetic fuses:

- **NVIDIA industrial precision** — acid greens, carbon blacks, telemetry-grade clarity
- **Research lab minimalism** — asymmetric grids, generous whitespace, editorial typography
- **Neural topology** — glassmorphism layers, ambient particle fields, signal-flow animations

The interface reads as a **high-performance reasoning dashboard**: cold, precise, alive. Every surface suggests depth — data planes floating in a void, separated by frosted glass, illuminated by status signals. It is not "dark mode"; it is a spatial compute environment.

---

## Colors

All color tokens defined as CSS custom properties. No hardcoded hex values permitted in component code.

### Background Hierarchy

| Role | CSS Variable | Value | Usage |
|------|-------------|-------|-------|
| Page background | `--bg-base` | `#030303` | Deepest canvas — body, empty states |
| Surface | `--bg-surface` | `#0A0A0F` | Cards, panels, code blocks |
| Elevated | `--bg-elevated` | `#111118` | Hover states, dropdowns, tooltips |
| Glass | `--bg-glass` | `rgba(17, 17, 24, 0.72)` | Frosted overlays, modals, floating panels |
| Input | `--bg-input` | `#0D0D14` | Form fields, search bars |

### Text Hierarchy

| Role | CSS Variable | Value | Opacity | Usage |
|------|-------------|-------|---------|-------|
| Primary text | `--text-primary` | `#F0F0F5` | 100% | Headings, body, labels |
| Secondary text | `--text-secondary` | `#A0A0B0` | 72% | Descriptions, metadata |
| Muted text | `--text-muted` | `#606070` | 48% | Timestamps, file paths, placeholders |
| Inverse text | `--text-inverse` | `#030303` | 100% | Text on accent buttons |

### Accent Spectrum (Neural Signals)

| Role | CSS Variable | Value | Usage |
|------|-------------|-------|-------|
| Primary accent | `--accent-primary` | `#76B900` | NVIDIA green — CTA, active states, success, thinking complete |
| Cyan signal | `--accent-cyan` | `#00F0FF` | Reasoning traces, data flow, thinking tokens |
| Purple signal | `--accent-purple` | `#B829DD` | RL/GRPO, reward signals, experimental features |
| Amber signal | `--accent-amber` | `#FFB800` | Budget limits, warnings, attention gates |
| Rose signal | `--accent-rose` | `#FF4D6D` | Failures, error modes, negative reward |

### State Colors

| Role | CSS Variable | Value | Usage |
|------|-------------|-------|-------|
| Success | `--state-success` | `#76B900` | Correct answers, completed phases, positive reward |
| Error | `--state-error` | `#FF4D6D` | Failures, OOM, incorrect answers, validation errors |
| Warning | `--state-warning` | `#FFB800` | Budget exhaustion, rate limits, pending review |
| Info | `--state-info` | `#00F0FF` | Tips, reasoning steps, active processing |

### Border & Divider

| Role | CSS Variable | Value | Usage |
|------|-------------|-------|-------|
| Default border | `--border-default` | `rgba(255, 255, 255, 0.06)` | Card outlines, separators |
| Hover border | `--border-hover` | `rgba(118, 185, 0, 0.3)` | Card hover glow, active focus rings |
| Active border | `--border-active` | `rgba(0, 240, 255, 0.4)` | Selected items, input focus |
| Error border | `--border-error` | `rgba(255, 77, 109, 0.5)` | Invalid fields, failed tests |

### Gradient Tokens

| Role | CSS Variable | Value |
|------|-------------|-------|
| Neural gradient | `--gradient-neural` | `linear-gradient(135deg, #76B900 0%, #00F0FF 50%, #B829DD 100%)` |
| Heat gradient | `--gradient-heat` | `linear-gradient(90deg, #76B900 0%, #FFB800 50%, #FF4D6D 100%)` |
| Edge glow | `--gradient-edge` | `linear-gradient(180deg, rgba(118,185,0,0.3) 0%, rgba(0,240,255,0.1) 100%)` |
| Glass shine | `--gradient-shine` | `linear-gradient(180deg, rgba(255,255,255,0.08) 0%, transparent 100%)` |

---

## Typography

### Font Stack

| Role | Font | Variable | Fallback |
|------|------|----------|----------|
| UI text / Display | **Space Grotesk** | `--font-sans` | `Inter, system-ui, -apple-system, sans-serif` |
| Body / Interface | **Inter** | `--font-body` | `system-ui, -apple-system, sans-serif` |
| Code / Mono | **JetBrains Mono** | `--font-mono` | `Fira Code, Menlo, Monaco, Consolas, monospace` |
| Accent / Labels | **Space Grotesk** | `--font-display` | `Inter, system-ui, sans-serif` |

*Note: Space Grotesk is used for headings, labels, and any uppercase tracking-wide text. Inter is used for body copy and interface text. JetBrains Mono is mandatory for all code, logs, metrics, and reasoning traces.*

### Type Scale

| Token | Size | Line Height | Letter Spacing | Weight | Usage |
|-------|------|-------------|----------------|--------|-------|
| `text-hero` | `4.5rem` (72px) | `1.05` | `-0.02em` | 700 | Page title, competition name |
| `text-h1` | `3rem` (48px) | `1.1` | `-0.015em` | 700 | Section headers |
| `text-h2` | `2.25rem` (36px) | `1.2` | `-0.01em` | 600 | Phase titles, card headers |
| `text-h3` | `1.5rem` (24px) | `1.3` | `0` | 600 | Sub-sections, metric labels |
| `text-body` | `1rem` (16px) | `1.6` | `0` | 400 | Paragraphs, descriptions |
| `text-small` | `0.875rem` (14px) | `1.5` | `0.01em` | 400 | Captions, metadata, timestamps |
| `text-xs` | `0.75rem` (12px) | `1.4` | `0.04em` | 500 | Tags, badges, labels (uppercase) |
| `text-mono` | `0.8125rem` (13px) | `1.5` | `0` | 400 | Code blocks, logs, paths |

### Typography Patterns

- **Headings**: Tight line-height, negative tracking. Display headings (`text-hero`, `text-h1`) receive a subtle atmospheric glow: `text-shadow: 0 0 40px rgba(118, 185, 0, 0.12)`.
- **Labels / Tags**: Always uppercase, `letter-spacing: 0.08em`, `font-size: 0.75rem`, `font-weight: 500`, semi-transparent (`--text-secondary`).
- **Code / Data**: Always `--font-mono`. Syntax highlighting uses the accent spectrum. Line numbers in `--text-muted`.
- **Metrics**: Large monospace numbers (`text-h2` or `text-h3`) with small uppercase labels beneath.

---

## Border Radius

| Context | Class | Value | Usage |
|---------|-------|-------|-------|
| Inline / small UI | `rounded-sm` | `6px` | Buttons, tags, badges, inputs |
| Cards / panels | `rounded-lg` | `16px` | Primary cards, metric panels, code blocks |
| Modals / overlays | `rounded-xl` | `20px` | Dialogs, drawers, toasts |
| Pills / capsules | `rounded-full` | `9999px` | Status badges, filter chips |
| Large surfaces | `rounded-2xl` | `24px` | Hero cards, feature sections |

*All radius values use a 4px grid base. No arbitrary radius values permitted.*

---

## Component Library

**shadcn/ui** on top of **Tailwind CSS v4**.

- Components live in `components/ui/`.
- Use the shadcn CLI to add new components: `npx shadcn add [component]`.
- Never write primitive components from scratch — extend shadcn primitives with the ATRD theme tokens.
- Custom components (reasoning trace, budget gauge, failure heatmap) live in `components/atrd/`.

### Required shadcn Components

| Component | Usage |
|-----------|-------|
| `button` | Primary, ghost, danger variants |
| `card` | Base for metric cards, phase cards, log cards |
| `dialog` | Modal overlays for detail views |
| `dropdown-menu` | Context menus, filter selectors |
| `input` | Search, configuration fields |
| `progress` | Budget forcing gauge, training progress |
| `scroll-area` | Code blocks, log viewers |
| `separator` | Section dividers |
| `sheet` | Mobile drawers, side panels |
| `skeleton` | Loading states for cards and metrics |
| `tabs` | Phase navigation, ablation views |
| `tooltip` | Icon explanations, metric definitions |

### Custom ATRD Components (to be built)

| Component | Path | Description |
|-----------|------|-------------|
| `ReasoningTrace` | `components/atrd/reasoning-trace.tsx` | Vertical timeline of thinking steps with expand/collapse |
| `BudgetGauge` | `components/atrd/budget-gauge.tsx` | Adaptive compute allocation slider with color morphing |
| `FailureHeatmap` | `components/atrd/failure-heatmap.tsx` | Grid visualization of failure mode frequencies |
| `PhaseStepper` | `components/atrd/phase-stepper.tsx` | P1→P4 vertical stepper with status animations |
| `MetricCard` | `components/atrd/metric-card.tsx` | Monospace number + label + sparkline |
| `LeaderboardBadge` | `components/atrd/leaderboard-badge.tsx` | Rank display with glow and award icons |
| `CodeBlock` | `components/atrd/code-block.tsx` | Syntax-highlighted block with line numbers and copy |
| `NeuralPulse` | `components/atrd/neural-pulse.tsx` | Status orb with active processing animation |

---

## Layout Patterns

### Editor / Dashboard Layout

Full-viewport split layout for the main competition dashboard:

```
┌─────────────────────────────────────────────────────────────┐
│  Navbar (fixed, 64px, bottom border)                      │
├──────────┬──────────────────────────────┬─────────────────┤
│          │                              │                 │
│  Left    │       Center Canvas          │   Right         │
│  Sidebar │       (scrollable content)     │   Sidebar       │
│  (280px) │                              │   (320px)       │
│          │                              │                 │
│  Phase   │  ┌────────────────────────┐  │  Metrics        │
│  Nav     │  │  Reasoning Trace       │  │  Panel          │
│  +       │  │  / Code / Viz          │  │  +              │
│  File    │  └────────────────────────┘  │  Budget         │
│  Tree    │                              │  Control        │
│          │                              │                 │
└──────────┴──────────────────────────────┴─────────────────┘
```

- **Left sidebar**: Fixed `280px`, collapsible to `72px` icon rail on tablet. Contains phase navigation (P1–P4) and file tree.
- **Center canvas**: Fluid, scrollable. Contains primary content — reasoning traces, code blocks, visualizations.
- **Right sidebar**: Fixed `320px`, collapsible to drawer on mobile. Contains live metrics, budget gauge, and training telemetry.
- **Navbar**: Fixed `64px`, full width. Logo, competition status, user actions.

### Sidebars

- **Fixed width** with `1px` right border (`--border-default`).
- **Collapsible**: Left sidebar has a collapse toggle. Right sidebar collapses to a floating action button on mobile.
- **Scrollable**: Independent scroll containers with custom thin scrollbar (4px width, `--accent-primary` thumb, `--bg-elevated` track).

### Modals / Overlays

- **Centered overlay** with `backdrop-blur-xl` (`blur(24px)`) and `bg-black/60`.
- **Modal container**: `bg-glass`, `rounded-xl` (20px), `border-default`, max-width `640px` (md) or `512px` (sm).
- **Entrance animation**: `scale(0.96) opacity-0` → `scale(1) opacity-100`, 200ms, `ease-spring`.
- **Exit animation**: Reverse, 150ms.

### Navbar

- **Height**: `64px` fixed.
- **Background**: `bg-glass` with `backdrop-blur-md` and bottom border (`--border-default`).
- **Left**: Logo mark + "ATRD" wordmark (Space Grotesk, 700).
- **Center**: Competition status badge ("Training", "Evaluating", "Complete") with `NeuralPulse` orb.
- **Right**: User avatar, settings gear, submit button (primary accent).

### Cards

**Base Card**:
- Background: `bg-glass`
- Backdrop filter: `blur(20px) saturate(140%)`
- Border: `1px solid --border-default`
- Border radius: `rounded-lg` (16px)
- Padding: `24px` (space-6)
- Shadow: `0 4px 24px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.04)`

**Hover State**:
- Border: `1px solid --border-hover`
- Shadow: `0 8px 32px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(118, 185, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.08)`
- Transform: `translateY(-2px)`
- Transition: `all 0.4s cubic-bezier(0.16, 1, 0.3, 1)`

**Variants**:
- **Metric Card**: `rounded-lg`, top accent bar (4px solid `--accent-primary`), monospace numbers.
- **Phase Card**: Left border (2px) colored by phase status. Complete = `--state-success`, Active = `--accent-cyan`, Pending = `--text-muted`.
- **Log Card**: `bg-surface`, no blur, monospace text, timestamp in `--text-muted`.

### Buttons

**Primary**:
- Background: `linear-gradient(135deg, --accent-primary, #5A8F00)`
- Color: `--text-inverse`
- Font: `--font-sans`, 600 weight
- Padding: `12px 24px`
- Radius: `rounded-sm` (6px)
- Shadow: `0 0 20px rgba(118, 185, 0, 0.25)`
- Hover: `brightness(1.1)`, shadow expands to `0 0 30px rgba(118, 185, 0, 0.4)`

**Ghost**:
- Background: transparent
- Border: `1px solid rgba(255, 255, 255, 0.12)`
- Color: `--text-secondary`
- Hover: border-color `--border-active`, color `--accent-cyan`, subtle glow

**Danger**:
- Background: `rgba(255, 77, 109, 0.1)`
- Border: `1px solid --accent-rose`
- Color: `--accent-rose`

---

## Icons

**Lucide React** exclusively.

- **Style**: Stroke-based only. No filled icons.
- **Stroke width**: `1.5px` (thin, precise, technical).
- **Sizes**:
  - `h-4 w-4` (16px): Inline text, tags, table cells
  - `h-5 w-5` (20px): Buttons, nav items, list items
  - `h-6 w-6` (24px): Standalone feature icons, empty states
  - `h-8 w-8` (32px): Hero section icons, large CTAs

### Icon Mapping

| Concept | Icon | Color Token |
|---------|------|-------------|
| Reasoning / Thinking | `Brain` | `--accent-cyan` |
| Dataset / Data | `Database` | `--text-secondary` |
| Training / Model | `Cpu` | `--accent-primary` |
| RL / Reward | `TrendingUp` | `--accent-purple` |
| Budget / Time | `Clock` | `--accent-amber` |
| Success / Correct | `CheckCircle2` | `--state-success` |
| Failure / Error | `XCircle` | `--state-error` |
| Submission / Export | `UploadCloud` | `--accent-cyan` |
| Notebook / Code | `FileCode` | `--text-secondary` |
| Settings | `Settings` | `--text-muted` |
| Expand / Collapse | `ChevronDown` / `ChevronRight` | `--text-secondary` |
| Copy | `Copy` | `--text-muted` |
| External Link | `ExternalLink` | `--accent-cyan` |
| Warning | `AlertTriangle` | `--state-warning` |
| Info | `Info` | `--state-info` |
| Spark / Award | `Zap` | `--accent-amber` |
| Leaderboard | `Trophy` | `--accent-primary` |
| Phase Complete | `CheckCircle` | `--state-success` |
| Phase Pending | `Circle` | `--text-muted` |
| Phase Active | `Loader2` (spinning) | `--accent-cyan` |

---

## Spacing System

Based on **4px grid** with exponential growth.

| Token | Value | Tailwind | Usage |
|-------|-------|----------|-------|
| `space-1` | `4px` | `p-1` / `gap-1` | Micro gaps, icon padding |
| `space-2` | `8px` | `p-2` / `gap-2` | Inline spacing, tight groups |
| `space-3` | `12px` | `p-3` / `gap-3` | Button padding-y, tag gaps |
| `space-4` | `16px` | `p-4` / `gap-4` | Card internal padding |
| `space-6` | `24px` | `p-6` / `gap-6` | Component gaps |
| `space-8` | `32px` | `p-8` / `gap-8` | Section sub-groups |
| `space-12` | `48px` | `p-12` / `gap-12` | Card external margins |
| `space-16` | `64px` | `p-16` / `gap-16` | Section breaks |
| `space-24` | `96px` | `p-24` / `gap-24` | Major section separators |
| `space-32` | `128px` | `p-32` / `gap-32` | Hero breathing room |

---

## Animation & Motion Tokens

### Easing Curves

| Name | Value | Usage |
|------|-------|-------|
| `ease-spring` | `cubic-bezier(0.16, 1, 0.3, 1)` | Card entrances, layout shifts, modals |
| `ease-smooth` | `cubic-bezier(0.4, 0, 0.2, 1)` | Color transitions, opacity, hover states |
| `ease-bounce` | `cubic-bezier(0.34, 1.56, 0.64, 1)` | Success states, metric pops, badges |
| `ease-linear` | `linear` | Progress bars, thinking tokens, loading |

### Duration Scale

| Token | Value | Usage |
|-------|-------|-------|
| `duration-instant` | `100ms` | Micro-interactions, checkbox toggles |
| `duration-fast` | `200ms` | Hover states, color shifts |
| `duration-normal` | `300ms` | Button presses, dropdowns |
| `duration-slow` | `400ms` | Card entrances, page transitions |
| `duration-dramatic` | `600ms` | Hero elements, major reveals |

### Key Animations

| Animation | Description | CSS |
|-----------|-------------|-----|
| `fade-up` | Staggered entrance | `opacity 0 → 1`, `translateY(20px) → 0`, `scale(0.98) → 1` |
| `neural-pulse` | Active processing | `box-shadow` ring expansion loop |
| `budget-flow` | Compute allocation | `width` fill with gradient shift |
| `float` | Ambient drift | `translateY` + `rotate` organic movement |
| `type-reveal` | Code reveal | `clip-path` inset wipe |
| `spin-slow` | Loading states | `rotate` 360° over 2s |

---

## Responsive Breakpoints

| Name | Width | Tailwind Prefix | Layout Behavior |
|------|-------|-----------------|-----------------|
| `xs` | `< 480px` | `xs:` | Single column, stacked, full-bleed cards, bottom drawer nav |
| `sm` | `480–767px` | `sm:` | 2-column grid where applicable, reduced padding |
| `md` | `768–1023px` | `md:` | 4-column grid, sidebars collapse to overlays |
| `lg` | `1024–1279px` | `lg:` | 8-column grid, full layout restored |
| `xl` | `≥ 1280px` | `xl:` | 12-column grid, maximum expression |
| `2xl` | `≥ 1536px` | `2xl:` | Ultra-wide spacing, larger type scale |

### Mobile Adaptations

- **Navigation**: Bottom sheet drawer (not top hamburger).
- **Cards**: Full-width, single column, `space-4` padding.
- **Typography**: `text-hero` scales to `2.5rem`, `text-h1` to `2rem`.
- **Animations**: Reduce particle count by 60%, disable cursor spotlight.
- **Code blocks**: Horizontal scroll with fade edge indicators, font-size `0.75rem`.
- **Touch**: Buttons min-height `48px`, active state uses `scale(0.98)` instead of hover glow.

---

## Z-Index Architecture

| Layer | Z-Index | Content |
|-------|---------|---------|
| Background | `-10` | Particle canvas, grid overlays, ambient effects |
| Content | `0` | Text, cards, images, primary interface |
| Floating | `10` | Sticky headers, toasts, floating action buttons |
| Overlay | `20` | Modals, drawers, dropdowns, tooltips |
| Glow | `30` | Cursor effects, spotlight, highest attention |

---

## Accessibility & Performance

### Accessibility

- **Reduced motion**: Respect `prefers-reduced-motion: reduce`. Disable all animations, instant transitions.
- **Color independence**: Never use color alone to indicate state. Always pair with icon + text.
- **Contrast ratios**: All text meets WCAG AA (4.5:1). `--text-primary` on `--bg-base` = 18.5:1.
- **Focus rings**: `2px solid --accent-cyan` with `2px` offset on all interactive elements.

### Performance

- **GPU acceleration**: Use `will-change: transform` on animated cards, remove after animation completes.
- **Particle canvas**: Throttle to `30fps` on mobile. Use `requestAnimationFrame`.
- **Font loading**: `font-display: swap` for all custom fonts. System fallbacks render immediately.
- **Glassmorphism fallback**: On browsers without `backdrop-filter`, fall back to solid `--bg-elevated`.

---

## Tailwind Configuration Reference

```javascript
// tailwind.config.js
module.exports = {
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        void: '#030303',
        surface: '#0A0A0F',
        elevated: '#111118',
        nvidia: '#76B900',
        cyan: '#00F0FF',
        purple: '#B829DD',
        amber: '#FFB800',
        rose: '#FF4D6D',
        'text-primary': '#F0F0F5',
        'text-secondary': '#A0A0B0',
        'text-muted': '#606070',
      },
      fontFamily: {
        display: ['Space Grotesk', 'Inter', 'system-ui'],
        sans: ['Inter', 'system-ui'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'fade-up': 'fadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) backwards',
        'neural-pulse': 'neuralPulse 2s ease-in-out infinite',
        'budget-flow': 'budgetFlow 1.5s ease-out forwards',
        'float': 'float 8s ease-in-out infinite',
        'spin-slow': 'spin 2s linear infinite',
      },
      keyframes: {
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(20px) scale(0.98)' },
          '100%': { opacity: '1', transform: 'translateY(0) scale(1)' },
        },
        neuralPulse: {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(118, 185, 0, 0.4)' },
          '50%': { boxShadow: '0 0 0 8px rgba(118, 185, 0, 0)' },
        },
        budgetFlow: {
          '0%': { width: '0%' },
          '100%': { width: '100%' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px) rotate(0deg)' },
          '33%': { transform: 'translateY(-12px) rotate(1deg)' },
          '66%': { transform: 'translateY(8px) rotate(-1deg)' },
        },
      },
      boxShadow: {
        'card': '0 4px 24px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.04)',
        'card-hover': '0 8px 32px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(118, 185, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.08)',
        'glow-nvidia': '0 0 20px rgba(118, 185, 0, 0.25)',
        'glow-cyan': '0 0 20px rgba(0, 240, 255, 0.25)',
        'glow-rose': '0 0 20px rgba(255, 77, 109, 0.25)',
      },
      borderRadius: {
        'sm': '6px',
        'lg': '16px',
        'xl': '20px',
        '2xl': '24px',
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
};
```

---

## Asset Guidelines

### Decorative Elements

- **Neural node SVGs**: Small circles (4–8px) with connecting bezier curves, used as section dividers. Stroke: `1px`, color `--text-muted`.
- **Grid corners**: L-shaped brackets at card corners (1px, `--text-muted`) for "technical blueprint" feel.
- **Status orbs**: 8px circles with `neural-pulse` animation, placed next to active process names.
- **Particle field**: 30–50 dots (2–4px), `--accent-cyan` and `--accent-primary`, slow drift, connecting lines within 100px.

### Logo & Branding

- **Logo mark**: Abstract neural node icon (3 connected circles) in `--accent-primary`.
- **Wordmark**: "ATRD" in Space Grotesk, 700 weight, `--text-primary`.
- **Subtitle**: "Adaptive Test-Time Reasoning Distillation" in `--font-sans`, 500 weight, `--text-secondary`, uppercase, tracking-wide.

---

*Document version: 1.0 — Aligned with ATRD Project Specification*
*Last updated: 2026-05-23*
