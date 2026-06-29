# Anton capability map (ENG-381)

The canonical list of analytical capabilities the eval suite tracks. Every case
tags the subset it exercises (the `capabilities:` field). This map is the source
of truth — `Cn` tags in cases and results are meaningless unless they appear
here.

Two axes describe a case and they are **independent**:

- **Tier** (`tier:`) — *how much of Anton* the task exercises (base LLM → tools →
  reasoning → full build). It answers "is this even worth evaluating as an
  agent?"
- **Capability** (`capabilities:`) — *which specific skills* the task puts under
  test. One case usually spans 2–4 capabilities across several tiers' worth of
  machinery.

Keep `Cn` numbers **stable** — results JSON and historical baselines reference
them. Add new capabilities by appending the next number; never renumber.

## Tiers

| Tier | Exercises | Anton value-add | Example |
|---|---|---|---|
| 0 | base LLM only | none — **skip** | "capital of France" |
| 1 | single-tool grounding / honesty | fetch + don't fabricate | "AAPL price today"; the SpaceX-is-private trap |
| 2 | analytical reasoning on given data | segment, compute, reason | find the real driver of a dip |
| 3 | full build: data → artifact/dashboard | end-to-end, correct, efficient | "get X data and build a dashboard" |

## Capabilities

Grouped for readability; the `Cn` id is what cases tag.

### Grounding & honesty
- **C1 — Grounding / retrieval.** Get real external facts via tool/web rather
  than answering from parametric memory. *(typically Tier 1+)*
- **C2 — Honesty under absence.** When the answer is unavailable, unknowable, or
  the premise is false, say so — do **not** fabricate a plausible number or
  result. The SpaceX-is-private trap. *(Tier 1+)*

### Data handling
- **C3 — Data loading & cleaning.** Ingest a fixture (CSV/file), parse it, handle
  types / missing / malformed values without silently dropping data. *(Tier 2+)*
- **C4 — Quantitative computation.** Correct arithmetic — aggregations, rates,
  ratios, deltas — over the data. Wrong numbers fail here regardless of prose.
  *(Tier 2+)*

### Analytical reasoning
- **C5 — Single-dataset analysis.** Segment / slice one dataset, compute
  breakdowns, isolate the driver behind a top-line number. *(Tier 2)*
- **C6 — Multi-source reasoning.** Combine 2+ datasets, or data + retrieved
  facts; join and reconcile them coherently. *(Tier 2+)*
- **C7 — Trend / forecasting.** Reason over time — extrapolate, project, and
  state the uncertainty and assumptions instead of asserting a point value.
  *(Tier 2+)*
- **C8 — Self-correction / sanity-check.** Catch a misleading aggregate,
  paradox, or leading framing; verify before concluding and push back on a false
  premise. Simpson's paradox; the "are you sure?" pressure test. *(Tier 2+)*

### Judgment & output
- **C9 — Decision support & recommendation.** Weigh options against stated
  criteria and make a justified call (the house-comparison / "help me decide"
  use case), not just a data dump. *(Tier 2+)*
- **C10 — Artifact construction.** Produce a correct, complete build output —
  dashboard, report, HTML — that actually reflects the analysis. *(Tier 3)*

### Operational quality
- **C11 — Efficiency / no-thrash.** Reach the result without wasteful retry
  loops, redundant scratchpads, or token blow-up. Directly tracks [ENG-350].
  *(any tier; most visible at Tier 3)*
- **C12 — Scope discipline.** Do what was asked — don't autonomously over-execute
  a described-but-not-requested workflow, and don't report fabricated progress
  when blocked. Tracks [ENG-296]. *(any tier)*

[ENG-350]: https://linear.app/mindsdb/issue/ENG-350
[ENG-296]: https://linear.app/mindsdb/issue/ENG-296

## Coverage matrix (current)

`✅` covered · `—` gap. This is the worklist for the rest of ENG-381.

| Capability | Covered by | Headroom? |
|---|---|---|
| C1 Grounding | — | — |
| C2 Honesty under absence | — | — |
| C3 Data loading | `reasoning-sales-dip-01`, `reasoning-ab-simpson-01` (implicit) | passes |
| C4 Quant computation | `reasoning-sales-dip-01`, `reasoning-ab-simpson-01` | passes |
| C5 Single-dataset analysis | `reasoning-sales-dip-01`, `reasoning-ab-simpson-01` | passes |
| C6 Multi-source | — | — |
| C7 Trend / forecasting | — | — |
| C8 Self-correction | `reasoning-sales-dip-01`, `reasoning-ab-simpson-01` | passes |
| C9 Decision support | — | — |
| C10 Artifact construction | — | — |
| C11 Efficiency | — *(no scorer yet)* | — |
| C12 Scope discipline | — | — |

**Reading of the gaps:** today the suite tests one analytical cluster (C3/C4/C5/C8)
on two cases that both pass — no Tier-1 grounding/honesty, no Tier-3 build, no
decision-support, and the two operational capabilities (C11/C12) have no scorer.
The baseline has **no headroom** until cases land on capabilities Anton can
currently miss (C2, C9, C10, C11).
