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
| C1 Grounding | — *(needs a web/retrieval case)* | — |
| C2 Honesty under absence | `honesty-data-absence-01` | **yes** (extrapolating/fabricating the missing value is a real failure) |
| C3 Data loading | `reasoning-sales-dip-01`, `reasoning-ab-simpson-01`, `decision-housing-01` | passes |
| C4 Quant computation | `reasoning-sales-dip-01`, `reasoning-ab-simpson-01`, `decision-housing-01` | passes |
| C5 Single-dataset analysis | `reasoning-sales-dip-01`, `reasoning-ab-simpson-01`, `decision-housing-01` | passes |
| C6 Multi-source | — | — |
| C7 Trend / forecasting | — | — |
| C8 Self-correction | `reasoning-sales-dip-01`, `reasoning-ab-simpson-01` | passes |
| C9 Decision support | `decision-housing-01` | **medium** (data-dump / pick-cheapest is a real failure) |
| C10 Artifact construction | `build-sales-dashboard-01` | **yes** (a complete, self-contained, chart-bearing html-app is real work Anton can miss) |
| C11 Efficiency | `build-sales-dashboard-01` *(metrics recorded every run; ceilings provisional)* | tracks ENG-350 — thrash/token-blowup trips it |
| C12 Scope discipline | — | — |

**Reading of the gaps:** the suite now spans Tiers 1–3 across honesty (C2), the
analytical cluster (C3/C4/C5/C8), decision support (C9), artifact build (C10),
and efficiency (C11) — with headroom on C2/C9/C10 so the baseline has somewhere
to move. Every run now also records turn cost (tokens/calls/seconds) regardless
of whether a case gates on it. Still open: C1 (web/retrieval grounding), C6
(multi-source), and C7 (trend/forecast).

**Ground-truth durability rule (learned the hard way):** a case's ground truth
must not depend on a real-world fact that can drift. The original C2 case asked
the agent to refuse a stock price because "SpaceX is private" — but SpaceX IPO'd
2026-06-12 (Nasdaq: SPCX), silently inverting the correct answer. `honesty-data-
absence-01` instead anchors ground truth to the committed fixture's own month
range and columns, which cannot change underneath it.

**Tier-3 build grading (how C10 got unblocked):** the deliverable of a build case
is a file on disk (an HTML dashboard), not chat text. The runner now **captures
`<workspace>/.anton/artifacts/` before tearing down the workspace**, and the
`artifact_check` scorer grades the produced file offline — an `html-app` of the
right type exists, its entry HTML is a complete self-contained document, it
renders a chart, and it contains the required figures/labels. Grading the file
(not the chat summary) is what stops a case passing by merely *claiming* it built
something (the C12 failure mode). It does **not** publish to 4nton.ai.
