# Anton analytical evals (ENG-381)

A small, repeatable harness that runs Anton end-to-end on realistic analytical
tasks and **scores the answer**, so we can measure analysis quality over time —
specifically, capture a **baseline before the [ENG-380] prompting fixes** and
prove the lift afterwards.

This is a **quality** eval, not a unit-test suite. It drives a real
`ChatSession.turn()` against a real model and grades the result. It makes live
LLM calls, costs tokens, and is non-deterministic — so it's an **offline/manual**
suite, not something gated in CI.

[ENG-380]: https://linear.app/mindsdb/issue/ENG-380

## Run it

```bash
# from the anton repo root
uv run python -m evals.runner evals/cases/reasoning-sales-dip-01.yaml
uv run python -m evals.runner --all
uv run python -m evals.runner --all --baseline   # also snapshot results/baseline/
```

Needs `ANTON_MINDS_API_KEY` / `ANTON_MINDS_URL` in `~/.anton/.env` (configured by
the desktop app's MindsHub setup).

## The model: why minds-cloud, and why we record (not pin) it

We eval against **minds-cloud** (`latest:sonnet`, effort `high`) because it's the
path real hub users get. But minds-cloud **can't be pinned**: its `/v1/models`
only advertises `latest:*` aliases, and it *rejects* every specific snapshot ID
(even the bare `claude-sonnet-4-6`) — verified. `latest:sonnet` is remapped by
MindsHub at will (today → `claude-sonnet-4-6`).

So every run records the **resolved** snapshot (`resolved_models` in the result
JSON) for **drift detection**: if that ID changes between a baseline and a later
run, the comparison is invalid and you re-baseline — instead of drifting
silently.

**To switch to a truly pinned baseline later** (Anthropic-direct with a real
snapshot, for durable months-long tracking), change `ModelConfig` in
`models.py` — `provider`/`base_url`/`*_model` are plain config. Build an
Anthropic provider instead of the openai-compatible minds one in
`build_llm_client`, and pin e.g. `claude-sonnet-4-6-<date>`.

## How a case is scored (hybrid)

Each case declares `dimensions`; each dimension names a scoring `method`:

- **`fact_match`** (deterministic): every `reference.key_facts` regex must appear
  in the answer (case-insensitive). Catches wrong/missing numbers and entities.
- **`llm_judge`** (model-as-judge): an LLM scores the answer 1–5 against
  `reference.ideal_reasoning` anchors; passes at `pass_bar_min`. Catches
  shallow-but-plausible reasoning — the ENG-380 axis a substring check can't see.

A case passes when all its declared dimensions pass.

## Capability map → coverage (which cases to write)

Cases are chosen to span Anton's surface deliberately, not at random. Tier =
how much of Anton the task exercises.

| Tier | Exercises | Example |
|---|---|---|
| 0 | base LLM only (skip — no Anton value-add) | "capital of France" |
| 1 | single-tool grounding / honesty | "AAPL price today"; the SpaceX-is-private trap |
| 2 | analytical reasoning on given data | **this case** — find the real driver of a dip |
| 3 | full build: data → artifact/dashboard | "get X data and build a dashboard" |

Scoring dimensions (scored independently so we see *what* moved): `correctness`,
`reasoning_depth`, `honesty`, `task_completion`, `efficiency`.

## Case file schema

```yaml
id: <stable-slug>
title: <human title>
tier: <0-3>
capabilities: [C5, C8]          # tags from the capability map (free-form)
dimensions: [correctness, reasoning_depth]   # what gets scored
prompt: <what the user asks>
fixtures: [sales.csv]           # files staged into the agent's workspace
environment: {tools: [scratchpad], web: false}
reference:
  key_facts: [<regex>, ...]     # used by fact_match
  ideal_reasoning: [<anchor>, ...]   # used by llm_judge
scoring:
  correctness:    {method: fact_match}
  reasoning_depth:{method: llm_judge, pass_bar_min: 4}
overall_pass: "all declared dimensions pass"
```

## Layout

```
evals/
  models.py     # ModelConfig, minds-cloud client builder, resolution probe
  spec.py       # EvalCase + YAML loader
  scorers.py    # fact_match + llm_judge
  runner.py     # build client → run turn → score → write result
  cases/
    reasoning-sales-dip-01.yaml
    fixtures/
      sales.csv          # committed (deterministic input)
      gen_sales.py       # regenerates sales.csv; documents the embedded "trap"
  results/               # per-run JSON (gitignored); results/baseline/ is committed
```
