# Anton Eval Harness

End-to-end evaluation framework that runs real multi-turn conversations against live LLM APIs and scores the results. Use it to establish baselines, catch regressions, and compare model/prompt changes.

## Quick Start

```bash
# Install eval dependencies
uv sync --extra eval

# Set your API key (or export in shell)
export ANTHROPIC_API_KEY=sk-ant-...

# List all cases with cost estimates
uv run --extra eval python -m evals --list

# Run the full suite (~$0.43, ~5 min)
uv run --extra eval python -m evals

# Run a single case
uv run --extra eval python -m evals --case fibonacci-20

# Run a category
uv run --extra eval python -m evals --category code_generation

# Run cases by tag
uv run --extra eval python -m evals --tag fast

# Cap spending
uv run --extra eval python -m evals --budget 0.10

# Save report to a custom path
uv run --extra eval python -m evals --output my_report.json

# Override models
uv run --extra eval python -m evals --planning-model claude-sonnet-4-6 --coding-model claude-haiku-4-5-20251001
```

### Via pytest

```bash
# Full suite
uv run --extra eval pytest evals/ -v

# Single case
uv run --extra eval pytest evals/ -v -k fibonacci-20

# Category
uv run --extra eval pytest evals/ -v -k code_generation
```

## Benchmark Suite

32 cases across 7 categories:

| Category | Cases | What It Tests |
|----------|-------|---------------|
| `simple_qa` | 4 | Conversational responses, no tools needed |
| `code_generation` | 5 | Scratchpad use, correct computation, output formatting |
| `multi_step` | 4 | Multiple tool rounds, chained reasoning |
| `memory` | 3 | Memorize tool use, scope handling, knowing when *not* to memorize |
| `error_recovery` | 3 | Code fails then self-corrects, division by zero, import errors |
| `context_pressure` | 1 | Multi-turn recall — remembers facts across 5 turns |
| `complex` | 12 | ETL pipelines, graph algorithms, matrix math, regex parsing, class design, schema modeling, timezone arithmetic, contradictory requirements, iterative debugging |

Estimated full-suite cost: **~$0.43 per run**.

## Architecture

```
evals/
├── __main__.py       # Standalone CLI entry point (python -m evals)
├── types.py          # Core dataclasses: EvalCase, EvalResult, TurnRecord, ScoreResult
├── loader.py         # Load & validate YAML benchmark cases
├── runner.py         # EvalRunner: orchestrates cases via ChatSession.turn()
├── scoring.py        # 8 scorer types (contains, regex, tool_called, llm_judge, etc.)
├── judge.py          # LLM-as-judge: Haiku grades responses against rubrics
├── report.py         # JSON reports + human-readable summary tables
├── conftest.py       # pytest fixtures (real Anthropic provider)
├── test_evals.py     # Parametrized pytest tests — 1 test per YAML case
├── cases/            # Declarative benchmark definitions (YAML)
│   ├── simple_qa.yaml
│   ├── code_generation.yaml
│   ├── multi_step.yaml
│   ├── memory.yaml
│   ├── error_recovery.yaml
│   ├── context_pressure.yaml
│   └── complex.yaml
└── results/          # JSON reports from runs (gitignored)
```

## Writing Benchmark Cases

Cases are defined in YAML files under `evals/cases/`. Each file contains a list of cases:

```yaml
- id: my-new-case                    # unique slug
  category: code_generation          # groups cases in reports
  description: "What this tests"
  estimated_cost_usd: 0.01           # rough cost estimate
  tags: [scratchpad, fast]           # for filtering with --tag
  turns:
    - user_input: "Compute 2 + 2"   # single-turn
  scorers:
    - type: contains                 # assert response contains "4"
      value: "4"
      weight: 1.0
```

### Multi-turn cases

```yaml
turns:
  - user_input: "Create a list of 5 numbers"
  - user_input: "Now sort them in reverse"      # follow-up turn
```

### Scorer Types

| Type | What It Checks | Key Fields |
|------|---------------|------------|
| `contains` | Response contains substring (case-insensitive) | `value` |
| `regex` | Response matches regex pattern | `value` (regex) |
| `not_contains` | Response does NOT contain substring | `value` |
| `tool_called` | A specific tool was invoked | `value` (tool name) |
| `tool_not_called` | A specific tool was NOT invoked | `value` (tool name) |
| `tool_call_count` | Exact number of tool invocations | `value`, `expected_count` |
| `code_output` | Scratchpad stdout contains substring | `value` |
| `llm_judge` | LLM (Haiku) grades response against a rubric | `value` (rubric text) |

Every scorer has a `weight` (default 1.0). The overall score is the weighted pass rate:

```
overall_score = sum(weight for passed scorers) / sum(all weights)
```

A case passes if `overall_score >= 0.5`.

### LLM-as-Judge Rubrics

For nuanced evaluation, use `llm_judge` with a plain-English rubric:

```yaml
scorers:
  - type: llm_judge
    value: |
      The response should correctly compute the mean as 44.5
      and identify 78, 89, 90 as outliers.
      Score 1 if correct, 0.5 if partially correct, 0 if wrong.
    weight: 2.0
```

The judge uses Haiku with `temperature=0`. Score >= 0.5 = pass.

## Reports

Each run saves a JSON report to `evals/results/<timestamp>.json` and prints a summary:

```
=== Anton Eval Results ===
Run: 20260317_134137 | Models: claude-sonnet-4-6 / claude-haiku-4-5-20251001

Category                 Pass  Total   Score   Tokens     Cost
------------------------------------------------------------
code_generation           5/5      5    1.00     1751 $ 0.0105
complex                  12/12    12    1.00     4200 $ 0.0300
memory                    3/3      3    1.00      412 $ 0.0023
...
------------------------------------------------------------
TOTAL                   32/32     32    1.00    17000 $ 0.1700

Time: 300.0s | Avg: 9.4s/case
```

The JSON report contains full details: per-case scores, per-turn data, tool calls, token usage, and latency.

## How It Works

1. **Loader** reads YAML files and builds `EvalCase` objects
2. **Runner** creates a fresh `ChatSession` per case (with real scratchpad subprocess, and real `Cortex` for memory cases)
3. For each turn, it calls `session.turn()` and records: response text, tool calls, tokens, latency
4. **Scorers** are applied post-hoc to the collected data
5. **LLM judge** (if used) sends the response + rubric to Haiku for semantic grading
6. **Reporter** aggregates results into JSON + human-readable summary

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for LLM calls |
| `ANTON_ANTHROPIC_API_KEY` | Alt | Alternative key variable (checked as fallback) |
