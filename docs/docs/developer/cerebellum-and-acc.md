---
title: "Error learning: cerebellum & ACC"
description: How Anton learns from failures — per-cell supervised error learning in the cerebellum, and turn-level pattern detection in the anterior cingulate cortex.
---

# Error learning: cerebellum & ACC

Anton has two error-learning systems at two time scales. The **cerebellum**
(`anton/core/memory/cerebellum.py`) learns from a *single failed scratchpad
cell*. The **ACC** (`anton/core/memory/acc.py`) learns from a *pattern across
multiple events within one turn*. Both are producers, not storage: their output
is `Engram` objects routed through `Cortex.encode()`, the same pipeline the
consolidator and the `memorize` tool use (see [Memory systems](/developer/memory-systems)).

## Cerebellum — per-cell supervised error learning

Brain analog: forward modeling and error correction. When a motor command is
issued, the cerebellum predicts the expected sensory consequences; when actual
feedback arrives, it computes the prediction error and refines future commands.

For Anton, the "motor command" is a scratchpad cell. Before the cell runs, the
LLM declares intent via the `one_line_description` field on the scratchpad tool —
that description **is** the forward model. After the cell runs, the actual
outcome (stdout, stderr, error) is compared against it, and meaningful
divergences become generalizable lessons that future code-generating calls see.

### Hooks live in the dispatcher, not the runtime

The cerebellum observes via two hooks fired from the scratchpad tool dispatcher
(`handle_scratchpad` in `anton/core/tools/tool_handlers.py`) — never from the
runtime backend:

```
handle_scratchpad (orchestration layer)
  ├─ build prelim Cell from tool input
  ├─ FIRE pre-execute observers ──→ Cerebellum.on_pre_execute (counter)
  ├─ pad.execute(code, ...)        (pure execution — runtime never sees observers)
  ├─ FIRE post-execute observers ─→ Cerebellum.on_post_execute (buffer if errored)
  └─ return formatted result
```

This decoupling is intentional. `LocalScratchpadRuntime`, `ScratchpadManager`,
and `RemoteScratchpadRuntime` are completely hook-agnostic — they don't import
the cerebellum, have no hook attributes, and never call observers. A new backend
inherits zero hook code because there is none to inherit. Observation is an
orchestration concern; the dispatcher is the only place where execution and
observation meet. (Observer firing is best-effort: a buggy observer logs a
warning and never kills a cell.)

### Cheap path: zero LLM calls on clean cells

`on_post_execute` checks `cell.error is None and not cell.stderr.strip()` and
returns immediately for clean cells — never buffered, no LLM call ever made.
The cost of running the cerebellum on a happy-path turn is **zero LLM calls**.

### Batched per-turn diff

Errored/warning cells accumulate in a buffer across the turn. At end-of-turn,
`_schedule_cerebellum_flush()` fires `Cerebellum.flush()` as a fire-and-forget
background task — the user gets their reply immediately:

1. Buffered cells are formatted into a compact post-mortem prompt.
2. One LLM call via `LLMClient.generate_object_code` (the cheap coding model)
   returns a `_DiffPassResult` Pydantic model with extracted lessons.
3. Each lesson becomes an `Engram` with `kind="lesson"`, `topic="scratchpad"`,
   `source="consolidation"`, routed through `Cortex.encode()`.
4. Future scratchpad cells see those lessons via the existing
   `recall_scratchpad_wisdom()` injection into the scratchpad tool description.

The cerebellum is a **producer only** — no parallel storage system, no separate
corrections file. Whatever the consolidator and `memorize` write to, the
cerebellum also writes to. Generated lessons land in `lessons.md` like any other
engram and get pruned by the same compaction loop.

| Method | Purpose |
|---|---|
| `on_pre_execute(cell)` | Pre-execute hook. Counter only in v1. |
| `on_post_execute(cell)` | Cheap path skips clean cells; errored/warning cells get buffered. |
| `flush()` | Batched diff pass on buffered cells, encode via Cortex, clear buffer. |
| `reset()` | Drop the buffer without encoding (turn cancelled mid-flight). |
| `buffered_count` | Cells waiting for the next flush. |

## ACC — turn-level pattern detection

Brain analog: the anterior cingulate cortex fires the *error-related negativity*
(ERN) ~80 ms after the brain notices an actual outcome diverged from an expected
one. Anton's ACC watches a turn unfold as a stream of typed events and, at
end-of-turn, extracts lessons from patterns that fired more than once — e.g.
scratchpad name switches (each name is a separate venv), oversized cells whose
`code` silently serialized to empty, or the same tool failing three times in a
row with the same args.

### Modes

Env var `ANTON_ACC_MODE` (mirrors `ANTON_MEMORY_MODE`); read in
`ChatSession.__init__`, default `passive`:

| Mode | Behaviour |
|---|---|
| `off` | ACC observes nothing — events drop at the safe-emit wrapper. |
| `passive` (default) | Layer 1 only. Lessons drain to memory at end-of-turn; the next turn's system prompt picks them up. Zero added surface in the turn loop. |
| `active` | Layer 1 + Layer 2. Lessons ALSO inject inline as text blocks alongside `tool_result` blocks, so the LLM sees them on its very next round. |

### Layer 1 — passive learning (emit sites + end-of-turn drain)

Emit sites call `session._acc_observe(kind, detail, severity=...)` — a safe
wrapper that no-ops when the ACC isn't attached, no-ops when the cortex is off,
and swallows `ValueError` on unknown kinds so emit-site drift never breaks a turn.

| Event kind | Emit site | File |
|---|---|---|
| `scratchpad_call` | After args validation in the `handle_scratchpad` exec branch | `core/tools/tool_handlers.py` |
| `scratchpad_result` | After `pad.execute()` returns a non-killed cell | `core/tools/tool_handlers.py` |
| `scratchpad_empty_code` | When `prepare_scratchpad_exec` rejects the call | `core/tools/tool_handlers.py` |
| `scratchpad_reset` | After `pad.reset()` in the reset action | `core/tools/tool_handlers.py` |
| `scratchpad_killed` | When the cell error starts with `Cancelled` / `Cell timed out` / `Cell killed` | `core/tools/tool_handlers.py` |
| `tool_call` | Top of the per-tool-call loop in `_stream_and_handle_tools` | `core/session.py` |
| `tool_result` | After result text is finalized | `core/session.py` |
| `history_repair` | After `_seal_dangling_tool_uses` inserts synthetic blocks | `core/session.py` |
| `cap_exhausted` | When the tool-round cap is exceeded | `core/session.py` |

`_schedule_acc_flush()` runs at end of `turn()` and `turn_stream()`, next to the
cerebellum flush. Detectors are pure functions (no LLM call); only the
`cortex.encode()` file I/O is wrapped in `asyncio.create_task`. Each `Lesson`
becomes an Engram with `kind=lesson.kind`, `scope="global"`,
`confidence="high"`, `source="consolidation"` — the `kind` (always/never/when)
was tagged by the detector, so lessons land in the right section of
`rules.md`. De-dupe uses a
caller-supplied predicate (`has_similar_lesson`, currently a cheap substring
match against `rules.md`).

### Layer 2 — mid-turn nudges (active mode only)

`_acc_maybe_nudge` runs after each tool-call round: `acc.at_round_n()`
re-evaluates every detector and returns only lessons whose detector hasn't
already nudged this turn. Each new lesson is appended as a
`{"type": "text", "text": "[Anton self-check — detector] rule"}` block inside
the same user message that carries the round's tool results — one nudge per
detector per turn. The mid-turn path deliberately skips `has_similar_lesson`:
if a rule is in `rules.md` but being violated right now, repeating it inline is
the whole point.

### Layer 3 — not yet wired (deliberate)

Retrieval-scored rule ranking at system-prompt assembly: score candidate rules
by relevance, load the top-K within budget, age out rules that never make the
cut. Needs an embedding index over rules plus a ranker call on the load path.

### Event vocabulary (9 kinds)

The `EVENT_KINDS` frozenset is a closed vocabulary; `observe()` raises
`ValueError` on unknown kinds. Tests assert every kind is read by at least one
detector (the single producer-only exception is `tool_call`, reserved for a
future `detect_orphaned_tool_call`).

| Kind | Detail shape | Read by |
|---|---|---|
| `scratchpad_call` | `{name, code_len, one_line_description}` | `detect_name_switch`, `detect_oversized_cell` |
| `scratchpad_result` | `{name, success, stdout_len, error}` | `detect_repeated_error_signature` |
| `scratchpad_empty_code` | `{name}` | `detect_oversized_cell` |
| `scratchpad_reset` | `{name, reason}` | `detect_reset_churn` |
| `scratchpad_killed` | `{name, reason}` | `detect_kill_loop` |
| `tool_call` | `{name, args_summary}` | producer-only (reserved) |
| `tool_result` | `{name, success, error}` | `detect_repeated_tool_error`, `detect_repeated_error_signature` |
| `history_repair` | `{reason}` | `detect_repair_churn` |
| `cap_exhausted` | `{}` | `detect_cap_exhausted` |

### Detectors (9 pure functions)

Detectors are stateless functions of `Sequence[Event] -> Lesson | None`.
Cross-detector dedupe at `at_end_of_turn()` collapses overlapping lessons.

| Detector | Fires when | Cognitive failure it learns from |
|---|---|---|
| `detect_name_switch` | 2+ distinct scratchpad names in one turn | Identity sprawl — each name is a separate venv |
| `detect_oversized_cell` | Empty-code drops OR 2+ cells over ~5 KB | Silent schema truncation of large `code` strings |
| `detect_repeated_tool_error` | 2+ consecutive failures of the same tool | Blind retry of the same tool |
| `detect_repeated_error_signature` | Same normalised error signature 3+ times across any producers | Blind retry across tools / arg tweaks |
| `detect_reset_churn` | 2+ scratchpad resets in one turn | Abandoning state instead of debugging in place |
| `detect_kill_loop` | 2+ cells killed on the same scratchpad name | Writing cells that hang — approach too heavy |
| `detect_severity_climb` | Strictly-increasing severity run of length 3+ ending at 5+ | Situation deteriorating without strategy change |
| `detect_repair_churn` | 3+ `history_repair` events in one turn | Malformed tool_use/result structure; conversation derailing |
| `detect_cap_exhausted` | A single `cap_exhausted` event | Round cap hit → mandatory post-mortem |

`detect_repeated_error_signature` runs error strings through
`_normalise_error_signature()` — a cheap regex pass collapsing paths, integers,
hex addresses, and short quoted tokens into placeholders — so
`engine='gmail-1'` and `engine='gmail-2'` hash to the same signature and the
loop is caught even when the LLM tweaks args between attempts.

Tests: `tests/test_acc.py` (pure detectors → state → JSON-fixture replay →
vocabulary discipline) plus wiring tests in `tests/test_session_acc_init.py`.
