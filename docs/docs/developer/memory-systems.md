---
title: Memory systems
description: Engrams, the hippocampus storage engine, the cortex coordinator, episodic memory, consolidation, and how memory reaches the LLM's prompts.
---

# Memory systems

All long-term declarative memory lives under `anton/core/memory/`; a small set
of legacy/orthogonal modules lives at `anton/memory/` (reconsolidator, history
store, session store, `/memory` command handlers). Procedural memory is covered
in [Skills internals](/developer/skills-internals), error learning in
[Cerebellum & ACC](/developer/cerebellum-and-acc).

## The Engram — fundamental unit of memory

Every memory trace is an `Engram` dataclass, defined in `anton/core/memory/base.py`:

```python
@dataclass
class Engram:
    text: str                                          # The memory content
    kind: "always" | "never" | "when" | "lesson" | "profile"  # Classification
    scope: "global" | "project"                        # Where to store it
    confidence: "high" | "medium" | "low" = "medium"   # Encoding gate signal
    topic: str = ""                                    # For lessons — topic slug
    source: "user" | "consolidation" | "llm" = "llm"   # Origin of the memory
```

Every producer — the `memorize` tool, the consolidator, the cerebellum, the ACC —
creates Engrams and routes them through one path:

```
Source (user/LLM/consolidation/cerebellum/ACC) → Engram created
  → Cortex.encoding_gate() — needs confirmation?
    → yes: queued for user review before next prompt
    → no:  Cortex.encode() → routes to correct Hippocampus by scope
           → Hippocampus writes to disk with file locking:
             profile = full atomic rewrite; rule = insert into rules.md
             section; lesson = append to lessons.md (+ topics/{slug}.md)
```

`base.py` also defines `HippocampusProtocol` — a `runtime_checkable` structural
`typing.Protocol` describing the contract of a single-scope memory store. The
file-backed `Hippocampus` satisfies it via structural subtyping; alternate
backends (database-backed, cloud-synced) plug into this seam without
inheriting from the file implementation.

## `hippocampus.py` — the storage engine

One instance per scope (global OR project). It doesn't decide what to remember —
it just reads and writes.

| Retrieval method | Reads | Brain analog |
|---|---|---|
| `recall_identity()` | `profile.md` | Medial PFC / Default Mode Network |
| `recall_rules()` | `rules.md` | Basal Ganglia + OFC |
| `recall_lessons(token_budget)` | `lessons.md` (budget-limited, most recent first) | Anterior Temporal Lobe |
| `recall_topic(slug)` | `topics/{slug}.md` | Cortical Association Areas |
| `recall_scratchpad_wisdom()` | "when" rules + scratchpad-related lessons + `topics/scratchpad-*.md` | Procedural priming |

| Encoding method | Writes | Behavior |
|---|---|---|
| `encode_rule(text, kind, confidence, source)` | `rules.md` under the right `## Always/Never/When` section | Deduplicates. Uses file lock. |
| `encode_lesson(text, topic, source)` | `lessons.md` + optionally `topics/{slug}.md` | Deduplicates. Append-only with lock. |
| `rewrite_identity(entries)` | `profile.md` | Full rewrite (atomic via `.tmp` + rename). |

### Memory entry format

Files are human-readable markdown; metadata hides in HTML comments:

```markdown
# Lessons
- CoinGecko free tier rate-limits at ~50 req/min <!-- topic:api-coingecko ts:2026-02-27 -->
```

| Field | Values | Meaning |
|---|---|---|
| `confidence` | `high`, `medium`, `low` | Drives the encoding gate in copilot mode |
| `source` | `user`, `consolidation`, `llm` | Where the memory originated |
| `ts` | `YYYY-MM-DD` | Encoding date; used for recency ordering |
| `topic` | slug string | Lessons only; cross-files into `topics/{slug}.md` |

## `cortex.py` — the executive coordinator

The Cortex manages two `Hippocampus` instances (global + project) and is the
encoding endpoint everything routes through.

| Method | Purpose |
|---|---|
| `build_memory_context()` | Assemble memories for system-prompt injection (~5800-token budget) |
| `get_scratchpad_context()` | Combine scratchpad wisdom from both scopes for the tool-description injection — the channel cerebellum lessons flow through |
| `encode(engrams)` | Route engrams to the right hippocampus by scope. Called by `memorize`, the consolidator, the cerebellum, and the ACC |
| `encoding_gate(engram)` | Mode-dependent check: does this engram need user confirmation? |
| `needs_compaction()` / `compact_all()` | Threshold check + LLM-assisted dedup/merge via `generate_object_code(_CompactionResult, ...)` |
| `maybe_update_identity(message)` | Identity-fact extraction via `generate_object_code(_IdentityFacts, ...)` — background task, fired by the session every 5 turns |

**Memory modes (the encoding gate).** Configured via `ANTON_MEMORY_MODE` or
`/setup` > Memory:

| Mode | Behavior | Brain analog |
|---|---|---|
| `autopilot` (default) | Anton decides what to save, no confirmation | High tonic NE — broad encoding |
| `copilot` | Auto-save high-confidence, confirm ambiguous after the answer | Moderate NE — selective |
| `off` | Never save (still reads existing memory) | Encoding suppressed |

Confirmations are *never* shown mid-turn — only after the full response, right
before the next prompt.

**Compaction (synaptic homeostasis).** When a memory file exceeds the
threshold (`_COMPACTION_THRESHOLD = 20` entries in `cortex.py`), session start
fires a background compaction: the coding model removes duplicates, merges
restatements, and drops superseded entries. Atomic rewrite (`.tmp` +
`os.rename`), exclusive lock, silently skipped on LLM failure — it never
corrupts existing memory.

## `consolidator.py` — sleep replay

After a scratchpad session ends, `should_replay(cells)` runs cheap heuristics
(no LLM call): replay if any cell errored (high-signal — errors are
emotional), the session was long (5+ cells), or any cell was cancelled/killed;
skip sessions with fewer than 2 cells (too short to learn from).

`replay_and_extract(cells, llm)` compresses the cell history into one line per
cell, sends it to the coding model via
`generate_object_code(_ConsolidatedLessons, ...)`, and returns Engrams with
`source="consolidation"` — rules (always/never/when) and lessons, each with
scope and confidence so the encoding gate knows what to do.

## `reconsolidator.py` — legacy migration

One-time format migration at startup (`anton/memory/reconsolidator.py`):
`needs_reconsolidation()` fires when old `.anton/context/` or
`.anton/learnings/` directories exist and the new `memory/` directory lacks
`rules.md`, `lessons.md`, and `profile.md`; `reconsolidate()` then converts
them into `memory/lessons.md` + `memory/topics/`. Synchronous, no LLM calls,
never deletes the old files.

## `episodes.py` — episodic memory

A complete timestamped JSONL log per session in `.anton/episodes/`. Roles:

| Role | What's logged |
|---|---|
| `user` | User's input |
| `assistant` | Anton's text response |
| `tool_call` | Tool invocation input (truncated to 500 chars) |
| `tool_result` | Tool output (truncated to 2000 chars) |
| `scratchpad` | Scratchpad cell stdout (truncated to 2000 chars) |

The `recall` tool searches it: case-insensitive substring match across all
JSONL files, newest first, with a `days_back` filter. The LLM calls it
explicitly — a normal tool call, not automatic (see
[Episodes & recall](/teach/episodes-and-recall)). Design principles:
**fire-and-forget** (`log()` never raises), **file locking**
(`fcntl.flock(LOCK_EX)` for concurrent appends), **truncation** to prevent
JSONL bloat, and a **toggle** (`ANTON_EPISODIC_MEMORY`, default on).

## How memory reaches the LLM

**Moment A — system prompt (strategic retrieval).** At turn start,
`cortex.build_memory_context()` assembles labeled sections
(`## Your Memory — Identity`, `## Your Memory — Global Rules`, ...) so the LLM
knows these are its own memories, not user instructions. `anton.md` is
injected *after* memory, giving user instructions higher priority.
**Moment B — scratchpad tool description (procedural priming).** When
scratchpads are active, `cortex.get_scratchpad_context()` appends relevant
lessons to the scratchpad tool's description — the LLM sees them exactly when
composing code.

### Context budget

| Section | Brain analog | Budget | Loaded when |
|---|---|---|---|
| Identity | mPFC / DMN | ~300 tokens | Always (system prompt) |
| Global rules | Basal Ganglia | ~1500 tokens | Always (system prompt) |
| Project rules | Basal Ganglia | ~1500 tokens | Always (system prompt) |
| Global lessons | ATL semantics | ~1000 tokens | Always (most recent first) |
| Project lessons | ATL semantics | ~1000 tokens | Always (most recent first) |
| Scratchpad wisdom | Procedural priming | ~500 tokens | Scratchpad active (tool desc) |
| Procedural memory list | Striatum (skill labels) | ~50 tokens per skill | When any skills are saved |
| Topic files | Cortical association | Unlimited | On demand |
| Skill procedures | Striatum (full skills) | Variable | On demand (`recall_skill`) |
| Episodic recall | MTL episodic | Variable | On demand (`recall`) |
| **Total in prompt** | **Working memory** | **~5800 tokens + ~50/skill** | ~3% of 200K context |

## The `memorize` tool

The LLM-facing encoding primitive. `handle_memorize()` in
`anton/core/tools/tool_handlers.py` converts each entry to an `Engram` with
`confidence="high"` and `source="user"` (explicit tool calls are trusted), then
encodes via a fire-and-forget `asyncio.create_task` — explicit memorize calls
never interrupt the turn; confirmations are reserved for the consolidator.

## Concurrency safety

| Operation | Scope | Strategy |
|---|---|---|
| Normal writes (rules, lessons) | Global | `fcntl.flock(LOCK_EX)` per file — append-only, no read-modify-write race |
| Normal writes | Project | No locking needed — one session per project |
| Compaction | Global | Exclusive lock + atomic rename (write `.tmp` then `os.rename`) |
| Identity updates | Global | Exclusive lock (full rewrite via `.tmp` + rename) |
| Concurrent compaction | Global | Other sessions skip — only one "sleeps" at a time |
