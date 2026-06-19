---
title: Brain mapping
description: The full neuro-analogy behind Anton's modules — the seven learning systems, the structure-to-module table, and the on-disk file layout.
---

# Brain mapping

Anton's architecture mirrors the brain's major learning systems, and the module
names in `anton/core/memory/` are literal: `hippocampus.py`, `cortex.py`,
`cerebellum.py`, `acc.py`. The **user docs deliberately use plain names**
(memory, lessons, rules, skills — see the [Glossary](/reference/glossary)); this
page is the rosetta stone that maps plain names to brain structures to code.

## The seven learning systems

| Brain Region | Function | Anton Equivalent |
|---|---|---|
| Prefrontal Cortex (PFC) | Executive control, planning, the "inner voice" | Orchestrator — decides what to work on, how, and when to stop |
| Working Memory (dlPFC) | Temporary reasoning space, ~4 slots | Scratchpads — isolated reasoning environments |
| Hippocampus | Episodic memory, records experiences | Experience Store — logs of problem + context + solution |
| Cortex (semantic memory) | Facts, rules, identity — the consolidated knowledge | Engrams — `lessons.md`, `rules.md`, `profile.md` |
| Striatum (procedural memory) | Habits and learned procedures — patterns of action | Skills — multi-stage reusable procedures with declarative + chunked + code representations |
| Cerebellum (per-cell error learning) | Supervised correction on a single action — "what I expected vs what happened" | Cerebellum — buffers errored scratchpad cells, extracts generalizable lessons via post-mortem |
| Anterior Cingulate Cortex (turn-level error detection) | Notices when the same kind of error pattern fires more than once within an episode — the brain's ERN | ACC — observes turn events, flags repeat patterns, produces lessons that flow through the same Engram pipeline |

These systems coexist the way they coexist in the brain: declarative and
procedural memory are dissociable (a person with hippocampal damage like H.M.
can lose new declarative memories but still learn motor skills); the cerebellum
operates in parallel with continued action rather than blocking it; and the ACC
watches the whole turn rather than any single cell, complementing rather than
replacing the cerebellum.

## Structure → module → behavior

| Brain Structure | Module | What It Does |
|---|---|---|
| **Hippocampus** (CA3/CA1) | `hippocampus.py` | The storage engine. Reads and writes individual memory traces (engrams) to markdown files. One instance per scope — it doesn't decide *what* to remember, just executes storage and retrieval. |
| **Prefrontal Cortex** (dlPFC/vmPFC) | `cortex.py` | The executive coordinator. Manages two hippocampi (global + project), decides which memories to load into the LLM's context window, gates whether new memories need confirmation. |
| **Medial Temporal Lobe** (episodic) | `episodes.py` | Raw episodic memory. Logs every conversation turn as timestamped JSONL — user input, assistant responses, tool calls, scratchpad output. Searchable via the `recall` tool. Like HSAM: never forgets. |
| **Hippocampal Replay** (SWS consolidation) | `consolidator.py` | After a scratchpad session ends, replays what happened in compressed form and extracts durable lessons via a fast LLM call. Like sleep — offline, post-hoc, selective. |
| **Striatum** (procedural memory) | `skills.py` | Long-term procedural memory. Stores reusable skills as multi-stage directories (declarative → chunks → code). The LLM retrieves skills on demand via the `recall_skill` tool, the way the basal ganglia activates a learned action sequence in response to a familiar context. |
| **Cerebellum** (supervised error learning) | `cerebellum.py` | Forward-model + error correction at the *single-cell* time scale. Observes every scratchpad cell via pre/post execute hooks, buffers errored/warning cells across the turn, and runs a post-mortem LLM diff to extract generalizable lessons. Operates in parallel with the agent — never blocks. |
| **Anterior Cingulate Cortex** (ERN) | `acc.py` | Pattern-level error detection at the *whole-turn* time scale. Watches a stream of typed events and runs pure-function detectors at end-of-turn. Lessons flow through the same `cortex.encode()` path the cerebellum uses; it does not own storage. |
| **Reconsolidation** (Nader et al.) | `reconsolidator.py` | One-time migration. When old memory formats are reactivated, they enter a labile state and get re-encoded in the new format. Preserves content, updates structure. |
| **Medial PFC / Default Mode Network** | `profile.md` | Always-on self-model. Identity facts (name, timezone, preferences) that contextualize all processing — you don't "look up" your own name. |
| **Basal Ganglia + OFC** | `rules.md` | Go/No-Go behavioral gates. The direct pathway enables ("always"), the indirect pathway suppresses ("never"), the OFC handles conditions ("when X → do Y"). |
| **Anterior Temporal Lobe** | `lessons.md` | Semantic knowledge hub. Facts that started as episodes but have been distilled into general knowledge. |
| **Cortical Association Areas** | `topics/*.md` | Deep domain expertise stored in specialized regions. Not all active simultaneously — retrieved when contextual cues indicate relevance. |
| **Locus Coeruleus-NE** | Memory modes | The encoding gate. Controls how aggressively Anton writes new memories — from broad/indiscriminate to fully suppressed. |
| **Synaptic Homeostasis** | Compaction | During "sleep", weak traces are pruned and redundant memories are merged, preventing unbounded growth. |

## File layout on disk

Two scopes: **global** (cross-project, under the home directory) and
**project** (workspace-specific, under the project's `.anton/`).

```
~/.anton/                              GLOBAL scope (cross-project)
├── memory/
│   ├── profile.md                     Identity — who the user is
│   ├── rules.md                       Always/never/when behavioral rules
│   ├── lessons.md                     Semantic facts from experience
│   └── topics/                        Deep domain expertise
│       └── *.md
└── skills/                            PROCEDURAL MEMORY (striatum)
    └── <label>/                       One directory per skill
        ├── SKILL.md                   name, description, instructions (agentskills.io format)
        ├── references/                Stage 2 — higher-level recipes/macros (optional, v2+)
        ├── scripts/                   Stage 3 — runnable helper modules (optional, v2+)
        │   └── __init__.py
        ├── requirements.txt           Stage 3 dependencies (optional)
        └── stats.json                 Per-stage usage counters (recommended/used)

<project>/.anton/                      PROJECT scope (workspace-specific)
├── memory/
│   ├── rules.md                       Project-specific rules
│   ├── lessons.md                     Project-specific knowledge (cerebellum writes here)
│   └── topics/
│       └── *.md
├── episodes/                          EPISODIC MEMORY (conversation archive)
│   ├── 20260227_143052.jsonl          One file per session (YYYYMMDD_HHMMSS)
│   └── 20260228_091522.jsonl
├── anton.md                           User-written project context (unchanged)
└── .env                               Secrets (unchanged)
```

Scope rules:

- **Profile is global-only** — identity is singular.
- **Rules and lessons exist at both scopes.**
- **Skills live globally** (one library across projects) at `~/.anton/skills/`.
- **`anton.md` stays user-written** and is not managed by the memory system —
  see [Project context](/teach/project-context).

## Reading order for new contributors

1. `anton/core/memory/base.py` — the `Engram` dataclass and `HippocampusProtocol`.
   Everything else routes through these.
2. `anton/core/memory/hippocampus.py` then `cortex.py` — storage, then coordination.
3. `anton/core/memory/cerebellum.py` and `acc.py` — the error-learning loop.
4. `anton/core/session.py` — where it all gets wired into the turn loop.

Continue with [Memory systems](/developer/memory-systems) for the detailed
walk-through of each module.
