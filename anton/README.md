# Inside Anton 

## Introduction
In 2015, after reading *How to Create a Mind* by Ray Kurzweil, I became convinced that we could programmatically build a mind by mirroring the brain’s core building blocks.
I tried. I failed — but I learned something important: one fundamental piece was missing. I called it the **Anticipation Block Architecture**. You can read about it [here](https://torrmal.github.io/2015/12/29/anticipation-loop/).

It turns out the world went on to build something remarkably similar: transformers and now, in 2026, LLMs have matured to the point where the ideas seeded by *How to Create a Mind* are no longer just philosophical — they’re implementable.
And here we are: Like an adrenaline junkie eyeing at a bungee looking for another fix, trying again: Meet **Anton**.

## A mini Mind

It is probably obvious now, but Anton has a brain-inspired architecture, and the more we build it the more it resembles/mirrors functional parts of the brain.  On the other hand we also understand that people don't need to know anything about the brain to play with Anton, so we mapped some of the places/files where users can have inputs, or investigate what's up, to names that make more sense than the scientific name of that function of the brain.

The current implementation has seven blocks, mapping the major learning systems:

| Brain Region                 | Function                                          | Anton Equivalent                                              |
|------------------------------|---------------------------------------------------|---------------------------------------------------------------|
| Prefrontal Cortex (PFC)      | Executive control, planning, the "inner voice"   | Orchestrator — decides what to work on, how, and when to stop |
| Working Memory (dlPFC)       | Temporary reasoning space, ~4 slots              | Scratchpads — isolated reasoning environments                 |
| Hippocampus                  | Episodic memory, records experiences             | Experience Store — logs of problem + context + solution       |
| Cortex (semantic memory)     | Facts, rules, identity — the consolidated knowledge | Engrams — `lessons.md`, `rules.md`, `profile.md`            |
| Striatum (procedural memory) | Habits and learned procedures — patterns of action | Skills — multi-stage reusable procedures with declarative + chunked + code representations |
| Cerebellum (per-cell error learning)  | Supervised correction on a single action — "what I expected vs what happened" | Cerebellum — buffers errored scratchpad cells, extracts generalizable lessons via post-mortem |
| Anterior Cingulate Cortex (turn-level error detection) | Notices when the same kind of error pattern fires more than once within an episode — the brain's ERN | ACC — observes turn events, flags repeat patterns, produces lessons that flow through the same Engram pipeline |

These seven systems coexist the way they coexist in the brain: declarative and procedural memory are dissociable (a person with hippocampal damage like H.M. can lose new declarative memories but still learn motor skills); the cerebellum operates in parallel with continued action rather than blocking it; and the ACC watches the whole turn rather than any single cell, complementing rather than replacing the cerebellum.



## Architecture of Anton

The high-level flow — how the executive, scratchpads, and the long-term stores collaborate on every turn:

```
  ┌────────────────────────────────────────────────────┐
  │              EXECUTIVE (the orchestrator)          │
  │                                                    │
  │  On new problem:                                   │
  │    1. Check SKILL LIBRARY → match?                 │
  │       YES → recall_skill(label) → load procedure   │
  │       NO  → open fresh scratchpad                  │
  │    2. Monitor scratchpad progress                  │
  │    3. Detect stuck/failure → pivot strategy        │
  │    4. On success → record to experience store      │
  └────────────┬────────────────────↑──────────────────┘
               │ spawns & monitors  │
               ▼                    │
  ┌──────────────────────────────────────────────────────┐
  │              SCRATCHPADS (working memory)            │
  │                                                      │
  │  Each scratchpad is:                                 │
  │  - An isolated reasoning environment (its own venv)  │
  │  - A chain-of-thought trace (code + observations)    │
  │  - Has a goal, constraints, and a budget             │
  │  - Can request sub-scratchpads (decomposition)       │
  │  - Can invoke the hypocampus in a loop               │
  │                                                      │
  │  Every cell execution fires pre/post hooks observed  │
  │  by the CEREBELLUM (post-mortem error learning).     │
  └──────┬──────────────┬───────────────────┬────────────┘
         │              │                   │
         │ on success   │ on cell errors    │ on success
         ▼              ▼                   ▼
  ┌────────────┐  ┌──────────────┐  ┌─────────────────────┐
  │ EXPERIENCE │  │  CEREBELLUM  │  │   SKILL LIBRARY     │
  │   STORE    │  │              │  │                     │
  │ (hipp.)    │  │ Buffers bad  │  │ /skill save → LLM   │
  │            │  │ cells, runs  │  │ drafts a procedure  │
  │ Episodes — │  │ post-mortem  │  │ with label + name + │
  │ JSONL log  │  │ via LLM,     │  │ description +       │
  │ of every   │  │ encodes new  │  │ declarative_md.     │
  │ turn.      │  │ lessons via  │  │                     │
  │            │  │ Cortex.      │  │ Future turns recall │
  │ Recall via │  │              │  │ the procedure via   │
  │ `recall`   │  │ Lessons feed │  │ recall_skill tool.  │
  │ tool.      │  │ next code    │  │                     │
  │            │  │ generation   │  │ Stored at           │
  │            │  │ (procedural  │  │ ~/.anton/skills/    │
  │            │  │ priming).    │  │   <label>/          │
  └────────────┘  └──────────────┘  └─────────────────────┘
```

The brain analog: the executive (PFC) plans and delegates to working memory (scratchpads), which can pull on procedural memory (striatum/skills) for known recipes and on declarative memory (hippocampus/cortex/engrams) for facts. The cerebellum runs in parallel with continued action — it never blocks the agent, it just refines future cells through supervised error learning.

And the Hipocampus also is controlled as follows:

```
                    ┌───────────────────────────────────────────────┐
                    │              CORTEX (cortex.py)               │
                    │     Prefrontal Cortex — Executive Control     │
                    │  Coordinates all memory systems, decides what │
                    │  to load into working memory (context window) │
                    └────────┬──────────────┬──────────────┬────────┘
                             │              │              │
               ┌─────────────┘       ┌──────┘              └──────┐
               ▼                     ▼                            ▼
    ┌──────────────────┐   ┌───────────────────┐        ┌───────────────────┐
    │   HIPPOCAMPUS    │   │  CONSOLIDATOR     │        │  RECONSOLIDATOR   │
    │ (hippocampus.py) │   │(consolidator.py)  │        │(reconsolidator.py)│
    │                  │   │                   │        │                   │
    │  Encodes & reads │   │ Sleep replay —    │        │ Reactivates old   │
    │  memory traces   │   │ reviews scratchpad│        │ memories, converts│
    │  at one scope    │   │ sessions offline, │        │ legacy formats to │
    │  (global / proj) │   │ extracts lessons  │        │ new schema        │
    └────────┬─────────┘   └───────────────────┘        └───────────────────┘
             │
             ▼
    ┌────────────────────────────────────────────────┐
    │         SEMANTIC MEMORY FILES (on disk)        │
    │                                                │
    │  profile.md   ← Identity (Default Mode Network)│
    │  rules.md     ← Behavioral gates (Basal Gangli)│
    │  lessons.md   ← Semantic facts (Temporal Lobe) │
    │  topics/*.md  ← Domain expertise (Association  │
    │                 Areas), loaded on demand       │
    └────────────────────────────────────────────────┘

    ┌───────────────────────────────────────────────┐
    │      EPISODIC MEMORY (episodes.py)            │
    │      Medial Temporal Lobe — raw experience    │
    │                                               │
    │  episodes/*.jsonl  ← One file per session     │
    │  Timestamped log of every turn, tool call,    │
    │  and scratchpad execution. Searchable via     │
    │  the `recall` tool.                           │
    └───────────────────────────────────────────────┘
```

## Brain Mapping

| Brain Structure | Module | What It Does |
|---|---|---|
| **Hippocampus** (CA3/CA1) | `hippocampus.py` | The storage engine. Reads and writes individual memory traces (engrams) to markdown files. One instance per scope — it doesn't decide *what* to remember, just executes storage and retrieval. |
| **Prefrontal Cortex** (dlPFC/vmPFC) | `cortex.py` | The executive coordinator. Manages two hippocampi (global + project), decides which memories to load into the LLM's context window, gates whether new memories need confirmation. |
| **Medial Temporal Lobe** (episodic) | `episodes.py` | Raw episodic memory. Logs every conversation turn as timestamped JSONL — user input, assistant responses, tool calls, scratchpad output. Searchable via the `recall` tool. Like HSAM: never forgets. |
| **Hippocampal Replay** (SWS consolidation) | `consolidator.py` | After a scratchpad session ends, replays what happened in compressed form and extracts durable lessons via a fast LLM call. Like sleep — offline, post-hoc, selective. |
| **Striatum** (procedural memory) | `skills.py` | Long-term procedural memory. Stores reusable skills as multi-stage directories (declarative → chunks → code). The LLM retrieves skills on demand via the `recall_skill` tool, the way the basal ganglia activates a learned action sequence in response to a familiar context. |
| **Cerebellum** (supervised error learning) | `cerebellum.py` | Forward-model + error correction at the *single-cell* time scale. Observes every scratchpad cell via pre/post execute hooks, buffers errored/warning cells across the turn, and runs a post-mortem LLM diff to extract generalizable lessons. Lessons flow through the existing wisdom-injection pipeline into future code generation. Operates in parallel with the agent — never blocks. |
| **Anterior Cingulate Cortex** (ERN — turn-level error detection) | `acc.py` | Pattern-level error detection at the *whole-turn* time scale. Watches a stream of typed events (scratchpad calls/results, tool calls/results, history repairs, round milestones) and runs pure-function detectors at end-of-turn. Lessons it emits flow through the same `cortex.encode()` path the cerebellum uses; it does not own storage. Implemented as a standalone module with passing tests; not yet wired into `ChatSession`. |
| **Reconsolidation** (Nader et al.) | `reconsolidator.py` | One-time migration. When old memory formats are reactivated, they enter a labile state and get re-encoded in the new format. Preserves content, updates structure. |
| **Medial PFC / Default Mode Network** | `profile.md` | Always-on self-model. Identity facts (name, timezone, preferences) that contextualize all processing — you don't "look up" your own name. |
| **Basal Ganglia + OFC** | `rules.md` | Go/No-Go behavioral gates. The direct pathway enables ("always"), the indirect pathway suppresses ("never"), the OFC handles conditions ("when X → do Y"). |
| **Anterior Temporal Lobe** | `lessons.md` | Semantic knowledge hub. Facts that started as episodes but have been distilled into general knowledge. |
| **Cortical Association Areas** | `topics/*.md` | Deep domain expertise stored in specialized regions. Not all active simultaneously — retrieved when contextual cues indicate relevance. |
| **Locus Coeruleus-NE** | Memory modes | The encoding gate. Controls how aggressively Anton writes new memories — from broad/indiscriminate to fully suppressed. |
| **Synaptic Homeostasis** | Compaction | During "sleep", weak traces are pruned and redundant memories are merged, preventing unbounded growth. |

## File Layout on Disk

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

Profile (`profile.md`) is global-only — identity is singular. Rules and lessons exist at both scopes. Skills live globally (one library across projects) at `~/.anton/skills/`. `anton.md` stays as the user-written instruction file and is not managed by the memory system.

## Memory Entry Format

All memory files are human-readable markdown. Metadata lives in HTML comments so the files look clean when you open them:

**rules.md:**
```markdown
# Rules

## Always
- Use httpx instead of requests <!-- confidence:high source:user ts:2026-02-27 -->
- Call progress() before llm.complete() in scratchpad <!-- confidence:high source:consolidation ts:2026-02-27 -->

## Never
- Use time.sleep() in scratchpad cells <!-- confidence:high source:consolidation ts:2026-02-27 -->

## When
- If fetching paginated API data → async + progress() between pages <!-- confidence:medium source:consolidation ts:2026-02-27 -->
```

**lessons.md:**
```markdown
# Lessons
- CoinGecko free tier rate-limits at ~50 req/min <!-- topic:api-coingecko ts:2026-02-27 -->
- Bitcoin price data via /coins/bitcoin/market_chart/range <!-- topic:api-coingecko ts:2026-02-27 -->
- pandas read_csv needs encoding='utf-8-sig' for BOM files <!-- topic:pandas ts:2026-02-27 -->
```

**profile.md:**
```markdown
# Profile
- Name: Jorge
- Timezone: PST
- Expertise: Python, data analysis, API integrations
- Communication: concise, direct
- Tools: prefers uv over pip, uses VS Code, macOS
```

### Metadata Fields

Each entry can carry HTML-comment metadata:

| Field | Values | Meaning |
|---|---|---|
| `confidence` | `high`, `medium`, `low` | How certain the system is. Drives the encoding gate in copilot mode. |
| `source` | `user`, `consolidation`, `llm` | Where the memory originated. User-sourced = explicit tool call or user request. Consolidation = extracted from scratchpad replay. LLM = the model decided to save it mid-conversation. |
| `ts` | `YYYY-MM-DD` | When the memory was encoded. Used for recency ordering in lessons. |
| `topic` | slug string | Topic tag for lessons. Used to cross-file into `topics/{slug}.md`. |

## Episodic Memory — Raw Conversation Archive

Episodic memory is a complete, timestamped log of everything that happens in a conversation. Brain analog: the **Medial Temporal Lobe** episodic memory system.

### File Format

Each session produces one JSONL file in `.anton/episodes/`:

```jsonl
{"ts":"2026-02-27T14:30:52","session":"20260227_143052","turn":1,"role":"user","content":"What's the bitcoin price?","meta":{}}
{"ts":"2026-02-27T14:30:55","session":"20260227_143052","turn":1,"role":"assistant","content":"Let me check that.","meta":{}}
{"ts":"2026-02-27T14:31:00","session":"20260227_143052","turn":1,"role":"tool_call","content":"{'action': 'exec', ...}","meta":{"tool":"scratchpad"}}
{"ts":"2026-02-27T14:31:02","session":"20260227_143052","turn":1,"role":"scratchpad","content":"$67,432","meta":{"description":"Fetch BTC"}}
{"ts":"2026-02-27T14:31:03","session":"20260227_143052","turn":1,"role":"tool_result","content":"[output]\n$67,432","meta":{"tool":"scratchpad"}}
```

### Roles

| Role | What's Logged |
|------|---------------|
| `user` | User's input (text or stringified multimodal content) |
| `assistant` | Anton's text response |
| `tool_call` | Tool invocation input (truncated to 500 chars) |
| `tool_result` | Tool output (truncated to 2000 chars) |
| `scratchpad` | Scratchpad cell stdout (truncated to 2000 chars) |

### The `recall` Tool

The LLM has a `recall` tool that searches episodic memory. It's included in the tool list when episodic memory is enabled.

```json
{
  "name": "recall",
  "input": {
    "query": "bitcoin",
    "max_results": 20,
    "days_back": 30
  }
}
```

Search is case-insensitive substring matching across all JSONL files, newest-first. The `days_back` parameter filters by session file timestamp.

**When recall happens:** The LLM decides to call the `recall` tool during conversation — typically when the user asks about previous sessions, past work, or "what did we talk about last time?" It's a standard tool call like `scratchpad` or `memorize`, not automatic.

### Design Principles

- **Fire-and-forget**: `log()` catches all exceptions and never raises. Logging never blocks the conversation.
- **File locking**: Uses `fcntl.flock(LOCK_EX)` for safe concurrent appends.
- **Truncation**: Tool inputs capped at 500 chars, results at 2000 chars — prevents JSONL bloat from large scratchpad outputs.
- **Toggle**: Controlled by `ANTON_EPISODIC_MEMORY` env var or `/setup` > Memory. Default: ON.

## How Memory Flows Through a Session

Memory reaches the LLM at two distinct moments:

### Moment A — System Prompt (Strategic Retrieval)

When a turn begins, the Cortex assembles memories into the system prompt. This is like the prefrontal cortex loading relevant memories into working memory before a task:

1. **Identity** (profile) — always loaded (~300 tokens)
2. **Global rules** — behavioral constraints (~1500 tokens)
3. **Project rules** — scope-specific constraints (~1500 tokens)
4. **Global lessons** — semantic knowledge, most recent first (~1000 tokens)
5. **Project lessons** — scope-specific facts, most recent first (~1000 tokens)

Total budget: ~5800 tokens, about 3% of a 200K context window.

The Cortex inserts these as labeled sections in the system prompt (`## Your Memory — Identity`, `## Your Memory — Global Rules`, etc.) so the LLM knows they're its own memories, not user instructions. The `anton.md` user-written context is injected *after* memory, giving user instructions higher priority.

### Moment B — Scratchpad Tool Description (Procedural Priming)

When scratchpads are active, relevant lessons are appended to the scratchpad tool description. The LLM sees them right when composing code — like procedural memory that activates automatically when you get on a bike:

```python
scratchpad_tool["description"] += f"\n\nLessons from past sessions:\n{wisdom}"
```

This combines all "when" rules + lessons with `scratchpad-*` topics from both scopes. The content comes from `cortex.get_scratchpad_context()`, which calls `recall_scratchpad_wisdom()` on both hippocampi.

## The `memorize` Tool

Anton has a tool called `memorize` that it can call during conversation to encode new memories. The LLM decides what to save and classifies each entry:

```json
{
  "entries": [
    {
      "text": "CoinGecko rate-limits at 50 req/min",
      "kind": "lesson",
      "scope": "global",
      "topic": "api-coingecko"
    },
    {
      "text": "Always use progress() for long API calls in scratchpad",
      "kind": "always",
      "scope": "global"
    }
  ]
}
```

**Entry kinds:**
- **always** — Something to always do. Written to the `## Always` section of `rules.md`.
- **never** — Something to never do. Written to `## Never`.
- **when** — A conditional rule ("if X then Y"). Written to `## When`.
- **lesson** — A factual discovery. Written to `lessons.md` and optionally to `topics/{slug}.md`.
- **profile** — A fact about the user. Rewrites `profile.md` as a coherent snapshot.

**Scope determines where the memory lives:**
- **global** — Universal knowledge useful across any project. Written to `~/.anton/memory/`.
- **project** — Specific to this workspace. Written to `<project>/.anton/memory/`.

**Handler flow** (in `tools.py`):
1. `handle_memorize()` receives the tool call input
2. Each entry is converted to an `Engram` with `confidence="high"` and `source="user"` (explicit tool calls are trusted)
3. The encoding gate is checked per engram — in autopilot/copilot mode, high-confidence entries auto-encode
4. Any entries needing confirmation are queued in `session._pending_memory_confirmations`
5. The confirmation UI shows before the next user prompt

## Memory Modes — The Encoding Gate

Like the Locus Coeruleus-Norepinephrine system that controls how aggressively the brain writes new memories, Anton has three memory modes:

| Mode | Behavior | Brain Analog |
|------|----------|---|
| **autopilot** (default) | Anton decides what to save, no confirmation | High tonic NE — broad encoding |
| **copilot** | Auto-save high-confidence memories, confirm ambiguous ones after the answer | Moderate NE — selective encoding |
| **off** | Never save (still reads existing memory) | Suppressed — encoding blocked |

Configure via `/setup` > Memory, or the `ANTON_MEMORY_MODE` environment variable.

**The encoding gate logic** (in `cortex.py`):
```python
def encoding_gate(self, engram: Engram) -> bool:
    """Returns True if user confirmation is needed."""
    if self.mode == "autopilot": return False   # never confirm
    if self.mode == "off":       return False   # won't reach encoding anyway
    # copilot: auto-encode high confidence, confirm rest
    return engram.confidence != "high"
```

**Important design rule:** Memory confirmations are *never* shown during scratchpad execution or while Anton is composing an answer. They only appear after the user has received their full response, right before the next prompt. This ensures memory never interrupts the workflow.

**Confirmation UX** (copilot mode, after the answer):
```
Lessons learned from this session:
  1. [always] Call progress() before long API calls in scratchpad
  2. [lesson] CoinGecko rate-limits at 50 req/min

Save to memory? (y/n/pick numbers): 1
Saved 1 entries.
```

## Consolidation — Learning from Scratchpad Sessions

After a scratchpad session ends, the Consolidator runs in the background — like hippocampal replay during sleep.

### When It Triggers

The `should_replay()` method uses heuristics (no LLM call) to decide if a session is worth reviewing:

| Condition | Why |
|---|---|
| Any cell had an error | High-signal learning opportunity — errors are emotional |
| Session was long (5+ cells) | Rich experience with enough steps to mine patterns |
| Any cell was cancelled/killed | Something went wrong — worth understanding what |
| Session had < 2 cells | Skipped — too short to learn from |

### What It Does

1. **Compresses** the cell history into a compact summary — one line per cell with description, status, and first output line. Error cells include a code snippet.
2. **Sends** the summary to the fast coding model with a structured extraction prompt.
3. **Parses** the JSON response into `Engram` objects with `source="consolidation"`.
4. **Routes** through the encoding gate: high-confidence auto-encode, medium-confidence queue for confirmation.

### What the LLM Extracts

The consolidation prompt asks for two types of memories:

- **Rules**: behavioral patterns
  - "Always call progress() before long API calls in scratchpad"
  - "Never use time.sleep() in scratchpad cells"
  - "If fetching paginated data → use async + progress()"

- **Lessons**: factual knowledge
  - "CoinGecko free tier rate-limits at ~50 req/min"
  - "pandas read_csv needs encoding='utf-8-sig' for BOM files"
  - "Bitcoin price data via /coins/bitcoin/market_chart/range"

Each extracted memory includes a `scope` (global vs. project) and `confidence` (high vs. medium) so the encoding gate knows how to handle it.

## Identity Extraction — The Default Mode Network

Every 5 conversation turns, the Cortex passively checks if the user's message reveals identity-relevant information — like the Default Mode Network monitoring for self-relevant signals.

**How it works:**
1. A fast LLM call with the user's message and a prompt asking for identity facts
2. Returns a JSON array like `["Name: Jorge", "Timezone: PST"]`
3. Merges with existing profile: facts with the same key prefix (e.g., `Name:`) are replaced, not duplicated
4. Rewrites `~/.anton/memory/profile.md` atomically (exclusive file lock + write `.tmp` + rename)

This runs as a background `asyncio.create_task()` — never blocks the conversation. Only fires when `memory_mode != "off"`.

## Compaction — Synaptic Homeostasis

When memory files grow past 50 entries, the Cortex triggers compaction at session start — like the Synaptic Homeostasis Hypothesis (Tononi-Cirelli) where sleep prunes overgrown synapses.

**Compaction uses the coding model to:**
1. Remove exact duplicates
2. Merge entries that say the same thing differently (keep the clearest version)
3. Remove entries superseded by newer, more specific ones

**Safety guarantees:**
- Rewrite is atomic: write `.tmp`, then `os.rename`
- Uses exclusive file lock to prevent concurrent compaction
- If the LLM call fails, compaction is silently skipped — never corrupts existing memory
- Conservative by default: the prompt tells the model "when in doubt, keep the entry"

Compaction runs as a background `asyncio.create_task()` at session start — doesn't block the first user prompt.

## Reconsolidation — Legacy Migration

On first run after upgrading, Anton automatically migrates old memory formats:

| Legacy Format | Source | Destination |
|---|---|---|
| `.anton/context/*.md` | SelfAwarenessContext files | `memory/lessons.md` + `memory/topics/` |
| `.anton/learnings/*.md` | LearningStore files | `memory/lessons.md` + `memory/topics/` |

**Detection** (`needs_reconsolidation()`): runs when old directories exist with files AND new `memory/` directory doesn't have `rules.md`, `lessons.md`, or `profile.md`.

**Migration logic:**
- Context files: each `.md` file becomes a topic. Lines are split, bullets stripped, short fragments (<6 chars) skipped. Source is set to `"user"`.
- Learning files: the `index.json` is read for topic metadata. Content is split into individual facts. Source is set to `"consolidation"`.
- Old files are preserved — nothing is deleted.
- Runs synchronously at startup (fast, no LLM calls needed).

## Procedural Memory — The Skills System

Skills are Anton's **procedural memory** — reusable workflows the user has marked as worth remembering. Brain analog: the **striatum** stores motor programs and habits, learned action sequences that fire when a familiar context is recognized. Anton's skill system mirrors this: when the LLM sees a request that matches a stored skill, it pulls the procedure into working memory and follows it instead of reasoning from scratch.

Skills are intentionally distinct from engrams. **Engrams hold facts** ("CoinGecko rate-limits at 50 req/min"), are loaded into every prompt unconditionally because they're cheap, and live in `lessons.md` / `rules.md` / `profile.md`. **Skills hold whole procedures** ("how to summarize a CSV end-to-end"), are NOT loaded into every prompt, and the LLM explicitly retrieves them via the `recall_skill` tool when it recognizes a match. Both systems coexist in the brain — declarative and procedural memory are dissociable — and both coexist in Anton.

### Skill Directory Format

Each skill is a directory at `~/.anton/skills/<label>/` containing multi-stage representations that coexist (rather than graduating between stages):

```
~/.anton/skills/csv-summary/
├── SKILL.md           ← agentskills.io format: frontmatter (name, description, metadata) + body
├── references/        ← Stage 2: higher-level recipes/macros (emerges with use, v2+)
├── scripts/           ← Stage 3: runnable helper modules (emerges with reliability, v2+)
│   └── __init__.py
├── requirements.txt   ← Stage 3 dependencies (optional)
└── stats.json         ← per-stage usage counters
```

The three stages mirror the cortico-striatal-cerebellar gradient:
- **Stage 1 (declarative)** — what the prefrontal cortex reads when first learning a skill. Slow, deliberate, fully flexible.
- **Stage 2 (chunks)** — chunked sub-procedures (associative striatum). Faster than Stage 1, still LLM-mediated.
- **Stage 3 (code)** — runnable helpers (sensorimotor striatum). Cheapest, fastest, used when context is highly familiar.

The executive picks the highest stage that's reliable enough for the current context. v1 only ships Stage 1; the directory format pre-allocates the other slots so consolidation can fill them later without a migration.

### Naming: `label`, not `slug`

Each skill's unique identifier is its `label`. In cognitive psychology, a *label* is the declarative handle by which a procedural memory is addressed in working memory — the verbal token the executive holds when deciding to invoke a stored procedure. It's deliberately distinct from `name` (the human-readable display like "CSV Summary") and `description` (the retrieval cue describing the matching context).

### How Skills Get Created

Skills are created manually in v1 via the `/skill save` command. The user runs it after a successful task; the LLM reads the recent scratchpad cells + chat history and drafts the skill via `LLMClient.generate_object` with a `_SkillDraft` Pydantic schema:

```
you> Take a quick look at sales_q3.csv

anton> [opens scratchpad, loads pandas, infers schema, prints describe(), plots distributions]
       Here's what I found...

you> /skill save csv summary
anton> Drafting a skill from recent work…
       Saved skill csv-summary → ~/.anton/skills/csv-summary/
       Name: CSV Summary
       Description: User asks to explore, summarize, or describe a CSV file.
```

Automatic skill extraction (the consolidator promoting recurring scratchpad patterns into skills) is a v2/v3 feature. v1 deliberately uses manual curation to learn what "good" skills look like before automating.

### How Skills Get Used

On every turn, the system prompt includes a compact `## Procedural memory` section listing every available skill as one line: `- <label> — <description>`. The full procedures stay on disk. When the LLM recognizes a match, it calls the `recall_skill` tool:

```
{"name": "recall_skill", "input": {"label": "csv-summary"}}
```

The tool reads the SKILL.md body and returns it as the tool result, which the LLM follows as guidance for the rest of the turn. Each successful recall increments `stats.json::stage_1::recommended` — that's the classifier signal, mechanically captured without any LLM compliance dance.

Brain analog: the prefrontal cortex doesn't keep every skill loaded. It has fast pattern recognition that flags "I might need skill X" and *retrieves* the skill into working memory only when it actually needs it. The `recall_skill` tool is exactly this retrieval operation.

### Skill Slash Commands

| Command | What it does |
|---|---|
| `/skill save [name hint]` | LLM drafts a new skill from recent work and saves it |
| `/skills list` (or `/skill list`) | Show all saved skills with usage counters |
| `/skill show <label>` | Print one skill's procedure + stats (typo-tolerant via closest_match) |
| `/skill remove <label>` | Delete a skill from disk |

### Typo Recovery

When the LLM passes a label that doesn't exist (typos, guesses), `recall_skill` uses `closest_match()` to find the nearest existing slug via difflib and returns that skill's procedure with a warning. The `recommended` counter is credited to the *resolved* label, not the input — so `recall_skill('csv_sumary')` still increments `csv-summary` in the stats. The LLM gets useful behavior even when it gets the spelling wrong.

## Cerebellum — Supervised Error Learning

The Cerebellum is Anton's **supervised error-correction system**. It observes every scratchpad cell and learns from the ones that diverge from intent. Brain analog: the cerebellum's classical role is *forward modeling and error correction* — when a motor command is issued, the cerebellum predicts the expected sensory consequences, and when actual feedback arrives, it computes the prediction error and uses it to refine future commands.

For Anton, the "motor command" is a scratchpad cell. Before the cell runs, the LLM declares its intent via the `one_line_description` field on the scratchpad tool. That description IS the forward model — the prediction of what the cell should do. After the cell runs, we have its actual outcome (stdout, stderr, error). The Cerebellum compares the two and, when they diverge meaningfully, encodes a generalizable lesson that future code-generating LLM calls will see.

### Decoupling: Hooks Live in the Dispatcher, Not the Runtime

The Cerebellum operates via two observer hooks called from the scratchpad tool dispatcher (`handle_scratchpad`), NOT from the runtime backend itself:

```
handle_scratchpad (orchestration layer)
  ├─ build prelim Cell from tool input
  ├─ FIRE pre-execute observers ──→ Cerebellum.on_pre_execute (counter)
  ├─ pad.execute(code, ...)        (pure execution — runtime never sees observers)
  ├─ FIRE post-execute observers ─→ Cerebellum.on_post_execute (buffer if errored)
  └─ return formatted result
```

This decoupling is intentional. `LocalScratchpadRuntime`, `ScratchpadManager`, and any future `RemoteScratchpadRuntime` are **completely hook-agnostic** — they don't import the Cerebellum, they don't have hook attributes, they never call observers. When a remote runtime backend is added, it inherits zero hook code because there is none to inherit. The orchestration layer is the only place where execution and observation meet.

### Cheap Path

Most cells succeed cleanly. The Cerebellum's `on_post_execute` hook checks `cell.error is None and not cell.stderr.strip()` and returns immediately for clean cells — they're never buffered, no LLM call is ever made for them. Only cells that errored or warned trigger the buffer. The cost of running the Cerebellum on a happy-path turn is **zero LLM calls**.

### Batched Per-Turn Diff

When errored cells exist, they accumulate in a buffer across the turn. At end-of-turn, `_schedule_cerebellum_flush()` fires `Cerebellum.flush()` as a fire-and-forget background task. The user gets their reply immediately while the diff runs in parallel:

1. The buffered cells get formatted into a compact post-mortem prompt
2. One LLM call via `LLMClient.generate_object_code` (the cheap coding model) returns a `_DiffPassResult` Pydantic model with extracted lessons
3. Each lesson is wrapped as an `Engram` with `kind="lesson"`, `topic="scratchpad"`, `source="consolidation"`, and routed through `Cortex.encode()` — the same path manual lessons and the consolidator already use
4. Future scratchpad cells see those lessons via the existing `recall_scratchpad_wisdom()` injection into the scratchpad tool description

The cerebellum is a **producer** only — it generates new lesson entries for the existing storage and retrieval pipeline. There's no parallel storage system, no separate `corrections.md` file. Whatever the consolidator and `/memorize` write to, the cerebellum also writes to.

Brain analog: cerebellar plasticity (LTD at parallel-fiber → Purkinje cell synapses) operates in parallel with continued action, never blocking it. Lessons compound silently across turns; future cells avoid traps that earlier cells fell into.

### The Generated Lessons Look Like

```markdown
- For CSV files with mixed column types, pass low_memory=False to pd.read_csv. <!-- topic:scratchpad source:consolidation ts:2026-04-11 -->
- Wrap pd.to_datetime() calls in errors='coerce' when the input may contain malformed strings. <!-- topic:scratchpad source:consolidation ts:2026-04-11 -->
```

These appear in `lessons.md` like any other engram, carry the same metadata, and get pruned by the same compaction loop when memory grows past threshold.

## Anterior Cingulate Cortex — Pattern-Level Error Detection

The Cerebellum learns from a single failed cell. The ACC learns from a *pattern across multiple events* within one turn. Brain analog: the anterior cingulate cortex fires the *error-related negativity* (ERN) ~80 ms after the brain notices that an actual outcome diverged from an expected one. That signal flows downstream to the dopaminergic midbrain (reward prediction error), the striatum (action policy update), and the dorsolateral PFC (strategy adjustment).

Anton's ACC watches a turn unfold, classifies events as they arrive, and at end-of-turn extracts actionable lessons from patterns that fired more than once. Real failure modes it's designed to catch:

- **Scratchpad name switch** — the LLM started in `build_pres`, switched to `write_html`, then `pres1`. Each scratchpad name is a separate isolated environment; variables in one don't exist in another. Burned 8 rounds on recovery in the original session.
- **Oversized cell drops** — large HTML strings serialised to empty `code` in the tool-call schema. Repeated >5KB cells fail the same way; the LLM didn't realise the schema was clipping it.
- **Repeated tool error** — the same tool failed three times in a row with the same args (the publish-from-chat bug — three identical failed calls before pivoting).

### Status

Implemented at `anton/core/memory/acc.py` with 51 passing tests, plus 14 wiring tests at `tests/test_session_acc_init.py`. Layers 1 and 2 are wired into `ChatSession`:

  - **Layer 1 — passive learning.** Emit sites fire `acc.observe(...)` at scratchpad/tool/repair/cap-exhaust hooks; `_schedule_acc_flush()` runs at end-of-turn alongside the cerebellum flush. Detected lessons become `Engram` objects whose `kind` (always/never/when) was tagged by the detector, so they flow into the right section of `rules.md`. Future turns pick them up via the existing memory→system-prompt pipeline.
  - **Layer 2 — mid-turn nudges.** `at_round_n()` runs after each tool-call round; when a NEW detector fires (one nudge per detector per turn), the lesson text gets appended as a `text` block inside the same user-role message that carries the round's `tool_result` blocks. The LLM sees the alarm on its very next round, not on the next turn. Off by default — gated on `ANTON_ACC_MODE=active`.

Modes (env var `ANTON_ACC_MODE`, mirrors `ANTON_MEMORY_MODE`):

| Mode | Behaviour |
|---|---|
| `off` | ACC observes nothing — events drop at the safe-emit wrapper. Use to disable the feature entirely. |
| `passive` (default) | Layer 1 only. Lessons drain to memory at end-of-turn; next turn's system prompt picks them up. Adds zero surface to the turn loop. |
| `active` | Layer 1 + Layer 2. Lessons ALSO inject inline as text blocks in `tool_results` so the LLM sees them on the very next round. Stronger learning signal; more invasive. |

What is NOT yet wired (deliberate):
  - **Layer 3 — retrieval-scored rule ranking.** At system-prompt assembly, score each candidate rule by relevance to the current turn's context and load the top-K within the token budget. Per-rule retrieval counters age out rules that never make the cut. Pairs with optional outcome-tracking (did the rule reduce its target pattern after it landed?). Needs a small embedding index over rules + a ranker call on the load path.

### Vocabulary discipline

The ACC enforces a closed event vocabulary via the `EVENT_KINDS` frozenset. `observe()` raises `ValueError` on unknown kinds. A test asserts that every kind in `EVENT_KINDS` is read by at least one detector — the previous wide `KNOWN_PRODUCER_ONLY` allowlist was deliberately collapsed to a single justified entry (`tool_call`, reserved for a future `detect_orphaned_tool_call`). A second test guards against dropped kinds (`context_compaction`, `round_milestone`) silently reappearing.

### Event vocabulary (9 kinds)

| Kind | Detail shape | Read by |
|---|---|---|
| `scratchpad_call` | `{name, code_len, one_line_description}` | `detect_name_switch`, `detect_oversized_cell` |
| `scratchpad_result` | `{name, success, stdout_len, error}` | `detect_repeated_error_signature` |
| `scratchpad_empty_code` | `{name}` | `detect_oversized_cell` |
| `scratchpad_reset` | `{name, reason}` | `detect_reset_churn` |
| `scratchpad_killed` | `{name, reason}` | `detect_kill_loop` |
| `tool_call` | `{name, args_summary}` | *(producer-only — reserved for future `detect_orphaned_tool_call`)* |
| `tool_result` | `{name, success, error}` | `detect_repeated_tool_error`, `detect_repeated_error_signature` |
| `history_repair` | `{reason}` | `detect_repair_churn` |
| `cap_exhausted` | `{}` | `detect_cap_exhausted` |

### Detectors (9 pure functions over the event stream)

Detectors are stateless functions of `Sequence[Event] → Lesson | None`. Each detector that fires produces a one-sentence rule. Cross-detector dedupe at `at_end_of_turn()` collapses overlapping lessons.

| Detector | Fires when | Cognitive failure it learns from |
|---|---|---|
| `detect_name_switch` | ≥2 distinct scratchpad names in one turn | Identity sprawl across scratchpads — each name is a separate venv. |
| `detect_oversized_cell` | Observed empty-code drops OR ≥2 cells over ~5 KB | Silent schema truncation of large `code` strings. |
| `detect_repeated_tool_error` | ≥2 consecutive failures of same tool | Blind retry of same tool. |
| `detect_repeated_error_signature` | Same normalised error signature ≥3 times across any producers | Blind retry across tools / arg tweaks (generalises the publish-from-chat + gmail bug patterns). |
| `detect_reset_churn` | ≥2 scratchpad resets in one turn | Abandoning state instead of debugging in place. |
| `detect_kill_loop` | ≥2 cells killed on the same scratchpad name | Writing cells that hang — approach too heavy. |
| `detect_severity_climb` | Per-producer strictly-increasing severity run of length ≥3 ending ≥5 | Situation deteriorating without strategy change — ERN crossed threshold. |
| `detect_repair_churn` | ≥3 `history_repair` events in one turn | LLM generating malformed tool_use/result structurally; conversation derailing. |
| `detect_cap_exhausted` | A single `cap_exhausted` event | Round cap hit → mandatory post-mortem rather than silent retry. |

### Error-signature normalisation

`detect_repeated_error_signature` runs each error string through `_normalise_error_signature()` before counting — a cheap regex pass that collapses paths, integers, hex addresses, and short quoted tokens into placeholders. That way `"Refusing to save record for engine='gmail-1'"` and `"Refusing to save record for engine='gmail-2'"` hash to the same signature and the detector catches the underlying loop even when the LLM tweaks args between attempts.

### Producer, not storage

Like the cerebellum, the ACC is a *producer*. It does not own storage. Lessons it generates flow into the same Engram pipeline that the cerebellum and consolidator already use. The de-dupe predicate is caller-supplied (`has_similar_lesson`) so the wiring layer can choose substring, embedding, or semantic similarity without changing ACC internals.

## Structured Output — `LLMClient.generate_object`

Anton has a single primitive for getting structured data out of the LLM, used by the cerebellum, the consolidator, the cortex's identity/compaction passes, the connect collector, the skill drafter, and the custom-datasource flow. It lives at `anton/llm/client.py`:

```python
async def generate_object(
    self,
    schema_class,        # A Pydantic BaseModel subclass, or list[Model]
    *,
    system: str,
    messages: list[dict],
    max_tokens: int | None = None,
):
    """Forced-tool-call structured output via the planning provider."""
```

There's also a paired `generate_object_code(...)` that uses the cheap *coding* provider — appropriate for fast/cheap structured tasks like the cerebellum's post-mortem and the cortex's identity extraction.

### How It Works

1. The Pydantic model is converted to a JSON schema via `model_json_schema()`
2. A synthetic tool is built whose `input_schema` is that JSON schema
3. The LLM provider is called with `tool_choice={"type": "tool", "name": tool_name}` — this *forces* the LLM to call the tool rather than returning text
4. The tool's input dict is validated through `model_validate()` and returned as a typed instance

### Why It Beats Asking for JSON in Text

| Old pattern (text JSON) | New pattern (`generate_object`) |
|---|---|
| "Return ONLY valid JSON, no commentary, no markdown fences" | Forced tool_choice — the LLM cannot return text |
| Manual `json.loads()` with try/except | Pydantic `model_validate()` with structural validation |
| Strip markdown fences with regex (`_strip_json_fences`) | Never needed — there's no text response to strip |
| Defensive `if not isinstance(data, dict): return` checks | Pydantic catches type errors at the schema layer |
| Field-by-field `.get(key, default)` extraction | Typed attribute access on the validated instance |

### The Shared Helper

The schema-derivation and validation logic lives in exactly one place — `anton/llm/structured.py` — and is shared by both `LLMClient.generate_object` (main process, async) and `_ScratchpadLLM.generate_object` (subprocess bridge, sync). Two pure helper functions:

```python
def build_structured_tool(schema_class) -> tuple[dict, type, bool]:
    """Pydantic model → (tool_dict, validator_class, is_list)."""

def unwrap_structured_response(tool_call_input, validator_class, is_list):
    """LLM tool call input → validated typed Pydantic instance."""
```

This pattern is what every extraction call site uses. Adding a new one is mechanical: define a Pydantic model with `Field(description=...)` on each field, call `await session._llm.generate_object(MySchema, ...)`, wrap in try/except for graceful degradation. The field descriptions on the Pydantic model double as the LLM's instructions — there's no separate prompt explaining the schema.

### Where It's Used

| Module | Schema | Provider | Purpose |
|---|---|---|---|
| `connect_collector.py::extract_variables` | `_ExtractionResult` | planning | Parse free-form credential input into structured fields |
| `commands/skills.py::handle_skill_save` | `_SkillDraft` | planning | LLM drafts a skill from recent scratchpad work |
| `commands/datasource.py::handle_add_custom_datasource` | `_CustomDatasourceSpec` | planning | LLM identifies a custom datasource's auth fields |
| `cortex.py::_compact_file` | `_CompactionResult` | **coding** | Memory deduplication during synaptic homeostasis |
| `cortex.py::maybe_update_identity` | `_IdentityFacts` | **coding** | Default-mode identity extraction every 5 turns |
| `consolidator.py::replay_and_extract` | `_ConsolidatedLessons` | **coding** | Sleep-replay extraction of lessons from scratchpad sessions |
| `cerebellum.py::_run_diff` | `_DiffPassResult` | **coding** | Post-mortem error learning from cell failures |

The split between *planning* and *coding* providers preserves the original intent of each call site — anything that previously used `_llm.code()` now uses `generate_object_code` (cheap, fast model), and anything that previously used `_llm.plan()` now uses `generate_object` (planning model).

## Concurrency Safety

| Operation | Scope | Strategy |
|---|---|---|
| Normal writes (rules, lessons) | Global | `fcntl.flock(LOCK_EX)` on each file — append-only, no read-modify-write race |
| Normal writes | Project | No locking needed — one session per project |
| Compaction | Global | Exclusive lock + atomic rename (write `.tmp` then `os.rename`) |
| Identity updates | Global | Exclusive lock (full rewrite via `.tmp` + rename) |
| Concurrent compaction | Global | Other sessions skip — only one "sleeps" at a time |

## The Engram — Fundamental Unit of Memory

Every memory trace is represented as an `Engram` dataclass, defined in `anton/core/memory/base.py`:

```python
@dataclass
class Engram:
    text: str                                          # The memory content
    kind: "always" | "never" | "when" | "lesson" | "profile"  # Classification
    scope: "global" | "project"                        # Where to store it
    confidence: "high" | "medium" | "low" = "medium"   # Encoding gate signal
    topic: str = ""                                    # For lessons — topic slug
    source: "user" | "consolidation" | "llm" = "llm"  # Origin of the memory
```

Named for Karl Lashley's *engram* — the hypothesized physical substrate of a memory trace. Each engram flows through the system:

```
Source (user/LLM/consolidation/cerebellum/ACC)
  → Engram created
    → Cortex.encoding_gate() — needs confirmation?
      → yes: queued for user review before next prompt
      → no:  Cortex.encode() → routes to correct Hippocampus by scope
              → Hippocampus writes to disk with file locking
                → profile: full rewrite (atomic)
                → rule: insert into correct section of rules.md
                → lesson: append to lessons.md + optionally topics/{slug}.md
```

### `HippocampusProtocol` — the storage interface

`base.py` also defines a `HippocampusProtocol` — a `runtime_checkable` structural `typing.Protocol` describing the public contract of a single-scope memory store (the `recall_*` and `encode_*` methods listed in the next section). The concrete file-backed `Hippocampus` in `hippocampus.py` satisfies it automatically via structural sub-typing. The protocol exists so alternate backends (database-backed, cloud-synced) can be substituted without inheriting from the file-based implementation. This is the seam Enterprise adapters plug into.

## Module Reference

The long-term memory system lives under `anton/core/memory/`. A small set of legacy / orthogonal modules still lives at the top level under `anton/memory/`.

```
anton/core/memory/                 LONG-TERM MEMORY (brain-mapped modules)
├── base.py             Engram dataclass + HippocampusProtocol (structural backend interface)
├── hippocampus.py      Hippocampus class — file-backed implementation of the protocol
├── cortex.py           Cortex class (executive declarative-memory coordinator)
├── episodes.py         Episode + EpisodicMemory class
├── consolidator.py     Consolidator class (sleep-replay → Engrams)
├── cerebellum.py       Cerebellum class (per-cell supervised error learning)
├── acc.py              AnteriorCingulate class (turn-level pattern error detection)
└── skills.py           Skill, SkillStore, SkillStats — procedural memory storage layer

anton/memory/                      LEGACY / ORTHOGONAL (not the brain-mapped memory system)
├── reconsolidator.py   needs_reconsolidation() + reconsolidate() — one-time format migration
├── manage.py           MemoryManage class — handlers for /memory and /setup > Memory, MEMORY_MODES dict
├── history_store.py    HistoryStore — chat session persistence (transcripts on disk)
├── store.py            SessionStore — session list / metadata (different from history_store)
└── learnings.py        [legacy] LearningStore — pre-Hippocampus format, kept only for migration

anton/core/llm/
├── client.py           LLMClient with plan/code/generate_object/generate_object_code
├── structured.py       build_structured_tool + unwrap_structured_response (shared helper)
└── ...                 anthropic.py, openai.py, provider.py, prompt_builder.py, prompts.py

anton/core/tools/
├── recall_skill.py     RECALL_SKILL_TOOL — the LLM's procedural memory retrieval primitive
├── tool_handlers.py    handle_scratchpad with pre/post-execute observer firing
└── ...                 registry.py, tool_defs.py
```

### `base.py` — Engram + HippocampusProtocol

`Engram` is the fundamental memory-trace dataclass (see "The Engram" section above). `HippocampusProtocol` is a `runtime_checkable` structural Protocol defining the read/write contract of a single-scope memory store. Alternate backends (Enterprise, cloud-synced, database-backed) satisfy the protocol by shape — no inheritance from the file-based class needed.

### `hippocampus.py` — Storage Engine

The Hippocampus handles one scope (global OR project) and is the canonical file-backed implementation of `HippocampusProtocol`. It doesn't decide what to remember — it just reads and writes.

**Retrieval methods:**
| Method | Reads | Brain Analog |
|---|---|---|
| `recall_identity()` | `profile.md` | Medial PFC / Default Mode Network |
| `recall_rules()` | `rules.md` | Basal Ganglia + OFC |
| `recall_lessons(token_budget)` | `lessons.md` (budget-limited, most recent first) | Anterior Temporal Lobe |
| `recall_topic(slug)` | `topics/{slug}.md` | Cortical Association Areas |
| `recall_scratchpad_wisdom()` | "when" rules + scratchpad-related lessons + `topics/scratchpad-*.md` | Procedural memory |

**Encoding methods:**
| Method | Writes | Behavior |
|---|---|---|
| `encode_rule(text, kind, confidence, source)` | `rules.md` under correct `## Always/Never/When` section | Deduplicates. Uses file lock. |
| `encode_lesson(text, topic, source)` | `lessons.md` + optionally `topics/{slug}.md` | Deduplicates. Append-only with lock. |
| `rewrite_identity(entries)` | `profile.md` | Full rewrite (atomic via `.tmp` + rename). |

### `cortex.py` — Executive Coordinator

The Cortex manages two Hippocampus instances and orchestrates all declarative memory operations. It is also the encoding endpoint that the cerebellum and consolidator route their generated lessons through.

| Method | Purpose |
|---|---|
| `build_memory_context()` | Assemble memories for system prompt injection (~5800 token budget) |
| `get_scratchpad_context()` | Combine scratchpad wisdom from both scopes for tool description injection. **This is the channel the cerebellum's lessons flow through into future code generation.** |
| `encode(engrams)` | Route engrams to correct hippocampus by scope. Returns action log. Called by `/memorize`, the consolidator, and the cerebellum. |
| `encoding_gate(engram)` | Check if an engram needs user confirmation (mode-dependent) |
| `needs_compaction()` | Check if any file exceeds the threshold |
| `compact_all()` | LLM-assisted deduplication + merge on all oversized files. Uses `generate_object_code(_CompactionResult, ...)`. |
| `maybe_update_identity(message)` | Extract identity facts from user message via `generate_object_code(_IdentityFacts, ...)`. Background, fires every 5 turns. |

### `episodes.py` — Episodic Memory

The EpisodicMemory handles raw conversation logging and recall.

| Method | Purpose |
|---|---|
| `start_session()` | Create a new JSONL file, return session ID |
| `log(episode)` | Append an Episode to the current session file (fire-and-forget) |
| `log_turn(turn, role, content, **meta)` | Convenience wrapper — builds Episode and calls log() |
| `recall(query, max_results, days_back)` | Search all JSONL files for matching episodes (newest first) |
| `recall_formatted(query, **kwargs)` | Return human-readable string of matching episodes |
| `session_count()` | Count the number of session JSONL files |

### `consolidator.py` — Scratchpad Replay

| Method | Purpose |
|---|---|
| `should_replay(cells)` | Heuristic check: errors, 5+ cells, or cancellations → True |
| `replay_and_extract(cells, llm)` | Compress cells → `generate_object_code(_ConsolidatedLessons, ...)` → return Engrams |

### `cerebellum.py` — Supervised Error Learning (per-cell)

| Method | Purpose |
|---|---|
| `on_pre_execute(cell)` | Pre-execute hook called by `handle_scratchpad`. Counter only in v1. |
| `on_post_execute(cell)` | Post-execute hook. Cheap path skips clean cells; errored/warning cells get buffered. |
| `flush()` | Run the batched diff pass on all buffered cells, encode lessons via Cortex, clear buffer. Fire-and-forget at end-of-turn. |
| `reset()` | Drop the buffer without encoding (used when a turn is cancelled mid-flight). |
| `buffered_count` | Number of cells waiting for the next flush. |
| `_run_diff(cells)` | Internal: send buffered cells to `generate_object_code(_DiffPassResult, ...)` and return validated lessons. |
| `_encode_lessons(lessons)` | Internal: wrap lessons as Engrams (`kind="lesson"`, `topic="scratchpad"`, `source="consolidation"`) and route through `Cortex.encode()`. |

### `acc.py` — Anterior Cingulate Cortex (turn-level)

Pattern-level error detection across a turn. Pure detectors as free functions plus a small stateful `AnteriorCingulate` that holds the event stream.

| Element | Purpose |
|---|---|
| `Event` | Dataclass with `kind`, `severity`, `detail`, `round_idx`. The atomic observation. |
| `Lesson` | Dataclass with `rule`, `triggers`, `detector`. What a detector emits. |
| `EVENT_KINDS` | Frozenset of 9 canonical event-kind strings. `observe()` rejects unknown kinds. |
| `DETECTORS` | Tuple of 9 pure detector functions — see the table in the ACC section above. |
| `AnteriorCingulate.observe(kind, ...)` | Append an `Event`. Rejects unknown kinds. |
| `AnteriorCingulate.at_end_of_turn(has_similar_lesson=...)` | Run all detectors, dedupe via caller-supplied predicate, return new `Lesson` list. |
| `AnteriorCingulate.clear()` | Drop the event stream (between turns). |
| `AnteriorCingulate.events` / `event_kind_counts` | Read-only views for inspection and testing. |

Tests live at `tests/test_acc.py` (44 tests, 4 layers: pure-function detectors → state tests → JSON-fixture replay → vocabulary discipline). Fixtures at `tests/fixtures/acc/{name_switch,oversized_cell,publish_failure_loop,reset_churn,kill_loop}.json`.

### `skills.py` — Procedural Memory Store

| Method | Purpose |
|---|---|
| `SkillStore.list_all()` | Return every loadable skill, sorted by label. |
| `SkillStore.list_summaries()` | Lightweight listing — `[{"label": "...", "name": "...", "description": "..."}]`. Used by the prompt builder to inject the procedural-memory section without loading any declarative content. |
| `SkillStore.load(label)` | Read a single skill by label. Returns None if absent or malformed.                                                                                                                           |
| `SkillStore.save(skill)` | Write the skill directory. Creates `SKILL.md`, `stats.json`. Never wipes accumulated counters.                                                                                               |
| `SkillStore.delete(label)` | Remove a skill directory.                                                                                                                                                                    |
| `SkillStore.increment_recommended(label, *, stage)` | Atomic-ish bump of the per-stage `recommended` counter (called by `recall_skill`).                                                                                                           |
| `SkillStore.closest_match(bad_label, *, cutoff=0.6)` | Difflib-based fuzzy match for typo recovery.                                                                                                                                                 |
| `make_unique_label(base, store)` | Generate a slug that doesn't collide with any existing skill (`csv-summary`, `csv-summary_2`, ...).                                                                                          |
| `slugify(text)` | Normalize arbitrary text into a kebab-case identifier.                                                                                                                                       |

### `tools/recall_skill.py` — Procedural Memory Retrieval Tool

The LLM-facing tool that pulls a skill into working memory. Lives alongside the other tool defs but is the only tool whose handler reads `session._skill_store`.

| Element | Purpose |
|---|---|
| `RECALL_SKILL_TOOL` | The `ToolDef` registered with the session — name, description, input_schema, handler. |
| `handle_recall_skill(session, tc_input)` | Resolve the label (with closest_match fallback for typos), increment the per-stage `recommended` counter, return a formatted procedure to the LLM as the tool result. |

### `llm/structured.py` — Shared Schema Helper

Two pure helper functions for forced-tool-call structured output. Used by both `LLMClient.generate_object` (main process, async) and `_ScratchpadLLM.generate_object` (subprocess bridge, sync) — they share this code via lazy imports so neither runtime forces pydantic at module load time.

| Function | Purpose |
|---|---|
| `build_structured_tool(schema_class)` | Pydantic model (or `list[Model]`) → `(tool_dict, validator_class, is_list)`. The `tool_dict` is ready to pass as `tools=[...]` with `tool_choice={"type": "tool", "name": tool_dict["name"]}`. |
| `unwrap_structured_response(tool_call_input, validator_class, is_list)` | Validate the LLM's tool call input via Pydantic and unwrap the wrapper if it was a list. Raises `pydantic.ValidationError` on schema drift. |

### `reconsolidator.py` — Legacy Migration

| Function | Purpose |
|---|---|
| `needs_reconsolidation(project_dir)` | Check if old formats exist and new ones don't |
| `reconsolidate(project_dir)` | Migrate `.anton/context/` and `.anton/learnings/` → `.anton/memory/` |

## Integration Points in chat.py

The memory + skills + cerebellum + ACC systems are wired into `ChatSession` (defined in `anton/chat.py` for the CLI entry-points; the runtime class actually lives in `anton/core/session.py`, with chat-loop wiring at the top level in `anton/chat.py`). The cerebellum wires via the dispatcher's scratchpad observer list; the ACC wires via direct `session._acc.observe(...)` calls at each emit site (broader emit footprint than scratchpad alone — also tools, history-repair, round-cap).

### ACC emit sites (Layer 1)

| Event kind | Emit site | File |
|---|---|---|
| `scratchpad_call` | After args validation in `handle_scratchpad` exec branch | `core/tools/tool_handlers.py` |
| `scratchpad_result` | After `pad.execute()` returns a non-killed cell | `core/tools/tool_handlers.py` |
| `scratchpad_empty_code` | When `prepare_scratchpad_exec` rejects the call | `core/tools/tool_handlers.py` |
| `scratchpad_reset` | After `pad.reset()` in the reset action | `core/tools/tool_handlers.py` |
| `scratchpad_killed` | After `pad.execute()` returns a cell whose `error` starts with `Cancelled`/`Cell timed out`/`Cell killed` | `core/tools/tool_handlers.py` |
| `tool_call` | At top of per-tc loop in `_stream_and_handle_tools` | `core/session.py` |
| `tool_result` | After result_text is finalized, before `tool_results.append` | `core/session.py` |
| `history_repair` | After `_seal_dangling_tool_uses` actually inserts synthetic blocks | `core/session.py` |
| `cap_exhausted` | When `tool_round > self._max_tool_rounds` | `core/session.py` |

### End-of-turn drain

`_schedule_acc_flush()` lives next to `_schedule_cerebellum_flush()` and runs at the same two spots — end of `turn()` and end of `turn_stream()`. Fire-and-forget: detectors are pure (no LLM call) and the only async work is `cortex.encode()`, but we still wrap the encode in `asyncio.create_task` so file I/O doesn't block the user-facing reply.

Each `Lesson` becomes an `Engram(text=rule, kind=lesson.kind, scope="global", confidence="high", source="consolidation")`. The `kind` is whatever the detector tagged (`always`/`never`/`when`) — no string-matching at the wiring layer. Lessons land in `~/.anton/memory/rules.md` under the corresponding `## Always` / `## Never` / `## When` section, and the next turn's system prompt picks them up via the existing `cortex.build_memory_context()` pipeline.

### Mid-turn nudge (`_acc_maybe_nudge`)

Called by `_stream_and_handle_tools` immediately after `tool_results` is built and before that user message gets appended to history. When `ANTON_ACC_MODE == "active"`:

  1. Runs `acc.at_round_n()` — re-evaluates every detector against the events buffered so far this turn and returns only lessons whose detector hasn't already nudged this turn.
  2. For each newly-fired lesson, appends `{"type": "text", "text": "[Anton self-check — <detector>] <rule>"}` to the `tool_results` content array.
  3. The LLM sees those text blocks alongside the tool_result blocks on its very next round.

One nudge per detector per turn — re-stating the same alarm round after round would inflate history without changing behaviour. Cleared on the same `clear()` boundary the event buffer uses. The mid-turn path deliberately does NOT consult `has_similar_lesson`: if a rule is already in `rules.md` but the LLM is violating it right now, repeating the rule inline is the whole point.

### `_acc_observe` safe-emit wrapper

Every emit site calls `session._acc_observe(kind, detail, ...)` rather than touching `session._acc` directly. This wrapper:
  - Returns silently when the ACC isn't attached (defensive).
  - Returns silently when the cortex is disabled (`mode == "off"`) — observation without persistence is pointless.
  - Catches `ValueError` from `observe()` on unknown kinds so emit-site drift never breaks a turn.

### De-dupe predicate

The ACC is constructed with `has_similar_lesson=_acc_has_similar`, a closure that does a cheap substring match against the current `rules.md` content. Prevents the same lesson being re-encoded on every turn. Embedding similarity is a v2 upgrade.

```
1. _chat_loop() startup:
   → Creates Cortex(global_hc=Hippocampus(global_dir), project_hc=Hippocampus(project_dir), mode, llm)
   → Creates EpisodicMemory(episodes_dir, enabled=settings.episodic_memory)
   → Starts episodic session if enabled
   → Runs reconsolidation if needed
   → Fires background compaction if needed

2. ChatSession.__init__():
   → Stores cortex as self._cortex
   → Stores episodic as self._episodic
   → Initializes self._skill_store = SkillStore() (procedural memory)
   → Initializes self._cerebellum = Cerebellum(cortex=self._cortex, llm=self._llm)
   → Initializes self._scratchpad_observers = [self._cerebellum]
   → Initializes self._pending_memory_confirmations = []

3. ChatSession._build_system_prompt():
   → Calls cortex.build_memory_context()  →  injected before anton.md
   → Passes self._skill_store to prompt builder
   → Builder appends "## Procedural memory" section listing all available skills

4. ChatSession._build_tools():
   → Calls cortex.get_scratchpad_context()  →  appended to scratchpad tool desc
   → Includes MEMORIZE_TOOL in tool list
   → Includes RECALL_TOOL when episodic memory is enabled
   → Includes RECALL_SKILL_TOOL (always available — no-op if no skills saved)

5. Tool dispatch (tools.py):
   → "memorize" → handle_memorize() → cortex.encode()
   → "recall" → handle_recall() → episodic.recall_formatted()
   → "recall_skill" → handle_recall_skill() → SkillStore.load() + increment_recommended()
   → "scratchpad" exec → handle_scratchpad() fires pre/post observers around pad.execute()

6. handle_scratchpad (tool_handlers.py) — observer dispatch:
   → Build prelim Cell from tool input (code + description + estimated_time)
   → _fire_pre_execute(session, prelim_cell) → cerebellum.on_pre_execute (counter)
   → pad.execute(...) — pure execution, runtime never sees observers
   → _fire_post_execute(session, cell) → cerebellum.on_post_execute (buffer if errored)

7. turn_stream():
   → Logs user input to episodic memory (before LLM call)
   → Logs assistant response to episodic memory (after LLM call)

8. _stream_and_handle_tools() tool loop:
   → Logs each tool_call to episodic memory
   → Logs each tool_result to episodic memory
   → Logs scratchpad cell output to episodic memory
   → _maybe_consolidate_scratchpads() → background asyncio.create_task

9. End of turn (turn / turn_stream):
   → Every 5 turns → cortex.maybe_update_identity() as background task
   → _schedule_cerebellum_flush() → fire-and-forget background task
     → Runs cerebellum diff on all buffered cells
     → Encodes extracted lessons via cortex.encode()
     → Lessons appear in next turn's scratchpad tool description automatically

10. Before user prompt (_chat_loop):
    → Show pending memory confirmations → user approves/rejects/picks

11. Slash commands for skills (chat.py):
    → /skill save [name hint] → handle_skill_save() → drafts via generate_object → SkillStore.save()
    → /skills or /skill list → handle_skills_list() → tabular display of skills + counters
    → /skill show <label> → handle_skill_show() → full procedure + stats
    → /skill remove <label> → handle_skill_remove() → SkillStore.delete()

12. /setup wizard (sub-menu):
    → Option 1: Models — provider, API key, planning & coding models
    → Option 2: Memory — memory mode (autopilot/copilot/off) + episodic toggle
    → Persisted to ANTON_MEMORY_MODE and ANTON_EPISODIC_MEMORY in .anton/.env

13. /memory (read-only dashboard):
    → Shows semantic memory counts (global/project rules, lessons, topics)
    → Shows episodic memory status (ON/OFF) and session count
    → No configuration prompts — directs to /setup > Memory

14. _rebuild_session():
    → Updates cortex._llm and cortex.mode when settings change
    → Propagates episodic memory instance
    → Re-creates cerebellum if llm or cortex changed
```

## Context Budget Summary

| Section | Brain Analog | Budget | Loaded When |
|---------|---|--------|-------------|
| Identity | mPFC / DMN | ~300 tokens | Always (system prompt) |
| Global rules | Basal Ganglia | ~1500 tokens | Always (system prompt) |
| Project rules | Basal Ganglia | ~1500 tokens | Always (system prompt) |
| Global lessons | ATL semantics | ~1000 tokens | Always (most recent first) |
| Project lessons | ATL semantics | ~1000 tokens | Always (most recent first) |
| Scratchpad wisdom | Procedural priming | ~500 tokens | Scratchpad active (tool desc). Cerebellum-generated lessons flow through here. |
| Procedural memory list | Striatum (skill labels) | ~50 tokens per skill (compact list) | Always — when any skills are saved. Full procedures NOT loaded; only labels + description. |
| Topic files | Cortical association | Unlimited | On demand |
| Skill procedures | Striatum (full skills) | Variable per skill | On demand (`recall_skill` tool) — only when the LLM recognizes a match |
| Episodic recall | MTL episodic | Variable | On demand (`recall` tool) |
| **Total in prompt** | **Working memory** | **~5800 tokens + ~50/skill** | ~3% of 200K context |

The procedural memory list scales linearly with the number of saved skills but stays cheap (~50 tokens each — slug + one-line `description`). The full skill procedures are *paid for only when retrieved*, the same way the prefrontal cortex doesn't keep every procedural memory loaded — it has fast pattern recognition that flags relevance and pulls the full procedure from storage on demand.
