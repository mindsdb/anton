# Inside Anton 

## Introduction 

Reading the book `How to create a mind` by Ray Kurzweil, back in 2015, a few years before we started MindsDB, I left believing, that one could programatically create Mind mirroring the brain's building blocks, however, after trying I came to the conclusion that one foundamental block needed to happen, and `Anticipation Block architecture`, you can read more about this thought [here](https://torrmal.github.io/2015/12/29/anticipation-loop/).

Well, turns out the world did build an architecture like that just a few years later, which is what we know as transformers, and LLMs today (2026) are at a point where the ideas that were seeded in us by 'How to create a Mind' are very viable from an implementation point of view. So 11 years later this try two ;)

## A mini Mind

It is probably obvious now, but Anton has a brain-inspired architecture, and the more we build it the more it resembles/mirrors functional parts of the brain.  On the other hand we also understand that people don't need to know anything about the brain to play with Anton, so we mapped some of the places/files where users can have inputs, or investigate what's up, to names that make more sense than the scientific name of that function of the brain.

The current implementation has three blocks:

| Brain Region                 | Function                                         | Anton Equivalent                                              |
|------------------------------|--------------------------------------------------|---------------------------------------------------------------|
| Prefrontal Cortex (PFC)      | Executive control, planning, the "inner voice"  | Orchestrator — decides what to work on, how, and when to stop |
| Working Memory (dlPFC)       | Temporary reasoning space, ~4 slots             | Scratchpads — isolated reasoning environments                 |
| Hippocampus                  | Episodic memory, records experiences            | Experience Store — logs of problem + context + solution       |



## Architecture of Anton

These three parts work in a very simple way:

```
  ┌────────────────────────────────────────────────────┐
  │              EXECUTIVE (the orchestrator)          │
  │                                                    │
  │  On new problem:                                   │
  │    1. Check SKILL LIBRARY → match?                 │
  │       YES → deploy skill's template scratchpad     │
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
  └────────────────────┬─────────────────────────────────┘
                       │ on success
                       ▼
  ┌──────────────────────────────────────────────────────┐
  │         EXPERIENCE STORE (hippocampus)               │
  │                                                      │
  │  Each entry:                                         │
  │  {                                                   │
  │    problem_signature: "...",                         │
  │    context: { what tools, what domain, what input }, │
  │    scratchpad_trace: [ step1, step2, ... ],          │
  │    outcome: success | failure,                       │
  │    cost: tokens/time spent,                          │
  │    salience: how important/novel was this            │
  │  }                                                   │
  │                                                      │
  │  Searchable by similarity (embeddings)               │
  └──────────────────────────────────────────────────────┘

```

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
└── memory/
    ├── profile.md                     Identity — who the user is
    ├── rules.md                       Always/never/when behavioral rules
    ├── lessons.md                     Semantic facts from experience
    └── topics/                        Deep domain expertise
        └── *.md

<project>/.anton/                      PROJECT scope (workspace-specific)
├── memory/
│   ├── rules.md                       Project-specific rules
│   ├── lessons.md                     Project-specific knowledge
│   └── topics/
│       └── *.md
├── episodes/                          EPISODIC MEMORY (conversation archive)
│   ├── 20260227_143052.jsonl          One file per session (YYYYMMDD_HHMMSS)
│   └── 20260228_091522.jsonl
├── anton.md                           User-written project context (unchanged)
└── .env                               Secrets (unchanged)
```

Profile (`profile.md`) is global-only — identity is singular. Rules and lessons exist at both scopes. `anton.md` stays as the user-written instruction file and is not managed by the memory system.

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

## Concurrency Safety

| Operation | Scope | Strategy |
|---|---|---|
| Normal writes (rules, lessons) | Global | `fcntl.flock(LOCK_EX)` on each file — append-only, no read-modify-write race |
| Normal writes | Project | No locking needed — one session per project |
| Compaction | Global | Exclusive lock + atomic rename (write `.tmp` then `os.rename`) |
| Identity updates | Global | Exclusive lock (full rewrite via `.tmp` + rename) |
| Concurrent compaction | Global | Other sessions skip — only one "sleeps" at a time |

## The Engram — Fundamental Unit of Memory

Every memory trace is represented as an `Engram` dataclass:

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
Source (user/LLM/consolidation)
  → Engram created
    → Cortex.encoding_gate() — needs confirmation?
      → yes: queued for user review before next prompt
      → no:  Cortex.encode() → routes to correct Hippocampus by scope
              → Hippocampus writes to disk with file locking
                → profile: full rewrite (atomic)
                → rule: insert into correct section of rules.md
                → lesson: append to lessons.md + optionally topics/{slug}.md
```

## Module Reference

```
anton/memory/
├── hippocampus.py      Engram + Hippocampus class
├── cortex.py           Cortex class
├── episodes.py         Episode + EpisodicMemory class
├── consolidator.py     Consolidator class
├── reconsolidator.py   needs_reconsolidation() + reconsolidate() functions
├── learnings.py        [legacy] LearningStore — replaced by Hippocampus
└── store.py            SessionStore — session history (orthogonal to long-term memory)
```

### `hippocampus.py` — Storage Engine

The Hippocampus handles one scope (global OR project). It doesn't decide what to remember — it just reads and writes.

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

The Cortex manages two Hippocampus instances and orchestrates all memory operations.

| Method | Purpose |
|---|---|
| `build_memory_context()` | Assemble memories for system prompt injection (~5800 token budget) |
| `get_scratchpad_context()` | Combine scratchpad wisdom from both scopes for tool description injection |
| `encode(engrams)` | Route engrams to correct hippocampus by scope. Returns action log. |
| `encoding_gate(engram)` | Check if an engram needs user confirmation (mode-dependent) |
| `needs_compaction()` | Check if any file exceeds 50 entries |
| `compact_all()` | LLM-assisted deduplication + merge on all oversized files |
| `maybe_update_identity(message)` | Extract identity facts from user message (fast model, background) |

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
| `replay_and_extract(cells, llm)` | Compress cells → fast LLM call → parse JSON → return Engrams |

### `reconsolidator.py` — Legacy Migration

| Function | Purpose |
|---|---|
| `needs_reconsolidation(project_dir)` | Check if old formats exist and new ones don't |
| `reconsolidate(project_dir)` | Migrate `.anton/context/` and `.anton/learnings/` → `.anton/memory/` |

## Integration Points in chat.py

The memory system is wired into `ChatSession` and `_chat_loop()`:

```
1. _chat_loop() startup:
   → Creates Cortex(global_dir, project_dir, mode, llm)
   → Creates EpisodicMemory(episodes_dir, enabled=settings.episodic_memory)
   → Starts episodic session if enabled
   → Runs reconsolidation if needed
   → Fires background compaction if needed

2. ChatSession.__init__():
   → Stores cortex as self._cortex
   → Stores episodic as self._episodic
   → Initializes self._pending_memory_confirmations = []

3. ChatSession._build_system_prompt():
   → Calls cortex.build_memory_context()  →  injected before anton.md

4. ChatSession._build_tools():
   → Calls cortex.get_scratchpad_context()  →  appended to scratchpad tool desc
   → Includes MEMORIZE_TOOL in tool list
   → Includes RECALL_TOOL when episodic memory is enabled

5. Tool dispatch (tools.py):
   → "memorize" → handle_memorize() → cortex.encode()
   → "recall" → handle_recall() → episodic.recall_formatted()

6. turn_stream():
   → Logs user input to episodic memory (before LLM call)
   → Logs assistant response to episodic memory (after LLM call)

7. _stream_and_handle_tools() tool loop:
   → Logs each tool_call to episodic memory
   → Logs each tool_result to episodic memory
   → Logs scratchpad cell output to episodic memory
   → _maybe_consolidate_scratchpads() → background asyncio.create_task

8. After turn (turn_stream):
   → Every 5 turns → cortex.maybe_update_identity() as background task

9. Before user prompt (_chat_loop):
   → Show pending memory confirmations → user approves/rejects/picks

10. /setup wizard (sub-menu):
    → Option 1: Models — provider, API key, planning & coding models
    → Option 2: Memory — memory mode (autopilot/copilot/off) + episodic toggle
    → Persisted to ANTON_MEMORY_MODE and ANTON_EPISODIC_MEMORY in .anton/.env

11. /memory (read-only dashboard):
    → Shows semantic memory counts (global/project rules, lessons, topics)
    → Shows episodic memory status (ON/OFF) and session count
    → No configuration prompts — directs to /setup > Memory

12. _rebuild_session():
    → Updates cortex._llm and cortex.mode when settings change
    → Propagates episodic memory instance
```

## Context Budget Summary

| Section | Brain Analog | Budget | Loaded When |
|---------|---|--------|-------------|
| Identity | mPFC / DMN | ~300 tokens | Always (system prompt) |
| Global rules | Basal Ganglia | ~1500 tokens | Always (system prompt) |
| Project rules | Basal Ganglia | ~1500 tokens | Always (system prompt) |
| Global lessons | ATL semantics | ~1000 tokens | Always (most recent first) |
| Project lessons | ATL semantics | ~1000 tokens | Always (most recent first) |
| Scratchpad wisdom | Procedural memory | ~500 tokens | Scratchpad active (tool desc) |
| Topic files | Cortical association | Unlimited | On demand |
| Episodic recall | MTL episodic | Variable | On demand (recall tool) |
| **Total in prompt** | **Working memory** | **~5800 tokens** | ~3% of 200K context |
