---
title: Skills internals
description: The procedural memory implementation — multi-stage skill directories, drafting via structured output, the recall_skill tool, and the SkillStore API.
---

# Skills internals

Skills are Anton's **procedural memory** — the striatum in the
[brain mapping](/developer/brain-mapping). The storage layer is
`anton/core/memory/skills.py` (`Skill`, `SkillStore`, `SkillStats`), the
retrieval tool is `anton/core/tools/recall_skill.py`, and the drafting flow is
`anton/commands/skills.py`. User-facing docs: [Skills](/teach/skills).

## Engrams vs skills

The two memory kinds are intentionally distinct (as declarative and procedural
memory are dissociable in the brain):

- **Engrams hold facts** ("CoinGecko rate-limits at 50 req/min"). Loaded into
  every prompt unconditionally because they're cheap. Live in `lessons.md` /
  `rules.md` / `profile.md`.
- **Skills hold whole procedures** ("how to summarize a CSV end-to-end").
  NOT loaded into every prompt — the LLM explicitly retrieves them via the
  `recall_skill` tool when it recognizes a match.

## Multi-stage directory format

Each skill is a directory at `~/.anton/skills/<label>/` containing multi-stage
representations that **coexist** (rather than graduating between stages):

```
~/.anton/skills/csv_summary/
├── meta.json          ← label, name, description, when_to_use, provenance, presence flags
├── declarative.md     ← Stage 1: step-by-step procedure the LLM reads (always present)
├── chunks.md          ← Stage 2: higher-level recipes/macros (emerges with use, v2+)
├── code/              ← Stage 3: runnable helper modules (emerges with reliability, v2+)
│   └── __init__.py
├── requirements.txt   ← Stage 3 dependencies (optional)
└── stats.json         ← per-stage usage counters
```

The three stages mirror the cortico-striatal-cerebellar gradient:

- **Stage 1 (declarative)** — what the prefrontal cortex reads when first
  learning a skill. Slow, deliberate, fully flexible.
- **Stage 2 (chunks)** — chunked sub-procedures (associative striatum). Faster
  than Stage 1, still LLM-mediated.
- **Stage 3 (code)** — runnable helpers (sensorimotor striatum). Cheapest,
  fastest, used when context is highly familiar.

The executive picks the highest stage reliable enough for the current context.
**v1 only ships Stage 1**; the directory format pre-allocates the other slots so
consolidation can fill them later without a migration.

## Naming: `label` vs `name` vs `when_to_use`

A skill's unique identifier is its **`label`** (snake_case slug, e.g.
`csv_summary`). In cognitive psychology, a *label* is the declarative handle by
which a procedural memory is addressed in working memory — the verbal token the
executive holds when deciding to invoke a stored procedure. It is deliberately
distinct from:

- **`name`** — the human-readable display ("CSV Summary"),
- **`when_to_use`** — the retrieval cue describing the matching context. This
  is the most important field: it's the one line the LLM sees in every prompt.

## Creation: `/skill save` and `_SkillDraft`

Skills are created manually in v1 via `/skill save [name hint]`
(`handle_skill_save` in `anton/commands/skills.py`). The handler bundles recent
scratchpad cells + chat history into a prompt and calls
`session._llm.generate_object(_SkillDraft, ...)` — forced-tool-call structured
output (see [LLM dispatch](/developer/llm-dispatch)). The `_SkillDraft` Pydantic
model has five fields, and the `Field(description=...)` strings double as the
LLM's instructions:

| Field | Required | Guidance baked into the schema |
|---|---|---|
| `label` | yes | snake_case, 2-4 words, captures the essence |
| `name` | yes | Human-readable display name |
| `description` | no | One sentence on what the skill does |
| `when_to_use` | yes | One sentence on when it applies — "what the classifier shows to the LLM next time" |
| `declarative_md` | yes | Numbered steps, specific decisions, written as instructions for a future agent — not a retrospective |

The result is saved via `SkillStore.save()`, with `make_unique_label()` avoiding
slug collisions (`csv_summary`, `csv_summary_2`, ...). Automatic skill
extraction (the consolidator promoting recurring patterns into skills) is a
v2/v3 feature — v1 deliberately uses manual curation.

## Retrieval: prompt section + `recall_skill`

On every turn, the prompt builder (`_build_procedural_memory_section` in
`anton/core/llm/prompt_builder.py`) injects a compact
`## Procedural memory (skills available)` section: one line per skill —
`` `label` — when_to_use `` — built from `SkillStore.list_summaries()` without
loading any declarative content (~50 tokens per skill). The full procedures
stay on disk.

When the LLM recognizes a match, it calls the `recall_skill` tool. The handler
(`handle_recall_skill` in `anton/core/tools/recall_skill.py`):

1. Loads the skill by label via `SkillStore.load(label)`.
2. On miss, tries `closest_match()` (typo recovery — see below); if nothing is
   close, returns the available labels so the LLM can self-correct.
3. Increments `stats.json` `stage_1.recommended` for the **resolved** label —
   a precise, mechanical classifier signal with no LLM compliance dance.
4. Returns the formatted Stage 1 procedure as the tool result, which the LLM
   follows as guidance for the rest of the turn.

The tool is registered unconditionally in `ChatSession._build_core_tools()` —
it's a no-op when no skills are saved.

### Typo recovery

When the LLM passes a label that doesn't exist, `closest_match()` finds the
nearest existing slug via difflib (cutoff 0.6) and returns that skill's
procedure with a warning prepended. The `recommended` counter is credited to
the *resolved* label, not the input — `recall_skill('csv_sumary')` still
increments `csv_summary`.

## `SkillStore` API

| Method | Purpose |
|---|---|
| `list_all()` | Every loadable skill, sorted by label |
| `list_summaries()` | Lightweight listing — label, name, when_to_use only. Used by the prompt builder |
| `load(label)` | Read one skill. Returns None if absent or malformed |
| `save(skill)` | Write the directory: `meta.json`, `declarative.md`, `stats.json`. Never wipes accumulated counters |
| `delete(label)` | Remove a skill directory |
| `increment_recommended(label, *, stage)` | Atomic-ish bump of the per-stage `recommended` counter (called by `recall_skill`) |
| `closest_match(bad_label, *, cutoff=0.6)` | Difflib fuzzy match for typo recovery |
| `make_unique_label(base, store)` | Collision-free slug generation |
| `slugify(text)` | Normalize arbitrary text into a snake_case identifier |

## Slash commands

| Command | What it does |
|---|---|
| `/skill save [name hint]` | LLM drafts a new skill from recent work and saves it |
| `/skills list` (or `/skill list`) | Show all saved skills with usage counters |
| `/skill show <label>` | Print one skill's procedure + stats (typo-tolerant via `closest_match`) |
| `/skill remove <label>` | Delete a skill from disk |

See [Slash commands](/reference/slash-commands) for the full command reference.
