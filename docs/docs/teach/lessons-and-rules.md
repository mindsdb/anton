---
title: Lessons and rules
description: The rules.md, lessons.md, and topics files — what they contain, how entries get there, and how to manage them from chat.
---

# Lessons and rules

Rules and lessons are the heart of Anton's everyday memory. Both are plain markdown files that exist at two scopes — global (`~/.anton/memory/`) and per-project (`.anton/memory/`) — and both are loaded into every conversation automatically.

- A **rule** shapes behavior: something to always do, never do, or do when a condition applies.
- A **lesson** is a fact: something Anton discovered while working that's worth keeping.

## The files

### `memory/rules.md`

Rules are organized under three sections — `## Always`, `## Never`, and `## When`:

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

### `memory/lessons.md`

Lessons are a flat list of facts, most recent first when loaded:

```markdown
# Lessons
- CoinGecko free tier rate-limits at ~50 req/min <!-- topic:api-coingecko ts:2026-02-27 -->
- Bitcoin price data via /coins/bitcoin/market_chart/range <!-- topic:api-coingecko ts:2026-02-27 -->
- pandas read_csv needs encoding='utf-8-sig' for BOM files <!-- topic:pandas ts:2026-02-27 -->
```

### `memory/topics/*.md`

When a lesson carries a topic tag, it's also cross-filed into a per-subject file — for example `topics/api-coingecko.md`. Topic files hold deeper notes per subject and are loaded on demand when the subject comes up, rather than in every conversation.

## The small annotations

Each entry carries a small HTML comment with metadata. It's invisible when the markdown is rendered, and you can ignore it when editing by hand:

| Field | Meaning |
|---|---|
| `confidence` | How certain Anton is (`high`, `medium`, `low`). In co-pilot mode, only high-confidence entries save without asking you. |
| `source` | Where the entry came from — `user` (you asked for it), `consolidation` (extracted from a work session), or `llm` (Anton decided mid-conversation). |
| `ts` | Date the entry was saved. Used to order lessons newest-first. |
| `topic` | The topic slug a lesson is cross-filed under. |

## How entries appear

Most entries are written by Anton itself as it works — when it discovers something, learns from an error, or reviews a finished session (see [What Anton remembers](/teach/memory-overview) for the full picture). You can also just tell it directly:

```
you> remember: always use httpx instead of requests
```

Anton classifies that as an Always rule and saves it on the spot.

## Managing from chat

Everything is numbered for easy reference:

| Command | What it does |
|---|---|
| `/memory rules` | Show all rules, numbered, grouped by scope and kind |
| `/memory rules delete <n>` | Delete rule number `n` |
| `/memory rules edit <n>` | Edit rule number `n` in place |
| `/memory lessons` | Show all lessons, numbered |
| `/memory lessons delete <n>` | Delete lesson number `n` |
| `/memory lessons edit <n>` | Edit lesson number `n` |
| `/memory vacuum` | Deduplicate and compact memory — merges entries that say the same thing, drops superseded ones |
| `/memory reset global` | Wipe all global memory (asks for confirmation) |
| `/memory reset project` | Wipe all project memory (asks for confirmation) |

Or skip the commands entirely: the files are yours. Open `rules.md` or `lessons.md` in any editor and change whatever you like — Anton picks up the edits in the next conversation.

## They're always loaded

Rules and lessons from both scopes are placed into every conversation automatically — Anton doesn't need to search for them. This is why it pays to keep them clean: a wrong rule gets followed everywhere. If you spot one, `/memory rules delete <n>` is the fastest fix.
