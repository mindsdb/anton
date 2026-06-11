---
title: What Anton remembers
description: How Anton's memory works — plain markdown files you can open, edit, and control, split into global and project scopes.
---

# What Anton remembers

Anton is an open-source AI coworker that executes tasks, connects to your tools and data, remembers lessons, and improves its workflows over time. The "remembers" part is what this section is about — and it works differently from most AI tools.

**Anton's memory is plain markdown files on your disk.** There is no opaque database, no cloud sync, no embedding store you can't inspect. Open the files in any editor, read what Anton knows, fix anything it got wrong, or delete what you don't want it to keep. Memory you can read is memory you can trust.

## Two scopes

Anton keeps memory at two levels:

| Scope | Location | What lives there |
|---|---|---|
| **Global** | `~/.anton/memory/` | Knowledge about *you* and the world — your identity and preferences, universal rules and lessons that apply everywhere. The skill library lives alongside it at `~/.anton/skills/`. |
| **Project** | `.anton/memory/` in each workspace | Knowledge about *this* workspace — project-specific rules, lessons learned while working here, and session logs. |

When you chat with Anton, both scopes are loaded automatically. A lesson learned in one project stays in that project; a lesson about an API rate limit that applies everywhere gets saved globally.

## What gets remembered

| Kind | Stored in | Example |
|---|---|---|
| **Identity / profile** | `~/.anton/memory/profile.md` | Your name, timezone, preferred tools, communication style. Global only — identity is singular. |
| **Rules** | `memory/rules.md` | Things to always do, never do, or do when a condition applies — "Always use httpx instead of requests." |
| **Lessons** | `memory/lessons.md` | Facts discovered while working — "CoinGecko free tier rate-limits at ~50 req/min." |
| **Topics** | `memory/topics/*.md` | Deeper notes organized per subject, loaded on demand when relevant. |
| **Episodes** | `.anton/episodes/*.jsonl` | A full timestamped log of every session — what you asked, what Anton did, what came back. |
| **Skills** | `~/.anton/skills/` | Reusable step-by-step procedures — "how to summarize a CSV end-to-end." Shared across all projects. |

## How memories get written

You never have to manage memory by hand (though you always can). Anton writes its own memory three ways:

1. **While it works.** Anton saves facts, rules, and profile details it judges worth keeping as the conversation happens — or immediately when you say something like "remember: always use httpx."
2. **From its mistakes.** When code Anton runs hits errors, a background pass reviews what went wrong after your answer is delivered and writes a generalizable lesson so the same trap is avoided next time.
3. **By reviewing its own sessions.** After a substantial work session ends, Anton replays it in the background and extracts durable rules and lessons from what happened.

### Memory modes

A memory mode controls how freely Anton saves:

| Mode | Behavior |
|---|---|
| **Autopilot** (default) | Anton decides what to remember — no confirmations. |
| **Co-pilot** | Saves the obvious, asks you to confirm the ambiguous. Confirmations only ever appear *after* you've received your full answer, never in the middle of work. |
| **Off** | Never saves anything new. Still reads existing memory. |

Change the mode via `/setup` and choose Memory, or set the `ANTON_MEMORY_MODE` environment variable (`autopilot`, `copilot`, or `off`).

## Inspecting memory anytime

Run `/memory` in chat for a status dashboard — entry counts per scope, your identity summary, topics, and episodic memory status. Sub-commands let you list, edit, and delete individual entries; see [Lessons and rules](/teach/lessons-and-rules). And since everything is markdown, you can always just open the files.

## The rest of this section

- [Lessons and rules](/teach/lessons-and-rules) — the rule and lesson files, how entries look, and how to manage them from chat.
- [Skills](/teach/skills) — saving whole procedures Anton can reuse across projects.
- [Episodes and recall](/teach/episodes-and-recall) — the session archive and how Anton searches its own history.
- [Project context](/teach/project-context) — `anton.md`, the one file that's yours alone.
- [The learnings command](/teach/learnings-cli) — the legacy learning store and its automatic migration.
