---
title: Workspace files
description: Every file and directory Anton keeps under the project .anton/ folder and the global ~/.anton/ folder.
---

# Workspace files

Anton keeps its state in two places: a `.anton/` folder inside each workspace (project scope) and `~/.anton/` in your home directory (global scope). Almost everything is plain text — and most of it is safe to open and edit.

## Project scope — `.anton/` in each workspace

| Path | What it is | Safe to edit by hand? |
|---|---|---|
| `.anton/.env` | Workspace configuration variables and local secrets (memory mode, toggles, keys scoped to this workspace) | Yes — standard `KEY=value` lines |
| `.anton/anton.md` | Your project context — instructions Anton reads at conversation start. Anton never writes to it. See [Project context](/teach/project-context) | Yes — it's yours by design |
| `.anton/memory/rules.md` | Project-specific behavioral rules under `## Always`, `## Never`, `## When` sections | Yes — keep the section headings and one bullet per entry |
| `.anton/memory/lessons.md` | Project-specific facts Anton learned while working here | Yes — one bullet per entry |
| `.anton/memory/topics/*.md` | Deeper project notes organized per subject, loaded on demand | Yes |
| `.anton/episodes/*.jsonl` | Session archive — one file per session, named `YYYYMMDD_HHMMSS.jsonl`. See [Episodes and recall](/teach/episodes-and-recall) | Better not — delete whole files if needed, or use `/memory episodes delete <n>` |
| `.anton/artifacts/` | User-facing outputs (HTML apps, docs, datasets). One subfolder per artifact, each carrying a `metadata.json` and a `README.md` rendered from it | The outputs are yours to use; avoid hand-editing `metadata.json` |
| `.anton/scratchpad-venvs/` | Isolated Python environments for Anton's code execution, one per scratchpad | No — managed automatically; safe to delete to reclaim space (rebuilt on demand) |
| `.anton/context/`, `.anton/learnings/` | Legacy formats from older versions, migrated automatically into `memory/` on first run after upgrading (originals preserved). See [The learnings command](/teach/learnings-cli) | Leave as-is |

## Global scope — `~/.anton/`

| Path | What it is | Safe to edit by hand? |
|---|---|---|
| `~/.anton/.env` | Global configuration — LLM provider and API keys, search-provider key, terms consent. Shared across all workspaces; workspace `.env` values win on conflict | Yes — standard `KEY=value` lines |
| `~/.anton/memory/profile.md` | Your identity profile — name, timezone, preferences. Global only; there is no project profile | Yes |
| `~/.anton/memory/rules.md` | Global rules that apply in every workspace | Yes |
| `~/.anton/memory/lessons.md` | Global lessons — knowledge useful across any project | Yes |
| `~/.anton/memory/topics/*.md` | Global per-subject notes | Yes |
| `~/.anton/skills/<label>/` | The skill library — one directory per saved skill, shared across all projects. See [Skills](/teach/skills) | Procedure yes, metadata carefully (below) |
| `~/.anton/skills/<label>/meta.json` | Skill metadata — label, name, when-to-use cue | Carefully — keep it valid JSON |
| `~/.anton/skills/<label>/declarative.md` | The step-by-step procedure Anton follows | Yes — hand-tuning the procedure is encouraged |
| `~/.anton/skills/<label>/chunks.md` | Optional higher-level recipes (present in later-stage skills) | Yes, if present |
| `~/.anton/skills/<label>/code/` | Optional runnable helper modules (present in later-stage skills) | Yes, if present |
| `~/.anton/skills/<label>/requirements.txt` | Optional dependencies for a skill's helper code | Yes, if present |
| `~/.anton/skills/<label>/stats.json` | Per-skill usage counters | No — let Anton track usage |
| `~/.anton/datasources.md` | Your custom data-source engine definitions — Anton appends here when it builds a custom integration, and you can add your own. See [Custom integrations](/connect/custom-integrations) | Yes |
| `~/.anton/scratchpad-venvs/` | Fallback location for scratchpad environments when no workspace path applies | No — managed automatically; safe to delete |

## Rules of thumb

- **Markdown files are yours.** Memory, topics, profile, skills' procedures, `anton.md`, `datasources.md` — open and edit freely. Anton picks up changes in the next conversation.
- **JSON and JSONL files are Anton's bookkeeping.** Prefer the chat commands (`/memory`, `/skill`) over hand-editing them.
- **Environments are disposable.** Anything under `scratchpad-venvs/` can be deleted; Anton rebuilds what it needs.
