---
title: Workspaces
description: How Anton organizes its state — the .anton/ folder, project vs global scope, and overriding the workspace.
---

# Workspaces

A workspace is the directory Anton works in. Everything Anton knows about a
project — its memory, episodes, configuration, and local secrets — lives in a
`.anton/` folder inside that directory.

## The `.anton/` layout

When you run `anton` in a directory:

| Path | What it holds |
| --- | --- |
| `.anton/` | Workspace folder: scratchpad state, episodic memory, local secrets |
| `.anton/anton.md` | Optional project context — Anton reads this at conversation start |
| `.anton/.env` | Workspace configuration variables (local file) |
| `.anton/episodes/` | Episodic memories, one file per session |
| `.anton/memory/rules.md` | Behavioral rules: always/never/when rules (e.g. never hardcode credentials) |
| `.anton/memory/lessons.md` | Factual knowledge: things Anton has learned (API quirks, patterns that worked) |
| `.anton/memory/topics/` | Topic-specific lessons — deeper notes organized by subject |
| `.anton/artifacts/` | Outputs Anton produces — one folder per artifact |

For a full file-by-file reference, see
[Workspace files](/reference/workspace-files).

## Project vs global scope

Anton keeps two layers of state:

- **Global — `~/.anton/`** — things that are *yours*, not the project's: your identity and profile, your API keys (`~/.anton/.env`), the skills library (`~/.anton/skills/`), and global memory. These follow you across every workspace.
- **Project — `<workspace>/.anton/`** — things that belong to *this project*: project memory (rules, lessons, topics), episodes, project context (`anton.md`), and project-level configuration.

When settings or env vars exist in both, the project `.anton/.env` wins and
the global `~/.anton/.env` fills in anything missing (API keys, etc.).

How memory layers combine is covered in
[Memory overview](/teach/memory-overview).

## How Anton picks the workspace at boot

1. If the current directory already has a `.anton/`, Anton uses it.
2. Otherwise it creates one (and tells you: `workspace is /path/.anton`).

So the simplest mental model is: `cd` to the project you want Anton to work
on, then run `anton`.

## Overriding the workspace

To run Anton against a different directory without changing into it:

```bash
anton --folder /path/to/workspace    # or: anton -f /path/to/workspace
```
