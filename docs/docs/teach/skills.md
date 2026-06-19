---
title: Skills
description: Save whole reusable procedures from work Anton just did, and let it recall and follow them in future sessions.
---

# Skills

Lessons hold *facts* — "CoinGecko rate-limits at 50 req/min." Skills hold *whole procedures* — "how to summarize a CSV end-to-end." When you've watched Anton do something well and you want it done the same way next time, you save it as a skill.

Skills live in a single library at `~/.anton/skills/`, one directory per skill, **shared across all your projects**. A skill saved while working in one workspace is available everywhere.

## Creating a skill

Do a task, then save it. Anton drafts the procedure from the work it just did — the recent conversation plus the code it ran:

```
you> Take a quick look at sales_q3.csv

anton> [opens scratchpad, loads pandas, infers schema, prints describe(), plots distributions]
       Here's what I found...

you> /skill save csv summary
anton> Drafting a skill from recent work…
       Saved skill csv_summary → ~/.anton/skills/csv_summary/
       Name: CSV Summary
       When to use: User asks to explore, summarize, or describe a CSV file.
```

The name hint after `/skill save` is optional — Anton will pick a sensible label either way. The skill is saved automatically (no interactive editing step) and is available immediately, including for the rest of the current session.

## Using a skill

You don't invoke skills manually. Every conversation includes a compact list of your saved skills — just each skill's label and a one-line "when to use" cue. When your request matches one, Anton recalls the full procedure from disk and follows it instead of reasoning from scratch.

Each recall bumps a usage counter, so `/skill list` shows you which skills actually earn their keep. Anton is also typo-tolerant here: if it asks for a label that doesn't quite exist, the closest match is used.

## Managing skills

| Command | What it does |
|---|---|
| `/skill save <name hint>` | Draft a new skill from recent work and save it (name hint optional) |
| `/skill list` | Show all saved skills with their "when to use" cues and recall counters |
| `/skill show <label>` | Print one skill's full procedure and stats — typo-tolerant, suggests the closest match |
| `/skill remove <label>` | Delete a skill from disk |

## What's inside a skill

Each skill is a small directory at `~/.anton/skills/` named by its label:

```
~/.anton/skills/csv_summary/
├── meta.json          name, description, when-to-use cue
├── declarative.md     the step-by-step procedure Anton follows
└── stats.json         usage counters
```

`declarative.md` is ordinary markdown — open it to read or hand-tune the procedure. The full directory format (including the optional later stages a skill can grow into) is covered in the developer guide at [Skills internals](/developer/skills-internals).

## When to save a skill

Good candidates are tasks that are **multi-step, repeatable, and were just done well**: a report you'll want monthly, a data-cleaning routine, a deployment checklist. Single facts belong in [lessons](/teach/lessons-and-rules) instead — Anton handles those on its own.
