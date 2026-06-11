---
title: Project context (anton.md)
description: The one file that's entirely yours — written instructions Anton reads at the start of every conversation in a workspace.
---

# Project context (anton.md)

Every workspace can have a `.anton/anton.md` file: plain-markdown instructions that Anton reads at the start of each conversation there.

Unlike the memory files, **`anton.md` is yours alone — Anton never writes to it.** Memory is what Anton learns; `anton.md` is what you declare. It will never be edited, compacted, or appended to by the memory system.

## What to put in it

- **What this project is** — one or two sentences of context so Anton doesn't have to infer it.
- **Conventions** — naming, formatting, style choices specific to this workspace.
- **Where the data lives** — file paths, database names, which connection is the source of truth.
- **Preferred outputs** — report format, chart style, where results should be written.
- **Always/never instructions for this workspace** — though if a rule should survive and evolve with Anton's own learning, it often fits better as a proper rule via `/memory` or a "remember:" request; see [Lessons and rules](/teach/lessons-and-rules). Use `anton.md` for instructions you want to control word-for-word.

## Your instructions win

`anton.md` is loaded *after* Anton's memory, so if anything in it conflicts with a remembered rule or lesson, your written instructions take priority.

## Example

A realistic `anton.md` for a data-analysis workspace:

```markdown
# Project: Q3 Retail Analytics

This workspace analyzes quarterly sales for the retail division.
Source data arrives as CSV exports in ./data/raw/ (one file per region,
UTF-8 with BOM). The warehouse connection "postgres-retail" is the
source of truth for historical numbers — prefer it over the CSVs when
both have the data.

## Conventions
- All money values are EUR. Never display USD.
- Fiscal quarters: Q3 = July 1 to September 30.
- Round percentages to one decimal place.

## Outputs
- Reports go to ./reports/ as a single self-contained HTML file.
- Charts: bar charts for regional comparisons, line charts for trends.
- Every report starts with a 3-bullet executive summary.

## Always / Never
- Always exclude the "internal-test" region from aggregates.
- Never modify anything under ./data/raw/ — treat it as read-only.
```

## Tips

- Keep it short and current. It's loaded into every conversation in the workspace, so a focused page beats an encyclopedia.
- If you find yourself repeating an instruction in chat, that's the signal to move it into `anton.md`.
- For knowledge Anton should *discover and refine on its own*, leave room for [memory](/teach/memory-overview) to do its job — `anton.md` is for the ground rules.
