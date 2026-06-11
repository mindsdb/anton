---
title: The learnings command
description: anton learnings lists the legacy learning store — older knowledge that is automatically migrated into the current memory files.
---

# The learnings command

```bash
anton learnings
```

Lists stored learnings as a table of topic and summary. If nothing has been recorded, it prints "No learnings recorded yet."

## This is the legacy store

`anton learnings` surfaces an **older format** of Anton's knowledge — markdown files under `.anton/learnings/` with a small index, from before the current memory system existed.

On the first run after upgrading, Anton automatically migrates old-format files into the current memory layout, one time only:

- `.anton/learnings/*.md` and `.anton/context/*.md` are split into individual facts and re-encoded into `memory/lessons.md` and `memory/topics/`.
- The old files are **preserved** — nothing is deleted, so `anton learnings` keeps working on whatever was there.
- The migration only runs when legacy files exist and the new memory files don't, so it never overwrites current memory.

## Where to look instead

For current knowledge, use the memory surface in chat:

- `/memory lessons` — list, edit, and delete what Anton knows now ([Lessons and rules](/teach/lessons-and-rules))
- `/memory` — the full status dashboard ([What Anton remembers](/teach/memory-overview))
