---
title: Sessions
description: Resume previous conversations, inspect session history, and share sessions between machines or teammates.
---

# Sessions

Every conversation with Anton is a session. Sessions are stored in your
workspace, so you can resume them later, inspect what happened, and even
export one for a teammate to import.

## Resuming a session

**From inside chat** — type `/resume` to see a picker of your recent
sessions and continue one where it left off.

**At launch** — start Anton with the resume flag:

```bash
anton --resume    # or: anton -r
```

## Listing sessions from the terminal

```bash
anton sessions
```

Prints a table of recent sessions with their ID, task, status, and a short
summary preview. To see one session's full details and summary:

```bash
anton session <id>
```

## Sharing sessions with `/share`

`/share` packages a session into a portable `.anton` file you can move
between machines or hand to a teammate. The export includes the conversation
history, a memory snapshot (lessons born in the session and project memories
it used), the scratchpad code cells with their output, and an
LLM-generated title and summary.

| Command | What it does |
| --- | --- |
| `/share export` | Export the current session to `.anton/output/` as a timestamped `.anton` file |
| `/share export --summary` | Lighter export: metadata and summary only, no full conversation history (recommended for long sessions) |
| `/share import <file>` | Import a `.anton` file: creates a new session with the conversation, memories, and scratchpad cells restored, then resumes it |
| `/share status` | Show whether the current session was imported, and from whom — plus which data sources it references and whether they're connected on this machine |
| `/share history` | List all exported and imported `.anton` files in this workspace |

A typical handoff:

```
# On machine A
/share export
  → .anton/output/pipeline-latency-root-cause-analysis_20260610_142233.anton

# On machine B (after copying the file over)
/share import ~/Downloads/pipeline-latency-root-cause-analysis_20260610_142233.anton
```

After import, Anton prints a briefing — title, who exported it and when,
summary, memory counts, and code cells — and you continue where the session
left off. If the session used data sources you haven't connected yet,
`/share status` flags them so you can run `/connect` first
(see [Data sources](/connect/data-sources)).

If you already have an active session, importing creates a new session; your
current work is preserved in history.
