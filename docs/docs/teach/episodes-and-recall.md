---
title: Episodes and recall
description: Every session is archived to disk, and Anton can search its own history when you ask about past work.
---

# Episodes and recall

Anton keeps a complete archive of your work sessions, called episodes. Every session is logged to `.anton/episodes/` in your workspace as one timestamped file per session (named like `20260227_143052.jsonl` — date and time the session started).

## What's recorded

Each episode file is an append-only log of everything that happened, turn by turn:

- your messages
- Anton's replies
- every tool call it made and what came back
- output from code it ran

Long entries are truncated for size — tool inputs and outputs are capped so the archive stays compact even after heavy sessions.

## Recall — Anton searches its own history

Anton can search this archive itself. Ask things like:

```
you> What did we do last week with the sales data?
you> What was that API endpoint we found for bitcoin prices?
```

Anton uses its recall tool to search the episode files — case-insensitive matching, newest results first, with an optional days-back window when the question implies a time range. It decides when to search on its own; you don't have to issue a command.

## Inspecting from chat

| Command | What it does |
|---|---|
| `/memory episodes` | Show the current conversation as a table of turns (question and answer per turn) |
| `/memory episodes delete <n>` | Delete turn number `n` from the archive — the live conversation history is rebuilt without it |

## Turning it off, wiping it

Episodic memory is **on by default**. To turn it off:

- run `/setup`, choose Memory, and toggle episodic memory, or
- set `ANTON_EPISODIC_MEMORY=false` in your environment or workspace `.env`.

When off, sessions aren't logged and the recall tool isn't offered — existing files are left in place.

To wipe the archive entirely, run `/memory reset episodic` (asks for confirmation).

## Privacy

Episodes are written to your local disk inside the workspace's `.anton/episodes/` folder and stay there. They are not uploaded anywhere. If a session touched something sensitive, you can delete individual turns with `/memory episodes delete <n>`, remove the session's file by hand, or reset the whole archive.
