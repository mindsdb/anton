---
title: Slash commands
description: Every in-chat slash command, grouped the same way as /help.
---

# Slash commands

Slash commands are typed at the chat prompt. Run `/help` anytime to see this list in the terminal; tab completion suggests commands and sub-commands as you type. For terminal commands run *outside* chat, see [CLI commands](/reference/cli-commands).

## LLM Provider

| Command | What it does |
|---|---|
| `/llm` | Change LLM provider or API key |
| `/minds` | Connect to a Minds server |

See [Pick a provider](/start/pick-a-provider).

## Data Connections

| Command | What it does |
|---|---|
| `/connect` | Connect a database or API to your Local Vault |
| `/list` | List all saved connections |
| `/edit` | Edit credentials for an existing connection |
| `/remove` | Remove a saved connection |
| `/test` | Test a saved connection |

See [Connecting data sources](/connect/data-sources).

## Workspace

| Command | What it does |
|---|---|
| `/setup` | Configure models and memory settings |
| `/memory` | View memory status and usage (dashboard; sub-commands below) |
| `/theme` | Switch theme — `/theme light` or `/theme dark`; bare `/theme` toggles |

### `/memory` sub-commands

| Command | What it does |
|---|---|
| `/memory` | Status dashboard |
| `/memory rules` | Show behavioral rules |
| `/memory rules delete <n>` | Delete rule number `n` |
| `/memory rules edit <n>` | Edit rule number `n` |
| `/memory lessons` | Show learned lessons |
| `/memory lessons delete <n>` | Delete lesson number `n` |
| `/memory lessons edit <n>` | Edit lesson number `n` |
| `/memory identity` | Show the identity profile |
| `/memory identity delete <n>` | Delete identity entry number `n` |
| `/memory identity edit <n>` | Edit identity entry number `n` |
| `/memory episodes` | Show episodic sessions as a table of turns |
| `/memory episodes delete <n>` | Delete turn number `n` |
| `/memory vacuum` | Deduplicate and compact memory |
| `/memory reset [global\|project\|episodic]` | Wipe a memory scope (asks for confirmation) |
| `/memory help` | Show all memory commands |

See [What Anton remembers](/teach/memory-overview) and [Lessons and rules](/teach/lessons-and-rules).

## Skills

| Command | What it does |
|---|---|
| `/skill save <name>` | Draft a skill from recent work and save it (name hint optional) |
| `/skill list` | Show all saved skills with usage counters |
| `/skill show <label>` | Print one skill's procedure and stats |
| `/skill remove <label>` | Delete a skill from disk |

See [Skills](/teach/skills).

## Chat Tools

| Command | What it does |
|---|---|
| `/goal` | Run a goal autonomously until complete — `/goal "objective" [--turns N]` |
| `/paste` | Attach an image from your clipboard |
| `/resume` | Continue a previous session |
| `/share` | Share sessions — `/share export`, `/share export --summary`, `/share import`, `/share status`, `/share history` |
| `/remote` | Set up or manage the remote scratchpad |
| `/publish` | Publish an HTML report to the web |
| `/unpublish` | Remove a published report |
| `/explain` | Show explainability details for the latest answer — summary, data sources used, and generated SQL |

See [Chat basics](/use/chat-basics) and [Sessions](/use/sessions).

## General

| Command | What it does |
|---|---|
| `/help` | Show the help menu |
| `exit` | Exit the chat |
