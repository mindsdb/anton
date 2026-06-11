---
title: CLI commands
description: Every anton terminal command — synopsis, behavior, and examples.
---

# CLI commands

This page covers the `anton` command-line surface. For commands you type *inside* a chat, see [Slash commands](/reference/slash-commands).

:::note For contributors
This page mirrors `anton/cli.py`. If you change the CLI, update this page in the same pull request.
:::

## `anton` — start a chat

```bash
anton [--folder PATH] [--resume]
```

Running `anton` with no subcommand starts an interactive chat in the current directory. On first run it walks you through accepting the terms and choosing an LLM provider; after that it drops you straight into the prompt. It also checks for updates and initializes the workspace's `.anton/` folder if needed.

| Option | Short | What it does |
|---|---|---|
| `--folder PATH` | `-f` | Use `PATH` as the workspace instead of the current directory |
| `--resume` | `-r` | Resume a previous chat session instead of starting fresh |

```bash
anton -f ~/projects/sales-analysis -r
```

The `--folder` and `--resume` options belong to the top-level command and also apply when running subcommands (for example, to point a subcommand at another workspace).

## `anton setup`

Configure the LLM provider, model, and API key. Runs the same guided flow as first-run onboarding: pick Minds-Enterprise-Cloud, a self-hosted Minds server, or bring your own key (Anthropic, OpenAI, Google Gemini, or any OpenAI-compatible endpoint). Validates the key with a live probe before saving. See [Pick a provider](/start/pick-a-provider).

```bash
anton setup
```

## `anton setup-search`

Configure an external web-search provider — Exa.ai or Brave Search. Only needed when your LLM endpoint is a generic OpenAI-compatible one without native web search (Anthropic, OpenAI, and mdb.ai provide search themselves). The key is validated and saved to the global `~/.anton/.env`, so it carries across all workspaces. See [Search providers](/configure/search-providers).

```bash
anton setup-search
```

## `anton dashboard`

Show the Anton status dashboard in the terminal. See [Dashboard](/use/dashboard).

```bash
anton dashboard
```

## `anton sessions`

List recent sessions as a table: ID, task, status, and a short summary preview.

```bash
anton sessions
```

## `anton session <id>`

Show one session's details — its task, status, and full summary. Get the ID from `anton sessions`. Exits with an error if the ID isn't found.

```bash
anton session 20260227_143052
```

## `anton learnings`

List stored learnings from the legacy learning store as a topic + summary table. See [The learnings command](/teach/learnings-cli) for what this is and how old learnings are migrated into current memory.

```bash
anton learnings
```

## `anton version`

Print the installed Anton version.

```bash
anton version
```

## `anton connect [slug]`

Connect a database or API to the Local Vault. With no argument, starts the interactive connect flow (same as `/connect` in chat). Pass an existing connection slug to reconnect using stored credentials without re-entering them.

```bash
anton connect
anton connect postgres-mydb
```

## `anton list`

List all saved data-source connections in the Local Vault.

```bash
anton list
```

## `anton edit <name>`

Edit credentials for an existing Local Vault connection, by slug.

```bash
anton edit postgres-mydb
```

## `anton remove <name>`

Remove a saved connection from the Local Vault, by slug.

```bash
anton remove postgres-mydb
```

## `anton test <name>`

Test a saved Local Vault connection by running its test snippet.

```bash
anton test postgres-mydb
```

For more on connections, see [Connecting data sources](/connect/data-sources).
