---
title: Analytics
description: What anonymous usage events Anton sends, what they contain, and how to opt out.
---

# Analytics

Anton collects anonymous usage events — for example "session started" or
"first query" — to help the MindsDB team understand how the product is used.

## What is sent

Each event is a single HTTP GET request carrying only:

- the action name (e.g. `anton_started`),
- a timestamp,
- an anonymous installation ID.

**No personal data or query content is ever sent** — no prompts, no file
contents, no hostnames. The installation ID is a one-way SHA-256 hash of the
machine's network adapter address, truncated to 16 hex characters; the raw
address never leaves your device. Events are fire-and-forget: they never
block Anton and failures are silently ignored.

## Opting out

Set the environment variable:

```bash
export ANTON_ANALYTICS_ENABLED=false
```

Or add it to your workspace config (`.anton/.env`):

```text
ANTON_ANALYTICS_ENABLED=false
```

To turn it off everywhere, put the same line in the global `~/.anton/.env`.
See [Environment variables](/configure/env-vars) for how the config files are
loaded, and [Security model](/configure/security) for the full picture of
what leaves your machine.
