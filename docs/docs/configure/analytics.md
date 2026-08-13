---
title: Analytics
description: What anonymous usage events Anton sends, what they contain, and how to opt out.
---

# Analytics

Anton collects anonymous usage events — for example "session started" or
"first query" — to help the MindsDB team understand how the product is used.

## What is sent

Every event carries:

- the action name (e.g. `anton_started`),
- a timestamp,
- an anonymous installation ID.

Some events also carry **anonymous measurements of that action** — for example
a datasource event names the engine (`postgres`), and the per-turn accounting
event carries token counts, model names, durations and an opaque conversation
id. These are numbers, names and identifiers only.

**No personal data or query content is ever sent** — no prompts, no message
text, no tool output, no file contents, no file paths, no credentials, no
hostnames, no email addresses. The installation ID is a one-way SHA-256 hash of
the machine's network adapter address, truncated to 16 hex characters; the raw
address never leaves your device. Events are fire-and-forget: they never block
Anton and a failure to send is never surfaced to you. Failures on the per-turn
accounting events are recorded in Anton's own debug log; failures on the other
events are discarded without a trace.

Two transports are in use, depending on the event: most events are a single
HTTP GET to the collector at `ANTON_ANALYTICS_URL`, while the per-turn
accounting events are an HTTPS POST whose body carries the measurements —
a body rather than a query string so the values do not end up in intermediate
access logs.

## Opting out

Set the environment variable:

```bash
export ANTON_ANALYTICS_ENABLED=false
```

Or add it to your workspace config (`.anton/.env`):

```text
ANTON_ANALYTICS_ENABLED=false
```

That switch covers every event on every transport.

### Turning off one transport only

`ANTON_ANALYTICS_URL` re-points the collector, and governs **only** the events
that use it — the per-turn accounting events are unaffected by it. To disable
those specifically, set an empty key:

```text
ANTON_POSTHOG_KEY=
```

To turn everything off everywhere, put `ANTON_ANALYTICS_ENABLED=false` in the
global `~/.anton/.env`.
See [Environment variables](/configure/env-vars) for how the config files are
loaded, and [Security model](/configure/security) for the full picture of
what leaves your machine.
