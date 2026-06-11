---
title: Security model
description: How the credential vault, scratchpad isolation, and local-first design keep your secrets and data under your control.
---

# Security model

Anton runs on your machine. Your code, files, and queries stay local unless
you explicitly ask Anton to send them somewhere. This page explains the
mechanisms behind that.

## The credential vault

Connection credentials are stored in a local vault at `~/.anton/data_vault/`
— one JSON file per connection, with the directory restricted to your user
account and each file readable only by you.

The key property: **secrets are never placed in LLM prompts.**

- At run time, credentials are injected as `DS_*` environment variables into
  the scratchpad process (for example `DS_PASSWORD`, or namespaced forms like
  `DS_POSTGRES_MYDB__PASSWORD` when several connections are loaded).
- The model writes code that reads `os.environ['DS_PASSWORD']` — it sees the
  variable *name*, never the value.
- Scratchpad output is scrubbed before it reaches the model: values of
  registered secret variables are redacted, so a stray `print` can't leak a
  password into the conversation.

See [Connecting things: overview](/connect/overview) for how credentials get
into the vault in the first place.

## Scratchpad isolation

All code Anton writes runs in scratchpads — not in your shell. With the local
backend, each scratchpad is a persistent subprocess running inside its own
dedicated Python virtual environment (created per scratchpad under the
workspace's `.anton/scratchpad-venvs/`, or `~/.anton/scratchpad-venvs/` when
no workspace applies). Packages Anton installs for one task don't pollute
your system Python or other scratchpads, and a scratchpad can be reset to a
clean state at any time.

## What leaves your machine

Three things, and only three:

1. **Prompts to your LLM provider.** Whatever you type, plus the context
   Anton assembles, goes to the provider you configured (Anthropic, OpenAI,
   Minds, or your own endpoint) and is governed by that provider's terms.
2. **Anonymous analytics.** Event names and timestamps only — no query
   content, no personal data. Opt out any time: see
   [Analytics](/configure/analytics).
3. **Whatever you ask Anton to send.** Emails, API calls, published
   dashboards — actions you request.

In the other direction, remember that **fetched web content is untrusted
input**: a page Anton reads can contain text designed to manipulate the
model. See [Web fetch](/connect/web-fetch).

## Windows firewall

On Windows, the scratchpad's Python needs outbound network access. The
installer offers to add a firewall rule; if you skipped it, run this in an
elevated PowerShell:

```powershell
netsh advfirewall firewall add rule name="Anton Scratchpad" dir=out action=allow program="$env:USERPROFILE\.anton\scratchpad-venv\Scripts\python.exe"
```

## Your responsibility

Anton acts on your behalf — sending emails, modifying data, calling APIs. As
the first-run terms screen puts it: you're responsible for what Anton does on
your behalf, so review proposed actions before authorizing them. This matters
most for [custom integrations](/connect/custom-integrations), where Anton
writes its own integration code.
