---
title: Search providers
description: Set up Exa.ai or Brave Search with anton setup-search for generic OpenAI-compatible endpoints.
---

# Search providers

`anton setup-search` configures an external search provider so the
`web_search` tool works on LLM endpoints that don't ship native search.

## When you need it

Only when your LLM provider is a **generic OpenAI-compatible endpoint**
(Together, Groq, Ollama, vLLM, OpenRouter, and similar). Anthropic, OpenAI,
and Minds (mdb.ai) all execute web search natively on your existing LLM key —
no extra setup, and `setup-search` is unnecessary. See
[Web search](/connect/web-search) for the full provider matrix.

When `anton setup` finishes configuring a custom OpenAI-compatible endpoint,
it offers this step automatically. You can run it again at any time:

```bash
anton setup-search
```

## Choosing a provider

| Provider | Character | Get a key at |
| --- | --- | --- |
| Exa.ai | AI-native semantic search | dashboard.exa.ai/api-keys |
| Brave Search | privacy-focused web search | api.search.brave.com/app/keys |

The setup screen shows which provider (if any) is currently configured, with
a masked tail of the active key so you can recognize it without exposing it.

## Key validation

Anton validates the key before saving it: it makes a small probe call to the
provider's search API and checks the response. If authentication fails or the
service errors, you can:

- **retry** — re-enter the key for the same provider (fix a typo without
  re-picking from the menu),
- **switch** — go back to the picker and try the other provider,
- **skip** — disable `web_search` for now.

## Switching providers

Re-run `anton setup-search` and pick the other provider. The previously
stored key for the old provider is left in place; only the active provider
selection changes.

## Skipping

Choosing **Skip** disables `web_search` — the tool stays unavailable until
you run `anton setup-search` again. If a working provider is already
configured, Anton asks for confirmation before clearing it, so a stray
keystroke can't wipe a working setup.

## Where the key lives

The provider choice and key are persisted to the **global** `~/.anton/.env`
(as `ANTON_EXTERNAL_SEARCH_PROVIDER` plus `ANTON_EXA_API_KEY` or
`ANTON_BRAVE_API_KEY`), so they survive across sessions and workspaces —
the same scope as your LLM keys. See
[Environment variables](/configure/env-vars).
