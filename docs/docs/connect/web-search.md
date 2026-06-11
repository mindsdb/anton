---
title: Web search
description: How Anton searches the live web — provider-native when possible, Exa.ai or Brave for generic endpoints.
---

# Web search

Anton can query the live web through the `web_search` tool, which is on by
default. How it executes depends on your LLM provider:

| Provider | `web_search` | Setup |
| --- | --- | --- |
| Anthropic BYOK | Anthropic native server tool | None — billed on your Anthropic key |
| OpenAI BYOK | OpenAI Responses API native | None — billed on your OpenAI key |
| Minds-Enterprise-Cloud (mdb.ai) | mdb.ai passthrough | None — billed on your Minds key |
| Generic OpenAI-compatible (Together, Groq, Ollama, vLLM, …) | Exa.ai or Brave (you choose) | Run `anton setup-search` once |

For the first three rows there is nothing to configure — the LLM provider
executes the search server-side and folds the results directly into its
response, billed on the key you already set up.

## Generic OpenAI-compatible endpoints

Generic endpoints don't ship a native search capability, so Anton uses an
external search provider — Exa.ai or Brave Search. After `anton setup`
finishes configuring a custom OpenAI-compatible endpoint, Anton offers to set
one up on the spot. You can also run (or re-run) that step at any time:

```bash
anton setup-search
```

You pick a provider, paste an API key, and Anton validates the key with a
probe call before saving it. The key is persisted to `~/.anton/.env`, so it
carries across sessions and workspaces — exactly like your LLM key. See
[Search providers](/configure/search-providers) for the full setup walkthrough
and where to get keys.

If you skip the step, `web_search` is unavailable on that endpoint until you
run `anton setup-search`.

## Opting out

To disable web search entirely, set:

```bash
export ANTON_WEB_SEARCH_ENABLED=false
```

or add the line to your workspace config (`.anton/.env`).

## Caveats

- Provider rate limits apply, on both LLM-native and external providers.
- For retrieving the contents of a specific URL, Anton uses a separate tool —
  see [Web fetch](/connect/web-fetch).
