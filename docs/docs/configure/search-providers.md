---
title: Search providers
description: Use Parallel Search by default or choose Exa.ai or Brave Search for generic OpenAI-compatible endpoints.
---

# Search providers

On generic OpenAI-compatible endpoints, Anton uses [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp) for `web_search` by default. It needs no Parallel account or API key. Search queries are sent to Parallel, and the free service is rate limited.

Anthropic, OpenAI, and MindsHub execute search through their own provider tools instead. See [Web search](/connect/web-search).

Run `anton setup-search` to change the generic endpoint provider:

```bash
anton setup-search
```

The choices are Parallel Search, Exa.ai, Brave Search, and Skip. Exa and Brave require their own API keys. Anton validates those keys before saving them in the global `~/.anton/.env`; switching providers leaves previous keys in place. Skip disables `web_search` until you choose a provider again. You can also disable search on every endpoint with `ANTON_WEB_SEARCH_ENABLED=false`.
