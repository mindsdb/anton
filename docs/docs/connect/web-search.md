---
title: Web search
description: How Anton searches the live web through model providers or its built-in Parallel Search fallback.
---

# Web search

Anton's `web_search` tool is on by default. The route depends on the LLM endpoint:

| Endpoint | Search route | Setup |
| --- | --- | --- |
| Anthropic BYOK | Anthropic native server tool | None beyond the Anthropic key |
| OpenAI BYOK | OpenAI Responses API native tool | None beyond the OpenAI key |
| MindsHub | MindsHub passthrough | None beyond the MindsHub key |
| Generic OpenAI-compatible (Together, Groq, Ollama, vLLM, …) | [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp) by default | No Parallel account or key |

On a generic endpoint, Anton sends the search query to Parallel's free, rate-limited MCP service and returns titles, URLs, and excerpts through its usual `web_search` tool. Run `anton setup-search` to select Exa.ai or Brave Search with your own key, or to disable search. See [Search providers](/configure/search-providers).

`web_fetch` remains a separate tool for retrieving a specific URL. To disable web search on any endpoint, set `ANTON_WEB_SEARCH_ENABLED=false`.
