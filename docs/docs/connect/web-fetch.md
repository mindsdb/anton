---
title: Web fetch
description: How Anton retrieves the contents of a URL, what to expect from different page types, and why fetched content is untrusted input.
---

# Web fetch

Anton can retrieve the contents of a URL through the `web_fetch` tool, which
is on by default. Like [web search](/connect/web-search), execution depends on
your LLM provider:

| Provider | `web_fetch` |
| --- | --- |
| Anthropic BYOK | Anthropic native server tool |
| OpenAI BYOK | covered by `web_search` |
| Minds-Enterprise-Cloud (mdb.ai) | mdb.ai passthrough |
| Generic OpenAI-compatible | built-in HTTP GET fallback (no key needed) |

The fallback needs no API key at all: Anton performs a plain HTTP GET,
follows redirects, and strips HTML down to readable plain text before handing
it to the model.

## What to expect

- **30-second timeout.** Slow pages fail rather than hang the conversation.
- **HTML is stripped to plain text.** Works best on article-style pages —
  blog posts, documentation, news.
- **Paywalls and JS-heavy single-page apps return little.** The fallback does
  not execute JavaScript, so pages that render client-side may come back
  nearly empty.

:::warning Treat fetched page content as untrusted input

Anything a web page says ends up in the model's context. A malicious page can
embed instructions intended for the model (prompt injection). Anton's
credential vault keeps secrets out of prompts, but you should still be
deliberate about which URLs you ask Anton to fetch, and review any actions it
proposes after reading external content. See
[Security model](/configure/security).

:::

## Opting out

To disable web fetch, set:

```bash
export ANTON_WEB_FETCH_ENABLED=false
```

or add the line to your workspace config (`.anton/.env`).
