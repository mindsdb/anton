---
title: Trace headers
description: Attach Langfuse trace headers to LLM requests on OpenAI-compatible endpoints.
---

# Trace headers

When the planning provider is OpenAI-compatible, Anton can attach trace
headers to its LLM requests so a router or observability proxy can attribute
traces to sessions:

- `Langfuse-Session-Id`
- `Langfuse-Tags`
- `Langfuse-Metadata`

This is useful when your requests pass through a gateway that records traces
— for example a self-hosted Langfuse proxy sitting in front of Ollama or
vLLM.

## Enabling

To emit the headers against any OpenAI-compatible endpoint, set:

```bash
export ANTON_LANGFUSE_HEADERS=1
```

Or add it to your workspace config (`.anton/.env`):

```text
ANTON_LANGFUSE_HEADERS=1
```

The headers only carry session attribution metadata — they don't change what
prompts are sent or where. See
[Environment variables](/configure/env-vars) for the config file loading
order.
