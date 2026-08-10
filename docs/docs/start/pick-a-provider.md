---
title: Pick a provider
description: Compare the LLM providers Anton supports and how to set each one up.
---

# Pick a provider

Anton works with any of these provider options. You choose one during
onboarding (the first time you run `anton`), and you can switch at any time
with `/llm` in chat or by running `anton setup`. All keys are persisted to
`~/.anton/.env`, so they carry across sessions and workspaces.

## At a glance

| Option | Default model | Web search / fetch | Best for |
| --- | --- | --- | --- |
| MindsHub — recommended | Resolved from the live catalogue (`sonnet` / `haiku` tier defaults) | Native passthrough, zero setup | Best overall experience |
| Anthropic (bring your own key) | `claude-sonnet-4-6` | Native server tools | Anthropic accounts |
| OpenAI (bring your own key) | `gpt-5.4` | Native via Responses API | OpenAI accounts |
| Google Gemini (bring your own key) | `gemini-3-flash-preview` | Needs `anton setup-search` | Gemini accounts |
| Custom OpenAI-compatible | You choose | Needs `anton setup-search` | Ollama, vLLM, Together, Groq, LM Studio, Azure, … |

## Option 1 — MindsHub (recommended)

[MindsHub](https://mindshub.ai) is the default and recommended choice. During
setup Anton resolves the planning and coding models from the server's live
model catalogue — picking the models your key is actually entitled to use —
and validates the connection with a real request against that configuration:

- Latest frontier models
- Faster responses
- Cost optimized
- Secure data connectors
- Native web search and fetch passthrough — no extra setup

During onboarding, if you don't have a MindsHub API key yet, Anton opens the
signup page for you — it takes a few seconds.

The MindsHub host itself is not prompted for. To point Anton at a non-default
host (staging, self-hosted), set `ANTON_MINDS_URL` in your environment or in
`~/.anton/.env` before running setup — setup honors it.

## Option 2 — Bring your own key

Choosing "Bring your own key" presents four sub-options. In each case Anton
validates the key with a quick probe call before saving it.

### Anthropic

Enter your Anthropic API key and a model (default: `claude-sonnet-4-6`). Web
search and web fetch run as Anthropic native server tools, billed on your
Anthropic key — no extra setup.

### OpenAI

Enter your OpenAI API key and a model (default: `gpt-5.4`). Web search runs
natively through the OpenAI Responses API; fetching is covered by the same
capability.

### Google Gemini

Anton talks to Gemini through Google's OpenAI-compatible endpoint. Get a key
at [aistudio.google.com/apikey](https://aistudio.google.com/apikey); the
default model is `gemini-3-flash-preview`. Because this is a generic
OpenAI-compatible endpoint, web search needs an external provider — run
`anton setup-search` to configure Exa.ai or Brave Search.

### Custom OpenAI-compatible endpoint

Works with Ollama, vLLM, Together, Groq, LM Studio, Azure, or any
OpenAI-compatible API. You provide:

- **Base URL** — e.g. `http://localhost:11434/v1` for Ollama
- **API key** — press Enter to skip if your endpoint doesn't need one
- **Model name**
- **API version** — leave blank for standard endpoints; required for Azure

After setup, Anton offers to configure a web search provider (Exa.ai or
Brave Search) since generic endpoints don't expose search natively. You can
run that step any time with `anton setup-search` — see
[Search providers](/configure/search-providers).

## Web search and fetch by provider

Anton exposes two web tools to the agent — `web_search` and `web_fetch` —
both on by default. How they execute depends on your provider:

| Provider | `web_search` | `web_fetch` | Setup |
| --- | --- | --- | --- |
| Anthropic BYOK | Anthropic native server tool | Anthropic native server tool | None — billed on your Anthropic key |
| OpenAI BYOK | OpenAI Responses API native | covered by `web_search` | None — billed on your OpenAI key |
| MindsHub | MindsHub passthrough | MindsHub passthrough | None — billed on your MindsHub key |
| Generic OpenAI-compatible (Together, Groq, Ollama, vLLM, …) | Exa.ai or Brave (you choose at setup) | stdlib HTTP GET (no key) | Run `anton setup-search` once |

To opt out, set `ANTON_WEB_SEARCH_ENABLED=false` and/or
`ANTON_WEB_FETCH_ENABLED=false` in your config. See
[Web search](/connect/web-search) and [Web fetch](/connect/web-fetch).

## Switching providers later

- In chat: type `/llm` to change provider, model, or API key.
- From the terminal: run `anton setup` to re-run the full provider selection.

Both flows validate the new configuration before saving it to `~/.anton/.env`.
