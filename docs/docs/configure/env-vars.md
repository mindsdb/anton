---
title: Environment variables
description: The authoritative reference for ANTON_* configuration variables and the .env loading chain.
---

# Environment variables

Every Anton setting can be set as an environment variable with the `ANTON_`
prefix. Most users never touch these directly — the `anton setup` and
`anton setup-search` flows write them to the right `.env` file for you — but
they are all available for scripting and overrides.

## The .env loading chain

Anton reads configuration from three locations:

1. `.env` in the current directory
2. `.anton/.env` in the workspace (project-local config)
3. `~/.anton/.env` (global config — LLM keys, search keys)

Variables set in your actual shell environment always take precedence over
values from any file. Setup flows write LLM and search provider keys to the
global file so they carry across workspaces, and workspace-specific settings
to the local one.

## Providers and models

| Variable | Default | What it does |
| --- | --- | --- |
| `ANTON_PLANNING_PROVIDER` | `anthropic` | Provider for the planning model (`anthropic`, `openai`, `openai-compatible`) |
| `ANTON_PLANNING_MODEL` | `claude-sonnet-4-6` | Model used for planning and conversation |
| `ANTON_CODING_PROVIDER` | `anthropic` | Provider for the coding model |
| `ANTON_CODING_MODEL` | `claude-haiku-4-5-20251001` | Model used for code generation in the scratchpad |
| `ANTON_MAX_TOKENS` | `8192` | Max output tokens per LLM call |
| `ANTON_ANTHROPIC_API_KEY` | unset | Anthropic API key |
| `ANTON_OPENAI_API_KEY` | unset | OpenAI (or OpenAI-compatible endpoint) API key |
| `ANTON_OPENAI_BASE_URL` | unset | Base URL for an OpenAI-compatible endpoint |
| `ANTON_OPENAI_API_VERSION` | unset | Azure OpenAI `api-version` query parameter |
| `ANTON_MINDS_ENABLED` | `true` | Allow using a Minds server as LLM provider |
| `ANTON_MINDS_API_KEY` | unset | Minds API key |
| `ANTON_MINDS_URL` | Minds cloud URL | Minds server URL |
| `ANTON_MINDS_MIND_NAME` | unset | Mind name to use on the Minds server |
| `ANTON_MINDS_SSL_VERIFY` | `true` | Verify SSL certificates when talking to the Minds server |

## Web tools

| Variable | Default | What it does |
| --- | --- | --- |
| `ANTON_WEB_SEARCH_ENABLED` | `true` | Enable the `web_search` tool — see [Web search](/connect/web-search) |
| `ANTON_WEB_FETCH_ENABLED` | `true` | Enable the `web_fetch` tool — see [Web fetch](/connect/web-fetch) |
| `ANTON_EXTERNAL_SEARCH_PROVIDER` | unset | External search provider for generic endpoints: `exa` or `brave` |
| `ANTON_EXA_API_KEY` | unset | Exa.ai API key |
| `ANTON_BRAVE_API_KEY` | unset | Brave Search API key |

The search provider variables are normally written by `anton setup-search` —
see [Search providers](/configure/search-providers).

## Memory

| Variable | Default | What it does |
| --- | --- | --- |
| `ANTON_MEMORY_ENABLED` | `true` | Master switch for the memory system |
| `ANTON_MEMORY_MODE` | `autopilot` | How lessons are saved: `autopilot`, `copilot`, or `off` — see [Memory overview](/teach/memory-overview) |
| `ANTON_EPISODIC_MEMORY` | `true` | Keep an episode archive of past sessions — see [Episodes and recall](/teach/episodes-and-recall) |

## Behavior

| Variable | Default | What it does |
| --- | --- | --- |
| `ANTON_THEME` | `auto` | Terminal color theme |
| `ANTON_DISABLE_AUTOUPDATES` | `false` | Skip automatic update checks — see [Updating](/start/updating) |
| `ANTON_ANALYTICS_ENABLED` | `true` | Anonymous usage events — see [Analytics](/configure/analytics) |
| `ANTON_LANGFUSE_HEADERS` | unset | Set to `1` to attach Langfuse trace headers on any OpenAI-compatible endpoint — see [Trace headers](/configure/trace-headers) |
| `ANTON_PROACTIVE_DASHBOARDS` | `false` | When `true`, Anton builds HTML dashboards proactively; when `false`, CLI output only |
| `ANTON_BACKEND` | `local` | Scratchpad backend: `local` or `remote` (remote requires Minds URL and API key) |
| `ANTON_PUBLISH_URL` | `https://4nton.ai` | Publish service used when sharing artifacts |
| `ANTON_TERMS_CONSENT` | `false` | Records that you accepted the terms screen (set by the first-run flow) |
| `ANTON_FIRST_RUN_DONE` | `false` | Records that first-run onboarding completed (set by the setup flow) |

A few additional internal tuning variables exist (tool-round limits, cell
timeouts); they are intentionally undocumented here as they are not meant for
everyday configuration.
