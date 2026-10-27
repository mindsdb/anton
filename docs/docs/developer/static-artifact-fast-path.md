---
title: Static artifact fast path
description: Opt-in direct generation for self-contained HTML pages, its three settings, and what each one changes.
---

# Static artifact fast path

An opt-in path for self-contained HTML pages built from data the agent can fetch now.
It skips the brief, PRD and spec steps, binds the exact source JSON into the page, and
keeps the normal static and browser checks. Everything is off unless a setting below is
set; with all three unset, generation is the same as without this code.

## Settings

| Variable | Value | Effect |
|---|---|---|
| `ANTON_STATIC_FAST_PATH` | `direct` | In act-first mode, with no attachments other than data files, discovery saves the complete data to `observed-data.json` in the artifact folder and a single writer call builds the page. Read at call time. |
| `ANTON_STATIC_SINGLE_CALL` | `on` | `generate_artifact` takes `name` for a new page and registers it in the same call, and the chat prompt uses the shorter single-call artifact instructions. Read when `anton.core.tools.tool_defs` is imported, so set it before the process starts. |
| `ANTON_STATIC_COMPONENT_WRITER` | `on` | With bound data, the writer returns a small program for the built-in presentation runtime (`presentation.js`: charts, tables, maps, globes, timelines and text, each chart with a Data table) instead of a whole HTML document. It answers `USE_HTML_WRITER` when the runtime can't express the request. |

`ANTON_STATIC_FAST_PATH` is the core; the other two are independent of it.

## How a page is built

1. Discovery runs with the usual source tools. Earlier tool results from the conversation
   are passed to it as data. A scratchpad cell can finish collection itself with
   `complete_static_artifact`, which the host accepts only after the cell succeeds and
   writes a fresh, credential-free `observed-data.json`.
2. If discovery settles on `html-app`, the writer builds the page. The host embeds the
   handoff verbatim at `window.ARTIFACT_DATA`; the model writes only presentation code.
   An edit without a new handoff keeps the page's previous data.
3. The page goes through `verify_frontend` and the browser check with the session's sandbox
   policy, screenshot setting and the previous page for comparison. A page that still fails
   the browser check after one repair ends as `BrowserCheckFailed`, like the normal pipeline.

Anything else (other types, other attachments, ask-first mode, or an existing page or
program over 24,000 characters) uses the normal pipeline, and so does a page the writer
cannot finish in one reply. The handoff and the saved program are internal files: they are
never listed or published.
