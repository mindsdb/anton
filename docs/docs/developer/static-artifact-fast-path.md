---
title: Static artifact fast path
description: Direct generation for self-contained HTML pages, and what it changes.
---

# Static artifact fast path

The default path for self-contained HTML pages built from data the agent can fetch now.
It skips the brief, PRD and spec steps, binds the exact source JSON into the page, and
keeps the normal static and browser checks. It has three parts:

- **Direct generation.** In act-first mode, with no attachments other than data files,
  discovery saves the complete data to `observed-data.json` in the artifact folder and a
  single writer call builds the page.
- **Single call.** `generate_artifact` takes `name` for a new page and registers it in the
  same call, and the chat prompt uses the shorter single-call artifact instructions (a
  static-only variant where the deployment offers no live apps). With
  direct reports off (`ANTON_DIRECT_REPORTS=false`) the page tools keep `slug` and the
  register-first instructions.
- **Component writer.** With bound data, the writer returns a small program for the
  built-in presentation runtime (`presentation.js`: charts, tables, maps, globes, timelines
  and text, each chart with a Data table) instead of a whole HTML document. It answers
  `USE_HTML_WRITER` when the runtime can't express the request, and the ordinary HTML
  writer builds the page.

## How a page is built

1. Discovery runs with the usual source tools. Earlier tool results from the conversation
   are passed in the kickoff user message as data, outside the system prompt. A scratchpad cell can finish collection itself with
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

## Current limits

The host checks that the handoff is fresh, parseable JSON without credential-shaped keys,
credential-bearing URLs or values that the output scrubber would redact. It deletes a rejected handoff.
It does not independently reconcile the model's aggregates with source rows or validate
their population, grain and reference. A browser check can confirm that the page renders,
but it cannot establish that a displayed figure is correct.

The generated HTML embeds the complete handoff so that controls can work without a
backend. Publishing or sharing that HTML therefore shares its source rows, including
fields the UI does not display. Use this path only when that data may be shared with
the artifact's readers. A data-minimizing publish path is still needed for private rows.
