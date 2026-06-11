---
title: How a turn works
description: What happens between typing a request and getting a result — scratchpads, tools, live progress, and lessons.
---

# How a turn works

You describe an outcome; Anton figures out the steps. That's the whole
contract. This page walks through what a single turn looks like from your
side of the screen.

## 1. You describe an outcome

```
you> I hold 50 AAPL, 200 NVDA, and 10 AMZN. Get today's prices, calculate
     my total portfolio value, and give me a dashboard.
```

No special syntax. Plain language, as much or as little detail as you like.

## 2. Anton decides how to get there

Depending on the request, Anton may:

- **Answer directly** — for questions that don't need any action.
- **Open a scratchpad** — an isolated code environment where it writes and runs code, inspects the results, and iterates until the work is done. Code is a means, not the end: the scratchpad is how Anton fetches data, crunches numbers, builds reports, and wires up integrations.
- **Call tools** — web search and web fetch ([how those work](/connect/overview)), your connected data sources, and its own memory to recall relevant lessons and rules.

## 3. You watch live progress

While Anton works, you see what it's doing in real time — the steps it takes,
code it runs, and output it gets. You stay in the loop the whole way; nothing
happens in a hidden batch.

## 4. Anton may note what it learned

When a piece of work taught Anton something durable — an API quirk, a pattern
that worked, a rule worth keeping — it can record it as a lesson. Depending on
your memory mode, lessons are either saved automatically or shown to you for
confirmation:

```
Lessons learned from this session:
  1. [lesson] The stock API rate-limits at 5 requests/sec; batch requests.
Save to memory? (y/n/pick numbers):
```

Confirmation only ever happens after your answer is delivered — never
mid-task. See [Memory overview](/teach/memory-overview) and
[Lessons and rules](/teach/lessons-and-rules).

## Useful commands during and after a turn

### `/goal` — autonomous multi-turn objectives

For bigger objectives, hand Anton a goal and let it run multiple turns on its
own until the objective is complete:

```
/goal "build a weekly sales report from the postgres db" --turns 10
```

The `--turns N` limit is optional.

### `/explain` — explainability for the latest answer

```
/explain
```

Shows how the latest answer was produced: a summary, the data sources used,
and any generated SQL (per query, with the datasource and engine it ran
against).

### `/publish` and `/unpublish` — share HTML reports

When Anton produces an HTML report, `/publish` puts it on the web and gives
you a link to share. `/unpublish` lists your published reports and lets you
remove one.

## Where to go next

- [Sessions](/use/sessions) — resuming and sharing conversations
- [Workspaces](/use/workspaces) — where Anton keeps its state
- [Connect your data](/connect/data-sources) — databases, APIs, and apps
