# Anton

```
        ▐
   ▄█▀██▀█▄   ♡♡♡♡
 ██  (°ᴗ°) ██
   ▀█▄██▄█▀          ▄▀█ █▄ █ ▀█▀ █▀█ █▄ █
    ▐   ▐            █▀█ █ ▀█  █  █▄█ █ ▀█
    ▐   ▐
```


Anton is an advanced AI coworker. You tell it what you need done and it figures out the rest.

## Quick start

**macOS / Linux:**
```bash
curl -sSf https://raw.githubusercontent.com/mindsdb/anton/main/install.sh | sh && export PATH="$HOME/.local/bin:$PATH" && anton
```

**Windows** (PowerShell):
```powershell
irm https://raw.githubusercontent.com/mindsdb/anton/main/install.ps1 | iex
```

Run it by simply typing the command:
```
anton
```

That drops you into a conversation with Anton. Talk to Anton like a person, for example, ask Anton this:

```
Information about inflation in the US is found on this website:
     https://www.bls.gov/news.release/cpi.nr0.htm
Plot me the consumer price items contributions to inflation, stacked per month.
```

What happens next is the interesting part. Anton doesn't have any particular skill to begin with. It figures it out live: fetches the page, parses the HTML, writes scratchpad code on the fly, and generates a stacked bar chart with the information you asked for — all in one conversation, with no setup.
That's the point: you describe a problem in plain language, and Anton assembles the toolchain, writes the code, and delivers the result.

<img width="400" height="701" alt="image" src="https://github.com/user-attachments/assets/6e495976-0638-44a3-9094-d6a91f92ea18" />

### Another example

You don't even need to specify a specific website. For example, whatup with crypto? Try asking Anton this:

```
Analyze how Bitcoin has evolved over the past 12 months by summarizing the
overall price trend and highlighting major dips and gains (e.g., moves
greater than ~5–10%). For each significant movement, find and briefly
explain relevant news or events that may have influenced the price,
including dates and credible sources. Use only publicly accessible
information and scrape the web as needed, assuming no API keys are available.
```

## How it works

Anton collaborates with people to solve problems, by self-evolving the skills and tools it needs, operating as follows.


**Chat + Scratchpad** — For most tasks, Anton works directly in conversation. It thinks through the problem, writes and runs Python in a persistent scratchpad (variables, imports, and data survive across steps), installs packages as needed, and iterates until it has your answer.

```
Task → Memory Recall → Planning → Scratchpad Building (if needed) → Execution
```

### What is a Scratchpad?

Anton's scratchpads are notebook-style environments it drives programmatically.

When you ask Anton for something, it writes and executes Python in the scratchpad. This makes Anton particularly great at **data analysis tasks**: counting, scraping, parsing, transforming, aggregating, calling API's and exploring datasets with real computation instead of LLM guesswork.

What the scratchpad handles well:
- **Data analysis** — Load a CSV, filter rows, compute aggregates, pivot tables, plot distributions
- **Text processing** — Parse logs, extract patterns, count tokens, transform formats
- **Math and counting** — Character counts, statistical calculations, combinatorics
- **Multi-step exploration** — Build up understanding incrementally, inspect intermediate results
- **LLM-powered computation** — Call `get_llm()` inside scratchpad code for AI-assisted analysis (classification, extraction, summarization) over your data

### Explainable by default

You can always ask Anton to explain what it did. Ask it to dump its scratchpad and you get a full notebook-style breakdown: every cell of code it ran, the outputs, and errors — so you can follow its reasoning step by step.

---

## Memory System

Anton has two complementary memory systems:

**Semantic memory** — Rules, lessons, identity, and domain expertise stored as human-readable markdown at two scopes (global and per-project). After scratchpad sessions, it automatically extracts lessons from errors and long runs.

**Episodic memory** — A complete, timestamped, searchable archive of every conversation. Stored as JSONL in `.anton/episodes/`, one file per session. Anton can search past conversations using its `recall` tool when you ask about previous sessions or past work.

Configure memory via `/setup` > Memory, or set environment variables:
- `ANTON_MEMORY_MODE` — Semantic memory encoding mode (autopilot / copilot / off). Default: **autopilot**.
- `ANTON_EPISODIC_MEMORY` — Episodic memory archive (true / false). Default: **true**.

Use `/memory` to view a read-only dashboard of both memory systems.

For the full architecture, file formats, and developer guide, see **[anton/README.md](anton/README.md)**.

---

## Workspace

When you run `anton` in a directory, it checks for an `.anton` folder. If the folder exists but no `anton.md`, Anton asks before setting up — it won't touch your stuff without permission.

**.anton/anton.md** — Write anything here. Project context, conventions, preferences. Anton reads it at the start of every conversation.

**Secret vault** — When Anton needs an API key or token, it asks you directly and stores the value in `.anton/.env`. The secret never passes through the LLM — Anton just gets told "the variable is set."

All data lives in `.anton/` in the current working directory. Override with `anton --folder /path`.


## Configuration

```
.anton/.env              # Workspace-local secrets and API keys
ANTON_ANTHROPIC_API_KEY  # Anthropic API key
ANTON_PLANNING_MODEL     # Model for planning (default: claude-sonnet-4-6)
ANTON_CODING_MODEL       # Model for coding (default: claude-haiku-4-5-20251001)
ANTON_MEMORY_MODE        # Memory encoding mode (default: autopilot)
ANTON_EPISODIC_MEMORY    # Episodic memory archive (default: true)
```

Env loading order: `cwd/.env` → `.anton/.env` → `~/.anton/.env`

## Manual install

If you already have [uv](https://docs.astral.sh/uv/):
```
uv tool install git+https://github.com/mindsdb/anton.git
```

## Upgrade / Uninstall

```
uv tool upgrade anton
uv tool uninstall anton
```

### Prerequisites

- **git** — required ([macOS](https://git-scm.com/downloads/mac) / `sudo apt install git` / `winget install Git.Git`)
- **Python 3.11+** — optional (uv downloads it automatically if missing)
- **curl** — macOS/Linux only, usually pre-installed
- Internet connection. No admin/sudo required (Windows install will optionally request admin to add a firewall rule for scratchpad internet access).

### Windows: scratchpad internet access

The install script adds a Windows Firewall rule so the scratchpad can reach the internet (for web scraping, API calls, etc.). If you skipped that step or installed manually, run this in an **admin PowerShell**:

```powershell
netsh advfirewall firewall add rule name="Anton Scratchpad" dir=out action=allow program="$env:USERPROFILE\.anton\scratchpad-venv\Scripts\python.exe"
```

## How is Anton different from Claude Code / Codex?

Anton is a *doing* tool not a coding tool. Tools like Claude-Code exist for your codebase — they read your repo, edit your files etc. The code they write *is* the focus.
Anton on the other hand, doesn't care or needs a coding repo. Yes, it writes code too, but that code is a means to an end, which is why we introduced the scratchpad logic, so Anton can fetch a page, parse a table, plot a chart, call an API, crunch some numbers, ... whatever it needs to solve a problem. The output is the answer, not the source file.

If you're coding a commercial app, use a coding agent. If you need something *done* — a dataset analyzed, a report generated, a workflow automated — talk to Anton.

## Is "Anton" a Mind?

Yes, at mindsDB we build AI systems that collaborate with people to accomplish tasks, inspired by the culture series books, so yes, Anton is a Mind :)

## Why the name "Anton"?

We really enjoyed the show *Silicon Valley*. Gilfoyle's AI — Son of Anton — was an autonomous system that wrote code, made its own decisions, and occasionally went rogue. We thought it was was great name for an AI that can learn on its own, so we kept Anton, dropped the "Son of".

## License

AGPL-3.0 license
