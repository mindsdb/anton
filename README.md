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
curl -sSf https://raw.githubusercontent.com/mindsdb/anton/main/install.sh | sh && export PATH="$HOME/.local/bin:$PATH" 
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
I hold 50 AAPL, 200 NVDA, and 10 AMZN. Get today's prices, calculate my
total portfolio value, show me the 30-day performance of each stock, and
any other information that might be useful. Give me a complete dashboard.
```

What happens next is the interesting part. Anton doesn't have any particular skill to begin with. It figures it out live: scrapes live prices, writes scratchpad code on the fly, crunches the numbers, and builds you a full dashboard — all in one conversation, with no setup.
That's the point: you describe a problem in plain language, and Anton assembles the toolchain, writes the code, and delivers the result.

<img width="800"  alt="image" src="https://github.com/user-attachments/assets/39ec8b3b-65e8-4e23-8861-c649969d4e1c" />


### Explainable by default

You can always ask Anton to explain what it did. Ask it to dump its scratchpad and you get a full notebook-style breakdown: every cell of code it ran, the outputs, and errors — so you can follow its reasoning step by step.


## How it works

For the full architecture of Anton, file formats, and developer guide, see **[anton/README.md](anton/README.md)**.


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

### Memory System that you can explore

Anton has two complementary memory systems that are meant to be human readable:

**Semantic memory** — Rules, lessons, identity, and domain expertise stored as human-readable markdown at two scopes (global and per-project). After scratchpad sessions, it automatically extracts lessons from errors and long runs.

**Episodic memory** — A complete, timestamped, searchable archive of every conversation. Stored as JSONL in `.anton/episodes/`, one file per session. Anton can search past conversations using its `recall` tool when you ask about previous sessions or past work.

Configure memory via `/setup` > Memory, or set environment variables:
- `ANTON_MEMORY_MODE` — Semantic memory encoding mode (autopilot / copilot / off). Default: **autopilot**.
- `ANTON_EPISODIC_MEMORY` — Episodic memory archive (true / false). Default: **true**.

Use `/memory` to view a read-only dashboard of both memory systems.


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

Anton is a *doing* tool not a coding tool. Tools like Claude-Code exist for your codebase. The code they write *is* the focus.
Anton on the other hand, doesn't care. Yes, it writes code too, and very well, but that code is a means to an end... Anton does whatever it needs to solve a problem you give it to solve. The output is the answers and insights you want, not what code it generates.

If you're coding a commercial app, use a coding agent. If you need something *done* — a dataset analyzed, a report generated, a workflow automated — talk to Anton.

## Is "Anton" a Mind?

Yes, at mindsDB we build AI systems that collaborate with people to accomplish tasks, inspired by the culture series books, so yes, Anton is a Mind :)

## Why the name "Anton"?

We really enjoyed the show *Silicon Valley*. Gilfoyle's AI — Son of Anton — was an autonomous system that wrote code, made its own decisions, and occasionally went rogue. We thought it was was great name for an AI that can learn on its own, so we kept Anton, dropped the "Son of".

## License

AGPL-3.0 license
