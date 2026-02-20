# Anton

```
        ▐
   ▄█▀██▀█▄   ♡♡♡♡
 ██  (°ᴗ°) ██
   ▀█▄██▄█▀          ▄▀█ █▄ █ ▀█▀ █▀█ █▄ █
    ▐   ▐            █▀█ █ ▀█  █  █▄█ █ ▀█
    ▐   ▐
```

A self-evolving autonomous system that collaborates with people to solve problems.

Anton is not a code assistant. It's a coworker with a computer. You tell it what you need done — the same way you'd ask a colleague — and it figures out the rest. If it needs code, it writes it. If it needs a tool it doesn't have, it builds one. If it needs to run five things in parallel, it spawns minions.

## Quick start

```bash
curl -sSf https://raw.githubusercontent.com/mindsdb/anton/main/install.sh | sh
anton
```

That drops you into a conversation. Talk to Anton like a person.

Try this:

```
you> Information about inflation in the US is found on this website,
     https://www.bls.gov/news.release/cpi.nr0.htm — plot me the CPI
     items stacked per month.
```

What happens next is the interesting part. Anton doesn't have a "fetch BLS data" skill or a "plot CPI" template. It figures it out live: fetches the page with `httpx`, parses the HTML table with BeautifulSoup, installs any missing packages into the scratchpad on the fly, builds a pandas DataFrame, and generates a stacked bar chart with matplotlib — all in one conversation, with no setup. If it hits a missing library, it `install`s it mid-flow and keeps going. You get a chart on your screen and can ask follow-up questions about the data because the state is still in memory.

That's the point: you describe a problem in plain language, and Anton assembles the toolchain, writes the code, and delivers the result — the same way a coworker would if you walked over to their desk and asked.

## How it works

Anton has a four-phase execution pipeline:

```
Task → Memory Recall → Planning → Skill Building (if needed) → Execution
```

1. **Memory recall** — Loads past session summaries, relevant learnings, skill-local notes, and project context from `.anton/`. Every task starts with what Anton already knows.

2. **Planning** — An LLM breaks the task into atomic steps, mapping each to a known skill. If a step needs something that doesn't exist, it's flagged for building.

3. **Skill building** — For unknown steps, Anton generates Python skill modules on the fly, validates them, and registers them. These persist in `.anton/skills/` so they're available next time.

4. **Execution** — Steps run in dependency order. Results, durations, and errors are logged. After completion, Anton extracts reusable learnings and records them.

The chat interface wraps this pipeline behind a conversational layer. Anton asks clarifying questions, interprets context, and only fires the pipeline when it has enough information.

## Workspace

When you run `anton` in a directory, it checks for an `anton.md` file. If the folder has existing files but no `anton.md`, Anton asks before setting up — it won't touch your stuff without permission.

Once initialized, the workspace looks like:

```
project/
├── anton.md              # Project context (read every conversation)
└── .anton/
    ├── .env              # Secrets (API keys, tokens — never pass through LLM)
    ├── context/          # Self-awareness files (project facts, conventions)
    ├── skills/           # User and auto-generated skills
    ├── sessions/         # Task transcripts and summaries
    ├── learnings/        # Extracted insights
    └── minions/          # One folder per minion (<id>/status.json, artifacts)
```

**anton.md** — Write anything here. Project context, conventions, preferences. Anton reads it at the start of every conversation.

**Secret vault** — When Anton needs an API key or token, it asks you directly and stores the value in `.anton/.env`. The secret never passes through the LLM — Anton just gets told "the variable is set."

All data lives in `.anton/` in the current working directory. Override with `anton --folder /path`.

## Scratchpad

Anton includes a persistent Python scratchpad — a notebook-style environment it drives programmatically. This makes Anton particularly great at **data analysis tasks**: counting, parsing, transforming, aggregating, and exploring datasets with real computation instead of LLM guesswork.

When you ask Anton to analyze data, it writes and executes Python in the scratchpad, building up state across cells exactly like a Jupyter notebook. Variables, imports, and intermediate results persist across steps. When you ask "how did you solve that?", it dumps a clean notebook-style summary of its work — code blocks, truncated output samples, error summaries — so you can follow the reasoning without wading through raw logs.

What the scratchpad handles well:
- **Data analysis** — Load a CSV, filter rows, compute aggregates, pivot tables, plot distributions
- **Text processing** — Parse logs, extract patterns, count tokens, transform formats
- **Math and counting** — Character counts, statistical calculations, combinatorics
- **Multi-step exploration** — Build up understanding incrementally, inspect intermediate results
- **LLM-powered computation** — Call `get_llm()` inside scratchpad code for AI-assisted analysis (classification, extraction, summarization) over your data

The scratchpad also has access to Anton's skills (`run_skill()`) and an `agentic_loop()` for multi-step AI workflows — so it can orchestrate tool-calling LLM loops right inside the notebook.

## Minions

Minions are background workers. Anton handles the conversation, but when a task is long-running, independent, or recurring, it spawns a minion to handle it separately.

### Spawning

In chat, Anton recognizes work that should run in the background — large batch jobs, monitoring tasks, periodic checks — and spawns a minion. From the CLI:

```bash
anton minion "check email and flag urgent items" --every 30m
anton minion "run test suite" --max-runs 1
anton minion "monitor API health" --every 5m --start 2025-01-15T09:00 --end 2025-01-15T17:00
```

Each minion gets its own directory (`.anton/minions/<id>/`) with a status file, session transcript, and any output artifacts. If scheduling options aren't passed via CLI, the minion can ask conversationally.

### Communication

Minions write their results to `.anton/minions/<id>/` — status.json, transcripts, and artifacts. The parent Anton reads these to check progress. No sockets, no IPC — just the filesystem. When a minion completes, its summary is available for the next conversation turn.

### Scheduling

Minions support flexible scheduling:
- `--every` — Repeat frequency (`5m`, `1h`, `30s`)
- `--start` / `--end` — Time window
- `--max-runs` — Cap on total executions
- `cron_expr` — Standard cron expressions for advanced scheduling

### Lifecycle

Statuses: `pending → running → completed | failed | killed`

Anton decides which minions to kill. When a minion is killed, its schedule is also removed — no orphaned jobs. Minions track their own run count and respect max_runs limits.

**Current state.** Data models (`MinionInfo`, `MinionRegistry`), CLI scaffolding, directory creation, and status persistence exist. Process spawning and real-time monitoring are not yet implemented.

## Commands

| Command | What it does |
|---|---|
| `anton` | Interactive chat |
| `anton run "task"` | Execute a task autonomously |
| `anton --folder /path` | Use a specific workspace |
| `anton skills` | List discovered skills |
| `anton sessions` | Browse past work |
| `anton learnings` | What Anton has learned |
| `anton minion "task"` | Spawn a background worker |
| `anton minion "task" --every 5m` | Spawn a recurring minion |
| `anton minions` | List tracked minions |

## Configuration

```
.anton/.env              # Workspace-local secrets and API keys
ANTON_ANTHROPIC_API_KEY  # Anthropic API key
ANTON_PLANNING_MODEL     # Model for planning (default: claude-sonnet-4-6)
ANTON_CODING_MODEL       # Model for coding (default: claude-opus-4-6)
```

Env loading order: `cwd/.env` → `.anton/.env` → `~/.anton/.env`

## Why "Anton"?

From *Silicon Valley*. Gilfoyle's AI — Son of Anton — was an autonomous system that wrote code, made its own decisions, and occasionally went rogue. We kept the name, dropped the "Son of." Same energy.

## License

MIT
