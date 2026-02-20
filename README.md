# Anton

A self-evolving autonomous system that collaborates with people to solve problems.

Anton is not a code assistant. It's a coworker with a computer. You tell it what you need done — the same way you'd ask a colleague — and it figures out the rest. If it needs code, it writes it. If it needs a tool it doesn't have, it builds one. If it needs to run five things in parallel, it spawns minions.

## Quick start

```bash
curl -sSf https://raw.githubusercontent.com/mindsdb/anton/main/install.sh | sh
anton
```

That drops you into a conversation. Talk to Anton like a person.

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

## Architecture

```
anton/
├── cli.py              # Typer CLI: chat, run, skills, sessions, minions
├── chat.py             # Multi-turn conversation with tool-call delegation
├── core/
│   ├── agent.py        # Orchestrator: memory → plan → build → execute
│   ├── planner.py      # LLM-powered task decomposition
│   ├── executor.py     # Step-by-step plan execution
│   └── estimator.py    # Duration tracking for ETAs
├── skill/
│   ├── base.py         # @skill decorator and SkillResult
│   ├── registry.py     # Discovery and lookup
│   ├── builder.py      # LLM-powered skill generation
│   └── loader.py       # Dynamic module loading
├── memory/
│   ├── store.py        # Session history (JSONL transcripts)
│   ├── learnings.py    # Extracted insights indexed by topic
│   └── context.py      # Builds memory context for the planner
├── context/
│   └── self_awareness.py  # .anton/context/ files injected into every LLM call
├── minion/
│   └── registry.py     # Minion lifecycle tracking
├── llm/
│   ├── client.py       # Planning vs coding model abstraction
│   ├── prompts.py      # System prompts
│   └── anthropic.py    # Anthropic provider
├── channel/            # Output rendering (terminal, future: slack, etc.)
└── events/             # Async event bus for status updates
```

### Data layout (workspace-relative)

```
.anton/
├── .env                # API keys (local to this workspace)
├── context/            # Self-awareness files (project facts, conventions)
├── skills/             # User and auto-generated skills
├── sessions/           # Task transcripts and summaries
└── learnings/          # Extracted insights
```

All data lives in `.anton/` in the current working directory. Override with `anton --folder /path`.

## Minions

Minions are background workers. The idea: Anton handles the conversation, but when a task is long-running, independent, or recurring, it spawns a minion to handle it separately.

### How it works (design)

**Spawning.** In chat, Anton recognizes work that should run in the background — large batch jobs, monitoring tasks, periodic checks — and spawns a minion. From the CLI: `anton minion "task" --folder /path`. Each minion is a separate `Agent.run()` in its own process and workspace.

**Communication.** Minions write their results to `.anton/minions/<id>/` — a session transcript, status file, and output artifacts. The parent Anton reads these to check progress. No sockets, no IPC — just the filesystem. Anton can poll minion status, and when a minion completes, its summary is available for the next conversation turn.

**Scheduling.** Minions can have a cron expression (`cron_expr` field). A scheduled minion re-runs on its cron schedule until killed. Use case: "check my email every 30 minutes and flag anything important."

**Lifecycle.** Anton decides which minions to kill. When a minion is killed, its cron schedule is also removed — no orphaned jobs. Statuses: `pending → running → completed | failed | killed`.

**Current state.** The data models (`MinionInfo`, `MinionRegistry`) and CLI scaffolding (`anton minion`, `anton minions`) exist. The actual process spawning, filesystem protocol, and chat-level tooling are not yet implemented.

## Commands

| Command | What it does |
|---|---|
| `anton` | Interactive chat |
| `anton run "task"` | Execute a task autonomously |
| `anton --folder /path` | Use a specific workspace |
| `anton skills` | List discovered skills |
| `anton sessions` | Browse past work |
| `anton learnings` | What Anton has learned |
| `anton minion "task"` | Spawn a background worker (scaffold) |
| `anton minions` | List tracked minions (scaffold) |

## Configuration

```
.anton/.env              # Workspace-local API keys
ANTON_ANTHROPIC_API_KEY  # Anthropic API key
ANTON_PLANNING_MODEL     # Model for planning (default: claude-sonnet-4-6)
ANTON_CODING_MODEL       # Model for coding (default: claude-opus-4-6)
```

Env loading order: `cwd/.env` → `.anton/.env` → `~/.anton/.env`

## Why "Anton"?

From *Silicon Valley*. Gilfoyle's AI — Son of Anton — was an autonomous system that wrote code, made its own decisions, and occasionally went rogue. We kept the name, dropped the "Son of." Same energy.

## License

MIT
