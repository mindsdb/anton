# Anton × Terminal-Bench 2.0

Harbor adapter for running Anton against [Terminal-Bench 2.0](https://github.com/terminal-bench/terminal-bench) tasks.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Harbor Framework                                     │
│                                                       │
│  ┌─────────────┐    ┌─────────────────────────────┐  │
│  │ Task        │    │ Docker Environment           │  │
│  │ instruction │    │                              │  │
│  │ + config    │    │  ┌────────────────────────┐  │  │
│  └──────┬──────┘    │  │ Container              │  │  │
│         │           │  │                        │  │  │
│         ▼           │  │  exec(cmd) ──► shell   │  │  │
│  ┌──────────────┐   │  │                        │  │  │
│  │ AntonAgent   │───┼──│  write_file ──► fs     │  │  │
│  │              │   │  │                        │  │  │
│  │ LLM loop:    │   │  └────────────────────────┘  │  │
│  │ plan → exec  │   │                              │  │
│  │ → observe    │   └─────────────────────────────┘  │
│  │ → repeat     │                                     │
│  └──────────────┘    ┌─────────────┐                 │
│                      │ Verifier    │                 │
│                      │ test.sh     │                 │
│                      └─────────────┘                 │
└──────────────────────────────────────────────────────┘
```

**Key design decision:** Instead of using Anton's `ChatSession` (which couples with a local scratchpad subprocess), the adapter builds a lightweight LLM planning loop directly on `LLMClient.plan()`. Shell commands are routed through Harbor's `environment.exec()` into the Docker container.

## Prerequisites

- Python **3.12+** (Harbor requirement)
- Docker (for container environments)
- `ANTHROPIC_API_KEY` environment variable

## Installation

```bash
# Install Harbor framework
pip install harbor>=0.1.45

# Or from source (recommended for latest):
pip install git+https://github.com/harbor-ai/harbor.git

# Install Anton (from repo root)
pip install -e .
```

## Quick Start

### Via Harbor CLI

```bash
# Single task
harbor run \
  --agent evals.terminal_bench.agent:AntonAgent \
  --model anthropic/claude-sonnet-4-6 \
  --task /path/to/terminal-bench/tasks/some-task

# Full benchmark suite
harbor run \
  --agent evals.terminal_bench.agent:AntonAgent \
  --model anthropic/claude-sonnet-4-6 \
  --tasks /path/to/terminal-bench/tasks/ \
  --output results/
```

### Programmatic

```python
import asyncio
from pathlib import Path
from evals.terminal_bench.agent import AntonAgent
from harbor.models.agent.context import AgentContext

async def main():
    agent = AntonAgent(
        logs_dir=Path("/tmp/anton-logs"),
        model_name="anthropic/claude-sonnet-4-6",
    )

    # setup() initializes the LLM client
    await agent.setup(environment)

    # run() executes the agent loop
    context = AgentContext()
    await agent.run(
        instruction="Fix the failing test in /app/tests/",
        environment=environment,
        context=context,
    )

    print(f"Tokens: {context.n_input_tokens} in / {context.n_output_tokens} out")
    print(f"Cost: ${context.cost_usd:.4f}")

asyncio.run(main())
```

## How It Works

1. **Setup**: Creates an `LLMClient` with `AnthropicProvider` from `ANTHROPIC_API_KEY`
2. **Run loop** (max 30 rounds):
   - Sends task instruction + conversation history to Claude Sonnet 4
   - LLM responds with tool calls (`execute` or `write_file`)
   - Tools are executed in the Docker container via `environment.exec()`
   - Results are fed back to the LLM
   - Repeats until the LLM responds without tool calls
3. **Tracking**: Token counts and cost are updated on `AgentContext` after every LLM call (available even on timeout)
4. **Logging**: Full conversation saved to `logs_dir/anton_agent_log.json`

## Tools

| Tool | Description |
|------|-------------|
| `execute` | Run any shell command in the container. Returns stdout, stderr, return code. |
| `write_file` | Write content to a file (base64-encoded to avoid escaping issues). |

## Configuration

| Env Variable | Required | Default | Description |
|-------------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key for Claude |

The model is specified via Harbor's `--model` flag (e.g., `anthropic/claude-sonnet-4-6`).

## Running Tests

```bash
# Unit tests (no Harbor or API key needed — all mocked)
uv run --extra dev pytest evals/terminal_bench/test_agent.py -v
```

## Cost Estimates

Based on Claude Sonnet 4 pricing ($3/MTok input, $15/MTok output):

| Scenario | Est. Tokens | Est. Cost |
|----------|------------|-----------|
| Simple task (5 rounds) | ~50K | ~$0.15 |
| Medium task (15 rounds) | ~200K | ~$0.60 |
| Hard task (30 rounds) | ~500K | ~$1.50 |
| Full Terminal-Bench Hard (44 tasks) | ~10M | ~$30-50 |
