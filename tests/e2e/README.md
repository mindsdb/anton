# Anton E2E Test Suite

End-to-end tests that run the Anton CLI as a subprocess. Two modes:

- **Stub** (default) — responses served by a local HTTP stub server, no API key needed, fully deterministic.
- **Live** — real LLM provider, requires a configured API key.

## Running

From the repo root:

```bash
# Stub mode (fast, no API key)
python -m pytest tests/e2e/

# Live mode (real LLM, slower)
python -m pytest tests/e2e/ --live

# Single scenario file
python -m pytest tests/e2e/test_tool_execution.py

# Verbose with test names
python -m pytest tests/e2e/ -v

# Stop on first failure
python -m pytest tests/e2e/ -x
```

## Stub vs Live

| | Stub | Live |
|---|---|---|
| API key required | No | Yes (uses env vars) |
| Speed | Fast (~1–3s per test) | Slow (~5–15s per test) |
| Deterministic | Yes | No |
| Tests marked `@pytest.mark.stub_only` | Run | Skipped |
| Timeouts | Base values | 5× base values |

Tests marked `@pytest.mark.stub_only` rely on inspecting exact request payloads or scripted LLM behaviour — they are automatically skipped in `--live` mode.

## Environment (stub mode)

The harness sets all required env vars automatically. You do not need a `.env` file.

## Environment (live mode)

Live mode inherits your shell environment and applies test-specific overrides (analytics off, no autoupdates, no colour). You need a valid provider configured, e.g.:

```bash
export ANTON_OPENAI_API_KEY=sk-...
export ANTON_PLANNING_PROVIDER=openai
export ANTON_CODING_PROVIDER=openai
python -m pytest tests/e2e/ --live
```

## Scenarios

| File | What it covers |
|---|---|
| `test_boot_config.py` | Clean boot, version/sessions subcommands, missing API key, invalid provider |
| `test_basic_chat.py` | Single/multi-turn responses, blank input, exit, history file |
| `test_tool_execution.py` | Scratchpad exec, tool result forwarding, syntax errors, reset, sequential rounds |
| `test_loop_safety.py` | Max tool rounds circuit breaker, continuation limits, turn timeouts, error nudges |
| `test_session_persistence.py` | Session file creation, listing, message content, concurrent sessions |
| `test_error_handling.py` | Missing API key message, invalid provider, HTTP 500, malformed JSON, large input |
| `test_scratchpad_resilience.py` | Output flooding, progress marker consumption, error cell recovery |
| `test_credential_scrubbing.py` | DS_* secret scrubbing before LLM calls |
| `test_circuit_breaker_evasion.py` | Alternating errors, MAX_TOOL_ROUNDS backstop |
| `test_model_call_deadline.py` | A silent model call ends at `ANTON_MODEL_CALL_IDLE_TIMEOUT_S`; a call that keeps writing is not cut |