# Anton Server

Run Anton as an HTTP service. Exposes an OpenAI-compatible **Responses API** so any client (the antontron desktop app, your scripts, another agent) can talk to a local or remote Anton over the wire.

The endpoint surface mirrors the hosted variant in `anton_servicesrepo/scratchpad_service`, so the same client code works against either backend.

---

## Install

The server requires `fastapi` and `uvicorn`. They are **not** part of the base install — `anton` stays lean for users who only want the CLI.

The first time you run `anton serve`, Anton checks for the extras and offers to install them automatically using whichever package manager you used to install Anton itself.

```bash
anton serve
# Server packages are required to run `anton serve`:
#   - fastapi>=0.100
#   - uvicorn>=0.20
# Install with uv? [Y/n]
```

If you'd rather install them yourself, pick the line that matches your install:

| How you installed anton | Command |
| --- | --- |
| `uv tool install` (the install.sh / install.ps1 default) | `uv tool install --with fastapi --with uvicorn --upgrade anton` |
| `uv pip` / a virtualenv | `uv pip install fastapi uvicorn` |
| Plain pip | `pip install 'anton[server]'` |
| Dev checkout | `pip install -e '.[server]'` |

---

## Run

```bash
anton serve                          # binds 127.0.0.1:8765
anton serve --host 0.0.0.0 --port 9000
anton serve --reload                 # autoreload (dev only)
```

The server runs against the workspace in your current directory — same rules as `anton`. Use `anton --folder /path/to/ws serve` to point it elsewhere.

On startup you'll see:

```
Anton serving http://127.0.0.1:8765 (workspace: /Users/you/projects/foo)
```

---

## Endpoints

### `GET /health`

```json
{
  "status": "ok",
  "version": "2.26.4.30.0",
  "workspace": "/Users/you/projects/foo",
  "sessions": ["20260429_171205"],
  "pads": ["default"],
  "pad_count": 1,
  "max_pads": 5,
  "last_activity": 1714499500.123
}
```

### `POST /v1/responses`

OpenAI Responses API. Streaming SSE by default; pass `"stream": false` for a single JSON response.

**Request**

```json
{
  "input": "What's in the inbox today?",
  "model": "anton",
  "stream": true,
  "conversation": "20260429_171205"
}
```

- `input` — string, or a list of `{role, content}` messages (the last `user` message is taken as the prompt).
- `conversation` — optional. Pass an existing id to resume; omit it and Anton generates one (timestamp format `YYYYMMDD_HHMMSS`) and returns it on `response.created`.
- `model` — echoed back; Anton uses whatever model is configured in your workspace.

**Streaming events**

```
event: response.created
data: {"type":"response.created","sequence_number":1,"response":{...},"conversation_id":"..."}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":2,"item_id":"msg-...","delta":"Hello"}

event: response.in_progress
data: {"type":"response.in_progress","sequence_number":3,"thought_role":"thought.scratchpad.start","content":"..."}

event: response.completed
data: {"type":"response.completed","sequence_number":N,"response":{...full final response...}}
```

`response.in_progress` events carry `thought_role` so clients can render scratchpad activity, recall/memorize hits, progress phases, and context compactions inline. Tool/thought roles match those exposed by the hosted service.

**Non-streaming response**

```json
{
  "id": "resp-abc123def456",
  "object": "response",
  "created_at": 1714499500,
  "status": "completed",
  "model": "anton",
  "output": [
    {
      "type": "message",
      "id": "msg-...",
      "status": "completed",
      "role": "assistant",
      "content": [{"type": "output_text", "text": "..."}]
    }
  ]
}
```

### Scratchpad endpoints (`/v1/scratchpad/*`)

Direct control over named `LocalScratchpadRuntime` instances — the same surface the hosted service exposes under `/scratchpad/*`, just under a `/v1/` prefix here. Pads are keyed by `name`; `"default"` is used when the field is omitted.

| Method | Path | Body / query | Purpose |
| --- | --- | --- | --- |
| POST | `/v1/scratchpad/start` | `ScratchpadStartRequest` | Create + start a named pad. |
| POST | `/v1/scratchpad/execute` | `ScratchpadExecRequest` | Run code, return the final `Cell`. |
| POST | `/v1/scratchpad/execute-stream` | `ScratchpadExecRequest` | Run code, stream progress + final `Cell` over SSE. |
| POST | `/v1/scratchpad/install` | `ScratchpadInstallRequest` | `pip install` packages into the pad's venv. |
| POST | `/v1/scratchpad/reset` | `{name}` | Kill the runtime, clear cells, restart. |
| POST | `/v1/scratchpad/cancel` | `{name}` | Cancel the currently running cell. |
| GET | `/v1/scratchpad/view` | `?name=...` | Cells + outputs as text. |
| GET | `/v1/scratchpad/notebook` | `?name=...` | Markdown notebook-style summary. |
| GET | `/v1/scratchpad/cells` | `?name=...` | All cells as structured data. |
| POST | `/v1/scratchpad/close` | `{name}` | Stop the pad, keep persistent state. |
| POST | `/v1/scratchpad/cleanup` | `{name}` | Stop + delete the pad's venv. |
| GET | `/v1/scratchpad/list` | — | Names of all active pads. |

**Request shapes**

```json
// /v1/scratchpad/start
{
  "name": "default",
  "coding_provider": "",
  "coding_model": "",
  "coding_api_key": "",
  "coding_base_url": ""
}
```

The four `coding_*` fields are optional — leave them blank and the server fills in from your `AntonSettings` (`coding_provider`, `coding_model`, the relevant API key, `openai_base_url`). The hosted variant requires them per request; this matches that signature so the same client works against either.

```json
// /v1/scratchpad/execute  and  /v1/scratchpad/execute-stream
{
  "name": "default",
  "code": "df.head()",
  "description": "peek at the data",
  "estimated_time": "~1s",
  "estimated_seconds": 1
}
```

```json
// /v1/scratchpad/install
{ "name": "default", "packages": ["pandas", "numpy"] }
```

```json
// reset / cancel / close / cleanup
{ "name": "default" }
```

**Streaming `/v1/scratchpad/execute-stream`**

```
data: {"type":"progress","message":"running cell..."}
data: {"type":"progress","message":"finalizing output"}
data: {"type":"cell","cell":{...full final Cell...}}
```

Errors mid-stream are emitted as `{"type":"error","error":"..."}` events, after which the SSE stream closes.

**Pool semantics.** Up to `ANTON_SERVER_MAX_PADS` pads (default 5) live concurrently. Calling `start` for a name beyond the cap returns `500` with the message *"Maximum concurrent scratchpads reached. Close an existing pad first."* — close one with `/v1/scratchpad/close` to free the slot.

---

## Conversations & history

Each `conversation` id maps 1:1 to a ChatSession. History is persisted under `<workspace>/.anton/episodes/<id>_history.json` so a conversation can be resumed across restarts — pass the same id to `/v1/responses`.

In-memory live sessions are capped (default 3) and evicted oldest-first when the cap is hit; an evicted conversation simply rebuilds from disk on the next request. Override with `ANTON_SERVER_MAX_SESSIONS=10`.

---

## Configuration

The server uses your existing Anton config (the same `.anton/.env` and global `~/.anton/.env`). LLM provider, API keys, memory mode — all picked up automatically.

| Env var | Default | Notes |
| --- | --- | --- |
| `ANTON_SERVER_MAX_SESSIONS` | `3` | Max concurrent live ChatSession instances. |
| `ANTON_SERVER_MAX_PADS` | `5` | Max concurrent live `LocalScratchpadRuntime` instances. |

---

## Auth

The local server has **no authentication** by default. It binds to `127.0.0.1` so only processes on the same machine can reach it — fine for the antontron desktop app and local scripts.

If you bind to a non-loopback address (`--host 0.0.0.0`), put a reverse proxy with auth in front of it. A bearer-token check matching the hosted variant will land here when we unify the two.

---

## What's not here yet

The hosted service exposes a few extra endpoints not yet ported:

- `/v1/chat/completions` — OpenAI Chat Completions shape
- `/v1/conversations` — explicit create / list / get / patch / delete
- `/v1/sessions` — list/close live sessions
- `/files/*` — read/list workspace output files

Coming next as the antontron client surfaces real needs. The eventual goal is to merge this with `anton_servicesrepo/scratchpad_service` so there is one server, runnable both locally and on a remote instance.

---

## Design notes

- **Built as a factory, not a module-level app.** `create_app(settings: AntonSettings) -> FastAPI` — the host process owns settings resolution, which keeps tests honest and lets us bind multiple instances if we ever need to.
- **Reuses Anton's existing pieces.** `LLMClient.from_settings`, `Workspace`, `HistoryStore`, `ChatSession` — no fork. Anything you change in core flows straight through.
- **Streaming model is the source of truth.** The non-streaming branch just collects `StreamTextDelta` events from the same stream the SSE branch serves, so both paths produce identical text.
