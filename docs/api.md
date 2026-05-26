<!-- Cross-repo links assume sibling clones under a single parent dir
     (e.g. ~/Projects/mindsdb/). These links are intentionally broken on
     github.com — the local layout is the source of truth. -->

# API endpoints

All routes mount under `/v1/` (with a legacy mirror at `/api/v1/`). New routes go on `/v1/*` only — the `/api/v1/*` mount is a compatibility shim, not a routing strategy.

## Route inventory

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/health` | Liveness probe |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat (streaming + non-streaming) |
| `POST` | `/v1/responses` | OpenAI Responses API shape |
| `GET / POST` | `/v1/conversations` | List + create conversations |
| `GET` | `/v1/conversations/{id}/messages` | Full message history |
| `GET / POST` | `/v1/minds` | List + create Minds |
| `GET / POST / DELETE` | `/v1/datasources` | Manage data-source connections |
| `GET / POST` | `/v1/minds/{mind_name}/memory` | Mind memory |
| `GET` | `/v1/tree` | Hierarchical tree view |
| `GET` | `/v1/limits` | Plan-quota visibility |

FastAPI auto-generates Swagger at `/docs` and ReDoc at `/redoc`. There is no committed JSON schema to update — the running service is the source of truth.

## Authentication

Every request must carry a Bearer token: either a Keycloak JWT or a MindsDB API key.

```
Authorization: Bearer <mdb-api-key-or-keycloak-jwt>
```

The token is validated upstream by [auth-service](../../auth/README.md)'s `/v1/authenticate`. Identity headers (`X-User-Id`, `X-Organization-Id`, `X-Billing-Period-Start`, `X-Billing-Period-End`) are injected by that step and read in `minds/api/v1/deps.py`. This service does not decode the JWT directly.

## Chat completions

**Non-streaming.**

```bash
curl -X POST http://localhost:9010/v1/chat/completions \
  -H "Authorization: Bearer <mdb-api-key-or-keycloak-jwt>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minds",
    "messages": [{"role": "user", "content": "Hello"}],
    "metadata": {"mdb_completions_session_id": 1748503096170}
  }'
```

**Streaming.** Add `"stream": true`. The response is `text/event-stream`; each event carries an OpenAI-compatible chunk.

```bash
curl -X POST http://localhost:9010/v1/chat/completions \
  -H "Authorization: Bearer <key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minds",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "metadata": {"mdb_completions_session_id": 123}
  }'
```

**Message roles.** `system` (instructions/context), `user`, `assistant`, `function` (function-call responses).

## Streaming architecture

Each streaming request creates a `Streamer` with an internal `asyncio.Queue`. Handlers push messages with `streamer.push(role=Role.system, content="...")`; the SSE consumer formats them as `text/event-stream`. Non-streaming requests use `StreamerCollector` instead, which gathers everything into a single JSON response — the same handler code works for both.

## Usage limits

Resource-creating endpoints (`/chat/completions`, `/responses`, `/minds POST`, `/datasources POST`) gate on `require_usage_available(limits_service, ResourceType.<TYPE>)` before the resource is created. The guard raises `UsageLimitExceededError` → HTTP 429. Limits come from the `mind-usage-limits` Statsig dynamic config; self-hosted mode is unlimited.

`-1` in the dynamic config means "unlimited"; `0` means "no resource of this type can be created"; only `-1` disables the cap.
