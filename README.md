<!-- Cross-repo links assume sibling clones under a single parent dir
     (e.g. ~/Projects/mindsdb/). These links are intentionally broken on
     github.com — the local layout is the source of truth. -->

# minds

An OpenAI-compatible LLM inference API built on FastAPI. Provides stateless (`/v1/chat/completions`) and stateful (`/v1/responses`) endpoints with multi-tenant support, token usage limits via Statsig, and observability via Langfuse.


## What this repo is

Exposes OpenAI-compatible `/chat/completions` (stateless passthrough), `/responses` (stateful with conversation history), and `/models` endpoints. Supports OpenAI, Anthropic, and Google Gemini with direct adapters. Fireworks models (Kimi, DeepSeek, Qwen) route via the Anthropic-compatible API. Seamless provider switching via model aliases. Built on FastAPI, SQLModel, Postgres 16, Statsig (usage limits + feature flags), and Langfuse for tracing.


## How it fits in the system

Clients call this service for LLM inference with standardized OpenAI request/response shapes. Every request is authenticated by the [auth](../auth/README.md) service, which injects identity headers (`X-User-Id`, `X-Organization-Id`, `X-Billing-Period-Start`, `X-Billing-Period-End`). The service routes requests to the appropriate LLM provider and persists conversation history in Postgres for stateful responses.

```mermaid
flowchart LR
    CLIENT[Client apps<br/>mindshub_frontend<br/>anton runtimes<br/>etc]
    THIS[minds inference API<br/>api.mindshub.ai/v1]
    AS[auth-service]
    LLM["LLM providers<br/>(OpenAI, Anthropic,<br/>Gemini, Fireworks)"]
    PG[(Postgres<br/>minds DB)]
    LF[Langfuse]

    CLIENT -->|/v1/chat/completions<br/>/v1/responses<br/>/v1/models| THIS
    THIS -->|/v1/authenticate<br/>+ identity headers| AS
    THIS --> LLM
    THIS --> PG
    THIS -. tracing .-> LF
```

## Quick start

```bash
cp .env.example .env                  # fill in OPENAI__API_KEY, ANTHROPIC__API_KEY, ...
make docker/run                       # Postgres + Redis + Langfuse + auto-migrate + minds on :9010
make docker/stop                      # tear down
```

Or run only the FastAPI side against your own Postgres:

```bash
make activate                         # creates ./env venv + installs dev deps
make migrate                          # alembic upgrade head
make run                              # uvicorn + watchfiles auto-reload on http://0.0.0.0:9010
```

**Example request:**
```bash
curl -X POST http://localhost:9010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-User-Id: user-123" \
  -H "X-Organization-Id: org-456" \
  -d '{
    "model": "latest:sonnet",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

## Repo layout

```
.
├── pyproject.toml             # FastAPI, SQLModel, SQLAlchemy 2, statsig, Langfuse
├── alembic.ini + alembic/     # database migrations
├── Dockerfile                 # the minds runtime image
├── Makefile                   # activate, run, migrate, test/unit, test/integration, docker/run, docker/build
├── docker-compose.yml         # Postgres 16 + Redis + Langfuse + the app — full local stack
├── pytest.ini + .coveragerc   # 85% unit coverage gate
├── minds/                     # the package
│   ├── server.py              # FastAPI app factory; mounts v1 router at /v1 and (legacy) /api/v1
│   ├── inference/             # LLM inference layer (adapter pattern for multi-provider support)
│   │   ├── adapter.py         # ProviderAdapter ABC interface
│   │   ├── service.py         # InferenceService orchestrator
│   │   ├── model_resolver.py  # model alias resolution (latest:sonnet → concrete config)
│   │   ├── types.py           # shared types (UsageBox, message converters)
│   │   └── providers/         # concrete provider adapters (OpenAI, Anthropic, Gemini, Fireworks)
│   ├── api/v1/
│   │   ├── router.py          # all v1 sub-routers
│   │   ├── deps.py            # dependency injection (context, session, services)
│   │   └── endpoints/
│   │       ├── health.py      # /v1/health — liveness + readiness checks
│   │       ├── chat.py        # /v1/chat/completions — OpenAI-compatible passthrough
│   │       ├── responses.py   # /v1/responses — stateful responses with conversation history
│   │       └── models.py      # /v1/models — list available LLM models
│   ├── handlers/              # request orchestration (bridges API → inference)
│   │   ├── openai_request_handler.py      # resolves model, calls inference
│   │   ├── chat_completions_request_handler.py  # /chat/completions entry point
│   │   └── responses_request_handler.py         # /responses entry point
│   ├── services/
│   │   ├── conversations.py   # stateful conversation storage (for /responses endpoint)
│   │   ├── limits.py          # token usage limit enforcement (Statsig + DB)
│   │   └── usage.py           # token usage aggregation
│   ├── requests/              # request parsing, streaming, tracing
│   │   ├── context.py         # Context extraction from HTTP headers
│   │   ├── chat_completions_request.py  # request schema
│   │   ├── responses_request.py         # responses request schema
│   │   ├── stream.py          # SSE streaming utilities
│   │   └── langfuse_tracing.py
│   ├── client/
│   │   └── openai_client.py   # OpenAI SDK wrapper
│   ├── model/                 # SQLModel ORM entities
│   │   ├── base.py            # BaseSQLModel (uuid PK, timestamps, soft delete)
│   │   ├── chat_completion.py # token tracking for inference
│   │   ├── conversation.py    # stateful conversation storage
│   │   └── message.py         # conversation messages
│   ├── schemas/               # Pydantic request/response DTOs
│   │   ├── chat.py            # message roles, OpenAI shapes
│   │   ├── conversations.py   # conversation DTOs
│   │   ├── limits.py          # usage limit configs
│   │   └── messages.py        # message response shapes
│   ├── common/
│   │   ├── settings/          # Pydantic BaseSettings for all config (providers, limits, Statsig, etc)
│   │   ├── guards/            # usage limit enforcement
│   │   ├── statsig/           # feature flags + dynamic config (limits, Langfuse gates)
│   │   ├── logger.py          # structured logging setup
│   │   ├── authentication.py  # header parsing
│   │   └── constants.py       # header names, etc
│   ├── db/pg_session.py       # SQLAlchemy engine + session factory
│   └── server.py              # FastAPI app factory
├── alembic/versions/          # one file per migration
├── deployment/
│   └── minds/                 # Helm chart + values-{prod,staging,dev}.yaml
├── tests/
│   ├── unit/                  # 85% coverage gate
│   └── integration/           # happy path tests against compose stack
└── .github/workflows/
    ├── dev-build-deploy.yml          # push to dev → dev cluster
    ├── staging-build-deploy.yml      # push to main → staging cluster
    ├── prod-build-deploy.yml         # release published → prod, gated
    ├── linter.yml                    # ruff check + format
    ├── tests-unit.yml                # pytest + coverage check (85%)
    └── tests-integration.yml         # integration tests
```

## Local development

**Prereqs.**
- Python 3.10+ (the project supports 3.10–3.13 per `pyproject.toml`).
- Docker + Docker Compose (the recommended dev loop).
- API keys for at least one LLM provider (OpenAI, Anthropic, Gemini, Fireworks).

**Service ports in the compose stack.**
- minds: `9010`
- Postgres: `35432`
- Redis: `16379`
- Langfuse UI: `13000`

**`.env` example.** `.env.example` is the canonical template. Required vars:

```env
# Database
DATABASE__URI=postgresql://minds:minds@localhost:35432/minds

# LLM provider credentials (set at least one)
OPENAI__API_KEY=<your OpenAI key>
ANTHROPIC__API_KEY=<your Anthropic key>
GEMINI__API_KEY=<your Gemini key>
FIREWORKS__API_KEY=<your Fireworks key>

# Langfuse (optional — set LANGFUSE_ENABLED=true to enable tracing)
LANGFUSE_ENABLED=false
LANGFUSE_HOST=http://localhost:13000
LANGFUSE_PUBLIC_KEY=<key>
LANGFUSE_SECRET_KEY=<key>

# Statsig (feature flags + token usage limits)
STATSIG__SDK_KEY=<key>
STATSIG__ENVIRONMENT=development      # development | staging | production
STATSIG__DISABLE_NETWORK=true         # true = offline, flags fall back to defaults
STATSIG__DISABLE_ALL_LOGGING=true     # mute Statsig internal logs

# Deployment mode
DEPLOYMENT_MODE=self-hosted           # self-hosted | cloud
                                      # self-hosted bypasses Statsig, all limits are unlimited
```

**Database setup.** The compose stack runs migrations automatically. To use an external Postgres:

```bash
psql -c "CREATE DATABASE minds; CREATE USER minds WITH PASSWORD 'minds'; GRANT ALL PRIVILEGES ON DATABASE minds TO minds;"
export DATABASE__URI="postgresql://minds:minds@localhost:5432/minds"
make migrate
```

**Generate or inspect a migration.**

```bash
python -m alembic revision --autogenerate -m "describe change"   # then HAND-CHECK the file
make migrate                                                      # alembic upgrade head
python -m alembic current
python -m alembic history
```

## Build, test, lint, format

| Step | Command |
|---|---|
| Unit tests | `make test/unit` |
| Unit tests + coverage gate (85%) | `make test/unit/coverage` |
| HTML coverage report | `make coverage/html` → `htmlcov/` |
| Integration tests | `make test/integration` |
| All tests | `make test` |
| Lint | `make lint` (ruff `select = E,F,UP,B,SIM,I`) |
| Format | `make format` |
| Build image | `make docker/build` |

`make run` starts the required Docker services (Postgres + Redis + Langfuse) if they aren't already up, then boots the FastAPI app with `watchfiles` auto-reload on `http://0.0.0.0:9010`. `make docker/run` boots the full stack:

- PostgreSQL 16 on `35432`
- Redis 7.2 on `36379`
- Langfuse 2.87 on `3001`
- Minds app on `9010`
- Migration service (runs `alembic upgrade head` automatically on startup)
- Autoheal (restarts unhealthy containers)

`make docker/stop` tears everything down.

## API

See [docs/api.md](docs/api.md) for the route inventory, request/response examples, and the streaming architecture overview. FastAPI auto-generates Swagger at `/docs` and ReDoc at `/redoc`; the running service is the source of truth.

## Deployment

Three workflows mirror the rest of MindsHub. Builds run on `mdb-dev` (newdev EKS), prod deploy on `mdb-prod` (newprod EKS). All workflows fan out from a shared linter + unit-test gate.

| Env | Trigger | Workflow |
|---|---|---|
| dev | push to `dev` | `dev-build-deploy.yml` |
| staging | push to `main` | `staging-build-deploy.yml` |
| prod | GitHub Release published | `prod-build-deploy.yml` (`environment: prod`, `@mindsdb/devops` reviewers) |

```mermaid
sequenceDiagram
    participant Eng as Engineer
    participant GH as GitHub
    participant L as linter + tests-unit (reusable)
    participant B as build (mdb-dev)
    participant Dep as deploy (mdb-prod)
    participant K8s as newprod EKS
    Eng->>GH: gh release create vYYYY.M.D
    GH->>L: release.published
    L->>B: pass
    B->>B: build + push to ECR
    B->>Dep: image ref
    Dep->>K8s: helm upgrade minds deployment/minds -f values-prod.yaml
    K8s-->>Eng: rollout
```

The `minds-flows` Helm chart (Prefect worker) deploys from the same release. Migrations run as a pre-deploy hook in the chart (`alembic upgrade head`).

## Conventions

- **`/v1/*` is canonical**, with a duplicate mount at `/api/v1/*` for backward compatibility. Register new endpoints under `/v1/<resource>/` only.
- **Identity comes from headers, not the JWT.** The auth-service injects `X-User-Id`, `X-Organization-Id`, `X-Billing-Period-Start`, `X-Billing-Period-End`. `minds/api/v1/deps.py` builds the `Context` from those headers — never decode the JWT directly.
- **Model resolution via aliases.** Clients request models like `latest:sonnet`, `latest:gpt-4o`, `latest:gemini-2.0-flash`. `ModelResolver` maps these to concrete provider configs. New aliases are added to `minds/inference/model_resolver.py` + tested in unit tests.
- **InferenceService is the single entry point for LLM calls.** Create a fresh `ProviderAdapter` per request (never reuse). The service handles state management (usage, output, artifacts) via `InferenceResult` and `UsageBox`.
- **Streaming responses via `Streamer` (`asyncio.Queue`).** Handlers push messages; the SSE consumer formats them. Non-streaming requests use `StreamerCollector`. The same handler code works for both.
- **Token usage limits gate inference endpoints.** Call `await require_usage_available(limits_service, ResourceType.TOKENS)` before inference. The guard raises `UsageLimitExceededError` → HTTP 429. Limits come from Statsig (`mind-usage-limits` dynamic config); self-hosted mode is unlimited.
- **Stateful responses via Conversations service.** `/v1/responses` persists messages and context. `ConversationsService` handles create, append, retrieve. No agent logic — just storage and context recovery.
- **Statsig flags live in `minds/common/statsig/feature_flags/<flag>.py`** with an `is_<flag>_enabled(context, settings=None)` helper. Self-hosted returns the configured default; cloud calls `check_gate(...)`. Wire new flags through `AppSettings.feature_flag_<name>: FeatureFlagSettings`.
- **Migrations live in `alembic/versions/`.** Autogenerate first (`alembic revision --autogenerate`), then hand-edit — autogenerate misses indexes and constraints regularly.
- **Branch + commit.** Topic branches off `main`; descriptive commit messages, no rigid prefix.

## Decisions

Architectural choices locked in for this service — don't redo them.

- **This service is inference-only: no agent orchestration, no data catalog, no Mind/Datasource CRUD.** It's a stateless + stateful LLM API. All complexity lives upstream (agent runtimes, orchestration layers). This service routes, traces, and limits. If you're adding an agent framework or catalog feature, you're building the wrong service.
- **Identity comes from headers, not the JWT.** The auth-service injects `X-User-Id`, `X-Organization-Id`, `X-Billing-Period-Start`, `X-Billing-Period-End`. Trust the upstream check; never decode the Bearer token.
- **InferenceService is the single router for all LLM calls.** Don't call provider adapters directly. InferenceService handles alias resolution, adapter creation, state management, and cleanup. It's stateless per-request: every call gets a fresh adapter.
- **Multi-provider switching via model aliases.** New providers are added to `minds/inference/providers/` as concrete `ProviderAdapter` implementations. Model aliases map via `ModelResolver` — no hardcoded provider selection in handlers.
- **Statsig is the feature-flag and dynamic-config layer.** Self-hosted bypasses it (returns defaults); cloud calls `check_gate(...)`. New limits and flags wire through `AppSettings` fields.
- **Token usage limits gate inference endpoints.** Statsig provides the cap; the DB tracks consumption. `-1` = unlimited; `0` = no requests; `N > 0` = firm ceiling. Self-hosted defaults to unlimited.

## Cross-repo touch-points

- **[auth](../auth/README.md)** — every inbound request is validated by the auth-service. The identity headers (`X-User-Id`, `X-Organization-Id`, billing-period bounds) are set by `/v1/authenticate`.
- **Client applications** — call `/v1/chat/completions`, `/v1/responses`, `/v1/models` with OpenAI-compatible request shapes.
- **LLM provider APIs** — (OpenAI, Anthropic, Google, Fireworks). Provider credentials are configured via env vars (`OPENAI__API_KEY`, `ANTHROPIC__API_KEY`, etc).
- **[terraform](../terraform/README.md)** — owns the RDS Postgres, ACM cert for `api.mindshub.ai`, and the EKS-OIDC role the pod uses.
- **[Kubernetes-Foundational-Services](../Kubernetes-Foundational-Services/README.md)** — owns the nginx-ingress, cert-manager, and external-secrets.
- **Langfuse** — (optional observability). When enabled via Statsig gate + env config, traces are sent to the Langfuse instance specified in `LANGFUSE_HOST`.
- **Statsig** — feature-flag and dynamic-config service. Token usage limits, Langfuse instrumentation gates, and other feature toggles are fetched from Statsig in cloud mode.

```mermaid
flowchart LR
    CLIENTS["Client apps<br/>(mindshub_frontend,<br/>anton runtimes, etc)"]
    THIS["minds<br/>inference API"]
    AUTH["auth-service"]
    LLM["LLM providers<br/>(OpenAI, Anthropic,<br/>Gemini, Fireworks)"]
    PG[(Postgres<br/>minds DB)]
    STATSIG["Statsig<br/>(limits, gates)"]
    LF["Langfuse<br/>(optional)"]

    CLIENTS -->|/v1/chat/completions<br/>/v1/responses<br/>/v1/models| THIS
    THIS --> AUTH
    THIS --> LLM
    THIS --> PG
    THIS -.->|limits, gates| STATSIG
    THIS -.->|traces| LF
```

## Common tasks (cookbook)

**Add support for a new LLM provider.**

1. Create `minds/inference/providers/<provider_name>_adapter.py` implementing `ProviderAdapter` ABC:
   - `async def complete(config, messages, stream, ...) -> StreamingResponse | JSONResponse`
   - `async def get_last_usage() -> tuple[int, int] | None` (input_tokens, output_tokens)
   - `def get_last_output() -> dict[str, Any] | None` (OpenAI-format assistant message)
   - `def get_last_artifacts() -> list[dict]` (provider-specific outputs)
2. Create a helper module `minds/inference/providers/<provider_name>.py` for message translation, request building, response parsing.
3. Add provider config to `AnthropicSettings`, `OpenAISettings`, etc in `minds/common/settings/app_settings.py` (API key, base URL, model names).
4. Register the provider in `InferenceService.complete()`: add a branch that detects the provider and instantiates your adapter.
5. Add model aliases in `ModelResolver` that map to your provider. Test the alias resolution.
6. Add unit tests in `tests/unit/inference/providers/test_<provider_name>_adapter.py` covering streaming, non-streaming, errors, usage tracking.
7. Run `make test/unit/coverage && make lint`.

**Add a database column or table.**

1. Edit the SQLModel in `minds/model/` (must inherit from `BaseSQLModel`).
2. Import the model in `alembic/env.py` so migrations discover it.
3. `python -m alembic revision --autogenerate -m "describe change"`.
4. Hand-check the migration file in `alembic/versions/` — autogenerate misses indexes, constraints, and enum changes.
5. `make migrate` locally, run `make test/unit`, commit both the model and the migration in the same PR.

**Add a Statsig feature flag.**

1. Create the gate in the Statsig console.
2. Add it to `AppSettings`:
   ```python
   feature_flag_<name>: FeatureFlagSettings = Field(
       default=FeatureFlagSettings(name="<kebab-name>", default_value=False)
   )
   ```
3. Add `minds/common/statsig/feature_flags/<name>.py` with `is_<name>_enabled(context: Context, settings: AppSettings | None = None) -> bool`. Self-hosted returns the default; cloud calls Statsig.
4. Use it in code. Set `STATSIG__DISABLE_NETWORK=true` for local dev — gates fall back to defaults.

**Add a token usage limit.**

1. Token limits already gate `/chat/completions` and `/responses`. If you need to add a new limit type (not just token-based), add it to `ResourceType` in `minds/common/guards/usage.py`.
2. Add a counter that reads usage from the `ChatCompletion` table (or your new table) scoped to the user + organization.
3. Update the `mind-usage-limits` Statsig dynamic config with the new limit section.
4. Call `await require_usage_available(limits_service, ResourceType.<TYPE>)` at the top of the protected endpoint.

**Bump a dependency.**

1. Edit `pyproject.toml`.
2. Run `make activate` to rebuild the venv; `make test` to catch breakage.

**Generate / refresh OpenAPI docs.** FastAPI auto-generates at `/docs` (Swagger) and `/redoc`. There is no committed JSON schema.

## Gotchas

- **The unit-test coverage gate is 85% (set in `.coveragerc` + `pytest.ini`).** New code with thin tests fails CI. Add tests in the same PR.
- **Identity comes from headers, not the JWT.** Do not decode the Bearer token — trust the auth-service headers.
- **`X-Billing-Period-Start` is ISO-8601.** When missing, usage falls back to lifetime, silently widening limits. Test both cases.
- **Statsig "unlimited" is `-1`, not `0` or `null`.** `0` = no requests allowed; only `-1` disables the cap.
- **Self-hosted bypasses Statsig entirely.** `DeploymentMode.SELF_HOSTED` makes all flags return their defaults without calling Statsig. Test the no-Statsig path when adding a new gate.
- **InferenceService creates a fresh adapter per request.** Never reuse adapters across requests. Stateful adapters cause concurrent-request races and leaked state.
- **ModelResolver is the single path for alias → provider mapping.** Don't hardcode provider selection in handlers. All aliases go through the resolver.
- **Langfuse is gated via the `enable_langfuse` Statsig gate.** When the gate returns false, tracing is skipped — do not depend on Langfuse spans being present.
- **Streaming responses are async by nature.** Token usage and final output are populated after the stream completes, read from `InferenceResult.usage_box`. Non-streaming reads them immediately.
- **`alembic upgrade head` runs in a chart pre-deploy hook.** A bad migration blocks the deploy. Test against a copy of prod schema before merging.
- **`/api/v1/*` is a legacy mount.** Register new routes on `/v1/*` only.
- **`pyproject.toml` and `pytest.ini` both exist.** The `pytest.ini` takes precedence for test config.

## Where to look when something breaks

- **LLM provider requests fail with 401/403.** Check the API key in the env (`OPENAI__API_KEY`, `ANTHROPIC__API_KEY`, etc) and the provider's base URL (`OPENAI__API_URL`, etc). Verify the key has the right permissions in the provider's console.
- **Streaming response cuts off mid-stream.** Check the LB / ingress timeouts — k8s nginx-ingress `proxy-read-timeout` must exceed the LLM's response time. Default 60s is often too short. Also check provider-side timeouts.
- **Model alias resolves to wrong provider.** Check `ModelResolver.resolve()` in `minds/inference/model_resolver.py` — ensure the alias is registered and maps to the correct provider config.
- **Token usage not tracked.** Check that the LLM provider adapter's `get_last_usage()` method is populating `InferenceResult.usage_box`. For streaming, verify the adapter populates usage *before* the stream ends (it's read after completion).
- **Postgres connection pool exhausted.** Check `DATABASE__POOL_SIZE` (default 5) + `DATABASE__MAX_OVERFLOW` (default 10). Long-running requests that don't close sessions are the usual culprit. See `minds/db/pg_session.py` for the session lifecycle.
- **Stateful responses (`/v1/responses`) lose context.** Ensure `ConversationsService.append_message()` is being called during the request, and that `get_conversation()` is loading the full history before building the context.

## Out of scope

- **Agent orchestration, data catalog, Mind/Datasource management.** This is a stateless + stateful inference API only. Data querying, agent workflows, and mind configuration are upstream concerns.
- **Identity and entitlements** — [auth](../auth/README.md). This service consumes identity headers only.
- **LLM provider implementations.** This service is a thin multi-provider router. Provider-specific logic (RAG, vision, tools, structured output) happens in the client or in an upstream orchestration layer.
- **Cluster networking, TLS, secrets management** — [Kubernetes-Foundational-Services](../Kubernetes-Foundational-Services/README.md).
- **DNS, ACM, RDS infrastructure** — [terraform](../terraform/README.md).
