<!-- Cross-repo links assume sibling clones under a single parent dir
     (e.g. ~/Projects/mindsdb/). These links are intentionally broken on
     github.com — the local layout is the source of truth. -->

# minds

The Minds backend — a FastAPI service serving MindsHub's data + LLM endpoints at `api.mindshub.ai/v1/*`.

## What this repo is

Exposes OpenAI-compatible `/chat/completions` and `/responses` (streaming + non-streaming), the `/minds` CRUD that wraps MindsDB-side agents, `/datasources` connection management, `/conversations` history, `/memory`, `/tree`, and `/limits` quota visibility. Built on FastAPI, SQLModel, Postgres 16, the MindsDB Python SDK, Statsig (feature-flag gates + per-user usage caps via dynamic config), and Langfuse for tracing.

## How it fits in the MindsHub system

The browser (via the [mindshub_frontend](../mindshub_frontend/README.md) SPA) and the agent runtimes ([anton](https://github.com/mindsdb/anton) running on `cowork` and on [mindshub_services](../mindshub_services/README.md)-provisioned Lightsail instances) call this service for all data + LLM operations. Every request is authenticated by the [auth](../auth/README.md) service via `/v1/authenticate`, which returns identity headers (`X-User-Id`, `X-Organization-Id`, `X-Billing-Period-Start`, `X-Billing-Period-End`) that this service reads off the inbound request. Outbound work calls MindsDB Cloud / on-prem via the MindsDB Python SDK using the caller's API key. See [System architecture (in mindshub_services)](../mindshub_services/README.md).

```mermaid
flowchart LR
    SPA[mindshub_frontend SPA]
    AGENT[anton runtime<br/>on cowork / Lightsail]
    THIS[minds backend<br/>api.mindshub.ai/v1]
    AS[auth-service]
    MDB[(MindsDB Cloud<br/>or on-prem)]
    PG[(Postgres<br/>minds DB)]
    LF[Langfuse]

    SPA -->|/v1/chat/completions<br/>/v1/minds<br/>/v1/datasources<br/>/v1/responses| THIS
    AGENT -->|/v1/chat/completions| THIS
    THIS -->|/v1/authenticate<br/>+ identity headers| AS
    THIS --> MDB
    THIS --> PG
    THIS -. tracing .-> LF
```

## Quick start

```bash
cp .env.example .env                  # fill in OPENAI__API_KEY, MINDSDB__URL, ...
make docker/run                       # Postgres + Redis + Langfuse + auto-migrate + minds on :9010
make docker/stop                      # tear down
```

Or run only the FastAPI side against your own Postgres:

```bash
make activate                         # creates ./env venv + installs dev deps
make migrate                          # alembic upgrade head
make run                              # uvicorn + watchfiles auto-reload on http://0.0.0.0:9010
```

## Repo layout

```
.
├── pyproject.toml             # FastAPI 0.128, SQLModel, SQLAlchemy 2, mindsdb_sdk, prefect, statsig, mind-castle
├── alembic.ini + alembic/     # database migrations
├── Dockerfile                 # the minds runtime image
├── Dockerfile.scratchpad      # the scratchpad-service variant baked for Lightsail snapshots
├── Makefile                   # activate, run, migrate, test/unit, test/integration, docker/run, docker/build
├── docker-compose.yml         # Postgres 16 + Redis + Langfuse + the app — full local stack
├── pytest.ini + .coveragerc   # 85% unit coverage gate
├── prefect.yaml / prefect.docker.yaml  # Prefect deployment manifests
├── minds/                     # the package
│   ├── server.py              # FastAPI app factory; mounts v1 router at /v1 and (legacy) /api/v1
│   ├── api/v1/
│   │   ├── router.py          # all v1 sub-routers
│   │   ├── deps.py            # auth dep injection (reads identity headers, builds Context)
│   │   └── endpoints/
│   │       ├── health.py      # /v1/health
│   │       ├── chat.py        # /v1/chat/completions — OpenAI-compatible (streaming + non-streaming)
│   │       ├── responses.py   # /v1/responses — OpenAI Responses API shape
│   │       ├── minds.py       # /v1/minds — CRUD on Mind entities
│   │       ├── datasources.py # /v1/datasources — connections to user data
│   │       ├── conversations.py # /v1/conversations — message history
│   │       ├── memory.py      # /v1/minds/{name}/memory
│   │       ├── tree.py        # /v1/tree
│   │       └── limits.py      # /v1/limits — plan-quota visibility
│   ├── handlers/              # chat_completions_request_handler, responses_request_handler, openai handler
│   ├── agents/                # agent orchestration (multi-step, tools)
│   ├── client/                # OpenAI + MindsDB SDK wrappers — create_mindsdb_client_from_request etc.
│   ├── common/
│   │   ├── guards/            # usage limit guard (Statsig dynamic config + DB counts → 429)
│   │   ├── statsig/           # Statsig client + per-flag helpers (feature_flags/) + dynamic_config/ (mind_limits)
│   │   ├── logging/           # structured logging + Langfuse spans
│   │   └── streaming/         # Streamer + StreamerCollector abstraction over asyncio.Queue
│   ├── db/                    # SQLModel session, pg_session, BaseSQLModel
│   ├── cache/                 # Redis-backed cache helpers
│   ├── handlers/              # chat + response request handlers (orchestration logic per endpoint)
│   └── jobs/                  # Prefect flows for offline / scheduled work
├── alembic/versions/          # one file per migration
├── deployment/
│   ├── minds/                 # minds Helm chart + values-{prod,staging,dev,alpha-dev,gitlab-dev,terabase-dev}.yaml
│   ├── minds-flows/           # Prefect-worker Helm chart
│   └── eval-job.yaml          # eval k8s Job manifest
├── tests/
│   ├── unit/                  # 85% coverage gate
│   └── integration/           # against compose stack
└── .github/workflows/
    ├── dev-build-deploy.yml          # push to dev → dev cluster
    ├── staging-build-deploy.yml      # push to main → staging cluster
    ├── prod-build-deploy.yml         # release published → prod, gated
    ├── linter.yml                    # ruff
    ├── tests-unit.yml                # pytest tests/unit/ + coverage check
    └── tests-integration.yml         # pytest tests/integration/ against ephemeral stack
```

## Local development

**Prereqs.**
- Python 3.10+ (the project supports 3.10–3.13 per `pyproject.toml`).
- Docker + Docker Compose (the recommended dev loop).
- An OpenAI API key and either MindsDB Cloud creds or a local MindsDB on `localhost:47334`.

**Service ports in the compose stack.**
- minds: `9010`
- Postgres: `35432`
- Redis: `36379`
- Langfuse UI: `3001`

**`.env` example.** `.env.example` is the canonical template. Required vars:

```env
# Auth-service this instance trusts for /v1/authenticate
AUTH_SERVICE_URL=https://auth.dev.mindshub.ai/v1

# Database
DATABASE__URI=postgresql://minds:minds@localhost:35432/minds
DATABASE__POOL_SIZE=20
DATABASE__MAX_OVERFLOW=20

# Service port
MIND_PORT=9010

# OpenAI passthrough
OPENAI__API_URL=https://api.openai.com/v1
OPENAI__API_KEY=<your key>
OPENAI__MODEL_NAME=gpt-4o
OPENAI__MAX_TOKENS=400000

# MindsDB SDK upstream
MINDSDB__URL=https://cloud.mindsdb.com

# Langfuse (optional — set LANGFUSE_ENABLED=true to instrument)
LANGFUSE_ENABLED=false
LANGFUSE_HOST=http://localhost:3001
LANGFUSE_PUBLIC_KEY=<key>
LANGFUSE_SECRET_KEY=<key>

# Statsig (feature flags + dynamic config)
STATSIG__SDK_KEY=<key>
STATSIG__ENVIRONMENT=development      # development | staging | production
STATSIG__DISABLE_NETWORK=true         # true = offline, flags fall back to defaults
STATSIG__DISABLE_ALL_LOGGING=true     # mute Statsig internal logs

# Self-hosted bypasses Statsig entirely — flags use the configured default.
DEPLOYMENT_MODE=cloud                 # cloud | self_hosted
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

- **`/v1/*` is canonical**, with a duplicate mount at `/api/v1/*` (`app.include_router(v1_router, prefix="/api")`) for older callers. Register new endpoints under `/v1/<resource>/` only.
- **Identity comes from headers, not the JWT.** The auth-service injects `X-User-Id`, `X-Organization-Id`, `X-Billing-Period-Start`, `X-Billing-Period-End` after `/v1/authenticate`. `minds/api/v1/deps.py` builds the `Context` from those headers — never decode the JWT directly here.
- **OpenAI-compatible streaming uses a `Streamer` (`asyncio.Queue`) producer-consumer pattern.** Handlers push messages via `streamer.push(role=..., content=...)`; the SSE consumer formats them. Non-streaming requests use `StreamerCollector` to gather into a single JSON response. The same handler code works for both.
- **MindsDB client comes from the request.** Call `create_mindsdb_client_from_request(request)` to extract the Bearer token and construct an authenticated MindsDB SDK client. Do not call `create_mindsdb_client(...)` with a hardcoded key for live requests. Example:
  ```python
  from minds.client.mindsdb import create_mindsdb_client_from_request

  client = create_mindsdb_client_from_request(request)
  models = client.models.list()
  databases = client.databases.list()
  ```
  Configure the upstream via `MINDSDB__URL` (default `https://cloud.mindsdb.com`).
- **Usage limits gate every resource-creating endpoint.** Call `await require_usage_available(limits_service, ResourceType.<TYPE>)` at the top of `/chat/completions`, `/responses`, `/minds POST`, `/datasources POST`. The guard raises `UsageLimitExceededError` → HTTP 429. Limits come from Statsig (`mind-usage-limits` dynamic config); self-hosted mode is unlimited.
- **Statsig flags live in `minds/common/statsig/feature_flags/<flag>.py`** with an `is_<flag>_enabled(context, settings=None)` helper. Self-hosted bypasses Statsig (returns the configured default); cloud calls `statsig.check_gate(user=build_statsig_user(context, settings), name=gate.name)`. Wire new flags through `AppSettings.feature_flag_<name>: FeatureFlagSettings`.
- **Statsig dynamic configs live in `minds/common/statsig/dynamic_config/<config>.py`**. Use the same self-hosted-vs-cloud branching as feature flags. Current example: `mind_limits.py` resolves `mind-usage-limits` for the usage guard.
- **Migrations live in `alembic/versions/`.** Autogenerate first (`alembic revision --autogenerate`), then hand-edit the file — autogenerate misses index and constraint changes regularly.
- **Branch + commit.** Topic branches off `main`; descriptive commit messages, no rigid prefix.

## Decisions

Architectural choices locked in for this service — don't redo them.

- **This service is reached at `api.mindshub.ai/v1/*` — no `/minds/` path prefix at the public layer.** The host is the service identifier. Paths under `/v1/<resource>/` describe resources, not the service. Don't propose a `/minds/` or `/anton/` segment in the public URL; new public surfaces are added by host (different subdomain) or by `/v1/<resource>/` (new resource on the same host).
- **Identity comes from headers, not the JWT this service receives.** `/v1/authenticate` on [auth](../auth/README.md) is the only validator; it injects `X-User-Id`, `X-Organization-Id`, `X-Billing-Period-Start`, `X-Billing-Period-End`. Do not decode the inbound Bearer token here. Trust the upstream check.
- **The MindsDB SDK is constructed per-request using the caller's token.** Call `create_mindsdb_client_from_request(request)`. Do not substitute a hardcoded admin token for live requests; that would silently route work to the wrong identity.
- **Statsig replaced LaunchDarkly as the feature-flag and dynamic-config layer.** Self-hosted bypasses Statsig entirely (returns the configured default); cloud calls `check_gate(...)`. New flags wire through `AppSettings.feature_flag_<name>: FeatureFlagSettings` and a helper in `minds/common/statsig/feature_flags/`.
- **Usage limits gate every resource-creating endpoint, with `-1` meaning "unlimited".** `0` means "no resource of this type can be created"; only `-1` disables the cap. Self-hosted defaults to unlimited.

## Cross-repo touch-points

- **[auth](../auth/README.md)** — every inbound request runs through the auth-service first. The identity headers this service reads (`X-User-Id`, `X-Organization-Id`, billing-period bounds) are set by auth-service's `/v1/authenticate`.
- **[mindshub_frontend](../mindshub_frontend/README.md)** — the SPA's `VITE_MINDS_API_URL` points at this service. The UI calls `/v1/minds`, `/v1/datasources`, `/v1/chat/completions`, `/v1/responses`, `/v1/conversations`, `/v1/limits` directly with CORS.
- **[anton](https://github.com/mindsdb/anton)** — the agent runtime calls `/v1/chat/completions` and `/v1/responses` from inside provisioned instances and from the Cowork desktop app. Its `minds_url` setting points here.
- **[mindshub_services](../mindshub_services/README.md)** — Lambdas do not call this service directly; the provisioned instances do.
- **MindsDB core engine** (separate product) — accessed via the MindsDB Python SDK with the caller's API key. URL is `MINDSDB__URL`, typically `https://cloud.mindsdb.com` in prod.
- **[terraform](../terraform/README.md)** — owns the RDS Postgres, ACM cert for `api.mindshub.ai`, and the EKS-OIDC role this service's pod uses.
- **[Kubernetes-Foundational-Services](../Kubernetes-Foundational-Services/README.md)** — owns the nginx-ingress that does the `/v1/X` → `/api/v1/X` rewrite, plus cert-manager + external-secrets.

```mermaid
flowchart LR
    SPA[mindshub_frontend] --> THIS[minds]
    AGENT[anton runtimes] --> THIS
    THIS --> AS[auth-service]
    THIS --> MDB[(MindsDB core)]
    TF[terraform] -. RDS + ACM + IAM .-> THIS
    KFS[K-F-S] -. ingress + secrets .-> THIS
```

## Common tasks (cookbook)

**Add a new `/v1/<resource>/` endpoint.**

1. Create `minds/api/v1/endpoints/<resource>.py` with `router = APIRouter()` + the route handlers.
2. Include it in `minds/api/v1/router.py`: `api_router.include_router(<resource>.router, prefix="/<resource>", tags=["<resource>"])`.
3. For resource-creating endpoints, gate with `await require_usage_available(limits_service, ResourceType.<TYPE>)`.
4. Add unit tests in `tests/unit/api/v1/test_<resource>.py`; add an integration test if the endpoint touches the DB.
5. Run `make test/unit/coverage && make lint`.

**Add a database column or table.**

1. Edit the SQLModel in `minds/db/` (or wherever the model lives).
2. `python -m alembic revision --autogenerate -m "describe change"`.
3. Open the generated file in `alembic/versions/` and hand-check it — autogenerate misses indexes, constraints, and many enum changes.
4. `make migrate` locally, run the affected tests, commit both the model and the migration in the same PR.

**Add a Statsig feature flag.**

1. Create the gate in the Statsig console (default-off in dev/staging/prod as appropriate).
2. Add it to `AppSettings`:
   ```python
   feature_flag_<name>: FeatureFlagSettings = Field(
       default=FeatureFlagSettings(name="<kebab-name>", default_value=False)
   )
   ```
3. Add `minds/common/statsig/feature_flags/<name>.py` with an `is_<name>_enabled(context: Context, settings: AppSettings | None = None) -> bool` helper. Follow the existing helpers (`langfuse_enabled.py`, `model_selection_enabled.py`) for the pattern: self-hosted returns `gate.default_value`; cloud calls `get_statsig(settings=settings).check_gate(user=build_statsig_user(context=context, settings=settings), name=gate.name)`.
4. Use the helper in code. Set `STATSIG__DISABLE_NETWORK=true` for local dev without Statsig connectivity — gates fall back to their configured defaults.

**Add a Statsig-driven usage limit.**

1. Add the resource type to the `ResourceType` enum in `minds/common/guards/usage.py`.
2. Add a counter implementation that queries the DB for the user's `lifetime` + `billing_cycle` counts. Billing cycle reads from the `X-Billing-Period-Start` header.
3. Update the `mind-usage-limits` Statsig dynamic config with the new resource section.
4. Call the guard from the relevant endpoint(s) before the resource-creating operation.

**Bump a dependency.**

1. Edit `pyproject.toml`.
2. Run `make activate` to rebuild the venv; `make test` to catch breakage.
3. `mindsdb_sdk` is pulled from a Git zip URL tracking the SDK repo's `main` branch — a breaking change there lands here on next `pip install`. Pin to a tag if upstream churn becomes a problem.

**Generate / refresh OpenAPI docs.** FastAPI auto-generates at `/docs` (Swagger) and `/redoc`. There is no committed JSON schema to update.

## Gotchas

- **The unit-test coverage gate is 85% (set in `.coveragerc` + `pytest.ini`).** A new endpoint with thin tests fails CI. Add the tests in the same PR as the code.
- **Identity comes from headers set by the auth-service, not from the JWT this service receives.** Do not decode the Bearer token here — trust the headers set by the upstream auth check.
- **`X-Billing-Period-Start` is ISO-8601.** When the header is missing, billing-cycle usage falls back to the lifetime count, which silently widens limits. Test both presence and absence.
- **Statsig "unlimited" is `-1`, not `0` or `null`.** `0` means "no resource of this type can be created"; only `-1` disables the cap.
- **Self-hosted mode (no Statsig) defaults all limits to unlimited.** Test the no-Statsig path when adding a new limit type.
- **`mindsdb_sdk` pulls from `main` on the upstream SDK repo.** Breaking changes there land in this repo on the next install.
- **Self-hosted deployments bypass Statsig entirely.** `DeploymentMode.SELF_HOSTED` makes every flag helper short-circuit to `gate.default_value` without calling Statsig. Test the no-Statsig path when adding a new gate.
- **Langfuse is gated per-user via the `enable_langfuse` Statsig gate.** When the gate returns false, the request runs the un-instrumented handler — do not depend on Langfuse spans being present in logging.
- **`alembic upgrade head` runs in a chart pre-deploy hook.** A bad migration blocks the deploy. Test against a copy of the prod schema before merging.
- **`/api/v1/*` is still mounted as a legacy duplicate.** Register new routes on `/v1/*` only. The legacy mount is a compatibility shim, not a routing strategy.
- **`pyproject.toml` `[tool.pytest.ini_options]` and `pytest.ini` both exist.** The `[tool.pytest]` block has `--maxfail=1 --disable-warnings`; for full failure output locally, override on the CLI.

## Where to look when something breaks

- **`/v1/authenticate` returns 401 from this service.** Check the auth-service URL in the env (`AUTH_SERVICE_URL`). Then check whether the inbound Bearer is a JWT or an API key; both should work but log differently in the auth-service.
- **Streaming response cuts off mid-stream.** Check the LB / ingress timeouts — k8s nginx-ingress `proxy-read-timeout` must exceed the LLM's response time. Default 60s is often too short.
- **MindsDB SDK calls fail with 401.** The Bearer token was not forwarded — `create_mindsdb_client_from_request(request)` is the only correct path; do not substitute `create_mindsdb_client(<env_var>)` for live requests.
- **Postgres connection pool exhausted.** `DATABASE__POOL_SIZE` (default 20) + `DATABASE__MAX_OVERFLOW` (default 20). A long-running request that doesn't close its session is the usual culprit; check `minds/db/pg_session.py` for the session lifecycle.
- **General infra context.** [../internal-documentation/infrastructure-oncall.md](../internal-documentation/infrastructure-oncall.md). Langfuse traces at the Langfuse UI host configured in `LANGFUSE_HOST`.

## Out of scope

- **Identity, API keys, entitlements** — [auth](../auth/README.md). This service only consumes the identity headers it sets.
- **Frontend UI** — [mindshub_frontend](../mindshub_frontend/README.md).
- **Agent provisioning** — [mindshub_services](../mindshub_services/README.md). Provisioned instances call into here, not the other way around.
- **Agent runtime library** — [anton](https://github.com/mindsdb/anton). The model-provider calls hit this service.
- **MindsDB core engine** — separate product (`mindsdb/mindsdb`). This service is a thin SDK consumer.
- **Cluster networking + TLS** — [Kubernetes-Foundational-Services](../Kubernetes-Foundational-Services/README.md).
- **DNS / ACM / RDS** — [terraform](../terraform/README.md).
