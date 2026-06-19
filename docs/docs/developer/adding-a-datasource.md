---
title: Adding a data source
description: How datasource engines are defined in datasources.md — the YAML schema, auth method variants, test snippets, user overrides, and a worked example.
---

# Adding a data source

Anton's datasource catalog is **markdown, not code**. Built-in engines are
defined in `anton/core/datasources/datasources.md`; the parser is
`anton/core/datasources/datasource_registry.py`. Each engine is a level-2
heading followed by a fenced `yaml` block, with free prose around it that Anton
reads for auth flows, common errors, and where to get tokens.

To add or override an engine, you usually don't touch the repo at all: put the
same format in `~/.anton/datasources.md` and Anton merges it **over** the
built-in registry at startup (`DatasourceRegistry._load()` parses built-ins
first, then user entries win by engine slug). Fields in user-defined engines
are treated as non-required so partial definitions still work. See
[Custom integrations](/connect/custom-integrations) for the user-facing flow.

## The YAML block schema

Parsed into `DatasourceEngine` / `DatasourceField` / `AuthMethod` dataclasses:

| Key | Type | Meaning |
|---|---|---|
| `engine` | string (required) | Unique slug, e.g. `postgres`. The registry key |
| `display_name` | string | Human name shown in `/connect` |
| `pip` | string | Package to install in the scratchpad before the test snippet runs |
| `name_from` | string or list | Which credential field(s) derive the default connection name (e.g. `database`, or `[account, database]` joined with `_`) |
| `popular` | bool | Surfaces the engine at the top of pickers |
| `fields` | list | The credentials to collect (when there's no auth choice) |
| `auth_method` | `choice` or empty | `choice` means the user picks from `auth_methods` first |
| `auth_methods` | list | Named variants, each with `name`, `display`, and its own `fields` |
| `test_snippet` | multiline string | Python that proves the connection works, reading only `DS_*` env vars |

Each entry in `fields`:

| Key | Type | Meaning |
|---|---|---|
| `name` | string | Field name; becomes the `DS_<NAME_UPPER>` env var |
| `required` | bool (default true) | Whether collection insists on it |
| `secret` | bool (default false) | Stored in the vault, masked in UIs |
| `description` | string | Shown to the user at collection time |
| `default` | string | Pre-filled value (e.g. `"5432"`) |

### OAuth2 variants

An auth method can carry an `oauth2` spec (see the HubSpot entry for the
canonical example):

```yaml
auth_methods:
  - name: oauth2
    display: "OAuth2 (for multi-account or publishable apps)"
    fields:
      - { name: client_id,     required: true,  secret: false, description: "OAuth2 client ID" }
      - { name: client_secret, required: true,  secret: true,  description: "OAuth2 client secret" }
    oauth2:
      auth_url: https://app.hubspot.com/oauth/authorize
      token_url: https://api.hubapi.com/oauth/v1/token
      scopes: [crm.objects.contacts.read, crm.objects.deals.read]
      store_fields: [access_token, refresh_token]
```

The prose below the block then tells Anton how to drive the flow with the
scratchpad: build the authorization URL, start a local HTTP server on port 8099
to catch the callback, open the browser with `webbrowser.open()`, exchange the
`code` at `token_url`, and store the fields listed in `store_fields` in the
vault.

## The one hard rule: secrets stay in env vars

Credentials are injected as `DS_*` environment variables before any scratchpad
code runs (namespaced per connection, e.g. `DS_POSTGRES_PROD_DB__HOST`, with
flat `DS_HOST`-style vars during active connection tests — see
`data_vault.py`). **Never embed raw secret values in code strings** — not in
`test_snippet`, not in prose examples. The test snippet must read everything
from `os.environ`. This keeps secrets out of LLM transcripts and episodic logs
(see [Security](/configure/security)).

## Prose guidance conventions

After the YAML block, write short prose that Anton will actually use:

- **Where to get credentials** — e.g. "HubSpot → Settings → Integrations →
  Private Apps → Create", or "generate an App Password at
  myaccount.google.com/apppasswords".
- **Common errors and what they mean** — e.g. `"password authentication
  failed"` → wrong password or user; `"could not connect to server"` → wrong
  host/port or firewall.
- **Auth-flow steps** for anything non-trivial (OAuth, key-pair auth).

## Worked example: a REST API source

Append this to `~/.anton/datasources.md` (or to the built-in file in a PR):

````markdown
## OpenWeather

```yaml
engine: openweather
display_name: OpenWeather
pip: requests
popular: false
fields:
  - { name: api_key, required: true,  secret: true,  description: "OpenWeather API key from home.openweathermap.org/api_keys" }
  - { name: units,   required: false, secret: false, description: "metric or imperial", default: "metric" }
test_snippet: |
  import requests, os
  r = requests.get(
      "https://api.openweathermap.org/data/2.5/weather",
      params={
          "q": "London",
          "appid": os.environ['DS_API_KEY'],
          "units": os.environ.get('DS_UNITS', 'metric'),
      },
      timeout=15,
  )
  r.raise_for_status()
  print("ok")
```

Get a key at home.openweathermap.org/api_keys — free tier allows 60 calls/min.
Common errors: 401 → key not yet activated (takes ~10 minutes after signup);
429 → rate limit exceeded.
````

Checklist for the example (and for any new engine):

1. `engine` slug is unique and lowercase.
2. Every secret field has `secret: true`.
3. `test_snippet` is self-contained, reads only `DS_*` env vars, ends with a
   `print("ok")` so success is unambiguous, and fails loudly otherwise.
4. `pip` names the one package the snippet imports (the scratchpad installs it
   before running the test).
5. Prose covers token acquisition and the two or three most common errors.

Validate by restarting Anton (the registry parses both files at startup;
malformed YAML blocks are skipped with a warning on stderr) and running
`/connect` — your engine appears by `display_name`, with fuzzy matching for
typos via `DatasourceRegistry.fuzzy_find()`. See also the
[data sources overview](/connect/data-sources).
