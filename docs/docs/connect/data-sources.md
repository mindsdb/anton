---
title: Data sources
description: Catalog of databases, warehouses, vector stores, SaaS apps, and email accounts Anton can connect to out of the box.
---

# Data sources

These are the engines Anton knows how to connect out of the box. For each one
it knows the required fields, how to test the connection, and common failure
modes. Connect any of them with `/connect` — see
[Connecting things: overview](/connect/overview) for the flow.

## Databases

| Engine | What you need |
| --- | --- |
| PostgreSQL | host, port (5432), database, user, password; optional schema and SSL |
| MySQL | host, port (3306), database, user, password; optional SSL certificates |
| MariaDB | same as MySQL (wire-compatible, same driver) |
| Microsoft SQL Server | host, port (1433), database, user, password; for Azure SQL use the `server` field (e.g. `myserver.database.windows.net`) |
| Oracle Database | user, password, host, port (1521), and a service name, SID, or full DSN |
| DuckDB | path to a `.duckdb` file or `:memory:`; optional MotherDuck token for cloud databases |
| TimescaleDB | host, port (5432), database, user, password — a PostgreSQL server with the `timescaledb` extension installed |

## Warehouses

| Engine | What you need |
| --- | --- |
| Snowflake | account identifier, user, database, plus either a password or a key pair (PEM private key); optional warehouse, schema, role |
| Google BigQuery | GCP project ID and dataset; a service account JSON key (pasted or a file path) with BigQuery Data Viewer and Job User roles |
| Amazon Redshift | cluster endpoint host, port (5439), database, user, password; SSL mode defaults to `require` |
| Databricks | server hostname, HTTP path, and a personal access token (User Settings → Developer → Access Tokens) |

## Vector stores

| Engine | What you need |
| --- | --- |
| pgvector | PostgreSQL connection details for a server with the `vector` extension installed (Supabase, Neon, and RDS all support it) |
| ChromaDB | server host and port (8000) for HTTP mode, or a local persist directory; in-memory mode needs nothing |

## SaaS and CRM

| Engine | What you need |
| --- | --- |
| Salesforce | username, password, plus consumer key and secret from a connected app (Setup → App Manager) |
| HubSpot | a Private App token (recommended; starts with `pat-na1-`), or OAuth2 client ID and secret |
| Shopify | store URL plus the client ID and client secret of a custom app (Settings → Apps → Develop apps) |
| NetSuite | account ID plus OAuth 1.0a consumer key/secret and token ID/secret from an integration record |
| BigCommerce | API base URL and an API token (Advanced Settings → API Accounts) |
| Comarch Optima (WebArm API) | API base URL and an API key; optional company selector |

## Email

| Engine | What you need |
| --- | --- |
| Gmail | your address and a 16-character app password from myaccount.google.com/apppasswords — requires 2-Factor Authentication on the Google account; no OAuth setup |
| Email (generic IMAP/SMTP) | address and password (or app-specific password); IMAP/SMTP server hostnames for non-Gmail providers |

## Connect, edit, remove, test

```text
/connect                      # in chat: pick an engine and enter credentials
anton connect                 # same flow from your shell
anton connect postgres-mydb   # reconnect a saved connection

/list                         # show saved connections and their status
/edit postgres-mydb           # update credentials (Enter keeps current value)
/remove postgres-mydb         # delete from the vault (with confirmation)
/test postgres-mydb           # re-run the connection test
```

Every engine with a test snippet is verified at connect time: Anton installs
the driver in a scratchpad, runs the test, and reports the exact error if it
fails. Credentials are stored in the local vault and injected as `DS_*`
environment variables at run time — see [Security model](/configure/security).

## Your own engine definitions

The built-in catalog is a markdown registry, and you can extend it. Add your
own engine definitions at `~/.anton/datasources.md` — Anton merges them over
the built-ins at startup. Engines you create through the custom-datasource
flow are written there automatically. For the definition format, see
[Adding a datasource](/developer/adding-a-datasource).
