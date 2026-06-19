---
title: "Connecting things: overview"
description: How Anton connects to databases, warehouses, and apps — and how the credential vault keeps secrets away from the model.
---

# Connecting things: overview

Anton is an open-source AI coworker that can execute tasks, connect to tools
and data, remember lessons, and improve its workflows over time. To work with
your data, it first needs a connection — and the whole flow is conversational.

## How `/connect` works

Type `/connect` in a chat session (or run `anton connect` from your shell):

```text
/connect

(anton) What would you like to connect?
  Examples: PostgreSQL, MySQL, Snowflake, BigQuery, Gmail, GitHub, HubSpot, Salesforce, Jira, REST API.
```

1. **Pick an engine.** Name any supported engine from the
   [data sources catalog](/connect/data-sources). If Anton doesn't recognize
   the name, it offers to set it up as a custom datasource — see
   [Custom integrations](/connect/custom-integrations).
2. **Anton collects the fields conversationally.** It shows what the engine
   needs, then asks for each value one at a time. You can also paste a
   connection string or several `key=value` pairs in one go and Anton extracts
   the fields for you. Type `help` for guidance on where to find a credential,
   or `skip` to save a partial connection and finish later with `/edit`.
3. **Secrets go into the local credential vault.** Passwords, tokens, and keys
   are stored on your machine and are never placed in LLM prompts. At run
   time they are injected as environment variables into the scratchpad — the
   model only ever sees variable names like `DS_PASSWORD`, never the values.
   See [Security model](/configure/security).
4. **Anton tests the connection.** It runs a short test snippet in an isolated
   scratchpad (installing the driver if needed). If the test fails, Anton
   shows the error and offers to let you re-enter credentials.

On success the connection is saved under a slug like `postgres-3f2a9c1b`
(engine plus a short generated name) and Anton is ready to query it.

## The five connection commands

| In chat | From your shell | What it does |
| --- | --- | --- |
| `/connect` | `anton connect` | Connect a new data source, or pass a saved slug to reconnect without re-entering credentials |
| `/list` | `anton list` | List all saved connections with their status |
| `/edit` | `anton edit NAME` | Update credentials for an existing connection (Enter keeps the current value) |
| `/remove` | `anton remove NAME` | Delete a connection from the vault (asks for confirmation) |
| `/test` | `anton test NAME` | Re-run the connection test for a saved connection |

`NAME` is the connection slug in `engine-name` format, e.g. `postgres-mydb`.
Running `/remove` with no argument shows a numbered list to pick from.

## Reconnecting

Saved connections survive across sessions. To reattach one in a new session,
pass its slug:

```text
/connect postgres-3f2a9c1b
```

or just ask in plain language — "connect to my Gmail and find unanswered
emails" — and Anton finds the credentials in the vault itself.

## Next steps

- Browse the full engine catalog: [Data sources](/connect/data-sources)
- Understand where secrets live: [Security model](/configure/security)
- No connector for your tool? [Custom integrations](/connect/custom-integrations)
