---
title: Custom integrations
description: When there is no connector for something, ask Anton — it builds the integration itself.
---

# Custom integrations

Anton doesn't wait for someone to build a connector. If you need to talk to a
service it has no built-in support for, ask — Anton writes the integration
code itself in the scratchpad, configures it, and gets it running.

## Example: WhatsApp

```text
Set up a WhatsApp integration so I can message you from my phone.
```

Anton figures out what the integration needs, writes the code, sets it up,
and keeps it running — so you can chat with it from WhatsApp, Telegram, or
whatever channel you need. The same pattern works for any API or tool: you
describe the outcome, Anton builds the plumbing.

## Custom datasources via `/connect`

The [connect flow](/connect/overview) also handles services that aren't in
the [built-in catalog](/connect/data-sources). Name any tool or service at the
`/connect` prompt:

```text
/connect

(anton) What would you like to connect?
> github
```

Anton asks how the service authenticates (a short description, no secrets),
then works out the connection spec itself: the credential fields, the pip
package to use, and a test snippet that verifies the connection. It collects
the credentials conversationally — type `help` on any field for guidance, or
paste several values at once — then tests the connection and saves it.

The generated engine definition is written to `~/.anton/datasources.md`, so
the new engine behaves like a built-in from then on: it shows up in `/list`,
and `/edit`, `/remove`, and `/test` all work. Removing the last connection
for a custom engine also removes its definition. You can edit the definitions
in that file by hand too — see
[Adding a datasource](/developer/adding-a-datasource) for the format.

## Set expectations honestly

- **This is agent-built code.** Anton writes the integration on the fly,
  guided by what it knows about the service. Review what the code does —
  especially anything that sends messages or modifies data on your behalf.
- **Credentials still go through the vault.** Custom connections use the same
  credential vault as built-in engines: secrets are stored locally and
  injected as `DS_*` environment variables at run time, never placed in LLM
  prompts. See [Security model](/configure/security).
- **Test snippets may be imperfect.** If a generated connection test fails,
  Anton shows the error and lets you correct the details or retry.
