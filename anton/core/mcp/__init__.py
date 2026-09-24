"""MCP (Model Context Protocol) client support for anton.

General infrastructure, not HubSpot-specific: the transport/session/discovery
core here works against any remote, OAuth-authenticated MCP server — HubSpot
is the first caller, but Linear and PostHog also run their own MCP servers,
so this is built to be reused rather than a one-off (ENG-1816).

Today, every other connector's "tool calling" is the LLM writing ad hoc
REST/GraphQL scratchpad code against env-var-injected bearer tokens. That
doesn't fit MCP's stateful, schema-negotiated protocol: a real session has to
be opened (``initialize`` + capability negotiation) and its tools discovered
*before* the model can be told what's callable, which the scratchpad's
ad hoc code model has no place to do. So MCP tools are registered as native
``ToolDef``s instead — the same mechanism anton's own built-in tools use —
and never touch the scratchpad at all.

Module map:
  - ``servers``  — the fixed per-engine MCP server URL table.
  - ``client``   — ``McpSession``: one open session (transport + init +
                   discovery + call), for the life of a turn.
  - ``access``   — fail-closed read/write/none tool classification.
  - ``registry`` — turns a discovered tool list into namespaced ``ToolDef``s,
                   applying the access filter.
  - ``errors``   — permanent (dead credential) vs. transient (rate limit /
                   transport hiccup) failure classification, with one silent
                   retry on transient.
  - ``wiring``   — the entry points other modules call: discover tools for a
                   turn's MCP connections (folded into ``ChatSessionConfig``
                   before the session is built — see its module docstring
                   for why it has to happen there, not after), and a
                   one-shot ``call_mcp_tool`` for a caller that just needs a
                   single result (e.g. an identity lookup) with no
                   registration involved.
"""
