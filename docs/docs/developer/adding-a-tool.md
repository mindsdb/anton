---
title: Adding a tool
description: Step-by-step walkthrough for adding a built-in tool — ToolDef, handler signature, registration — and when a tool is warranted at all.
---

# Adding a tool

## First: do you actually need a tool?

Anton's philosophy is that **most work goes through the scratchpad**. The agent
doesn't need a `read_csv` tool or a `http_get` tool — it writes Python in a
scratchpad cell and gets exactly the capability it needs, with credentials
already injected as `DS_*` env vars and an LLM bridge available via `get_llm()`.

Tools are reserved for **primitives the model must call directly** — operations
that:

- need to run in the **main process**, not the scratchpad subprocess
  (e.g. `memorize` writes through the Cortex; `recall_skill` reads the
  session's skill store);
- must return content the model consumes **as part of the conversation**, not
  as program output (e.g. `read_image` returns vision blocks);
- are part of the **turn protocol itself** (e.g. `create_artifact` claims a
  folder the renderer watches).

If your idea is "let Anton do X with an API", the answer is almost always a
[data source definition](/developer/adding-a-datasource) plus scratchpad code —
not a tool. A senior reviewer will push back on tool proposals that the
scratchpad already covers.

## The pieces

A tool is three things (see [Tool system](/developer/tool-system) for the full
machinery):

1. A `ToolDef` — name, LLM-facing description, JSON `input_schema`, and a
   handler (defined in `anton/core/tools/tool_defs.py`, or in a dedicated file
   like `recall_skill.py`).
2. An async **handler** with the signature
   `async def handle_mytool(session: "ChatSession", tc_input: dict) -> str` —
   it receives the live session and the tool-call input dict, and returns the
   result string sent back to the LLM (vision tools may return a list of
   content blocks instead).
3. **Registration** in `ChatSession._build_core_tools()`
   (`anton/core/session.py`), optionally guarded by a condition (workspace
   bound, episodic enabled, ...). Embedding hosts can instead pass extra
   ToolDefs via the session's `_extra_tools`.

## Walkthrough: `recall_skill.py`, the cleanest template

`anton/core/tools/recall_skill.py` is the best file to copy because the
definition, schema, handler, and docs live together in ~130 lines.

**1. The description — written for the LLM, not for humans:**

```python
_DESCRIPTION = (
    "Retrieve a procedural skill from long-term memory into your working "
    "context. Call this when you recognize that one of the skills listed in "
    "the '## Procedural memory' section of your system prompt applies to the "
    "user's current request. ..."
)
```

Say *when* to call the tool and what comes back. Bad descriptions are the
number-one cause of tools being over- or under-used.

**2. The input schema — plain JSON Schema:**

```python
_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "description": "The skill label to recall, e.g. 'csv_summary'. ...",
        },
    },
    "required": ["label"],
}
```

**3. The handler — validate defensively, return strings, never raise:**

```python
async def handle_recall_skill(session: "ChatSession", tc_input: dict) -> str:
    label_in = (tc_input.get("label") or "").strip()
    if not label_in:
        return "ERROR: recall_skill requires a non-empty 'label' parameter. ..."

    store = getattr(session, "_skill_store", None)
    if store is None:
        return "ERROR: no skill store is wired into this session. ..."

    skill = store.load(label_in)
    # ... typo recovery via store.closest_match(label_in) ...
    store.increment_recommended(skill.label, stage=1)
    return _format_skill_response(skill, warning=warning)
```

Handler conventions visible here:

- **Read session state via `getattr` with a fallback** — sessions can be built
  without every subsystem, and a tool must degrade to a clear error string,
  not an `AttributeError`.
- **Return errors as strings the LLM can act on** ("Available skills: ...") —
  exceptions from handlers turn into opaque failures; informative strings let
  the model self-correct on the next round.
- **Capture mechanical signals in the handler** (the `recommended` counter),
  not via LLM conventions.

**4. The ToolDef:**

```python
RECALL_SKILL_TOOL = ToolDef(
    name="recall_skill",
    description=_DESCRIPTION,
    input_schema=_INPUT_SCHEMA,
    handler=handle_recall_skill,
)
```

**5. Registration** — in `ChatSession._build_core_tools()`:

```python
# Procedural memory retrieval — always available, no-op if no skills.
self.tool_registry.register_tool(RECALL_SKILL_TOOL)
```

If your tool only makes sense under a condition, guard it the way `recall`
(episodic enabled) and the artifact tools (workspace bound) are guarded —
hiding a tool entirely beats registering one that returns errors.

## Checklist

1. Define `_DESCRIPTION`, `_INPUT_SCHEMA`, the handler, and the `ToolDef` —
   ideally in one new file under `anton/core/tools/`.
2. Register it in `_build_core_tools()` with the right guard.
3. Make the handler total: every input shape returns a string; no path raises.
4. Keep results within reason — large outputs bloat history (episodic logging
   truncates tool results at 2000 chars; your tool result itself goes into the
   LLM history uncut, so truncate big payloads yourself).
5. Optional: set `ToolDef.prompt` if the tool needs a system-prompt fragment
   (it is appended by the prompt builder; most tools don't need this).
6. Add tests under `tests/` — handler unit tests can call
   `await handle_mytool(fake_session, {...})` directly; see
   `tests/test_acc.py` and friends for session-faking patterns, and
   [Contributing](/developer/contributing) for how to run the suite.
