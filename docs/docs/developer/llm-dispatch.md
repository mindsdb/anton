---
title: LLM dispatch
description: Provider abstraction, the planning/coding model split, LLMClient, forced-tool-call structured output, prompt assembly, and trace headers.
---

# LLM dispatch

Everything LLM-related lives under `anton/core/llm/`:

| File | Role |
|---|---|
| `provider.py` | `LLMProvider` ABC + the wire types: `LLMResponse`, `ToolCall`, `Usage`, and the `StreamEvent` union (`StreamTextDelta`, `StreamToolUseStart/Delta/End`, `StreamComplete`, `StreamTaskProgress`, `StreamToolResult`, `StreamContextCompacted`) |
| `anthropic.py` | `AnthropicProvider` — direct Anthropic API |
| `openai.py` | `OpenAIProvider` — OpenAI, Azure-style, and any openai-compatible endpoint; carries a `flavor` (OpenAI / mdb.ai passthrough / generic compatible) that decides native web-tool routing |
| `client.py` | `LLMClient` — the facade the rest of the codebase calls |
| `structured.py` | Shared helpers for forced-tool-call structured output |
| `prompt_builder.py` | `ChatSystemPromptBuilder` — system prompt assembly |
| `prompts.py` | The base prompt templates |
| `tracing.py` | Per-turn `TraceContext` for outbound telemetry headers |

## Planning vs coding providers

Anton runs two provider/model pairs, configured independently in settings:

- **planning** — the user-facing turn loop. Smart, slower, more expensive.
- **coding** — fast/cheap structured work: scratchpad code help, memory
  compaction, identity extraction, consolidation, cerebellum post-mortems,
  history summarization.

`LLMClient.from_settings()` builds both from `AntonSettings`
(`planning_provider` / `planning_model` / `coding_provider` / `coding_model`).
Provider strings are `anthropic`, `openai`, or `openai-compatible`. For the
compatible case, a flavor resolver distinguishes mdb.ai passthrough (when the
base URL matches the configured Minds URL) from a generic third-party endpoint —
that decides whether `web_search`/`web_fetch` execute server-side or fall back
to handler-dispatched tools (see [Tool system](/developer/tool-system)).

## `LLMClient` surface

| Method | Provider | Returns |
|---|---|---|
| `plan(...)` / `plan_stream(...)` | planning | `LLMResponse` / `StreamEvent` iterator — the main turn loop |
| `code(...)` | coding | `LLMResponse` — free-text completions on the cheap model |
| `generate_object(Schema, ...)` | planning | A validated Pydantic instance |
| `generate_object_code(Schema, ...)` | coding | Same, on the cheap model |

## Structured output: `generate_object`

Anton has a single primitive for getting structured data out of the LLM. How it
works:

1. The Pydantic model is converted to a JSON schema via `model_json_schema()`.
2. A synthetic tool is built whose `input_schema` is that schema.
3. The provider is called with `tool_choice={"type": "tool", "name": tool_name}` —
   this *forces* the LLM to call the tool rather than returning text.
4. The tool-call input is validated through `model_validate()` and returned as
   a typed instance (or `list[Model]` for list annotations).

### Why it beats asking for JSON in text

| Old pattern (text JSON) | New pattern (`generate_object`) |
|---|---|
| "Return ONLY valid JSON, no commentary, no markdown fences" | Forced tool_choice — the LLM cannot return text |
| Manual `json.loads()` with try/except | Pydantic `model_validate()` with structural validation |
| Strip markdown fences with regex | Never needed — there's no text response to strip |
| Defensive `isinstance` checks | Pydantic catches type errors at the schema layer |
| Field-by-field `.get(key, default)` extraction | Typed attribute access on the validated instance |

### The shared helper: `structured.py`

Schema derivation and validation live in exactly one place — two pure functions
shared by `LLMClient.generate_object` (main process, async) and
`_ScratchpadLLM.generate_object` (subprocess bridge, sync — see
[Scratchpad runtime](/developer/scratchpad-runtime)):

| Function | Purpose |
|---|---|
| `build_structured_tool(schema_class)` | Pydantic model (or `list[Model]`) → `(tool_dict, validator_class, is_list)`. List inputs are wrapped in a synthetic `_ArrayWrapper` with an `items` field, because many providers refuse top-level arrays in tool input schemas |
| `unwrap_structured_response(tool_call_input, validator_class, is_list)` | Validate via Pydantic and unwrap the list wrapper. Raises `pydantic.ValidationError` on schema drift |

Pydantic is imported lazily so the module is importable without it.

Adding a new extraction call site is mechanical: define a Pydantic model with
`Field(description=...)` on each field (the descriptions double as the LLM's
instructions — no separate prompt explains the schema), call
`await session._llm.generate_object(MySchema, ...)`, wrap in try/except for
graceful degradation.

### Where `generate_object` is used

| Module | Schema | Provider | Purpose |
|---|---|---|---|
| `connect_collector.py::extract_variables` | `_ExtractionResult` | planning | Parse free-form credential input into structured fields |
| `commands/skills.py::handle_skill_save` | `_SkillDraft` | planning | Draft a skill from recent scratchpad work |
| `commands/datasource.py::handle_add_custom_datasource` | `_CustomDatasourceSpec` | planning | Identify a custom datasource's auth fields |
| `cortex.py::_compact_file` | `_CompactionResult` | **coding** | Memory deduplication during synaptic homeostasis |
| `cortex.py::maybe_update_identity` | `_IdentityFacts` | **coding** | Default-mode identity extraction every 5 turns |
| `consolidator.py::replay_and_extract` | `_ConsolidatedLessons` | **coding** | Sleep-replay extraction of lessons |
| `cerebellum.py::_run_diff` | `_DiffPassResult` | **coding** | Post-mortem error learning from cell failures |

The planning/coding split preserves each call site's original intent: anything
that previously used `_llm.code()` now uses `generate_object_code`, anything
that used `_llm.plan()` uses `generate_object`.

## Prompt assembly: `prompt_builder.py`

`ChatSystemPromptBuilder.build(...)` concatenates, in order:

1. **prefix** (from `SystemPromptContext`)
2. The base `CHAT_SYSTEM_PROMPT` (runtime identity, visualization rules,
   current datetime)
3. **Tool prompts** — optional `ToolDef.prompt` fragments
4. **Memory context** — `cortex.build_memory_context()` output
5. **Project context** — `anton.md` (after memory, so user instructions win)
6. Self-awareness context (legacy) and datasource context
7. **Procedural memory section** — one line per skill from
   `SkillStore.list_summaries()`, telling the LLM to call `recall_skill(label)`
8. **suffix**

Order matters: later sections carry more influence, which is why `anton.md`
lands after the memory sections.

## Trace headers

`tracing.py` defines a `TraceContext` (`session_id`, `turn_id`, `harness`) held
in a `ContextVar`. `ChatSession.turn_stream` installs it for the duration of a
turn, so nested calls — structured output, the cerebellum's diff, scratchpad
coding calls in the same task — inherit the same trace without threading kwargs.

The OpenAI provider reads it when talking to MindsHub and attaches
`Langfuse-Session-Id`, `Langfuse-Tags`, and `Langfuse-Metadata` headers.
Against any other openai-compatible endpoint the headers are opt-in via
`ANTON_LANGFUSE_HEADERS=1`. Direct Anthropic and raw OpenAI ignore the context.
See [Trace headers](/configure/trace-headers).
