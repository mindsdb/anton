"""What the agent says when asked which model is serving the conversation.

ENG-1638. The system prompt used to hand the model one ``RUNTIME IDENTITY``
block built from *configuration* and then tell it "you already know what model
you are running on — NEVER ask". Configuration is the right input for exactly
one of the two questions that block answered:

- **"What should code you write call?"** — the configured provider/SDK. Correct,
  and kept (``build_runtime_context``).
- **"What is answering me right now?"** — NOT configuration. ``mindshub_air`` is
  a gateway alias, not a model; a mis-applied local endpoint (ENG-1634) leaves
  the configured name pointing at a model that never ran; and on the web pod the
  block rendered empty while the mandate stayed, so the model denied being the
  model it was (a Grok session answering "No — I'm Anton, not Grok").

This module owns the second question. The rule mirrors how every other surface
in the product already names a model (the composer picker, billing, the
website): the alias the user picked, never the vendor model underneath it.

- ``mindshub_air`` → **"MindsHub Air"**, always. It is a product tier whose
  backing model changes without notice (pricing page / FAQ), so the agent must
  neither name nor deny what is underneath — it says the model is not
  disclosed. Every other catalog alias is *named after* its model
  ("Grok 4.6", "GPT 5.6 Luna"), so for those the served model is the honest
  answer and reveals nothing the picker didn't.
- Anything else → the model the provider **actually reported serving**
  (``LLMResponse.model``, echoed by MindsHub, OpenAI, Anthropic, Ollama and
  LM Studio alike), falling back to the requested id before the first response
  arrives — labelled as unconfirmed, so a turn-1 answer is still honest.
- Nothing known → the agent is told to say it cannot verify, and never to guess
  or deny. The old prompt had no such fallback; that absence is what turned a
  wrong input into a confident falsehood.

Kept free of heavy imports so the cloud pod (``anton.cloud_turn``) can use it
without pulling in the CLI's rich/typer stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anton.config.settings import AntonSettings

#: The MindsHub free-tier alias. Same literal as ``minds_client.MINDS_FREE_TIER_MODEL``;
#: duplicated rather than imported so this module stays import-light.
MINDSHUB_AIR_ALIAS = "mindshub_air"

#: Aliases the agent reports by their product label instead of the served model.
#: Seeded from the catalog's ``label`` for the one alias whose label does NOT
#: name the model underneath. Every other MindsHub alias is named after its
#: model, so reporting the served id for those matches the picker already.
#: Reading ``label`` live from ``/v1/models`` is the eventual source; a static
#: map is enough while Air is the only opaque tier.
OPAQUE_ALIAS_LABELS: dict[str, str] = {MINDSHUB_AIR_ALIAS: "MindsHub Air"}

#: ``response.model`` from a BYOK / local server is untrusted text headed into
#: the system prompt. Cap it and keep it on one line so it can carry a name and
#: nothing else.
_MAX_MODEL_NAME_LEN = 80


#: The deprecated pin prefix cowork-server still resolves (``latest:sonnet``).
#: A stale ``latest:mindshub_air`` pin is overlaid onto settings verbatim, so
#: without stripping it the Air rule would miss and the served vendor id would
#: leak — the exact thing this module exists to prevent.
_LEGACY_PIN_PREFIX = "latest:"


def sanitize_model_name(value: object) -> str | None:
    """A model id safe to interpolate into the prompt, or ``None``.

    Non-strings (a Mock in tests, ``None`` from an SDK that omitted the field)
    are dropped rather than stringified. Newlines and control characters are
    removed so the value cannot start a new prompt line; the length is capped;
    the deprecated ``latest:`` pin prefix is dropped so an alias compares (and
    reads) the same however it was stored.
    """
    if not isinstance(value, str):
        return None
    cleaned = "".join(ch for ch in value if ch.isprintable() and ch not in "\r\n").strip()
    if cleaned.lower().startswith(_LEGACY_PIN_PREFIX):
        cleaned = cleaned[len(_LEGACY_PIN_PREFIX):].strip()
    if not cleaned:
        return None
    return cleaned[:_MAX_MODEL_NAME_LEN]


def serving_model_lines(
    *, requested: object, served: object
) -> list[str]:
    """Prompt lines describing the model serving this conversation.

    ``requested`` is the planning model id anton asked for (``settings.planning_model``);
    ``served`` is what the provider reported back on the last planning response
    (``LLMResponse.model``), ``None`` before the first response.

    Returns ``[]`` when neither is known — the caller then emits the
    cannot-verify fallback instead of a mandate.
    """
    req = sanitize_model_name(requested)
    srv = sanitize_model_name(served)

    if req in OPAQUE_ALIAS_LABELS:
        label = OPAQUE_ALIAS_LABELS[req]
        return [
            f"- Serving model: {label} (MindsDB's hosted model tier).",
            f"- MindsHub does not disclose which model serves {label}. If asked what "
            f"is underneath, say it is not disclosed — never name, guess, or deny an "
            f"underlying vendor model.",
        ]
    if srv:
        return [f"- Serving model: {srv} (as reported by the provider on its last response)."]
    if req:
        return [
            f"- Serving model: {req} (the model that was requested; the provider has "
            f"not yet confirmed it)."
        ]
    return []


_CANNOT_VERIFY_LINE = (
    "- The harness did not report which model is serving this conversation. If "
    "asked, say you cannot verify it — do not guess, and do not deny being any "
    "particular model."
)

_IDENTITY_RULE_LINE = (
    "- If asked which model or provider is serving this conversation, answer from "
    "the serving-model line above and nothing else. Do not infer it from your "
    "training, your style, or any configured model id."
)

_CONFIGURED_HEADER = (
    "CONFIGURED LLM (what code you write should call — not necessarily what is "
    "serving this conversation):"
)

_CONFIGURED_RULE_LINE = (
    "- When building tools or code that needs an LLM, use this configured provider "
    "and SDK. Do not ask the user which LLM or API to use — the configuration above "
    "is the answer."
)


def build_runtime_identity_section(
    *, identity_lines: list[str], configured_block: str
) -> str:
    """Render the ``RUNTIME IDENTITY`` system-prompt section.

    Two sub-blocks, one per question:

    - identity — ``identity_lines`` from :func:`serving_model_lines`, or the
      cannot-verify fallback when empty. There is always an identity answer;
      what is never emitted is a claim that the model "already knows".
    - configured — ``configured_block`` from :func:`build_runtime_context`
      (provider + model ids for code the agent writes). Omitted entirely when
      empty, so a host that injects nothing gets no dangling reference to
      "the runtime info above".
    """
    out = ["RUNTIME IDENTITY:"]
    if identity_lines:
        out.extend(identity_lines)
        out.append(_IDENTITY_RULE_LINE)
    else:
        out.append(_CANNOT_VERIFY_LINE)

    configured = configured_block.strip()
    if configured:
        out.append("")
        out.append(_CONFIGURED_HEADER)
        out.append(configured)
        out.append(_CONFIGURED_RULE_LINE)
    return "\n".join(out)


def build_runtime_context(settings: "AntonSettings") -> str:
    """The configured-LLM block: provider + model ids, plus a connected Mind.

    This is the input for *code the agent writes*, never for "which model is
    serving this conversation" — that is :func:`serving_model_lines`. The
    workspace path and memory mode used to ride along here; neither serves
    either question and the path leaked into traces, so they are gone
    (ENG-1638 security note).
    """
    ctx = (
        f"- Provider: {settings.planning_provider}\n"
        f"- Planning model: {settings.planning_model}\n"
        f"- Coding model: {settings.coding_model}"
    )
    if settings.minds_api_key and (
        settings.minds_mind_name or settings.minds_datasource
    ):
        engine = settings.minds_datasource_engine or "unknown"
        ctx += f"\n\n**CONNECTED MIND (Minds):**\n"
        if settings.minds_mind_name:
            ctx += f"- Mind: {settings.minds_mind_name}\n"
        if settings.minds_datasource:
            ctx += (
                f"- Datasource: {settings.minds_datasource}\n"
                f"- Engine: {engine}\n"
            )
        ctx += (
            f"- Minds URL: {settings.minds_url}\n"
            f"- To query data, use the scratchpad with the built-in `query_minds_data()` function.\n"
            f"  It is pre-loaded in the scratchpad namespace — DO NOT import it. Just call it directly.\n"
            f'  Example: result = query_minds_data("SELECT * FROM users LIMIT 5")\n'
            f"  Returns dict with 'type', 'data' (list of rows), 'column_names', 'error_message'.\n"
            f'  Optional: query_minds_data("SELECT ...", datasource="other_ds")\n'
        )
        if settings.minds_datasource:
            ctx += f"- Write SQL appropriate for the {engine} engine.\n"
    return ctx
