"""Custom datasource creation (LLM-assisted)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from anton.connect_collector import extract_variables
from anton.commands.datasource.helpers import show_credential_help
from anton.core.datasources.datasource_registry import DatasourceEngine, DatasourceField
from anton.utils.datasources import persist_custom_engine
from anton.utils.prompt import prompt_or_cancel

if TYPE_CHECKING:
    from rich.console import Console
    from anton.chat import ChatSession
    from anton.core.datasources.datasource_registry import DatasourceRegistry


class _CustomDatasourceField(BaseModel):
    """One credential field in a custom-datasource spec."""

    name: str = Field(
        ...,
        description=(
            "snake_case field name (e.g. 'host', 'api_key'). Must be a "
            "valid Python identifier; this becomes both the on-disk key "
            "and the env var suffix (DS_<NAME>)."
        ),
    )
    value: str = Field(
        default="",
        description=(
            "Inline value if the user already provided one in their "
            "description, otherwise empty string."
        ),
    )
    secret: bool = Field(
        default=False,
        description=(
            "True if the field is sensitive (passwords, API keys, "
            "tokens) — affects how it's stored and prompted for."
        ),
    )
    required: bool = Field(
        default=False,
        description=(
            "Always False for custom datasources — all fields are optional "
            "by design. Never set this to True."
        ),
    )
    description: str = Field(
        default="",
        description=(
            "One-line description shown to the user when prompting "
            "for this field."
        ),
    )


class _CustomDatasourceSpec(BaseModel):
    """Structured output of the LLM call in handle_add_custom_datasource."""

    display_name: str = Field(
        ...,
        description="Human-readable name for the service (e.g. 'GitHub API').",
    )
    pip: str = Field(
        default="",
        description=(
            "pip-installable package name (or space-separated names) "
            "needed to interact with this service. Empty string if no "
            "extra package is required (e.g. plain HTTPS via stdlib)."
        ),
    )
    test_snippet: str = Field(
        default="",
        description=(
            "Python code that tests the connection using os.environ "
            "vars DS_FIELDNAME (uppercase field name with DS_ prefix) "
            "and prints 'ok' on success. Empty string if untestable."
        ),
    )
    fields: list[_CustomDatasourceField] = Field(
        default_factory=list,
        description=(
            "Credential fields the user will need to provide. List in "
            "the order they should be prompted."
        ),
    )


async def collect_custom_credentials(
    console: "Console",
    session: "ChatSession",
    registry: "DatasourceRegistry",
    display_name: str,
    fields: list[DatasourceField],
    credentials: dict[str, str],
    *,
    edit_existing: bool = False,
) -> bool:
    """Collect or edit custom datasource credentials sequentially."""
    fields_by_name = {f.name: f for f in fields}
    slug_hint = re.sub(r"[^\w]", "_", display_name.lower()).strip("_")
    known_engine_slugs = [e.engine for e in registry.all_engines()]
    skipped: set[str] = set()

    def _label_for(f: DatasourceField) -> str:
        return (f.description or "").strip() or f.name

    def _looks_structured(text: str) -> bool:
        return "=" in text or "://" in text or "\n" in text

    if edit_existing:
        for current in fields:
            while True:
                current_value = credentials.get(current.name, "")
                label = f"(anton) {_label_for(current)}"
                if current.secret and current_value:
                    label = f"{label} [••••••••]"
                    value = await prompt_or_cancel(label, password=True)
                elif current_value:
                    value = await prompt_or_cancel(label, default=current_value)
                else:
                    value = await prompt_or_cancel(label, password=current.secret)

                if value is None:
                    return False

                stripped = value.strip()
                if not stripped:
                    break

                lowered = stripped.lower()
                if lowered == "skip":
                    return True
                if lowered == "help":
                    await show_credential_help(
                        console, session, display_name, current, fields,
                    )
                    continue

                credentials[current.name] = stripped
                break
        return True

    while True:
        remaining = [
            f for f in fields
            if f.name not in credentials and f.name not in skipped
        ]
        if not remaining:
            break
        current = remaining[0]
        value = await prompt_or_cancel(
            f"(anton) {_label_for(current)}",
            password=current.secret,
        )
        if value is None:
            return False
        stripped = value.strip()
        if not stripped:
            skipped.add(current.name)
            continue
        lowered = stripped.lower()
        if lowered == "skip":
            break
        if lowered == "help":
            await show_credential_help(
                console, session, display_name, current, fields,
            )
            continue
        if len(remaining) > 1 and _looks_structured(stripped):
            extracted = await extract_variables(
                stripped,
                expected_fields=fields,
                current_engine=slug_hint,
                current_engine_display=display_name,
                known_engine_slugs=known_engine_slugs,
                session=session,
            )
            if extracted.variables:
                filled_here: list[str] = []
                for k, val in extracted.variables.items():
                    if k in fields_by_name and val:
                        credentials[k] = val
                        filled_here.append(k)
                if filled_here:
                    console.print(
                        f"[anton.muted]        Got: {', '.join(filled_here)}[/]"
                    )
                    continue
        credentials[current.name] = stripped

    return True


async def handle_add_custom_datasource(
    console: "Console",
    name: str,
    registry: "DatasourceRegistry",
    session: "ChatSession",
    *,
    known_service: bool = False,
):
    """Ask for the tool name, use the LLM to identify required fields, then collect credentials."""

    console.print()
    if name:
        tool_name = name
    else:
        tool_name = await prompt_or_cancel(
            "(anton) Name of the tool or service?",
        )
        if not tool_name or not tool_name.strip():
            return None
        tool_name = tool_name.strip()

    if tool_name and not tool_name.isdigit():
        console.print()
        console.print(
            f"[anton.cyan](anton)[/] Setting up [bold]{tool_name}[/] as a custom datasource."
        )

    if known_service:
        user_answer = ""
        console.print("[anton.muted]        Figuring out the connection…[/]")
    else:
        user_answer = await prompt_or_cancel(
            f"(anton) How does {tool_name} authenticate? (short description, no secrets)",
        )
        if user_answer is None:
            return None
        console.print("[anton.muted]        Figuring out the connection…[/]")

    llm_prompt = f"The user wants to connect to {repr(tool_name)}."
    if user_answer:
        llm_prompt += f" They said: {user_answer}"
    else:
        llm_prompt += " Determine the standard authentication fields for this service."
    llm_prompt += (
        "\n\nReturn the connection spec following the schema you've been given. "
        "For test_snippet, write Python that uses os.environ['DS_<FIELDNAME>'] "
        "vars (uppercase, DS_ prefix) and prints 'ok' on success."
        "\n\nIMPORTANT: Every field in the generated spec must have required=False. "
        "Never set required=True for any field in a custom datasource. "
        "All fields are optional by design — the user will provide whatever "
        "they have and Anton will ask for more only if needed."
    )

    try:
        spec: _CustomDatasourceSpec = await session._llm.generate_object(
            _CustomDatasourceSpec,
            system="You are a data source connection expert.",
            messages=[{"role": "user", "content": llm_prompt}],
            max_tokens=1024,
        )
    except Exception:
        console.print(
            "[anton.warning]        Couldn't identify connection details. Try again.[/]"
        )
        console.print()
        return None

    test_snippet = spec.test_snippet.strip()
    fields: list[DatasourceField] = []
    for f in spec.fields:
        if not f.name:
            continue
        fields.append(
            DatasourceField(
                name=f.name,
                required=False,
                secret=f.secret,
                description=f.description,
            )
        )

    if not fields:
        console.print("[anton.warning]    Couldn't identify any connection fields.[/]")
        console.print()
        return None

    display_name = spec.display_name or name
    pip_pkg = spec.pip

    # Collect inline values and show a compact one-line summary.
    credentials: dict[str, str] = {}
    summary_parts: list[str] = []
    for f, raw in zip(fields, spec.fields):
        inline_value = (raw.value or "").strip()
        if inline_value:
            credentials[f.name] = inline_value
        if f.secret and inline_value:
            summary_parts.append(f"[bold]{f.name}[/] (secret, provided)")
        elif f.secret:
            summary_parts.append(f"[bold]{f.name}[/] (secret)")
        elif inline_value:
            summary_parts.append(f"[bold]{f.name}[/]={inline_value}")
        else:
            summary_parts.append(f"[bold]{f.name}[/]")
    console.print()
    console.print("        [anton.muted]Fields:[/] " + ", ".join(summary_parts))
    console.print()

    # Sequential collection — one field at a time, consistent with the
    # built-in connect flow. Enter skips the current field; 'skip' stops
    # collection; 'help' shows credential guidance; structured input with
    # multiple values is parsed via LLM extraction.
    if not await collect_custom_credentials(
        console,
        session,
        registry,
        display_name,
        fields,
        credentials,
    ):
        return None

    # Custom datasources never have required fields.
    for f in fields:
        f.required = False

    engine_def = persist_custom_engine(
        registry,
        display_name,
        fields,
        test_snippet=test_snippet,
        pip=pip_pkg,
    )
    if engine_def is None:
        console.print(
            "[anton.warning]Could not validate engine definition — "
            "credentials saved but engine not written to datasources.md.[/]"
        )
        engine_def = DatasourceEngine(
            engine=re.sub(r"[^\w]", "_", display_name.lower()).strip("_"),
            display_name=display_name,
            pip=pip_pkg,
            fields=fields,
            test_snippet=test_snippet,
            custom=True,
        )

    return engine_def, credentials
