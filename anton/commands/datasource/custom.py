"""Custom datasource creation (LLM-assisted)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from anton.core.datasources.datasource_registry import DatasourceEngine, DatasourceField
from anton.utils.datasources import remove_engine_block
from anton.utils.prompt import prompt_or_cancel
from anton.commands.datasource.helpers import show_credential_help

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
        default=True,
        description="True if the connection cannot be tested without this field.",
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
            "(anton) What is the name of the tool or service?",
        )
        if not tool_name or not tool_name.strip():
            return None
        tool_name = tool_name.strip()

    if known_service:
        user_answer = ""
        console.print("[anton.muted]        Working out the connection details…[/]")
    else:
        user_answer = await prompt_or_cancel(
            f"(anton) How do you authenticate with {tool_name}? "
            "Describe what credentials you have (don't paste actual values)",
        )
        if not user_answer or not user_answer.strip():
            return None
        console.print()
        console.print("[anton.muted]    Got it — working out the connection details…[/]")

    llm_prompt = f"The user wants to connect to {repr(tool_name)}."
    if user_answer:
        llm_prompt += f" They said: {user_answer}"
    else:
        llm_prompt += " Determine the standard authentication fields for this service."
    llm_prompt += (
        "\n\nReturn the connection spec following the schema you've been given. "
        "For test_snippet, write Python that uses os.environ['DS_<FIELDNAME>'] "
        "vars (uppercase, DS_ prefix) and prints 'ok' on success."
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
                required=f.required,
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

    # Show summary
    console.print()
    console.print("      [bold]── What I'll save ──────────────────────────[/]")
    credentials: dict[str, str] = {}
    for f, raw in zip(fields, spec.fields):
        inline_value = (raw.value or "").strip()
        if f.secret and inline_value:
            console.print(
                f"        • [bold]{f.name:<14}[/] (secret — provided, stored securely)"
            )
            credentials[f.name] = inline_value
        elif f.secret:
            console.print(
                f"        • [bold]{f.name:<14}[/] (secret — I'll ask for this)"
            )
        else:
            val_display = inline_value or "[anton.muted]<to be collected>[/]"
            console.print(f"        • [bold]{f.name:<14}[/] {val_display}")
            if inline_value:
                credentials[f.name] = inline_value
    console.print()

    help_answer = await prompt_or_cancel(
        "(anton) Do you need instructions on how to obtain these credentials?",
        choices=["y", "n"], default="n",
    )
    if help_answer is None:
        return None
    if help_answer.strip().lower() == "y":
        await show_credential_help(
            console, session, display_name, None, fields,
        )

    # Prompt for any secret fields not provided inline
    for f, raw in zip(fields, spec.fields):
        if not f.secret:
            continue
        if (raw.value or "").strip():
            continue
        value = await prompt_or_cancel(f"(anton) {f.name}", password=True)
        if value is None:
            return None
        if value:
            credentials[f.name] = value

    # Prompt for any required non-secret fields not provided inline
    for f, raw in zip(fields, spec.fields):
        if f.secret:
            continue
        if not f.required:
            continue
        if f.name in credentials:
            continue
        value = await prompt_or_cancel(f"(anton) {f.name}")
        if value is None:
            return None
        if value:
            credentials[f.name] = value

    # Offer to collect optional non-secret fields
    for f, raw in zip(fields, spec.fields):
        if f.secret or f.required or f.name in credentials:
            continue
        value = await prompt_or_cancel(f"(anton) {f.name} (optional — press Enter to skip)")
        if value is None:
            return None
        if value:
            credentials[f.name] = value

    if not credentials:
        console.print("[anton.warning]        No credentials collected. Aborting.[/]")
        console.print()
        return None

    # Build engine slug and write definition to ~/.anton/datasources.md
    slug = re.sub(r"[^\w]", "_", display_name.lower()).strip("_")
    field_lines = "\n".join(
        f"  - {{ name: {f.name}, required: {str(f.required).lower()}, "
        f'secret: {str(f.secret).lower()}, description: "{f.description}" }}'
        for f in fields
    )
    test_snippet_yaml = ""
    if test_snippet:
        indented = "\n".join(f"  {line}" for line in test_snippet.splitlines())
        test_snippet_yaml = f"test_snippet: |\n{indented}\n"

    yaml_block = (
        f"\n---\n\n## {display_name}\n"
        "```yaml\n"
        f"engine: {slug}\n"
        f"display_name: {display_name}\n"
        + (f"pip: {pip_pkg}\n" if pip_pkg else "")
        + f"fields:\n{field_lines}\n"
        + test_snippet_yaml
        + "```\n"
    )
    user_ds_path = Path("~/.anton/datasources.md").expanduser()
    tmp_path = user_ds_path.with_suffix(".tmp")

    existing = (
        user_ds_path.read_text(encoding="utf-8") if user_ds_path.is_file() else ""
    )
    existing = remove_engine_block(existing, slug)

    tmp_path.write_text(existing + yaml_block, encoding="utf-8")

    parsed = registry.validate_file(tmp_path)
    if slug in parsed:
        shutil.move(str(tmp_path), str(user_ds_path))
    else:
        tmp_path.unlink(missing_ok=True)
        console.print(
            "[anton.warning]Could not validate engine definition — "
            "credentials saved but engine not written to datasources.md.[/]"
        )

    registry.reload()
    engine_def = registry.get(slug)
    if engine_def is None:
        engine_def = DatasourceEngine(
            engine=slug,
            display_name=display_name,
            pip=pip_pkg,
            fields=fields,
            test_snippet=test_snippet,
        )

    missing_required = [f.name for f in fields if f.required and f.name not in credentials]
    if missing_required:
        console.print(
            "[anton.warning]    Cannot save — missing required fields: "
            f"{', '.join(missing_required)}. Aborting.[/]"
        )
        console.print()
        return None

    return engine_def, credentials
