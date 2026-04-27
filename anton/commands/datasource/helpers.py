"""Shared helpers for datasource commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markdown import Markdown
from rich.padding import Padding

from anton.utils.prompt import prompt_or_cancel

if TYPE_CHECKING:
    from rich.console import Console
    from anton.chat import ChatSession
    from anton.core.datasources.datasource_registry import DatasourceField


async def prompt_field_value(
    field: "DatasourceField",
    credentials: dict[str, str],
) -> bool:
    """Prompt for one datasource field, respecting secret/required/default.

    Shared by the `/edit` flow and the failed-test retry flow. Updates
    ``credentials`` in place. Returns ``False`` if the user cancelled
    (Esc), ``True`` otherwise. An empty response keeps the current value
    when one exists, otherwise leaves the field unset.
    """
    current = credentials.get(field.name, "")
    field_label = f"(anton) {field.name}"
    if not field.required:
        field_label += " (optional)"

    if field.secret:
        masked = "••••••••" if current else ""
        label = f"{field_label} [{masked}]" if masked else field_label
        value = await prompt_or_cancel(label, password=True)
        if value is None:
            return False
        if value:
            credentials[field.name] = value
        return True

    if current:
        value = await prompt_or_cancel(field_label, default=current)
        if value is None:
            return False
        credentials[field.name] = value if value else current
        return True

    if field.default:
        value = await prompt_or_cancel(field_label, default=field.default)
        if value is None:
            return False
        if value:
            credentials[field.name] = value
        return True

    value = await prompt_or_cancel(field_label)
    if value is None:
        return False
    if value:
        credentials[field.name] = value
    return True


async def show_credential_help(
    console: "Console",
    session: "ChatSession",
    service_name: str,
    current_field,
    all_fields: list,
) -> None:
    """Use the LLM to explain how to obtain credentials."""
    field_descriptions = ", ".join(
        f"{f.name} ({f.description})" for f in all_fields
    )
    storage_note = (
        "The credentials will be stored securely in Anton's Local Vault — "
        "do NOT suggest storage tips, password managers, or safe-keeping advice."
    )
    if current_field is not None:
        prompt = (
            f"I'm connecting to {service_name} and need to provide: {field_descriptions}\n\n"
            f"I need help with the '{current_field.name}' field"
            f" ({current_field.description}).\n\n"
            "Give me a brief step-by-step guide on where and how to get this credential. "
            f"Be concise — numbered steps, no fluff. {storage_note}"
        )
        heading = f"[anton.cyan](anton)[/] How to get [bold]{current_field.name}[/]:"
    else:
        prompt = (
            f"I'm connecting to {service_name} and need these credentials: {field_descriptions}\n\n"
            "Give me a brief step-by-step guide on where and how to obtain each of these. "
            f"Be concise — numbered steps, no fluff. {storage_note}"
        )
        heading = f"[anton.cyan](anton)[/] How to get credentials for [bold]{service_name}[/]:"

    console.print()
    console.print("[anton.muted]        Looking up instructions…[/]")

    try:
        resp = await session._llm.plan(
            system="You are a helpful assistant that guides users through obtaining credentials for services.",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=512,
        )
        help_text = (resp.content or "").strip()
    except Exception:
        help_text = "Sorry, couldn't fetch help right now. Try checking the service's documentation."

    console.print()
    console.print(heading)
    console.print()
    console.print(Padding(Markdown(help_text), (0, 0, 0, 8)))
    console.print()
