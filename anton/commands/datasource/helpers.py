"""Shared helpers for datasource commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markdown import Markdown
from rich.padding import Padding

if TYPE_CHECKING:
    from rich.console import Console
    from anton.chat import ChatSession


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
