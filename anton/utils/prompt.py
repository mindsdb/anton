"""Interactive prompt helpers for Anton's CLI commands."""

from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PTStyle

from rich.console import Console


MINDS_KEYS = {
    "ANTON_MINDS_API_KEY",
    "ANTON_MINDS_URL",
    "ANTON_MINDS_MIND_NAME",
    "ANTON_MINDS_DATASOURCE",
    "ANTON_MINDS_DATASOURCE_ENGINE",
    "ANTON_MINDS_SSL_VERIFY",
}

LLM_KEYS = {
    "ANTON_PLANNING_PROVIDER",
    "ANTON_CODING_PROVIDER",
    "ANTON_PLANNING_MODEL",
    "ANTON_CODING_MODEL",
    "ANTON_ANTHROPIC_API_KEY",
    "ANTON_OPENAI_API_KEY",
    "ANTON_OPENAI_BASE_URL",
}

SECRET_PATTERNS = ("KEY", "TOKEN", "SECRET", "PAT", "PASSWORD")


def mask_secret(value: str, *, keep: int = 4) -> str:
    if len(value) <= keep * 2:
        return "*" * max(len(value), 3)
    return f"{value[:keep]}...{value[-keep:]}"


def is_secret_key(key: str) -> bool:
    upper = key.upper()
    return any(p in upper for p in SECRET_PATTERNS)


def display_value(key: str, value: str) -> str:
    if is_secret_key(key) and value:
        return mask_secret(value)
    return value or "[dim]<empty>[/]"


async def prompt_or_cancel(
    label: str,
    *,
    default: str = "",
    password: bool = False,
    choices: list[str] | None = None,
    choices_display: str = "",
    allow_cancel: bool = True,
) -> str | None:
    """Prompt for free-text input; return None if the user presses Esc.

    Fully async via prompt_toolkit's prompt_async() — event loop never blocked.
    Only Esc is bound for cancellation; Ctrl+C propagates as KeyboardInterrupt.
    If `choices` is given, re-prompts until input matches or user presses Esc.
    If `choices_display` is given, uses it for the styled bracket text instead of
    joining `choices` (useful when the display text differs from strict validation).
    If `allow_cancel` is False, Esc is ignored and no footer is shown.
    """
    _esc = False
    bindings = KeyBindings()

    if allow_cancel:
        @bindings.add("escape")
        def _on_esc(event):
            nonlocal _esc
            _esc = True
            event.app.exit(result="")

    pt_style = PTStyle.from_dict({"bottom-toolbar": "noreverse nounderline bg:default"})

    def _toolbar():
        if not allow_cancel:
            return ""
        return HTML("<style fg='#ff69b4'>⏵⏵ Esc to cancel</style>")

    opts_text = choices_display or ("/".join(choices) if choices else "")

    if password:
        suffix = " (hidden):"
    elif opts_text and default:
        suffix = (
            f" <b><ansimagenta>[{opts_text}]</ansimagenta></b>"
            f" <b><ansicyan>({default})</ansicyan></b>:"
        )
    elif opts_text:
        suffix = f" <b><ansimagenta>[{opts_text}]</ansimagenta></b>:"
    elif default:
        suffix = f" <b><ansicyan>({default})</ansicyan></b>:"
    else:
        suffix = ":"

    pt_session: PromptSession[str] = PromptSession(
        mouse_support=False,
        bottom_toolbar=_toolbar,
        style=pt_style,
        key_bindings=bindings,
        is_password=password,
    )

    from anton.channel.theme import get_palette as _get_palette
    _prompt_color = _get_palette().prompt

    space = '\u2009'
    if label.startswith("(anton) "):
        body = label[len("(anton) "):]
        message = HTML(f"<b><style fg='{_prompt_color}'>(anton)</style></b> {body}{suffix}{space}")
    else:
        message = HTML(f"{label}{suffix}{space}")

    while True:
        _esc = False
        result = await pt_session.prompt_async(message)
        if _esc:
            return None
        val = result.strip() if result else default
        if choices is None or val in choices:
            break

    if not val and default:
        return default
    return val


async def prompt_minds_api_key(
    console: Console,
    *,
    current_key: str,
    allow_empty_keep: bool,
) -> str | None:
    prompt = "API key"
    if current_key:
        masked = mask_secret(current_key)
        if allow_empty_keep:
            prompt += f" (Enter to keep {masked})"
        else:
            prompt += f" (current: {masked}; Enter to cancel)"

    api_key = (await prompt_or_cancel(prompt, default="", password=True) or "").strip()
    if api_key:
        return api_key
    if current_key and allow_empty_keep:
        return current_key
    return None
