"""The language the user's app is set to, as one line of the system prompt (ENG-4149).

A host that knows the user's UI language passes it as
``ChatSessionConfig.ui_locale`` (cowork-server reads it from the request's
``Accept-Language``). The session then adds :func:`ui_locale_line` to the
system prompt. The line does not force a language: the user's own message wins,
and the UI language only decides when the message does not say (a one-word
reply, a pasted table, a code snippet).

Only the tags in :data:`UI_LOCALE_NAMES` are accepted. Anything else, an
unsupported language or a malformed value, reads as no locale, so the prompt is
exactly what it was before this existed. Supporting another language is one
entry in that table.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Supported UI locale tags and the English name the prompt uses for each.
UI_LOCALE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "pt-BR": "Brazilian Portuguese",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh-CN": "Simplified Chinese",
}

_BY_LOWER = {tag.lower(): tag for tag in UI_LOCALE_NAMES}


def validated_ui_locale(value: object) -> str | None:
    """The supported tag ``value`` names, in its canonical case, else None.

    Hosts supply this, so it is untrusted. Matching ignores case and
    surrounding space (``"PT-br"`` reads as ``"pt-BR"``). Never raises: a bad
    locale must not fail a turn.
    """
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    tag = _BY_LOWER.get(cleaned.lower())
    if tag is None:
        # Bounded so a pathological header cannot flood the log.
        logger.warning("ignoring unsupported ui_locale %r (expected one of %s)", cleaned[:32], sorted(UI_LOCALE_NAMES))
    return tag


def ui_locale_line(ui_locale: str | None) -> str:
    """The system prompt's language line for ``ui_locale``, or "" for none."""
    tag = validated_ui_locale(ui_locale)
    if tag is None:
        return ""
    name = UI_LOCALE_NAMES[tag]
    return (
        f"## Language\n"
        f"The user's app is set to {name} ({tag}). Reply in the language the user "
        f"writes in; if that is unclear (e.g. a very short message, or only data or "
        f"code), reply in {name}. The same rule applies to everything the user reads, "
        f"including the short step descriptions you write for your tool calls."
    )


__all__ = ["UI_LOCALE_NAMES", "ui_locale_line", "validated_ui_locale"]
