"""Turning the service's page snapshot into what the model and Jev read."""

from __future__ import annotations

import json
import re

#: Jev picks a target from at most this many elements: a choice over hundreds
#: of options is past what one question should carry.
SHORTLIST_SIZE = 40
_WORD = re.compile(r"[a-z0-9]{3,}")


def element_line(el: dict) -> str:
    line = f"[{el['id']}] {el.get('role', '')} {json.dumps(el.get('name', ''), ensure_ascii=False)}"
    if el.get("value"):
        line += f" value={json.dumps(el['value'], ensure_ascii=False)}"
    if "checked" in el:
        line += " (checked)" if el["checked"] else " (unchecked)"
    if el.get("type") == "password":
        line += " (password)"
    if el.get("disabled"):
        line += " (disabled)"
    if el.get("offscreen"):
        line += " (offscreen)"
    if el.get("options"):
        line += f" options={json.dumps(el['options'], ensure_ascii=False)}"
    return line


def format_page(page: dict, *, max_text: int = 3000) -> str:
    """The plain-text page the model reads: one ``[17] button "Add to Cart"`` per line."""
    lines = [f"URL: {page.get('url', '')}", f"Title: {page.get('title', '')}", ""]
    lines += [element_line(el) for el in page.get("elements", [])]
    if page.get("truncated"):
        lines.append("… more elements not shown")
    text = (page.get("text") or "")[:max_text]
    if text:
        lines += ["", "Page text:", text]
    return "\n".join(lines)


def signature(page: dict) -> str:
    """Changes when the page does: URL plus every element's id, name and value."""
    parts = [page.get("url", "")]
    parts += [f"{el.get('id')}:{el.get('name')}:{el.get('value', '')}:{el.get('checked', '')}" for el in page.get("elements", [])]
    return "|".join(parts)


def shortlist(elements: list[dict], goal: str, inputs: dict | None = None, size: int = SHORTLIST_SIZE) -> list[dict]:
    """The elements most likely to matter for the goal, best first.

    Ranked by shared words with the goal and the input names, then on-screen
    before off-screen, then page order. Disabled elements are dropped: there
    is nothing to do with them.
    """
    words = set(_WORD.findall(goal.lower()))
    for key in inputs or {}:
        words |= set(_WORD.findall(str(key).lower()))

    def score(item: tuple[int, dict]) -> tuple:
        index, el = item
        name = f"{el.get('name', '')} {el.get('role', '')}".lower()
        overlap = len(words & set(_WORD.findall(name)))
        return (-overlap, bool(el.get("offscreen")), index)

    usable = [(i, el) for i, el in enumerate(elements) if not el.get("disabled")]
    return [el for _, el in sorted(usable, key=score)[:size]]
