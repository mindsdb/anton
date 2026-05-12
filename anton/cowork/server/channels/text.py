"""Outbound text splitting honoring per-platform character limits.

Slack messages have a soft 3,000-char readability ceiling and a 40,000-char hard
cap; WhatsApp text is capped at 4,096; Telegram at 4,096; Discord at 2,000.
Without splitting, adapters silently truncate mid-response — splitting on
paragraph boundaries preserves Markdown structure when possible.

The algorithm matches nanoclaw's ``splitForLimit`` for cross-tool consistency.
"""
from __future__ import annotations


def split_for_limit(text: str, limit: int) -> list[str]:
    """Split ``text`` into chunks no larger than ``limit`` chars.

    Preference order: blank-line break, then newline, then space, then a hard
    char cut as a last resort. Each chunk has whitespace trimmed at the seam.

    A fenced code block that straddles a chunk boundary will render as two
    independent blocks on the receiver — same behavior as manually re-opening
    the fence; we don't try to balance fences here.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n\n", 0, limit)
        if cut <= 0:
            cut = remaining.rfind("\n", 0, limit)
        if cut <= 0:
            cut = remaining.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        chunk = remaining[:cut].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut:].lstrip()

    if remaining:
        chunks.append(remaining)
    return chunks
