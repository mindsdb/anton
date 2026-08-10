"""ENG-1092: current user turn carries its send time as a bracketed prefix,
so the live clock no longer sits in the (cached) system prompt."""

from __future__ import annotations

from datetime import datetime, timezone

from anton.core.session import _stamp_user_content

_NOW = datetime(2026, 7, 28, 13, 33)


def test_stamps_string_content():
    assert _stamp_user_content("да", _NOW) == "[2026-07-28 13:33] да"


def test_format_matches_cowork_server_history_stamp():
    # Must match anton_harness/harness.py `%Y-%m-%d %H:%M` so a message reads
    # identically live and when replayed from persisted history (cache-safe).
    stamped = _stamp_user_content("hello", _NOW)
    assert stamped.startswith("[2026-07-28 13:33] ")


def test_empty_string_passes_through():
    assert _stamp_user_content("", _NOW) == ""


def test_list_content_passes_through_unchanged():
    blocks = [{"type": "text", "text": "hi"}, {"type": "image", "source": {}}]
    assert _stamp_user_content(blocks, _NOW) == blocks


def test_utc_aware_datetime_renders_utc_wall_clock():
    # The call site passes datetime.now(timezone.utc); strftime must emit the
    # UTC wall-clock (no offset), matching cowork-server's UTC created_at stamp.
    # A local-time datetime would drift by the TZ offset — the ENG-1092 bug.
    dt = datetime(2026, 7, 31, 15, 14, tzinfo=timezone.utc)
    assert _stamp_user_content("x", dt) == "[2026-07-31 15:14] x"
