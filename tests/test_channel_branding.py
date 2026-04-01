from __future__ import annotations

import re
from io import StringIO
from unittest.mock import MagicMock, patch

from rich.console import Console

from anton.channel.branding import TAGLINES, pick_tagline
from anton.channel.theme import build_rich_theme


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _make_console() -> Console:
    return Console(
        file=StringIO(),
        theme=build_rich_theme("dark"),
        force_terminal=True,
        width=80,
    )


class TestTaglines:
    def test_pick_tagline_returns_from_list(self):
        tagline = pick_tagline()
        assert tagline in TAGLINES

    def test_pick_tagline_deterministic_with_seed(self):
        a = pick_tagline(seed=42)
        b = pick_tagline(seed=42)
        assert a == b


class TestRenderBanner:
    def test_banner_contains_version(self):
        from anton.channel.branding import render_banner

        console = _make_console()
        render_banner(console)
        output = _strip_ansi(console.file.getvalue())
        from anton import __version__
        assert f"v{__version__}" in output

    def test_banner_contains_robot(self):
        from anton.channel.branding import render_banner

        console = _make_console()
        render_banner(console)
        output = _strip_ansi(console.file.getvalue())
        assert "(\u00b0\u1d17\u00b0)" in output  # robot face

    def test_banner_contains_block_name(self):
        from anton.channel.branding import render_banner

        console = _make_console()
        render_banner(console)
        output = _strip_ansi(console.file.getvalue())
        # The block-letter ANTON name (top row)
        assert "\u2584\u2580\u2588" in output


class TestRenderDashboard:
    def test_dashboard_contains_commands(self):
        from anton.channel.branding import render_dashboard

        mock_settings = MagicMock()
        mock_settings.memory_enabled = False
        mock_settings.coding_model = "claude-opus-4-6"

        with patch("anton.config.settings.AntonSettings", return_value=mock_settings):
            console = _make_console()
            render_dashboard(console)
            output = _strip_ansi(console.file.getvalue())
            assert "Commands" in output
            assert "Status" in output
