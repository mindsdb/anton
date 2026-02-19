from __future__ import annotations

import re
from io import StringIO
from unittest.mock import MagicMock, patch

from rich.console import Console

from anton.channel.branding import ASCII_LOGO, TAGLINES, pick_tagline
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
    def test_taglines_non_empty(self):
        assert len(TAGLINES) >= 16

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
        assert "v0.1.0" in output


class TestRenderDashboard:
    def test_dashboard_contains_commands(self):
        from anton.channel.branding import render_dashboard

        mock_settings = MagicMock()
        mock_settings.skills_dir = "skills"
        mock_settings.user_skills_dir = "~/.anton/skills"
        mock_settings.memory_enabled = False
        mock_settings.coding_model = "claude-opus-4-6"

        mock_registry = MagicMock()
        mock_registry.list_all.return_value = []

        with patch("anton.config.settings.AntonSettings", return_value=mock_settings), \
             patch("anton.skill.registry.SkillRegistry", return_value=mock_registry):
            console = _make_console()
            render_dashboard(console)
            output = _strip_ansi(console.file.getvalue())
            assert "Commands" in output
            assert "Status" in output
