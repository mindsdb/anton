from __future__ import annotations

from unittest.mock import patch

from rich.theme import Theme

from anton.channel.theme import (
    DARK_PALETTE,
    LIGHT_PALETTE,
    build_rich_theme,
    detect_color_mode,
    get_palette,
)


class TestDetectColorMode:
    def test_anton_theme_env_dark(self):
        with patch.dict("os.environ", {"ANTON_THEME": "dark"}, clear=False):
            assert detect_color_mode() == "dark"

    def test_anton_theme_env_light(self):
        with patch.dict("os.environ", {"ANTON_THEME": "light"}, clear=False):
            assert detect_color_mode() == "light"

    def test_anton_theme_env_case_insensitive(self):
        with patch.dict("os.environ", {"ANTON_THEME": "LIGHT"}, clear=False):
            assert detect_color_mode() == "light"

    def test_colorfgbg_dark(self):
        env = {"COLORFGBG": "15;0"}
        with patch.dict("os.environ", env, clear=False):
            # Remove ANTON_THEME if present
            with patch.dict("os.environ", {"ANTON_THEME": ""}, clear=False):
                assert detect_color_mode() == "dark"

    def test_colorfgbg_light(self):
        env = {"COLORFGBG": "0;15"}
        with patch.dict("os.environ", env, clear=False):
            with patch.dict("os.environ", {"ANTON_THEME": ""}, clear=False):
                assert detect_color_mode() == "light"

    def test_default_is_dark(self):
        with patch.dict("os.environ", {"ANTON_THEME": "", "COLORFGBG": ""}, clear=False):
            with patch("anton.channel.theme.os.uname") as mock_uname:
                mock_uname.return_value.sysname = "Linux"
                assert detect_color_mode() == "dark"

    def test_macos_dark_mode(self):
        with patch.dict("os.environ", {"ANTON_THEME": "", "COLORFGBG": ""}, clear=False):
            with patch("anton.channel.theme.os.uname") as mock_uname:
                mock_uname.return_value.sysname = "Darwin"
                with patch("anton.channel.theme.subprocess.run") as mock_run:
                    mock_run.return_value.returncode = 0
                    assert detect_color_mode() == "dark"

    def test_macos_light_mode(self):
        with patch.dict("os.environ", {"ANTON_THEME": "", "COLORFGBG": ""}, clear=False):
            with patch("anton.channel.theme.os.uname") as mock_uname:
                mock_uname.return_value.sysname = "Darwin"
                with patch("anton.channel.theme.subprocess.run") as mock_run:
                    mock_run.return_value.returncode = 1
                    assert detect_color_mode() == "light"


class TestPalettes:
    def test_dark_palette_cyan(self):
        assert DARK_PALETTE.cyan == "#00FFFF"

    def test_light_palette_cyan(self):
        assert LIGHT_PALETTE.cyan == "#006B6B"

    def test_get_palette_dark(self):
        assert get_palette("dark") is DARK_PALETTE

    def test_get_palette_light(self):
        assert get_palette("light") is LIGHT_PALETTE


class TestBuildRichTheme:
    def test_returns_theme_object(self):
        theme = build_rich_theme("dark")
        assert isinstance(theme, Theme)

    def test_theme_has_anton_keys(self):
        theme = build_rich_theme("dark")
        assert "anton.cyan" in theme.styles
        assert "phase.planning" in theme.styles
        assert "phase.executing" in theme.styles
