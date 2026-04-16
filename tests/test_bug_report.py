"""Tests for bug report functionality."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from anton.commands.bug_report import handle_report_bug
from anton.config.settings import AntonSettings
from anton.diagnostics import collect_diagnostics, save_diagnostics_file


class TestDiagnostics:
    """Test diagnostic collection functionality."""

    def test_collect_diagnostics_basic(self):
        """Test basic diagnostic collection without optional components."""
        settings = AntonSettings()
        diagnostics = collect_diagnostics(settings)

        # Check required fields
        assert "timestamp" in diagnostics
        assert "anton_version" in diagnostics
        assert "system_info" in diagnostics
        assert "packages" in diagnostics
        assert "config" in diagnostics
        assert "datasources" in diagnostics
        assert "workspace" in diagnostics
        assert "memory" in diagnostics
        assert "conversation" in diagnostics
        assert "recent_logs" in diagnostics

        # Check system info
        sys_info = diagnostics["system_info"]
        assert "platform" in sys_info
        assert "python_version" in sys_info
        assert "os_name" in sys_info

        # Check config sanitization
        config = diagnostics["config"]
        assert (
            "anthropic_api_key" not in config
            or config["anthropic_api_key"] == "***REDACTED***"
        )
        assert (
            "openai_api_key" not in config
            or config["openai_api_key"] == "***REDACTED***"
        )
        assert (
            "minds_api_key" not in config or config["minds_api_key"] == "***REDACTED***"
        )

    def test_collect_diagnostics_with_session(self):
        """Test diagnostic collection with an active session."""
        settings = AntonSettings()

        # Mock session
        session = MagicMock()
        session._history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
        ]
        session._turn_count = 2

        diagnostics = collect_diagnostics(settings, session=session)

        assert diagnostics["conversation"] is not None
        assert diagnostics["conversation"]["turn_count"] == 2
        assert diagnostics["conversation"]["history_length"] == 4
        assert len(diagnostics["conversation"]["history_sample"]) == 4

    def test_collect_diagnostics_with_workspace(self):
        """Test diagnostic collection with workspace info."""
        settings = AntonSettings()

        # Mock workspace
        workspace = MagicMock()
        workspace.base = Path("/test/workspace")
        workspace.name = "test-workspace"
        workspace.git_root = Path("/test/workspace/.git")

        diagnostics = collect_diagnostics(settings, workspace=workspace)

        assert diagnostics["workspace"] is not None
        assert diagnostics["workspace"]["base"] == "/test/workspace"
        assert diagnostics["workspace"]["name"] == "test-workspace"
        assert diagnostics["workspace"]["has_git"] is True

    def test_collect_diagnostics_with_cortex(self):
        """Test diagnostic collection with memory/cortex info."""
        settings = AntonSettings(memory_mode="autopilot")

        # Mock cortex
        cortex = MagicMock()
        cortex.enabled = True
        cortex._episodic = MagicMock()
        cortex._episodic.enabled = True

        diagnostics = collect_diagnostics(settings, cortex=cortex)

        assert diagnostics["memory"]["enabled"] is True
        assert diagnostics["memory"]["mode"] == "autopilot"
        assert diagnostics["memory"]["episodic_enabled"] is True

    def test_save_diagnostics_file(self, tmp_path):
        """Test saving diagnostics to file."""
        diagnostics = {
            "timestamp": "2024-01-01T00:00:00",
            "anton_version": "1.0.0",
            "system_info": {"platform": "test"},
        }

        output_file = save_diagnostics_file(diagnostics, tmp_path)

        assert output_file.exists()
        assert output_file.name.startswith("bug_report_")
        assert output_file.suffix == ".json"

        # Verify content
        with open(output_file) as f:
            saved_data = json.load(f)
        assert saved_data == diagnostics


class TestBugReportCommand:
    """Test bug report command handler."""

    @pytest.mark.asyncio
    async def test_handle_report_bug_cancelled(self):
        """Test bug report cancelled by user."""
        console = Console()
        settings = AntonSettings()

        with patch("anton.commands.bug_report.prompt_or_cancel", return_value="n"):
            await handle_report_bug(console, settings, None, None, None)
            # Should return early without error

    @pytest.mark.asyncio
    async def test_handle_report_bug_with_description(self):
        """Test bug report with user description."""
        console = Console()
        settings = AntonSettings(minds_api_key="test-key")

        # Mock the prompts
        prompt_responses = ["y", "y", "This is a test bug description"]
        with patch(
            "anton.commands.bug_report.prompt_or_cancel", side_effect=prompt_responses
        ):
            with patch("anton.commands.bug_report.collect_diagnostics") as mock_collect:
                with patch(
                    "anton.commands.bug_report.save_diagnostics_file"
                ) as mock_save:
                    with patch(
                        "anton.commands.bug_report.publish_bug_report"
                    ) as mock_publish:
                        mock_collect.return_value = {"test": "data"}
                        mock_save.return_value = Path("/test/bug_report.json")

                        await handle_report_bug(console, settings, None, None, None)

                        # Verify diagnostics were collected
                        mock_collect.assert_called_once()

                        # Verify description was added
                        saved_diagnostics = mock_save.call_args[0][0]
                        assert (
                            saved_diagnostics["user_description"]
                            == "This is a test bug description"
                        )

                        # Verify publish was called
                        mock_publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_report_bug_no_api_key(self):
        """Test bug report when API key needs to be entered."""
        console = Console()
        settings = AntonSettings(minds_api_key=None)  # Explicitly set no API key

        # Mock the prompts: consent, bug description, has API key, API key
        prompt_responses = ["y", "n", "n", "test-api-key"]
        with patch(
            "anton.commands.bug_report.prompt_or_cancel", side_effect=prompt_responses
        ) as mock_prompt:
            with patch("anton.commands.bug_report.collect_diagnostics") as mock_collect:
                with patch(
                    "anton.commands.bug_report.save_diagnostics_file"
                ) as mock_save:
                    with patch(
                        "anton.commands.bug_report.publish_bug_report"
                    ) as mock_publish:
                        with patch(
                            "anton.commands.bug_report.webbrowser.open"
                        ) as mock_browser:
                            mock_collect.return_value = {"test": "data"}
                            mock_save.return_value = Path("/test/bug_report.json")

                            await handle_report_bug(console, settings, None, None, None)

                            # Debug: Check how many times prompt was called
                            assert mock_prompt.call_count == 4, (
                                f"Expected 4 prompts, got {mock_prompt.call_count}"
                            )

                            # Verify browser was opened for registration
                            mock_browser.assert_called_once()

                            # Verify API key was set
                            assert settings.minds_api_key == "test-api-key"

                            # Verify publish was called
                            mock_publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_report_bug_collection_error(self):
        """Test bug report when diagnostic collection fails."""
        console = Console()
        settings = AntonSettings()

        with patch("anton.commands.bug_report.prompt_or_cancel", return_value="y"):
            with patch(
                "anton.commands.bug_report.collect_diagnostics",
                side_effect=Exception("Test error"),
            ):
                await handle_report_bug(console, settings, None, None, None)
                # Should handle error gracefully

    @pytest.mark.asyncio
    async def test_handle_report_bug_publish_error(self):
        """Test bug report when publishing fails."""
        console = Console()
        settings = AntonSettings(minds_api_key="test-key")

        prompt_responses = ["y", "n"]
        with patch(
            "anton.commands.bug_report.prompt_or_cancel", side_effect=prompt_responses
        ):
            with patch("anton.commands.bug_report.collect_diagnostics") as mock_collect:
                with patch(
                    "anton.commands.bug_report.save_diagnostics_file"
                ) as mock_save:
                    with patch(
                        "anton.commands.bug_report.publish_bug_report",
                        side_effect=Exception("Network error"),
                    ):
                        mock_collect.return_value = {"test": "data"}
                        mock_save.return_value = Path("/test/bug_report.json")

                        await handle_report_bug(console, settings, None, None, None)
                        # Should handle error gracefully
