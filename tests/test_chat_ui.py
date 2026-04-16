from __future__ import annotations

from unittest.mock import MagicMock, patch

from anton.chat_ui import PHASE_LABELS, StreamDisplay, _MAX_DESC, _tool_display_text



class TestStreamDisplay:
    def _make_display(self):
        console = MagicMock()
        toolbar = {"stats": "", "status": ""}
        return StreamDisplay(console, toolbar=toolbar), console

    @patch("anton.chat_ui.Live")
    def test_start_creates_live(self, MockLive):
        display, console = self._make_display()
        display.start()
        MockLive.assert_called_once()
        MockLive.return_value.start.assert_called_once()

    @patch("anton.chat_ui.Live")
    def test_append_text_updates_buffer(self, MockLive):
        display, console = self._make_display()
        display.start()
        live = MockLive.return_value

        display.append_text("Hello ")
        display.append_text("world!")

        # Before any tool use, text goes to _initial_text
        assert display._initial_text == "Hello world!"
        assert live.update.call_count == 2

    @patch("anton.chat_ui.Live")
    def test_finish_stops_live_and_prints(self, MockLive):
        display, console = self._make_display()
        display.start()
        live = MockLive.return_value

        display.append_text("test output")
        display.finish()

        live.stop.assert_called_once()
        # Should print the response and stats
        assert console.print.call_count >= 2

    @patch("anton.chat_ui.Live")
    def test_abort_stops_live_cleanly(self, MockLive):
        display, console = self._make_display()
        display.start()
        live = MockLive.return_value

        display.abort()

        live.stop.assert_called_once()
        # abort should NOT print anything
        console.print.assert_not_called()

    @patch("anton.chat_ui.Live")
    def test_update_progress_updates_spinner(self, MockLive):
        display, console = self._make_display()
        display.start()
        live = MockLive.return_value

        display.update_progress("executing", "Step 1/3: read file", eta=10.0)

        # Should have been called: once for start (initial spinner), once for update_progress
        assert live.update.call_count >= 1

    @patch("anton.chat_ui.Live")
    def test_update_progress_without_eta(self, MockLive):
        display, console = self._make_display()
        display.start()
        live = MockLive.return_value

        display.update_progress("planning", "Analyzing task...")

        assert live.update.call_count >= 1

    def test_phase_labels_cover_all_phases(self):
        expected = {"memory_recall", "planning", "executing", "complete", "failed", "scratchpad"}
        assert expected == set(PHASE_LABELS.keys())


class TestActivityTracking:
    def _make_display(self):
        console = MagicMock()
        toolbar = {"stats": "", "status": ""}
        return StreamDisplay(console, toolbar=toolbar), console

    @patch("anton.chat_ui.Live")
    def test_tool_use_creates_activity(self, MockLive):
        display, _ = self._make_display()
        display.start()

        display.on_tool_use_start("tool_1", "scratchpad")

        assert len(display._activities) == 1
        assert display._activities[0].tool_id == "tool_1"
        assert display._activities[0].name == "scratchpad"

    @patch("anton.chat_ui.Live")
    def test_json_delta_accumulation(self, MockLive):
        display, _ = self._make_display()
        display.start()

        display.on_tool_use_start("tool_1", "scratchpad")
        display.on_tool_use_delta("tool_1", '{"action":')
        display.on_tool_use_delta("tool_1", ' "exec", "name": "main"}')
        display.on_tool_use_end("tool_1")

        act = display._activities[0]
        assert act.description == "exec"  # no Scratchpad() wrapper

    @patch("anton.chat_ui.Live")
    def test_finish_prints_activity_summary(self, MockLive):
        display, console = self._make_display()
        display.start()

        # Initial text before tools
        display.append_text("Let me check...")

        display.on_tool_use_start("tool_1", "scratchpad")
        display.on_tool_use_delta("tool_1", '{"action": "exec", "name": "pad"}')
        display.on_tool_use_end("tool_1")

        # Answer text after tools
        display.append_text("Here's what I found...")
        display.finish()

        # finish should print: muted initial, activity tree, anton> + answer markdown, trailing newline
        assert console.print.call_count >= 4

    @patch("anton.chat_ui.Live")
    def test_no_activities_no_tree(self, MockLive):
        display, console = self._make_display()
        display.start()

        display.append_text("Just text, no tools")
        display.finish()

        # Should print: anton> prefix, markdown, trailing newline — but no activity tree
        # The first print should NOT be a Text with tool labels
        calls = console.print.call_args_list
        # With no activities, the first call is the "anton> " prefix
        from rich.text import Text as RichText
        first_arg = calls[0][0][0] if calls[0][0] else None
        assert isinstance(first_arg, RichText)
        assert "anton>" in first_arg.plain

    @patch("anton.chat_ui.Live")
    def test_multiple_tool_calls(self, MockLive):
        display, _ = self._make_display()
        display.start()

        display.on_tool_use_start("tool_1", "scratchpad")
        display.on_tool_use_delta("tool_1", '{"action": "exec", "name": "pad"}')
        display.on_tool_use_end("tool_1")

        display.on_tool_use_start("tool_2", "memorize")
        display.on_tool_use_delta("tool_2", '{"entries": [{"text": "test", "kind": "lesson", "scope": "project"}]}')
        display.on_tool_use_end("tool_2")

        assert len(display._activities) == 2
        # Scratchpad now shows just the description (no wrapper)
        assert display._activities[0].description == "exec"
        # Memorize now shows a witty phrase (random, so just check it's a string)
        assert display._activities[1].description  # non-empty

    def test_malformed_json_fallback(self):
        # Bad JSON should not crash — falls back to a default
        result = _tool_display_text("scratchpad", "{broken json")
        assert result == "Running code"

    def test_tool_display_text_truncation(self):
        long_desc = "a" * 100
        result = _tool_display_text("scratchpad", f'{{"one_line_description": "{long_desc}"}}')
        # No wrapper — just the truncated description
        assert len(result) <= _MAX_DESC
        assert result.endswith("\u2026")

    def test_tool_display_text_unknown_tool(self):
        result = _tool_display_text("some_new_tool", '{"foo": "bar"}')
        # Unknown tools get a generic phrase from _GENERIC_TOOL_PHRASES
        assert isinstance(result, str)
        assert len(result) > 0

    def test_scratchpad_display_uses_one_line_description(self):
        """one_line_description should be used directly (no Scratchpad() wrapper)."""
        result = _tool_display_text(
            "scratchpad",
            '{"action": "exec", "name": "pad", "one_line_description": "Install packages"}',
        )
        assert result == "Install packages"

    def test_scratchpad_display_falls_back_to_action(self):
        """Without one_line_description, scratchpad shows the action."""
        result = _tool_display_text(
            "scratchpad",
            '{"action": "exec", "name": "pad"}',
        )
        assert result == "exec"

    @patch("anton.chat_ui.Live")
    def test_text_routes_to_initial_before_tools(self, MockLive):
        display, _ = self._make_display()
        display.start()

        display.append_text("Let me check...")
        assert display._initial_text == "Let me check..."
        assert display._buffer == ""
        assert not display._in_tool_phase

    @patch("anton.chat_ui.Live")
    def test_text_routes_to_buffer_after_tools(self, MockLive):
        display, _ = self._make_display()
        display.start()

        display.append_text("Initial text")
        display.on_tool_use_start("tool_1", "scratchpad")
        display.append_text("Answer text")

        assert display._initial_text == "Initial text"
        assert display._buffer == "Answer text"
        assert display._in_tool_phase
