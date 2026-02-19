from __future__ import annotations

from unittest.mock import MagicMock, patch

from anton.chat_ui import PHASE_LABELS, THINKING_MESSAGES, TOOL_MESSAGES, StreamDisplay


class TestMessageLists:
    def test_thinking_messages_non_empty(self):
        assert len(THINKING_MESSAGES) > 0

    def test_tool_messages_non_empty(self):
        assert len(TOOL_MESSAGES) > 0


class TestStreamDisplay:
    def _make_display(self):
        console = MagicMock()
        return StreamDisplay(console), console

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

        assert display._buffer == "Hello world!"
        assert live.update.call_count == 2

    @patch("anton.chat_ui.Live")
    def test_finish_stops_live_and_prints(self, MockLive):
        display, console = self._make_display()
        display.start()
        live = MockLive.return_value

        display.append_text("test output")
        display.finish(input_tokens=128, output_tokens=342, elapsed=2.3, ttft=0.48)

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
        expected = {"memory_recall", "planning", "skill_discovery", "skill_building", "executing", "complete", "failed"}
        assert expected == set(PHASE_LABELS.keys())
