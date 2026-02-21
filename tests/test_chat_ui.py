from __future__ import annotations

from unittest.mock import MagicMock, patch

from anton.chat_ui import (
    PHASE_LABELS,
    THINKING_MESSAGES,
    TOOL_MESSAGES,
    StreamDisplay,
    _TOOL_LABELS,
    _Step,
)


class TestMessageLists:
    def test_thinking_messages_non_empty(self):
        assert len(THINKING_MESSAGES) > 0

    def test_tool_messages_non_empty(self):
        assert len(TOOL_MESSAGES) > 0


class TestToolLabels:
    def test_tool_labels_has_execute_task(self):
        assert "execute_task" in _TOOL_LABELS

    def test_tool_labels_has_scratchpad(self):
        assert "scratchpad" in _TOOL_LABELS

    def test_tool_labels_has_minds(self):
        assert "minds" in _TOOL_LABELS


class TestStep:
    def test_step_defaults(self):
        step = _Step(label="Thinking...")
        assert step.label == "Thinking..."
        assert step.done is False

    def test_step_done(self):
        step = _Step(label="Done", done=True)
        assert step.done is True


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
        # Should have one initial thinking step
        assert len(display._steps) == 1
        assert not display._steps[0].done

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
    def test_append_text_marks_steps_done(self, MockLive):
        display, console = self._make_display()
        display.start()

        display.append_text("Hello")

        # All steps should be marked done on first text
        for step in display._steps:
            assert step.done

    @patch("anton.chat_ui.Live")
    def test_show_tool_execution_adds_step(self, MockLive):
        display, console = self._make_display()
        display.start()

        display.show_tool_execution("scratchpad")

        assert len(display._steps) == 2
        assert display._steps[0].done  # thinking step marked done
        assert not display._steps[1].done  # new tool step active
        assert display._steps[1].label == "Running scratchpad"

    @patch("anton.chat_ui.Live")
    def test_show_tool_execution_unknown_tool(self, MockLive):
        display, console = self._make_display()
        display.start()

        display.show_tool_execution("unknown_tool")

        assert len(display._steps) == 2
        assert display._steps[1].label == "unknown_tool"

    @patch("anton.chat_ui.Live")
    def test_show_tool_execution_with_description(self, MockLive):
        display, console = self._make_display()
        display.start()

        # First call without description adds the step
        display.show_tool_execution("scratchpad")
        assert len(display._steps) == 2
        assert display._steps[1].label == "Running scratchpad"

        # Second call WITH description updates the active step's label
        display.show_tool_execution("scratchpad", "installing pandas")
        assert len(display._steps) == 2  # no new step added
        assert display._steps[1].label == "Running scratchpad — installing pandas"

    @patch("anton.chat_ui.Live")
    def test_show_tool_execution_collapse_repeated(self, MockLive):
        display, console = self._make_display()
        display.start()

        # First scratchpad call
        display.show_tool_execution("scratchpad")
        display.show_tool_execution("scratchpad", "installing pandas")

        # Second scratchpad call — marks previous done, adds new
        display.show_tool_execution("scratchpad")
        display.show_tool_execution("scratchpad", "fetching data")

        assert len(display._steps) == 3  # thinking + 2 scratchpad steps
        assert display._steps[1].done  # first scratchpad marked done
        assert display._steps[1].label == "Running scratchpad — installing pandas"
        assert not display._steps[2].done  # second scratchpad active
        assert display._steps[2].label == "Running scratchpad — fetching data"

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
    def test_finish_prints_steps_when_multiple(self, MockLive):
        display, console = self._make_display()
        display.start()

        display.show_tool_execution("scratchpad")
        display.append_text("result")
        display.finish(input_tokens=10, output_tokens=20, elapsed=1.0, ttft=0.1)

        # Should print step lines + blank + response + stats + spacing
        assert console.print.call_count >= 4

    @patch("anton.chat_ui.Live")
    def test_finish_skips_steps_when_single(self, MockLive):
        display, console = self._make_display()
        display.start()

        display.append_text("quick reply")
        display.finish(input_tokens=10, output_tokens=20, elapsed=1.0, ttft=0.1)

        # Only 1 step (thinking) — step list should NOT be printed
        # Should print: response prefix, response, stats, 2x spacing
        calls = [str(c) for c in console.print.call_args_list]
        # No step checkmark lines in output
        step_lines = [c for c in calls if "✓" in c]
        assert len(step_lines) == 0

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

    @patch("anton.chat_ui.Live")
    def test_show_tool_result_marks_steps_done(self, MockLive):
        display, console = self._make_display()
        display.start()

        display.show_tool_result("some result")

        for step in display._steps:
            assert step.done
        assert "some result" in display._buffer
