from __future__ import annotations

import asyncio
import os
import threading

from unittest.mock import MagicMock, patch

from anton.chat_ui import (
    PHASE_LABELS,
    EscapeWatcher,
    QuestionRenderTracker,
    StreamDisplay,
    _MAX_DESC,
    _tool_display_text,
)



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
    def test_append_text_accumulates_in_pending(self, MockLive):
        display, console = self._make_display()
        display.start()
        live = MockLive.return_value

        display.append_text("Hello ")
        display.append_text("world!")

        # All streamed text accumulates in the single _pending buffer
        assert display._pending == "Hello world!"
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

    @patch("anton.chat_ui.Live")
    def test_stop_spinner_for_input_stops_the_live(self, MockLive):
        # CLIElicitor calls this right before prompt_toolkit takes the
        # terminal, so the spinner must already be gone by then — not just
        # queued to stop once the out-of-band event is drained.
        display, console = self._make_display()
        display.start()
        live = MockLive.return_value

        display.stop_spinner_for_input()

        live.stop.assert_called_once()
        assert display._live is None

    @patch("anton.chat_ui.Live")
    def test_reasoning_start_restarts_the_spinner_after_interactive_stopped_it(self, MockLive):
        """Live-testing feedback (ENG-969, generate_prd): `phase="interactive"`
        tears the Live context down (`_live = None`). A `reasoning_start`
        that follows it directly — with no tool-result line printed in
        between to implicitly restart the spinner, as generate_prd's phase 1
        (after ask_user) and phase 2 (after show_and_confirm) both do — must
        not silently no-op on a `None` Live."""
        display, console = self._make_display()
        display.start()

        display.update_progress("interactive", "")
        assert display._live is None

        display.update_progress("reasoning_start", "Thinking...")

        assert display._live is not None
        # `MockLive(...)` returns the same mock instance every call, so this
        # is `display.start()`'s initial `.start()` plus the restart's.
        assert MockLive.return_value.start.call_count == 2

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
        from rich.markdown import Markdown as RichMarkdown
        from rich.text import Text as RichText

        display, console = self._make_display()
        display.start()

        # Preamble before the tool — must flush dimmed AT on_tool_use_start
        display.append_text("Let me check...")
        display.on_tool_use_start("tool_1", "scratchpad")

        muted_before_finish = [
            c.args[0].plain
            for c in console.print.call_args_list
            if c.args
            and isinstance(c.args[0], RichText)
            and c.args[0].style == "anton.muted"
        ]
        assert muted_before_finish == ["Let me check..."]

        display.on_tool_use_delta("tool_1", '{"action": "exec", "name": "pad"}')
        display.on_tool_use_end("tool_1")

        # Answer text after the tool
        display.append_text("Here's what I found...")

        calls_before_finish = len(console.print.call_args_list)
        display.finish()
        finish_calls = console.print.call_args_list[calls_before_finish:]

        # finish() prints NO muted inner-speech (it was flushed earlier) …
        assert not [
            c
            for c in finish_calls
            if c.args
            and isinstance(c.args[0], RichText)
            and c.args[0].style == "anton.muted"
        ]
        # … and prints the final answer as a single Markdown block.
        markdowns = [
            c.args[0].markup
            for c in finish_calls
            if c.args and isinstance(c.args[0], RichMarkdown)
        ]
        assert markdowns == ["Here's what I found..."]

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
    def test_preamble_flushed_dimmed_at_tool_start(self, MockLive):
        from rich.text import Text as RichText

        display, console = self._make_display()
        display.start()

        display.append_text("Initial text")
        display.on_tool_use_start("tool_1", "scratchpad")

        # Preamble printed dimmed at the tool boundary, accumulator cleared
        muted = [
            c.args[0].plain
            for c in console.print.call_args_list
            if c.args
            and isinstance(c.args[0], RichText)
            and c.args[0].style == "anton.muted"
        ]
        assert muted == ["Initial text"]
        assert display._pending == ""

        # Subsequent text accumulates fresh
        display.append_text("Answer text")
        assert display._pending == "Answer text"

    @patch("anton.chat_ui.Live")
    def test_multiround_preambles_flushed_separately(self, MockLive):
        from rich.markdown import Markdown as RichMarkdown
        from rich.text import Text as RichText

        display, console = self._make_display()
        display.start()

        # Round 1: preamble → tool
        display.append_text("Now launching the backend:")
        display.on_tool_use_start("t1", "scratchpad")
        display.on_tool_use_delta("t1", '{"action": "exec", "name": "p"}')
        display.on_tool_use_end("t1")

        # Round 2: preamble → tool
        display.append_text("Launched! Checking the API:")
        display.on_tool_use_start("t2", "scratchpad")
        display.on_tool_use_delta("t2", '{"action": "exec", "name": "p"}')
        display.on_tool_use_end("t2")

        # Trailing text after the last tool = the real final answer
        display.append_text("Everything works.")
        display.finish()

        muted = [
            c.args[0].plain
            for c in console.print.call_args_list
            if c.args
            and isinstance(c.args[0], RichText)
            and c.args[0].style == "anton.muted"
        ]
        # Both preambles printed live, in order, each on its own line
        assert muted == [
            "Now launching the backend:",
            "Launched! Checking the API:",
        ]

        markdowns = [
            c.args[0].markup
            for c in console.print.call_args_list
            if c.args and isinstance(c.args[0], RichMarkdown)
        ]
        # Final answer is a single block — NOT concatenated with the preambles
        assert markdowns == ["Everything works."]

    @patch("anton.chat_ui.Live")
    def test_consecutive_tools_no_text_no_flush(self, MockLive):
        from rich.text import Text as RichText

        display, console = self._make_display()
        display.start()

        display.on_tool_use_start("t1", "scratchpad")
        display.on_tool_use_end("t1")
        display.on_tool_use_start("t2", "scratchpad")
        display.on_tool_use_end("t2")

        muted = [
            c
            for c in console.print.call_args_list
            if c.args
            and isinstance(c.args[0], RichText)
            and getattr(c.args[0], "style", None) == "anton.muted"
        ]
        assert muted == []

    @patch("anton.chat_ui.Live")
    def test_turn_ending_with_tool_prints_no_answer(self, MockLive):
        from rich.markdown import Markdown as RichMarkdown

        display, console = self._make_display()
        display.start()

        display.append_text("Preamble")
        display.on_tool_use_start("t1", "scratchpad")
        display.on_tool_use_end("t1")
        display.finish()

        markdowns = [
            c
            for c in console.print.call_args_list
            if c.args and isinstance(c.args[0], RichMarkdown)
        ]
        # No trailing text → no anton> answer block
        assert markdowns == []

    @patch("anton.chat_ui.Live")
    def test_no_tools_single_markdown_answer(self, MockLive):
        from rich.markdown import Markdown as RichMarkdown

        display, console = self._make_display()
        display.start()

        display.append_text("Hello ")
        display.append_text("world!")
        display.finish()

        markdowns = [
            c.args[0].markup
            for c in console.print.call_args_list
            if c.args and isinstance(c.args[0], RichMarkdown)
        ]
        assert markdowns == ["Hello world!"]


class TestQuestionRenderTracker:
    """CLIElicitor.ask() waits on this before starting prompt_toolkit, so a
    published question is guaranteed to already be on screen — otherwise
    prompt_toolkit's own (synchronous, pre-await) render races the
    out-of-band print of the question and either can land first."""

    async def test_mark_rendered_before_waiting_resolves_immediately(self):
        tracker = QuestionRenderTracker()
        tracker.mark_rendered("q1")

        await asyncio.wait_for(tracker.wait_for_render("q1", timeout=5), timeout=0.2)

    async def test_wait_for_render_unblocks_as_soon_as_marked(self):
        tracker = QuestionRenderTracker()
        waiter = asyncio.ensure_future(tracker.wait_for_render("q1", timeout=5))
        await asyncio.sleep(0)  # let the waiter register itself
        assert not waiter.done()

        tracker.mark_rendered("q1")

        await asyncio.wait_for(waiter, timeout=0.2)

    async def test_wait_for_render_gives_up_after_timeout_if_never_marked(self):
        tracker = QuestionRenderTracker()

        await asyncio.wait_for(
            tracker.wait_for_render("q1", timeout=0.05), timeout=0.2
        )

    async def test_unrelated_question_id_does_not_unblock_a_different_one(self):
        tracker = QuestionRenderTracker()
        waiter = asyncio.ensure_future(tracker.wait_for_render("q1", timeout=0.05))
        await asyncio.sleep(0)

        tracker.mark_rendered("q2")

        assert not waiter.done()
        await asyncio.wait_for(waiter, timeout=0.2)  # only resolves via its own timeout


class TestEscapeWatcherStdinReads:
    """`_watch()` selects in an executor but reads on the event loop's own
    thread, and stdin is shared with prompt_toolkit's reader while an
    interactive prompt is up. If that reader consumes the bytes in between,
    a blocking read here freezes the whole event loop until the next
    keystroke — which is how an `ask_user` prompt ended up missing its
    `Esc to cancel` toolbar until a key was pressed: the redraw scheduled by
    the CPR response sat in a loop that was no longer running."""

    def test_read_available_returns_empty_instead_of_blocking(self):
        read_fd, write_fd = os.pipe()  # blocking, and deliberately left empty
        result: dict[str, bytes] = {}

        def call() -> None:
            result["data"] = EscapeWatcher._read_available(read_fd, 1)

        worker = threading.Thread(target=call, daemon=True)
        try:
            worker.start()
            worker.join(timeout=2)
            assert not worker.is_alive(), (
                "read blocked on an fd with nothing to read — on the event "
                "loop's thread that stalls every pending callback"
            )
            assert result["data"] == b""
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_read_available_reads_pending_bytes_and_restores_blocking_mode(self):
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, b"\x1b[A")

            assert EscapeWatcher._read_available(read_fd, 1) == b"\x1b"
            assert EscapeWatcher._read_available(read_fd, 32) == b"[A"
            # Left as found: stdin is shared with prompt_toolkit and with the
            # user's shell after the run.
            assert os.get_blocking(read_fd)
        finally:
            os.close(read_fd)
            os.close(write_fd)
