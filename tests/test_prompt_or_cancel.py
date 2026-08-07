"""anton.utils.prompt: the bottom-toolbar height fix in prompt_or_cancel()."""

from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import BufferControl

from anton.utils.prompt import _pin_input_window_height


def _default_buffer_window(session: PromptSession) -> Window:
    """Find the Window rendering session.default_buffer — mirrors the walk
    _pin_input_window_height itself does, so the test observes the same
    node the fix is meant to change."""
    seen: set[int] = set()

    def walk(container):
        if id(container) in seen:
            return None
        seen.add(id(container))
        if isinstance(container, Window) and isinstance(container.content, BufferControl):
            if container.content.buffer is session.default_buffer:
                return container
        for child in container.get_children():
            found = walk(child)
            if found is not None:
                return found
        return None

    window = walk(session.layout.container)
    assert window is not None, "default_buffer window not found in layout"
    return window


def test_pin_input_window_height_stops_it_from_extending():
    # Without the fix, PromptSession leaves this window free to extend —
    # which is exactly what lets it swallow the space reserved for the
    # bottom toolbar and push the toolbar down to the terminal's last row.
    session = PromptSession(bottom_toolbar=lambda: "hint")
    window = _default_buffer_window(session)
    assert window.dont_extend_height() is False

    _pin_input_window_height(session)
    assert window.dont_extend_height() is True


def test_pin_input_window_height_is_safe_without_a_bottom_toolbar():
    # A session with no bottom_toolbar at all has nothing this fix needs to
    # protect, but the walk still finds the same window unconditionally —
    # confirm that is harmless rather than assuming it.
    session = PromptSession()
    _pin_input_window_height(session)  # must not raise
    assert _default_buffer_window(session).dont_extend_height() is True
