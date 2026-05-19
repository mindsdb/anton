"""Clipboard and file-attachment helpers for the chat loop."""

from __future__ import annotations

import asyncio
import base64
import sys
from pathlib import Path

from rich.console import Console

from anton.clipboard import (
    IMAGE_REF_REGEX,
    PastedImageRegistry,
    clipboard_unavailable_reason,
    is_image_path,
    replace_image_paths_in_pasted,
)


MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB base64 limit per Anthropic API


class ImageTooLargeError(Exception):
    """Raised when an image exceeds the Claude API base64 size limit."""

    def __init__(self, size_bytes: int) -> None:
        self.size_bytes = size_bytes
        super().__init__(
            f"Image too large: {self._format_size(size_bytes)} "
            "(base64 would exceed 5 MB)"
        )

    @staticmethod
    def _format_size(nbytes: int) -> str:
        """Format bytes as human-readable size."""
        for unit in ("B", "KB", "MB", "GB"):
            if nbytes < 1024:
                return f"{nbytes:.0f}{unit}" if unit == "B" else f"{nbytes:.1f}{unit}"
            nbytes /= 1024
        return f"{nbytes:.1f}TB"


def human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.0f}{unit}" if unit == "B" else f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


def format_file_message(text: str, paths: list[Path], console: Console) -> str:
    """Rewrite user input to include file contents for detected paths."""
    parts: list[str] = []

    remaining = text
    for p in paths:
        for representation in (str(p), f"'{p}'", f'"{p}"', str(p).replace(" ", "\\ ")):
            remaining = remaining.replace(representation, "")
    remaining = remaining.strip()

    if remaining:
        parts.append(remaining)
    else:
        if len(paths) == 1:
            parts.append(f"Analyze this file: {paths[0].name}")
        else:
            names = ", ".join(p.name for p in paths)
            parts.append(f"Analyze these files: {names}")

    for p in paths:
        suffix = p.suffix.lower()
        size = p.stat().st_size

        console.print(f"  [anton.muted]attached: {p.name} ({human_size(size)})[/]")

        # Image check runs before the size guard: `read_image` accepts files
        # up to ~3.75 MB raw, well above 512 KB, so the generic 'too large
        # to inline' message would silently hide the tool hint for typical
        # photos and screenshots.
        if is_image_path(p.name):
            parts.append(
                f'\n<file path="{p}">\n(Image file — {human_size(size)}. '
                f'Call the `read_image` tool with file_path="{p}" to view it, or use the scratchpad to process it)\n</file>'
            )
            continue

        if size > 512_000:
            parts.append(
                f'\n<file path="{p}">\n(File too large to inline — {human_size(size)}. '
                f"Use the scratchpad to read it.)\n</file>"
            )
            continue

        if suffix in (
            ".ico",
            ".pdf", ".zip", ".tar", ".gz", ".exe", ".dll", ".so",
            ".pyc", ".pyo", ".whl", ".egg", ".db", ".sqlite",
        ):
            parts.append(
                f'\n<file path="{p}">\n(Binary file — {human_size(size)}. '
                f"Use the scratchpad to process it.)\n</file>"
            )
            continue

        try:
            content = p.read_text(errors="replace")
        except Exception:
            parts.append(f'\n<file path="{p}">\n(Could not read file.)\n</file>')
            continue

        parts.append(f'\n<file path="{p}">\n{content}\n</file>')

    return "\n".join(parts)


def format_clipboard_image_message(uploaded: object, user_text: str = "") -> list[dict]:
    """Build a multimodal LLM message for a clipboard image upload."""
    import base64

    if uploaded.size_bytes * 4 // 3 > MAX_IMAGE_BYTES:
        raise ImageTooLargeError(uploaded.size_bytes)

    text = (
        user_text.strip()
        if user_text
        else "I've pasted an image from my clipboard. Analyze it."
    )
    text += (
        f"\n\nThe image is also saved at: {uploaded.path}\n"
        f"({uploaded.width}x{uploaded.height}, {human_size(uploaded.size_bytes)}). "
        f"If you need to process it programmatically, use that path in the scratchpad."
    )

    image_data = Path(uploaded.path).read_bytes()
    b64 = base64.standard_b64encode(image_data).decode("ascii")

    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64,
            },
        },
        {
            "type": "text",
            "text": text,
        },
    ]


async def ensure_clipboard(console: Console) -> bool:
    """Check clipboard support; offer to install Pillow if missing.

    Returns True if clipboard is ready to use, False otherwise.
    """
    reason = clipboard_unavailable_reason()
    if reason is None:
        return True
    if reason == "unsupported_platform":
        console.print("[anton.warning]Clipboard is not supported on this platform.[/]")
        return False
    if reason == "missing_linux_clipboard_tools":
        console.print(
            "[anton.warning]Clipboard on Linux requires [b]wl-clipboard[/] (Wayland) "
            "or [b]xclip[/] (X11). Install one and try again.[/]"
        )
        return False
    # reason == "missing_pillow"
    console.print("[anton.muted]Clipboard image support requires Pillow.[/]")
    answer = console.input("[bold]Install Pillow now? (y/n):[/] ").strip().lower()
    if answer not in ("y", "yes"):
        console.print("[anton.muted]Skipped.[/]")
        return False
    console.print("[anton.muted]Installing Pillow...[/]")
    import subprocess

    proc = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: subprocess.run(
            ["uv", "pip", "install", "--python", sys.executable, "Pillow"],
            capture_output=True,
            timeout=120,
        ),
    )
    if proc.returncode == 0:
        console.print("[anton.success]Pillow installed. Clipboard is now available.[/]")
        return True

    proc = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: subprocess.run(
            [sys.executable, "-m", "pip", "install", "Pillow"],
            capture_output=True,
            timeout=120,
        ),
    )
    if proc.returncode == 0:
        console.print("[anton.success]Pillow installed. Clipboard is now available.[/]")
        return True
    console.print("[anton.error]Failed to install Pillow.[/]")
    return False


# -------------------------- pasted-image plumbing ---------------------------
#
# When the user drags an image into the terminal, we want the path to be swapped
# for a [Image #N] chip in the prompt buffer. Three pieces:
#   1. ``make_image_paste_bindings`` — prompt_toolkit KeyBindings that intercept
#      bracketed paste and rewrite image paths to refs.
#   2. ``ImageRefLexer`` — colorizes [Image #N] in the input line.
#   3. ``build_image_ref_message`` — at submit time, expands refs into a list
#      of multimodal content blocks for the LLM.


def _media_type_for(fmt: str) -> str:
    f = (fmt or "PNG").lower()
    if f in ("jpg", "jpeg"):
        return "image/jpeg"
    if f in ("png", "gif", "webp", "bmp"):
        return f"image/{f}"
    return "image/png"


def make_image_paste_bindings(registry: PastedImageRegistry, console: Console):
    """Build a ``KeyBindings`` that swaps image paths in pasted text for [Image #N]."""
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys

    bindings = KeyBindings()

    @bindings.add(Keys.BracketedPaste, eager=True)
    def _(event):
        data = event.data.replace("\r\n", "\n").replace("\r", "\n")
        rewritten, registered = replace_image_paths_in_pasted(data, registry)
        event.current_buffer.insert_text(rewritten)

    return bindings


def attach_image_path_detector(buffer, registry: PastedImageRegistry) -> None:
    """Watch a Buffer for image paths *pasted* in, swap them for [Image #N].

    Most Linux terminals (GNOME Terminal, Konsole, …) deliver drag-and-drop as a
    burst of normal keypresses rather than bracketed paste, so the
    BracketedPaste handler never fires. We distinguish paste from typing by the
    inter-event interval: a paste burst arrives in <1 ms gaps, manual typing
    has 50–200 ms gaps even at 300 wpm. After ≥5 consecutive changes with
    sub-20 ms gaps we schedule a 50 ms-debounced check that runs the path
    detector on the settled buffer.
    """
    import asyncio
    import time

    from prompt_toolkit.document import Document

    in_handler: list[bool] = [False]
    pending: list[asyncio.TimerHandle | None] = [None]
    last_ts: list[float] = [0.0]
    burst_size: list[int] = [0]
    def _do_check():
        pending[0] = None
        burst_size[0] = 0
        if in_handler[0]:
            return
        text = buffer.text
        if not text or "." not in text:
            return

        # Insert space before processing if buffer doesn't end with whitespace
        needs_space = text and text[-1] not in (' ', '\n', '\t')
        if needs_space:
            in_handler[0] = True
            try:
                buffer.insert_text(' ')
            finally:
                in_handler[0] = False
            text = buffer.text

        rewritten, registered = replace_image_paths_in_pasted(text, registry)
        if not registered:
            return

        cursor = buffer.cursor_position
        new_cursor = max(0, min(cursor + (len(rewritten) - len(text)), len(rewritten)))
        in_handler[0] = True
        try:
            buffer.set_document(Document(rewritten, new_cursor), bypass_readonly=True)
        finally:
            in_handler[0] = False

    def _on_change(_):
        if in_handler[0]:
            return
        now = time.monotonic()
        delta = now - last_ts[0]
        last_ts[0] = now
        if delta < 0.02:
            burst_size[0] += 1
        else:
            burst_size[0] = 1
        if burst_size[0] < 5:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if pending[0] is not None:
            pending[0].cancel()
        pending[0] = loop.call_later(0.05, _do_check)

    buffer.on_text_changed += _on_change


class ImageRefLexer:
    """prompt_toolkit Lexer that highlights [Image #N] chips in the input line."""

    def lex_document(self, document):
        def get_line(lineno: int):
            line = document.lines[lineno]
            fragments: list[tuple[str, str]] = []
            pos = 0
            for m in IMAGE_REF_REGEX.finditer(line):
                if m.start() > pos:
                    fragments.append(("", line[pos:m.start()]))
                fragments.append(("class:image-ref", m.group(0)))
                pos = m.end()
            if pos < len(line):
                fragments.append(("", line[pos:]))
            return fragments

        return get_line


def build_image_ref_message(
    text: str, registry: PastedImageRegistry
) -> tuple[str | list[dict], set[int]]:
    """Expand any [Image #N] placeholders in *text* into multimodal content blocks.

    Returns ``(content, referenced_ids)``. If no refs are present, ``content``
    is the original text string (unchanged). If refs are present but a referenced
    ID is missing from the registry (e.g. the file is gone), the placeholder is
    kept verbatim in the text block.
    """
    matches = list(IMAGE_REF_REGEX.finditer(text))
    if not matches:
        return text, set()

    referenced: set[int] = set()
    blocks: list[dict] = []
    text_buf: list[str] = []

    def flush_text():
        if not text_buf:
            return
        joined = "".join(text_buf)
        if joined.strip():
            blocks.append({"type": "text", "text": joined.strip()})
        text_buf.clear()

    pos = 0
    for m in matches:
        if m.start() > pos:
            text_buf.append(text[pos:m.start()])

        ref_id = int(m.group(1))
        item = registry.get(ref_id)
        if item is None:
            text_buf.append(m.group(0))
        else:
            referenced.add(ref_id)
            try:
                raw = item.path.read_bytes()
            except OSError:
                text_buf.append(m.group(0))
            else:
                flush_text()
                fmt = item.format
                # BMP is not supported by OpenAI or Claude APIs — convert to PNG on the fly.
                if fmt.upper() == "BMP":
                    import io
                    from PIL import Image as _PILImage
                    buf = io.BytesIO()
                    _PILImage.open(io.BytesIO(raw)).save(buf, format="PNG")
                    raw = buf.getvalue()
                    fmt = "PNG"
                if len(raw) * 4 // 3 > MAX_IMAGE_BYTES:
                    raise ImageTooLargeError(len(raw))
                b64 = base64.standard_b64encode(raw).decode("ascii")
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": _media_type_for(fmt),
                        "data": b64,
                    },
                })
        pos = m.end()

    if pos < len(text):
        text_buf.append(text[pos:])
    flush_text()

    if not any(b["type"] == "image" for b in blocks):
        return text, referenced
    return blocks, referenced
