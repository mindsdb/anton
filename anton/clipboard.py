"""Clipboard image/file paste support.

Grabs images or file paths from the system clipboard (macOS/Windows/Linux),
saves uploads to .anton/uploads/, and provides path-parsing utilities
shared with the drag-and-drop logic in chat.py.
"""

from __future__ import annotations

import os
import re
import json
import time
import shlex
import shutil
import hashlib
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


IMAGE_EXTENSION_REGEX = re.compile(r"\.(png|jpe?g|gif|webp|bmp)$", re.IGNORECASE)
IMAGE_REF_REGEX = re.compile(r"\[Image #(\d+)\]")
AT_PATH_REGEX = re.compile(
    r'(?:(?<=\s)|^)@(?:"([^"]+)"|\'([^\']+)\'|(\S+))'
)


@dataclass
class ClipboardImage:
    """An image grabbed from the clipboard."""

    image: Any  # PIL.Image.Image (typed as Any to avoid hard Pillow dep)
    width: int
    height: int
    mode: str


@dataclass
class ClipboardResult:
    """Result of inspecting the system clipboard."""

    image: ClipboardImage | None = None
    file_paths: list[Path] = field(default_factory=list)
    text: str = ""


@dataclass
class UploadedFile:
    """Metadata for a saved clipboard image."""

    path: Path
    original_type: str
    width: int
    height: int
    size_bytes: int
    format: str


@dataclass
class PastedImage:
    """An image attached via drag-and-drop, addressable by [Image #id] in the prompt."""

    id: int
    path: Path
    size_bytes: int
    format: str  # uppercase, e.g. "PNG", "JPEG"


class PastedImageRegistry:
    """Per-chat-session registry mapping pasted-image IDs to file metadata."""

    def __init__(self) -> None:
        self._next_id = 1
        self._items: dict[int, PastedImage] = {}

    def add(self, path: Path) -> PastedImage:
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = 0
        ext = path.suffix.lstrip(".").lower()
        fmt = {"jpg": "JPEG"}.get(ext, ext.upper() or "PNG")
        item = PastedImage(
            id=self._next_id,
            path=path,
            size_bytes=size_bytes,
            format=fmt,
        )
        self._items[self._next_id] = item
        self._next_id += 1
        return item

    def get(self, ref_id: int) -> PastedImage | None:
        return self._items.get(ref_id)

    def prune_unused(self, kept_ids: set[int]) -> None:
        for rid in list(self._items.keys()):
            if rid not in kept_ids:
                self._items.pop(rid, None)


def format_image_ref(ref_id: int) -> str:
    return f"[Image #{ref_id}]"


def is_image_path(token: str) -> bool:
    return bool(IMAGE_EXTENSION_REGEX.search(token))


def replace_image_paths_in_pasted(text: str, registry: PastedImageRegistry) -> tuple[str, list[PastedImage]]:
    """Find image file paths in pasted text, register them, replace each with [Image #N].

    Returns ``(rewritten_text, registered_images)``. Paths that don't exist or
    aren't absolute are left untouched. Multiple representations of the same
    path (quoted, escaped) are handled.
    """
    registered: list[PastedImage] = []
    out_lines: list[str] = []

    for line in text.splitlines(keepends=True):
        body = line.rstrip("\n").rstrip("\r")
        suffix = line[len(body):]

        try:
            tokens = shlex.split(body)
        except ValueError:
            tokens = []

        for token in tokens:
            if len(token) < 2 or not is_image_path(token):
                continue
            candidate = Path(token)
            try:
                if not (candidate.is_absolute() and candidate.is_file()):
                    continue
            except OSError:
                continue
            item = registry.add(candidate)
            registered.append(item)
            ref = format_image_ref(item.id)
            path_str = str(candidate)
            for rep in (
                f"'{path_str}'",
                f'"{path_str}"',
                path_str.replace(" ", "\\ "),
                path_str,
            ):
                if rep and rep in body:
                    body = body.replace(rep, ref, 1)
                    break

        out_lines.append(body + suffix)

    return "".join(out_lines), registered


def replace_at_image_paths(
    text: str, registry: PastedImageRegistry
) -> tuple[str, list[PastedImage]]:
    """Find @<path> references to image files, register them, replace each with [Image #N].

    Supports absolute paths and paths relative to cwd. Non-image or non-existent
    paths are left untouched (the literal "@<path>" stays in the text).
    """
    registered: list[PastedImage] = []

    def _sub(match: re.Match) -> str:
        token = match.group(1) or match.group(2) or match.group(3) or ""
        if not token or not is_image_path(token):
            return match.group(0)
        try:
            candidate = Path(token).expanduser()
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate).resolve()
            if not candidate.is_file():
                return match.group(0)
        except OSError:
            return match.group(0)
        item = registry.add(candidate)
        registered.append(item)
        return format_image_ref(item.id)

    new_text = AT_PATH_REGEX.sub(_sub, text)
    return new_text, registered


def _linux_clipboard_tool() -> str | None:
    """Pick a Linux clipboard CLI tool (wl-paste/xclip), or None if absent."""
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-paste"):
        return "wl-paste"
    if shutil.which("xclip"):
        return "xclip"
    if shutil.which("wl-paste"):
        return "wl-paste"
    return None


def is_clipboard_supported() -> bool:
    """Return True if we can attempt clipboard image grabs on this platform."""
    return clipboard_unavailable_reason() is None


def clipboard_unavailable_reason() -> str | None:
    """Return a reason string if clipboard is unavailable, or None if OK.

    Possible reasons: unsupported_platform, missing_pillow,
    missing_linux_clipboard_tools.
    """
    system = platform.system()
    if system not in ("Darwin", "Windows", "Linux"):
        return "unsupported_platform"
    try:
        from PIL import ImageGrab  # noqa: F401
    except ImportError:
        return "missing_pillow"
    if system == "Linux" and _linux_clipboard_tool() is None:
        return "missing_linux_clipboard_tools"
    return None


def grab_clipboard() -> ClipboardResult:
    """Inspect the system clipboard; try image first, then text/file paths."""
    result = ClipboardResult()

    # Try image via Pillow
    img = _grab_image()
    if img is not None:
        result.image = ClipboardImage(
            image=img,
            width=img.size[0],
            height=img.size[1],
            mode=img.mode,
        )
        return result

    # Fall back to text (may contain file paths)
    text = _grab_text()
    if text:
        # Check if the text looks like file paths
        paths = parse_dropped_paths(text)
        if paths:
            result.file_paths = paths
        else:
            result.text = text

    return result


def _grab_image() -> Any | None:
    """Attempt to grab an image from the clipboard via Pillow.

    Returns a PIL Image or None.  On macOS, ``grabclipboard()`` may return
    a *list* of file paths when the user copied files in Finder — we skip
    those (they'll be handled by the text path).
    """
    try:
        from PIL import ImageGrab
    except ImportError:
        return None

    try:
        clip = ImageGrab.grabclipboard()
    except Exception:
        return None

    if clip is None:
        return None

    # macOS quirk: Finder-copied files come back as a list of paths
    if isinstance(clip, list):
        return None

    # Ensure it's an actual PIL Image
    try:
        from PIL import Image

        if isinstance(clip, Image.Image):
            return clip
    except Exception:
        pass

    return None


def _grab_text() -> str:
    """Grab text from the clipboard using platform-native CLI tools."""
    system = platform.system()
    try:
        if system == "Darwin":
            return subprocess.run(
                ["pbpaste"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        elif system == "Windows":
            return subprocess.run(
                ["powershell", "-Command", "Get-Clipboard"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        elif system == "Linux":
            tool = _linux_clipboard_tool()
            if tool == "wl-paste":
                return subprocess.run(
                    ["wl-paste", "--no-newline"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout
            elif tool == "xclip":
                return subprocess.run(
                    ["xclip", "-selection", "clipboard", "-o"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout
    except Exception:
        pass
    return ""


def save_clipboard_image(image: Any, uploads_dir: Path) -> UploadedFile:
    """Save a PIL Image to *uploads_dir* as PNG and write a .meta.json sidecar.

    Parameters
    ----------
    image:
        A ``PIL.Image.Image`` (or the ``ClipboardImage.image`` attribute).
    uploads_dir:
        Directory to save into (created if missing).

    Returns
    -------
    UploadedFile with the saved path and metadata.
    """
    uploads_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    # Hash a few pixels for uniqueness
    raw = image.tobytes()[:4096]
    h = hashlib.sha256(raw).hexdigest()[:8]
    filename = f"clipboard_{ts}_{h}.png"
    filepath = uploads_dir / filename

    image.save(filepath, format="PNG")
    size_bytes = filepath.stat().st_size

    # Sidecar metadata
    meta = {
        "source": "clipboard",
        "timestamp": ts,
        "width": image.size[0],
        "height": image.size[1],
        "mode": image.mode,
        "format": "PNG",
        "size_bytes": size_bytes,
    }
    meta_path = filepath.with_suffix(".png.meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    return UploadedFile(
        path=filepath,
        original_type="clipboard",
        width=image.size[0],
        height=image.size[1],
        size_bytes=size_bytes,
        format="PNG",
    )


def cleanup_old_uploads(uploads_dir: Path, max_age_days: int = 7) -> int:
    """Delete uploads older than *max_age_days*.  Returns count of files removed."""
    if not uploads_dir.is_dir():
        return 0

    cutoff = time.time() - (max_age_days * 86400)
    removed = 0

    for f in list(uploads_dir.iterdir()):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            continue

    return removed


def parse_dropped_paths(text: str) -> list[Path]:
    r"""Detect file paths from terminal drag-and-drop or clipboard text.

    When users drag files into the terminal, the shell pastes the path as:
      - /path/to/file           (macOS/Linux, no spaces)
      - /path/to/file\ name    (macOS, escaped spaces)
      - '/path/to/file name'   (macOS, quoted)
      - "C:\Users\foo\file"    (Windows, quoted)
      - C:\Users\foo\file      (Windows, no spaces)
    Multiple files may be separated by spaces or newlines.
    """
    paths: list[Path] = []

    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            tokens = shlex.split(line)
        except ValueError:
            tokens = [line]

        for token in tokens:
            if len(token) < 2:
                continue
            candidate = Path(token)
            try:
                if candidate.is_absolute() and candidate.exists():
                    paths.append(candidate)
            except OSError:
                continue

    return paths
