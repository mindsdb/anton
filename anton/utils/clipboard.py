"""Clipboard and file-attachment helpers for the chat loop."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from rich.console import Console

from anton.clipboard import clipboard_unavailable_reason

_SENSITIVE_PATTERNS = (
    ".ssh/",
    ".aws/",
    ".gnupg/",
    ".anton/.env",
    ".anton/data_vault/",
    ".env",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".docker/config.json",
    "credentials.json",
    ".git/config",
    ".kube/config",
)


def _is_sensitive_path(p: Path) -> bool:
    """Check if a path matches known sensitive file patterns."""
    s = str(p)
    return any(pattern in s for pattern in _SENSITIVE_PATTERNS)


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
        if _is_sensitive_path(p):
            console.print(
                f"  [anton.warning]⚠ Skipped {p.name} — this looks like a "
                f"sensitive file (credentials, keys). Use the scratchpad "
                f"to read it locally if needed.[/]"
            )
            parts.append(
                f'\n<file path="{p}">\n'
                f"(Sensitive file — not sent to the LLM. "
                f"If you need to process it, use the scratchpad.)\n</file>"
            )
            continue

        suffix = p.suffix.lower()
        size = p.stat().st_size

        console.print(f"  [anton.muted]attached: {p.name} ({human_size(size)})[/]")

        if size > 512_000:
            parts.append(
                f'\n<file path="{p}">\n(File too large to inline — {human_size(size)}. '
                f"Use the scratchpad to read it.)\n</file>"
            )
            continue

        if suffix in (
            ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
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
