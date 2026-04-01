from __future__ import annotations

import os
from dataclasses import dataclass

from rich.theme import Theme


@dataclass(frozen=True)
class Palette:
    cyan: str
    cyan_dim: str
    prompt: str
    user_prompt: str
    success: str
    error: str
    warning: str
    muted: str


DARK_PALETTE = Palette(
    cyan="#22d3ee",
    cyan_dim="#0891b2",
    prompt="#22d3ee",
    user_prompt="#00ff9f",
    success="#2FBF71",
    error="#FF6B6B",
    warning="#FFB020",
    muted="#6B7280",
)

LIGHT_PALETTE = Palette(
    cyan="#006B6B",
    cyan_dim="#004D4D",
    prompt="#005F5F",
    user_prompt="#1A7F42",
    success="#1A7F42",
    error="#DC2626",
    warning="#D97706",
    muted="#9CA3AF",
)


def detect_color_mode() -> str:
    override = os.environ.get("ANTON_THEME", "").lower()
    if override in ("dark", "light"):
        return override
    return "dark"


def get_palette(mode: str | None = None) -> Palette:
    if mode is None:
        mode = detect_color_mode()
    return LIGHT_PALETTE if mode == "light" else DARK_PALETTE


def build_rich_theme(mode: str) -> Theme:
    p = get_palette(mode)
    return Theme(
        {
            "anton.cyan": p.cyan,
            "anton.cyan_dim": p.cyan_dim,
            "anton.prompt": f"bold {p.prompt}",
            "anton.glow": f"bold {p.cyan}",
            "anton.heading": f"bold {p.cyan}",
            "anton.success": p.success,
            "anton.error": p.error,
            "anton.warning": p.warning,
            "anton.muted": p.muted,
            "phase.planning": "bold blue",
            "phase.skill_discovery": f"bold {p.cyan}",
            "phase.skill_building": "bold magenta",
            "phase.executing": f"bold {p.warning}",
            "phase.complete": f"bold {p.success}",
            "phase.failed": f"bold {p.error}",
            # Rich Markdown styles
            "markdown.h1": f"bold {p.cyan}",
            "markdown.h2": f"bold {p.cyan}",
            "markdown.h3": f"bold {p.cyan}",
            "markdown.h4": f"bold {p.cyan_dim}",
            "markdown.strong": "bold",
            "markdown.emph": "italic",
            "markdown.code": f"bold {p.warning}",
            "markdown.link": f"underline {p.cyan}",
            "markdown.link_url": p.cyan_dim,
            "markdown.item.bullet": p.cyan,
            "markdown.item.number": p.cyan,
            "markdown.block_quote": f"italic {p.muted}",
            "markdown.hr": p.muted,
        }
    )
