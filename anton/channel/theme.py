from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from rich.theme import Theme


@dataclass(frozen=True)
class Palette:
    cyan: str
    cyan_dim: str
    success: str
    error: str
    warning: str
    muted: str


DARK_PALETTE = Palette(
    cyan="#00FFFF",
    cyan_dim="#008B8B",
    success="#2FBF71",
    error="#FF6B6B",
    warning="#FFB020",
    muted="#6B7280",
)

LIGHT_PALETTE = Palette(
    cyan="#006B6B",
    cyan_dim="#004D4D",
    success="#1A7F42",
    error="#DC2626",
    warning="#D97706",
    muted="#9CA3AF",
)


def detect_color_mode() -> str:
    override = os.environ.get("ANTON_THEME", "").lower()
    if override in ("dark", "light"):
        return override

    colorfgbg = os.environ.get("COLORFGBG", "")
    if colorfgbg:
        parts = colorfgbg.split(";")
        try:
            bg = int(parts[-1])
            return "light" if bg > 6 else "dark"
        except (ValueError, IndexError):
            pass

    if os.uname().sysname == "Darwin":
        try:
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode != 0:
                return "light"
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    return "dark"


def get_palette(mode: str) -> Palette:
    return LIGHT_PALETTE if mode == "light" else DARK_PALETTE


def build_rich_theme(mode: str) -> Theme:
    p = get_palette(mode)
    return Theme(
        {
            "anton.cyan": p.cyan,
            "anton.cyan_dim": p.cyan_dim,
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
        }
    )
