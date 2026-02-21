from __future__ import annotations

import random
import sys
import time
from typing import TYPE_CHECKING

from rich.columns import Columns
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from anton import __version__

if TYPE_CHECKING:
    from rich.console import Console

TAGLINES = [
    "autonomous by design",
    "your problem, my obsession",
    "no meetings, just results",
    "ctrl+c is my safe word",
    "thinks while you sleep",
    "less overthinking, more solving",
    "the coworker who actually listens",
    "turning 'hmm' into 'done'",
    "ask me anything, regret nothing",
    "breaking assumptions so you don't have to",
    "coffee not required",
    "one question away from useful",
    "like a coworker who reads the docs",
    "the intern who never sleeps",
    "you talk, I figure it out",
    "curiosity-driven problem solving",
]

BUBBLE_PHRASES = [
    "\u2661\u2661\u2661\u2661",
    "sup",
    "let's go",
    "ask me",
    "ready",
    "hi boss",
    "what's the plan?",
    "I'm here",
    "try me",
]

# Mouth chars: smile, talking open, talking mid
_MOUTH_SMILE = "\u1d17"   # ᴗ
_MOUTH_OPEN = "o"
_MOUTH_MID = "\u203f"     # ‿
_MOUTH_TALK = [_MOUTH_OPEN, _MOUTH_MID]


def pick_tagline(seed: int | None = None) -> str:
    rng = random.Random(seed)
    return rng.choice(TAGLINES)


def _build_robot_text(mouth: str, bubble: str) -> Text:
    """Build the full robot as a Rich Text object with styling."""
    g = "bold cyan"
    m = "dim"
    # Pad bubble to avoid layout jitter (longest phrase is ~16 chars)
    padded = bubble.ljust(16)
    lines = [
        (f"        \u2590\n", g),
        (f"   \u2584\u2588\u2580\u2588\u2588\u2580\u2588\u2584   ", g),
        (f"{padded}\n", g),
        (f" \u2588\u2588", g),
        (f"  (\u00b0{mouth}\u00b0) ", m),
        (f"\u2588\u2588\n", g),
        (f"   \u2580\u2588\u2584\u2588\u2588\u2584\u2588\u2580", g),
        (f"          \u2584\u2580\u2588 \u2588\u2584 \u2588 \u2580\u2588\u2580 \u2588\u2580\u2588 \u2588\u2584 \u2588\n", g),
        (f"    \u2590   \u2590", g),
        (f"            \u2588\u2580\u2588 \u2588 \u2580\u2588  \u2588  \u2588\u2584\u2588 \u2588 \u2580\u2588\n", g),
        (f"    \u2590   \u2590\n", g),
    ]
    text = Text()
    for content, style in lines:
        text.append(content, style=style)
    return text


def _render_robot_static(console: Console, bubble: str = "\u2661\u2661\u2661\u2661") -> None:
    """Render the static ASCII robot (used as fallback)."""
    g = "anton.glow"
    m = "anton.muted"
    padded = bubble.ljust(16)
    console.print(f"[{g}]        \u2590[/]")
    console.print(f"[{g}]   \u2584\u2588\u2580\u2588\u2588\u2580\u2588\u2584[/]   [{g}]{padded}[/]")
    console.print(f"[{g}] \u2588\u2588[/]  [{m}](\u00b0\u1d17\u00b0)[/] [{g}]\u2588\u2588[/]")
    console.print(
        f"[{g}]   \u2580\u2588\u2584\u2588\u2588\u2584\u2588\u2580[/]"
        f"          [{g}]\u2584\u2580\u2588 \u2588\u2584 \u2588 \u2580\u2588\u2580 \u2588\u2580\u2588 \u2588\u2584 \u2588[/]"
    )
    console.print(
        f"[{g}]    \u2590   \u2590[/]"
        f"            [{g}]\u2588\u2580\u2588 \u2588 \u2580\u2588  \u2588  \u2588\u2584\u2588 \u2588 \u2580\u2588[/]"
    )
    console.print(f"[{g}]    \u2590   \u2590[/]")


def _animate_banner(console: Console) -> None:
    """Run a quick typing animation: hearts → one random phrase → hearts."""
    rng = random.Random()
    middle = [p for p in BUBBLE_PHRASES if p != "\u2661\u2661\u2661\u2661"]
    phrase = rng.choice(middle)

    type_speed = 0.05
    pause_after = 0.35
    clear_speed = 0.02

    with Live(
        _build_robot_text(_MOUTH_SMILE, "\u2661\u2661\u2661\u2661"),
        console=console,
        refresh_per_second=30,
        transient=True,
    ) as live:
        time.sleep(0.3)

        # Type the phrase
        for j in range(1, len(phrase) + 1):
            mouth = _MOUTH_TALK[j % 2]
            live.update(_build_robot_text(mouth, phrase[:j]))
            time.sleep(type_speed)

        # Hold with smile
        live.update(_build_robot_text(_MOUTH_SMILE, phrase))
        time.sleep(pause_after)

        # Clear back
        for j in range(len(phrase), 0, -1):
            live.update(_build_robot_text(_MOUTH_SMILE, phrase[:j - 1]))
            time.sleep(clear_speed)

        # Back to hearts
        live.update(_build_robot_text(_MOUTH_SMILE, "\u2661\u2661\u2661\u2661"))
        time.sleep(0.15)

    _render_robot_static(console, "\u2661\u2661\u2661\u2661")


def render_banner(console: Console, *, animate: bool = True) -> None:
    tagline = pick_tagline()

    # Animate only on interactive terminals
    if animate and sys.stdout.isatty():
        try:
            _animate_banner(console)
        except Exception:
            _render_robot_static(console)
    else:
        _render_robot_static(console)

    console.print(f"[anton.cyan_dim] {'━' * 40}[/]")
    console.print(
        f" v{__version__} \u2014 [anton.muted]\"{tagline}\"[/]",
    )


def render_dashboard(console: Console) -> None:
    from pathlib import Path

    from anton.config.settings import AntonSettings

    settings = AntonSettings()
    tagline = pick_tagline()

    _render_robot_static(console)
    console.print(f"[anton.cyan_dim] {'━' * 40}[/]")
    console.print(
        f" v{__version__} \u2014 [anton.muted]\"{tagline}\"[/]",
    )
    console.print(f"[anton.cyan_dim] {'━' * 40}[/]")
    console.print()

    # Count skills
    from anton.skill.registry import SkillRegistry

    registry = SkillRegistry()
    builtin = Path(__file__).resolve().parent.parent.parent / settings.skills_dir
    registry.discover(builtin)
    user_dir = Path(settings.user_skills_dir).expanduser()
    registry.discover(user_dir)
    skill_count = len(registry.list_all())

    # Count sessions
    session_count = 0
    if settings.memory_enabled:
        try:
            from anton.memory.store import SessionStore

            memory_dir = Path(settings.memory_dir).expanduser()
            store = SessionStore(memory_dir)
            session_count = len(store.list_sessions())
        except Exception:
            pass

    from anton.channel.theme import detect_color_mode

    mode = detect_color_mode()

    commands_content = (
        "[anton.cyan]run[/] <task>    Execute a task\n"
        "[anton.cyan]skills[/]        List skills\n"
        "[anton.cyan]sessions[/]      Browse sessions\n"
        "[anton.cyan]learnings[/]     Review learnings\n"
        "[anton.cyan]channels[/]      List channels\n"
        "[anton.cyan]version[/]       Show version"
    )

    memory_label = "enabled" if settings.memory_enabled else "disabled"
    model_label = settings.coding_model
    if len(model_label) > 16:
        model_label = model_label[:16] + "\u2026"

    status_content = (
        f"[anton.cyan]Skills[/]    {skill_count} loaded\n"
        f"[anton.cyan]Memory[/]    {memory_label}\n"
        f"[anton.cyan]Sessions[/]  {session_count} stored\n"
        f"[anton.cyan]Channel[/]   cli\n"
        f"[anton.cyan]Theme[/]     {mode}\n"
        f"[anton.cyan]Model[/]     {model_label}"
    )

    commands_panel = Panel(
        commands_content,
        title="Commands",
        border_style="anton.cyan_dim",
        width=30,
    )
    status_panel = Panel(
        status_content,
        title="Status",
        border_style="anton.cyan_dim",
        width=26,
    )

    console.print(Columns([commands_panel, status_panel], padding=(0, 1)))
    console.print()
    console.print(
        ' [anton.muted]Quick start:[/] [anton.cyan]anton run "fix the failing tests"[/]'
    )
    console.print()
