from __future__ import annotations

import random
from typing import TYPE_CHECKING

from rich.columns import Columns
from rich.panel import Panel
from rich.text import Text

from anton import __version__

if TYPE_CHECKING:
    from rich.console import Console

ASCII_LOGO = (
    " \u2584\u2580\u2588 \u2588\u2584 \u2588 \u2580\u2588\u2580 \u2588\u2580\u2588 \u2588\u2584 \u2588\n"
    " \u2588\u2580\u2588 \u2588 \u2580\u2588  \u2588  \u2588\u2584\u2588 \u2588 \u2580\u2588"
)

TAGLINES = [
    "autonomous by design",
    "your code, my plan",
    "no meetings, just results",
    "ctrl+c is my safe word",
    "ships while you sleep",
    "less yak-shaving, more shipping",
    "pair programming without the small talk",
    "turning TODOs into DONEs",
    "git push and chill",
    "breaking prod so you don't have to",
    "coffee not required",
    "one task, zero excuses",
    "like a coworker who reads the docs",
    "the intern who never sleeps",
    "sudo make me a sandwich",
    "async everything, regret nothing",
]


def pick_tagline(seed: int | None = None) -> str:
    rng = random.Random(seed)
    return rng.choice(TAGLINES)


def render_banner(console: Console) -> None:
    tagline = pick_tagline()
    logo = Text(ASCII_LOGO, style="anton.cyan")
    console.print(logo)
    console.print(
        f" v{__version__} \u2014 [anton.muted]\"{tagline}\"[/]",
    )
    console.print()


def render_dashboard(console: Console) -> None:
    from pathlib import Path

    from anton.config.settings import AntonSettings

    settings = AntonSettings()
    tagline = pick_tagline()

    logo = Text(ASCII_LOGO, style="anton.cyan")
    console.print(logo)
    console.print(
        f" v{__version__} \u2014 [anton.muted]\"{tagline}\"[/]",
    )
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
