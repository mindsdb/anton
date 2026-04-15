from __future__ import annotations

import os
import subprocess
from pathlib import Path

from rich.console import Console

from anton.utils.prompt import prompt_or_cancel
from anton.config.settings import AntonSettings


MEMORY_MODES = {
    "autopilot": "Autopilot — Anton decides what to remember",
    "copilot": "Co-pilot — save obvious, confirm ambiguous",
    "off": "Off — never save (still reads existing)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_scope(args: list[str], idx: int = 0) -> str:
    """Return 'global', 'project', or 'both' (default) from args[idx]."""
    if idx < len(args) and args[idx] in ("global", "project"):
        return args[idx]
    return "both"


def _numbered_bullets(text: str) -> list[tuple[int, str, str]]:
    """
    Parse a markdown string and return [(n, section_heading, bullet_text), ...].
    section_heading is the nearest ## / # heading above each bullet.
    """
    result: list[tuple[int, str, str]] = []
    section = ""
    n = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            section = stripped.lstrip("#").strip()
        elif stripped.startswith("- "):
            n += 1
            result.append((n, section, stripped[2:]))
    return result


def _delete_bullet(path: Path, n: int) -> bool:
    """Remove the n-th bullet entry (1-indexed) from a markdown file."""
    if not path.exists():
        return False
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    count = 0
    for i, line in enumerate(lines):
        if line.lstrip().startswith("- "):
            count += 1
            if count == n:
                lines.pop(i)
                path.write_text("".join(lines), encoding="utf-8")
                return True
    return False


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MemoryManage:
    """Handler for the /memory command and its sub-commands."""

    def __init__(
        self,
        console: Console,
        settings: AntonSettings,
        cortex: "Cortex | None",
        episodic: "EpisodicMemory | None" = None,
    ) -> None:
        self.console = console
        self.settings = settings
        self.cortex = cortex
        self.episodic = episodic

        self.SUBCOMMANDS: dict[str, object] = {
            "help":        self.help,
            "rules":       self.rules,
            "lessons":     self.lessons,
            "profile":     self.profile,
            "episodes":    self.episodes,
            "edit":        self.edit,
            "delete":      self.delete,
            "prune":       self.prune,
            "vacuum":      self.vacuum,
            "consolidate": self.consolidate,
            "reset":       self.reset,
        }

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def handle(self, cmd: str) -> None:
        """Dispatch /memory [sub-command] or show the status dashboard."""
        sub_cmd = cmd.removeprefix("/memory").strip()
        parts = sub_cmd.split()

        if len(parts) == 0:
            return self.info()

        if parts[0] not in self.SUBCOMMANDS:
            c = self.console
            c.print()
            c.print(f"[anton.error]Unknown command: {cmd}[/]")
            c.print()
            return self.help()

        handler = self.SUBCOMMANDS[parts[0]]

        return handler(parts[1:])

    def help(self) -> None:
        """Show available /memory sub-commands."""
        c = self.console
        c.print()
        c.print("[anton.cyan]Memory commands[/]")
        c.print()
        c.print("  [bold]/memory[/]                                        — status dashboard")
        c.print()
        c.print("  [bold dim]Inspect[/]")
        c.print("  [bold]/memory rules [global|project][/]                 — show behavioral rules")
        c.print("  [bold]/memory lessons [global|project][/]               — show learned lessons")
        c.print("  [bold]/memory profile [global|project][/]               — show identity profile")
        c.print("  [bold]/memory episodes [query][/]                       — search episodic logs")
        c.print()
        c.print("  [bold dim]Edit[/]")
        c.print("  [bold]/memory edit rules|lessons|profile [global|project][/]     — open in $EDITOR")
        c.print("  [bold]/memory delete rule|lesson|profile <n> [global|project][/] — remove entry #n")
        c.print()
        c.print("  [bold dim]Maintenance[/]")
        c.print("  [bold]/memory prune[/]                                  — remove outdated/low-value entries")
        c.print("  [bold]/memory vacuum[/]                                 — deduplicate and compact")
        c.print("  [bold]/memory consolidate[/]                            — trigger manual consolidation")
        c.print("  [bold]/memory reset [project|global|all][/]             — wipe a scope")
        c.print()
        c.print("  [bold]/memory help[/]                                   — show this message")
        c.print()

    # ------------------------------------------------------------------
    # Inspect
    # ------------------------------------------------------------------

    def rules(self, scope: str = "both") -> None:
        """Display stored rules, numbered for easy reference."""
        if self.cortex is None:
            self.console.print("[anton.warning]Memory system not initialized.[/]")
            return
        for label, hc in self._pick_hc(scope):
            self._print_entries(f"{label} — Rules", hc.recall_rules() or "")

    def lessons(self, scope: str = "both") -> None:
        """Display stored lessons, numbered for easy reference."""
        if self.cortex is None:
            self.console.print("[anton.warning]Memory system not initialized.[/]")
            return
        for label, hc in self._pick_hc(scope):
            self._print_entries(f"{label} — Lessons", hc._read_full_lessons() or "")

    def profile(self, scope: str = "both") -> None:
        """Display the identity profile, numbered for easy reference."""
        if self.cortex is None:
            self.console.print("[anton.warning]Memory system not initialized.[/]")
            return
        for label, hc in self._pick_hc(scope):
            self._print_entries(f"{label} — Profile", hc.recall_identity() or "")

    def episodes(self, query: str = "") -> None:
        """Search and display episodic memory logs."""
        if self.episodic is None:
            self.console.print("[anton.warning]Episodic memory not initialized.[/]")
            return
        q = query.strip() or "*"
        formatted = self.episodic.recall_formatted(q, max_results=20)
        self.console.print()
        self.console.print(f"[anton.cyan]Episodes[/] [dim](query: {q!r})[/]")
        self.console.print()
        if formatted:
            self.console.print(formatted)
        else:
            self.console.print("  [dim](no episodes found)[/]")
        self.console.print()

    # ------------------------------------------------------------------
    # Edit / Delete
    # ------------------------------------------------------------------

    def edit(self, args: list[str]) -> None:
        """Open a memory file in $EDITOR."""
        if self.cortex is None:
            self.console.print("[anton.warning]Memory system not initialized.[/]")
            return

        kind = args[0] if args else None
        scope = _parse_scope(args, 1)
        if scope == "both":
            scope = "project"

        file_map = {"rules": "rules.md", "lessons": "lessons.md", "profile": "profile.md"}
        if kind not in file_map:
            self.console.print(
                f"[anton.warning]Unknown kind {kind!r}. Use: rules, lessons, profile[/]"
            )
            return

        _, hc = self._pick_hc(scope)[0]
        path: Path = hc._dir / file_map[kind]
        path.touch()

        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "nano"
        self.console.print(f"[dim]Opening {path} in {editor}…[/]")
        subprocess.call([editor, str(path)])

    def delete(self, args: list[str]) -> None:
        """Delete a specific numbered entry from a memory file.

        Usage: delete rule|lesson|profile <n> [global|project]
        """
        if self.cortex is None:
            self.console.print("[anton.warning]Memory system not initialized.[/]")
            return

        if len(args) < 2:
            self.console.print(
                "[anton.warning]Usage: /memory delete rule|lesson|profile <n> [global|project][/]"
            )
            return

        kind = args[0]
        try:
            n = int(args[1])
        except ValueError:
            self.console.print(
                f"[anton.warning]Entry number must be an integer, got {args[1]!r}[/]"
            )
            return

        scope = _parse_scope(args, 2)
        if scope == "both":
            scope = "project"

        file_map = {"rule": "rules.md", "lesson": "lessons.md", "profile": "profile.md"}
        if kind not in file_map:
            self.console.print(
                f"[anton.warning]Unknown kind {kind!r}. Use: rule, lesson, profile[/]"
            )
            return

        _, hc = self._pick_hc(scope)[0]
        path: Path = hc._dir / file_map[kind]

        if _delete_bullet(path, n):
            self.console.print(f"  Deleted {kind} #{n} from {scope} memory.")
        else:
            self.console.print(
                f"[anton.warning]Entry #{n} not found in {scope} {kind}s.[/]"
            )

    # ------------------------------------------------------------------
    # Maintenance (stubs)
    # ------------------------------------------------------------------

    def prune(self) -> None:
        """Remove outdated or low-value memory entries."""
        pass

    def vacuum(self) -> None:
        """Deduplicate and compact memory storage."""
        pass

    def consolidate(self) -> None:
        """Trigger manual memory consolidation."""
        pass

    def reset(self, scope: str = "all") -> None:
        """Wipe a memory scope ('project', 'global', or 'all')."""
        pass

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def info(self) -> None:
        """Show memory status — read-only dashboard."""
        console = self.console
        console.print()
        console.print("[anton.cyan]Memory Status[/]")
        console.print()

        mode_label = MEMORY_MODES.get(self.settings.memory_mode, self.settings.memory_mode)
        console.print(f"  Mode:  [bold]{mode_label}[/]")
        console.print()

        if self.cortex is None:
            console.print("  [anton.warning]Memory system not initialized.[/]")
            console.print()
            return

        global_total = self._show_scope("Global Memory", self.cortex.global_hc)
        project_total = self._show_scope("Project Memory", self.cortex.project_hc)

        total = global_total + project_total
        console.print(f"  Total entries: [bold]{total}[/]")
        if self.cortex.needs_compaction():
            console.print("  [anton.warning]Compaction needed (>50 entries in a scope)[/]")
        console.print()

        if self.episodic is not None:
            status = "[bold]ON[/]" if self.episodic.enabled else "[dim]OFF[/]"
            sessions = self.episodic.session_count()
            console.print(f"  [anton.cyan]Episodic Memory[/]")
            console.print(f"    Status:    {status}")
            console.print(f"    Sessions:  {sessions}")
            console.print()

        console.print("[dim]  Use [markdown.code]/setup[/] > Memory to change configuration.[/]")
        console.print("[dim]  Run [markdown.code]/memory help[/] for all commands.[/]")
        console.print()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------


    def _show_scope(self, label: str, hc) -> int:
        console = self.console
        identity = hc.recall_identity()
        rules = hc.recall_rules()
        lessons_raw = hc._read_full_lessons()
        rule_count = (
            sum(1 for ln in rules.splitlines() if ln.strip().startswith("- "))
            if rules
            else 0
        )
        lesson_count = (
            sum(1 for ln in lessons_raw.splitlines() if ln.strip().startswith("- "))
            if lessons_raw
            else 0
        )
        topics: list[str] = []
        if hc._topics_dir.is_dir():
            topics = [
                p.stem for p in sorted(hc._topics_dir.iterdir()) if p.suffix == ".md"
            ]

        console.print(f"  [anton.cyan]{label}[/] [dim]({hc._dir})[/]")
        if identity:
            entries = [
                ln.strip()[2:]
                for ln in identity.splitlines()
                if ln.strip().startswith("- ")
            ]
            if entries:
                console.print(
                    f"    Identity:  {', '.join(entries[:3])}"
                    + (" ..." if len(entries) > 3 else "")
                )
            else:
                console.print("    Identity:  [dim](set)[/]")
        else:
            console.print("    Identity:  [dim](empty)[/]")
        console.print(f"    Rules:     {rule_count}")
        console.print(f"    Lessons:   {lesson_count}")
        if topics:
            console.print(f"    Topics:    {', '.join(topics)}")
        else:
            console.print("    Topics:    [dim](none)[/]")
        console.print()
        return rule_count + lesson_count

    def _pick_hc(self, scope: str) -> list[tuple[str, object]]:
        """Return [(label, hc)] for the given scope string."""
        if scope == "global":
            return [("Global", self.cortex.global_hc)]
        if scope == "project":
            return [("Project", self.cortex.project_hc)]
        return [("Global", self.cortex.global_hc), ("Project", self.cortex.project_hc)]

    def _print_entries(self, title: str, raw: str) -> None:
        """Print numbered bullet entries under a heading, grouped by markdown section."""
        c = self.console
        c.print()
        c.print(f"[anton.cyan]{title}[/]")
        c.print()
        entries = _numbered_bullets(raw)
        if not entries:
            c.print("  [dim](empty)[/]")
        else:
            current_section: str | None = None
            for n, section, text in entries:
                if section != current_section:
                    if section:
                        c.print(f"  [bold]{section}[/]")
                    current_section = section
                c.print(f"    [dim]{n:>3}.[/]  {text}")
        c.print()




async def handle_setup_memory(
    console: Console,
    settings: AntonSettings,
    workspace: "Workspace",
    cortex: "Cortex | None",
    episodic: "EpisodicMemory | None" = None,
) -> None:
    """Setup sub-menu: memory mode and episodic memory toggle."""
    console.print()
    console.print("[anton.cyan]Memory configuration[/]")
    console.print()

    console.print("  Memory mode:")
    console.print(
        r"    [bold]1[/]  Autopilot — Anton decides what to remember       [dim]\[recommended][/]"
    )
    console.print(
        r"    [bold]2[/]  Co-pilot — save obvious, confirm ambiguous        [dim]\[selective][/]"
    )
    console.print(
        r"    [bold]3[/]  Off — never save memory (still reads existing)    [dim]\[suppressed][/]"
    )
    console.print()

    mode_map = {"1": "autopilot", "2": "copilot", "3": "off"}
    current_mode_num = {"autopilot": "1", "copilot": "2", "off": "3"}.get(
        settings.memory_mode, "1"
    )
    mode_choice = await prompt_or_cancel(
        "(anton) Memory mode", choices=["1", "2", "3"], default=current_mode_num
    )
    if mode_choice is None:
        console.print()
        return
    memory_mode = mode_map[mode_choice]
    settings.memory_mode = memory_mode
    workspace.set_secret("ANTON_MEMORY_MODE", memory_mode)
    if cortex is not None:
        cortex.mode = memory_mode

    if episodic is not None:
        console.print()
        ep_status = "ON" if episodic.enabled else "OFF"
        console.print(
            f"  Episodic memory (conversation archive): Currently [bold]{ep_status}[/]"
        )
        toggle = await prompt_or_cancel(
            "(anton) Toggle episodic memory?", choices=["y", "n"], default="n"
        )
        if toggle is None:
            toggle = "n"
        if toggle == "y":
            new_state = not episodic.enabled
            episodic.enabled = new_state
            settings.episodic_memory = new_state
            workspace.set_secret(
                "ANTON_EPISODIC_MEMORY", "true" if new_state else "false"
            )
            console.print(f"  Episodic memory: [bold]{'ON' if new_state else 'OFF'}[/]")

    console.print()
    console.print("[anton.success]Configuration updated.[/]")
    console.print()