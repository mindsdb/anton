from __future__ import annotations

import os
import subprocess
from pathlib import Path

from prompt_toolkit import PromptSession

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
            "identity":     self.identity,
            "episodes":    self.episodes,
            "vacuum": self.vacuum,
            "reset": self.reset
        }

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def handle(self, cmd: str) -> None:
        """Dispatch /memory [sub-command] or show the status dashboard."""
        sub_cmd = cmd.removeprefix("/memory").strip()
        parts = sub_cmd.split()

        if len(parts) == 0:
            return self.info()

        if self.cortex is None:
            self.console.print("  [anton.warning]Memory system not initialized.[/]")
            self.console.print()
            return

        if parts[0] not in self.SUBCOMMANDS:
            c = self.console
            c.print()
            c.print(f"[anton.error]Unknown command: {cmd}[/]")
            c.print()
            return await self.help()

        handler = self.SUBCOMMANDS[parts[0]]

        return await handler(*parts[1:])

    async def help(self) -> None:
        """Show available /memory sub-commands."""
        c = self.console
        c.print()
        c.print("[anton.cyan]Memory commands[/]")
        c.print()
        c.print("  [bold]/memory[/]                               — status dashboard")
        c.print()
        c.print("  [bold dim]Inspect[/]")
        c.print("  [bold]/memory rules[/]                         — show behavioral rules")
        c.print("  [bold]/memory rules delete <n>[/]              — delete rule #n")
        c.print("  [bold]/memory rules edit <n>[/]                — edit rule #n")
        c.print()
        c.print("  [bold]/memory lessons[/]                       — show learned lessons")
        c.print("  [bold]/memory lessons delete <n>[/]            — delete lesson #n")
        c.print("  [bold]/memory lessons edit <n>[/]              — edit lesson #n")
        c.print()
        c.print("  [bold]/memory identity[/]                      — show identity profile")
        c.print("  [bold]/memory identity delete <n>[/]           — delete identity entry #n")
        c.print("  [bold]/memory identity edit <n>[/]             — edit identity entry #n")
        c.print()
        c.print("  [bold]/memory episodes[/]                      — show episodic sessions")
        c.print("  [bold]/memory episodes delete <n>[/]           — delete session #n")
        c.print()
        c.print("  [bold dim]Maintenance[/]")
        c.print("  [bold]/memory vacuum[/]                        — deduplicate and compact")
        c.print("  [bold]/memory reset global|project|episodic[/] — wipe a scope")
        c.print()
        c.print("  [bold]/memory help[/]                          — show this message")
        c.print()

    # ------------------------------------------------------------------
    # Inspect
    # ------------------------------------------------------------------

    async def _handle_menu_command(self, action, num, global_items, project_items, methods: dict = None):
        index = {
            **global_items,
            **project_items,
        }
        nums = list(index.keys())

        if not nums:
            return self.console.print("Nothing to act on — memory is empty.")

        if num is None:
            return self.console.print(f"Choose item to {action}: {min(nums)}-{max(nums)}")

        if num.isdigit():
            num = int(num)
        if num not in index:
            return self.console.print(f"Item {num} not found, choose number between {min(nums)} and {max(nums)}")

        if action not in methods.keys():
            return self.console.print(f"Unknown action use one of: {','.join(methods.keys())}")

        id = index[num].id
        if num in global_items:
            method = getattr(self.cortex.global_hc, methods[action])
        else:
            method = getattr(self.cortex.project_hc, methods[action])

        if action == 'delete':
            method(id)
            self.console.print("[anton.cyan]Deleted[/]")

        elif action == 'edit':
            text = await prompt_or_cancel("Edit the text>", default_text=index[num].text)

            if text is None:
                return False

            method(id, text)
            self.console.print("[anton.cyan]Updated[/]")
        return True

    async def rules(self, action: str = None, num: str = None) -> None:
        """Display stored rules, numbered for easy reference."""
        global_items = dict(enumerate(self.cortex.global_hc.get_rules(), start=1))
        project_items = dict(enumerate(self.cortex.project_hc.get_rules(), start=len(global_items) + 1))

        if action is not None:
            updated = await self._handle_menu_command(action, num, global_items, project_items, methods={
                'delete': 'del_rule',
                'edit': 'update_rule',
            })
            if updated:
                return await self.rules()
            return

        for scope_title, items in [("Global", global_items), ("Project", project_items)]:
            if not items:
                continue
            self._print_title(scope_title)
            prev_kind = None
            for n, engram in items.items():
                if engram.kind != prev_kind:
                    self.console.print(f"  [dim]{engram.kind}[/]")
                    prev_kind = engram.kind
                self._print_numbered_item(n, engram)

        self.console.print("  [bold]/memory rules delete <n>[/]  — delete rule #n")
        self.console.print("  [bold]/memory rules edit <n>[/]    — edit rule #n")

    async def lessons(self, action: str = None, num: str = None) -> None:
        """Display stored lessons, numbered for easy reference."""
        global_items = dict(enumerate(self.cortex.global_hc.get_lessons(), start=1))
        project_items = dict(enumerate(self.cortex.project_hc.get_lessons(), start=len(global_items) + 1))

        if action is not None:
            updated = await self._handle_menu_command(action, num, global_items, project_items, methods={
                'delete': 'del_lesson',
                'edit': 'update_lesson',
            })
            if updated:
                return await self.lessons()
            return

        for scope_title, items in [("Global", global_items), ("Project", project_items)]:
            if not items:
                continue
            self._print_title(scope_title)
            for i, item in items.items():
                self._print_numbered_item(i, item)

        self.console.print("  [bold]/memory lessons delete <n>[/]  — delete lesson #n")
        self.console.print("  [bold]/memory lessons edit <n>[/]    — edit lesson #n")


    async def identity(self, action: str = None, num: str = None) -> None:
        """Display the identity profile, numbered for easy reference."""
        global_items = dict(enumerate(self.cortex.global_hc.get_identities(), start=1))
        project_items = dict(enumerate(self.cortex.project_hc.get_identities(), start=len(global_items) + 1))

        if action is not None:
            updated = await self._handle_menu_command(action, num, global_items, project_items, methods={
                'delete': 'del_identity',
                'edit': 'update_identity',
            })
            if updated:
                return await self.identity()
            return

        for scope_title, items in [("Global", global_items), ("Project", project_items)]:
            if not items:
                continue
            self._print_title(scope_title)
            for i, item in items.items():
                self._print_numbered_item(i, item)

        self.console.print("  [bold]/memory identity delete <n>[/]  — delete identity entry #n")
        self.console.print("  [bold]/memory identity edit <n>[/]    — edit identity entry #n")

    async def episodes(self, action: str = None, num: str = None) -> None:
        if self.episodic is None:
            self.console.print("[anton.warning]Episodic memory not initialized.[/]")
            return

        items = dict(enumerate(self.episodic.get_episodes(), start=1))

        if action is not None:
            nums = list(items.keys())

            if not nums:
                return self.console.print("Nothing to act on — no episodes.")

            if num is None:
                return self.console.print(f"Choose item to {action}: {min(nums)}-{max(nums)}")

            if num.isdigit():
                num = int(num)
            if num not in items:
                return self.console.print(f"Item {num} not found, choose number between {min(nums)} and {max(nums)}")

            if action == 'delete':
                session_id = items[num].session
                self.episodic.del_episode(session_id)
                self.console.print("[anton.cyan]Deleted[/]")
                return await self.episodes()
            else:
                return self.console.print(f"Unknown action: {action}")

        self._print_title("Episodic Memory")
        max_shown_items = 50
        if len(items) > max_shown_items:
            # keep only last items
            items = dict(sorted(items.items())[-max_shown_items:])
            self.console.print(f"Only the last {max_shown_items} are shown:")
        if len(items) == 0:
            print("(no items)")
        for i, item in items.items():
            content = item.content.replace('\n', ' ')
            if len(content) > 100:
                content = content[:97] + "..."
            self.console.print(f"    [dim]{i:>3}.[/]  {content}")

        self.console.print("  [bold]/memory episodes delete <n>[/]  — delete session #n")

    async def vacuum(self):
        toggle = await prompt_or_cancel(
            "(anton) Are you sure to vacuum memory?", choices=["y", "n"], default="n"
        )
        if toggle == "y":
            await self.cortex.vacuum()
            self.console.print(f"Vacuum finished")
        else:
            self.console.print(f"Canceled")

    async def reset(self, scope: str = None) -> None:

        scopes = ['global', 'project', 'episodic']

        self.console.print("Clearing memory")
        if scope is None:
            return self.console.print(f"Choose a scope to reset memory: {', '.join(scopes)}")

        if scope not in scopes:
            return self.console.print(f"Unknown scope: {scope}, choose one of: {', '.join(scopes)}")

        toggle = await prompt_or_cancel(
            f"(anton) Are you sure to reset {scope} memory? Type 'reset' to confirm",
            choices=["reset", "n"], default="n"
        )
        if toggle != "reset":
            return self.console.print(f"Canceled")

        if scope == 'global':
            self.cortex.global_hc.clear()
        if scope == 'project':
            self.cortex.project_hc.clear()
        if scope == 'episodic':
            self.episodic.clear()

        return self.console.print(f"Reset finished: {scope}")

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

        global_total = self._info_scope("Global Memory", self.cortex.global_hc)
        project_total = self._info_scope("Project Memory", self.cortex.project_hc)

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

    def _info_scope(self, label: str, hc) -> int:
        console = self.console
        identities = [entry.text for entry in hc.get_identities()]
        rule_count = len(hc.get_rules())
        lessons = hc.get_lessons()
        topics = {
            entry.topic
            for entry in lessons
            if entry.topic
        }

        console.print(f"  [anton.cyan]{label}[/] [dim]({hc._dir})[/]")
        if identities:
            console.print(
                f"    Identity:  {', '.join(identities[:3])}"
                + (" ..." if len(identities) > 3 else "")
            )
        else:
            console.print("    Identity:  [dim](empty)[/]")
        console.print(f"    Rules:     {rule_count}")
        console.print(f"    Lessons:   {len(lessons)}")
        if topics:
            console.print(f"    Topics:    {', '.join(topics)}")
        else:
            console.print("    Topics:    [dim](none)[/]")
        console.print()
        return rule_count + len(lessons)

    def _print_title(self, title: str) -> None:
        c = self.console
        c.print()
        c.print(f"[anton.cyan]{title}[/]")

    def _print_numbered_item(self, n, entry) -> None:
        self.console.print(f"    [dim]{n:>3}.[/]  {entry.text}")
        meta_parts = []
        for field in ("confidence", "source", "topic"):
            val = getattr(entry, field, None)
            if val is not None and val not in ("medium", "llm"):
                meta_parts.append(f"{field}:{val}")
        if (ua := getattr(entry, "updated_at", None)) is not None:
            meta_parts.append(f"updated:{ua.strftime('%Y-%m-%d')}")
        if meta_parts:
            self.console.print(f"         [anton.cyan]{' '.join(meta_parts)}[/]")


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