"""Workspace initialization and management for Anton.

Handles:
- anton.md creation and reading (project context file)
- .env secret vault (store secrets without passing through LLM)
- Non-empty folder detection and user confirmation
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

ANTON_MD_TEMPLATE = """\
# Anton Workspace

Created: {date}

<!-- Add project context, conventions, and notes below.
     Anton reads this file at the start of every conversation. -->
"""


class Workspace:
    """Manages the .anton/ workspace directory and its files."""

    def __init__(self, base: Path) -> None:
        self._base = base
        self._anton_dir = base / ".anton"
        self._anton_md = self._anton_dir / "anton.md"
        self._env_file = self._anton_dir / ".env"
        self._anton_md_last_read: datetime | None = None

    @property
    def base(self) -> Path:
        return self._base

    @property
    def anton_md_path(self) -> Path:
        return self._anton_md

    @property
    def env_path(self) -> Path:
        return self._env_file

    # ── Folder state checks ──────────────────────────────────────

    def is_initialized(self) -> bool:
        """Check if this workspace has been initialized (anton.md exists)."""
        return self._anton_md.is_file()

    def has_non_anton_files(self) -> bool:
        """Check if the folder contains files that aren't part of Anton."""
        if not self._base.exists():
            return False
        for item in self._base.iterdir():
            name = item.name
            # Skip Anton's own files/dirs
            if name in (".anton", ".env"):
                continue
            # Skip common hidden files
            if name.startswith("."):
                continue
            return True
        return False

    def needs_confirmation(self) -> bool:
        """Check if the user should confirm before initializing.

        Returns True if the folder is non-empty and doesn't have anton.md.
        """
        return not self.is_initialized() and self.has_non_anton_files()

    # ── Initialization ───────────────────────────────────────────

    def initialize(self) -> list[str]:
        """Create the workspace structure. Returns list of actions taken."""
        actions: list[str] = []

        # Create .anton/ directory and memory subdirectory
        self._anton_dir.mkdir(parents=True, exist_ok=True)
        (self._anton_dir / "memory").mkdir(exist_ok=True)
        actions.append(f"Created {self._anton_dir}")

        # Create anton.md if it doesn't exist
        if not self._anton_md.is_file():
            self._anton_md.write_text(
                ANTON_MD_TEMPLATE.format(date=datetime.now().strftime("%Y-%m-%d"))
            )
            actions.append(f"Created {self._anton_md}")

        # Create .env if it doesn't exist
        if not self._env_file.is_file():
            self._env_file.write_text("# Anton environment variables\n")
            actions.append(f"Created {self._env_file}")

        return actions

    # ── anton.md reading ─────────────────────────────────────────

    def read_anton_md(self) -> str | None:
        """Read anton.md content. Returns None if it doesn't exist."""
        if not self._anton_md.is_file():
            return None
        return self._anton_md.read_text()

    def anton_md_modified_since_last_read(self) -> bool:
        """Check if anton.md has been modified since last read_anton_md_tracked()."""
        if not self._anton_md.is_file():
            return False
        mtime = datetime.fromtimestamp(self._anton_md.stat().st_mtime)
        if self._anton_md_last_read is None:
            return True
        return mtime > self._anton_md_last_read

    def read_anton_md_tracked(self) -> str | None:
        """Read anton.md and track the read timestamp."""
        content = self.read_anton_md()
        if content is not None:
            self._anton_md_last_read = datetime.now()
        return content

    def build_anton_md_context(self) -> str:
        """Build a prompt section from anton.md content, if any."""
        content = self.read_anton_md_tracked()
        if not content or not content.strip():
            return ""

        return (
            "\n\n## Project Context (anton.md)\n"
            "The following was written by the user in .anton/anton.md:\n\n"
            f"{content.strip()}\n"
        )

    # ── Secret vault (.env management) ───────────────────────────

    def load_env(self) -> dict[str, str]:
        """Load all variables from .anton/.env. Returns key=value dict."""
        result: dict[str, str] = {}
        if not self._env_file.is_file():
            return result
        for line in self._env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip()
        return result

    def get_secret(self, key: str) -> str | None:
        """Get a specific secret from .anton/.env."""
        env = self.load_env()
        return env.get(key)

    def has_secret(self, key: str) -> bool:
        """Check if a secret exists in .anton/.env."""
        return self.get_secret(key) is not None

    def set_secret(self, key: str, value: str) -> None:
        """Store a secret in .anton/.env without passing it through the LLM.

        The value is written directly to the .env file, and the
        environment variable is set in the current process.
        """
        self._anton_dir.mkdir(parents=True, exist_ok=True)

        # Read existing lines
        lines: list[str] = []
        replaced = False
        if self._env_file.is_file():
            for line in self._env_file.read_text().splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    existing_key = stripped.partition("=")[0].strip()
                    if existing_key == key:
                        lines.append(f"{key}={value}")
                        replaced = True
                        continue
                lines.append(line)

        if not replaced:
            lines.append(f"{key}={value}")

        self._env_file.write_text("\n".join(lines) + "\n")

        # Also set in current process environment
        os.environ[key] = value

    def remove_secret(self, key: str) -> bool:
        """Remove a secret from .anton/.env.

        Returns True if the key was found and removed, False otherwise.
        """
        if not self._env_file.is_file():
            return False

        lines: list[str] = []
        found = False
        for line in self._env_file.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                existing_key = stripped.partition("=")[0].strip()
                if existing_key == key:
                    found = True
                    continue
            lines.append(line)

        if found:
            self._env_file.write_text("\n".join(lines) + "\n")
            os.environ.pop(key, None)

        return found

    def apply_env_to_process(self) -> int:
        """Load .anton/.env variables into os.environ. Returns count loaded."""
        env = self.load_env()
        count = 0
        for key, value in env.items():
            if key not in os.environ:
                os.environ[key] = value
                count += 1
        return count
