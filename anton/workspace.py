"""Workspace initialization and management for Anton.

Handles:
- anton.md creation and reading (project context file)
- .env secret vault (store secrets without passing through LLM)
- Non-empty folder detection and user confirmation
"""

from __future__ import annotations

import errno
import os
from datetime import datetime
from pathlib import Path

from anton.config.settings import AntonSettings

ANTON_MD_TEMPLATE = """\
# Anton Workspace

Created: {date}

<!-- Add project context, conventions, and notes below.
     Anton reads this file at the start of every conversation. -->
"""


_SECRET_FILE_MODE = 0o600


def _write_private(path: Path, text: str) -> None:
    """Write `text` to `path`, readable and writable only by its owner.

    The file holds secrets, so it is created 0600 rather than under the
    process umask. An existing file is tightened before the write, so a
    mode an earlier version left behind does not outlive the next secret.
    UTF-8, not the host locale, to match how the file is read back.
    """
    if path.exists():
        os.chmod(path, _SECRET_FILE_MODE)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _SECRET_FILE_MODE)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


class Workspace:
    """Manages the .anton/ workspace directory and its files."""

    def __init__(self, base: Path, settings: AntonSettings | None = None) -> None:
        self._base = base
        self._anton_dir = base / ".anton"
        self._anton_md = self._anton_dir / "anton.md"
        self._env_file = self._anton_dir / ".env"
        self._anton_md_last_read: datetime | None = None

        # Reuse a caller's settings so the cloud pod's dotenv-disabled
        # AntonSettings isn't bypassed by a second one built here. Only
        # `artifacts_dir` is read; None = desktop behaviour unchanged.
        settings = settings or AntonSettings()
        self._artifacts_dir = self._anton_dir / settings.artifacts_dir

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

    def initialize(self, *, create_anton_md: bool = True) -> list[str]:
        """Create the workspace structure. Returns list of actions taken.

        Retries once on ESTALE: on a shared NFS/EFS workspace another
        client (cowork-server re-staging `.anton/anton.md`, or a workspace
        wipe) can delete an inode this pod still has cached, so the first
        stat fails with a stale handle. That failed stat drops the cached
        dentry, so a second pass does a fresh lookup and succeeds.

        ``create_anton_md=False`` is the cloud mode (ENG-1817): there
        ``.anton/anton.md`` is owned by cowork-server, which stages the
        project's instructions into it before every turn and clears it when
        they are removed. A template written by the pod would be treated as
        a stale staged copy and deleted — and under gVisor the pod then
        keeps hitting the dead inode via its cached handle (ESTALE), which
        the retry above cannot recover. Not creating the file removes the
        ownership fight entirely; skipping the stat also removes the one
        ESTALE-prone call from the cloud turn's critical path.
        """
        try:
            return self._initialize_once(create_anton_md=create_anton_md)
        except OSError as exc:
            if exc.errno != errno.ESTALE:
                raise
            return self._initialize_once(create_anton_md=create_anton_md)

    def _initialize_once(self, *, create_anton_md: bool = True) -> list[str]:
        actions: list[str] = []

        # Create .anton/ directory and memory subdirectory
        self._anton_dir.mkdir(parents=True, exist_ok=True)
        (self._anton_dir / "memory").mkdir(exist_ok=True)
        actions.append(f"Created {self._anton_dir}")

        # Create anton.md if it doesn't exist
        if create_anton_md and not self._anton_md.is_file():
            self._anton_md.write_text(
                ANTON_MD_TEMPLATE.format(date=datetime.now().strftime("%Y-%m-%d")),
                encoding="utf-8",
            )
            actions.append(f"Created {self._anton_md}")

        # Create .env if it doesn't exist
        if not self._env_file.is_file():
            _write_private(self._env_file, "# Anton environment variables\n")
            actions.append(f"Created {self._env_file}")

        # Visible artifacts directory at the workspace root. Replaces
        # the legacy hidden `.anton/output/` dump — one folder per
        # artifact, each owning its own metadata.json + README.md.
        # Idempotent: existing artifact subfolders are left alone.
        if not self._artifacts_dir.exists():
            self._artifacts_dir.mkdir(parents=True, exist_ok=True)
            actions.append(f"Created {self._artifacts_dir}")

        return actions

    @property
    def artifacts_dir(self) -> Path:
        """Where artifacts live. Created lazily by `initialize()`."""
        return self._artifacts_dir

    # ── anton.md reading ─────────────────────────────────────────

    def read_anton_md(self) -> str | None:
        """Read anton.md content. Returns None if it doesn't exist."""
        if not self._anton_md.is_file():
            return None
        return self._anton_md.read_text(encoding="utf-8")

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
        for line in self._env_file.read_text(encoding="utf-8").splitlines():
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
            for line in self._env_file.read_text(encoding="utf-8").splitlines():
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

        _write_private(self._env_file, "\n".join(lines) + "\n")

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
        for line in self._env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                existing_key = stripped.partition("=")[0].strip()
                if existing_key == key:
                    found = True
                    continue
            lines.append(line)

        if found:
            _write_private(self._env_file, "\n".join(lines) + "\n")
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
