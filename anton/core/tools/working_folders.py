"""Folders a host lets the agent pick and attach files from, beside the workspace.

Only a host building a session can name them (``ChatSessionConfig.working_folders``);
no setting or environment variable reaches this list. The host vouches for each
folder, so this module only checks that an entry is a real, absolute directory.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def normalize_working_folders(paths: Iterable[Path | str] | None) -> tuple[Path, ...]:
    """The usable folders among `paths`, resolved, in order and without repeats.

    An entry that is not absolute, does not exist, or is not a directory is
    dropped and logged: a relative entry would otherwise resolve against the
    process's working directory, a root the host never named.
    """
    folders: list[Path] = []
    for raw in paths or ():
        path = Path(raw)
        if not path.is_absolute():
            _log.warning("Ignoring working folder %s: not an absolute path", path)
            continue
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError):
            _log.warning("Ignoring working folder %s: it does not resolve", path, exc_info=True)
            continue
        if not resolved.is_dir():
            _log.warning("Ignoring working folder %s: not a directory", path)
            continue
        if resolved not in folders:
            folders.append(resolved)
    return tuple(folders)


def working_folder_roots(session: Any) -> tuple[Path, ...]:
    """The session's working folders, or none for a session that has no such list.

    Only a real sequence of paths counts, so a mock or a differently shaped
    session reads as having no working folders, never as an open fence.
    """
    value = getattr(session, "_working_folders", ())
    if not isinstance(value, (tuple, list)):
        return ()
    return tuple(path for path in value if isinstance(path, Path))
