"""Directory snapshot + diff helpers for per-turn artifact tracking.

A `DirSnapshot` is a `{relative_path: (mtime, size)}` map captured
before a conversation turn fires. After the turn returns we diff
against a fresh snapshot — every entry that's new or changed marks
its parent artifact folder as "touched this turn" so the store can
append a provenance record.

The shape is intentionally minimal — no hashing, no inode tracking.
Files only change when their mtime or size changes, and that's
enough signal for "did this turn touch this file?". False positives
(e.g. `touch` without a real edit) are vanishingly rare in practice
and only result in an extra benign provenance entry.

Symlinks and non-regular files are ignored so a stray socket or
named pipe under `artifacts/` can't trip the snapshotter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

# {relative_path -> (mtime_ns, size_bytes)}
# Using mtime_ns (not float seconds) avoids the rounding artefacts
# that bite on filesystems with sub-second resolution.
DirSnapshot = Dict[str, tuple[int, int]]


def snapshot_dir(root: Path) -> DirSnapshot:
    """Return a `{relative_path -> (mtime_ns, size)}` map for every
    regular file under `root`, recursively. Returns an empty dict
    when `root` doesn't exist (which is fine — diffing an empty
    snapshot against a populated one yields the full add list).
    """
    out: DirSnapshot = {}
    if not root.exists() or not root.is_dir():
        return out
    for p in root.rglob("*"):
        try:
            if not p.is_file() or p.is_symlink():
                continue
            stat = p.stat()
        except (OSError, FileNotFoundError):
            # Race against a concurrent write — skip the file. The
            # next snapshot pass will pick it up.
            continue
        rel = str(p.relative_to(root))
        out[rel] = (stat.st_mtime_ns, stat.st_size)
    return out


def diff_snapshots(before: DirSnapshot, after: DirSnapshot) -> list[str]:
    """Relative paths that are present-and-different (or new) in
    `after` vs `before`. Deletions are NOT returned — provenance
    tracks file *creation/modification*, not removal.

    Result order is sorted for deterministic provenance entries.
    """
    changed: list[str] = []
    for rel, sig_after in after.items():
        sig_before = before.get(rel)
        if sig_before is None or sig_before != sig_after:
            changed.append(rel)
    changed.sort()
    return changed


def _files_by_artifact(rels: Iterable[str]) -> Dict[str, list[str]]:
    """Group changed file paths by their top-level artifact folder.

    A path under `artifacts/dashboard/data/prices.csv` belongs to
    the `dashboard` artifact; the function returns:
        {"dashboard": ["data/prices.csv"]}

    Files at the top level of the artifacts root (no subfolder)
    are skipped — those are user-stashed files that don't belong
    to any artifact.
    """
    grouped: Dict[str, list[str]] = {}
    for rel in rels:
        parts = rel.split("/", 1)
        if len(parts) < 2:
            # Top-level file directly under artifacts/ — not part
            # of any artifact subfolder, ignore.
            continue
        slug, inner = parts
        grouped.setdefault(slug, []).append(inner)
    return grouped
