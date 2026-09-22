"""Files the user attached to the conversation, carried into the pipeline.

Until 2026-09-18 the pipeline could not see them at all (S-01): the paths
an upload gets (`.cowork/files/<uuid>/<name>` in the app, `.anton/uploads/`
for a CLI paste) live only in the outer conversation, the generator's
`read_file` is fenced to the artifact folder, and the html-app task forbids
referencing any other file of the artifact. A request over "this CSV" or
"put my logo on it" had no path into the run.

Two kinds, handled differently:

- DATA files (csv, json, xlsx, ...) are read where every other source is
  read — by the gathering step, in the scratchpad, from the absolute path
  this module lists in the kickoff. Their content reaches the generators the
  usual way, as `## Data` notes. They are never copied into the artifact:
  an html-app embeds the aggregated data, and a published copy of the raw
  file would be dead weight (or worse, more than the user meant to share).
- ASSETS (images, fonts, anything else) are copied into the artifact folder
  by the orchestrator right before generation — `static/` for a fullstack
  app, the root for an html-app — and the generators are told the relative
  name to reference. The copy is a deterministic step, not a model action:
  a model cannot emit a binary file, and base64 in a reply is exactly the
  size problem the write protocol exists to avoid.

Which paths are accepted is decided here too, and strictly: the path comes
from the calling model, and a wrong one is not a failed run but a leak. A
path like `.anton/.env` has no data suffix, so without a fence it would be
copied into the artifact folder and published with it; `.anton/secrets.json`
would be read by the gathering step and its content carried into the PRD.
`accept_reason` below says what qualifies — an upload the host staged, or a
plain file inside the workspace outside any dot-directory. Everything else
is dropped with the reason, for both kinds of file.
"""

from __future__ import annotations

import mimetypes
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

# Suffixes the gathering step reads as data rather than ships as an asset.
DATA_SUFFIXES: frozenset[str] = frozenset({
    ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xlsx", ".xls", ".parquet",
    ".txt", ".md",
})

# An asset above this is not copied: the artifact folder is what gets
# published, and the browser gate loads the page from disk in 8 seconds.
ASSET_MAX_BYTES = 15 * 1024 * 1024

KIND_DATA = "data"
KIND_ASSET = "asset"

ATTACHMENTS_HEADER = "## Attached files"

# Where each host stages a conversation upload, relative to the workspace:
# the CLI's clipboard/paste directory and the cloud turn's shared mount.
# The cowork app keeps its uploads OUTSIDE the workspace, under
# `<COWORK_HOME>/files/<uuid>/<name>` where COWORK_HOME is `~/.cowork` or a
# `~/.cowork-<kind>` dev variant — matched by that layout instead.
UPLOAD_ROOTS_IN_WORKSPACE: tuple[tuple[str, ...], ...] = (
    (".anton", "uploads"),
    ("attachments",),
)
_COWORK_HOME_PREFIX = ".cowork"
_COWORK_FILES_DIRNAME = "files"

REFUSED_OUTSIDE = "outside the workspace and not a conversation upload"
REFUSED_HIDDEN = "a dot-file or inside a dot-directory, which only a conversation upload may be"


@dataclass
class Attachment:
    path: str  # absolute source path, as the conversation showed it
    name: str
    size: int
    kind: str
    # Relative path inside the artifact once an asset has been copied; ""
    # for data files and for assets the staging step skipped.
    staged: str = ""
    # Why staging skipped this one, when it did.
    skipped: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: object) -> "Attachment | None":
        if not isinstance(raw, dict):
            return None
        try:
            return cls(
                path=str(raw["path"]), name=str(raw["name"]),
                size=int(raw.get("size", 0)), kind=str(raw.get("kind", KIND_ASSET)),
                staged=str(raw.get("staged", "")), skipped=str(raw.get("skipped", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None


def _kind(path: Path) -> str:
    return KIND_DATA if path.suffix.lower() in DATA_SUFFIXES else KIND_ASSET


def _is_cowork_upload(resolved: Path) -> bool:
    """`.../.cowork[-<kind>]/files/<...>/<name>` — the cowork app's upload
    store, which lives outside any workspace."""
    parts = resolved.parts
    for i in range(len(parts) - 2):
        if parts[i].startswith(_COWORK_HOME_PREFIX) and parts[i + 1] == _COWORK_FILES_DIRNAME:
            return True
    return False


def refusal_reason(resolved: Path, workspace: Path | None) -> str | None:
    """Why a RESOLVED file path may not be attached, or None when it may.

    Checked on the resolved path so a symlink inside the workspace cannot
    point the copy at something outside it. Three ways in:

    - a cowork upload, by layout (see `_is_cowork_upload`);
    - a file under one of `UPLOAD_ROOTS_IN_WORKSPACE`;
    - a file inside the workspace with no dot-component in its relative
      path — a user's own project file, never `.anton/`, `.git/`, `.env`.

    Without a workspace only the first applies: there is nothing to be
    "inside" of, and guessing would be the leak this function exists to
    stop.
    """
    if _is_cowork_upload(resolved):
        return None
    if workspace is None:
        return REFUSED_OUTSIDE
    try:
        rel = resolved.relative_to(workspace.resolve())
    except (ValueError, OSError):
        return REFUSED_OUTSIDE
    for root in UPLOAD_ROOTS_IN_WORKSPACE:
        if rel.parts[: len(root)] == root and len(rel.parts) > len(root):
            return None
    if any(part.startswith(".") for part in rel.parts):
        return REFUSED_HIDDEN
    return None


def resolve_attachments(
    paths, *, workspace: Path | None = None
) -> tuple[list[Attachment], list[str]]:
    """Turn the tool's `attachments` argument into records, dropping what is
    not a readable, attachable file. Returns (kept, reasons for the dropped
    ones).

    `workspace` is the session's workspace base; `refusal_reason` says which
    paths pass. Duplicates collapse on the resolved path. Nothing here
    raises: a wrong path is the calling agent's mistake, recorded in the
    trace, and must not cost the run.
    """
    kept: list[Attachment] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for raw in paths or ():
        text = str(raw or "").strip()
        if not text:
            continue
        p = Path(text).expanduser()
        try:
            resolved = p.resolve()
            if not p.is_file():
                dropped.append(f"{text}: not a file")
                continue
            size = p.stat().st_size
        except OSError as exc:
            dropped.append(f"{text}: {exc.__class__.__name__}")
            continue
        reason = refusal_reason(resolved, workspace)
        if reason is not None:
            dropped.append(f"{text}: {reason}")
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        kept.append(Attachment(path=key, name=p.name, size=size, kind=_kind(p)))
    return kept, dropped


def stage_assets(attachments: list[Attachment], artifact_path: Path, *, static: bool) -> list[str]:
    """Copy every asset into the artifact (`static/` when `static`), return
    the relative paths written. Data files are left where they are.

    Idempotent: a retry or a resumed run copies the same source over the same
    name. Oversized assets are skipped with the reason on the record, so the
    generator is not told to reference a file that is not there.
    """
    dest_dir = artifact_path / "static" if static else artifact_path
    written: list[str] = []
    for att in attachments:
        if att.kind != KIND_ASSET:
            continue
        if att.size > ASSET_MAX_BYTES:
            att.staged = ""
            att.skipped = f"larger than {ASSET_MAX_BYTES // (1024 * 1024)} MB, not copied"
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(att.path, dest_dir / att.name)
        except OSError as exc:
            att.staged = ""
            att.skipped = f"copy failed: {exc.__class__.__name__}"
            continue
        att.staged = (dest_dir / att.name).relative_to(artifact_path).as_posix()
        att.skipped = ""
        written.append(att.staged)
    return written


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def render_for_gathering(attachments: list[Attachment]) -> str:
    """The `## Attached files` section of the gathering kickoff, or ""."""
    if not attachments:
        return ""
    lines = [ATTACHMENTS_HEADER]
    for att in attachments:
        mime = mimetypes.guess_type(att.name)[0] or "unknown type"
        role = (
            "data file — read it in the scratchpad from this path in this step"
            if att.kind == KIND_DATA
            else "asset — copied into the artifact folder before generation; "
            "the page will reference it by file name"
        )
        lines.append(f"- `{att.path}` ({mime}, {_human_size(att.size)}): {role}")
    return "\n".join(lines)


def render_for_generation(attachments: list[Attachment]) -> str:
    """The `## Attached files` section of the generation kickoff, or "".

    Lists what the page may reference (staged assets, by relative name) and
    what it may not (data files, which arrive as `## Data`; assets the staging
    step skipped). The relative name is the same whether the page is opened
    from disk or served: for a fullstack app both the page and the asset sit
    in `static/`.
    """
    if not attachments:
        return ""
    lines = [ATTACHMENTS_HEADER]
    for att in attachments:
        mime = mimetypes.guess_type(att.name)[0] or "unknown type"
        if att.kind == KIND_DATA:
            lines.append(
                f"- `{att.name}` ({mime}): data file, read during gathering — "
                "use its content from `## Data`; it is not in the artifact folder."
            )
        elif att.staged:
            lines.append(
                f"- `{att.name}` ({mime}, {_human_size(att.size)}): already in the "
                f"artifact folder next to the page — reference it as `{att.name}` "
                f'(for example `<img src="{att.name}">`).'
            )
        else:
            lines.append(
                f"- `{att.name}` ({mime}): NOT available — {att.skipped or 'not copied'}. "
                "Do not reference it."
            )
    return "\n".join(lines)


__all__ = [
    "ASSET_MAX_BYTES", "ATTACHMENTS_HEADER", "Attachment", "DATA_SUFFIXES",
    "KIND_ASSET", "KIND_DATA", "REFUSED_HIDDEN", "REFUSED_OUTSIDE",
    "UPLOAD_ROOTS_IN_WORKSPACE", "refusal_reason", "render_for_gathering",
    "render_for_generation", "resolve_attachments", "stage_assets",
]
