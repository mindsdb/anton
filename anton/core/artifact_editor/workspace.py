"""The artifact folder the artifact editor reads and writes through.

The editor's model sees an artifact as a handful of text files and changes
them with a diff. This module is the only door between that model and the
folder, and every rule about what the model may see or touch lives here:

* **What is content.** An artifact folder also holds the store's
  housekeeping (`metadata.json`, `.revisions/`, the backend log), the
  generation pipeline's inputs (`prd.md`, `observed-data.json`, ...), Python
  bytecode, dependency folders and binaries. None of it is what the user
  means by "the artifact", so `map()` hides it and `refuse_reason()` refuses
  to edit it. The names come from `anton.core.artifacts.internal_files`, the
  same definitions `ArtifactStore` uses to build an artifact's `files[]`.
* **What each kind of artifact may contain.** An `html-app` is one
  self-contained page: the preview serves that file alone, and the render
  check fails a page that loads a sibling (`separate_file`). A fullstack app
  has a fixed layout: `backend.py` and `requirements.txt` at the root (plus
  `state_manifest.json` for the stateful kind) and every frontend file under
  `static/`, the only folder the backend serves (the build-fullstack-backend
  skill and `generate_artifact/prompts.py`).
* **What the model must not retype.** A generated page carries its source
  data as an embedded JSON block (`observed_data.bind`), often far larger than
  the page around it. `read()` shows that block as ONE placeholder line, and
  `write()` puts the original block back, so the data can be neither dropped
  nor rewritten by an edit. A page compiled from a presentation program
  (`component_page.document`) is shown as its program alone, between the two
  markers the generator itself reads it back from; see `_component_bounds`.
* **Documents are packages of parts.** A Word, Excel or PowerPoint file
  (`.docx`, `.xlsx`, `.pptx`) is a zip of XML parts, and a change to one is a
  change to a part. Each package's content parts are listed as files of their
  own, addressed `<package path>/<part path>` (`report.docx/word/document.xml`),
  and read, searched and patched like any text file. A read unpacks the part;
  a write repacks the whole package and lands it like any other file, so the
  all-or-nothing and no-symlink rules below cover it, and a snapshot of a part
  is a snapshot of its package's bytes. A PDF is not a package of text and is
  never edited here.
* **How writes land.** All paths are checked before anything is written, then
  every file is staged next to its target and renamed into place; a failure
  part way puts back the files already replaced. Every path below the folder
  is opened one component at a time with `O_NOFOLLOW` (`anton.utils.nofollow`),
  so a symlink planted in the folder is refused rather than followed.

Everything here is synchronous and bounded. `search()` in particular never
arms SIGALRM (the old `anton/core/yolo/workspace.py` did, which only works on
the main thread and blocks the event loop while it waits); it is kept cheap
by refusing patterns that backtrack badly and by capping what it scans.
"""

from __future__ import annotations

import errno
import hashlib
import io
import os
import re
import stat
import sys
import time
import uuid
import zipfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from anton.core.artifacts.internal_files import (
    GENERATION_INPUT_FILES,
    HOUSEKEEPING_DIRS,
    NON_CONTENT_NAMES,
    PRESENTATION_PROGRAM_FILENAME,
    REVISIONS_DIRNAME,
    is_bytecode_name,
)
from anton.core.artifacts.models import ARTIFACT_TYPES
from anton.utils.nofollow import open_nofollow, opened_dir_nofollow

try:  # Python 3.11+
    import re._parser as _sre_parse  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - older interpreters
    import sre_parse as _sre_parse  # type: ignore[no-redef]

__all__ = [
    "ArtifactWorkspace",
    "MapEntry",
    "SearchHit",
    "Snapshot",
    "WorkspaceError",
]


class WorkspaceError(Exception):
    """A path or an edit the workspace refuses. The message is model-facing."""


@dataclass(frozen=True)
class MapEntry:
    path: str  # artifact-relative, forward slashes
    bytes: int  # size of the view `read()` returns (masked), or on-disk size


@dataclass(frozen=True)
class SearchHit:
    path: str
    line: int  # 1-based, in the masked view `read()` returns
    text: str  # the matching line, clipped around the match


# ── limits ────────────────────────────────────────────────────────────

# The largest view `read()` hands the model, measured AFTER masking: a 300 KB
# page whose 280 KB is embedded data is a 20 KB read.
MAX_READ_BYTES = 256 * 1024
# The most bytes read from disk for one file. Covers the 4 MB observed-data
# handoff cap (`observed_data.read_handoff`) embedded into a page, with room.
MAX_RAW_BYTES = 16 * 1024 * 1024
# The map is a listing for a model to choose from, not an inventory.
MAX_MAP_FILES = 1000
SNIFF_BYTES = 8192

# Whole-artifact preload: the numbers yolocoder settled on
# (`internal/agent/whole_context.go`): sent up front only when the whole
# eligible set fits, since the read it saves costs a round trip.
PRELOAD_MAX_FILES = 40
PRELOAD_MAX_BYTES = 48 * 1024  # including the "--- path ---" framing
PRELOAD_MAX_FILE_BYTES = 32 * 1024

# Search caps. Hits are what the model reads; files and bytes bound the work.
SEARCH_MAX_HITS = 60
SEARCH_MAX_HITS_PER_FILE = 10
SEARCH_MAX_FILES = 200
SEARCH_MAX_BYTES = 4 * 1024 * 1024
# A long line (minified code, a one-line page) is matched in windows of this
# many characters, overlapping so a match across a boundary is still found.
# Bounding the subject bounds what a polynomial pattern can cost.
SEARCH_WINDOW = 1000
SEARCH_OVERLAP = 200
# Checked between windows: a soft stop, not an interrupt.
SEARCH_DEADLINE_S = 2.0
SEARCH_SHOWN_CHARS = 240

# Folders that are never artifact content at any depth: dependencies, build
# output, caches, VCS (yolocoder's `generatedPath`, the old yolo `SKIP_DIRS`),
# plus the store's revision history wherever it turns up.
SKIP_DIRS = frozenset({
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    REVISIONS_DIRNAME,
})

# Known binary formats, refused without opening them. Anything else is sniffed.
BINARY_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".tif", ".tiff", ".avif", ".heic",
    ".pdf", ".zip", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".tar",
    ".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt", ".odt", ".ods", ".odp",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".mp3", ".mp4", ".wav", ".ogg", ".webm", ".mov", ".avi", ".m4a", ".flac",
    ".sqlite", ".sqlite3", ".db", ".parquet", ".feather", ".arrow", ".pkl", ".pickle", ".npy", ".npz",
    ".so", ".dylib", ".dll", ".exe", ".bin", ".wasm", ".class", ".jar", ".pyc", ".pyo",
})

# Preload order (Mochi's `preloadWhole`): what changes least first, so an
# edit invalidates a cached prompt only from the first changed file on. Code
# changes less than the content it presents.
CODE_SUFFIXES = frozenset({
    ".html", ".htm", ".css", ".scss", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".vue", ".svelte",
    ".py", ".json", ".toml", ".yaml", ".yml", ".sh", ".sql",
})
CONTENT_SUFFIXES = frozenset({".md", ".mdx", ".txt", ".csv", ".tsv", ".svg", ".xml", ".ini", ".cfg"})

HTML_SUFFIXES = frozenset({".html", ".htm"})
FULLSTACK_KINDS = frozenset({"fullstack-stateless-app", "fullstack-stateful-app"})
FULLSTACK_ROOT_FILES = ("backend.py", "requirements.txt")
STATE_MANIFEST = "state_manifest.json"

# Word, Excel and PowerPoint packages: zips of XML parts (ECMA-376, Open
# Packaging Conventions). A PDF is a binary page description, not one of them.
OOXML_SUFFIXES = frozenset({".docx", ".xlsx", ".pptx"})
PDF_SUFFIX = ".pdf"
# The largest package read: the same bound cowork-server keeps for a document
# (`publish.PUBLISH_FILE_MAX_BYTES`), above MAX_RAW_BYTES for text.
MAX_PACKAGE_BYTES = 25 * 1024 * 1024
# A repack rewrites every member, so the members and what they unpack to are
# bounded too: a small zip can claim to hold gigabytes.
MAX_PACKAGE_MEMBERS = 4000
MAX_PACKAGE_UNPACKED_BYTES = 256 * 1024 * 1024
# Parts that can be read and patched: XML and relationship parts. Media,
# embedded objects and printer settings are binary and stay as they are.
PART_SUFFIXES = (".xml", ".rels")

_DOCX_TEXT = r"word/(?:document|footnotes|endnotes|comments)\.xml|word/(?:header|footer)\d*\.xml"
_XLSX_TEXT = r"xl/workbook\.xml|xl/sharedStrings\.xml|xl/worksheets/sheet\d+\.xml"
_PPTX_TEXT = r"ppt/slides/slide\d+\.xml"
# The parts listed for each kind of package: the ones that carry its content
# and its formatting. Relationships, `[Content_Types].xml` and document
# properties are left off the list (a change rarely needs them) but can still
# be read and patched by path when one does, such as adding a slide.
_LISTED_PARTS = {
    ".docx": re.compile(rf"(?:{_DOCX_TEXT}|word/(?:styles|numbering|settings)\.xml|word/theme/theme\d*\.xml)"),
    ".xlsx": re.compile(rf"(?:{_XLSX_TEXT}|xl/styles\.xml|xl/theme/theme\d*\.xml|xl/charts/chart\d+\.xml)"),
    ".pptx": re.compile(
        rf"(?:{_PPTX_TEXT}|ppt/presentation\.xml|ppt/slideLayouts/slideLayout\d+\.xml"
        r"|ppt/slideMasters/slideMaster\d+\.xml|ppt/theme/theme\d+\.xml|ppt/notesSlides/notesSlide\d+\.xml"
        r"|ppt/charts/chart\d+\.xml)"
    ),
}
# The parts preloaded with a small artifact: where the words and numbers are.
# Styles, themes, masters and layouts are mostly boilerplate, often larger than
# the content itself, and read on demand.
_PRELOAD_PARTS = {
    ".docx": re.compile(_DOCX_TEXT),
    ".xlsx": re.compile(_XLSX_TEXT),
    ".pptx": re.compile(_PPTX_TEXT),
}
# The part a request is most likely about, read first when nothing is preloaded.
_MAIN_PARTS = {".docx": "word/document.xml", ".xlsx": "xl/worksheets/sheet1.xml", ".pptx": "ppt/slides/slide1.xml"}
_PACKAGE_NAMES = {".docx": "Word document", ".xlsx": "Excel workbook", ".pptx": "PowerPoint presentation"}

_IS_WINDOWS = sys.platform == "win32"
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_TMP_SUFFIX = ".anton-tmp"


# ── masking ───────────────────────────────────────────────────────────

# The binding `observed_data.bind` inserts: a run of `<meta ... data-cowork-observed>`
# and `<script ... data-cowork-observed>...</script>` elements. Each element
# pattern is the one `observed_data.strip_binding` removes, so what is masked
# here is exactly what generation would strip and re-bind.
_OBSERVED_SCRIPT = r"<script\b(?=[^>]*\bdata-cowork-observed\b)[^>]*>.*?</script\b[^>]*>"
_OBSERVED_META = r"<meta\b(?=[^>]*\bdata-cowork-observed\b)[^>]*>"
_OBSERVED_ELEMENT = rf"(?:{_OBSERVED_SCRIPT}|{_OBSERVED_META})"
_OBSERVED_RUN = re.compile(rf"{_OBSERVED_ELEMENT}(?:\s*{_OBSERVED_ELEMENT})*", re.S | re.I)

# Any placeholder this module writes, whichever file it came from.
_PLACEHOLDER_RE = re.compile(r"anton:keep [a-z-]+ [0-9a-f]{8}")


def _component_markers() -> tuple[str, str]:
    from anton.core.tools.generate_artifact.component_page import BEGIN, END

    return BEGIN, END


@dataclass(frozen=True)
class _Mask:
    start: int
    end: int
    placeholder: str
    # Newlines added so the placeholder sits on its own line; removed again
    # (when still there) on the way back.
    pre: str
    post: str


def _size(n: int) -> str:
    return f"{n} B" if n < 1024 else f"{n / 1024:.1f} KB"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:8]


def _mask_at(text: str, start: int, end: int, placeholder: str) -> _Mask:
    pre = "\n" if start > 0 and text[start - 1] != "\n" else ""
    post = "\n" if end < len(text) and text[end] != "\n" else ""
    return _Mask(start, end, placeholder, pre, post)


def _component_bounds(text: str) -> tuple[int, int] | None:
    """Where the program sits in a page compiled from a presentation program.

    `component_page.document` compiles `CoworkPage(config)` JavaScript into a
    page: the presentation CSS and ~45 KB of host runtime, then the program
    between `/* cowork-presentation-begin */` and `/* cowork-presentation-end */`.
    The program in the PAGE is the source of truth: `fast_static.run` reads
    it back with `component_page.existing_program(page)` and recompiles
    everything around it from the runtime files shipped with anton.
    `presentation-program.js` is only a recovery copy (`generate_artifact`
    reports it as `last_saved_component_program` when generation fails).

    So the editable part of such a page is the program and nothing else:
    an edit to the CSS or the runtime in the page would be silently lost the
    next time the page is generated, and an edit to the runtime is an edit to
    host code the page shares with every other generated page.
    """
    begin, end = _component_markers()
    if text.count(begin) != 1 or text.count(end) != 1:
        return None
    start = text.index(begin)
    stop = text.index(end) + len(end)
    if stop <= start:
        return None
    return start, stop


def _masks_for(path: str, text: str) -> list[_Mask]:
    """The regions of `text` the model sees as placeholders, in order."""
    if os.path.splitext(path)[1].lower() not in HTML_SUFFIXES:
        return []
    bounds = _component_bounds(text)
    if bounds is not None:
        start, stop = bounds
        masks = []
        if start > 0:
            shell = text[:start]
            masks.append(_mask_at(text, 0, start, (
                f"<!-- anton:keep presentation-shell {_digest(shell)} - the compiled page, its styles and "
                f"runtime ({_size(len(shell.encode()))}), rebuilt from the program below. Keep this line "
                "exactly; edit only the program between the presentation markers. -->"
            )))
        if stop < len(text):
            tail = text[stop:]
            masks.append(_mask_at(text, stop, len(text), (
                f"<!-- anton:keep presentation-tail {_digest(tail)} - the end of the compiled page. "
                "Keep this line exactly. -->"
            )))
        return masks
    masks = []
    for match in _OBSERVED_RUN.finditer(text):
        block = match.group(0)
        masks.append(_mask_at(text, match.start(), match.end(), (
            f"<!-- anton:keep observed-data {_digest(block)} - {_size(len(block.encode()))} of embedded "
            "source data (window.ARTIFACT_DATA), restored on save. Keep this line exactly; read the data "
            "through window.ARTIFACT_DATA, never copy it. -->"
        )))
    return masks


def _apply_masks(text: str, masks: list[_Mask]) -> str:
    out: list[str] = []
    cursor = 0
    for mask in masks:
        out.append(text[cursor:mask.start])
        out.append(mask.pre + mask.placeholder + mask.post)
        cursor = mask.end
    out.append(text[cursor:])
    return "".join(out)


def _restore_masks(path: str, masked: str, original: str, masks: list[_Mask]) -> str:
    """`masked` with every placeholder replaced by the block it stands for."""
    problems = []
    for mask in masks:
        count = masked.count(mask.placeholder)
        if count == 0:
            problems.append(
                f"{path}: the line `{mask.placeholder[:60]}...` was removed or changed. It stands for content "
                "that is restored on save; put the line back exactly as read."
            )
        elif count > 1:
            problems.append(
                f"{path}: the line `{mask.placeholder[:60]}...` appears {count} times; keep exactly one copy, "
                "where it was."
            )
    stray = masked
    for mask in masks:
        stray = stray.replace(mask.placeholder, "")
    if not problems and _PLACEHOLDER_RE.search(stray):
        problems.append(
            f"{path}: contains an `anton:keep` placeholder that does not belong to this file. "
            "Remove it; placeholders only stand for content in the file they were read from."
        )
    if problems:
        raise WorkspaceError("\n".join(problems))
    for mask in masks:
        index = masked.index(mask.placeholder)
        left = masked[:index]
        right = masked[index + len(mask.placeholder):]
        if mask.pre and left.endswith(mask.pre):
            left = left[: -len(mask.pre)]
        if mask.post and right.startswith(mask.post):
            right = right[len(mask.post):]
        masked = left + original[mask.start:mask.end] + right
    return masked


# ── document parts ────────────────────────────────────────────────────

# A part is usually written as one line: the whole body of a document after
# its XML declaration. A patch places lines, so the view adds a line break
# after every tag that is followed by another tag (directly, or after line
# breaks of its own), and the write takes exactly that one break out again.
# Adding one and removing one is exact both ways, so a part that comes back
# unchanged is byte-identical. A text element (`<w:t>`, `<a:t>`, `<t>`, ...)
# is never broken, so no text a reader of the document sees changes.
_TAG_BEFORE_TAG = re.compile(r"(<[^<>]*>)(\n*)(?=<)")
_TAG_BREAK_TAG = re.compile(r"(<[^<>]*>)\n(\n*)(?=<)")
_TEXT_ELEMENT_OPEN = re.compile(r"<(?:[\w.-]+:)?(?:t|delText|instrText)(?:\s[^<>]*)?(?<!/)>")


def _part_view(xml: str) -> str:
    """The part with a line break added after every tag that a tag follows."""
    return _TAG_BEFORE_TAG.sub(
        lambda m: m.group(0) if _TEXT_ELEMENT_OPEN.fullmatch(m.group(1)) else m.group(0) + "\n", xml
    )


def _part_unview(view: str) -> str:
    """`_part_view` undone: the one added line break after each such tag taken out again."""
    return _TAG_BREAK_TAG.sub(
        lambda m: m.group(0) if _TEXT_ELEMENT_OPEN.fullmatch(m.group(1)) else m.group(1) + m.group(2), view
    )


def _natural(name: str) -> list:
    """Sort key putting slide2.xml before slide10.xml."""
    return [int(piece) if piece.isdigit() else piece for piece in re.split(r"(\d+)", name)]


def _zip_member(zf: zipfile.ZipFile, part: str) -> zipfile.ZipInfo | None:
    """The member for `part`. Part names compare case-insensitively (OPC)."""
    try:
        return zf.getinfo(part)
    except KeyError:
        pass
    lower = part.lower()
    for info in zf.infolist():
        if info.filename.lower() == lower:
            return info
    return None


def _copy_info(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    """A fresh header for `info`'s member: same name, date, method and attributes."""
    copy = zipfile.ZipInfo(info.filename, info.date_time)
    copy.compress_type = info.compress_type
    copy.external_attr = info.external_attr
    copy.create_system = info.create_system
    copy.comment = info.comment
    return copy


# ── search safety ─────────────────────────────────────────────────────

_REPEATS = {
    getattr(_sre_parse, name)
    for name in ("MAX_REPEAT", "MIN_REPEAT")
    if hasattr(_sre_parse, name)
}
_POSSESSIVE = getattr(_sre_parse, "POSSESSIVE_REPEAT", None)
_UNBOUNDED = _sre_parse.MAXREPEAT


def _backtracks_badly(pattern: str) -> bool:
    """Whether `pattern` has a shape that can backtrack exponentially or worse.

    No interrupt is armed while a regex runs (there is none that works off the
    main thread without blocking it), so a pattern that could run away is not
    started. Refused shapes: a variable repeat inside another variable repeat
    (`(a+)+`), an alternation inside one whose branches can start alike
    (`(a|ab)*`), back-references, and three or more unbounded repeats of
    anything but one literal character (`.*a.*b.*c`), which cost a power of
    the line length. Everything ordinary (`function\\s+draw\\w*`,
    `(foo|bar)+`, `.*TODO.*`) passes.
    """
    try:
        tree = _sre_parse.parse(pattern)
    except Exception:
        return True
    unbounded = 0

    def first_literal(items) -> int | None:
        for op, av in items:
            if op is _sre_parse.LITERAL:
                return av
            if op is _sre_parse.SUBPATTERN:
                return first_literal(av[-1])
            return None
        return None

    def walk(items, inside: bool) -> bool:
        nonlocal unbounded
        for op, av in items:
            if op in _REPEATS:
                low, high, body = av
                variable = high != low
                if variable and inside:
                    return True
                if high is _UNBOUNDED or high == _UNBOUNDED:
                    single = len(body) == 1 and body[0][0] is _sre_parse.LITERAL
                    if not single:
                        unbounded += 1
                if walk(body, inside or variable):
                    return True
            elif _POSSESSIVE is not None and op is _POSSESSIVE:
                if walk(av[2], inside):
                    return True
            elif op is _sre_parse.SUBPATTERN:
                if walk(av[-1], inside):
                    return True
            elif op is _sre_parse.BRANCH:
                branches = av[1]
                if inside:
                    firsts = [first_literal(b) for b in branches]
                    if None in firsts or len(set(firsts)) != len(firsts):
                        return True
                for branch in branches:
                    if walk(branch, inside):
                        return True
            elif op in (_sre_parse.ASSERT, _sre_parse.ASSERT_NOT):
                if walk(av[1], inside):
                    return True
            elif op in (_sre_parse.GROUPREF, _sre_parse.GROUPREF_EXISTS):
                return True
            elif getattr(_sre_parse, "ATOMIC_GROUP", None) is not None and op is _sre_parse.ATOMIC_GROUP:
                if walk(av, inside):
                    return True
        return False

    if walk(tree.data if hasattr(tree, "data") else list(tree), False):
        return True
    return unbounded >= 3


def _windows(line: str) -> Iterator[tuple[int, str]]:
    if len(line) <= SEARCH_WINDOW:
        yield 0, line
        return
    step = SEARCH_WINDOW - SEARCH_OVERLAP
    for offset in range(0, len(line), step):
        yield offset, line[offset: offset + SEARCH_WINDOW]
        if offset + SEARCH_WINDOW >= len(line):
            return


def _clip_around(line: str, at: int) -> str:
    line = line.rstrip("\r")
    if len(line) <= SEARCH_SHOWN_CHARS:
        return line.strip()
    start = max(0, min(at - SEARCH_SHOWN_CHARS // 4, len(line) - SEARCH_SHOWN_CHARS))
    piece = line[start: start + SEARCH_SHOWN_CHARS]
    return ("…" if start else "") + piece + ("…" if start + SEARCH_SHOWN_CHARS < len(line) else "")


# ── paths ─────────────────────────────────────────────────────────────


def _clean(path: str) -> str:
    """`path` as a normalised artifact-relative POSIX path, or WorkspaceError."""
    if not isinstance(path, str) or not path.strip():
        raise WorkspaceError("an empty path names no file")
    raw = path.strip()
    if "\0" in raw or "\\" in raw:
        raise WorkspaceError(f"{raw}: not a valid artifact path")
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise WorkspaceError(f"{raw}: use a path relative to the artifact folder, not an absolute one")
    while raw.startswith("./"):
        raw = raw[2:]
    parts = raw.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise WorkspaceError(f"{raw}: is outside the artifact folder or not a plain relative path")
    return "/".join(parts)


def _looks_binary(head: bytes, *, whole: bool) -> bool:
    """NUL bytes or invalid UTF-8 in the first SNIFF_BYTES (`whole`: the entire file)."""
    if b"\0" in head:
        return True
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        # A multi-byte character cut by the sniff window is not an error.
        return whole or not (exc.reason == "unexpected end of data" and exc.start >= len(head) - 3)
    return False


def _replace_at(src: str, dst: str, dir_fd: int | None) -> None:
    """One rename inside a pinned folder. A seam for tests to inject a failure."""
    if dir_fd is None:
        os.replace(src, dst)
    else:
        os.replace(src, dst, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)


@dataclass
class Snapshot:
    """What some paths held at one moment; `restore()` puts exactly that back.

    A path that did not exist is deleted by `restore()`, so a file the edit
    created goes away again.
    """

    _workspace: ArtifactWorkspace
    _saved: dict[str, tuple[bytes, int] | None] = field(default_factory=dict)

    @property
    def paths(self) -> list[str]:
        return list(self._saved)

    def restore(self) -> None:
        self._workspace._commit(dict(self._saved))


class ArtifactWorkspace:
    """Read, search and write one artifact folder on the model's behalf."""

    def __init__(
        self,
        folder: Path,
        *,
        kind: str,
        primary: str | None = None,
        sandbox: object | None = None,
    ) -> None:
        if kind not in ARTIFACT_TYPES:
            raise ValueError(f"unknown artifact kind {kind!r}; expected one of {', '.join(ARTIFACT_TYPES)}")
        self.folder = Path(os.path.realpath(folder))
        self.kind = kind
        self.sandbox = sandbox
        self.primary = _clean(primary) if primary else self._default_primary()
        # Why the last search was narrowed (literal fallback, caps); "" when it wasn't.
        self.search_note = ""

    # ── policy ───────────────────────────────────────────────────────

    def _default_primary(self) -> str | None:
        if self.kind in FULLSTACK_KINDS:
            return "static/index.html"
        if self.kind != "html-app":
            return None
        try:
            pages = sorted(
                entry.name for entry in os.scandir(self.folder)
                if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith((".html", ".htm"))
            )
        except OSError:
            pages = []
        return pages[0] if len(pages) == 1 else "index.html"

    def _check_read(self) -> None:
        if self.sandbox is not None and not self.sandbox.allows_read(self.folder):
            raise WorkspaceError("this artifact is outside the files this conversation may read")

    def _check_write(self) -> None:
        """Under a sandbox, write only where the person's own cells may write.

        `CellSandbox` grants writes on its project, home, tmp and shared
        folders (`_writable_dirs`); a policy that says so itself is asked.
        """
        if self.sandbox is None:
            return
        allows_write = getattr(self.sandbox, "allows_write", None)
        if callable(allows_write):
            allowed = allows_write(self.folder)
        else:
            roots = [getattr(self.sandbox, name, None) for name in ("project_dir", "home_dir", "tmp_dir", "shared_dir")]
            target = os.path.realpath(self.folder)
            allowed = False
            for root in roots:
                if root is None:
                    continue
                real = os.path.realpath(root)
                if target == real or target.startswith(real.rstrip(os.sep) + os.sep):
                    allowed = True
                    break
        if not allowed:
            raise WorkspaceError("this artifact is outside the files this conversation may change")

    def _hidden_reason(self, rel: str) -> str | None:
        """Why `rel` is not artifact content, or None. Mirrors `map()`."""
        reason = self._non_content_reason(rel)
        if reason:
            return reason
        suffix = os.path.splitext(rel)[1].lower()
        if suffix in OOXML_SUFFIXES:
            main = _MAIN_PARTS[suffix]
            return (
                f"{rel}: a {_PACKAGE_NAMES[suffix]}, a package of XML parts. Read and edit its parts by path "
                f"instead, such as {rel}/{main}."
            )
        if suffix == PDF_SUFFIX:
            return f"{rel}: a PDF, which cannot be edited here; only text files and Word, Excel and PowerPoint parts can"
        if suffix in BINARY_SUFFIXES:
            return f"{rel}: a binary file; only text files can be read or edited"
        return None

    def _non_content_reason(self, rel: str) -> str | None:
        """Why `rel` is housekeeping, an input, bytecode or a dependency, or None."""
        parts = rel.split("/")
        top = parts[0]
        if top == PRESENTATION_PROGRAM_FILENAME and len(parts) == 1:
            where = f" in {self.primary}" if self.primary else ""
            return (
                f"{rel}: a saved copy of the presentation program, not the artifact. Edit the program "
                f"between the presentation markers{where}; this copy is kept in step on save."
            )
        if top in GENERATION_INPUT_FILES:
            return f"{rel}: an input the artifact was generated from, not part of the artifact"
        if top in NON_CONTENT_NAMES:
            what = "folder" if top in HOUSEKEEPING_DIRS else "file"
            return f"{rel}: a housekeeping {what} the artifact store manages, not part of the artifact"
        if any(is_bytecode_name(part) for part in parts):
            return f"{rel}: Python bytecode, not part of the artifact"
        for part in parts[:-1]:
            if part in SKIP_DIRS:
                return f"{rel}: inside {part}/, which holds dependencies, build output or history, not the artifact"
        return None

    def _kind_reason(self, rel: str, *, create: bool) -> str | None:
        if self.kind == "html-app":
            if create and rel != self.primary:
                return (
                    f"{rel}: this artifact is a single self-contained page ({self.primary}). The preview serves "
                    "that one file only, so a page that loads another file fails the render check "
                    f"(separate_file). Put the code, styles or data inline in {self.primary} instead."
                )
            return None
        if self.kind in FULLSTACK_KINDS:
            allowed = set(FULLSTACK_ROOT_FILES)
            if self.kind == "fullstack-stateful-app":
                allowed.add(STATE_MANIFEST)
            if rel in allowed or rel.startswith("static/"):
                return None
            manifest = f", {STATE_MANIFEST}" if self.kind == "fullstack-stateful-app" else ""
            return (
                f"{rel}: a {self.kind} has a fixed layout: backend.py, requirements.txt{manifest} at the root, "
                "and every frontend file (HTML, CSS, JS, images, data) under static/, the only folder the "
                "backend serves. Put backend code in backend.py and frontend files under static/."
            )
        return None

    def _link_reason(self, rel: str) -> str | None:
        """A symlink, a file used as a folder, or a folder at `rel`."""
        current = self.folder
        parts = rel.split("/")
        for i, part in enumerate(parts):
            current = current / part
            try:
                st = os.lstat(current)
            except FileNotFoundError:
                return None
            except OSError as exc:
                return f"{rel}: cannot be checked ({exc.strerror})"
            if stat.S_ISLNK(st.st_mode):
                return f"{rel}: {'/'.join(parts[: i + 1])} is a symlink; links are never followed or replaced"
            last = i == len(parts) - 1
            if not last and not stat.S_ISDIR(st.st_mode):
                return f"{rel}: {'/'.join(parts[: i + 1])} is a file, not a folder"
            if last and not stat.S_ISREG(st.st_mode):
                return f"{rel}: not a regular file"
        return None

    def refuse_reason(self, path: str, *, create: bool) -> str | None:
        """Why an edit to `path` is not allowed, or None when it is."""
        try:
            rel = _clean(path)
        except WorkspaceError as exc:
            return str(exc)
        split = self._package_split(rel)
        if split is not None:
            return self._part_reason(rel, *split, create=create)
        reason = self._hidden_reason(rel) or self._link_reason(rel)
        if reason:
            return reason
        if not create and self._is_binary_file(rel):
            return f"{rel}: a binary file; only text files can be read or edited"
        return self._kind_reason(rel, create=create)

    # ── document packages ────────────────────────────────────────────

    def _package_split(self, rel: str) -> tuple[str, str] | None:
        """(package path, part path) when `rel` names a part inside a document package.

        The first path component with a package suffix that is not a folder on
        disk is the package; everything after it is the part.
        """
        parts = rel.split("/")
        for i, name in enumerate(parts[:-1]):
            if os.path.splitext(name)[1].lower() not in OOXML_SUFFIXES:
                continue
            package = "/".join(parts[: i + 1])
            try:
                if stat.S_ISDIR(os.lstat(self._target(package)).st_mode):
                    continue
            except OSError:
                pass
            return package, "/".join(parts[i + 1:])
        return None

    def package_of(self, path: str) -> str | None:
        """The document package `path` addresses a part of, or None for a plain file."""
        try:
            split = self._package_split(_clean(path))
        except WorkspaceError:
            return None
        return split[0] if split is not None else None

    def _part_reason(self, rel: str, package: str, part: str, *, create: bool) -> str | None:
        reason = self._non_content_reason(package) or self._link_reason(package)
        if reason:
            return reason
        if not self._exists(package):
            return f"{rel}: there is no document {package} in this artifact"
        if not part.lower().endswith(PART_SUFFIXES):
            return f"{rel}: a binary part (media or an embedded object); only the XML parts of {package} can be edited"
        try:
            with self._package(package) as (zf, _data):
                info = _zip_member(zf, part)
        except WorkspaceError as exc:
            return str(exc)
        if info is None and not create:
            return f"{rel}: {package} has no part {part}"
        if info is not None and info.is_dir():
            return f"{rel}: a folder inside {package}, not a part"
        return None

    @contextmanager
    def _package(self, package: str) -> Iterator[tuple[zipfile.ZipFile, bytes]]:
        """The package's zip, opened from bytes read no-follow, and those bytes."""
        self._check_read()
        raw = self._read_raw(package)
        if raw is None:
            raise WorkspaceError(f"{package}: no such document")
        try:
            zf = zipfile.ZipFile(io.BytesIO(raw[0]))
        except (zipfile.BadZipFile, OSError, ValueError, EOFError) as exc:
            raise WorkspaceError(f"{package}: not a readable document package ({exc})") from exc
        with zf:
            members = zf.infolist()
            if len(members) > MAX_PACKAGE_MEMBERS:
                raise WorkspaceError(f"{package}: holds {len(members)} parts, more than this editor opens")
            if sum(info.file_size for info in members) > MAX_PACKAGE_UNPACKED_BYTES:
                raise WorkspaceError(f"{package}: unpacks to more than {_size(MAX_PACKAGE_UNPACKED_BYTES)}")
            yield zf, raw[0]

    @staticmethod
    def _member_bytes(zf: zipfile.ZipFile, info: zipfile.ZipInfo, rel: str) -> bytes:
        if info.file_size > MAX_RAW_BYTES:
            raise WorkspaceError(f"{rel}: too large ({_size(info.file_size)})")
        try:
            with zf.open(info) as handle:
                data = handle.read(MAX_RAW_BYTES + 1)
        except (zipfile.BadZipFile, RuntimeError, NotImplementedError, OSError, EOFError, ValueError) as exc:
            raise WorkspaceError(f"{rel}: cannot be unpacked ({exc})") from exc
        if len(data) > MAX_RAW_BYTES:
            raise WorkspaceError(f"{rel}: too large")
        return data

    def _part_text(self, zf: zipfile.ZipFile, info: zipfile.ZipInfo, rel: str) -> str:
        return self._decode(rel, self._member_bytes(zf, info, rel))

    def _listed_parts(self, package: str, zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
        """The package's content and formatting parts, in reading order."""
        pattern = _LISTED_PARTS[os.path.splitext(package)[1].lower()]
        found = [info for info in zf.infolist() if not info.is_dir() and pattern.fullmatch(info.filename)]
        return sorted(found, key=lambda info: _natural(info.filename))

    def _package_entries(self, package: str) -> list[MapEntry]:
        """One MapEntry per listed part of `package`, sized as `read()` shows it."""
        out: list[MapEntry] = []
        try:
            with self._package(package) as (zf, _data):
                for info in self._listed_parts(package, zf):
                    rel = f"{package}/{info.filename}"
                    try:
                        view = _part_view(self._part_text(zf, info, rel))
                    except WorkspaceError:
                        continue
                    out.append(MapEntry(rel, len(view.encode("utf-8"))))
        except WorkspaceError:
            return []
        return out

    def _part_masked(self, rel: str, package: str, part: str) -> str | None:
        reason = self._non_content_reason(package)
        if reason:
            raise WorkspaceError(reason)
        if not part.lower().endswith(PART_SUFFIXES):
            raise WorkspaceError(f"{rel}: a binary part (media or an embedded object); only XML parts can be read")
        if not self._exists(package):
            return None
        with self._package(package) as (zf, _data):
            info = _zip_member(zf, part)
            if info is None or info.is_dir():
                return None
            return _part_view(self._part_text(zf, info, rel))

    def _repack(self, package: str, edits: dict[str, str]) -> tuple[bytes, int] | None:
        """The package's new bytes with each part in `edits` replaced (views unviewed), or None when unchanged.

        Every other member is carried over unchanged, in its order and with its
        compression, so `[Content_Types].xml` stays first as Office expects. A
        part that does not exist yet is added at the end.
        """
        current = self._read_raw(package)
        if current is None:
            raise WorkspaceError(f"{package}: no such document")
        with self._package(package) as (zf, _data):
            replaced: dict[str, bytes] = {}
            added: dict[str, bytes] = {}
            for part, view in edits.items():
                rel = f"{package}/{part}"
                info = _zip_member(zf, part)
                text = _part_unview(view)
                try:
                    data = text.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise WorkspaceError(f"{rel}: the text cannot be saved as UTF-8 ({exc.reason})") from exc
                if info is None:
                    if _PLACEHOLDER_RE.search(text):
                        raise WorkspaceError(
                            f"{rel}: contains an `anton:keep` placeholder that does not belong to this file. "
                            "Remove it; placeholders only stand for content in the file they were read from."
                        )
                    added[part] = data
                elif data != self._member_bytes(zf, info, rel):
                    replaced[info.filename] = data
            if not replaced and not added:
                return None
            out = io.BytesIO()
            try:
                with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
                    for info in zf.infolist():
                        body = replaced.get(info.filename)
                        if body is None:
                            body = self._member_bytes(zf, info, f"{package}/{info.filename}")
                        dst.writestr(_copy_info(info), body)
                    stamp = time.localtime()[:6]
                    for part, body in added.items():
                        info = zipfile.ZipInfo(part, stamp)
                        info.compress_type = zipfile.ZIP_DEFLATED
                        dst.writestr(info, body)
            except (zipfile.BadZipFile, RuntimeError, NotImplementedError, ValueError) as exc:
                raise WorkspaceError(f"{package}: could not be repacked ({exc})") from exc
        return out.getvalue(), current[1]

    def read_package(self, path: str) -> bytes | None:
        """The bytes of the document package `path` is (or names a part of), read no-follow."""
        rel = _clean(path)
        split = self._package_split(rel)
        package = split[0] if split is not None else rel
        if os.path.splitext(package)[1].lower() not in OOXML_SUFFIXES:
            raise WorkspaceError(f"{rel}: not a Word, Excel or PowerPoint document")
        reason = self._non_content_reason(package)
        if reason:
            raise WorkspaceError(reason)
        self._check_read()
        raw = self._read_raw(package)
        return raw[0] if raw is not None else None

    def pdf_files(self) -> list[str]:
        """The artifact's PDFs: content, but never edited here."""
        try:
            self._check_read()
        except WorkspaceError:
            return []
        return sorted(rel for rel, _st in self._walk() if rel.lower().endswith(PDF_SUFFIX))

    def entry_file(self) -> str | None:
        """The file an edit most likely starts in: the entry page, or the primary document's main part."""
        primary = self.primary
        if primary and os.path.splitext(primary)[1].lower() not in OOXML_SUFFIXES:
            return primary
        entries = [entry.path for entry in self.map()]
        packages = [primary] if primary else list(dict.fromkeys(p for p in map(self.package_of, entries) if p))
        if len(packages) != 1:
            return None
        package = packages[0]
        main = f"{package}/{_MAIN_PARTS[os.path.splitext(package)[1].lower()]}"
        if main in entries:
            return main
        listed = [p for p in entries if p.startswith(package + "/")
                  and _PRELOAD_PARTS[os.path.splitext(package)[1].lower()].fullmatch(p[len(package) + 1:])]
        return listed[0] if listed else None

    # ── raw I/O ──────────────────────────────────────────────────────

    def _target(self, rel: str) -> Path:
        return self.folder / rel

    def _read_raw(self, rel: str) -> tuple[bytes, int] | None:
        """(bytes, permission bits) of `rel`, or None when nothing is there.

        Opened like `nofollow.read_bytes_nofollow`: one component at a time
        with O_NOFOLLOW, then required to be a regular file with no other
        hard link, and read no further than MAX_RAW_BYTES (MAX_PACKAGE_BYTES
        for a document package).
        """
        try:
            fd = open_nofollow(self.folder, self._target(rel), os.O_RDONLY | _O_NONBLOCK)
        except (FileNotFoundError, NotADirectoryError):
            return None
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise WorkspaceError(f"{rel}: a symlink; links are never followed") from exc
            raise WorkspaceError(f"{rel}: cannot be opened ({exc.strerror or exc})") from exc
        with os.fdopen(fd, "rb") as handle:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise WorkspaceError(f"{rel}: not a regular file")
            if not _IS_WINDOWS and st.st_nlink != 1:
                raise WorkspaceError(f"{rel}: has another hard link, so it is never read or written")
            limit = MAX_PACKAGE_BYTES if os.path.splitext(rel)[1].lower() in OOXML_SUFFIXES else MAX_RAW_BYTES
            if st.st_size > limit:
                raise WorkspaceError(f"{rel}: too large ({_size(st.st_size)})")
            data = handle.read(limit + 1)
        if len(data) > limit:
            raise WorkspaceError(f"{rel}: too large")
        return data, stat.S_IMODE(st.st_mode)

    def _is_binary_file(self, rel: str) -> bool:
        try:
            fd = open_nofollow(self.folder, self._target(rel), os.O_RDONLY | _O_NONBLOCK)
        except OSError:
            return False
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return False
            head = handle.read(SNIFF_BYTES + 1)
            return _looks_binary(head[:SNIFF_BYTES], whole=len(head) <= SNIFF_BYTES)

    def _decode(self, rel: str, data: bytes) -> str:
        if b"\0" in data:
            raise WorkspaceError(f"{rel}: a binary file; only text files can be read or edited")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceError(f"{rel}: not UTF-8 text; only text files can be read or edited") from exc

    # ── map ──────────────────────────────────────────────────────────

    def _walk(self) -> Iterator[tuple[str, os.stat_result]]:
        """(relative path, lstat) of each candidate content file, unsorted.

        Pruned like `store.iter_content_files` (top-level NON_CONTENT_NAMES,
        bytecode at any depth, symlinks never followed), and additionally
        SKIP_DIRS at any depth and known binary suffixes, except document
        packages and PDFs, which `map()` and `pdf_files()` sort out.
        """
        stack: list[tuple[str, str]] = [(os.fspath(self.folder), "")]
        while stack:
            current, prefix = stack.pop()
            try:
                with os.scandir(current) as it:
                    entries = list(it)
            except OSError:
                continue
            for entry in entries:
                name = entry.name
                if not prefix and name in NON_CONTENT_NAMES:
                    continue
                if is_bytecode_name(name) or name.endswith(_TMP_SUFFIX):
                    continue
                rel = f"{prefix}{name}"
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if name not in SKIP_DIRS:
                            stack.append((entry.path, rel + "/"))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    suffix = os.path.splitext(name)[1].lower()
                    if suffix in BINARY_SUFFIXES and suffix not in OOXML_SUFFIXES and suffix != PDF_SUFFIX:
                        continue
                    yield rel, entry.stat(follow_symlinks=False)
                except OSError:
                    continue

    def map(self) -> list[MapEntry]:
        """The artifact's text content files, sorted by path.

        HTML pages are sized as the model will read them (masked), every other
        file by its size on disk. A document package is listed as its content
        and formatting parts (`report.docx/word/document.xml`), each sized as
        `read()` shows it; a PDF is not listed.
        """
        try:
            self._check_read()
        except WorkspaceError:
            return []
        found: list[MapEntry] = []
        for rel, st in sorted(self._walk()):
            if len(found) >= MAX_MAP_FILES:
                break
            suffix = os.path.splitext(rel)[1].lower()
            if suffix == PDF_SUFFIX:
                continue
            if suffix in OOXML_SUFFIXES:
                found += self._package_entries(rel)[: MAX_MAP_FILES - len(found)]
                continue
            if self._is_binary_file(rel):
                continue
            size = st.st_size
            if os.path.splitext(rel)[1].lower() in HTML_SUFFIXES:
                try:
                    view = self.read(rel)
                except WorkspaceError:
                    view = None
                if view is not None:
                    size = len(view.encode("utf-8"))
            found.append(MapEntry(rel, size))
        return found

    # ── read ─────────────────────────────────────────────────────────

    def _masked(self, rel: str) -> str | None:
        """The masked view of `rel`, with no size cap; None when absent."""
        split = self._package_split(rel)
        if split is not None:
            return self._part_masked(rel, *split)
        reason = self._hidden_reason(rel)
        if reason:
            raise WorkspaceError(reason)
        self._check_read()
        raw = self._read_raw(rel)
        if raw is None:
            return None
        text = self._decode(rel, raw[0])
        return _apply_masks(text, _masks_for(rel, text))

    def read(self, path: str) -> str | None:
        """The text the model sees for `path`, None when it does not exist.

        Raises WorkspaceError for a path that is not artifact content, a
        link, a binary, or a view larger than MAX_READ_BYTES after masking.
        """
        rel = _clean(path)
        view = self._masked(rel)
        if view is None:
            return None
        size = len(view.encode("utf-8"))
        if size > MAX_READ_BYTES:
            raise WorkspaceError(
                f"{rel}: too large to read whole ({_size(size)}, limit {_size(MAX_READ_BYTES)}). "
                "Search it for the part you need instead."
            )
        return view

    # ── preload ──────────────────────────────────────────────────────

    def _preloadable(self, path: str) -> bool:
        split = self._package_split(path)
        if split is not None:
            package, part = split
            return bool(_PRELOAD_PARTS[os.path.splitext(package)[1].lower()].fullmatch(part))
        name = path.rsplit("/", 1)[-1].lower()
        if name.endswith(".lock") or "-lock." in name or name == "lock.json" or ".min." in name:
            return False
        suffix = os.path.splitext(name)[1]
        return suffix in CODE_SUFFIXES or suffix in CONTENT_SUFFIXES

    def preload(self) -> dict[str, str] | None:
        """Every eligible file's view when the whole set fits, else None.

        All or nothing: a set the model sees only part of reads as the whole
        project, which is worse than seeing a map and choosing. Eligible: text
        files by suffix, no lockfiles or minified bundles, and a document's
        content parts (its body, sheets and shared strings, or slides), not
        its styles, themes, masters or layouts, which are read when needed.
        Code files come before content files, then by path.
        """
        entries = [entry for entry in self.map() if self._preloadable(entry.path)]
        if not entries or len(entries) > PRELOAD_MAX_FILES:
            return None
        entries.sort(key=lambda e: (os.path.splitext(e.path)[1].lower() not in CODE_SUFFIXES, e.path))
        out: dict[str, str] = {}
        total = 0
        for entry in entries:
            try:
                view = self.read(entry.path)
            except WorkspaceError:
                return None
            if view is None:
                return None
            size = len(view.encode("utf-8"))
            if size > PRELOAD_MAX_FILE_BYTES:
                return None
            total += len(f"--- {entry.path} ---\n".encode()) + size + 1
            if total > PRELOAD_MAX_BYTES:
                return None
            out[entry.path] = view
        return out

    # ── search ───────────────────────────────────────────────────────

    def search(self, pattern: str) -> list[SearchHit]:
        """Lines matching `pattern` in the artifact's content files.

        A regular expression, case-sensitive. A pattern that does not compile,
        or one shaped to backtrack badly, is searched for as literal text
        instead, and `search_note` says so. Matches are against the masked
        views, so line numbers are the ones `read()` shows and embedded data
        is never searched.
        """
        self.search_note = ""
        query = (pattern or "").strip()
        if not query:
            self.search_note = "empty pattern"
            return []
        notes: list[str] = []
        try:
            compiled = re.compile(query)
        except re.error as exc:
            compiled = re.compile(re.escape(query))
            notes.append(f"not a valid regular expression ({exc}); searched as literal text")
        else:
            if _backtracks_badly(query):
                compiled = re.compile(re.escape(query))
                notes.append(
                    "the pattern could backtrack badly (nested or stacked repeats, overlapping "
                    "alternatives, back-references); searched as literal text. Simplify it to use regex."
                )
        hits: list[SearchHit] = []
        scanned_files = scanned_bytes = 0
        deadline = time.monotonic() + SEARCH_DEADLINE_S
        stopped = ""
        for entry in self.map():
            if len(hits) >= SEARCH_MAX_HITS:
                stopped = f"stopped at {SEARCH_MAX_HITS} matches; narrow the pattern"
                break
            if scanned_files >= SEARCH_MAX_FILES or scanned_bytes >= SEARCH_MAX_BYTES:
                stopped = "stopped before searching every file (search size cap); narrow the pattern"
                break
            try:
                view = self._masked(entry.path)
            except WorkspaceError:
                continue
            if view is None:
                continue
            scanned_files += 1
            scanned_bytes += len(view)
            per_file = 0
            for number, line in enumerate(view.split("\n"), start=1):
                if time.monotonic() > deadline:
                    stopped = f"stopped after {SEARCH_DEADLINE_S:g}s; narrow the pattern"
                    break
                for offset, window in _windows(line):
                    match = compiled.search(window)
                    if match:
                        hits.append(SearchHit(entry.path, number, _clip_around(line, offset + match.start())))
                        per_file += 1
                        break
                if per_file >= SEARCH_MAX_HITS_PER_FILE or len(hits) >= SEARCH_MAX_HITS:
                    break
            if stopped:
                break
        if stopped:
            notes.append(stopped)
        self.search_note = "; ".join(notes)
        return hits

    # ── write ────────────────────────────────────────────────────────

    def _exists(self, rel: str) -> bool:
        split = self._package_split(rel)
        if split is not None:
            package, part = split
            try:
                with self._package(package) as (zf, _data):
                    return _zip_member(zf, part) is not None
            except WorkspaceError:
                return False
        try:
            os.lstat(self._target(rel))
        except OSError:
            return False
        return True

    def write(self, files: dict[str, str]) -> None:
        """Write the model's (masked) texts, all or nothing.

        Every path is checked first (`refuse_reason`); then each text is
        unmasked against the file as it is on disk now; then every file is
        written by staging it next to its target and renaming it into place,
        keeping an existing file's permission bits. If any rename fails the
        files already replaced are put back and created ones removed. Parts of
        a document package are gathered per package, and the repacked package
        is one of the files written, in the same transaction.
        """
        self._check_write()
        cleaned: dict[str, str] = {}
        problems: list[str] = []
        for path, text in files.items():
            try:
                rel = _clean(path)
            except WorkspaceError as exc:
                problems.append(str(exc))
                continue
            reason = self.refuse_reason(rel, create=not self._exists(rel))
            if reason:
                problems.append(reason)
            elif rel in cleaned:
                problems.append(f"{rel}: given twice")
            else:
                cleaned[rel] = text
        if problems:
            raise WorkspaceError("\n".join(problems))

        changes: dict[str, tuple[bytes, int] | None] = {}
        package_edits: dict[str, dict[str, str]] = {}
        for rel, text in cleaned.items():
            split = self._package_split(rel)
            if split is not None:
                package_edits.setdefault(split[0], {})[split[1]] = text
                continue
            current = self._read_raw(rel)
            if current is None:
                if _PLACEHOLDER_RE.search(text):
                    raise WorkspaceError(
                        f"{rel}: contains an `anton:keep` placeholder that does not belong to this file. "
                        "Remove it; placeholders only stand for content in the file they were read from."
                    )
                new_text = text
            else:
                old_text = self._decode(rel, current[0])
                new_text = _restore_masks(rel, text, old_text, _masks_for(rel, old_text))
                self._check_component(rel, old_text, new_text, changes)
            try:
                data = new_text.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise WorkspaceError(f"{rel}: the text cannot be saved as UTF-8 ({exc.reason})") from exc
            if current is not None and data == current[0]:
                continue
            changes[rel] = (data, current[1] if current is not None else -1)
        for package, edits in package_edits.items():
            repacked = self._repack(package, edits)
            if repacked is not None:
                changes[package] = repacked
        if changes:
            self._commit(changes)

    def _check_component(
        self, rel: str, old: str, new: str, changes: dict[str, tuple[bytes, int] | None]
    ) -> None:
        """Hold an edit to a compiled presentation page to its program.

        Called after unmasking, so the shell and tail are already the bytes on
        disk; what is left to check is that nothing was added beside the
        program, and that the program is one `component_page.document` would
        compile. The recovery copy (`presentation-program.js`), when there is
        one, is kept in step for the primary page.
        """
        bounds = _component_bounds(old)
        if bounds is None:
            return
        begin, end = _component_markers()
        new_bounds = _component_bounds(new)
        if new_bounds is None:
            raise WorkspaceError(
                f"{rel}: the presentation markers `{begin}` and `{end}` must each stay exactly once, around "
                "the program."
            )
        start, stop = bounds
        new_start, new_stop = new_bounds
        if new[:new_start] != old[:start] or new[new_stop:] != old[stop:]:
            raise WorkspaceError(
                f"{rel}: this page is compiled from the presentation program between `{begin}` and `{end}`. "
                "Only that program can change; anything added outside the markers is lost when the page is "
                "next generated. Move the change into the program."
            )
        old_program = old[start + len(begin): stop - len(end)]
        program = new[new_start + len(begin): new_stop - len(end)]
        if program == old_program:
            return
        from anton.core.tools.generate_artifact import component_page

        try:
            component_page.document(program)
        except ValueError as exc:
            raise WorkspaceError(f"{rel}: the presentation program would not compile: {exc}") from exc
        if rel != self.primary:
            return
        try:
            copy = self._read_raw(PRESENTATION_PROGRAM_FILENAME)
        except WorkspaceError:
            return
        if copy is not None:
            changes[PRESENTATION_PROGRAM_FILENAME] = (program.strip().encode("utf-8") + b"\n", copy[1])

    # ── snapshot ─────────────────────────────────────────────────────

    def snapshot(self, paths: Iterable[str]) -> Snapshot:
        """Capture each path's bytes and mode, or its absence.

        A part is captured as its whole package, so `restore()` puts the
        package back byte for byte, whatever was repacked into it since.
        """
        saved: dict[str, tuple[bytes, int] | None] = {}
        for path in paths:
            rel = _clean(path)
            split = self._package_split(rel)
            if split is not None:
                rel = split[0]
            if rel not in saved:
                saved[rel] = self._read_raw(rel)
        return Snapshot(self, saved)

    # ── the transaction ──────────────────────────────────────────────

    @contextmanager
    def _parent(self, rel: str, *, create: bool) -> Iterator[tuple[int | None, str]]:
        """(descriptor or None, folder path) for `rel`'s parent, opened no-follow."""
        parts = tuple(rel.split("/"))
        with opened_dir_nofollow(self.folder, parts[:-1], create=create) as dir_fd:
            yield dir_fd, os.fspath(self.folder.joinpath(*parts[:-1]))

    @staticmethod
    def _at(dir_fd: int | None, folder: str, name: str) -> str:
        return name if dir_fd is not None else os.path.join(folder, name)

    def _stage(self, rel: str, data: bytes, mode: int) -> str:
        """Write `data` to a fresh temp file beside `rel`; return the temp's name."""
        name = rel.rsplit("/", 1)[-1]
        tmp = f".{name}.{uuid.uuid4().hex[:8]}{_TMP_SUFFIX}"
        with self._parent(rel, create=True) as (dir_fd, folder):
            if dir_fd is not None:
                try:
                    held = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    if not stat.S_ISREG(held.st_mode):
                        raise OSError(errno.EINVAL, "not a regular file", rel)
            fd = os.open(
                self._at(dir_fd, folder, tmp),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC | getattr(os, "O_BINARY", 0),
                0o666,
                dir_fd=dir_fd,
            )
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    if mode >= 0 and hasattr(os, "fchmod"):
                        os.fchmod(handle.fileno(), mode)
            except BaseException:
                with _quiet():
                    os.unlink(self._at(dir_fd, folder, tmp), dir_fd=dir_fd)
                raise
        return tmp

    def _discard(self, rel: str, tmp: str) -> None:
        with _quiet(), self._parent(rel, create=False) as (dir_fd, folder):
            os.unlink(self._at(dir_fd, folder, tmp), dir_fd=dir_fd)

    def _swap(self, rel: str, tmp: str) -> None:
        name = rel.rsplit("/", 1)[-1]
        with self._parent(rel, create=False) as (dir_fd, folder):
            _replace_at(self._at(dir_fd, folder, tmp), self._at(dir_fd, folder, name), dir_fd)

    def _remove(self, rel: str) -> None:
        name = rel.rsplit("/", 1)[-1]
        try:
            with self._parent(rel, create=False) as (dir_fd, folder):
                os.unlink(self._at(dir_fd, folder, name), dir_fd=dir_fd)
        except FileNotFoundError:
            pass

    def _commit(self, changes: dict[str, tuple[bytes, int] | None]) -> None:
        """Make every path hold its new bytes (None: be absent), or none of them.

        Phase one stages every new file beside its target, so a full disk or
        a refused folder fails before anything visible changed. Phase two
        renames (or deletes) each into place; a failure there puts back every
        path already changed, from bytes read just before.
        """
        self._check_write()
        originals: dict[str, tuple[bytes, int] | None] = {}
        for rel in changes:
            originals[rel] = self._read_raw(rel)
        missing_dirs = self._missing_parents(changes)

        staged: dict[str, str] = {}
        try:
            for rel, change in changes.items():
                if change is not None:
                    data, mode = change
                    staged[rel] = self._stage(rel, data, mode)
        except (OSError, WorkspaceError) as exc:
            for rel, tmp in staged.items():
                self._discard(rel, tmp)
            self._remove_dirs(missing_dirs)
            raise WorkspaceError(f"could not write {rel}: {_reason(exc)}; nothing was changed") from exc

        done: list[str] = []
        try:
            for rel, change in changes.items():
                if change is None:
                    self._remove(rel)
                else:
                    self._swap(rel, staged[rel])
                    del staged[rel]
                done.append(rel)
        except (OSError, WorkspaceError) as exc:
            for left_rel, tmp in staged.items():
                self._discard(left_rel, tmp)
            for back in reversed(done):
                with _quiet():
                    original = originals[back]
                    if original is None:
                        self._remove(back)
                    else:
                        self._swap(back, self._stage(back, original[0], original[1]))
            self._remove_dirs(missing_dirs)
            raise WorkspaceError(f"could not write {rel}: {_reason(exc)}; every file was put back") from exc

    def _missing_parents(self, changes: dict[str, tuple[bytes, int] | None]) -> list[Path]:
        """Folders the change would create, deepest first, to remove on failure."""
        missing: set[Path] = set()
        for rel, change in changes.items():
            if change is None:
                continue
            parent = self._target(rel).parent
            while parent != self.folder and not os.path.lexists(parent):
                missing.add(parent)
                parent = parent.parent
        return sorted(missing, key=lambda p: len(p.parts), reverse=True)

    def _remove_dirs(self, dirs: list[Path]) -> None:
        for folder in dirs:
            with _quiet():
                if os.path.isdir(folder) and not os.path.islink(folder):
                    os.rmdir(folder)


@contextmanager
def _quiet() -> Iterator[None]:
    try:
        yield
    except (OSError, WorkspaceError):
        pass


def _reason(exc: BaseException) -> str:
    if isinstance(exc, OSError) and exc.strerror:
        return exc.strerror
    return str(exc) or type(exc).__name__
