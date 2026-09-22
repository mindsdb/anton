"""Structural lint for `.pptx`/`.docx` artifacts: is this a package Office can open?

A deck the agent wrote and then called validated did not open (ENG-2175).
The shapes that produce that are all visible without an Office install:
a truncated write (no zip central directory), a member whose bytes are
damaged, a hand-rolled package missing `[Content_Types].xml` or
`_rels/.rels`, a relationship pointing at a part that was never written, a
malformed XML part, an `r:id` with no matching relationship, or a valid
package of the wrong kind (a Word document named `.pptx`).

Stdlib only (`zipfile` + `xml.etree`): python-pptx is not a dependency,
and reading a package's parts and relationships needs nothing it adds.

Known limitation: this checks the package, not the schema. A part that
parses but carries an element PowerPoint rejects still passes; the
LibreOffice oracle (`office_open_check.py`) is the second line for that.
"""

from __future__ import annotations

import posixpath
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree as ET

_CT_NS = "{http://schemas.openxmlformats.org/package/2006/content-types}"
_PR_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
# Attributes in this namespace (`r:id`, `r:embed`, `r:link`, ...) name a
# relationship of the part they sit in.
_R_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_OFFICE_DOCUMENT_REL = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
)

# Main-part content type prefix per format. Prefix, not exact match, so the
# template/slideshow/macro-enabled variants of the same format still count.
_MAIN_CONTENT_TYPE = {
    "pptx": ("application/vnd.openxmlformats-officedocument.presentationml.", "a presentation"),
    "docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.", "a document"),
}

# Enough to tell the agent what to fix without flooding its tool result.
_MAX_FINDINGS = 20

# A zip bomb must not stall the agent's turn: parts above this are not parsed.
_MAX_XML_PART_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class PackageFinding:
    """One reason the package will not open as the format its name claims."""

    detail: str

    def message(self) -> str:
        return f"file will not open as saved: {self.detail}"


def lint_pptx(path: Path) -> list[PackageFinding] | None:
    return lint_ooxml(path, "pptx")


def lint_docx(path: Path) -> list[PackageFinding] | None:
    return lint_ooxml(path, "docx")


def lint_ooxml(path: Path, kind: str) -> list[PackageFinding] | None:
    """Check `path` is a well-formed OOXML package of `kind`.

    Returns `None` only when the file could not be read at all (missing,
    permission denied), so nothing is claimed either way. A file that reads
    but is not a valid package is a finding, never `None`: for this format
    "does it open" is the whole question. Never raises.
    """
    try:
        if not path.is_file():
            return None
    except OSError:
        return None
    try:
        findings = _lint(path, kind)
    except OSError:
        return None
    except Exception as exc:  # anything else zipfile/expat can throw is a broken file
        findings = [PackageFinding(f"not a valid {kind} package ({type(exc).__name__}: {exc})")]
    if len(findings) > _MAX_FINDINGS:
        extra = len(findings) - _MAX_FINDINGS
        findings = findings[:_MAX_FINDINGS] + [PackageFinding(f"and {extra} more problems")]
    return findings


def _lint(path: Path, kind: str) -> list[PackageFinding]:
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        return [PackageFinding(f"not a valid {kind} package, the zip container is damaged or truncated ({exc})")]

    with zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            return [PackageFinding(f"not a valid {kind} package, part '{bad_member}' is corrupt (checksum mismatch)")]

        names = {n for n in zf.namelist() if not n.endswith("/")}
        findings: list[PackageFinding] = []

        parsed: dict[str, ET.Element] = {}
        for name in sorted(names):
            if not name.endswith((".xml", ".rels")):
                continue
            if zf.getinfo(name).file_size > _MAX_XML_PART_BYTES:
                continue
            try:
                parsed[name] = ET.fromstring(zf.read(name))
            except ET.ParseError as exc:
                findings.append(PackageFinding(f"part '{name}' is not well-formed XML ({exc})"))

        content_types = parsed.get("[Content_Types].xml")
        if content_types is None:
            if "[Content_Types].xml" not in names:
                findings.append(PackageFinding("required part '[Content_Types].xml' is missing"))
            return findings
        defaults, overrides = _content_types(content_types)

        for part in sorted(names - {"[Content_Types].xml"}):
            if part not in overrides and _extension(part) not in defaults:
                findings.append(PackageFinding(f"part '{part}' has no content type in [Content_Types].xml"))
        for part in sorted(set(overrides) - names):
            findings.append(PackageFinding(f"[Content_Types].xml lists part '{part}', which is missing"))

        if "_rels/.rels" not in names:
            findings.append(PackageFinding("required part '_rels/.rels' is missing"))
            return findings

        rels_by_source = {
            _rels_source(name): _relationships(root)
            for name, root in parsed.items()
            if name.endswith(".rels")
        }

        for source, rels in sorted(rels_by_source.items()):
            base = posixpath.dirname(source)
            for rid, (target, external) in sorted(rels.items()):
                if external:
                    continue
                resolved = _resolve(base, target)
                if resolved not in names:
                    where = source or "the package"
                    findings.append(
                        PackageFinding(f"relationship {rid} of '{where}' points at '{resolved}', which is missing")
                    )

        main = [
            _resolve("", target)
            for target, external, rel_type in _relationships_with_type(parsed.get("_rels/.rels"))
            if rel_type == _OFFICE_DOCUMENT_REL and not external
        ]
        prefix, noun = _MAIN_CONTENT_TYPE[kind]
        if not main:
            findings.append(PackageFinding("'_rels/.rels' names no main document part"))
        elif main[0] in names:
            main_type = overrides.get(main[0]) or defaults.get(_extension(main[0]), "")
            if not (main_type.startswith(prefix) and main_type.endswith("main+xml")):
                findings.append(
                    PackageFinding(
                        f"main part '{main[0]}' is not {noun} (content type '{main_type or 'none'}'),"
                        f" so the file is not a .{kind}"
                    )
                )

        for name, root in sorted(parsed.items()):
            if name.endswith(".rels") or name == "[Content_Types].xml":
                continue
            known = rels_by_source.get(name, {})
            missing = sorted(
                {
                    value
                    for el in root.iter()
                    for attr, value in el.attrib.items()
                    if attr.startswith(_R_NS) and value and value not in known
                }
            )
            for rid in missing:
                findings.append(
                    PackageFinding(f"part '{name}' refers to relationship '{rid}', which its .rels does not define")
                )

    return findings


def _content_types(root: ET.Element) -> tuple[dict[str, str], dict[str, str]]:
    defaults = {
        el.get("Extension", "").lower(): el.get("ContentType", "")
        for el in root.iter(f"{_CT_NS}Default")
    }
    overrides = {
        el.get("PartName", "").lstrip("/"): el.get("ContentType", "")
        for el in root.iter(f"{_CT_NS}Override")
    }
    return defaults, overrides


def _relationships_with_type(root: ET.Element | None) -> list[tuple[str, bool, str]]:
    if root is None:
        return []
    return [
        (el.get("Target", ""), el.get("TargetMode") == "External", el.get("Type", ""))
        for el in root.iter(f"{_PR_NS}Relationship")
    ]


def _relationships(root: ET.Element) -> dict[str, tuple[str, bool]]:
    return {
        el.get("Id", ""): (el.get("Target", ""), el.get("TargetMode") == "External")
        for el in root.iter(f"{_PR_NS}Relationship")
    }


def _extension(part: str) -> str:
    """Lower-case extension as [Content_Types].xml keys it. Not
    `posixpath.splitext`, which reads `_rels/.rels` as having none."""
    base = posixpath.basename(part)
    return base.rsplit(".", 1)[1].lower() if "." in base else ""


def _rels_source(rels_name: str) -> str:
    """`ppt/slides/_rels/slide1.xml.rels` -> `ppt/slides/slide1.xml`;
    `_rels/.rels` -> `` (the package itself)."""
    folder, file = posixpath.split(rels_name)
    parent = posixpath.dirname(folder)
    source = file[: -len(".rels")]
    return posixpath.join(parent, source) if source else ""


def _resolve(base: str, target: str) -> str:
    target = unquote(target)
    joined = target.lstrip("/") if target.startswith("/") else posixpath.join(base, target)
    return posixpath.normpath(joined)
