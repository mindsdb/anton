"""Keep a requested geographic view attached to its page for this turn.

Generation knows its entry page. Direct writes can identify only an explicit
primary or a sole HTML page; uncertain attribution remains advisory. The
requirement comes from the request, never from a repaired page's contents.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
from typing import TYPE_CHECKING

from anton.core.artifacts.geography import GeographyCheck, GeographyRequirement, check_geography, classify_request
from anton.utils.nofollow import read_bytes_nofollow

if TYPE_CHECKING:
    from anton.core.session import ChatSession
    from anton.core.tools.generate_artifact.state import GenState
    from anton.core.tools.tool_handlers import PageKey, _ChangedPage


_MULTIPLE_OUTPUTS = re.compile(
    r"\b(?:separate|another|additional)\b|\band\s+(?:a\s+|supporting\s+)?"
    r"(?:notes|summary|table|report|dashboard|page|artifact)\b", re.I,
)
_SUPPORTING_PAGE = re.compile(r"\b(?:notes|sources|appendix|readme|methodology|caveats)\b", re.I)
_SUPPORTING_OUTPUT = re.compile(
    r"^\s*(?:(?:please\s+)?(?:create|build|write|generate|make|produce|add)\s+"
    r"(?:(?:a|an|the|supporting|separate)\s+)*)?" + _SUPPORTING_PAGE.pattern,
    re.I,
)


def _merge(left: GeographyRequirement | None, right: GeographyRequirement | None) -> GeographyRequirement | None:
    if left is None:
        return right
    if right is None:
        return left
    kind = left.kind if left.kind == right.kind else "map_and_globe"
    return GeographyRequirement(kind, required=left.required or right.required)


def requirements_of(session: ChatSession) -> dict[PageKey, GeographyRequirement]:
    requirements = getattr(session, "_geography_requirements", None)
    if not isinstance(requirements, dict):
        requirements = {}
        session._geography_requirements = requirements
    return requirements


def generated_requirement(state: GenState, entry: Path) -> GeographyRequirement | None:
    from anton.core.tools.tool_handlers import PageKey

    key = PageKey(state.slug, entry.relative_to(state.artifact_path).as_posix())
    requirements = requirements_of(state.session)
    # Use the active user request when a delegate abbreviates it. For several
    # requested outputs, the delegate's specific request identifies its page.
    requirement = classify_request(state.user_request)
    active = getattr(state.session, "_artifact_geography_request", "")
    active = active if isinstance(active, str) else ""
    original = classify_request(active)
    supporting_only = requirement is None and bool(_SUPPORTING_OUTPUT.search(state.user_request))
    if original is not None and not supporting_only:
        if not _MULTIPLE_OUTPUTS.search(active):
            requirement = _merge(requirement, original)
        elif requirement is None:
            requirement = replace(original, required=False)
    # A repair's abbreviated request cannot remove a view. A stronger request
    # may add another kind of view to the same page.
    requirement = _merge(requirements.get(key), requirement)
    if requirement is not None:
        requirements[key] = requirement
    return requirements.get(key)


def check_pages(
    store,
    pages: list[_ChangedPage],
    *,
    request: str,
    requirements: dict[PageKey, GeographyRequirement],
    size_limit: int,
) -> dict[PageKey, GeographyCheck]:
    """Check edited pages, retaining a known requirement across repairs.

    Multiple changed artifacts need geographic metadata to identify a target.
    Once a target is known, a later, unrelated artifact does not inherit it.
    """
    request = request if isinstance(request, str) else ""
    intent = classify_request(request)
    multiple_outputs = bool(_MULTIPLE_OUTPUTS.search(request))
    by_slug: dict[str, list[_ChangedPage]] = {}
    for page in pages:
        by_slug.setdefault(page.slug, []).append(page)
    if intent is not None:
        for slug, group in by_slug.items():
            metadata = store.read_metadata(slug)
            primary = metadata.primary if metadata is not None else None
            target = next((page for page in group if page.key(store).page == primary), None)
            if target is None and len(group) == 1:
                target = group[0]
            if target is None:
                for page in group:
                    if page.edited:
                        requirements.setdefault(page.key(store), replace(intent, required=False))
                continue
            key = target.key(store)
            if key in requirements and requirements[key].required:
                continue
            description = f"{metadata.name} {metadata.description}" if metadata is not None else ""
            corroborating = classify_request(description)
            if corroborating is None and _SUPPORTING_PAGE.search(description):
                continue
            known_elsewhere = any(other.slug != slug and value.required for other, value in requirements.items())
            if known_elsewhere and corroborating is None:
                continue
            confident = (
                len(by_slug) == 1 and not multiple_outputs
            ) or (corroborating is not None and corroborating.required)
            # Separate geographic outputs inherit their own attributed view,
            # not every view requested across the whole turn.
            attributed = (
                corroborating
                if multiple_outputs and intent.kind == "map_and_globe"
                and corroborating is not None and corroborating.required
                else intent
            )
            requirements[key] = attributed if confident else replace(attributed, required=False)

    findings = {}
    for page in pages:
        key = page.key(store)
        requirement = requirements.get(key)
        if requirement is None or not page.edited or page.size > size_limit:
            continue
        try:
            html = read_bytes_nofollow(store.root / page.slug, page.path, limit=size_limit).decode("utf-8", errors="replace")
        except OSError:
            continue
        findings[key] = check_geography(html, requirement)
    return findings
