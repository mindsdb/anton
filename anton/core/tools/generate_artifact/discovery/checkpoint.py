"""`discovery.json` — the phase boundary that outlives the process.

The pipeline hands everything phase E needs across one boundary: the brief,
the PRD, the rendered data/web notes, the declared sources. On the hot path
that boundary is `GenState` in memory. This module is the same boundary
written down, so a second call — after `needs_confirmation`, after a budget
stop, after a failure — does not have to re-gather anything.

It also answers, deterministically, WHERE a repeat call should re-enter the
pipeline. Two independent checks, and neither guesses intent from free text:

  1. `request_fingerprint` — is this the same work? Only `user_request` is
     compared, normalized. The other three input fields are re-typed in the
     model's own words on every call, so a decision built on them would
     almost never match.
  2. `pipeline_stage` — how far did the previous call get? This, not a
     `confirmed` flag, is what says whether the brief was ever agreed:
     two fields describing one fact drift apart silently.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from anton.core.artifacts.internal_files import DISCOVERY_FILENAME

# ── Pipeline stages ─────────────────────────────────────────────────────────
# Written at each transition. A failure or a budget stop does NOT advance the
# stage: it stays at what the run actually completed, which is what makes a
# continuation cheap.
STAGE_AWAITING_CONFIRMATION = "awaiting_confirmation"
STAGE_PRD_WRITTEN = "prd_written"
STAGE_SPEC_WRITTEN = "spec_written"
STAGE_GENERATED = "generated"

# ── Entry points ────────────────────────────────────────────────────────────
ENTRY_FULL = "full"                    # A -> B -> C -> D -> E
ENTRY_CONFIRM = "resume_confirm"       # restore A, run B per design 2.6.1, C, D, E
ENTRY_SPEC = "resume_spec"             # restore A-C, run D, E
ENTRY_GENERATE = "resume_generate"     # restore A-D, run E
ENTRY_NEW_ITERATION = "new_iteration"  # restore A, run B with confirmation, C, D, E

_ENTRY_BY_STAGE = {
    STAGE_AWAITING_CONFIRMATION: ENTRY_CONFIRM,
    STAGE_PRD_WRITTEN: ENTRY_SPEC,
    STAGE_SPEC_WRITTEN: ENTRY_GENERATE,
    STAGE_GENERATED: ENTRY_NEW_ITERATION,
}

_WHITESPACE = re.compile(r"\s+")

# Separator that cannot occur in model-written text, so field boundaries stay
# unambiguous: without it ("ab", "") and ("a", "b") hash identically.
_FIELD_SEP = "\x00"


def normalize(text: str) -> str:
    """Trim the edges and collapse internal whitespace runs.

    The outer model re-types these fields every call; a stray newline must
    not cost a full re-run of gathering, including its questions to the user.
    """
    return _WHITESPACE.sub(" ", (text or "").strip())


def fingerprint(*parts: str) -> str:
    return hashlib.sha256(
        _FIELD_SEP.join(normalize(p) for p in parts).encode("utf-8")
    ).hexdigest()


def request_fingerprint(user_request: str) -> str:
    """Identity of the WORK. Decides full re-run vs. resume."""
    return fingerprint(user_request)


def call_fingerprint(
    agent_understanding: str, known_data: str, user_preferences: str
) -> str:
    """Identity of the CALL's soft fields. An optimization only.

    Used to skip a brief redraw when nothing changed. A wrong answer in
    either direction is cheap: a needless redraw costs one call, a missed one
    shows the user an unchanged brief they will still see reflected in the PRD.
    """
    return fingerprint(agent_understanding, known_data, user_preferences)


@dataclass
class DiscoveryCheckpoint:
    request_fingerprint: str = ""
    call_fingerprint: str = ""
    pipeline_stage: str = ""
    artifact_type: str = ""
    gathering_complete: bool = False
    declared_sources: list[str] = field(default_factory=list)
    # Sources with nothing executed against them. Persisted alongside the
    # rest because the emergency data loop's entry condition reads it, and a
    # condition that only exists in memory is not computable on a cold start.
    unverified_sources: list[str] = field(default_factory=list)
    brief_markdown: str = ""
    data_notes: str = ""
    web_notes: str = ""


def load(artifact_path: Path) -> DiscoveryCheckpoint | None:
    """Read the checkpoint, or None when there is nothing usable.

    Every failure mode degrades to None rather than raising: an artifact from
    before this feature has no file, and a corrupted one must cost a re-run,
    not the whole call. Unknown keys are dropped so a file written by a newer
    build does not break an older one.
    """
    path = artifact_path / DISCOVERY_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    known = {f.name for f in fields(DiscoveryCheckpoint)}
    return DiscoveryCheckpoint(**{k: v for k, v in raw.items() if k in known})


def save(artifact_path: Path, checkpoint: DiscoveryCheckpoint) -> None:
    (artifact_path / DISCOVERY_FILENAME).write_text(
        json.dumps(asdict(checkpoint), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def decide_entry(
    checkpoint: DiscoveryCheckpoint | None, *, request_fp: str
) -> str:
    """Where a call should enter the pipeline. Pure function, no I/O.

    An unrecognised stage falls back to the full path: the only safe reading
    of a checkpoint this build does not understand is "start over".
    """
    if checkpoint is None:
        return ENTRY_FULL
    if checkpoint.request_fingerprint != request_fp:
        return ENTRY_FULL
    return _ENTRY_BY_STAGE.get(checkpoint.pipeline_stage, ENTRY_FULL)
