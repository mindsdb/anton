"""The ENG-2248 verdict migration is NOT measurement-only — proof, both ways.

`_apply_error_tracking` opens with `if ok is not None:` and uses the handler's
verdict DIRECTLY to drive `error_streak[tool_name]`, which fires the resilience
nudge at `resilience_nudge_at` and the circuit breaker at
`max_consecutive_errors`. So every `ok=` in a migrated handler is a behaviour
decision, not a label, and these tests exist to make that explicit instead of
discovered in production.

Two claims, one per direction:

* **Tier 1 is free.** A migrated SUCCESS must behave exactly as the bare-string
  return did — the streak resets either way, because the legacy substring
  matcher finds none of its five markers in a success body. If this ever
  diverges, `ok=True` has stopped being a no-op and the "zero behaviour risk"
  argument for tier 1 is void.
* **Tier 2 is a real, accepted change.** A migrated FAILURE now reaches the
  streak where a bare string did not. Two in a row nudges, five trips the
  breaker. That is intended for calls that cannot succeed on retry, and it is
  asserted here so nobody has to infer it from the diff.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools.registry import ToolOutcome
from tests.conftest import make_mock_llm

_LEGACY_MARKERS = ("[error]", "Task failed:", "failed", "timed out", "Rejected:")


def _bare_session():
    """Minimal session: only `_apply_error_tracking` is under test."""
    return ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))


# ── Tier 1: a migrated success must be behaviourally identical ───────


def test_a_migrated_success_resets_the_streak_exactly_as_a_bare_string_did():
    """The whole "tier 1 is free" claim, pinned.

    Runs the SAME success body twice — once as `ok=None` (what shipped before
    the migration, classified by the legacy matcher) and once as `ok=True` —
    and requires an identical streak outcome. Not "both are falsy": the same
    value, from a primed non-zero streak, so a reset is observable.
    """
    session = _bare_session()
    body = "# Skill recalled: `csv-summary`\n\nLoad the CSV, infer types."

    # Precondition that makes this test able to fail: the success body must
    # contain none of the legacy markers, which is WHY ok=None reset the
    # streak. If a future success body happens to contain "failed", the two
    # paths genuinely diverge and this test should be the thing that says so.
    assert not any(m in body for m in _LEGACY_MARKERS)

    outcomes = {}
    for label, ok in (("legacy", None), ("migrated", True)):
        streak = {"recall_skill": 3}          # primed, so a reset is visible
        session._apply_error_tracking(
            body, "recall_skill", streak, set(), ok=ok
        )
        outcomes[label] = streak["recall_skill"]

    assert outcomes["legacy"] == outcomes["migrated"] == 0, outcomes


def test_a_migrated_success_appends_nothing_to_what_the_model_reads():
    """Tier 1 must not change the tool result the model sees, either."""
    session = _bare_session()
    body = "# Skill recalled: `csv-summary`"
    legacy = session._apply_error_tracking(body, "recall_skill", {}, set(), ok=None)
    migrated = session._apply_error_tracking(body, "recall_skill", {}, set(), ok=True)
    assert legacy == migrated == body


# ── Tier 2: the accepted behaviour change ────────────────────────────


def test_a_migrated_failure_now_reaches_the_streak_where_a_bare_string_did_not():
    """The change, isolated to one call.

    `recall_skill`'s missing-label text carries none of the five legacy
    markers, so before the migration it RESET the streak. Now it increments.
    """
    session = _bare_session()
    body = (
        "ERROR: recall_skill requires a non-empty 'label' parameter. "
        "Pick one from the procedural memory list in your system prompt."
    )
    assert not any(m in body for m in _LEGACY_MARKERS)   # why it used to reset

    legacy_streak = {"recall_skill": 0}
    session._apply_error_tracking(body, "recall_skill", legacy_streak, set(), ok=None)
    assert legacy_streak["recall_skill"] == 0, "pre-migration behaviour changed"

    migrated_streak = {"recall_skill": 0}
    session._apply_error_tracking(body, "recall_skill", migrated_streak, set(), ok=False)
    assert migrated_streak["recall_skill"] == 1


def test_two_consecutive_migrated_failures_nudge_the_model_accepted():
    """ACCEPTED CONSEQUENCE, asserted rather than discovered.

    At `resilience_nudge_at` the tool result gains advice text the model reads.
    For a call that cannot succeed on retry — no label supplied — that is the
    intended outcome of the migration.
    """
    session = _bare_session()
    body = "ERROR: recall_skill requires a non-empty 'label' parameter."
    streak: dict[str, int] = {}
    nudged: set[str] = set()

    first = session._apply_error_tracking(body, "recall_skill", streak, nudged, ok=False)
    assert streak["recall_skill"] == 1
    assert first == body, "no nudge before the threshold"

    second = session._apply_error_tracking(body, "recall_skill", streak, nudged, ok=False)
    assert streak["recall_skill"] == session._resilience_nudge_at
    assert len(second) > len(body), "the nudge is appended to what the model reads"
    assert "recall_skill" in nudged


def test_five_consecutive_migrated_failures_trip_the_breaker_accepted():
    """ACCEPTED CONSEQUENCE: at `max_consecutive_errors` the model is told to stop."""
    session = _bare_session()
    body = "ERROR: recall_skill requires a non-empty 'label' parameter."
    streak: dict[str, int] = {}
    nudged: set[str] = set()

    out = body
    for _ in range(session._max_consecutive_errors):
        out = session._apply_error_tracking(body, "recall_skill", streak, nudged, ok=False)

    assert streak["recall_skill"] == session._max_consecutive_errors
    assert "Stop retrying this approach" in out
    assert f"failed {session._max_consecutive_errors} times" in out


def test_one_success_clears_a_streak_built_from_migrated_failures():
    """The streak is consecutive, so the blast radius of tier 2 is bounded.

    Without this, "two failures nudge" reads as a permanent state. It is not:
    a single success resets the counter AND re-arms the nudge.
    """
    session = _bare_session()
    fail = "ERROR: recall_skill requires a non-empty 'label' parameter."
    streak: dict[str, int] = {}
    nudged: set[str] = set()
    for _ in range(3):
        session._apply_error_tracking(fail, "recall_skill", streak, nudged, ok=False)
    assert streak["recall_skill"] == 3 and "recall_skill" in nudged

    session._apply_error_tracking("# Skill recalled: `x`", "recall_skill",
                                  streak, nudged, ok=True)
    assert streak["recall_skill"] == 0
    assert "recall_skill" not in nudged


# ── Tier 3: the deliberately-unmigrated returns ──────────────────────


@pytest.mark.asyncio
async def test_the_no_match_family_is_still_unverdicted_on_purpose():
    """Tier 3 must stay `ok=None` — a bare string, not a ToolOutcome.

    `NO MATCH` is the most common non-success on the highest-volume tool. If a
    later change quietly gives it a verdict, normal skill exploration starts
    feeding the breaker, and this test is the thing that notices.
    """
    import tempfile
    from pathlib import Path

    from anton.core.memory.skills import SkillStore
    from anton.core.tools.recall_skill import handle_recall_skill

    with tempfile.TemporaryDirectory() as tmp:
        store = SkillStore(root=Path(tmp) / "skills")
        # Same minimal shape `test_recall_skill.py::_session_with` uses.
        session = SimpleNamespace(_skill_store=store)
        result = await handle_recall_skill(session, {"label": "does_not_exist"})

    assert not isinstance(result, ToolOutcome), (
        "NO MATCH gained a verdict — see the tier-3 comment in recall_skill.py; "
        "this needs its own ticket and a before/after on the nudge rate"
    )
    assert "NO MATCH" in result


# ── The migrated handlers themselves, end to end ─────────────────────
#
# The tests above pin the MECHANISM (`_apply_error_tracking`). These pin the
# handlers: that each one emits the verdict the tier table claims, and that the
# text the model reads is byte-identical to what it read before the migration.


def _read_image(tmp_path, **args):
    import asyncio

    from anton.core.tools.tool_handlers import handle_read_image

    session = SimpleNamespace(_workspace=SimpleNamespace(base=tmp_path))
    return asyncio.run(handle_read_image(session, args))


# A real 1x1 PNG, so the success path runs the whole base64/media-type chain
# rather than a mocked stand-in.
_PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d4944415478da63f8ffff3f0005fe02fea735d18b00"
    "00000049454e44ae426082"
)


@pytest.mark.parametrize("args, expected_text, expected_reason", [
    ({}, "Error: file_path is required.", "missing_name"),
    ({"file_path": "  "}, "Error: file_path is required.", "missing_name"),
])
def test_read_image_malformed_calls_are_failures_with_the_same_text(
    tmp_path, args, expected_text, expected_reason
):
    outcome = _read_image(tmp_path, **args)
    assert outcome.content == expected_text       # unchanged for the model
    assert outcome.ok is False                    # newly visible to the streak
    assert outcome.reason == expected_reason


def test_read_image_absent_file_is_a_failure_that_cannot_trip_the_breaker(tmp_path):
    """`ok=False` for the streak, TIER_SELF for the ledger — both deliberate.

    The streak SHOULD climb: re-reading a path that does not exist is thrash.
    The root-cause ledger should NOT, because the agent chose the path and can
    list the directory — resolving away from `external_wall` exactly as
    `artifact_not_found` does.
    """
    from anton.core.root_cause import classify

    outcome = _read_image(tmp_path, file_path="nope.png")
    assert outcome.ok is False
    assert outcome.content.startswith("Error: file not found: ")
    assert not classify(outcome.reason).trip_eligible


def test_read_image_a_non_image_is_a_failure_the_model_can_fix(tmp_path):
    from anton.core.root_cause import classify

    (tmp_path / "notes.txt").write_text("not a picture")
    outcome = _read_image(tmp_path, file_path="notes.txt")
    assert outcome.ok is False
    assert "is not a supported image format" in outcome.content
    assert not classify(outcome.reason).trip_eligible


def test_read_image_success_is_the_same_content_blocks_plus_a_verdict(tmp_path):
    (tmp_path / "shot.png").write_bytes(_PNG_1X1)
    outcome = _read_image(tmp_path, file_path="shot.png")

    assert outcome.ok is True
    # The model's view is unchanged: the same two blocks, same order.
    assert isinstance(outcome.content, list)
    assert [b["type"] for b in outcome.content] == ["image", "text"]
    assert outcome.content[0]["source"]["media_type"] == "image/png"
    assert outcome.content[1]["text"].startswith("Loaded shot.png (")


def test_read_image_never_pairs_a_list_with_a_failure_verdict():
    """The seam that keeps two documented gaps unreachable.

    Both tool loops' list arms skip the nudge/breaker TEXT and skip
    `_record_root_cause` (see the NOTE at session.py's non-streaming list arm
    and `_tool_failure_cause`'s docstring). A handler returning
    `ToolOutcome(content=[...], ok=False)` would therefore climb the streak,
    trip nothing the model can see, and put a cause on no turn tally.

    `read_image` is the first and only multimodal handler, and it returns a
    list ONLY on success. This is an AST scan rather than a call-path test
    because the property worth protecting is "nobody writes that shape",
    across the whole tree — not "today's inputs don't reach it".
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "anton"
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "ToolOutcome"):
                continue
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            content = kwargs.get("content")
            if not isinstance(content, (ast.List, ast.ListComp)):
                continue
            ok = kwargs.get("ok")
            if not (isinstance(ok, ast.Constant) and ok.value is True):
                offenders.append(f"{path.name}:{node.lineno}")

    assert not offenders, (
        "multimodal ToolOutcome without ok=True: " + ", ".join(offenders)
        + " — a list result must be a success, or the list arms of both tool "
          "loops must first learn to append the nudge and record the cause"
    )


def _skill_draft(tmp_path, store=None, **args):
    import asyncio

    from anton.core.tools.skill_draft import handle_create_skill_draft

    session = SimpleNamespace(_skill_drafts_root=tmp_path, _skill_store=store)
    return asyncio.run(handle_create_skill_draft(session, args))


def test_create_skill_draft_success_is_a_verdict_over_the_same_json(tmp_path):
    import json

    outcome = _skill_draft(tmp_path / "drafts", name="Competitive Analysis")
    assert outcome.ok is True
    payload = json.loads(outcome.content)        # still the same JSON envelope
    assert "skill_file" in payload and "error" not in payload


def test_create_skill_draft_with_no_store_is_an_accepted_wall(tmp_path):
    """The one tier-2 case in this PR that IS trip-eligible, stated on purpose.

    No drafts root means the host wired no store: no argument the agent can
    pass will work, and `store_unavailable` was already TIER_WALL before this
    PR (`recall_skill` uses it). So five of these in a row can trip the
    breaker — which is the correct outcome for a capability that is absent.
    """
    from anton.core.root_cause import classify

    outcome = _skill_draft(None, name="anything")
    assert outcome.ok is False
    assert "unavailable" in outcome.content
    assert classify(outcome.reason).trip_eligible
