"""Pins the _VerifierVerdict STUCK definition's environment-wall coverage (ENG-836).

Honest scope note: these are wording pins, not behavioural coverage — they
exist so a later edit can't silently drop the sentences whose effect was
measured live (0-1/12 STUCK without them vs 12/12 with them on a blocked-task
transcript, controls unmoved; see the 2026-08-04 A/B on ENG-836). Behavioural
verifier coverage is ENG-1211's scope.
"""

from __future__ import annotations

from anton.core.session import _VerifierVerdict


def _status_description() -> str:
    return _VerifierVerdict.model_fields["status"].description


def test_stuck_definition_names_environment_walls():
    desc = _status_description()
    # The uninstallable-OS-dependency case (no root, package manager blocked)
    # must be named INSIDE the STUCK bullet — the ENG-836 incident class.
    stuck = desc[desc.index("- STUCK:"):]
    assert "cannot be installed" in stuck
    assert "no root/sudo" in stuck
    assert "package manager blocked" in stuck


def test_stuck_definition_covers_repeated_workarounds():
    # The blocked loop's signature: the assistant keeps proposing another
    # approach to the same wall. Without this sentence the assistant's own
    # "Next I'll retry..." narration satisfies INCOMPLETE's "could keep going
    # on its own" and the loop force-continues into the wall.
    stuck = _status_description()
    stuck = stuck[stuck.index("- STUCK:"):]
    assert "Repeated failed workarounds for the same underlying blocker" in stuck
    assert "even if the assistant says it will try another approach" in stuck


def test_recovered_error_rule_is_intact():
    # ENG-1134's guard must survive the STUCK extension: a tool error the
    # assistant recovered from stays COMPLETE-eligible.
    desc = _status_description()
    assert "RECOVERED" in desc
    assert "do NOT mark a turn incomplete just because an earlier tool call failed" in desc


def test_close_to_done_defaults_false_and_is_bounded():
    # `close_to_done` unlocks extra spend on an INCOMPLETE verdict with no
    # human in the loop, so an under-specified model self-report degrades the
    # whole feature to a no-op: wording that stays a small, bounded claim
    # (not "probably fine") is what keeps a model from defaulting to it.
    desc = _VerifierVerdict.model_fields["close_to_done"].description
    assert _VerifierVerdict.model_fields["close_to_done"].default is False
    assert "roughly 1-3 more tool calls" in desc
    assert "nothing blocking or uncertain" in desc
    assert "an unsure model errs toward asking the user" in desc
