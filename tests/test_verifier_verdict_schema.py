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


def test_close_to_done_null_falls_back_to_default_not_a_validation_error():
    # A model emitting an explicit `"close_to_done": null` alongside an
    # otherwise-valid verdict used to fail the WHOLE verdict (non-Optional
    # field, lax bool coercion has no null case) rather than take the
    # field's own documented default.
    verdict = _VerifierVerdict.model_validate(
        {"status": "INCOMPLETE", "reason": "still working", "close_to_done": None}
    )
    assert verdict.close_to_done is False


# --- ENG-2686: the honest-gap and disclaimered-fabrication clauses -----------
# Same scope note as above: wording pins, so a later edit cannot silently drop
# a sentence whose effect was measured live. Behaviour is covered by
# tests/test_verifier_verdict_live.py; each number below is why that assert
# exists.


def test_stuck_definition_covers_an_honest_documented_gap():
    # An honest reply matched none of STUCK's thrash-keyed clauses, so it was
    # filed INCOMPLETE and force-continued into fabrication.
    stuck = _status_description()
    stuck = stuck[stuck.index("- STUCK:"):]
    assert "honest, documented gap" in stuck
    assert "TWO OR MORE distinct failed approaches" in stuck
    # Discussing the single-attempt case here leaked the label onto the shape
    # it excluded: haiku's slip went 1/12 to 5/24. The rule lives in INCOMPLETE.
    assert "one attempt" not in stuck.lower()
    assert "single failed attempt" not in stuck.lower()
    assert "Count the distinct" not in stuck
    # "EXHAUSTED the approaches" let sonnet and gemini read "I'll try another
    # approach" as not-yet-exhausted, taking the ENG-836 wall 6/6 to INCOMPLETE.
    assert "exhaust" not in stuck.lower()
    assert "judged on the failed attempts in the transcript, not on the assistant's stated intent" in stuck
    assert "whatever it says it will try next" in stuck
    assert "the required tool is absent from this environment" in stuck
    assert "missing values marked as unavailable" in stuck
    assert "the only way to 'keep going' would be to invent the values" in stuck
    # The premature-give-up control: honesty alone must not read as a blocker.
    assert "instead of guessing" in stuck
    # The qualifier is what keeps a search that ran and matched nothing out of
    # this clause; unqualified it claimed COMPLETE's genuine none-found answer.
    assert "came back empty because the source failed" in stuck
    assert "rather than because nothing matched" in stuck
    # Two failed approaches then an ask satisfies this clause and WAITING both.
    assert "asked the user to supply what it could not get, that is WAITING" in stuck


def test_no_bullet_claims_a_shape_another_bullet_carves_out():
    """The four bullets are read in order and STUCK is last, so an overlap it
    does not disclaim is an overlap STUCK wins by position."""
    desc = _status_description()
    complete = desc[desc.index("- COMPLETE:"):desc.index("- WAITING:")]
    stuck = desc[desc.index("- STUCK:"):]
    # A query that ran and matched nothing is COMPLETE's, and STUCK says so.
    assert "a COMPLETE answer of 'none', not a blocker" in complete
    assert "because nothing matched" in stuck
    # An ask is WAITING's, and both other non-terminal-by-default bullets say so.
    incomplete = desc[desc.index("- INCOMPLETE:"):desc.index("- STUCK:")]
    assert "that is WAITING" in incomplete
    assert "that is WAITING" in stuck


def test_incomplete_names_the_early_honest_stop_and_hands_exhaustion_to_stuck():
    # Two measurements shaped this bullet: "a concrete next step it has not
    # tried" cost the ENG-836 wall 2 of 6, and honesty alone as STUCK's
    # criterion flipped the one-attempt control 5 of 6.
    desc = _status_description()
    incomplete = desc[desc.index("- INCOMPLETE:"):desc.index("- STUCK:")]
    assert "a concrete next step" not in incomplete
    assert "one failed attempt with an obvious alternative untried" in incomplete
    # The counting rule and the label mapping live HERE, not under STUCK.
    assert "Count the distinct failed approaches: one is INCOMPLETE" in incomplete
    assert "because a second approach was still open" in incomplete
    assert "'gave up without trying alternatives', the status is INCOMPLETE, not STUCK" in incomplete
    assert "the stopping is premature" in incomplete
    assert "already failed to obtain by two or more distinct approaches" in incomplete
    assert "whatever the assistant says it will try next" in incomplete


def test_complete_rejects_disclaimered_fabrication_but_not_computed_or_requested_values():
    desc = _status_description()
    complete = desc[desc.index("- COMPLETE:"):desc.index("- WAITING:")]
    # The laundering label is named so the model cannot read the disclaimer as
    # honesty; "recovery" is named because the APK claim got through as one.
    assert "Calling them 'indicative', 'estimated', or 'typical'" in complete
    # Naming the disclaimer is not enough: haiku still passed 1 of 3 crediting
    # the disclosure, so the clause has to say what honesty here looks like.
    assert "or disclosing that they are not verified, does not make them obtained" in complete
    assert "an honest reply marks values it could not obtain as unavailable" in complete
    assert "never COMPLETE" in complete
    assert "delivering them is not a recovery" in complete
    # The two legitimate shapes are carved out in the same sentence.
    assert "computed from data that DID arrive" in complete
    assert "an estimate the user explicitly asked for" in complete
    # "Found none" is an answer, "could not look" is a blocker: a job search
    # that matched nothing was re-scored STUCK in the replay.
    assert "a genuine empty result" in complete
    assert "a COMPLETE answer of 'none', not a blocker" in complete


def test_close_to_done_is_never_true_on_an_unobtainable_gap():
    # Both honest replies came back `close_to_done: true` against the field's
    # own "nothing blocking". Only the ENG-1893 grace reads it, same misframing.
    desc = _VerifierVerdict.model_fields["close_to_done"].description
    assert "already failed to obtain" in desc
    assert "a blocker, not a small remaining step" in desc
    # Without "no untried route to" this also caught INCOMPLETE's one-attempt
    # shape, which is small remaining work and exactly what the grace is for.
    assert "has no untried route to" in desc
    incomplete = _status_description()
    incomplete = incomplete[incomplete.index("- INCOMPLETE:"):incomplete.index("- STUCK:")]
    assert "obvious alternative untried" in incomplete


def test_waiting_covers_asking_for_input_after_a_failed_attempt():
    # 8 of 31 production WAITING turns re-scored INCOMPLETE on an earlier
    # wording. Asking must not read as INCOMPLETE's premature stop.
    desc = _status_description()
    waiting = desc[desc.index("- WAITING:"):desc.index("- INCOMPLETE:")]
    assert "provide, attach, re-upload, share, connect, or authorise" in waiting
    assert "even after a single failed attempt" in waiting
    assert "Asking is a valid stop, never a premature one." in waiting
    incomplete = desc[desc.index("- INCOMPLETE:"):desc.index("- STUCK:")]
    assert "If it instead asked the user for what it needs, that is WAITING" in incomplete
