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
# Same scope note as above: wording pins so a later edit can't silently drop
# sentences whose effect was measured live (fixture eval + a shadow replay of
# stored production verifier requests; numbers on the ticket). Behavioural
# coverage is tests/test_verifier_verdict_live.py (`honest_unobtainable_data`,
# `disclaimered_fabrication` and the three risk controls).


def test_stuck_definition_covers_an_honest_documented_gap():
    # An assistant that tried, could not obtain the data or tool, and said so
    # instead of guessing must be STUCK even with a partial deliverable. Before
    # this sentence the honest reply matched none of STUCK's thrash-keyed
    # clauses and was filed INCOMPLETE — with the blocker named in the
    # verifier's own reason — then force-continued into fabrication.
    stuck = _status_description()
    stuck = stuck[stuck.index("- STUCK:"):]
    assert "honest, documented gap" in stuck
    assert "TWO OR MORE distinct failed approaches" in stuck
    # haiku slipped to STUCK on the one-attempt control while its own reason
    # argued INCOMPLETE — 1/12, then 5/24 once the STUCK bullet started
    # DISCUSSING the single-attempt case (the label leaked onto the shape it
    # was excluding). So the STUCK bullet must not talk about one attempt at
    # all; that rule lives in the INCOMPLETE bullet.
    assert "one attempt" not in stuck.lower()
    assert "single failed attempt" not in stuck.lower()
    assert "Count the distinct" not in stuck
    # v2 said "EXHAUSTED the approaches available to it" and sonnet + gemini
    # read "I'll try another approach" as not-yet-exhausted, dropping the
    # ENG-836 wall from 6/6 STUCK to 6/6 INCOMPLETE. Exhaustion is now defined
    # by the failed attempts on record, not by the assistant's stated intent.
    assert "exhaust" not in stuck.lower()
    assert "judged on the failed attempts in the transcript, not on the assistant's stated intent" in stuck
    assert "whatever it says it will try next" in stuck
    assert "the required tool is absent from this environment" in stuck
    assert "missing values marked as unavailable" in stuck
    assert "the only way to 'keep going' would be to invent the values" in stuck
    # The premature-give-up control: honesty alone must not read as a blocker.
    assert "instead of guessing" in stuck


def test_incomplete_names_the_early_honest_stop_and_hands_exhaustion_to_stuck():
    # Two live findings shaped this bullet. A first draft said INCOMPLETE needs
    # "a concrete next step it has not tried", and the ENG-836 environment wall
    # lost 2 of 6 on mindshub_air because "I'll try another approach" read as
    # that step. And the one-attempt give-up control flipped to STUCK 5 of 6
    # because honesty alone read as a blocker. So INCOMPLETE now names the
    # early honest stop explicitly, and defers only EXHAUSTED gaps to STUCK.
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
    # v3 lost 1 of 3 on haiku to a COMPLETE that credited the "clear disclosure
    # that these are indicative estimates" — the disclaimer itself was being
    # read as honesty, so the clause now says what honesty about an
    # unobtained value looks like (marking it unavailable) and that a
    # disclosed-but-supplied value is not it.
    assert "or disclosing that they are not verified, does not make them obtained" in complete
    assert "an honest reply marks values it could not obtain as unavailable" in complete
    assert "never COMPLETE" in complete
    assert "delivering them is not a recovery" in complete
    # And the two legitimate shapes the rule must NOT catch are carved out in
    # the same sentence, not left to inference.
    assert "computed from data that DID arrive" in complete
    assert "an estimate the user explicitly asked for" in complete
    # Shadow replay: a job search that ran and found no matching listings was
    # re-scored STUCK. "Found none" is an answer, "could not look" is a blocker.
    assert "a genuine empty result" in complete
    assert "a COMPLETE answer of 'none', not a blocker" in complete


def test_close_to_done_is_never_true_on_an_unobtainable_gap():
    # Both honest replies in the incident came back `close_to_done: true`,
    # despite the field's own "nothing blocking or uncertain". Its only effect
    # is the spend-ceiling grace (ENG-1893), but it is the same misframing.
    desc = _VerifierVerdict.model_fields["close_to_done"].description
    assert "already tried and failed to obtain" in desc
    assert "a blocker, not a small remaining step" in desc


def test_waiting_covers_asking_for_input_after_a_failed_attempt():
    # Shadow replay of 214 production verdicts on an earlier ENG-2686 wording:
    # 8 of 31 WAITING turns re-scored INCOMPLETE, all "I tried, it failed,
    # please attach / share / connect it". Asking must never read as the
    # premature stop the INCOMPLETE bullet describes.
    desc = _status_description()
    waiting = desc[desc.index("- WAITING:"):desc.index("- INCOMPLETE:")]
    assert "provide, attach, re-upload, share, connect, or authorise" in waiting
    assert "even after a single failed attempt" in waiting
    assert "Asking is a valid stop, never a premature one." in waiting
    incomplete = desc[desc.index("- INCOMPLETE:"):desc.index("- STUCK:")]
    assert "If it instead asked the user for what it needs, that is WAITING" in incomplete
