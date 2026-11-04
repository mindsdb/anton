"""What Jev is asked about a sidekick at work (`jev.decide`, role `decision`)."""

from __future__ import annotations

# Every check asks all four: one request costs the same either way, and the
# policy reads `done` only when the sidekick is finishing.
TAKE_BACK = "take_back"
STUCK = "stuck"
OFF_SCOPE = "off_scope"
DONE = "done"

# The whole budget of one check. Jev answers in well under a second; a slower
# reply is not worth holding a delegation round for, and the check fails open.
DECISION_TIMEOUT_S = 2.0


def sidekick_questions() -> dict:
    """The four yes/no (`noul`) questions, each answered as a probability of yes."""
    return {
        TAKE_BACK: {
            "type": "noul",
            "instructions": (
                "Should the main agent take this task back from the sidekick now? Yes when "
                "the work needs a decision the brief does not settle, input only the user "
                "can give, or judgment beyond the brief, or when the sidekick is making "
                "things worse."
            ),
        },
        STUCK: {
            "type": "noul",
            "instructions": (
                "Is the sidekick stuck: repeating a step that keeps failing, going round in "
                "circles, or making no progress over its last few rounds?"
            ),
        },
        OFF_SCOPE: {
            "type": "noul",
            "instructions": (
                "Is the sidekick working on something the brief did not ask for?"
            ),
        },
        DONE: {
            "type": "noul",
            "instructions": (
                "Does the transcript show the done-when condition is met, with evidence "
                "rather than only a claim?"
            ),
        },
    }


def sidekick_state(*, brief: str, done_when: str, transcript: str) -> dict:
    """The state one check is about."""
    return {
        "request": f"{brief.strip()}\n\nDone when: {done_when.strip()}",
        "transcript": transcript,
    }
