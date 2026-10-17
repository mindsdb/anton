"""Persistence lines in both conversation-discipline postures.

A hand-back used to tell a user who had just said "keep going" that it had
"stopped, as requested", and the model checked in mid-task on its own. Both
postures now say not to invent a stopping point after "keep going", and how
to describe a stop caused by an automatic limit. Pins that the lines survive
edits to the blocks around them.
"""

from anton.core.llm.prompts import (
    CONVERSATION_DISCIPLINE_ACT_FIRST,
    CONVERSATION_DISCIPLINE_ASK_FIRST,
)


def test_both_postures_attribute_a_limit_stop_to_the_limit():
    for discipline in (CONVERSATION_DISCIPLINE_ACT_FIRST, CONVERSATION_DISCIPLINE_ASK_FIRST):
        assert "automatic limit" in discipline
        assert "Never say you stopped because the user asked unless they did" in discipline


def test_both_postures_honor_keep_going():
    assert "don't invent a stopping point" in CONVERSATION_DISCIPLINE_ACT_FIRST
    assert "keep going and not stop" in CONVERSATION_DISCIPLINE_ASK_FIRST


def test_act_first_finishes_what_it_starts():
    assert "Finish what you start" in CONVERSATION_DISCIPLINE_ACT_FIRST
