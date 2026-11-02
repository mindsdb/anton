"""Persistence lines in both conversation-discipline postures.

The model checked in mid-task on its own, and stopped again after a user had
said "keep going". Both postures now say not to invent a stopping point after
"keep going". How to describe a stop caused by an automatic limit is left to
the limit hand-back, which says it at the moment of the pause. Pins that the
lines survive edits to the blocks around them.
"""

from anton.core.llm.prompts import (
    CONVERSATION_DISCIPLINE_ACT_FIRST,
    CONVERSATION_DISCIPLINE_ASK_FIRST,
)


def test_both_postures_honor_keep_going():
    assert "don't invent a stopping point" in CONVERSATION_DISCIPLINE_ACT_FIRST
    assert "keep going and not stop" in CONVERSATION_DISCIPLINE_ASK_FIRST


def test_act_first_finishes_what_it_starts():
    assert "Finish what you start" in CONVERSATION_DISCIPLINE_ACT_FIRST
