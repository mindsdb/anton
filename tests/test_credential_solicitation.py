"""The prompt must not ask the user for a credential in chat.

Users pasted live secrets into conversation because the prompt asked them to
and, on a host with no credential tool, chat was the only way to comply. The
value then lives in the transcript and in every trace built from it.

Assertions target the ask-shaped clause rather than the word "credential":
CHAT_SYSTEM_PROMPT legitimately documents the DS_* vault contract, and that
has to stay.
"""

from __future__ import annotations

from anton.core.llm.prompts import (
    CHAT_SYSTEM_PROMPT,
    CONVERSATION_DISCIPLINE_ACT_FIRST,
    CONVERSATION_DISCIPLINE_ASK_FIRST,
)

_DISCIPLINES = (CONVERSATION_DISCIPLINE_ACT_FIRST, CONVERSATION_DISCIPLINE_ASK_FIRST)


class TestNoCredentialSolicitation:
    def test_chat_prompt_does_not_list_credentials_as_a_thing_to_ask_for(self):
        assert "credentials they haven't shared" not in CHAT_SYSTEM_PROMPT

    def test_no_discipline_names_credentials_as_a_reason_to_stop_and_ask(self):
        """Both disciplines, because either can be the active one and the
        act-first variant is the default (ChatSessionConfig.act_first)."""
        for discipline in _DISCIPLINES:
            assert "credentials or access you can't obtain" not in discipline

    def test_the_vault_contract_survives(self):
        """The DS_* block is how scratchpad code reaches a connection at all.
        Removing the solicitation must not take it with it."""
        assert "DS_<ENGINE>_<NAME>__<FIELD>" in CHAT_SYSTEM_PROMPT
        assert "Connected data source credentials are injected" in CHAT_SYSTEM_PROMPT

    def test_the_prohibition_is_present_and_names_no_tool(self):
        """Where to supply a credential differs per host, so the base prompt
        carries only the rule; the tool prompt or the cloud-turn suffix names
        the place. Same split as the ask_user carve-out."""
        assert "A credential must never arrive as chat text" in CHAT_SYSTEM_PROMPT
        for tool in ("request_credentials", "connect_new_datasource"):
            assert tool not in CHAT_SYSTEM_PROMPT


class TestPastedCredentialIsNotPersisted:
    """A pasted secret must not be written anywhere durable on a host with no
    vault. The pod allowlists `memorize` and `create_skill_draft`, memory
    writes are relayed back and re-sent on later turns, and the relay's
    scrubber matches four shapes that a GitHub PAT, a WordPress application
    password and an SMTP password all miss. Storing it would turn one exposed
    trace into permanent exposure.
    """

    def test_the_rule_forbids_storing_the_value_at_all(self):
        assert "no tool call, no file and no store" in CHAT_SYSTEM_PROMPT

    def test_the_rule_says_to_have_it_rotated(self):
        """Rotation is the only remedy once a value is in the transcript, and
        it is the whole of the correct response where nothing can store it."""
        assert "should be rotated" in CHAT_SYSTEM_PROMPT
