"""The prompt must not ask the user for a credential in chat.

Users pasted live secrets into conversation because the prompt asked them to
and, on a host with no credential tool, chat was the only way to comply. The
value then lives in the transcript and in every trace built from it.

Assertions target the ask-shaped clause rather than the word "credential":
CHAT_SYSTEM_PROMPT legitimately documents the DS_* vault contract, and that
has to stay.
"""

from __future__ import annotations

import re

from anton.core.llm.prompts import (
    CHAT_SYSTEM_PROMPT,
    CONVERSATION_DISCIPLINE_ACT_FIRST,
    CONVERSATION_DISCIPLINE_ASK_FIRST,
)

_DISCIPLINES = (CONVERSATION_DISCIPLINE_ACT_FIRST, CONVERSATION_DISCIPLINE_ASK_FIRST)


def _is_negated(sentence: str) -> bool:
    """Whether a sentence forbids rather than instructs.

    Apostrophes are stripped first: a word-boundary match never fires inside
    "don't" (the `n` is preceded by a word character), so a prohibition
    written that way would otherwise read as a solicitation. Stripping leaves
    "dont" as its own token while "hasn't" becomes "hasnt", which is not in
    the set — so "if the user hasn't shared credentials yet, ask ..." still
    counts as a solicitation, which it is.
    """
    plain = sentence.replace("\u2019", "").replace("'", "")
    return re.search(r"\b(never|not|cannot|dont|doesnt|wont|isnt)\b", plain, re.IGNORECASE) is not None


def _ask_sentences(text: str) -> list[str]:
    """Sentences that instruct asking the user for something."""
    return [s for s in text.split(".") if re.search(r"\bask", s, re.IGNORECASE)]


class TestNoCredentialSolicitation:
    def test_no_ask_instruction_names_a_credential_as_the_thing_to_ask_for(self):
        """Asserted over every ask-shaped sentence rather than the one phrase
        that was removed, so a rewording is caught too. The prohibition itself
        mentions credentials while telling the model not to ask, so a sentence
        only counts as a solicitation when it is not negated.
        """
        # Deliberately not bare "token" or "secret": this prompt also discusses
        # LLM tokens, and a scan that collides with those fails on edits that
        # have nothing to do with credentials.
        subjects = re.compile(
            r"credential|api key|password|private key|access token|secret value",
            re.IGNORECASE,
        )
        for text in (CHAT_SYSTEM_PROMPT, *_DISCIPLINES):
            for sentence in _ask_sentences(text):
                if subjects.search(sentence) is None:
                    continue
                assert _is_negated(sentence), sentence

    def test_the_removed_solicitations_stay_removed(self):
        """The two literals that were actually there, pinned so a revert is
        visible as a failure rather than as a silently weaker prompt."""
        assert "credentials they haven't shared" not in CHAT_SYSTEM_PROMPT
        assert "credentials or access you can't obtain" not in CONVERSATION_DISCIPLINE_ACT_FIRST

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
