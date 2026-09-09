"""`_is_exit_command` — what ends the interactive chat loop.

The help menu and the completer list the exit command among the slash
commands, so the slash form has to leave the chat instead of reaching the
unknown-command fallthrough. `tests/e2e/scenarios/test_boot_config.py` pins
that end to end through the real CLI; this file pins the edges around it.
"""

from __future__ import annotations

import pytest

from anton.chat import _is_exit_command
from anton.commands.ui import COMMANDS, Command


@pytest.mark.parametrize("text", ["exit", "quit", "bye"])
def test_bare_exit_words_still_end_the_chat(text: str):
    assert _is_exit_command(text)


@pytest.mark.parametrize("text", ["/exit", "/quit", "/bye"])
def test_slash_forms_end_the_chat(text: str):
    assert _is_exit_command(text)


@pytest.mark.parametrize("text", ["/EXIT", "Exit", "  /exit  "])
def test_case_and_surrounding_space_are_ignored(text: str):
    assert _is_exit_command(text)


def test_arguments_after_the_slash_form_are_tolerated():
    # Every other command in the dispatch chain matches on its first token,
    # so a stray argument must not fall through to "Unknown command: /exit".
    assert _is_exit_command("/exit now")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "/",
        "/exiting",
        "//exit",
        "/help",
        "exit now",
        "how do I exit",
        "/Users/someone/exit",
    ],
)
def test_everything_else_is_a_message_for_the_agent(text: str):
    assert not _is_exit_command(text)


def test_the_advertised_exit_entry_is_one_the_loop_accepts():
    # The menu advertising a command the dispatch does not take is the whole
    # bug: the entry has to be the slash form, and it has to end the chat.
    advertised = [
        item
        for item in COMMANDS
        if isinstance(item, Command) and _is_exit_command(item.command)
    ]
    assert advertised, "COMMANDS no longer advertises a command that ends the chat"
    assert all(item.command.startswith("/") for item in advertised)
