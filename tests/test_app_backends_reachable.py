"""`ChatSessionConfig.app_backends_reachable`: False where the user's browser
cannot open a backend the session starts (cloud turns), so full-stack apps
reach the user only through their published preview."""
from __future__ import annotations

from types import SimpleNamespace

from anton.core.artifacts.backend_launcher import app_backends_reachable


def test_the_flag_defaults_to_reachable(make_session):
    assert make_session()._app_backends_reachable is True
    assert make_session(app_backends_reachable=False)._app_backends_reachable is False


def test_the_reader_treats_a_session_without_the_attribute_as_reachable():
    assert app_backends_reachable(SimpleNamespace()) is True
    assert app_backends_reachable(SimpleNamespace(_app_backends_reachable=True)) is True
    assert app_backends_reachable(SimpleNamespace(_app_backends_reachable=False)) is False
