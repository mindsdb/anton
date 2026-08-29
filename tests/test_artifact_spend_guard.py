"""SpendGuard: the two-level budget check inside generate_artifact (I-20)."""
from __future__ import annotations

from types import SimpleNamespace

from anton.core.tools.generate_artifact.spend import WIND_DOWN_ROUNDS, SpendGuard


def _session(reached: bool):
    return SimpleNamespace(spend_ceiling_reached=lambda: reached)


def test_below_ceiling_is_not_winding_down():
    guard = SpendGuard(session=_session(False))
    assert guard.should_wind_down() is False
    assert guard.winding_down is False


def test_reaching_the_ceiling_starts_wind_down():
    guard = SpendGuard(session=_session(True))
    assert guard.should_wind_down() is True
    assert guard.winding_down is True


def test_wind_down_is_sticky_even_if_the_ceiling_stops_reporting():
    flag = {"reached": True}
    guard = SpendGuard(session=SimpleNamespace(spend_ceiling_reached=lambda: flag["reached"]))
    assert guard.should_wind_down() is True
    flag["reached"] = False
    assert guard.should_wind_down() is True


def test_the_guard_holds_no_round_counter():
    """The closing-round budget is PER WRITE LOOP, not per run.

    The fullstack path runs two write loops under `asyncio.gather`, both
    sharing this guard. A counter here would give the pair WIND_DOWN_ROUNDS
    between them — roughly one round each — and one round is not enough to
    both emit a final chunk and call `finish`. So the guard only latches;
    each loop counts its own rounds down from WIND_DOWN_ROUNDS.
    """
    guard = SpendGuard(session=_session(True))
    assert not hasattr(guard, "wind_down_rounds_left")
    assert not hasattr(guard, "consume_wind_down_round")
    assert WIND_DOWN_ROUNDS >= 2


def test_a_session_without_the_public_probe_never_winds_down():
    # bench_generate.py and most unit tests pass a bare stub session.
    guard = SpendGuard(session=SimpleNamespace())
    assert guard.should_wind_down() is False


def test_a_probe_that_raises_is_treated_as_below_ceiling():
    def boom() -> bool:
        raise RuntimeError("no turn in progress")

    guard = SpendGuard(session=SimpleNamespace(spend_ceiling_reached=boom))
    assert guard.should_wind_down() is False
