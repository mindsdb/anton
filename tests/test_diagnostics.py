"""The stdlib adapter keeps diagnostic values out of every output format."""

import json
import logging
import sys

import pytest
from structlog.contextvars import bound_contextvars

from anton.diagnostics import make_formatter


@pytest.mark.parametrize("format", ["json", "text"])
def test_exception_chain_and_args_are_private(monkeypatch, format):
    monkeypatch.setenv("LOG_FORMAT", format)
    try:
        try:
            raise ValueError("private-inner")
        except ValueError:
            raise RuntimeError("private-outer")
    except RuntimeError as exc:
        record = logging.LogRecord(
            "runtime", logging.ERROR, __file__, 1, "Failure: %s", (exc,), sys.exc_info()
        )
    record.password = "private-extra"
    with bound_contextvars(conversation_id="conversation-1"):
        output = make_formatter().format(record)
    assert "private-" not in output
    assert "conversation-1" in output
    assert "RuntimeError" in output and "ValueError" in output
    assert "test_exception_chain_and_args" in output
    if format == "json":
        event = json.loads(output)
        assert event["event"] == "operation_failed"
        assert event["error_type"] == "RuntimeError"
        assert event["log_level"] == "ERROR"


def test_exception_as_argument_without_exc_info_is_private():
    record = logging.LogRecord(
        "runtime",
        logging.WARNING,
        __file__,
        1,
        "Failure: %s",
        (RuntimeError("private-argument"),),
        None,
    )
    output = make_formatter().format(record)
    assert "private-argument" not in output
    assert "RuntimeError" in output


async def test_concurrent_contexts_reset():
    import asyncio

    from structlog.contextvars import get_contextvars

    async def operation(value):
        with bound_contextvars(conversation_id=value):
            await asyncio.sleep(0)
            record = logging.LogRecord(
                "runtime", logging.WARNING, __file__, 1, "Operation failed", (), None
            )
            return json.loads(make_formatter().format(record))["conversation_id"]

    assert await asyncio.gather(operation("one"), operation("two")) == ["one", "two"]
    assert "conversation_id" not in get_contextvars()


@pytest.mark.parametrize("log_format", ["json", "text"])
@pytest.mark.parametrize("level", [None, "DEBUG"])
def test_resident_entrypoint_keeps_setup_and_fork_failures_safe(monkeypatch, capsys, log_format, level):
    from types import SimpleNamespace

    from anton.cloud_turn import resident

    monkeypatch.setenv("LOG_FORMAT", log_format)
    if level is None:
        monkeypatch.delenv("LOG_LEVEL", raising=False)
    else:
        monkeypatch.setenv("LOG_LEVEL", level)
    monkeypatch.setattr(resident.logger, "handlers", [])
    monkeypatch.setattr(resident.logger, "level", resident.logger.level)
    monkeypatch.setattr(resident.logger, "propagate", resident.logger.propagate)
    monkeypatch.setattr(resident, "forbid_same_user_access", lambda: None)
    root = logging.getLogger()
    original_root = (root.handlers[:], root.level)

    def fail_preload():
        raise RuntimeError("private-preload-value")

    def fail_fork():
        raise OSError("private-fork-value")

    def serve(entry):
        assert resident._prepare(fail_preload) is False
        closed = []
        worker = object.__new__(resident._Resident)
        worker._fork(SimpleNamespace(close=lambda: closed.append(True)), [])
        assert closed == [True]
        resident.logger.info("Resident progress")
        return 0

    monkeypatch.setattr(resident.os, "fork", fail_fork)
    monkeypatch.setattr(resident, "serve", serve)
    assert resident.main() == 0
    assert (root.handlers, root.level) == original_root
    output = capsys.readouterr()
    assert output.out == ""
    assert "private-" not in output.err
    assert ("Resident progress" in output.err) is (level == "DEBUG")
    assert "RuntimeError" in output.err and "OSError" in output.err
    if log_format == "json":
        events = [json.loads(line) for line in output.err.splitlines()]
        failures = [event for event in events if event["log_level"] == "ERROR"]
        assert [event["error_type"] for event in failures] == ["RuntimeError", "OSError"]
        assert all(event["logger"] == resident.__name__ for event in events)
        assert all(event["event"] == "operation_failed" for event in failures)
        assert "fail_preload" in failures[0]["exception"]
        assert "fail_fork" in failures[1]["exception"]
