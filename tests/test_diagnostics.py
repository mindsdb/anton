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
