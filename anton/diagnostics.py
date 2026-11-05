"""Stdlib logging with a shared JSON envelope and explicit diagnostic fields."""

import logging
import os
import sys
import traceback
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from uuid import UUID

import structlog

EVENT_FIELDS = frozenset(
    [
        "event",
        "reason",
        "outcome",
        "operation",
        "provider",
        "error_type",
        "error_class",
        "provider_error",
        "status_code",
        "request_id",
        "request_id_source",
        "correlation_id",
        "entry_id",
        "scratchpad_id",
        "pod_name",
        "conversation_id",
        "project_id",
        "schedule_id",
        "turn_id",
        "job_id",
        "trace_id",
        "method",
        "route",
        "duration_ms",
        "user_id",
        "organization_id",
        "workspace_id",
        "attempt",
        "attempts",
        "count",
        "model_alias",
        "retryable",
    ]
)


def safe_exception_traceback(exc_info: tuple) -> str | None:
    """Keep types and stack locations, excluding values, source, notes and locals."""
    if not exc_info or exc_info[1] is None:
        return None
    lines, seen = [], set()

    def visit(error, tb):
        if error is None or id(error) in seen:
            return
        seen.add(id(error))
        for frame in traceback.extract_tb(tb):
            lines.append(
                f'  File "{Path(frame.filename).name}", line {frame.lineno}, in {frame.name}'
            )
        lines.append(f"{type(error).__name__}: [error details omitted]")
        if isinstance(error, BaseExceptionGroup):
            for child in error.exceptions:
                visit(child, child.__traceback__)
        cause = error.__cause__ or (
            None if error.__suppress_context__ else error.__context__
        )
        if cause is not None:
            visit(cause, cause.__traceback__)

    visit(exc_info[1], exc_info[2])
    return "\n".join(lines)


def _event_schema(logger, method_name, event_dict):
    record = event_dict["_record"]
    if event_dict.get("exc_info"):
        # Exception messages may repeat a request or provider payload.
        event_dict["message"] = "Operation failed"
        event_dict.setdefault("event", "operation_failed")
        event_dict.setdefault("operation", record.funcName)
    args = event_dict.pop("positional_args", ())
    # Keep static message templates; formatting positional values can expose
    # provider replies, generated code, private paths or credential strings.
    if (
        args
        and isinstance(record.msg, str)
        and not getattr(record, "_diagnostic_safe_template", False)
    ):
        event_dict["message"] = record.msg
    if not record.name.startswith(("anton", "__main__")):
        event_dict["message"] = "Dependency diagnostic"
        event_dict.setdefault("event", "dependency_diagnostic")
    errors = [
        arg
        for arg in (args.values() if isinstance(args, dict) else args)
        if isinstance(arg, BaseException)
    ]
    if errors:
        event_dict["message"] = "Operation failed"
        event_dict["error_type"] = type(errors[0]).__name__
        event_dict.setdefault("event", "operation_failed")
        event_dict.setdefault("operation", record.funcName)
    if not isinstance(record.msg, str):
        event_dict["message"] = "Operation diagnostic"
    event_dict.setdefault(
        "event",
        "operation_failed"
        if record.levelno >= logging.WARNING
        else "operation_progress",
    )
    event_dict.setdefault("operation", record.funcName)
    payload = {
        "logged_at": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
        "logger": record.name,
        "log_level": record.levelname,
        "message": event_dict["message"],
    }
    for field in EVENT_FIELDS:
        value = event_dict.get(field)
        if isinstance(value, UUID):
            value = str(value)
        if type(value) in (str, bool, int) or (
            type(value) is float and isfinite(value)
        ):
            payload[field] = value
    exc_info = event_dict.get("exc_info")
    if exc_info:
        payload["exc_info"] = exc_info
        if isinstance(exc_info, tuple) and exc_info[1] is not None:
            payload["error_type"] = type(exc_info[1]).__name__
    return payload


def make_formatter():
    log_format = os.environ.get("LOG_FORMAT", "json")
    if log_format not in {"json", "text"}:
        raise ValueError("LOG_FORMAT must be json or text")
    renderer = (
        structlog.processors.JSONRenderer(
            separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        if log_format == "json"
        else structlog.processors.KeyValueRenderer(
            key_order=["logged_at", "log_level", "logger", "message"]
        )
    )
    return structlog.stdlib.ProcessorFormatter(
        pass_foreign_args=True,
        foreign_pre_chain=[
            structlog.processors.EventRenamer("message"),
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.ExtraAdder(allow=EVENT_FIELDS),
        ],
        processors=[
            _event_schema,
            structlog.processors.ExceptionRenderer(safe_exception_traceback),
            renderer,
        ],
    )


def configure_logging(level: str | None = None, *, stream=None) -> None:
    """Configure process diagnostics without redirecting protocol output."""
    level = (level or os.environ.get("LOG_LEVEL", "WARNING")).upper()
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(make_formatter())
    handler.setLevel(level)
    logging.basicConfig(level=level, handlers=[handler], force=True)
    for name in (
        "httpx",
        "httpx2",
        "httpcore",
        "httpcore2",
        "openai",
        "anthropic",
        "urllib3",
        "kubernetes",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    usage_handler = logging.StreamHandler(stream or sys.stderr)
    usage_handler.setFormatter(logging.Formatter("%(message)s"))
    usage = logging.getLogger("anton.usage")
    usage.setLevel(logging.INFO)
    usage.handlers = [usage_handler]
    usage.propagate = False
