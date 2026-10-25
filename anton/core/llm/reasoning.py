"""Task-local reasoning effort for phases that need a different allocation."""
from contextlib import contextmanager
from contextvars import ContextVar

_UNSET = object()
_effort = ContextVar('anton_reasoning_effort', default=_UNSET)


def current_effort(default):
    value = _effort.get()
    return default if value is _UNSET else value


@contextmanager
def scoped_effort(value):
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError('Reasoning effort must be a nonempty provider-supported string or None.')
    token = _effort.set(value)
    try:
        yield
    finally:
        _effort.reset(token)
