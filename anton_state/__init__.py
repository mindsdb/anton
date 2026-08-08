"""anton_state — unified STATE API for fullstack-stateful artifacts."""

__version__ = "0.1.0"

from anton_state.base import Driver, Item, Store
from anton_state.errors import (
    ConditionalCheckFailed,
    StateError,
    StateThrottled,
    StateUnavailable,
    StateValidationError,
)
from anton_state.factory import from_backend_state, open_store
from anton_state.odm import Collection
from anton_state.schema import Attr, Index, StateSchema
