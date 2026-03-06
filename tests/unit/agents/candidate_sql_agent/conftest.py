import sys
from types import ModuleType
from unittest.mock import Mock


def _ensure_module(name: str) -> ModuleType:
    mod = sys.modules.get(name)
    if isinstance(mod, ModuleType):
        return mod
    mod = ModuleType(name)
    sys.modules[name] = mod
    return mod


# Some environments running unit tests may not have optional deps installed.
# Stub them so that importing agent modules is possible without installing extras.

# - stub minds.agents.helpers to avoid importing full LLM/config + model stack during unit tests
helpers_name = "minds.agents.helpers"
if helpers_name not in sys.modules:
    helpers = ModuleType(helpers_name)

    helpers.current_date_time_layer = lambda: "The current date and time is 2000-01-01 00:00:00"
    helpers.mind_layer = lambda _mind: ""
    helpers.charting_layer = lambda: ""
    helpers.model_for = lambda _mind: Mock()
    helpers.is_native_query_mode_enabled = lambda _mind, _settings: False
    sys.modules[helpers_name] = helpers

# - langfuse is used for instrumentation in some code paths
langfuse = _ensure_module("langfuse")
if not hasattr(langfuse, "observe"):
    langfuse.observe = lambda f=None, **_: (lambda *a, **k: f(*a, **k)) if f else (lambda x: x)
if not hasattr(langfuse, "get_client"):
    langfuse.get_client = lambda *args, **kwargs: Mock()

# - mind_castle is used by SQLAlchemy type helpers in models
mc = _ensure_module("mind_castle")
mc_sqlalchemy_type = _ensure_module("mind_castle.sqlalchemy_type")
if not hasattr(mc_sqlalchemy_type, "SecretData"):
    from sqlalchemy.types import String, TypeDecorator

    class SecretData(TypeDecorator):  # noqa: N801 (keep name to match expected symbol)
        """Minimal stand-in for mind_castle SecretData SQLAlchemy type."""

        impl = String
        cache_ok = True

        def __init__(self, *args, **kwargs):
            super().__init__()

        def process_bind_param(self, value, dialect):
            return value

        def process_result_value(self, value, dialect):
            return value

    mc_sqlalchemy_type.SecretData = SecretData
