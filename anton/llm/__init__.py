"""Compatibility shim for the legacy `anton.llm` package.

The canonical implementation lives in `anton.core.llm`.
This package re-exports it so existing imports keep working:
- `from anton.llm.client import LLMClient`
- `from anton.llm.provider import LLMResponse`
"""

from anton.core.llm import *  # noqa: F401,F403
from anton.core.llm import __all__ as __all__  # re-export

