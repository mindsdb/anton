"""Compatibility shim for `anton.llm.anthropic`.

The canonical implementation lives in `anton.core.llm.anthropic`.

Note: we re-export the imported Anthropic SDK module as `anthropic` so test patches like
`patch("anton.llm.anthropic.anthropic")` keep working.
"""

from anton.core.llm.anthropic import *  # noqa: F401,F403
from anton.core.llm.anthropic import anthropic as anthropic  # noqa: F401

