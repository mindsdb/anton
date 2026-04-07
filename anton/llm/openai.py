"""Compatibility shim for `anton.llm.openai`.

The canonical implementation lives in `anton.core.llm.openai`.

Note: we re-export the imported OpenAI SDK module as `openai` so test patches like
`patch("anton.llm.openai.openai")` keep working.
"""

from anton.core.llm.openai import *  # noqa: F401,F403
from anton.core.llm.openai import openai as openai  # noqa: F401

