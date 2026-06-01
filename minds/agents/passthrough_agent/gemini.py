"""DEPRECATED: Backward compatibility re-export.

This module is superseded by minds.inference.providers.gemini.

All new code should import from minds.inference.providers.gemini instead.
This file is kept only for test compatibility and will be removed
in the next major version.
"""

from minds.inference.providers.gemini import *  # noqa: F401, F403

__all__ = [  # noqa: F405
    "_chat_messages_to_gemini",
    "_chat_tool_choice_to_gemini",
    "_gemini_finish_reason_to_openai",
    "_gemini_first_candidate",
    "_gemini_parts_for",
    "_gemini_response_to_openai",
    "_get_gemini_client",
    "_translate_tools_for_gemini",
    "proxy_gemini",
    "stream_gemini_as_openai",
]
