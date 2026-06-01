"""DEPRECATED: Backward compatibility re-export.

This module is superseded by minds.inference.providers.openai.

All new code should import from minds.inference.providers.openai instead.
This file is kept only for test compatibility and will be removed
in the next major version.
"""

from minds.inference.providers.openai import *  # noqa: F401, F403

__all__ = [  # noqa: F405
    "_chat_messages_to_responses_input",
    "_chat_tool_choice_to_responses",
    "_collect_responses_server_artifacts",
    "_get_openai_client",
    "_responses_response_to_chat_completion",
    "_translate_tools_for_openai",
    "proxy_openai",
    "stream_openai_responses_as_chat",
]
