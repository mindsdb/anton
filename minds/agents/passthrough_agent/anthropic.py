"""DEPRECATED: Backward compatibility re-export.

This module is superseded by minds.inference.providers.anthropic.

All new code should import from minds.inference.providers.anthropic instead.
This file is kept only for test compatibility and will be removed
in the next major version.
"""

from minds.inference.providers.anthropic import *  # noqa: F401, F403

__all__ = [  # noqa: F405
    "_anthropic_response_to_openai",
    "_collect_anthropic_server_artifacts",
    "_openai_messages_to_anthropic",
    "_openai_tool_choice_to_anthropic",
    "_translate_tools_for_anthropic",
    "_get_anthropic_client",
    "proxy_anthropic",
    "stream_anthropic_as_openai",
]
