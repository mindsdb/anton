"""The browser tool: drives a browser the user shares with the agent.

The host says where that browser is: the MindsHub browser router for cloud
turns, or the Cowork desktop app's own browser on the user's machine. Anton
reads pages as numbered elements and acts on them by number, and the user
watches beside the chat and takes over when a step is theirs. See tool.py
for what the model sees, and models.py for the answers both hosts give.
"""

from anton.core.browser.config import BrowserConfig

__all__ = ["BrowserConfig"]
