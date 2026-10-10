"""The browser tool: drives the user's MindsHub browser instance (br-<hash>).

The instance runs a Playwright + Chromium API (mindshub_services
snapshots/browser/service). Anton reads pages as numbered elements, acts on
them by number, and the user watches and takes over in a live viewer the host
shows beside the chat. See tool.py for what the model sees.
"""

from anton.core.browser.config import BrowserConfig

__all__ = ["BrowserConfig"]
