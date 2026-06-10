"""Test fixtures for provider adapter tests."""

import os


def pytest_configure(config):
    """Set required Anthropic environment variables before module imports."""
    os.environ.setdefault("ANTHROPIC__WEB_SEARCH_TOOL_TYPE", "web_search_20250305")
    os.environ.setdefault("ANTHROPIC__WEB_FETCH_TOOL_TYPE", "web_fetch_20250910")
    os.environ.setdefault("ANTHROPIC__WEB_FETCH_BETA_HEADER", "web-fetch-2025-09-10")
