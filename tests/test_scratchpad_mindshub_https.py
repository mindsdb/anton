"""A saved http:// MindsHub URL reaches the scratchpad over https.

``AntonSettings`` loads such a URL as https, but the pad reads ``ANTON_MINDS_URL``,
``ANTON_OPENAI_BASE_URL`` and ``OPENAI_BASE_URL`` from its own environment as they
stand. The runtime moves them to https before the pad starts, so code in the pad
never sends the key in clear. Other URLs reach the pad as saved.
"""

from __future__ import annotations

import pytest

from anton.core.backends.local import LocalScratchpadRuntime

_PRINT_URLS = (
    "import os\n"
    "for name in ('ANTON_MINDS_URL', 'ANTON_OPENAI_BASE_URL', 'OPENAI_BASE_URL'):\n"
    "    print(name, os.environ.get(name, ''))\n"
)


async def _urls_the_pad_sees(tmp_path) -> dict[str, str]:
    pad = LocalScratchpadRuntime(
        name="mindshub-https",
        coding_provider="anthropic",
        coding_model="",
        coding_api_key="",
        coding_base_url="",
        workspace_path=tmp_path,
    )
    await pad.start()
    try:
        cell = await pad.execute(_PRINT_URLS)
    finally:
        await pad.close()
    assert cell.error is None, cell.error
    return dict(line.split(" ", 1) if " " in line else (line, "") for line in cell.stdout.splitlines())


@pytest.mark.parametrize(
    ("saved", "seen"),
    [
        ("http://api.mindshub.ai", "https://api.mindshub.ai"),
        ("http://localhost:11434", "http://localhost:11434"),
        ("http://api.example.com", "http://api.example.com"),
    ],
)
async def test_the_pad_sees_mindshub_urls_over_https(tmp_path, monkeypatch, saved, seen):
    monkeypatch.setenv("ANTON_MINDS_URL", saved)
    monkeypatch.setenv("ANTON_OPENAI_BASE_URL", f"{saved}/v1")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    urls = await _urls_the_pad_sees(tmp_path)

    assert urls["ANTON_MINDS_URL"] == seen
    assert urls["ANTON_OPENAI_BASE_URL"] == f"{seen}/v1"
    assert urls["OPENAI_BASE_URL"] == f"{seen}/v1"
