"""Unit tests for MindsHub flavor detection, verified frontier context windows,
Turn 1/2+ serving_model_lines cache stability, and lazy write_check import.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

from anton.core.llm.client import _resolve_openai_compatible_flavor
from anton.core.llm.identity import serving_model_lines
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import _DEFAULT_CONTEXT_WINDOW, context_window


def test_mindshub_flavor_detection_with_default_minds_url():
    s = SimpleNamespace(
        openai_base_url="https://api.mindshub.ai/v1",
        minds_url="https://mdb.ai",
    )
    assert _resolve_openai_compatible_flavor(s) == OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH

    third_party = SimpleNamespace(
        openai_base_url="https://api.openrouter.ai/v1",
        minds_url="https://mdb.ai",
    )
    assert _resolve_openai_compatible_flavor(third_party) == OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC


def test_context_window_verified_and_conservative():
    assert context_window("kimi-k3") == 1_048_576
    assert context_window("accounts/fireworks/models/kimi-k3") == 1_048_576
    assert context_window("moonshotai/Kimi-K3") == 1_048_576
    assert context_window("kimi-k2.5") == 256_000
    assert context_window("kimi-k2") == 128_000
    assert context_window("glm-5-3") == 200_000
    assert context_window("deepseek-v4.1-flash") == 200_000
    assert context_window("gemini-3.1-pro") == 1_000_000
    # Broad/unknown models must NOT be overestimated
    assert context_window("deepseek-v3") == _DEFAULT_CONTEXT_WINDOW
    assert context_window("qwen/qwen3.5-9b") == _DEFAULT_CONTEXT_WINDOW


def test_serving_model_lines_cache_stable_when_matching():
    t1 = serving_model_lines(requested="kimi-k3", served=None)
    t2 = serving_model_lines(requested="kimi-k3", served="accounts/fireworks/models/kimi-k3")
    assert t1 == t2
    # Different served model still reports accurately
    t_other = serving_model_lines(requested="kimi-k3", served="glm-5-3")
    assert "glm-5-3" in t_other[0]


def test_write_check_is_lazily_imported():
    code = (
        "import sys, anton.core.session; "
        "assert 'anton.core.utils.write_check' not in sys.modules, 'write_check loaded eagerly!'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
