"""ENG-824 regression: the scratchpad must be UTF-8-safe regardless of the
host locale code page (e.g. GBK/cp936 on Chinese Windows)."""

from __future__ import annotations

import pytest

import anton.core.backends.local as local


def test_utf8_env_forces_utf8_mode():
    env = local._utf8_env({"FOO": "bar"})
    assert env["PYTHONUTF8"] == "1"
    assert env["FOO"] == "bar"  # base is preserved
    # Deliberately NOT set — a bare PYTHONIOENCODING makes stdio strict, which
    # would re-introduce a crash on exotic output (UTF-8 mode already covers it).
    assert "PYTHONIOENCODING" not in env


def test_utf8_env_respects_explicit_override():
    # setdefault: an operator who deliberately set UTF-8 mode off keeps it.
    env = local._utf8_env({"PYTHONUTF8": "0"})
    assert env["PYTHONUTF8"] == "0"


def test_boot_script_must_be_read_as_utf8():
    # The boot script contains non-ASCII (…, —). Reading it with a non-UTF-8
    # host-locale default (GBK on Chinese Windows) throws a UnicodeDecodeError
    # before the scratchpad can start — the exact ENG-824 crash. Guard that (a)
    # UTF-8 decodes cleanly and (b) the content genuinely needs UTF-8, so a
    # locale-default read is unsafe and must not regress.
    raw = local._BOOT_SCRIPT_PATH.read_bytes()
    raw.decode("utf-8")  # the fix: explicit UTF-8 succeeds
    with pytest.raises(UnicodeDecodeError):
        raw.decode("gbk")  # a GBK-locale default read would crash
