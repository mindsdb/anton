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


def test_boot_script_content_requires_utf8():
    # The boot script contains non-ASCII (…, —). Reading it with a non-UTF-8
    # host-locale default (GBK on Chinese Windows) throws a UnicodeDecodeError
    # before the scratchpad can start — the exact ENG-824 crash. Guard that (a)
    # UTF-8 decodes cleanly and (b) the content genuinely needs UTF-8, so a
    # locale-default read is unsafe and must not regress.
    raw = local._BOOT_SCRIPT_PATH.read_bytes()
    raw.decode("utf-8")  # the fix: explicit UTF-8 succeeds
    with pytest.raises(UnicodeDecodeError):
        raw.decode("gbk")  # a GBK-locale default read would crash


def test_boot_script_is_read_as_utf8(monkeypatch):
    # Pin the *read*, not just the file's bytes: if someone drops the explicit
    # encoding="utf-8" (reverting to a host-locale default), this fails even on a
    # UTF-8 CI box — where a bytes-only check would still pass. Spy on
    # Path.read_text and assert the boot-script read passes encoding="utf-8".
    seen = {}
    real_read_text = local.Path.read_text

    def spy(self, *args, **kwargs):
        if self == local._BOOT_SCRIPT_PATH:
            seen["encoding"] = kwargs.get("encoding")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(local.Path, "read_text", spy)
    assert local._read_boot_script()  # exercises the real read path
    assert seen.get("encoding") == "utf-8"


def test_parent_venv_pth_is_written_as_utf8(tmp_path, monkeypatch):
    # Sibling of the boot-script read: the child reads .pth files as UTF-8 under
    # UTF-8 mode, so the parent must *write* _parent_venv.pth as UTF-8 or a
    # non-ASCII site-packages path (e.g. a CJK Windows user dir) corrupts.
    # Pin the write encoding without spinning up a real venv.
    import builtins
    import site

    # Force the "running inside a venv" branch deterministically.
    monkeypatch.setattr(local.sys, "prefix", "/venv")
    monkeypatch.setattr(local.sys, "base_prefix", "/usr")
    monkeypatch.setattr(site, "getsitepackages", lambda: ["/parent/site-packages"])

    child_site = tmp_path / "lib" / "python3.11" / "site-packages"
    child_site.mkdir(parents=True)

    # Bypass the heavy __init__ (no network/subprocess) — the method only reads
    # self._venv_dir.
    runtime = object.__new__(local.LocalScratchpadRuntime)
    runtime._venv_dir = str(tmp_path)

    seen = {}
    real_open = builtins.open

    def spy_open(file, *args, **kwargs):
        if str(file).endswith("_parent_venv.pth"):
            seen["encoding"] = kwargs.get("encoding")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    runtime._setup_parent_site_packages()

    assert seen.get("encoding") == "utf-8"  # assert before any later read
    written = (child_site / "_parent_venv.pth").read_text(encoding="utf-8")
    assert "/parent/site-packages" in written
