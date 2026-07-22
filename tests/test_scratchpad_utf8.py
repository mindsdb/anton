"""ENG-824 regression: the scratchpad must be UTF-8-safe regardless of the
host locale code page (e.g. GBK/cp936 on Chinese Windows)."""

from __future__ import annotations

import pytest

import anton.core.backends.local as local
import anton.core.backends.wire as wire


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


# ── ENG-981: heal lone surrogates before compile() (folded into the belt hotfix)
#
# The scratchpad child crashes at compile() on a lone surrogate in the cell
# source — a non-ASCII Windows path byte (Área de Trabalho, an emoji) surrogate-
# escaped upstream and passed through the belt's lenient surrogateescape stdin.
# compile() is always strict, so heal_surrogate_source() cleans the source first.

def test_heal_recovers_byte_escaped_multibyte_path():
    mangled = 'open(r"C:\\Users\\\udcc3\udc81rea\\f.html")\n'   # "Área" split into surrogates
    healed = wire.heal_surrogate_source(mangled)
    assert healed == 'open(r"C:\\Users\\Área\\f.html")\n'
    compile(healed, "<scratchpad>", "exec")


def test_heal_recovers_escaped_emoji():
    mangled = 'x = "\udcf0\udc9f\udc8e\udc89"\n'                 # 🎉 = f0 9f 8e 89, escaped
    healed = wire.heal_surrogate_source(mangled)
    assert healed == 'x = "🎉"\n'
    compile(healed, "<scratchpad>", "exec")


def test_heal_preserves_recoverable_char_in_mixed_cell():
    mixed = 'p = r"C:\\\udcc3\udc81rea"  # noqa\nq = "\udc81"\n'
    healed = wire.heal_surrogate_source(mixed)
    assert "Área" in healed
    assert not any("\ud800" <= ch <= "\udfff" for ch in healed)
    compile(healed, "<scratchpad>", "exec")


def test_heal_preserves_recoverable_char_beside_unmappable_surrogate():
    # A recoverable byte-escaped "Á" and a high/unpaired surrogate (\ud800,
    # outside DC80..DCFF) in the same cell: the unmappable one is scrubbed WITHOUT
    # dragging "Á" into a full-cell scrub.
    mixed = 'p = r"C:\\\udcc3\udc81rea"\nq = "\ud800"\n'
    healed = wire.heal_surrogate_source(mixed)
    assert "Área" in healed
    assert not any("\ud800" <= ch <= "\udfff" for ch in healed)
    compile(healed, "<scratchpad>", "exec")


def test_heal_scrubs_truly_lone_surrogate_so_compile_succeeds():
    healed = wire.heal_surrogate_source('x = "a\udc81b"\n')
    assert "\udc81" not in healed
    compile(healed, "<scratchpad>", "exec")


def test_heal_is_noop_on_clean_source():
    clean = 'print("Área de Trabalho 🎉")\n'
    assert wire.heal_surrogate_source(clean) == clean


def test_bare_compile_on_lone_surrogate_still_raises():
    with pytest.raises(UnicodeEncodeError):
        compile('x = "a\udc81b"\n', "<scratchpad>", "exec")
