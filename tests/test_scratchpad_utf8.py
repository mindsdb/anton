"""ENG-824 regression: the scratchpad must be UTF-8-safe regardless of the
host locale code page (e.g. GBK/cp936 on Chinese Windows)."""

from __future__ import annotations

import inspect
import re

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


# ── ENG-940: the chat/scratchpad-path reads are launcher-independent ──────────
#
# ENG-824 shipped the PYTHONUTF8 "belt"; these pin the "suspenders" — explicit
# encoding="utf-8" on the reads themselves — so they can't regress to the host
# locale default on a launcher that doesn't set the env (bare CLI, Docker,
# OpenClaw), which would re-crash on a GBK/CJK Windows host.

def test_scratchpad_script_read_is_locale_independent(tmp_path):
    # Models ENG-824's root-caused crash site (`code = script_path.read_text()`):
    # the scratchpad script holds arbitrary model-written non-ASCII (string
    # literals, comments). A host-locale (GBK) default read crashes on UTF-8
    # bytes — the reported failure. Explicit encoding="utf-8" is locale-
    # independent. Proven on a real fixture, no dependence on the interpreter's
    # UTF-8 mode / PYTHONUTF8.
    script = tmp_path / "cell.py"
    body = 'label = "café — 数据 リポート"\nprint(label)\n'
    script.write_text(body, encoding="utf-8")

    raw = script.read_bytes()
    assert any(b > 0x7F for b in raw)  # genuinely non-ASCII → a locale read is at risk

    # The fix: an explicit UTF-8 read returns the script verbatim, independent
    # of the host locale / PYTHONUTF8.
    assert script.read_text(encoding="utf-8") == body

    # A host-locale (GBK) default read is unsafe — it either raises (the
    # reported ENG-824 crash) or silently corrupts the source. Payload-
    # independent: we don't rely on which failure mode these bytes trigger.
    try:
        assert raw.decode("gbk") != raw.decode("utf-8")  # mojibake
    except UnicodeDecodeError:
        pass  # crash — the exact ENG-824 failure mode


def test_chat_module_has_no_bare_read_text():
    # Regression guard (ENG-940): every text read in anton/chat.py must pass an
    # explicit encoding, so it can't silently revert to the host-locale default
    # (the ENG-824 crash at `code = script_path.read_text()`). Fails the instant
    # a `.read_text(...)` call omits encoding=.
    import anton.chat as chat

    src = inspect.getsource(chat)
    bare = [
        m.group(0)
        for m in re.finditer(r"\.read_text\(([^)]*)\)", src)
        if "encoding=" not in m.group(1)
    ]
    assert not bare, f"bare read_text() in anton/chat.py (add encoding='utf-8'): {bare}"


# ── ENG-940: the *encode*-side sibling (write path, lone surrogates) ──────────
#
# Distinct from the decode crash above: on a non-UTF-8 host, a non-ASCII
# Windows path (pt-BR "Área de Trabalho", an emoji filename) is surrogate-
# escaped into lone surrogates (\udcXX) when decoded. When such a string reaches
# the strict UTF-8 encode of the cell payload it raises "surrogates not allowed"
# and kills the whole session (users sabrina/eddie/janis). surrogatepass keeps
# the host-side encode from crashing; the subprocess (UTF-8 mode) decodes it.

def test_cell_payload_encode_is_lossless_over_the_pipe():
    # NOTE: this pins the *transport* only — that surrogateescape encode (parent)
    # round-trips through the subprocess's surrogateescape stdin decode. It does
    # NOT by itself fix the crash: the real raiser is compile() in the child on a
    # lone surrogate, healed by heal_surrogate_source (ENG-981), tested below.
    # (The `surrogatepass` alternative would fail this round-trip, re-mangling
    # the bytes — that's the only claim this test makes.)
    code = 'open(r"C:\\Users\\\udc81\udc9d\\index.html", "w")  # 🎉\n'
    payload = code + "\n" + local.CELL_DELIM + "\n"
    encoded = local._encode_cell_payload(payload)
    assert isinstance(encoded, bytes)
    assert encoded.decode("utf-8", errors="surrogateescape") == payload


def test_cell_payload_encode_preserves_accented_and_emoji():
    # Ordinary (well-formed) accented-Latin + emoji content must pass through
    # byte-for-byte — the transport must not mangle the common case.
    code = 'label = "Área de Trabalho 🎉 café"\nprint(label)\n'
    payload = code + "\n" + local.CELL_DELIM + "\n"

    encoded = local._encode_cell_payload(payload)
    assert encoded.decode("utf-8") == payload  # strict decode: no corruption


# ── ENG-981: heal lone surrogates before compile() (the real encode-side fix) ──
#
# The raiser is compile() in the child on a lone surrogate in the cell source
# (a non-ASCII Windows path byte, surrogate-escaped upstream and passed through
# the belt's lenient surrogateescape stdin). compile() is always strict, so the
# source must be cleaned first. These pin heal_surrogate_source, which the child
# calls right before compile().

def test_heal_recovers_byte_escaped_multibyte_path():
    # The common case: a real char whose UTF-8 bytes were escaped byte-wise
    # upstream (Á = C3 81 -> \udcc3\udc81). Heal must reassemble the *correct*
    # character so the cell compiles AND references the right path.
    mangled = 'open(r"C:\\Users\\\udcc3\udc81rea\\f.html")\n'   # "Área" split into surrogates
    healed = wire.heal_surrogate_source(mangled)
    assert healed == 'open(r"C:\\Users\\Área\\f.html")\n'
    compile(healed, "<scratchpad>", "exec")  # no UnicodeEncodeError


def test_heal_recovers_escaped_emoji():
    mangled = 'x = "\udcf0\udc9f\udc8e\udc89"\n'                 # 🎉 = f0 9f 8e 89, escaped
    healed = wire.heal_surrogate_source(mangled)
    assert healed == 'x = "🎉"\n'
    compile(healed, "<scratchpad>", "exec")


def test_heal_preserves_recoverable_char_in_mixed_cell():
    # A recoverable byte-escaped path AND an unrelated lone byte in the same
    # cell: the recoverable "Área" must survive (only the stray byte is scrubbed)
    # — a strict-decode-then-full-scrub would mojibake the whole thing.
    mixed = 'p = r"C:\\\udcc3\udc81rea"  # noqa\nq = "\udc81"\n'
    healed = wire.heal_surrogate_source(mixed)
    assert "Área" in healed
    assert not any("\ud800" <= ch <= "\udfff" for ch in healed)  # no residual
    compile(healed, "<scratchpad>", "exec")


def test_heal_scrubs_truly_lone_surrogate_so_compile_succeeds():
    # A genuinely lone surrogate (not a valid byte sequence): can't be recovered,
    # but must not crash compile() — replaced rather than raised.
    mangled = 'x = "a\udc81b"\n'
    healed = wire.heal_surrogate_source(mangled)
    assert "\udc81" not in healed
    compile(healed, "<scratchpad>", "exec")  # previously: UnicodeEncodeError


def test_heal_is_noop_on_clean_source():
    clean = 'print("Área de Trabalho 🎉")\n'   # real chars, no surrogates
    assert wire.heal_surrogate_source(clean) == clean


def test_bare_compile_on_lone_surrogate_still_raises():
    # Guard the premise: without the heal, this is the exact ENG-981 crash.
    with pytest.raises(UnicodeEncodeError):
        compile('x = "a\udc81b"\n', "<scratchpad>", "exec")
