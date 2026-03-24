from __future__ import annotations

import io
import json
import logging
import runpy
import sys
import types
from pathlib import Path

import pytest

_BOOT_PATH = Path(__file__).resolve().parents[5] / "minds" / "agents" / "anton_agent" / "anton" / "scratchpad_boot.py"


def _run_boot_once(
    *,
    stdin_text: str,
    monkeypatch: pytest.MonkeyPatch,
    clear_optional_env: bool = True,
) -> str:
    # The boot script imports `dill` unconditionally (it is installed in the scratchpad runtime),
    # but the unit-test environment may not include it. Provide a tiny stub so the script can run.
    class _FakeDill(types.SimpleNamespace):
        @staticmethod
        def dump(obj, fp):  # noqa: ANN001
            import pickle

            return pickle.dump(obj, fp)

        @staticmethod
        def load(fp):  # noqa: ANN001
            import pickle

            return pickle.load(fp)

    monkeypatch.setitem(sys.modules, "dill", _FakeDill())

    if clear_optional_env:
        # Ensure optional integrations are off for deterministic behavior.
        monkeypatch.delenv("ANTON_SCRATCHPAD_MODEL", raising=False)
        monkeypatch.delenv("ANTON_SCRATCHPAD_PROVIDER", raising=False)
        monkeypatch.delenv("ANTON_MINDS_URL", raising=False)
        monkeypatch.delenv("ANTON_MINDS_DATASOURCE", raising=False)
        monkeypatch.delenv("ANTON_MINDS_SSL_VERIFY", raising=False)
        monkeypatch.delenv("ANTON_MINDS_DATASOURCES_JSON", raising=False)
        monkeypatch.delenv("ANTON_MINDS_USER_ID", raising=False)
        monkeypatch.delenv("ANTON_MINDS_ORG_ID", raising=False)
        monkeypatch.delenv("ANTON_UV_PATH", raising=False)

    old_stdout, old_stderr, old_stdin = sys.stdout, sys.stderr, sys.stdin
    out = io.StringIO()
    err = io.StringIO()

    root = logging.getLogger()
    prior_handlers = list(root.handlers)
    prior_level = root.level

    try:
        sys.stdout = out
        sys.stderr = err
        sys.stdin = io.StringIO(stdin_text)
        runpy.run_path(str(_BOOT_PATH), run_name="__main__")
        return out.getvalue()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.stdin = old_stdin

        # Cleanup handlers added by scratchpad_boot.
        for h in list(root.handlers):
            if h not in prior_handlers:
                root.removeHandler(h)
        root.setLevel(prior_level)


def _extract_single_result(output: str) -> dict:
    start = output.index("__ANTON_RESULT__\n") + len("__ANTON_RESULT__\n")
    end = output.index("__ANTON_RESULT_END__\n", start)
    payload = output[start:end].strip()
    return json.loads(payload)


def test_scratchpad_boot_exec_prints_and_returns_json(monkeypatch):
    out = _run_boot_once(stdin_text="print('hi')\n__ANTON_CELL_END__\n", monkeypatch=monkeypatch)
    result = _extract_single_result(out)
    assert result["stdout"] == "hi\n"
    assert result["error"] is None


def test_scratchpad_boot_empty_cell(monkeypatch):
    out = _run_boot_once(stdin_text="\n__ANTON_CELL_END__\n", monkeypatch=monkeypatch)
    result = _extract_single_result(out)
    assert result["stdout"] == ""
    assert result["stderr"] == ""
    assert result["error"] is None


def test_scratchpad_boot_exception_traceback(monkeypatch):
    out = _run_boot_once(stdin_text="1/0\n__ANTON_CELL_END__\n", monkeypatch=monkeypatch)
    result = _extract_single_result(out)
    assert result["error"] is not None
    assert "ZeroDivisionError" in result["error"]


def test_scratchpad_boot_module_not_found_without_name_sets_traceback(monkeypatch):
    # Raising ModuleNotFoundError directly typically produces .name == None,
    # so the auto-install branch should be skipped.
    out = _run_boot_once(stdin_text="raise ModuleNotFoundError('x')\n__ANTON_CELL_END__\n", monkeypatch=monkeypatch)
    result = _extract_single_result(out)
    assert result["error"] is not None
    assert "ModuleNotFoundError" in result["error"]


def test_scratchpad_boot_auto_install_fail_uses_fake_subprocess(monkeypatch):
    # Trigger ModuleNotFoundError with a populated .name by importing a missing module.
    missing = "anton_definitely_missing_pkg"

    class _Proc:
        returncode = 1
        stderr = b"nope"

    fake_subprocess = types.SimpleNamespace(run=lambda *a, **k: _Proc())
    monkeypatch.setitem(sys.modules, "subprocess", fake_subprocess)

    out = _run_boot_once(stdin_text=f"import {missing}\n__ANTON_CELL_END__\n", monkeypatch=monkeypatch)
    result = _extract_single_result(out)
    assert result["error"] is not None
    assert "Auto-install failed" in result["error"]


def test_scratchpad_boot_auto_install_success_injects_module(monkeypatch):
    missing = "anton_missing_pkg_ok"

    class _Proc:
        returncode = 0
        stderr = b""

    def _run(argv, *a, **k):
        # Simulate install side-effect.
        sys.modules[missing] = types.ModuleType(missing)
        return _Proc()

    fake_subprocess = types.SimpleNamespace(run=_run)
    monkeypatch.setitem(sys.modules, "subprocess", fake_subprocess)

    out = _run_boot_once(stdin_text=f"import {missing}\n__ANTON_CELL_END__\n", monkeypatch=monkeypatch)
    result = _extract_single_result(out)
    assert result["error"] is None
    assert result.get("auto_installed") == [missing]


def test_scratchpad_boot_injects_progress_sample_and_minds_helpers(monkeypatch):
    # Enable Minds helpers.
    monkeypatch.setenv("ANTON_MINDS_URL", "http://minds")
    monkeypatch.setenv("ANTON_MINDS_DATASOURCE", "ds")
    monkeypatch.setenv("ANTON_MINDS_DATASOURCES_JSON", '[{"name":"ds","engine":"postgres"}]')
    monkeypatch.setenv("ANTON_MINDS_SSL_VERIFY", "false")
    monkeypatch.setenv("ANTON_MINDS_USER_ID", "u1")
    monkeypatch.setenv("ANTON_MINDS_ORG_ID", "o1")

    # Fake urllib.request to avoid network.
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"type":"ok","data":[[1]],"column_names":["x"]}'

    class _FakeReq:
        def __init__(self, url, data=None, method=None):
            self.url = url
            self.data = data
            self.method = method
            self.headers = {}

        def add_header(self, k, v):
            self.headers[k] = v

    fake_urllib = types.ModuleType("urllib.request")
    fake_urllib.Request = _FakeReq
    fake_urllib.urlopen = lambda req, context=None, timeout=60: _Resp()
    fake_urllib.HTTPError = type("HTTPError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "urllib.request", fake_urllib)
    import urllib as _urllib_pkg

    monkeypatch.setattr(_urllib_pkg, "request", fake_urllib, raising=False)

    code = "\n".join(
        [
            "progress('p1')",
            "print(list_datasources())",
            "print(query_minds_data('select 1'))",
            "sample({'a': 1, 'b': 2})",
        ]
    )
    out = _run_boot_once(stdin_text=code + "\n__ANTON_CELL_END__\n", monkeypatch=monkeypatch, clear_optional_env=False)

    # progress() writes to real stdout with marker
    assert "__ANTON_PROGRESS__ p1" in out

    result = _extract_single_result(out)
    assert "[{'name': 'ds', 'engine': 'postgres'}]" in result["stdout"]
    assert "'type': 'ok'" in result["stdout"]


def test_scratchpad_boot_query_minds_data_http_error_branch(monkeypatch):
    monkeypatch.setenv("ANTON_MINDS_URL", "http://minds")
    monkeypatch.setenv("ANTON_MINDS_DATASOURCE", "ds")
    monkeypatch.setenv("ANTON_MINDS_DATASOURCES_JSON", "[]")
    monkeypatch.setenv("ANTON_MINDS_SSL_VERIFY", "true")

    class HTTPError(Exception):
        def __init__(self, code=500, reason="no"):
            super().__init__("http")
            self.code = code
            self.reason = reason

        def read(self):
            return b"bad"

    class _FakeReq:
        def __init__(self, url, data=None, method=None):
            self.url = url
            self.data = data
            self.method = method
            self.headers = {}

        def add_header(self, k, v):
            self.headers[k] = v

    fake_urllib = types.ModuleType("urllib.request")
    fake_urllib.Request = _FakeReq
    fake_urllib.HTTPError = HTTPError
    fake_urllib.urlopen = lambda *a, **k: (_ for _ in ()).throw(HTTPError())
    monkeypatch.setitem(sys.modules, "urllib.request", fake_urllib)
    import urllib as _urllib_pkg

    monkeypatch.setattr(_urllib_pkg, "request", fake_urllib, raising=False)

    out = _run_boot_once(
        stdin_text="print(query_minds_data('select 1'))\n__ANTON_CELL_END__\n",
        monkeypatch=monkeypatch,
        clear_optional_env=False,
    )
    result = _extract_single_result(out)
    assert "'type': 'error'" in result["stdout"]
    assert "HTTP 500" in result["stdout"]


def test_scratchpad_boot_injects_get_llm_and_generate_object(monkeypatch):
    # Enable LLM helpers and provide fake llm.* modules.
    monkeypatch.setenv("ANTON_SCRATCHPAD_MODEL", "m")
    monkeypatch.setenv("ANTON_SCRATCHPAD_PROVIDER", "openai")

    class _ToolCall:
        def __init__(self, input_):
            self.id = "t1"
            self.name = "Tool"
            self.input = input_

    class _Resp:
        def __init__(self, content="", tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []

    class OpenAIProvider:
        def __init__(self, api_key: str = "", **_kwargs):
            self.api_key = api_key

        async def complete(self, **_kwargs):
            return _Resp(content="ok", tool_calls=[_ToolCall({"items": [{"x": 1}]})])

    llm_pkg = types.ModuleType("llm")
    llm_openai = types.ModuleType("llm.openai")
    llm_openai.OpenAIProvider = OpenAIProvider
    monkeypatch.setitem(sys.modules, "llm", llm_pkg)
    monkeypatch.setitem(sys.modules, "llm.openai", llm_openai)

    code = "\n".join(
        [
            "from pydantic import BaseModel",
            "class M(BaseModel):",
            "    x: int",
            "llm = get_llm()",
            "items = llm.generate_object(list[M], system='s', messages=[{'role':'user','content':'x'}])",
            "print(items[0].x)",
        ]
    )
    out = _run_boot_once(stdin_text=code + "\n__ANTON_CELL_END__\n", monkeypatch=monkeypatch, clear_optional_env=False)
    result = _extract_single_result(out)
    assert result["error"] is None
    assert result["stdout"].strip().endswith("1")
