import contextlib
import io
import json
import logging as _logging
import os
import sys
import traceback
from urllib.parse import urljoin

# This does not need to be a project dependency because it is only used by the boot script.
import dill

_CELL_DELIM = "__ANTON_CELL_END__"
_RESULT_START = "__ANTON_RESULT__"
_RESULT_END = "__ANTON_RESULT_END__"

# --- Optional: persist scratchpad interpreter session across restarts (dill) ---
_PERSIST_SESSION = os.environ.get("ANTON_SCRATCHPAD_PERSIST_SESSION", "true").lower() in {"1", "true", "yes", "on"}
_SESSION_PATH = os.environ.get("ANTON_SCRATCHPAD_SESSION_PATH", "/anton_scratchpad_session.pkl")


def _load_namespace() -> tuple[dict, str | None]:
    try:
        with open(_SESSION_PATH, "rb") as f:
            ns = dill.load(f)
        if not isinstance(ns, dict):
            raise TypeError("Session file did not contain a namespace dict")
        ns.setdefault("__builtins__", __builtins__)
        return ns, None
    except FileNotFoundError:
        return {"__builtins__": __builtins__}, None
    except Exception:
        return (
            {"__builtins__": __builtins__},
            "Failed to load scratchpad session; starting fresh.\n" + traceback.format_exc(),
        )


def _dump_namespace(ns: dict) -> str | None:
    try:
        with open(_SESSION_PATH, "wb") as f:
            dill.dump(ns, f)
        return None
    except Exception:
        return "Failed to dump scratchpad session.\n" + traceback.format_exc()


# Persistent namespace across cells
namespace, _load_err = _load_namespace()

# --- Inject get_llm() for LLM access from scratchpad code ---
_scratchpad_model = os.environ.get("ANTON_SCRATCHPAD_MODEL", "")
if _scratchpad_model:
    try:
        import asyncio as _llm_asyncio

        _scratchpad_provider_name = os.environ.get("ANTON_SCRATCHPAD_PROVIDER", "anthropic")
        _scratchpad_api_key = os.environ.get("ANTON_SCRATCHPAD_API_KEY", "")
        if _scratchpad_provider_name == "openai":
            from llm.openai import OpenAIProvider as _ProviderClass
        else:
            from llm.anthropic import AnthropicProvider as _ProviderClass

        _llm_provider = _ProviderClass(api_key=_scratchpad_api_key)
        _llm_model = _scratchpad_model

        _LLM_HEARTBEAT_INTERVAL = 10

        async def _run_with_heartbeat(coro):
            """Run an async coroutine while emitting progress heartbeats."""

            async def _heartbeat():
                elapsed = 0
                while True:
                    await _llm_asyncio.sleep(_LLM_HEARTBEAT_INTERVAL)
                    elapsed += _LLM_HEARTBEAT_INTERVAL
                    _real_stdout.write(_PROGRESS_MARKER + f" Waiting for LLM… ({elapsed}s)\n")
                    _real_stdout.flush()

            beat = _llm_asyncio.create_task(_heartbeat())
            try:
                return await coro
            finally:
                beat.cancel()
                with contextlib.suppress(_llm_asyncio.CancelledError):
                    await beat

        class _ScratchpadLLM:
            """Sync LLM wrapper for scratchpad use."""

            @property
            def model(self):
                return _llm_model

            def complete(self, *, system, messages, tools=None, tool_choice=None, max_tokens=4096):
                return _llm_asyncio.run(
                    _run_with_heartbeat(
                        _llm_provider.complete(
                            model=_llm_model,
                            system=system,
                            messages=messages,
                            tools=tools,
                            tool_choice=tool_choice,
                            max_tokens=max_tokens,
                        )
                    )
                )

            async def complete_async(self, *, system, messages, tools=None, tool_choice=None, max_tokens=4096):
                return await _run_with_heartbeat(
                    _llm_provider.complete(
                        model=_llm_model,
                        system=system,
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        max_tokens=max_tokens,
                    )
                )

            def generate_object(self, schema_class, *, system, messages, max_tokens=4096):
                from pydantic import BaseModel as _BaseModel

                is_list = hasattr(schema_class, "__origin__") and schema_class.__origin__ is list
                if is_list:
                    inner_class = schema_class.__args__[0]

                    class _ArrayWrapper(_BaseModel):
                        items: list[inner_class]

                    schema = _ArrayWrapper.model_json_schema()
                    tool_name = f"{inner_class.__name__}_array"
                else:
                    schema = schema_class.model_json_schema()
                    tool_name = schema_class.__name__

                tool = {
                    "name": tool_name,
                    "description": f"Generate structured output matching the {tool_name} schema.",
                    "input_schema": schema,
                }

                response = self.complete(
                    system=system,
                    messages=messages,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": tool_name},
                    max_tokens=max_tokens,
                )

                if not response.tool_calls:
                    raise ValueError("LLM did not return structured output.")

                raw = response.tool_calls[0].input

                if is_list:
                    wrapper = _ArrayWrapper.model_validate(raw)
                    return wrapper.items
                return schema_class.model_validate(raw)

        _scratchpad_llm_instance = _ScratchpadLLM()

        def get_llm():
            """Get a pre-configured LLM client. No API keys needed."""
            return _scratchpad_llm_instance

        def agentic_loop(*, system, user_message, tools, handle_tool, max_turns=10, max_tokens=4096):
            """Run a synchronous LLM tool-call loop."""
            llm = get_llm()
            messages = [{"role": "user", "content": user_message}]

            response = None
            for _ in range(max_turns):
                response = llm.complete(
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                )

                if not response.tool_calls:
                    return response.content

                assistant_content = []
                if response.content:
                    assistant_content.append({"type": "text", "text": response.content})
                for tc in response.tool_calls:
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.input,
                        }
                    )
                messages.append({"role": "assistant", "content": assistant_content})

                tool_results = []
                for tc in response.tool_calls:
                    try:
                        result = handle_tool(tc.name, tc.input)
                    except Exception as exc:
                        result = f"Error: {exc}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": result,
                        }
                    )
                messages.append({"role": "user", "content": tool_results})

            return response.content if response else ""

        namespace["get_llm"] = get_llm
        namespace["agentic_loop"] = agentic_loop
    except Exception:
        pass  # LLM not available — not fatal

# --- Inject query_minds_data() for Minds datasource access from scratchpad ---
_minds_url = os.environ.get("ANTON_MINDS_URL", "")
_minds_user_id = os.environ.get("ANTON_MINDS_USER_ID", "")
_minds_org_id = os.environ.get("ANTON_MINDS_ORG_ID", "")
_minds_default_datasource = os.environ.get("ANTON_MINDS_DATASOURCE", "")
_minds_datasources_json = os.environ.get("ANTON_MINDS_DATASOURCES_JSON", "[]")
_minds_ssl_verify = os.environ.get("ANTON_MINDS_SSL_VERIFY", "true").lower() != "false"

# Parse available datasources
try:
    _available_datasources = json.loads(_minds_datasources_json)
except Exception:
    _available_datasources = []

if _minds_url and _minds_default_datasource:
    try:
        import ssl as _minds_ssl
        import urllib.request as _minds_urllib

        def list_datasources():
            """Return list of available datasources: [{"name": ..., "engine": ...}, ...]"""
            return _available_datasources

        def query_minds_data(query, datasource=None):
            """Query a Minds datasource with SQL.

            Args:
                query: SQL query string
                datasource: Name of the datasource to query. If None, uses default.
                           Available datasources: use list_datasources() to see them.
            """
            # TODO: This should ideally run via the SDK.
            ds = datasource or _minds_default_datasource
            base = _minds_url.rstrip("/") + "/"
            path = f"api/v1/datasources/{ds}/query"
            url = urljoin(base, path)
            payload = json.dumps({"query": query, "native_query": True}).encode()

            req = _minds_urllib.Request(url, data=payload, method="POST")
            # Use internal API headers (no Bearer token needed for internal calls)
            if _minds_user_id:
                req.add_header("X-User-Id", _minds_user_id)
            if _minds_org_id:
                req.add_header("X-Organization-Id", _minds_org_id)
            req.add_header("Content-Type", "application/json")
            req.add_header("Accept", "application/json")

            ctx = None
            if not _minds_ssl_verify:
                ctx = _minds_ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = _minds_ssl.CERT_NONE

            try:
                with _minds_urllib.urlopen(req, context=ctx, timeout=60) as resp:
                    return json.loads(resp.read().decode())
            except _minds_urllib.HTTPError as e:
                body = ""
                with contextlib.suppress(Exception):
                    body = e.read().decode()
                return {
                    "type": "error",
                    "data": None,
                    "column_names": None,
                    "error_message": f"HTTP {e.code}: {body or e.reason}",
                }
            except Exception as e:
                return {
                    "type": "error",
                    "data": None,
                    "column_names": None,
                    "error_message": str(e),
                }

        namespace["query_minds_data"] = query_minds_data
        namespace["list_datasources"] = list_datasources
    except Exception:
        pass  # Minds query not available — not fatal

# Read-execute loop
_real_stdout = sys.stdout
_real_stdin = sys.stdin

_PROGRESS_MARKER = "__ANTON_PROGRESS__"


def progress(message=""):
    """Signal that long-running work is still active. Resets the inactivity timer."""
    _real_stdout.write(_PROGRESS_MARKER + " " + str(message) + "\n")
    _real_stdout.flush()


namespace["progress"] = progress

# --- Variable inspector ---


def sample(var, mode="preview", _name=None):
    """Inspect a variable with type-aware formatting."""
    _MAX_PREVIEW = 2000
    _MAX_FULL = 10000
    limit = _MAX_PREVIEW if mode == "preview" else _MAX_FULL

    header = f"[sample:{type(var).__name__}]"
    if _name:
        header = f"[sample:{_name} ({type(var).__name__})]"

    lines = [header]

    # --- pandas DataFrame ---
    try:
        import pandas as _pd

        if isinstance(var, _pd.DataFrame):
            lines.append(f"Shape: {var.shape[0]} rows x {var.shape[1]} cols")
            lines.append(f"Columns: {list(var.columns)}")
            lines.append(f"Dtypes:\n{var.dtypes.to_string()}")
            if mode == "preview":
                lines.append(f"\nHead (5 rows):\n{var.head().to_string()}")
                if var.shape[0] > 5:
                    lines.append(f"\nTail (3 rows):\n{var.tail(3).to_string()}")
                nulls = var.isnull().sum()
                nulls = nulls[nulls > 0]
                if len(nulls) > 0:
                    lines.append(f"\nNull counts:\n{nulls.to_string()}")
            else:
                lines.append(f"\nDescribe:\n{var.describe(include='all').to_string()}")
                n = min(50, var.shape[0])
                lines.append(f"\nFirst {n} rows:\n{var.head(n).to_string()}")
                nulls = var.isnull().sum()
                nulls = nulls[nulls > 0]
                if len(nulls) > 0:
                    lines.append(f"\nNull counts:\n{nulls.to_string()}")
            print(_truncate_sample("\n".join(lines), limit))
            return

        if isinstance(var, _pd.Series):
            lines.append(f"Length: {len(var)}, Dtype: {var.dtype}, Name: {var.name}")
            if mode == "preview":
                lines.append(f"\nHead (10):\n{var.head(10).to_string()}")
            else:
                lines.append(f"\nDescribe:\n{var.describe().to_string()}")
                n = min(50, len(var))
                lines.append(f"\nFirst {n}:\n{var.head(n).to_string()}")
            print(_truncate_sample("\n".join(lines), limit))
            return
    except ImportError:
        pass

    # --- numpy array ---
    try:
        import numpy as _np

        if isinstance(var, _np.ndarray):
            lines.append(f"Shape: {var.shape}, Dtype: {var.dtype}")
            if mode == "preview":
                flat = var.flatten()
                n = min(10, len(flat))
                lines.append(f"First {n} values: {flat[:n].tolist()}")
                if len(flat) > 10:
                    lines.append(f"Last 3 values: {flat[-3:].tolist()}")
                lines.append(f"Min: {var.min()}, Max: {var.max()}, Mean: {var.mean():.4g}")
            else:
                lines.append(f"Min: {var.min()}, Max: {var.max()}, Mean: {var.mean():.4g}, Std: {var.std():.4g}")
                lines.append(f"\n{repr(var)}")
            print(_truncate_sample("\n".join(lines), limit))
            return
    except ImportError:
        pass

    # --- dict ---
    if isinstance(var, dict):
        lines.append(f"Keys ({len(var)}): {list(var.keys())[:20]}")
        if len(var) > 20:
            lines[-1] += f" ... (+{len(var) - 20} more)"
        if mode == "preview":
            for i, (k, v) in enumerate(var.items()):
                if i >= 10:
                    lines.append(f"  ... ({len(var) - 10} more entries)")
                    break
                val_repr = repr(v)
                if len(val_repr) > 120:
                    val_repr = val_repr[:120] + "..."
                lines.append(f"  {k!r}: {val_repr}")
        else:
            import json as _json

            try:
                lines.append(_json.dumps(var, indent=2, default=str))
            except (TypeError, ValueError):
                lines.append(repr(var))
        print(_truncate_sample("\n".join(lines), limit))
        return

    # --- list / tuple ---
    if isinstance(var, list | tuple):
        lines.append(f"Length: {len(var)}")
        if len(var) > 0:
            lines.append(
                f"Item types: {type(var[0]).__name__}"
                + (" (mixed)" if len(var) > 1 and type(var[0]) is not type(var[-1]) else "")
            )
        if mode == "preview":
            n = min(5, len(var))
            for i in range(n):
                val_repr = repr(var[i])
                if len(val_repr) > 200:
                    val_repr = val_repr[:200] + "..."
                lines.append(f"  [{i}] {val_repr}")
            if len(var) > 5:
                lines.append(f"  ... ({len(var) - 5} more)")
                val_repr = repr(var[-1])
                if len(val_repr) > 200:
                    val_repr = val_repr[:200] + "..."
                lines.append(f"  [{len(var) - 1}] {val_repr}")
        else:
            for i, item in enumerate(var):
                val_repr = repr(item)
                if len(val_repr) > 500:
                    val_repr = val_repr[:500] + "..."
                lines.append(f"  [{i}] {val_repr}")
        print(_truncate_sample("\n".join(lines), limit))
        return

    # --- set / frozenset ---
    if isinstance(var, set | frozenset):
        lines.append(f"Length: {len(var)}")
        items = sorted(var, key=repr)
        if mode == "preview":
            for item in items[:10]:
                lines.append(f"  {repr(item)}")
            if len(items) > 10:
                lines.append(f"  ... ({len(items) - 10} more)")
        else:
            for item in items:
                lines.append(f"  {repr(item)}")
        print(_truncate_sample("\n".join(lines), limit))
        return

    # --- str ---
    if isinstance(var, str):
        lines.append(f"Length: {len(var)}")
        if mode == "preview":
            preview = var[:500]
            if len(var) > 500:
                preview += f"\n... ({len(var) - 500} more chars)"
            lines.append(preview)
        else:
            lines.append(var)
        print(_truncate_sample("\n".join(lines), limit))
        return

    # --- bytes ---
    if isinstance(var, bytes):
        lines.append(f"Length: {len(var)} bytes")
        if mode == "preview":
            lines.append(repr(var[:200]))
            if len(var) > 200:
                lines.append(f"... ({len(var) - 200} more bytes)")
        else:
            lines.append(repr(var))
        print(_truncate_sample("\n".join(lines), limit))
        return

    # --- fallback: any object ---
    lines.append(f"Type: {type(var).__module__}.{type(var).__qualname__}")
    attrs = [a for a in dir(var) if not a.startswith("_")]
    if attrs:
        lines.append(f"Attributes ({len(attrs)}): {attrs[:20]}")
        if len(attrs) > 20:
            lines[-1] += f" ... (+{len(attrs) - 20} more)"
    r = repr(var)
    if mode == "preview" and len(r) > 500:
        r = r[:500] + "..."
    lines.append(f"Repr: {r}")
    print(_truncate_sample("\n".join(lines), limit))


def _truncate_sample(text, max_chars):
    """Truncate sample output to max_chars."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated, {len(text)} chars total)"


namespace["sample"] = sample

# --- Logging capture ---


class _CellLogHandler(_logging.Handler):
    """Logging handler that writes to whichever StringIO is current."""

    def __init__(self):
        super().__init__(level=_logging.INFO)
        self.buf = None
        self.setFormatter(_logging.Formatter("%(name)s: %(message)s"))

    def emit(self, record):
        if self.buf is not None:
            with contextlib.suppress(Exception):
                self.buf.write(self.format(record) + "\n")


_cell_log_handler = _CellLogHandler()
_logging.root.addHandler(_cell_log_handler)
_logging.root.setLevel(_logging.INFO)

while True:
    lines = []
    eof = False
    try:
        while True:
            line = _real_stdin.readline()
            if not line:
                eof = True
                break
            stripped = line.rstrip("\r\n")
            if stripped == _CELL_DELIM:
                break
            lines.append(line)
    except EOFError:
        eof = True
    if eof:
        break

    code = "".join(lines)
    if not code.strip():
        result = {"stdout": "", "stderr": "", "logs": "", "error": None}
        _real_stdout.write(_RESULT_START + "\n")
        _real_stdout.write(json.dumps(result) + "\n")
        _real_stdout.write(_RESULT_END + "\n")
        _real_stdout.flush()
        continue

    out_buf = io.StringIO()
    err_buf = io.StringIO()
    log_buf = io.StringIO()
    error = None
    _cell_log_handler.buf = log_buf

    sys.stdout = out_buf
    sys.stderr = err_buf
    _auto_installed = []
    try:
        compiled = compile(code, "<scratchpad>", "exec")
        exec(compiled, namespace)
    except ModuleNotFoundError as _mnf:
        _missing = _mnf.name
        if _missing:
            sys.stdout = _real_stdout
            sys.stderr = sys.__stderr__
            _cell_log_handler.buf = None
            _real_stdout.write(_PROGRESS_MARKER + " " + f"Installing {_missing}..." + "\n")
            _real_stdout.flush()
            import subprocess as _sp

            _uv_path = os.environ.get("ANTON_UV_PATH", "")
            if _uv_path:
                _pip = _sp.run(
                    [_uv_path, "pip", "install", "--python", sys.executable, _missing],
                    capture_output=True,
                    timeout=120,
                )
            else:
                _pip = _sp.run(
                    [sys.executable, "-m", "pip", "install", _missing],
                    capture_output=True,
                    timeout=120,
                )
            out_buf = io.StringIO()
            err_buf = io.StringIO()
            log_buf = io.StringIO()
            _cell_log_handler.buf = log_buf
            sys.stdout = out_buf
            sys.stderr = err_buf
            if _pip.returncode == 0:
                _auto_installed.append(_missing)
                try:
                    exec(compiled, namespace)
                except Exception:
                    error = traceback.format_exc()
            else:
                error = (
                    f"ModuleNotFoundError: No module named '{_missing}'\nAuto-install failed:\n{_pip.stderr.decode()}"
                )
        else:
            error = traceback.format_exc()
    except Exception:
        error = traceback.format_exc()
    finally:
        sys.stdout = _real_stdout
        sys.stderr = sys.__stderr__
        _cell_log_handler.buf = None

    # Persist session after each cell (best-effort).
    _dump_namespace(namespace)

    result = {
        "stdout": out_buf.getvalue(),
        "stderr": err_buf.getvalue(),
        "logs": log_buf.getvalue(),
        "error": error,
    }
    if _auto_installed:
        result["auto_installed"] = _auto_installed
    _real_stdout.write(_RESULT_START + "\n")
    _real_stdout.write(json.dumps(result) + "\n")
    _real_stdout.write(_RESULT_END + "\n")
    _real_stdout.flush()
