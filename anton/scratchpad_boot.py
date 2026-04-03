import io
import json
import os
import sys
import traceback

_CELL_DELIM = "__ANTON_CELL_END__"
_RESULT_START = "__ANTON_RESULT__"
_RESULT_END = "__ANTON_RESULT_END__"

# Persistent namespace across cells
namespace = {"__builtins__": __builtins__}

# --- Inject get_llm() for LLM access from scratchpad code ---
_scratchpad_model = os.environ.get("ANTON_SCRATCHPAD_MODEL", "")
if _scratchpad_model:
    try:
        import asyncio as _llm_asyncio

        _scratchpad_provider_name = os.environ.get("ANTON_SCRATCHPAD_PROVIDER", "anthropic")
        if _scratchpad_provider_name in ("openai", "openai-compatible"):
            from anton.llm.openai import OpenAIProvider as _ProviderClass
        else:
            from anton.llm.anthropic import AnthropicProvider as _ProviderClass

        _llm_ssl_verify = os.environ.get("ANTON_MINDS_SSL_VERIFY", "true").lower() != "false"
        if _scratchpad_provider_name in ("openai", "openai-compatible"):
            # Explicitly pass base_url so Minds/openai-compatible endpoints work.
            # The OpenAI SDK may or may not pick up OPENAI_BASE_URL from env,
            # so we pass it directly to be safe.
            _llm_base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("ANTON_OPENAI_BASE_URL")
            _llm_api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTON_OPENAI_API_KEY")
            _llm_provider = _ProviderClass(
                api_key=_llm_api_key or None,
                base_url=_llm_base_url or None,
                ssl_verify=_llm_ssl_verify,
            )
        else:
            _llm_provider = _ProviderClass()  # Anthropic doesn't need ssl_verify
        _llm_model = _scratchpad_model

        _LLM_HEARTBEAT_INTERVAL = 10  # seconds between heartbeats during LLM calls

        async def _run_with_heartbeat(coro):
            """Run an async coroutine while emitting progress heartbeats.

            LLM API calls can block for 30s+.  Without heartbeats, the
            scratchpad inactivity timeout (30s) kills the process.  This
            wrapper runs a heartbeat task alongside the real work.
            """
            async def _heartbeat():
                elapsed = 0
                while True:
                    await _llm_asyncio.sleep(_LLM_HEARTBEAT_INTERVAL)
                    elapsed += _LLM_HEARTBEAT_INTERVAL
                    _real_stdout.write(
                        _PROGRESS_MARKER + f" Waiting for LLM… ({elapsed}s)\n"
                    )
                    _real_stdout.flush()

            beat = _llm_asyncio.create_task(_heartbeat())
            try:
                return await coro
            finally:
                beat.cancel()
                try:
                    await beat
                except _llm_asyncio.CancelledError:
                    pass

        class _ScratchpadLLM:
            """Sync LLM wrapper for scratchpad use. Mirrors SkillLLM interface."""

            @property
            def model(self):
                return _llm_model

            def complete(self, *, system, messages, tools=None, tool_choice=None, max_tokens=4096):
                """Call the LLM synchronously. Returns an LLMResponse.

                Automatically emits progress heartbeats every 10s so that
                long API calls don't trip the scratchpad inactivity timeout.
                """
                return _llm_asyncio.run(_run_with_heartbeat(
                    _llm_provider.complete(
                        model=_llm_model,
                        system=system,
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        max_tokens=max_tokens,
                    )
                ))

            async def complete_async(self, *, system, messages, tools=None, tool_choice=None, max_tokens=4096):
                """Call the LLM asynchronously. Returns an LLMResponse.

                Use this inside async code (e.g. asyncio.gather) for concurrent
                LLM calls.  Emits heartbeats automatically like complete().
                """
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
                """Generate a structured object matching a Pydantic model.

                Uses tool_choice to force the LLM to return structured data.
                Supports single models and list[Model].

                Args:
                    schema_class: A Pydantic BaseModel subclass, or list[Model].
                    system: System prompt.
                    messages: Conversation messages.
                    max_tokens: Max tokens for the LLM call.

                Returns:
                    An instance of schema_class (or a list of instances).
                """
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

                import json as _json
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
            """Run a synchronous LLM tool-call loop.

            The LLM reasons, calls tools via handle_tool(name, inputs) -> str,
            and iterates until it produces a final text response.

            Args:
                system: System prompt for the LLM.
                user_message: Initial user message.
                tools: Tool definitions (Anthropic tool schema format).
                handle_tool: Callback (tool_name, tool_input) -> result_string.
                max_turns: Safety limit on LLM round-trips (default 10).
                max_tokens: Max tokens per LLM call.

            Returns:
                The final text response from the LLM.
            """
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

                # Build assistant message with text + tool_use blocks
                assistant_content = []
                if response.content:
                    assistant_content.append({"type": "text", "text": response.content})
                for tc in response.tool_calls:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    })
                messages.append({"role": "assistant", "content": assistant_content})

                # Execute each tool and collect results
                tool_results = []
                for tc in response.tool_calls:
                    try:
                        result = handle_tool(tc.name, tc.input)
                    except Exception as exc:
                        result = f"Error: {exc}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result,
                    })
                messages.append({"role": "user", "content": tool_results})

            # Hit max_turns
            return response.content if response else ""

        namespace["get_llm"] = get_llm
        namespace["agentic_loop"] = agentic_loop
    except Exception:
        pass  # LLM not available — not fatal (e.g. anthropic not installed)

# --- Inject query_minds_data() for Minds datasource access from scratchpad ---
_minds_datasource = os.environ.get("ANTON_MINDS_DATASOURCE", "")
_minds_api_key = os.environ.get("ANTON_MINDS_API_KEY", "")
_minds_url = os.environ.get("ANTON_MINDS_URL", "")
if _minds_datasource and _minds_api_key and _minds_url:
    try:
        import ssl as _minds_ssl
        import urllib.request as _minds_urllib

        _minds_ssl_verify = os.environ.get("ANTON_MINDS_SSL_VERIFY", "true").lower() != "false"

        def query_minds_data(query, datasource=None):
            """Query a Minds datasource with SQL. Returns dict with type, data, column_names, error_message."""
            ds = datasource or _minds_datasource
            url = f"{_minds_url}/api/v1/datasources/{ds}/query"
            payload = json.dumps({"query": query, "native_query": True}).encode()

            req = _minds_urllib.Request(url, data=payload, method="POST")
            req.add_header("Authorization", f"Bearer {_minds_api_key}")
            req.add_header("Content-Type", "application/json")
            req.add_header("Accept", "application/json")
            req.add_header("User-Agent", "Mozilla/5.0 (compatible; Anton/1.0; +https://github.com/mindsdb/anton)")
            req.add_header("Accept-Language", "en-US,en;q=0.9")
            req.add_header("Accept-Encoding", "identity")
            req.add_header("Connection", "keep-alive")

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
                try:
                    body = e.read().decode()
                except Exception:
                    pass
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
    except Exception:
        pass  # Minds query not available — not fatal

# Read-execute loop
_real_stdout = sys.stdout
_real_stdin = sys.stdin

_PROGRESS_MARKER = "__ANTON_PROGRESS__"
_MAX_OUTPUT = 10_000

def progress(message=""):
    """Signal that long-running work is still active. Resets the inactivity timer."""
    _real_stdout.write(_PROGRESS_MARKER + " " + str(message) + "\n")
    _real_stdout.flush()

namespace["progress"] = progress

def sample(var, mode="preview", _name=None):
    """Inspect a variable with type-aware formatting.

    Args:
        var: The variable to inspect.
        mode: "preview" (default) — compact summary. "full" — complete dump.
        _name: Optional label printed as header (auto-detected when possible).

    Prints formatted output to stdout (captured by the cell).
    """
    _MAX_PREVIEW = 2000
    _MAX_FULL = 10000
    limit = _MAX_PREVIEW if mode == "preview" else _MAX_FULL

    header = f"[sample:{type(var).__name__}]"
    if _name:
        header = f"[sample:{_name} ({type(var).__name__})]"

    lines = [header]

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

    if isinstance(var, (list, tuple)):
        kind = type(var).__name__
        lines.append(f"Length: {len(var)}")
        if len(var) > 0:
            lines.append(f"Item types: {type(var[0]).__name__}" +
                         (f" (mixed)" if len(var) > 1 and type(var[0]) != type(var[-1]) else ""))
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

    if isinstance(var, (set, frozenset)):
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

    lines.append(f"Type: {type(var).__module__}.{type(var).__qualname__}")
    # Show public attributes
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
# Libraries like httpx, urllib3, etc. use Python logging. By default these
# messages are silently dropped (no handler configured). We set up a handler
# that writes to a per-cell StringIO so the LLM can see connection info,
# warnings, and errors from libraries.
import logging as _logging

class _CellLogHandler(_logging.Handler):
    """Logging handler that writes to whichever StringIO is current."""
    def __init__(self):
        super().__init__(level=_logging.INFO)
        self.buf = None
        self.setFormatter(_logging.Formatter("%(name)s: %(message)s"))

    def emit(self, record):
        if self.buf is not None:
            try:
                self.buf.write(self.format(record) + "\n")
            except Exception:
                pass

_cell_log_handler = _CellLogHandler()
_logging.root.addHandler(_cell_log_handler)
_logging.root.setLevel(_logging.INFO)

while True:
    lines = []
    eof = False
    try:
        # Use explicit readline() instead of iterating stdin.  On Windows,
        # Python's file iterator over a pipe uses internal block buffering
        # (~8 KB) and won't yield lines until the buffer fills or the pipe
        # closes — causing a deadlock.  readline() returns immediately on \n.
        while True:
            line = _real_stdin.readline()
            if not line:
                # EOF — parent closed stdin
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
        # Auto-install the missing module and retry the cell once
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
                    capture_output=True, timeout=120,
                )
            else:
                _pip = _sp.run(
                    [sys.executable, "-m", "pip", "install", _missing],
                    capture_output=True, timeout=120,
                )
            # Reset buffers and retry
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
                    f"ModuleNotFoundError: No module named '{_missing}'\n"
                    f"Auto-install failed:\n{_pip.stderr.decode()}"
                )
        else:
            error = traceback.format_exc()
    except Exception:
        error = traceback.format_exc()
    finally:
        sys.stdout = _real_stdout
        sys.stderr = sys.__stderr__
        _cell_log_handler.buf = None

    stdout_val = out_buf.getvalue()
    if len(stdout_val) > _MAX_OUTPUT:
        stdout_val = stdout_val[:_MAX_OUTPUT] + f"\n\n... (truncated, {len(stdout_val)} chars total)"
    result = {
        "stdout": stdout_val,
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
