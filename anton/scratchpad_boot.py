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
        if _scratchpad_provider_name == "openai":
            from anton.llm.openai import OpenAIProvider as _ProviderClass
        else:
            from anton.llm.anthropic import AnthropicProvider as _ProviderClass

        _llm_provider = _ProviderClass()  # reads API key from env
        _llm_model = _scratchpad_model

        class _ScratchpadLLM:
            """Sync LLM wrapper for scratchpad use. Mirrors SkillLLM interface."""

            @property
            def model(self):
                return _llm_model

            def complete(self, *, system, messages, tools=None, tool_choice=None, max_tokens=4096):
                """Call the LLM synchronously. Returns an LLMResponse."""
                return _llm_asyncio.run(_llm_provider.complete(
                    model=_llm_model,
                    system=system,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=max_tokens,
                ))

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

# --- Inject minds_client when a Mind is connected ---
_minds_connection_raw = os.environ.get("MINDS_CONNECTION", "")
if _minds_connection_raw:
    try:
        _minds_conn = json.loads(_minds_connection_raw)
        _minds_url = _minds_conn.get("url", "")
        _minds_api_key = _minds_conn.get("api_key") or os.environ.get("MINDS_API_KEY", "")
        _minds_mind_name = _minds_conn.get("mind_name", "")
        if _minds_url and _minds_api_key and _minds_mind_name:
            from anton.minds_query import MindsQueryClient
            _minds_verify_ssl = _minds_conn.get("verify_ssl", True)
            _minds_client_instance = MindsQueryClient(
                mindsserver_url=_minds_url,
                api_key=_minds_api_key,
                mind_name=_minds_mind_name,
                verify_ssl=_minds_verify_ssl,
                progress_fn=progress,
            )
            namespace["minds_client"] = _minds_client_instance
    except Exception as _minds_exc:
        import traceback as _tb
        sys.stderr.write(f"[scratchpad] minds_client init failed: {_minds_exc}\n")
        sys.stderr.write(_tb.format_exc() + "\n")
        sys.stderr.flush()

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

    # --- set / frozenset ---
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
    try:
        for line in _real_stdin:
            stripped = line.rstrip("\n")
            if stripped == _CELL_DELIM:
                break
            lines.append(line)
        else:
            # EOF — parent closed stdin
            break
    except EOFError:
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

    result = {
        "stdout": out_buf.getvalue(),
        "stderr": err_buf.getvalue(),
        "logs": log_buf.getvalue(),
        "error": error,
    }
    _real_stdout.write(_RESULT_START + "\n")
    _real_stdout.write(json.dumps(result) + "\n")
    _real_stdout.write(_RESULT_END + "\n")
    _real_stdout.flush()
