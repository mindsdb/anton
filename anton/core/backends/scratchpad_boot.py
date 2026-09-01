import contextlib
import io
import json
import os
import sys
import traceback

import dill

from anton.core.backends.wire import (
    CELL_DELIM,
    MISSING_MODULE_HINT,
    RESULT_START,
    RESULT_END,
    heal_surrogate_source,
)


# --- Python session persistence and namespace injection ---
PERSIST_SESSION = os.environ.get("ANTON_SCRATCHPAD_PERSIST_SESSION", "false").lower() in {"1", "true", "yes", "on"}

# NO default path (ENG-1124). The historical default was "/anton_scratchpad_session.pkl"
# — the filesystem ROOT, which no Cowork process can write to: macOS desktop gets EROFS
# from the sealed system volume, and the hosted container runs as non-root `anton`
# (uid 1000) so `/` is EACCES. Every dump failed, every failure was discarded, and
# session persistence was silently a no-op for ~2 months while reporting as enabled.
# An unset path now disables persistence *loudly* (see `_load_namespace`) rather than
# aiming writes at somewhere unwritable. The caller owns the path: `local.py` composes
# a per-pad, per-conversation one under the workspace and creates the directory.
SESSION_PATH = os.environ.get("ANTON_SCRATCHPAD_SESSION_PATH", "")

# The agent's workspace root, set by `local.py` ONLY when a real workspace is bound
# (ENG-1366). Used to resolve agent-authored modules while the snapshot is loaded —
# see `_workspace_on_syspath`. Empty means "no workspace", and every use is inert.
WORKSPACE_PATH = os.environ.get("ANTON_SCRATCHPAD_WORKSPACE_PATH", "")

# Snapshot size ceiling. Persisting after every cell costs nothing while it always
# fails; once it works, a namespace holding large frames would be rewritten on each
# cell. Above the cap we skip the write and say so, rather than stalling every cell on
# hundreds of MB of pickling.
_SESSION_MAX_DEFAULT = 100 * 1024 * 1024
try:
    SESSION_MAX_BYTES = int(
        os.environ.get("ANTON_SCRATCHPAD_SESSION_MAX_BYTES", str(_SESSION_MAX_DEFAULT))
    )
except ValueError:
    # A malformed override must never stop the scratchpad from booting.
    SESSION_MAX_BYTES = _SESSION_MAX_DEFAULT

# Never snapshot these: rebuilt on every boot, and per-cell state.
_SESSION_EXCLUDE_ALWAYS = frozenset({"__builtins__", "_anton_explainability_queries"})

# Helpers this module injects. They are excluded because they are rebuilt fresh each
# boot and some (`get_llm`, `web_search`, `query_minds_data`) close over live network
# clients that must not be pickled or carried between processes.
#
# Excluded BY IDENTITY, not by name (`_INJECTED_HELPERS` below). Excluding by name
# silently destroyed agent data: `sample` and `progress` are perfectly ordinary
# variable names (`sample = df.sample(100)`), and a name-based skip meant the agent's
# value was dropped from the snapshot and then re-injected as the helper on the next
# turn — so `print(sample)` returned `<function sample>` with no error. If the agent
# rebinds one of these names, its value is data and gets persisted like any other.
_INJECTED_HELPER_NAMES = frozenset(
    {"get_llm", "agentic_loop", "web_search", "query_minds_data", "progress", "sample"}
)

# The helper objects this boot actually created, recorded by `_inject_helper` at the
# point of definition. It must NOT be built from `namespace` after injection: on a
# restore the namespace already holds the agent's own value under that name, and
# `setdefault` leaves it there — so reading it back recorded the agent's DATA as though
# it were our helper, and the next dump then excluded it. Symptom: the value survived
# exactly one restart and became `<function sample>` on the second.
_INJECTED_HELPERS: dict = {}


def _inject_helper(name: str, fn) -> None:
    """Expose a helper to scratchpad code, and remember the object we injected.

    `setdefault`, not assignment: injections run after `_load_namespace`, so a plain
    assign would clobber an agent value restored under this name. Recording and
    injecting in one place keeps the two from drifting apart — that drift was the bug.
    """
    _INJECTED_HELPERS[name] = fn
    namespace.setdefault(name, fn)

# Names whose value dill could not pickle, mapped to the id() of the value that failed.
# Pickling is all-or-nothing per file, so ONE unpicklable object used to lose the whole
# namespace — and the objects that fail are exactly the ones long stateful tasks hold:
# DB connections (`sqlite3.Connection`, psycopg2, pyodbc), sockets (SMTP for a mail
# campaign), and generators. Verified with dill 0.4.1. Remembering the offenders keeps
# the expensive per-key scan to the first cell that hits one; keying on id() means a
# rebind to something picklable is retried rather than excluded forever.
_UNPICKLABLE: dict = {}

# Byte marker of dill's file-handle reconstructor inside a pickle stream. A pickled
# file object is stored as a CALL to `dill._dill._create_filehandle(name, mode, …)`,
# and that call runs `open(name, mode)` at LOAD time — for a `'w'` handle (even an
# already-closed one, e.g. `f` surviving a `with open(path, 'w') as f:` block) the
# open() itself truncates the file on disk. In cloud mode the snapshot is reloaded on
# every turn, so one leftover handle wiped artifact files turn after turn (ENG-1726).
# Scanning the serialised bytes instead of isinstance-checking the value catches
# handles nested inside containers too, and leaves harmless io.StringIO/io.BytesIO
# (which pickle by contents, not by reconstructor) alone.
_FILEHANDLE_MARKER = b"_create_filehandle"

# Persistence notes for the next cell's `logs`. Deliberately NOT `error` — a snapshot
# problem must not make the cell look failed, or it would feed the consecutive-error
# circuit breaker and the resilience nudge.
_session_notes: list[str] = []


# Snapshot envelope (ENG-1366). The payload is name -> individually-pickled bytes
# rather than one pickle of the whole namespace, so ONE value that cannot be rebuilt no
# longer discards every other variable: a pickle stream is a sequential program, and an
# exception part-way through it kills the remainder. The envelope holds nothing but
# str/int/bytes, so opening it can never fail on something the agent made.
#
# A TUPLE, deliberately not a dict. An anton predating this format loads the file and
# accepts ANY dict as the namespace itself — so a dict envelope would hand it a
# namespace containing `values` and `__anton_snapshot__` and nothing else, with the
# agent's real variables silently gone and NOTHING reported. That is precisely the
# silent no-op ENG-1124 was filed to end, and a rollback or an older desktop build
# resuming the conversation is enough to reach it. A non-dict trips that older loader's
# `isinstance(ns, dict)` guard instead, so it starts fresh AND says so. Verified against
# the real staging loader, both ways.
_SNAPSHOT_MAGIC = "__anton_snapshot__"
_SNAPSHOT_VERSION = 2


def _resolved_workspace() -> str | None:
    """The workspace root to make importable during a load, or None.

    Host-supplied (`conversation.project.path` -> `local.py`), never model-influenced:
    the pad NAME reaches the snapshot filename, but nothing the model chooses reaches
    this path. Re-resolved and required to be a real directory here anyway, so a
    malformed value is inert rather than trusted.
    """
    if not WORKSPACE_PATH:
        return None
    try:
        path = os.path.realpath(WORKSPACE_PATH)
    except OSError:
        return None
    return path if os.path.isdir(path) else None


@contextlib.contextmanager
def _workspace_on_syspath():
    """Make agent-authored modules importable for the duration of a snapshot load.

    Python puts the *script's* directory on `sys.path`, never the cwd — so a helper the
    agent wrote into the workspace and imported on turn 1 (having added the path itself)
    is unimportable in the fresh process that loads the snapshot. dill stores such
    objects by reference to their defining module, so that import failure is what
    discarded the whole namespace.

    APPENDED, never inserted at the front: the workspace is the agent's own directory
    and may contain files named after stdlib modules (`types.py`, `logging.py`). At the
    front those would shadow the real ones for the rest of the process; at the back they
    resolve only when nothing else claims the name. Verified both ways.

    Scoped to the load and removed afterwards, so the agent's own cell imports resolve
    exactly as they did before this change.
    """
    path = _resolved_workspace()
    if path is None or path in sys.path:
        yield
        return
    sys.path.append(path)
    try:
        yield
    finally:
        try:
            sys.path.remove(path)
        except ValueError:  # a cell rearranged sys.path mid-load
            pass


def _decode_values(blobs: dict) -> tuple[dict, dict]:
    """Unpickle each value independently. Returns (namespace, {name: reason})."""
    ns: dict = {}
    dropped: dict = {}
    for name, blob in blobs.items():
        if _FILEHANDLE_MARKER in blob:
            # Snapshots written before ENG-1726 can still hold pickled file handles;
            # materialising one re-opens the file in its original mode, truncating it
            # when that mode is 'w'. Refuse to load rather than damage workspace files.
            dropped[name] = "file handle: restoring would re-open (and truncate) the file"
            continue
        try:
            ns[name] = dill.loads(blob)
        except Exception as exc:
            # Per-value, so an unresolvable reference costs one variable, not all of
            # them. Most common cause: an object whose defining module is a .py file
            # the agent wrote and has since renamed or deleted.
            dropped[name] = f"{type(exc).__name__}: {exc}"
    return ns, dropped


# How many per-name failure reasons to spell out. Every dropped NAME is always listed
# — the agent needs those to know what to rebuild — but the reasons are tracebacks-worth
# of text and this note lands on `logs`, i.e. straight into the model's context on the
# next turn. A namespace that drops fifty names would otherwise spend more context
# explaining the loss than the loss cost.
_MAX_DROP_REASONS = 5


def _dropped_load_note(dropped: dict) -> str:
    names = sorted(dropped)
    shown = names[:_MAX_DROP_REASONS]
    detail = "; ".join(f"{name} ({dropped[name]})" for name in shown)
    if len(names) > len(shown):
        detail += f"; … and {len(names) - len(shown)} more"
    return (
        "Scratchpad session restored, but these variables could not be rebuilt and are "
        "now undefined: " + ", ".join(names) + ". Everything else was restored. This "
        "usually means an object whose class or function lives in a .py file this "
        "scratchpad wrote, which has since been renamed, moved or deleted — recreate "
        "those objects rather than relying on them persisting. Details: " + detail
    )


def _load_namespace() -> tuple[dict, str | None]:
    if not PERSIST_SESSION:
        return {"__builtins__": __builtins__}, None
    if not SESSION_PATH:
        return (
            {"__builtins__": __builtins__},
            "Scratchpad session persistence is enabled but "
            "ANTON_SCRATCHPAD_SESSION_PATH is unset, so nothing will survive this "
            "process. Variables and imports are lost when this scratchpad restarts.",
        )
    try:
        # The workspace covers BOTH reads: the per-value decode below, and a legacy
        # single-stream snapshot, which resolves its module references during
        # `dill.load` itself.
        with _workspace_on_syspath():
            with open(SESSION_PATH, "rb") as f:
                raw = dill.load(f)
            if isinstance(raw, tuple) and raw and raw[0] == _SNAPSHOT_MAGIC:
                if len(raw) != 3 or raw[1] != _SNAPSHOT_VERSION:
                    # A snapshot from a NEWER anton than this one. Refuse it rather
                    # than guess at its shape; the except below reports and starts
                    # fresh, which is the honest degradation.
                    raise TypeError(
                        f"Unsupported scratchpad snapshot version: {raw[1:2]}"
                    )
                ns, dropped = _decode_values(raw[2] or {})
            elif not isinstance(raw, dict):
                raise TypeError("Session file did not contain a namespace dict")
            else:
                # A snapshot written before this format existed: one stream holding the
                # real objects. Still readable, so a conversation in flight across the
                # upgrade keeps its state instead of paying a cold turn. It is
                # all-or-nothing by construction; the next dump rewrites it as v2.
                ns, dropped = raw, {}
        ns.setdefault("__builtins__", __builtins__)
        return ns, (_dropped_load_note(dropped) if dropped else None)
    except FileNotFoundError:
        # First cell of a brand-new pad — expected, not a problem.
        return {"__builtins__": __builtins__}, None
    except Exception:
        # Now reachable only for a torn/unreadable file or a legacy snapshot that still
        # will not load. Degrading to a fresh namespace is the old behaviour; the
        # difference is that now it is reported.
        return (
            {"__builtins__": __builtins__},
            "Failed to load scratchpad session; starting fresh.\n" + traceback.format_exc(),
        )


class _TooBig(Exception):
    """Raised mid-write once the snapshot passes SESSION_MAX_BYTES."""


class _CappedWriter:
    """File wrapper that aborts the write once it exceeds `limit` bytes.

    Checking the size *after* serialising bounded what we keep but not what it cost:
    a multi-GB namespace still paid a full serialise + write on every cell before being
    deleted, which is the exact cost the cap exists to avoid. Failing fast mid-write
    bounds both.
    """

    def __init__(self, fh, limit: int) -> None:
        self._fh = fh
        self._limit = limit
        self.written = 0

    def write(self, chunk) -> int:
        self.written += len(chunk)
        if self.written > self._limit:
            raise _TooBig()
        return self._fh.write(chunk)

    def flush(self) -> None:
        self._fh.flush()


def _snapshot_payload(ns: dict) -> dict:
    """The subset of `ns` worth persisting."""
    return {
        k: v
        for k, v in ns.items()
        if k not in _SESSION_EXCLUDE_ALWAYS
        # Injected helpers are skipped by IDENTITY, not by name: if the agent rebinds
        # `sample` or `progress` to its own data, that value is data and gets saved.
        and not (k in _INJECTED_HELPERS and v is _INJECTED_HELPERS[k])
        # Known-unpicklable, and still the same object that failed before.
        and _UNPICKLABLE.get(k) != id(v)
    }


def _encode_values(payload: dict) -> tuple[dict, list]:
    """Pickle each value on its own. Returns (blobs, dropped names).

    Isolating values at WRITE time is what lets the load be partial, and it also makes
    an unpicklable value ordinary rather than exceptional: the old code discovered the
    offender by re-walking the whole namespace after a failed bulk dump. Raises
    `_TooBig` as soon as the accumulated size passes the cap, so an oversized namespace
    stops costing a full serialise before being thrown away.
    """
    blobs: dict = {}
    dropped: list = []
    total = 0
    for key, value in payload.items():
        try:
            blob = dill.dumps(value)
        except Exception:
            # Live resources — DB connections, sockets, generators. Remembered by id()
            # so the retry is skipped next cell but a rebind to something picklable is
            # not excluded forever.
            _UNPICKLABLE[key] = id(value)
            dropped.append(key)
            continue
        if _FILEHANDLE_MARKER in blob:
            # File handles pickle without error but must never be persisted: loading
            # them re-runs open() and truncates 'w'-mode files (ENG-1726). Registered
            # in _UNPICKLABLE like any live resource, so the warning fires once and a
            # rebind to real data is retried.
            _UNPICKLABLE[key] = id(value)
            dropped.append(key)
            continue
        total += len(blob)
        if total > SESSION_MAX_BYTES:
            raise _TooBig()
        blobs[key] = blob
    return blobs, dropped


def _write_snapshot(blobs: dict, tmp_path: str) -> None:
    """Serialise the snapshot envelope to `tmp_path`, aborting past the size cap."""
    envelope = (_SNAPSHOT_MAGIC, _SNAPSHOT_VERSION, blobs)
    with open(tmp_path, "wb") as f:
        # _CappedWriter still guards the envelope overhead on top of the value bytes
        # already counted in `_encode_values`.
        dill.dump(envelope, _CappedWriter(f, SESSION_MAX_BYTES))


def _dump_namespace(ns: dict) -> str | None:
    if not PERSIST_SESSION or not SESSION_PATH:
        return None
    payload = _snapshot_payload(ns)
    # Include the pid: a pad can briefly overlap with its own replacement, and two
    # writers sharing one .tmp would corrupt each other's snapshot.
    tmp_path = f"{SESSION_PATH}.{os.getpid()}.tmp"

    def _discard_tmp() -> None:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    try:
        blobs, dropped = _encode_values(payload)
        # Write-then-replace. This process is killed abruptly at the end of a turn (and
        # by the inactivity watchdog mid-cell), so writing straight to SESSION_PATH
        # would eventually leave a torn pickle that the next process fails to load.
        # os.replace is atomic within a filesystem.
        _write_snapshot(blobs, tmp_path)
        os.replace(tmp_path, SESSION_PATH)
    except _TooBig:
        _discard_tmp()
        return (
            f"Scratchpad session NOT saved: the namespace exceeds the "
            f"{SESSION_MAX_BYTES:,}-byte snapshot cap. Variables will not survive a "
            "restart of this scratchpad — write large results to disk (or an artifact) "
            "instead of holding them in memory."
        )
    except Exception:
        _discard_tmp()
        return "Failed to dump scratchpad session.\n" + traceback.format_exc()
    if dropped:
        return (
            "Scratchpad session saved, but these variables could not be preserved and "
            "will be undefined if this scratchpad restarts: "
            + ", ".join(sorted(dropped))
            + ". Objects holding live resources (database connections, sockets, "
            "generators, open or closed file handles) cannot be saved — recreate "
            "them rather than relying on them persisting."
        )
    return None


# Persistent namespace across cells. Keep the load note — discarding it is how the
# broken path went unnoticed; it is surfaced on the first cell's `logs`.
namespace, _session_load_note = _load_namespace()
if _session_load_note:
    _session_notes.append(_session_load_note)
namespace["_anton_explainability_queries"] = []

# --- Inject get_llm() for LLM access from scratchpad code ---
_scratchpad_model = os.environ.get("ANTON_SCRATCHPAD_MODEL", "")
if _scratchpad_model:
    try:
        import asyncio as _llm_asyncio

        _scratchpad_provider_name = os.environ.get(
            "ANTON_SCRATCHPAD_PROVIDER", "anthropic"
        )
        if _scratchpad_provider_name in ("openai", "openai-compatible"):
            from anton.core.llm.openai import OpenAIProvider as _ProviderClass
        else:
            from anton.core.llm.anthropic import AnthropicProvider as _ProviderClass

        _llm_ssl_verify = (
            os.environ.get("ANTON_MINDS_SSL_VERIFY", "true").lower() != "false"
        )
        if _scratchpad_provider_name in ("openai", "openai-compatible"):
            # Explicitly pass base_url so Minds/openai-compatible endpoints work.
            # The OpenAI SDK may or may not pick up OPENAI_BASE_URL from env,
            # so we pass it directly to be safe.
            _llm_base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get(
                "ANTON_OPENAI_BASE_URL"
            )
            _llm_api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
                "ANTON_OPENAI_API_KEY"
            )
            _llm_api_version = os.environ.get("ANTON_OPENAI_API_VERSION") or None
            _llm_provider_kwargs: dict = {
                "api_key": _llm_api_key or None,
                "base_url": _llm_base_url or None,
                "ssl_verify": _llm_ssl_verify,
                "api_version": _llm_api_version,
            }
            # OpenAIProvider already defaults to supports_vision=True /
            # vision_format="openai" — every gateway we route openai-compatible
            # requests through (MindsHub/mdb.ai included) normalizes a standard
            # OpenAI image_url block into whatever the resolved backend natively
            # needs, so no per-host override is needed here. Forcing
            # vision_format="anthropic" for every mdb.ai/mindshub.ai host used
            # to assume every model behind it was Claude, which broke non-Claude
            # models sharing the same gateway (ENG-1992).
            # Resolve the OpenAI "flavor" so the injected web_search() helper can
            # route through whatever native web tooling the endpoint exposes.
            # Detect Minds/mdb.ai by HOST, not provider name (always "openai" for
            # an OpenAIProvider even on the minds gateway) — see resolve_web_flavor.
            _llm_provider_kwargs["flavor"] = _ProviderClass.resolve_web_flavor(
                _scratchpad_provider_name, _llm_base_url
            )
            _llm_provider = _ProviderClass(**_llm_provider_kwargs)
        else:
            _llm_provider = _ProviderClass()  # Anthropic doesn't need ssl_verify
        _llm_model = _scratchpad_model

        _LLM_HEARTBEAT_INTERVAL = 10  # seconds between heartbeats during LLM calls

        async def _run_with_heartbeat(coro):
            """Run an async coroutine while emitting progress heartbeats.

            Survival is covered by the cell-level liveness heartbeat thread;
            these progress lines exist for the user — "Waiting for LLM… (Ns)"
            is visible status during an in-cell LLM call that can block 30s+.
            """

            async def _heartbeat():
                elapsed = 0
                while True:
                    await _llm_asyncio.sleep(_LLM_HEARTBEAT_INTERVAL)
                    elapsed += _LLM_HEARTBEAT_INTERVAL
                    # Same lock as every other _real_stdout writer (see _wire_lock
                    # below): the liveness heartbeat thread is a genuine OS thread
                    # that writes for the whole cell span, including in-cell LLM
                    # calls, so an unlocked write here could land inside another
                    # writer's multi-line block and tear the parent's protocol.
                    with _wire_lock:
                        _real_stdout.write(
                            PROGRESS_MARKER + f" Waiting for LLM… ({elapsed}s)\n"
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

            def complete(
                self, *, system, messages, tools=None, tool_choice=None, max_tokens=4096
            ):
                """Call the LLM synchronously. Returns an LLMResponse.

                Emits "Waiting for LLM…" progress every 10s so the user sees
                status during long API calls (liveness is handled by the
                cell-level heartbeat thread).
                """
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

            async def complete_async(
                self, *, system, messages, tools=None, tool_choice=None, max_tokens=4096
            ):
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

            def generate_object(
                self, schema_class, *, system, messages, max_tokens=4096
            ):
                """Generate a structured object matching a Pydantic model.

                Uses tool_choice to force the LLM to return structured data.
                Supports single models and list[Model].

                The schema-building and unwrapping logic is shared with
                `LLMClient.generate_object` (in the main process) via
                `anton.core.llm.structured` — only the actual provider
                call differs between the two runtime contexts (sync
                subprocess here, async planning there).

                Args:
                    schema_class: A Pydantic BaseModel subclass, or list[Model].
                    system: System prompt.
                    messages: Conversation messages.
                    max_tokens: Max tokens for the LLM call.

                Returns:
                    An instance of schema_class (or a list of instances).
                """
                from anton.core.llm.structured import (
                    build_structured_tool,
                    looks_truncated,
                    raise_unusable_tool_call,
                    unwrap_structured_response,
                )

                tool, validator_class, is_list = build_structured_tool(
                    schema_class
                )

                response = self.complete(
                    system=system,
                    messages=messages,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": tool["name"]},
                    max_tokens=max_tokens,
                )

                if not response.tool_calls or any(
                    tc.repaired for tc in response.tool_calls
                ):
                    # Same classification as the async path (ENG-1081), and the
                    # same reason `parse_error` is left to the validation branch.
                    # Nothing retries here, but the message reaches the model as
                    # a traceback, so "you ran out of budget" is actionable
                    # where "did not return structured output" was not.
                    raise_unusable_tool_call(
                        response, tool_name=tool["name"], budget=max_tokens
                    )

                try:
                    return unwrap_structured_response(
                        response.tool_calls[0].input, validator_class, is_list
                    )
                except Exception:
                    if looks_truncated(response, max_tokens):
                        raise_unusable_tool_call(
                            response, tool_name=tool["name"], budget=max_tokens
                        )
                    raise

        _scratchpad_llm_instance = _ScratchpadLLM()

        def get_llm():
            """Get a pre-configured LLM client. No API keys needed."""
            return _scratchpad_llm_instance

        def agentic_loop(
            *, system, user_message, tools, handle_tool, max_turns=10, max_tokens=4096
        ):
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
            from anton.core.llm.provider import damaged_tool_call_result

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
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.input,
                        }
                    )
                messages.append({"role": "assistant", "content": assistant_content})

                # Execute each tool and collect results
                tool_results = []
                for tc in response.tool_calls:
                    # Arguments the model never finished are answered, not
                    # executed — same refusal as the session's tool loops, via
                    # the same shared builder.
                    damaged = damaged_tool_call_result(tc)
                    if damaged is not None:
                        tool_results.append(damaged)
                        continue

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

            # Hit max_turns
            return response.content if response else ""

        _WEB_SEARCH_SYSTEM = (
            "You are a web research assistant. Use your web search tool to find "
            "current, accurate information and answer the user's query directly "
            "and thoroughly. Prefer recent, reputable sources, and cite the source "
            "URLs you relied on inline or in a short list at the end."
        )

        def web_search(query, *, max_tokens=4096):
            """Answer a query using the configured LLM's native web search.

            Routes ``query`` to a single LLM completion with the provider's
            server-side web search tool enabled — Anthropic ``web_search``,
            OpenAI (BYOK) Responses-API ``web_search``, or the Minds Cloud /
            mdb.ai passthrough, depending on the configured provider. The model
            performs the search server-side and writes the answer in one round
            trip; this returns that narrative answer (a string), which usually
            includes the source links the model cited.

            If the configured provider/endpoint exposes no native web search
            (e.g. a generic OpenAI-compatible endpoint), a short explanatory
            message is returned instead of raising.
            """
            available = _llm_provider.native_web_tools()
            if "web_search" not in available:
                return (
                    "web_search is unavailable: the configured LLM "
                    "provider/endpoint does not expose a native web search tool. "
                    "Native web search is available on Anthropic, OpenAI (BYOK), "
                    "and Minds Cloud (mdb.ai)."
                )
            response = _llm_asyncio.run(
                _run_with_heartbeat(
                    _llm_provider.complete(
                        model=_llm_model,
                        system=_WEB_SEARCH_SYSTEM,
                        messages=[{"role": "user", "content": query}],
                        native_web_tools=available & {"web_search", "web_fetch"},
                        max_tokens=max_tokens,
                    )
                )
            )
            return response.content

        _inject_helper("get_llm", get_llm)
        _inject_helper("agentic_loop", agentic_loop)
        _inject_helper("web_search", web_search)
    except Exception:
        pass  # LLM not available — not fatal (e.g. anthropic not installed)

# --- Inject query_minds_data() for Minds datasource access from scratchpad ---
_minds_datasource = os.environ.get("ANTON_MINDS_DATASOURCE", "")
_minds_api_key = os.environ.get("ANTON_MINDS_API_KEY", "")
_minds_url = os.environ.get("ANTON_MINDS_URL", "")
_minds_engine = os.environ.get("ANTON_MINDS_DATASOURCE_ENGINE", "")
if _minds_datasource and _minds_api_key and _minds_url:
    try:
        import ssl as _minds_ssl
        import urllib.request as _minds_urllib

        _minds_ssl_verify = (
            os.environ.get("ANTON_MINDS_SSL_VERIFY", "true").lower() != "false"
        )

        def query_minds_data(query, datasource=None):
            """Query a Minds datasource with SQL. Returns dict with type, data, column_names, error_message."""
            ds = datasource or _minds_datasource
            url = f"{_minds_url}/api/v1/datasources/{ds}/query"
            payload = json.dumps({"query": query, "native_query": True}).encode()

            req = _minds_urllib.Request(url, data=payload, method="POST")
            req.add_header("Authorization", f"Bearer {_minds_api_key}")
            req.add_header("Content-Type", "application/json")
            req.add_header("Accept", "application/json")
            req.add_header(
                "User-Agent",
                "Mozilla/5.0 (compatible; Anton/1.0; +https://github.com/mindsdb/anton)",
            )
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
                    parsed = json.loads(resp.read().decode())
                    namespace.setdefault("_anton_explainability_queries", []).append({
                        "datasource": ds,
                        "sql": query,
                        "engine": _minds_engine or None,
                        "status": "ok",
                        "error_message": None,
                    })
                    return parsed
            except _minds_urllib.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode()
                except Exception:
                    pass
                namespace.setdefault("_anton_explainability_queries", []).append({
                    "datasource": ds,
                    "sql": query,
                    "engine": _minds_engine or None,
                    "status": "error",
                    "error_message": f"HTTP {e.code}: {body or e.reason}",
                })
                return {
                    "type": "error",
                    "data": None,
                    "column_names": None,
                    "error_message": f"HTTP {e.code}: {body or e.reason}",
                }
            except Exception as e:
                namespace.setdefault("_anton_explainability_queries", []).append({
                    "datasource": ds,
                    "sql": query,
                    "engine": _minds_engine or None,
                    "status": "error",
                    "error_message": str(e),
                })
                return {
                    "type": "error",
                    "data": None,
                    "column_names": None,
                    "error_message": str(e),
                }

        _inject_helper("query_minds_data", query_minds_data)
    except Exception:
        pass  # Minds query not available — not fatal

# Read-execute loop
_real_stdout = sys.stdout
_real_stdin = sys.stdin

import threading

from anton.core.backends.wire import (
    HEARTBEAT_MARKER,
    PROGRESS_MARKER,
    STDOUT_CHUNK_MARKER,
)

# All _real_stdout writes go through this lock: the heartbeat thread writes
# concurrently with the main thread's progress()/result emission, and a torn
# line would corrupt the parent's line-oriented protocol.
_wire_lock = threading.Lock()

# Env override exists for tests only; <= 0 disables the heartbeat entirely
# (restoring pre-heartbeat watchdog behavior). Same guard as SESSION_MAX_BYTES:
# a malformed override must never stop the scratchpad from booting.
try:
    _HEARTBEAT_INTERVAL = float(
        os.environ.get("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "10")
    )
except ValueError:
    _HEARTBEAT_INTERVAL = 10.0

# Per-tick cap on salvage chunk size: bounds a single wire line even when a
# cell floods stdout between ticks; the remainder ships on later ticks.
_CHUNK_MAX = 8_192

# The heartbeat thread reads this; the main loop points "buf" at the current
# cell's out_buf (it is swapped on auto-install retry). "shipped" is used by
# the stdout-salvage chunking (see the main loop).
_cell_out: dict = {"buf": None, "shipped": 0}


def _heartbeat_loop(stop: threading.Event) -> None:
    # A tick with new cell output ships it as a salvage chunk instead of a
    # bare beat — any line is liveness, and the parent keeps the chunks so a
    # killed cell can still report what it printed before dying. json.dumps
    # gives one-line framing (newlines/unicode escaped); a torn trailing
    # line read off the StringIO mid-write is acceptable in a salvage path.
    while not stop.wait(_HEARTBEAT_INTERVAL):
        buf = _cell_out["buf"]
        text = buf.getvalue() if buf is not None else ""
        new = text[_cell_out["shipped"] :]
        with _wire_lock:
            if new:
                chunk = new[:_CHUNK_MAX]
                _cell_out["shipped"] += len(chunk)
                _real_stdout.write(
                    STDOUT_CHUNK_MARKER + " " + json.dumps(chunk) + "\n"
                )
            else:
                _real_stdout.write(HEARTBEAT_MARKER + "\n")
            _real_stdout.flush()


_MAX_OUTPUT = 10_000


def progress(message=""):
    """Signal that long-running work is still active. Resets the inactivity timer."""
    with _wire_lock:
        _real_stdout.write(PROGRESS_MARKER + " " + str(message) + "\n")
        _real_stdout.flush()


_inject_helper("progress", progress)


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
                lines.append(
                    f"Min: {var.min()}, Max: {var.max()}, Mean: {var.mean():.4g}"
                )
            else:
                lines.append(
                    f"Min: {var.min()}, Max: {var.max()}, Mean: {var.mean():.4g}, Std: {var.std():.4g}"
                )
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
            lines.append(
                f"Item types: {type(var[0]).__name__}"
                + (
                    f" (mixed)"
                    if len(var) > 1 and type(var[0]) != type(var[-1])
                    else ""
                )
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


_inject_helper("sample", sample)

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
            if stripped == CELL_DELIM:
                break
            lines.append(line)
    except EOFError:
        eof = True
    if eof:
        break

    code = "".join(lines)
    # Heal lone surrogates before compile() — a non-ASCII Windows path byte can
    # arrive surrogate-escaped over stdin and would crash compile() with
    # "surrogates not allowed" (ENG-981).
    code = heal_surrogate_source(code)
    if not code.strip():
        result = {"stdout": "", "stderr": "", "logs": "", "error": None}
        with _wire_lock:
            _real_stdout.write(RESULT_START + "\n")
            _real_stdout.write(json.dumps(result) + "\n")
            _real_stdout.write(RESULT_END + "\n")
            _real_stdout.flush()
        continue

    # Liveness heartbeat: a daemon thread pings the real pipe while this cell
    # runs, so the parent's inactivity watchdog sees activity through a
    # deliberate sleep or a blocking call, not just through stdout/progress().
    _hb_stop = threading.Event()
    _hb_thread = None
    if _HEARTBEAT_INTERVAL > 0:
        _hb_thread = threading.Thread(
            target=_heartbeat_loop, args=(_hb_stop,), daemon=True
        )
        _hb_thread.start()
    try:
        _cwd_before = os.getcwd()
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        log_buf = io.StringIO()
        # Reset "shipped" BEFORE re-pointing "buf": the heartbeat thread may
        # read between the two assignments, and the worst case must be a
        # re-shipped duplicate chunk, never a skipped one.
        _cell_out["shipped"] = 0
        _cell_out["buf"] = out_buf
        error = None
        namespace["_anton_explainability_queries"] = []
        _cell_log_handler.buf = log_buf

        sys.stdout = out_buf
        sys.stderr = err_buf
        try:
            compiled = compile(code, "<scratchpad>", "exec")
            exec(compiled, namespace)
        except ModuleNotFoundError as _mnf:
            # Don't pip-install a name pulled from an exception — it may be a
            # hallucinated import, and the string is attacker-controllable.
            # Hint goes before the traceback: callers key off its last line.
            hint = ""
            if _mnf.name:
                hint = MISSING_MODULE_HINT.format(name=_mnf.name)
            error = hint + traceback.format_exc()
        except Exception:
            error = traceback.format_exc()
        finally:
            sys.stdout = _real_stdout
            sys.stderr = sys.__stderr__
            _cell_log_handler.buf = None

        stdout_val = out_buf.getvalue()
        if len(stdout_val) > _MAX_OUTPUT:
            stdout_val = (
                stdout_val[:_MAX_OUTPUT]
                + f"\n\n... (truncated, {len(stdout_val)} chars total)"
            )

        # Warn-only chdir visibility (ENG-578 fix #5): a cell's os.chdir
        # silently persists into every later cell of this pad — tell the
        # model instead of resetting, so deliberate chdir workflows keep
        # working. Appended AFTER the truncation clamp so the note can never
        # be truncated away, and regardless of `error` so a cell that raised
        # after the chdir still reports it.
        _cwd_after = os.getcwd()
        if _cwd_after != _cwd_before:
            stdout_val = (
                stdout_val
                + ("\n" if stdout_val and not stdout_val.endswith("\n") else "")
                + f"Note: this cell changed the working directory from {_cwd_before} "
                f"to {_cwd_after}; it persists for subsequent cells in this scratchpad.\n"
            )

        # Persist session after each cell. Keep the return value: a swallowed failure here
        # is exactly how ENG-1124 stayed invisible for two months.
        _session_dump_note = _dump_namespace(namespace)
        if _session_dump_note:
            _session_notes.append(_session_dump_note)

        # Surface persistence problems on `logs`, never on `error` — see `_session_notes`.
        logs_val = log_buf.getvalue()
        if _session_notes:
            logs_val = (logs_val.rstrip("\n") + "\n\n" if logs_val.strip() else "") + "\n".join(
                _session_notes
            )
            _session_notes.clear()
    finally:
        # Always stop the heartbeat, even if result assembly above raised —
        # otherwise the thread leaks and keeps ticking into the next cell's
        # read loop (see test_no_stray_beats_corrupt_next_cell).
        _hb_stop.set()
        if _hb_thread is not None:
            _hb_thread.join(timeout=2)
        _cell_out["buf"] = None

    result = {
        "stdout": stdout_val,
        "stderr": err_buf.getvalue(),
        "logs": logs_val,
        "error": error,
        "explainability_queries": list(namespace.get("_anton_explainability_queries", [])),
    }
    with _wire_lock:
        _real_stdout.write(RESULT_START + "\n")
        _real_stdout.write(json.dumps(result) + "\n")
        _real_stdout.write(RESULT_END + "\n")
        _real_stdout.flush()
