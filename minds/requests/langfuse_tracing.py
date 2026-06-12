import contextlib
import functools
import json
import traceback
from typing import Any

from minds.common.logger import get_logger
from minds.requests.context import Context, create_langfuse_context


class _NoOpObservation:
    """No-op observation returned by _NoOpLangfuseClient.start_observation."""

    def end(self, **kwargs):
        return self

    def update(self, **kwargs):
        return self

    def update_trace(self, **kwargs):
        return self


logger = get_logger(__name__)


class _NoOpLangfuseClient:
    def get_current_trace_id(self):
        return "disabled"

    def update_current_generation(self, **kwargs):
        pass

    def start_observation(self, **kwargs):
        return _NoOpObservation()

    def get_current_observation_id(self):
        return "disabled"

    def update_current_trace(self, **kwargs):
        pass


def get_client():
    """Lazy accessor for the langfuse client. Defers the (heavy) langfuse import
    until first request, so workers don't pay the cost during startup."""
    try:
        from langfuse import get_client as _langfuse_get_client

        return _langfuse_get_client()
    except ImportError:
        return _NoOpLangfuseClient()


def lazy_observe(**observe_kwargs):
    """Lazy variant of langfuse's @observe decorator.

    Defers importing langfuse until the wrapped function is first called, so
    decorating a module-level function does not trigger the import at boot.
    """

    def decorator(fn):
        _wrapped = None
        _langfuse_available = True

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            nonlocal _wrapped, _langfuse_available
            if _wrapped is None:
                try:
                    from langfuse import observe

                    _wrapped = observe(**observe_kwargs)(fn)
                except ImportError:
                    _wrapped = fn
                    _langfuse_available = False
            if not _langfuse_available:
                # langfuse's observe wrapper pops these reserved call-time
                # kwargs; without it the raw function would receive them and
                # raise TypeError. Strip them so trace propagation degrades to
                # a no-op when langfuse isn't installed.
                for reserved in ("langfuse_trace_id", "langfuse_parent_observation_id", "langfuse_public_key"):
                    kwargs.pop(reserved, None)
            return await _wrapped(*args, **kwargs)

        return wrapper

    return decorator


def setup_langfuse_observation(context: Context):
    """
    Set up Langfuse observation for the current request context.
    This function creates a Langfuse context from the provided request context,
    updates the current observation, and retrieves the trace ID.

    When the request carries the Langfuse-proxy convention headers
    (``Langfuse-Session-Id`` / ``Langfuse-Tags`` / ``Langfuse-Metadata``)
    the trace is also stamped with ``session_id``, the client-supplied tags
    are merged into our identity tags, the metadata blob is merged into the
    trace's metadata, and — when both ``harness`` and ``turn_id`` are present
    in the client metadata — the trace's display name is set to
    ``"{harness}:turn-{turn_id}"`` so dashboards can scan multi-turn loops
    without opening each trace.

    Args:
            context (Context): The context for the request, including user and metadata.
    Returns:
            str: The trace ID for the current observation, or a default request ID if an error occurs.
    """
    logger.debug(f"Setting up Langfuse observation for context: {context}")

    try:
        current_langfuse_context = create_langfuse_context(context)

        tags_string = [str(item) for item in current_langfuse_context.tags]
        # Merge our identity metadata with the client-supplied Langfuse-Metadata.
        # Identity wins on conflict because we record user_id / org / request_id
        # for filtering and they should be authoritative even if the client
        # passes the same keys.
        metadata_dict: dict[str, Any] = dict(current_langfuse_context.extra_metadata or {})
        metadata_dict.update(current_langfuse_context.metadata.model_dump())

        langfuse_client = get_client()

        update_kwargs: dict[str, Any] = {
            "user_id": str(context.user_id),
            "metadata": metadata_dict,
            "tags": tags_string,
        }
        if current_langfuse_context.session_id:
            update_kwargs["session_id"] = current_langfuse_context.session_id
        # When we've adopted an upstream trace (Langfuse-Trace-Id present) the
        # trace name belongs to the caller (e.g. Anton's root). Renaming it to
        # our "harness:turn-N" would clobber the owner's trace name, so only set
        # the display name when this request owns the trace.
        if current_langfuse_context.trace_name and not context.langfuse_trace_id:
            update_kwargs["name"] = current_langfuse_context.trace_name

        # update_current_trace is unaware of the harness convention; pass the
        # subset of keyword args it supports, fall back to the legacy 3-arg
        # call if the SDK rejects the new keys (defensive against older
        # stubs in tests).
        try:
            langfuse_client.update_current_trace(**update_kwargs)
        except TypeError as exc:
            logger.warning(
                "update_current_trace rejected extended kwargs, retrying with legacy args: %s",
                exc,
            )
            langfuse_client.update_current_trace(
                user_id=str(context.user_id),
                metadata=metadata_dict,
                tags=tags_string,
            )

        trace_id = langfuse_client.get_current_trace_id()
        logger.debug(f"Trace ID: {trace_id}")

        if trace_id:
            logger.debug(
                f"Created langfuse context with trace ID: {trace_id} session_id="
                f"{current_langfuse_context.session_id} name={current_langfuse_context.trace_name}"
            )
        else:
            logger.error("Failed to retrieve trace ID from Langfuse context.")

    except Exception as e:
        logger.error(f"Error updating Langfuse observation: {e}")
        logger.error(traceback.format_exc())


def get_langfuse_trace_id():
    """
    Get the trace ID for the current Langfuse observation.
    Returns:
            str: The trace ID for the current observation, or a default request ID if an error occurs.
    """
    try:
        langfuse_client = get_client()
        return langfuse_client.get_current_trace_id()
    except Exception as e:
        logger.error(f"Error getting Langfuse trace ID: {e}")
        logger.error(traceback.format_exc())
        return None


def capture_langfuse_generation_context() -> dict | None:
    """
    Capture the current Langfuse trace_id + observation_id so a generation can
    be enriched (or a child generation attached) AFTER the surrounding @observe
    scope has closed (e.g. inside a streaming background task that outlives the
    decorated request handler).

    Must be called while the @observe-decorated function is still on the call
    stack — typically right after setup_langfuse_observation(...).

    Returns:
        A dict {"trace_id": str, "parent_span_id": str} compatible with
        langfuse.types.TraceContext, or None if Langfuse is disabled / no
        active span / on error.
    """
    try:
        langfuse_client = get_client()
        trace_id = langfuse_client.get_current_trace_id()
        observation_id = langfuse_client.get_current_observation_id()

        if not trace_id or trace_id == "disabled":
            return None

        ctx: dict = {"trace_id": trace_id}
        if observation_id and observation_id != "disabled":
            ctx["parent_span_id"] = observation_id
        return ctx
    except Exception as e:
        logger.error(f"Error capturing Langfuse generation context: {e}")
        logger.error(traceback.format_exc())
        return None


def update_generation_usage(
    usage: tuple[int, int] | None,
    *,
    model: str | None = None,
    trace_context: dict | None = None,
    name: str = "llm-usage",
    input: Any = None,
    output: Any = None,
    metadata: dict | None = None,
    trace_input: Any = None,
    trace_output: Any = None,
) -> None:
    """
    Record LLM token usage on a Langfuse generation.

    Two modes:
    - In-scope (``trace_context is None``): updates the *current* @observe
      generation in place via ``client.update_current_generation``. Use this
      from any code path that runs synchronously inside the @observe-decorated
      handler (all non-streaming paths).
    - Detached (``trace_context`` provided): creates a child generation
      observation attached to the trace via
      ``client.start_observation(trace_context=..., as_type="generation")`` and
      ends it. Use this when usage is captured AFTER the @observe scope has
      closed (every streaming path: the body iterator is consumed by the ASGI
      server only after the decorated handler returns).

    Token counts are emitted under Langfuse's canonical generic keys
    (``input`` / ``output`` / ``total``) so cost rolls up via Langfuse's model
    registry against the supplied ``model``.

    Args:
        usage: (input_tokens, output_tokens). If None, this is a no-op.
        model: Model identifier as stored on the DB row (e.g. "gpt-4o-mini").
        trace_context: Optional dict with "trace_id" and optional
            "parent_span_id". If provided, a child generation is attached to
            that trace; otherwise the current generation is updated in place.
        name: Name for the child generation in detached mode (ignored in
            in-scope mode).
        input: Optional input payload to attach (Langfuse displays it).
        output: Optional output payload to attach (Langfuse displays it).
        metadata: Optional arbitrary metadata to attach to the generation
            (Langfuse displays it and lets you filter traces by it). Used by
            passthrough flows to record alias / provider / reasoning_effort
            so downstream analytics can slice by alias surface separately
            from the concrete upstream model.
        trace_input: Optional turn-level input to stamp on the *trace* (not the
            generation). Used for whole-turn input→output evals when many calls
            share one adopted trace: pass the turn's logical input (idempotent
            across calls). ``None`` leaves the trace input untouched.
        trace_output: Optional turn-level output to stamp on the *trace*. Set it
            to each call's output; with many calls on one trace the last call to
            finish wins, leaving the turn's final answer as the trace output.
            ``None`` leaves the trace output untouched. Both trace fields are
            written on the same observation we already create/update here, so
            there is no extra Langfuse call beyond a single attribute write.
    """
    if usage is None:
        logger.debug("update_generation_usage: usage is None, skipping")
        return

    input_tokens, output_tokens = usage
    in_t = int(input_tokens or 0)
    out_t = int(output_tokens or 0)
    usage_details = {
        "input": in_t,
        "output": out_t,
        "total": in_t + out_t,
    }

    # Only the fields the caller actually supplied — passing None would not
    # clobber (Langfuse drops None) but building the dict keeps the calls clean
    # and lets us skip the trace write entirely when there's nothing to set.
    trace_io_kwargs = {k: v for k, v in (("input", trace_input), ("output", trace_output)) if v is not None}

    try:
        langfuse_client = get_client()
        if trace_context is None:
            langfuse_client.update_current_generation(
                model=model,
                usage_details=usage_details,
                input=input,
                output=output,
                metadata=metadata,
            )
            if trace_io_kwargs:
                langfuse_client.update_current_trace(**trace_io_kwargs)
            logger.debug(
                f"Updated current Langfuse generation with usage_details={usage_details} "
                f"model={model} metadata={metadata} trace_io={list(trace_io_kwargs)}"
            )
        else:
            obs = langfuse_client.start_observation(
                trace_context=trace_context,
                name=name,
                as_type="generation",
                model=model,
                usage_details=usage_details,
                input=input,
                output=output,
                metadata=metadata,
            )
            # Stamp turn-level I/O on the trace via this same observation, so we
            # don't create an extra span just to carry trace attributes.
            if trace_io_kwargs:
                with contextlib.suppress(Exception):
                    obs.update_trace(**trace_io_kwargs)
            # Defensive: ending may fail on stub clients; this is non-fatal.
            with contextlib.suppress(Exception):
                obs.end()
            logger.debug(
                "Attached child Langfuse generation "
                f"trace_id={trace_context.get('trace_id')} "
                f"parent_span_id={trace_context.get('parent_span_id')} "
                f"usage_details={usage_details} model={model} metadata={metadata} trace_io={list(trace_io_kwargs)}"
            )
    except Exception as e:
        logger.error(f"Error updating Langfuse generation usage: {e}")
        logger.error(traceback.format_exc())


def record_tool_call_spans(
    *,
    tool_calls: list[dict] | None,
    trace_context: dict | None,
    metadata: dict | None = None,
) -> None:
    """
    Emit one Langfuse child span per ``tool_call`` returned by the model.

    Today's passthrough surfaces ``tool_calls`` in the final ChatCompletion
    response and expects the client to execute each one and post the result
    back on the next turn. From Langfuse's perspective every tool call is its
    own unit of work — we record it as a ``span`` observation with
    ``name="tool:{tool_name}"``, ``input`` = parsed JSON arguments (falling
    back to the raw string if the model emitted malformed JSON), and
    ``metadata`` carrying the call id plus any caller-provided extras (alias,
    provider, etc.). This makes tool selection + arguments filterable in the
    Langfuse UI without re-parsing the parent generation's output.

    No tool *result* span is emitted here — the result arrives on the next
    request as a ``tool``-role message and lands on that turn's trace via
    the input payload, not this one.

    Args:
        tool_calls: ChatCompletion ``tool_calls`` list (OpenAI shape, with
            ``id`` and ``function.name`` / ``function.arguments``). ``None``
            or empty → no-op.
        trace_context: Detached-mode trace context (from
            :func:`capture_langfuse_generation_context`). When ``None``,
            spans are attached to the current ``@observe`` scope. Pass the
            captured context whenever this runs after the parent scope has
            closed (streaming body iterators).
        metadata: Extra metadata merged into every emitted span (e.g.
            ``{"alias": "sonnet", "provider": "anthropic"}``).
    """
    # Defensive: callers occasionally hand us non-list values (a Mock from a
    # test, or a None when no tool_calls were emitted). Anything other than a
    # non-empty list is a no-op — silently rather than raising, since this is
    # an observability side effect and shouldn't fail the request.
    if not isinstance(tool_calls, list) or not tool_calls:
        return

    base_metadata = dict(metadata or {})

    try:
        langfuse_client = get_client()
    except Exception as exc:  # pragma: no cover - defensive against stub clients
        logger.warning(f"Could not get Langfuse client for tool-call span emission: {exc}")
        return

    for tc in tool_calls:
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        name = fn.get("name") or "unknown"
        raw_args = fn.get("arguments")
        parsed_args: Any
        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                parsed_args = {"_raw_arguments": raw_args}
        else:
            parsed_args = raw_args if raw_args is not None else {}

        span_metadata = {**base_metadata, "call_id": tc.get("id")}

        try:
            kwargs: dict[str, Any] = {
                "name": f"tool:{name}",
                "as_type": "span",
                "input": parsed_args,
                "metadata": span_metadata,
            }
            if trace_context is not None:
                kwargs["trace_context"] = trace_context
            obs = langfuse_client.start_observation(**kwargs)
            with contextlib.suppress(Exception):
                obs.end()
        except Exception as exc:
            logger.warning(f"Failed to record tool-call span for {name!r}: {exc}")


def record_search_tool_spans(
    *,
    artifacts: list[dict] | None,
    trace_context: dict | None,
    metadata: dict | None = None,
) -> None:
    """Emit one Langfuse child span per server-side search/fetch we executed.

    Unlike :func:`record_tool_call_spans` — which traces tool calls the
    *client* is expected to run on its next turn — these are tools **we**
    executed server-side inside the Fireworks external-search loop, so each span
    carries both the ``input`` (query / url) and the ``output`` (the results we
    fed back to the model). They nest under the same parent as the generation
    (detached ``trace_context`` for streaming, current ``@observe`` scope
    otherwise), matching how ``record_tool_call_spans`` nests.

    Crucially these spans carry **no** ``model`` or ``usage_details``: the
    search provider (Exa) is priced per request, not per token, so attaching
    token usage would mis-attribute the upstream model's per-token cost to the
    search. Token cost stays solely on the parent generation; the search's own
    cost (and identity) lives here as plain span metadata, fully separated.

    Args:
        artifacts: The ``external_search`` server artifacts recorded by the
            loop, each ``{"tool", "provider", "input", "results"?, "error"?}``.
            Non-list / empty → no-op.
        trace_context: Detached-mode trace context, or ``None`` to attach to
            the current ``@observe`` scope.
        metadata: Extra metadata merged into every span (alias / provider so a
            "show every web_search run during this turn" filter is one click).
    """
    if not isinstance(artifacts, list) or not artifacts:
        return

    base_metadata = dict(metadata or {})

    try:
        langfuse_client = get_client()
    except Exception as exc:  # pragma: no cover - defensive against stub clients
        logger.warning(f"Could not get Langfuse client for search-tool span emission: {exc}")
        return

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        tool = artifact.get("tool") or "search"
        span_metadata = {**base_metadata, "search_provider": artifact.get("provider")}
        if artifact.get("error"):
            span_metadata["error"] = artifact["error"]

        try:
            kwargs: dict[str, Any] = {
                "name": f"tool:{tool}",
                "as_type": "span",
                "input": artifact.get("input"),
                "output": artifact.get("results") or artifact.get("error"),
                "metadata": span_metadata,
            }
            if trace_context is not None:
                kwargs["trace_context"] = trace_context
            obs = langfuse_client.start_observation(**kwargs)
            with contextlib.suppress(Exception):
                obs.end()
        except Exception as exc:
            logger.warning(f"Failed to record search-tool span for {tool!r}: {exc}")
