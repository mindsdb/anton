"""Deterministic FSM that drives artifact generation (see design spec).

`run(state)` walks the graph nodes. Diamond nodes call
`session._llm.generate_object(...)`; the data-fetch and code-generation nodes
reuse `engine._run_loop`; verification uses `verifiers`.
"""
from __future__ import annotations

import inspect
import re

from anton.core.artifacts.internal_files import (
    API_SPEC_FILENAME,
    PRD_FILENAME,
    TECH_SPEC_FILENAME,
)

from . import engine, prompts
from .discovery import checkpoint as cp
from .discovery.orchestrator import CANCELLED, run_discovery
from .discovery.notes import (  # noqa: F401  (EXEC_* re-exported: tests import them from here)
    EXEC_CODE_MAX,
    EXEC_NOTES_MAX,
    EXEC_OUTPUT_MAX,
    render_exec_notes as _render_exec_notes,
)
from .prompts import HTML_APP_DEFAULT_PRIMARY
from .state import (
    DATA_LOOP_MAX,
    FetchVerdict,
    GenState,
    RequiredData,
    VerifyResult,
)


async def _decide(state: GenState, schema, prompt_pair, node: str) -> object:
    system, user = prompt_pair
    result = await state.session._llm.generate_object(
        schema, system=system, messages=[{"role": "user", "content": user}]
    )
    value = result.model_dump() if hasattr(result, "model_dump") else result
    state.trace_log.llm_call(
        node=node, method="generate_object",
        system=system, messages=[{"role": "user", "content": user}],
        value=value,
    )
    state.trace_log.verdict(node=node, schema=getattr(schema, "__name__", str(schema)), value=value)
    return result


async def _fetch_data_sample(state: GenState) -> str:
    """Run a scratchpad loop that pulls a data sample; return its summary."""
    result = await engine._run_loop(
        session=state.session,
        system=prompts.build_fetch_data_system_prompt(
            state.artifact_path,
            datasource_context=_datasource_context(state.session),
            public_sources=_public_data_sources_skill(state.session),
        ),
        kickoff=prompts.build_fetch_data_kickoff(state),
        artifact_path=state.artifact_path,
        require_files=False,
        node_label="fetch_data_sample",
        trace=state.trace_log,
    )
    if isinstance(result, str):
        # Loop failed — surface as a note so the next is_data_enough sees it.
        return f"(data fetch step reported: {result})"
    summary = result.get("summary") or "(fetch produced no summary)"
    exec_notes = _render_exec_notes(result.get("scratchpad_execs") or [])
    if exec_notes:
        summary += "\n\n" + exec_notes
    return summary


def _needs_data_loop(state: GenState) -> bool:
    """Whether the emergency data loop has to run at all.

    Decided from STATE, never from what happened during this call: on a cold
    start the gathering phase ran in a different process, so any condition
    phrased as "phase A did X" is simply not computable. Both inputs are
    restored from `discovery.json`.

    Two ways in:
      1. gathering never finished — the loop ran out of rounds without
         calling `finish_gathering`, so nothing vouches for the data;
      2. a declared source has nothing executed against it.

    Condition 2 is tracked as an explicit list rather than inferred from
    whether `data_notes` is empty, and that distinction carries the whole
    correction path: a user asking for a chart over a NEW source arrives with
    notes that are far from empty — they hold the previous gathering's cells.
    Matching source names against exec text would be guesswork, so the list
    is maintained where the facts are known.

    On the normal path the list is empty by construction and the loop costs
    nothing.
    """
    if not state.gathering_complete:
        return True
    return bool(state.unverified_sources)


async def _data_phase(state: GenState) -> str | None:
    """The emergency data loop: define -> can-fetch -> fetch, bounded.

    `is_data_enough` used to stand in front of this as an LLM call. It is
    gone: the gathering phase already answered that question by calling
    `finish_gathering`, and asking a second model to re-derive the same
    verdict from a summary of the first one's work cost a call per run and
    added nothing.
    """
    if not _needs_data_loop(state):
        state.record("data_check", "skipped", "gathering already covered the data")
        return None

    last_reasoning = "the gathering phase did not verify the data it declared"
    while state.data_iterations < DATA_LOOP_MAX:
        state.step_started("define_required_data")
        required: RequiredData = await _decide(
            state, RequiredData, prompts.build_required_data_prompt(state),
            "define_required_data",
        )
        required_text = "\n".join(
            f"- {it.name} — from {it.where} ({it.why})" for it in required.items
        ) or required.reasoning
        state.record("define_required_data", "done", required_text)
        last_reasoning = required.reasoning

        state.step_started("is_possible_to_fetch")
        can: FetchVerdict = await _decide(
            state, FetchVerdict,
            prompts.build_can_fetch_prompt(state, required_text),
            "is_possible_to_fetch",
        )
        if not can.possible:
            state.record("is_possible_to_fetch", "no", can.reasoning)
            state.error = f"not enough data: {can.reasoning}"
            return state.error
        state.record("is_possible_to_fetch", "yes", can.reasoning)

        state.step_started("fetch_data_sample")
        notes = await _fetch_data_sample(state)
        state.data_iterations += 1
        state.data_notes = (state.data_notes + "\n\n" + notes).strip()
        # The loop fetched exactly what was declared unverified, so the list
        # clears. Not narrowed item by item: the fetch step works from
        # `define_required_data`'s consolidated list, not per source.
        state.unverified_sources = []
        state.record("fetch_data_sample", "done", notes[:200])

        if not _needs_data_loop(state):
            return None

    state.error = (
        f"not enough data: the data loop did not converge within {DATA_LOOP_MAX} "
        f"iterations. Last assessment: {last_reasoning}"
    )
    return state.error


from . import verifiers
from .state import GEN_LOOP_MAX_RETRIES, GEN_VERIFY_MAX_RETRIES


async def _write_tech_spec(state: GenState) -> str | None:
    state.step_started("make_tech_spec")
    if state.messages:
        # Hot path: this node is the last one that sees the gathering
        # conversation, so it continues it rather than being handed a summary.
        system = state.pipeline_system
        user = prompts.build_tech_spec_instruction(state)
        messages = state.messages
        tools = state.pipeline_tools
    else:
        # Cold start: there is no conversation. The node gets what the
        # pre-merge generator got — brief, PRD, notes, journal — assembled
        # from whatever `discovery.json` restored.
        system, user = prompts.build_tech_spec_prompt(state)
        messages = None
        tools = None
    body, trunc_err = await engine._plan_whole_document(
        state.session, system=system, user=user, node_label="make_tech_spec",
        messages=messages, tools=tools,
        trace=state.trace_log,
        on_retry=lambda: state.step_started("make_tech_spec", attempt=1),
    )
    if trunc_err is not None:
        # Deliberately terminal. Writing the cut spec would put it into
        # `_spec_context` for both generators, which would then build half a
        # system with nothing reporting the loss (ENG-1116).
        state.record("make_tech_spec", "fail", trunc_err)
        state.error = trunc_err
        return state.error
    if not body:
        state.error = "make_tech_spec: model returned an empty specification."
        return state.error
    (state.artifact_path / TECH_SPEC_FILENAME).write_text(body, encoding="utf-8")
    if TECH_SPEC_FILENAME not in state.internal_files:
        state.internal_files.append(TECH_SPEC_FILENAME)
    state.record("make_tech_spec", "done", "wrote spec.md")
    return None


def _spec_context(state: GenState) -> str:
    """Brief + gathered data + tech spec — the shared context handed to
    api-spec and generation nodes (explicit inter-node data flow)."""
    parts = [state.brief.strip()]
    prd = prompts.prd_section(state)
    if prd:
        parts.append(prd)
    if state.data_notes.strip():
        parts.append("## Data\n" + state.data_notes.strip())
    if state.web_notes.strip():
        parts.append(state.web_notes.strip())
    spec_path = state.artifact_path / TECH_SPEC_FILENAME
    if spec_path.is_file():
        parts.append("## Technical specification\n" + spec_path.read_text(encoding="utf-8"))
    journal = state.journal()
    if journal:
        parts.append("## Progress journal (steps completed so far)\n" + journal)
    return "\n\n".join(parts)


async def _make_api_spec(state: GenState) -> str | None:
    state.step_started("make_api_spec")
    stateless = state.artifact_type == "fullstack-stateless-app"
    # Same split as `_write_tech_spec`: continue the shared history when there
    # is one, fall back to the assembled context on a cold start.
    spec_or_err = await engine._generate_api_spec(
        state.session, _spec_context(state), stateless=stateless,
        trace=state.trace_log, node_label="make_api_spec",
        messages=state.messages or None,
        tools=state.pipeline_tools if state.messages else None,
        system_override=state.pipeline_system if state.messages else None,
        on_retry=lambda: state.step_started("make_api_spec", attempt=1),
    )
    if spec_or_err.startswith("Error:"):
        state.error = f"make_api_spec: {spec_or_err}"
        return state.error
    state.api_spec = spec_or_err
    (state.artifact_path / API_SPEC_FILENAME).write_text(spec_or_err, encoding="utf-8")
    if API_SPEC_FILENAME not in state.internal_files:
        state.internal_files.append(API_SPEC_FILENAME)
    state.record("make_api_spec", "done", "wrote openapi.json")
    return None


def _vault(session):
    """The session's vault, or a local one. Mirrors _map_datasources and handle_update_artifact."""
    from anton.core.datasources.data_vault import LocalDataVault

    return getattr(session, "_data_vault", None) or LocalDataVault()


def _list_connections(vault) -> list[dict]:
    """`list_connections()` as a list, or [] when this is not a real vault.

    A returned coroutine is closed explicitly: in tests the session is an
    `AsyncMock` whose `_data_vault` is truthy while `list_connections()` yields a
    coroutine. Swallowing the TypeError is not enough — an abandoned coroutine
    surfaces as a RuntimeWarning at GC time, outside any `catch_warnings` block.
    """
    try:
        raw = vault.list_connections()
    except Exception:  # noqa: BLE001
        return []
    if inspect.iscoroutine(raw):
        raw.close()
        return []
    try:
        return [c for c in raw if isinstance(c, dict)]
    except TypeError:
        return []


def _datasource_context(session) -> str:
    """The `## Connected Data Sources` section (DS_* names, no values), or ""."""
    try:
        vault = _vault(session)
        if not _list_connections(vault):
            return ""
        from anton.utils.datasources import build_datasource_context

        return build_datasource_context(vault) or ""
    except Exception:  # noqa: BLE001
        return ""


_PUBLIC_SOURCES_SKILL = "public-data-sources"


def _public_data_sources_skill(session) -> str:
    """Body of the built-in public-data-sources skill, or "".

    The catalog must not be copied into the prompt — it already lives in
    `builtin_skills/public-data-sources/SKILL.md` and is maintained there. The
    sub-generator has no `recall_skill`, so the body is fed in directly from the
    same `SkillStore` the main agent uses.

    In tests the session is an `AsyncMock`, and `store.load(...)` returns a
    **coroutine**. Type-checking `declarative_md` alone is not enough: an
    abandoned coroutine surfaces as a RuntimeWarning at GC time, in the output of
    an arbitrary test rather than the one that created it. So it is closed
    explicitly — the same treatment as in `_list_connections`. (In `_pads_dict` no
    coroutine appears at all, because the mock is rejected by type before any
    call; here the call is unavoidable.)
    """
    try:
        store = getattr(session, "_skill_store", None)
        skill = store.load(_PUBLIC_SOURCES_SKILL) if store is not None else None
        if inspect.iscoroutine(skill):
            skill.close()
            return ""
        body = getattr(skill, "declarative_md", None)
        return body if isinstance(body, str) else ""
    except Exception:  # noqa: BLE001
        return ""


def _known_connection_hints(session) -> list[str]:
    """`<slug> → DS_<PREFIX>__<FIELD>` for every connection. Never raises.

    Hands over the ready env prefix, not just the slug: turning `<engine>-<name>`
    into `DS_<ENGINE>_<NAME>` sanitises special characters (`prod-db.eu` →
    `PROD_DB_EU`, see `_slug_env_prefix`). Asking the model to derive it itself
    means asking it to repeat exactly the normalisation it already got wrong —
    otherwise it would not have reached this error.
    """
    try:
        from anton.core.artifacts.models import DatasourceRef

        hints = []
        for c in _list_connections(_vault(session)):
            if not (c.get("engine") and c.get("name")):
                continue
            ref = DatasourceRef(engine=c["engine"], name=c["name"])
            hints.append(f"{ref.slug} → {ref.env_prefix}__<FIELD>")
        return sorted(hints)
    except Exception:  # noqa: BLE001
        return []


def _map_datasources(session, ds_keys: list[str]) -> tuple[list, list[str]]:
    """Map DS_* env keys used by backend.py to vault `DatasourceRef`s.

    Returns `(refs, unmapped)`. A key maps when some connection's
    `DS_<ENGINE>_<NAME>` prefix matches `DS_..._<FIELD>`. Mirrors the vault
    lookup already used by `handle_update_artifact` (tool_handlers.py): the
    session attribute is `_data_vault`, falling back to `LocalDataVault()`.
    """
    from anton.core.artifacts.models import DatasourceRef
    from anton.core.datasources.data_vault import LocalDataVault, _slug_env_prefix

    vault = getattr(session, "_data_vault", None) or LocalDataVault()
    conns = [
        (c["engine"], c["name"])
        for c in vault.list_connections()
        if c.get("engine") and c.get("name")
    ]
    refs: list = []
    unmapped: list[str] = []
    for key in ds_keys:
        match = next(
            ((e, n) for (e, n) in conns if key.startswith(_slug_env_prefix(e, n) + "__")),
            None,
        )
        if match is None:
            unmapped.append(key)
        else:
            ref = DatasourceRef(engine=match[0], name=match[1])
            if ref.slug not in {r.slug for r in refs}:
                refs.append(ref)
    return refs, unmapped


async def _declare_datasources(state: GenState, refs: list) -> None:
    """Persist the mapped `DatasourceRef`s into artifact metadata."""
    if not refs:
        return
    from anton.core.tools.tool_handlers import _artifact_store

    store = _artifact_store(state.session)
    if store is not None:
        store.update(state.slug, datasources=refs)
        state.record("declare_datasources", "done", ", ".join(r.slug for r in refs))


async def _gen_verify_backend(state: GenState, extra_context: str = "") -> str | None:
    stateless = state.artifact_type == "fullstack-stateless-app"
    system = prompts.build_backend_system_prompt(
        state.artifact_path,
        stateless=stateless,
        datasource_context=_datasource_context(state.session),
    )
    verdict = None  # guards the terminal message when no attempt ever verified
    last_loop_error: str | None = None  # see the comment in _gen_verify_frontend
    loop_failures = 0  # separate budgets — see _gen_verify_frontend
    verify_failures = 0
    extra = ("\n\n" + extra_context) if extra_context else ""
    for attempt in range(GEN_LOOP_MAX_RETRIES + GEN_VERIFY_MAX_RETRIES + 1):
        state.step_started("generate_backend", attempt=attempt)
        kickoff = prompts.build_backend_kickoff(_spec_context(state), state.api_spec or "{}") + extra
        if stateless:
            injections = [(
                "backend.py",
                "backend.py written. Now write requirements.txt listing EVERY "
                "package imported in backend.py (one per line). Then call finish.",
            )]
        else:
            injections = [
                (
                    "backend.py",
                    "backend.py written. Now write state_manifest.json — the flat "
                    "key-schema object from the DURABLE STATE rules, with every "
                    "Collection name the code uses listed in `collections`.",
                ),
                (
                    "state_manifest.json",
                    "state_manifest.json written. Now write requirements.txt "
                    "listing EVERY package imported in backend.py (one per line, "
                    "but NEVER `anton_state` — it is injected at runtime). Then "
                    "call finish.",
                ),
            ]
        result = await engine._run_loop(
            session=state.session, system=system, kickoff=kickoff,
            artifact_path=state.artifact_path,
            node_label="generate_backend", attempt=attempt, trace=state.trace_log,
            step_injections=injections,
        )
        if isinstance(result, str):
            state.record("generate_backend", "error", result)
            last_loop_error = result
            loop_failures += 1
            if loop_failures > GEN_LOOP_MAX_RETRIES:
                break
            extra = f"\n\n## Previous attempt failed\n{result}\nFix it and try again."
            continue
        for f in result["files_written"]:
            if f not in state.files_written:
                state.files_written.append(f)
        state.record("generate_backend", "done", result.get("summary", ""))

        state.step_started("verify_backend", attempt=attempt)
        verdict, ds_keys = await verifiers.verify_backend(
            scratchpad_pool=state.session._scratchpads,
            slug=state.slug, artifact_path=state.artifact_path,
            artifact_type=state.artifact_type,
        )
        state.trace_log.verifier(
            node="verify_backend", ok=verdict.ok,
            errors=list(verdict.errors), warnings=list(verdict.warnings),
        )
        if verdict.ok:
            # Spec: DS_* keys with no matching vault connection are errors.
            refs, unmapped = _map_datasources(state.session, ds_keys)
            if unmapped:
                known = _known_connection_hints(state.session)
                msg = (
                    "backend reads DS_* env keys with no matching vault "
                    "connection: " + ", ".join(unmapped)
                    + ". Available connections and their env-var namespaces: "
                    + ("; ".join(known) or "(none)")
                    + ". Use one of those namespaces verbatim."
                )
                state.record("verify_backend", "fail", msg)
                verdict.errors.append(msg)
                verify_failures += 1
                if verify_failures > GEN_VERIFY_MAX_RETRIES:
                    break
                extra = "\n\n## Verification failed — fix these\n- " + msg
                continue
            state.record("verify_backend", "ok", "; ".join(verdict.warnings))
            await _declare_datasources(state, refs)
            return None
        state.record("verify_backend", "fail", "; ".join(verdict.errors))
        verify_failures += 1
        if verify_failures > GEN_VERIFY_MAX_RETRIES:
            break
        extra = (
            "\n\n## Verification failed — fix these\n"
            + "\n".join(f"- {e}" for e in verdict.errors)
            + ("\nWarnings:\n" + "\n".join(f"- {w}" for w in verdict.warnings) if verdict.warnings else "")
        )
    # See the twin comment in _gen_verify_frontend.
    if verify_failures > GEN_VERIFY_MAX_RETRIES and verdict is not None:
        state.error = (
            f"Backend verification failed after {verify_failures} attempt(s): "
            + "; ".join(verdict.errors)
        )
    else:
        state.error = "Backend generation failed: " + (
            last_loop_error or "generation did not produce a verifiable backend"
        )
    return state.error


def _read_frontend_html(state: GenState, written: list[str]) -> str | None:
    if state.is_fullstack:
        entry = state.artifact_path / "static" / "index.html"
        return entry.read_text(encoding="utf-8") if entry.is_file() else None
    # html-app: pick ONLY among what this run actually wrote. `primary` is the
    # expectation, `written` is the fact, and the fact is what must be verified:
    # otherwise, with no primary set, candidate #1 becomes the default
    # `dashboard.html`, and a leftover file of that name from a previous
    # generation would shadow the fresh `report.html`. Within what was written,
    # primary takes priority — for the case where the loop wrote several .html.
    written_html = [rel for rel in written if rel.endswith(".html")]
    target = state.primary or HTML_APP_DEFAULT_PRIMARY
    for rel in [r for r in written_html if r == target] + written_html:
        p = state.artifact_path / rel
        if p.is_file():
            return p.read_text(encoding="utf-8")
    return None


async def _gen_verify_frontend(state: GenState) -> str | None:
    if state.is_fullstack:
        system = prompts.build_frontend_system_prompt(state.artifact_path)
    else:
        system = prompts.build_subagent_system_prompt(
            state.artifact_type, state.artifact_path, primary=state.primary
        )
    verdict = None  # guards the terminal message when no attempt ever verified
    # The loop's failure reason (round budget, no tool calls) is kept separately:
    # it cannot ride on VerifyResult — a variable argument breaks the contract
    # lock's AST walk (test_no_unresolvable_rule_literals).
    last_loop_error: str | None = None
    # Separate budgets: a loop failure must not consume the retry reserved for
    # fixing verifier findings (and vice versa). The range bound is the sum —
    # each iteration spends exactly one of the two budgets, and the inner
    # checks break as soon as either is exhausted.
    loop_failures = 0
    verify_failures = 0
    extra = ""
    for attempt in range(GEN_LOOP_MAX_RETRIES + GEN_VERIFY_MAX_RETRIES + 1):
        state.step_started("generate_frontend", attempt=attempt)
        if attempt > 0:
            # With append, a retry would extend the truncated remains of the
            # previous attempt. Delete deterministically: the "first chunk is
            # always mode=\"w\"" rule is in the prompt, but cannot be relied on.
            # Only for attempt > 0: before the first attempt the file may well be
            # a working previous version of the artifact.
            entry = (
                state.artifact_path / "static" / "index.html"
                if state.is_fullstack
                else state.artifact_path / (state.primary or HTML_APP_DEFAULT_PRIMARY)
            )
            entry.unlink(missing_ok=True)
        if state.is_fullstack:
            kickoff = prompts.build_frontend_kickoff(_spec_context(state), state.api_spec or "{}") + extra
        else:
            kickoff = prompts.build_user_kickoff(_spec_context(state)) + extra
        result = await engine._run_loop(
            session=state.session, system=system, kickoff=kickoff,
            artifact_path=state.artifact_path,
            node_label="generate_frontend", attempt=attempt, trace=state.trace_log,
        )
        if isinstance(result, str):
            state.record("generate_frontend", "error", result)
            last_loop_error = result
            loop_failures += 1
            if loop_failures > GEN_LOOP_MAX_RETRIES:
                break
            extra = f"\n\n## Previous attempt failed\n{result}\nFix it and try again."
            continue
        for f in result["files_written"]:
            if f not in state.files_written:
                state.files_written.append(f)
        state.record("generate_frontend", "done", result.get("summary", ""))

        state.step_started("verify_frontend", attempt=attempt)
        html = _read_frontend_html(state, result["files_written"])
        if html is None:
            # One channel: this message rides on VerifyResult like every other
            # check — otherwise the contract lock cannot see it and the terminal
            # error loses its cause (verdict used to stay None).
            verdict = VerifyResult(errors=[
                "No HTML entry file was written. Write static/index.html "
                "(or the html-app page)."
            ])
        else:
            verdict = verifiers.verify_frontend(html, is_fullstack=state.is_fullstack)
        state.trace_log.verifier(
            node="verify_frontend", ok=verdict.ok,
            errors=list(verdict.errors), warnings=list(verdict.warnings),
        )
        if verdict.ok:
            # The model may have named the file differently — bring metadata in
            # line with the fact, or the renderer opens the wrong file. Only after
            # successful verification: if both attempts fail, the attempt-1 cleanup
            # deletes the file and a primary written earlier would dangle.
            if not state.is_fullstack:
                actual = next(
                    (f for f in result["files_written"] if f.endswith(".html")), None
                )
                if actual and actual != (state.primary or HTML_APP_DEFAULT_PRIMARY):
                    from anton.core.tools.tool_handlers import _artifact_store

                    store = _artifact_store(state.session)
                    if store is not None:
                        store.update(state.slug, primary=actual)
                    state.primary = actual
            state.record("verify_frontend", "ok", "; ".join(verdict.warnings))
            return None
        state.record("verify_frontend", "fail", "; ".join(verdict.errors))
        verify_failures += 1
        if verify_failures > GEN_VERIFY_MAX_RETRIES:
            break
        extra = (
            "\n\n## Verification failed — fix these\n"
            + "\n".join(f"- {e}" for e in verdict.errors)
            + ("\nWarnings:\n" + "\n".join(f"- {w}" for w in verdict.warnings) if verdict.warnings else "")
        )
    # Name the cause that actually exhausted its budget: "verification failed"
    # only when the verifier's own retry budget ran out. The old single wording
    # claimed a verification retry that, on the loop-failure path, never
    # happened.
    if verify_failures > GEN_VERIFY_MAX_RETRIES and verdict is not None:
        state.error = (
            f"Frontend verification failed after {verify_failures} attempt(s): "
            + "; ".join(verdict.errors)
        )
    else:
        state.error = "Frontend generation failed: " + (
            last_loop_error or "generation did not produce a verifiable frontend"
        )
    return state.error


# ---------------------------------------------------------------------------
# run_app / verify_fullstack + run() assembly
# ---------------------------------------------------------------------------

import asyncio
import json
import urllib.error
import urllib.request

from .state import RUNAPP_MAX_RETRIES


async def _launch_backend(**kwargs):
    """Indirection so tests can monkeypatch the real launcher."""
    from anton.core.artifacts.backend_launcher import launch_artifact_backend

    return await launch_artifact_backend(**kwargs)


def _safe_get_paths(api_spec: str | None) -> list[str]:
    """GET paths from openapi.json with no `{param}` and no required params."""
    if not api_spec:
        return []
    try:
        spec = json.loads(api_spec)
    except json.JSONDecodeError:
        return []
    out: list[str] = []
    for path, ops in (spec.get("paths") or {}).items():
        if "{" in path:
            continue
        get = (ops or {}).get("get")
        if not isinstance(get, dict):
            continue
        params = get.get("parameters") or []
        if any(p.get("required") for p in params if isinstance(p, dict)):
            continue
        out.append(path)
    return out


async def _probe_app(state: GenState, port: int) -> str | None:
    """Health check + safe GET routes. Returns an error string or None."""
    base = f"http://127.0.0.1:{port}"

    def _get(url: str) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return resp.status, resp.read(4096).decode(errors="replace")
        except urllib.error.HTTPError as e:
            return e.code, ""
        except Exception as e:  # noqa: BLE001
            return 0, str(e)

    status, _ = await asyncio.to_thread(_get, base + "/api/health")
    if status != 200:
        return f"health check GET /api/health returned {status} (expected 200)."
    for path in _safe_get_paths(state.api_spec):
        status, _ = await asyncio.to_thread(_get, base + path)
        if status == 0 or status >= 500:
            return f"endpoint GET {path} failed with status {status}."
    return None


async def _tail_log(state: GenState, limit: int = 2000) -> str:
    log = state.artifact_path / "backend.log"
    if not log.is_file():
        return ""
    text = log.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


async def _run_and_verify_app(state: GenState) -> str | None:
    # `_tracked_backends` is initialised on ChatSession (session.py), but use the
    # same defensive getattr-or-create as handle_launch_backend for parity with
    # non-standard session objects.
    tracked = getattr(state.session, "_tracked_backends", None)
    if not isinstance(tracked, dict):
        tracked = {}
        state.session._tracked_backends = tracked
    problem = ""
    for attempt in range(RUNAPP_MAX_RETRIES + 1):
        state.step_started("run_app", attempt=attempt)
        launch = await _launch_backend(
            slug=state.slug,
            artifact_folder=state.artifact_path,
            scratchpad_pool=state.session._scratchpads,
            tracked_backends=tracked,
            health_path="/api/health",
        )
        if isinstance(launch, str):  # launcher error (install / readiness)
            state.record("run_app", "fail", launch)
            problem = launch
        else:
            port = launch["port"]
            from anton.core.tools.tool_handlers import _artifact_store

            store = _artifact_store(state.session)
            if store is not None:
                store.update(state.slug, port=port)
            state.record("run_app", "done", f"port {port}")
            state.step_started("verify_fullstack", attempt=attempt)
            probe_err = await _probe_app(state, port)
            if probe_err is None:
                state.record("verify_fullstack", "ok", f"port {port}")
                return None
            state.record("verify_fullstack", "fail", probe_err)
            problem = probe_err

        if attempt >= RUNAPP_MAX_RETRIES:
            break
        # One backend-loop retry with the failure + log tail passed as kickoff
        # context (NOT persisted into data_notes, so later steps stay clean).
        tail = await _tail_log(state)
        regen_err = await _gen_verify_backend(
            state,
            extra_context=(
                f"## Previous launch failure\n{problem}\n\n"
                f"## backend.log tail\n{tail}"
            ),
        )
        if regen_err is not None:
            state.error = regen_err
            return regen_err

    state.error = (
        f"Application failed to launch: {problem}\n\n"
        f"--- backend.log tail ---\n{await _tail_log(state)}"
    )
    return state.error


def _result_shell(state: GenState) -> dict:
    return {
        "files_written": state.files_written,
        # Generation input, not output: it physically sits in the artifact folder
        # but is not an artifact for the user (see the design spec, 3.6).
        "internal_files": state.internal_files,
        "summary": "; ".join(f"{s.node}:{s.outcome}" for s in state.trace),
        "trace": [{"node": s.node, "outcome": s.outcome, "detail": s.detail} for s in state.trace],
    }


def _save_checkpoint(state: GenState, stage: str) -> None:
    """Write `discovery.json` at a stage transition.

    Every field phase E reads goes in, because the boundary has to survive
    the process: `data_notes` and `web_notes` live only in memory otherwise,
    and the pad-inspection step that used to rebuild them on a cold start is
    gone.
    """
    cp.save(
        state.artifact_path,
        cp.DiscoveryCheckpoint(
            request_fingerprint=cp.request_fingerprint(state.user_request),
            call_fingerprint=cp.call_fingerprint(
                state.agent_understanding, state.known_data, state.user_preferences
            ),
            pipeline_stage=stage,
            artifact_type=state.final_artifact_type or state.artifact_type,
            gathering_complete=state.gathering_complete,
            declared_sources=list(state.declared_sources),
            unverified_sources=list(state.unverified_sources),
            brief_markdown=state.brief,
            data_notes=state.data_notes,
            web_notes=state.web_notes,
        ),
    )


def _invalidate_specs(state: GenState) -> None:
    """Drop specs written for a previous PRD.

    Runs after phase C succeeds, on EVERY new `prd.md` — not only when the
    request changed. The most common correction keeps the same request and
    only fixes the brief, and a spec built from the un-corrected requirements
    is exactly what a later cold start would pick up.

    Deliberately not run before phase A: a cancelled or early-failing run
    must not destroy the previous, working state (same principle as
    invariant 9 for the entry file).
    """
    for name in (TECH_SPEC_FILENAME, API_SPEC_FILENAME):
        path = state.artifact_path / name
        if path.is_file():
            path.unlink()
        if name in state.internal_files:
            state.internal_files.remove(name)
    state.api_spec = None


def _finish(state: GenState) -> dict:
    _save_checkpoint(state, cp.STAGE_GENERATED)
    return {"status": "generated", **_result_shell(state)}


def _cancelled(state: GenState) -> dict:
    """Nothing written, nothing deleted.

    The folder may already hold a previous run's work; cancelling means "do
    not rebuild", not "erase what is there".
    """
    return {
        "status": "cancelled",
        "reason": "user declined the brief",
        "qa_log": state.qa_log_markdown(),
    }


def _needs_confirmation(state: GenState) -> dict:
    return {
        "status": "needs_confirmation",
        "brief_summary": state.brief,
        "prd_path": str(state.artifact_path / PRD_FILENAME),
        "artifact_type": state.final_artifact_type or state.artifact_type,
        "qa_log": state.qa_log_markdown(),
    }


def _stopped_over_budget(state: GenState, detail: str) -> dict:
    # `brief_summary` travels with every budget stop: the run can end before
    # the user has seen a brief at all, and then this is the only way to show
    # them one.
    return {
        "status": "stopped_over_budget",
        "detail": detail,
        "brief_summary": state.brief,
        **_result_shell(state),
    }


async def run(state: GenState, *, entry: str = cp.ENTRY_FULL) -> dict | str:
    """Walk the whole pipeline from wherever this call is entitled to start."""
    if entry in (cp.ENTRY_FULL, cp.ENTRY_CONFIRM, cp.ENTRY_NEW_ITERATION):
        stage = await run_discovery(state, entry=entry)
        if stage == CANCELLED:
            return _cancelled(state)
        if stage == cp.STAGE_AWAITING_CONFIRMATION:
            _save_checkpoint(state, cp.STAGE_AWAITING_CONFIRMATION)
            # Same stage on disk, two different things to tell the user.
            # "Show the brief and get agreement" and "we stopped because this
            # got expensive — continue?" are not the same question, and the
            # second has to mention the budget or the user never learns why
            # the work stopped short.
            if state.winding_down():
                return _stopped_over_budget(
                    state, "budget reached while agreeing the brief"
                )
            return _needs_confirmation(state)
        _invalidate_specs(state)
        _save_checkpoint(state, cp.STAGE_PRD_WRITTEN)

    if entry != cp.ENTRY_GENERATE:
        # Before the spec phase, not only before generation: `make_tech_spec`
        # is the single most expensive call in the pipeline (it carries the
        # whole shared history), and by here `prd.md` plus a `PRD_WRITTEN`
        # checkpoint are already on disk, so stopping is cheap to resume.
        if state.winding_down():
            return _stopped_over_budget(
                state, "budget reached before the specification"
            )
        err = await _data_phase(state)
        if err is not None:
            return err
        err = await _write_tech_spec(state)
        if err is not None:
            return err
        if state.is_fullstack:
            err = await _make_api_spec(state)
            if err is not None:
                return err
        _save_checkpoint(state, cp.STAGE_SPEC_WRITTEN)
        # The shared history has done its job: `spec.md` is the recoding
        # point now. Dropping it here, in one place, is what keeps the
        # generation rounds as small as they are today.
        state.messages = []

    if state.winding_down():
        return _stopped_over_budget(
            state, "budget reached before file generation started"
        )

    if not state.is_fullstack:
        err = await _gen_verify_frontend(state)
        if err is not None:
            return err
        return _finish(state)

    back_err, front_err = await asyncio.gather(
        _gen_verify_backend(state), _gen_verify_frontend(state)
    )
    if back_err is not None:
        return back_err
    if front_err is not None:
        return front_err
    # Over the ceiling the backend is not launched at all — no process
    # started, no port written into metadata. Both loops closed, so the files
    # are there; what is missing is only the launch, and that is what the
    # continuation does.
    if state.winding_down():
        return _stopped_over_budget(
            state, "budget reached before launching the backend"
        )
    err = await _run_and_verify_app(state)
    if err is not None:
        return err
    return _finish(state)
