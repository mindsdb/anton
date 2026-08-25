"""Deterministic FSM that drives artifact generation (see design spec).

`run(state)` walks the graph nodes. Diamond nodes call
`session._llm.generate_object(...)`; the data-fetch and code-generation nodes
reuse `engine._run_loop`; verification uses `verifiers`.
"""
from __future__ import annotations

import inspect
import re

from . import engine, prompts
from .prompts import HTML_APP_DEFAULT_PRIMARY
from .state import (
    DATA_LOOP_MAX,
    DataVerdict,
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


# Caps for the exec-code record appended to data_notes: per-cell code, per-cell
# output snippet, and the whole section. Oldest cells are dropped first — the
# most recent ones are the ones that worked.
EXEC_CODE_MAX = 2000
EXEC_OUTPUT_MAX = 300
EXEC_NOTES_MAX = 8000


def _render_exec_notes(
    execs: list[dict], *, header: str = "### Code executed while fetching"
) -> str:
    """Deterministic record of the Python the fetch step ran.

    Appended to data_notes so later steps (tech spec, backend generation) see
    the exact working data-access code instead of relying on the model's
    `finish` summary to mention it.
    """
    blocks: list[str] = []
    for e in execs:
        code = (e.get("code") or "").strip()
        if not code:
            continue
        if len(code) > EXEC_CODE_MAX:
            code = code[:EXEC_CODE_MAX] + "\n# … truncated …"
        out = " ".join((e.get("output") or "").split())
        if len(out) > EXEC_OUTPUT_MAX:
            out = out[:EXEC_OUTPUT_MAX] + " …"
        block = f"Scratchpad `{e.get('name')}`:\n```python\n{code}\n```"
        if out:
            block += f"\nOutput: {out}"
        blocks.append(block)
    dropped = 0
    while blocks and sum(len(b) for b in blocks) > EXEC_NOTES_MAX:
        blocks.pop(0)
        dropped += 1
    if not blocks:
        return ""
    if dropped:
        header += f" (first {dropped} cell(s) omitted for size)"
    return header + "\n" + "\n\n".join(blocks)


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


def _pads_dict(session) -> dict:
    """Live scratchpad runtimes, or {} when this is not a real manager.

    The `isinstance(..., dict)` check does double duty. The real
    `ScratchpadManager.pads` is a genuine dict (`backends/manager.py:33` returns
    `self._pads`), while in tests the session is an `AsyncMock` whose `pads` is a
    mock. Rejecting it by type means we never call `.keys()` on it, so we never
    create an un-awaited coroutine — one would surface as a RuntimeWarning at GC
    time and pollute the output of every test using a mock session. See
    `_list_connections`, where such a coroutine has to be closed by hand; here it
    simply never comes into existence.
    """
    try:
        pads = session._scratchpads.pads
    except Exception:  # noqa: BLE001
        return {}
    return pads if isinstance(pads, dict) else {}


def _pads_named_in_brief(session, brief: str) -> list[str]:
    """Names of live scratchpads mentioned in `brief` — the caller passes the
    brief and the PRD joined, since either may be where a pad is named.

    Matching runs from the live pads to the brief text, not the other way round.
    Persistent pads cannot be enumerated (`Cell` does not store its pad name,
    `base.py:15-26`), and on this branch `get_or_create` does NOT materialise a
    pad from replayed cells: `ChatSessionConfig.cells` is declared as `None`
    (`session.py:261`) and no production caller sets it. So we work only with
    what is already live in the process — and a false match costs a few tokens in
    the notes rather than a venv provisioning.

    Word boundaries are mandatory: a pad named `dash` would otherwise match the
    word "dashboard", which a dashboard brief is guaranteed to contain.
    """
    matched = [
        name
        for name in _pads_dict(session)
        if isinstance(name, str)
        and name.strip()
        and re.search(rf"\b{re.escape(name)}\b", brief)
    ]
    return sorted(matched)


def _live_pad_names(session) -> list[str]:
    """Live pad names — used only for the skip record. Never raises."""
    return sorted(str(k) for k in _pads_dict(session))


def _inspect_named_scratchpads(state: GenState) -> None:
    """Read the cells of pads named in the brief or the PRD into `data_notes`.

    Reads `pad.cells` directly (`base.py:52`) rather than going through `dump`:
    that returns a ready-made markdown string from `render_notebook()`, which
    `_render_exec_notes` cannot consume, and it carries every cell's full code
    with no truncation. `pad.cells` maps into the required shape as-is, so the
    existing caps come for free.

    The step is synchronous and side-effect free: no `await`, no `get_or_create`.
    """
    state.step_started("inspect_scratchpads")
    # The PRD is searched alongside the brief: `generate_prd` is required to
    # cite the scratchpad and cell behind every data source it describes, so
    # on the normal path the PRD is where a pad name actually appears — the
    # brief may never mention one. Matching only the brief would re-fetch data
    # the PRD already points at.
    matched = _pads_named_in_brief(
        state.session, "\n\n".join(p for p in (state.brief, state.prd) if p)
    )
    if not matched:
        live = _live_pad_names(state.session)
        detail = (
            "no scratchpad named in the brief matched the live ones: "
            + ", ".join(live)
            if live
            else "no live scratchpads in this session"
        )
        state.record("inspect_scratchpads", "skipped", detail)
        return

    pads = _pads_dict(state.session)
    execs: list[dict] = []
    for name in matched:
        for cell in getattr(pads[name], "cells", None) or []:
            code = getattr(cell, "code", "") or ""
            if not code.strip():
                continue
            execs.append(
                {
                    "name": name,
                    "code": code,
                    # stdout or the error text: a failed cell would otherwise
                    # land in the notes as code with no result — indistinguishable
                    # from a successful one that printed nothing. "What was tried
                    # and what did not work" is exactly what the inspection needs.
                    "output": getattr(cell, "stdout", "") or getattr(cell, "error", "") or "",
                }
            )

    # Own header: the default "### Code executed while fetching" would be a lie
    # here — nothing was fetched, these are the main agent's cells. It would also
    # contradict the `is_data_enough` instruction, which teaches the node to count
    # this data as available "regardless of who obtained it".
    notes = _render_exec_notes(
        execs,
        header="### Cells the main agent already ran in: " + ", ".join(matched),
    )
    if not notes:
        state.record(
            "inspect_scratchpads", "empty",
            "pads named in the brief have no cells: " + ", ".join(matched),
        )
        return

    state.data_notes = (state.data_notes + "\n\n" + notes).strip()
    state.record(
        "inspect_scratchpads", "done",
        f"{len(execs)} cell(s) from {', '.join(matched)}",
    )


async def _data_phase(state: GenState) -> str | None:
    """is_data_enough ↔ define_required_data → is_possible_to_fetch → fetch.

    Before the first `is_data_enough`, inspect the pads named in the brief: what
    the outer agent already gathered does not need gathering again. The phase
    itself is not trimmed — only its input changes.
    """
    _inspect_named_scratchpads(state)
    last_reasoning = ""
    for _ in range(DATA_LOOP_MAX + 1):
        state.step_started("is_data_enough")
        verdict: DataVerdict = await _decide(
            state, DataVerdict, prompts.build_data_enough_prompt(state), "is_data_enough"
        )
        if verdict.enough:
            state.record("is_data_enough", "yes", verdict.reasoning)
            return None
        state.record("is_data_enough", "no", verdict.reasoning)

        if state.data_iterations >= DATA_LOOP_MAX:
            break

        state.step_started("define_required_data")
        required: RequiredData = await _decide(
            state, RequiredData, prompts.build_required_data_prompt(state), "define_required_data"
        )
        required_text = "\n".join(
            f"- {it.name} — from {it.where} ({it.why})" for it in required.items
        ) or required.reasoning
        state.record("define_required_data", "done", required_text)
        last_reasoning = required.reasoning

        state.step_started("is_possible_to_fetch")
        can: FetchVerdict = await _decide(
            state, FetchVerdict, prompts.build_can_fetch_prompt(state, required_text), "is_possible_to_fetch"
        )
        if not can.possible:
            state.record("is_possible_to_fetch", "no", can.reasoning)
            state.error = f"not enough data: {can.reasoning}"
            return state.error
        state.record("is_possible_to_fetch", "yes", can.reasoning)
        last_reasoning = can.reasoning

        state.step_started("fetch_data_sample")
        notes = await _fetch_data_sample(state)
        state.data_iterations += 1
        state.data_notes = (state.data_notes + "\n\n" + notes).strip()
        state.record("fetch_data_sample", "done", notes[:200])

    state.error = (
        f"not enough data: the data loop did not converge within {DATA_LOOP_MAX} "
        f"iterations. Last assessment: {last_reasoning}"
    )
    return state.error


# ---------------------------------------------------------------------------
# Spec + generation nodes
# ---------------------------------------------------------------------------

from . import verifiers
from .state import GEN_VERIFY_MAX_RETRIES


async def _write_tech_spec(state: GenState) -> str | None:
    state.step_started("make_tech_spec")
    system, user = prompts.build_tech_spec_prompt(state)
    resp = await state.session._llm.plan(
        system=system, messages=[{"role": "user", "content": user}]
    )
    state.trace_log.llm_call(
        node="make_tech_spec", method="plan",
        system=system, messages=[{"role": "user", "content": user}],
        response=resp,
    )
    body = (getattr(resp, "content", "") or "").strip()
    if not body:
        state.error = "make_tech_spec: model returned an empty specification."
        return state.error
    (state.artifact_path / "spec.md").write_text(body, encoding="utf-8")
    if "spec.md" not in state.internal_files:
        state.internal_files.append("spec.md")
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
    spec_path = state.artifact_path / "spec.md"
    if spec_path.is_file():
        parts.append("## Technical specification\n" + spec_path.read_text(encoding="utf-8"))
    journal = state.journal()
    if journal:
        parts.append("## Progress journal (steps completed so far)\n" + journal)
    return "\n\n".join(parts)


async def _make_api_spec(state: GenState) -> str | None:
    state.step_started("make_api_spec")
    stateless = state.artifact_type == "fullstack-stateless-app"
    spec_or_err = await engine._generate_api_spec(
        state.session, _spec_context(state), stateless=stateless,
        trace=state.trace_log, node_label="make_api_spec",
    )
    if spec_or_err.startswith("Error:"):
        state.error = f"make_api_spec: {spec_or_err}"
        return state.error
    state.api_spec = spec_or_err
    (state.artifact_path / "openapi.json").write_text(spec_or_err, encoding="utf-8")
    if "openapi.json" not in state.internal_files:
        state.internal_files.append("openapi.json")
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
    extra = ("\n\n" + extra_context) if extra_context else ""
    for attempt in range(GEN_VERIFY_MAX_RETRIES + 1):
        state.step_started("generate_backend", attempt=attempt)
        kickoff = prompts.build_backend_kickoff(_spec_context(state), state.api_spec or "{}") + extra
        result = await engine._run_loop(
            session=state.session, system=system, kickoff=kickoff,
            artifact_path=state.artifact_path,
            node_label="generate_backend", attempt=attempt, trace=state.trace_log,
            step_injections=[(
                "backend.py",
                "backend.py written. Now write requirements.txt listing EVERY "
                "package imported in backend.py (one per line). Then call finish.",
            )],
        )
        if isinstance(result, str):
            extra = f"\n\n## Previous attempt failed\n{result}\nFix it and try again."
            state.record("generate_backend", "error", result)
            last_loop_error = result
            continue
        for f in result["files_written"]:
            if f not in state.files_written:
                state.files_written.append(f)
        state.record("generate_backend", "done", result.get("summary", ""))

        state.step_started("verify_backend", attempt=attempt)
        verdict, ds_keys = await verifiers.verify_backend(
            scratchpad_pool=state.session._scratchpads,
            slug=state.slug, artifact_path=state.artifact_path,
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
                extra = "\n\n## Verification failed — fix these\n- " + msg
                continue
            state.record("verify_backend", "ok", "; ".join(verdict.warnings))
            await _declare_datasources(state, refs)
            return None
        state.record("verify_backend", "fail", "; ".join(verdict.errors))
        extra = (
            "\n\n## Verification failed — fix these\n"
            + "\n".join(f"- {e}" for e in verdict.errors)
            + ("\nWarnings:\n" + "\n".join(f"- {w}" for w in verdict.warnings) if verdict.warnings else "")
        )
    detail = (
        "; ".join(verdict.errors)
        if verdict is not None
        else (last_loop_error or "generation did not produce a verifiable backend")
    )
    state.error = "Backend verification failed after retry: " + detail
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
    extra = ""
    for attempt in range(GEN_VERIFY_MAX_RETRIES + 1):
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
            extra = f"\n\n## Previous attempt failed\n{result}\nFix it and try again."
            state.record("generate_frontend", "error", result)
            last_loop_error = result
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
        extra = (
            "\n\n## Verification failed — fix these\n"
            + "\n".join(f"- {e}" for e in verdict.errors)
            + ("\nWarnings:\n" + "\n".join(f"- {w}" for w in verdict.warnings) if verdict.warnings else "")
        )
    detail = (
        "; ".join(verdict.errors)
        if verdict is not None
        else (last_loop_error or "generation did not produce a verifiable frontend")
    )
    state.error = "Frontend verification failed after retry: " + detail
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


def _success(state: GenState) -> dict:
    return {
        "files_written": state.files_written,
        # Generation input, not output: it physically sits in the artifact folder
        # but is not an artifact for the user (see the design spec, 3.6).
        "internal_files": state.internal_files,
        "summary": "; ".join(f"{s.node}:{s.outcome}" for s in state.trace),
        "trace": [{"node": s.node, "outcome": s.outcome, "detail": s.detail} for s in state.trace],
    }


async def run(state: GenState) -> dict | str:
    # is_data_enough loop
    err = await _data_phase(state)
    if err is not None:
        return err
    # make_tech_spec
    err = await _write_tech_spec(state)
    if err is not None:
        return err
    # is_fullstack (deterministic)
    if not state.is_fullstack:
        err = await _gen_verify_frontend(state)
        if err is not None:
            return err
        return _success(state)
    # fullstack: make_api_spec → parallel backend/frontend gen+verify
    err = await _make_api_spec(state)
    if err is not None:
        return err
    back_err, front_err = await asyncio.gather(
        _gen_verify_backend(state), _gen_verify_frontend(state)
    )
    if back_err is not None:
        return back_err
    if front_err is not None:
        return front_err
    # run_app → verify_fullstack (with one backend-loop retry)
    err = await _run_and_verify_app(state)
    if err is not None:
        return err
    return _success(state)
