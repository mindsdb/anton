"""Opt-in direct generation for self-contained pages whose source data is already observed.

This path shares the normal writer and verifiers. It removes document-planning
round trips, not data or browser checks. Requests needing binary assets, a server,
or user confirmation continue through the existing pipeline. Validated data-file
attachments use the same discovery and structured handoff as other sources.
"""
from __future__ import annotations

import asyncio
import copy
import inspect
import json
import os
import re

from anton.utils.datasources import scrub_output
from anton.utils.nofollow import read_bytes_nofollow

from . import sub_tools, verifiers
from .engine import _drain_stream
from .orchestrator import _finish, _stopped_over_budget

MAX_SOURCE_CHARS = 96000
MAX_PAGE_CHARS = 18000
SOURCE_TOOLS = {"scratchpad", "read_text_file", "web_fetch", "web_search", "recall_skill"}


def _connection_procedures(session):
    """Hand matching saved procedures to discovery even when the caller didn't recall them."""
    from .orchestrator import _list_connections, _vault

    store = getattr(session, "_skill_store", None)
    if store is None:
        return ""
    procedures = []
    for connection in _list_connections(_vault(session))[:16]:
        label = connection.get("name")
        if not isinstance(label, str):
            continue
        try:
            skill = store.load(label)
            if inspect.iscoroutine(skill):
                skill.close()
                continue
            body = getattr(skill, "declarative_md", None)
            if isinstance(body, str) and body:
                procedures.append(f"Connected source procedure {label}:\n{body[:24000]}")
        except Exception:
            continue  # Ordinary source discovery still works without a matching skill.
    return "\n\n".join(procedures)[:MAX_SOURCE_CHARS]


def _integer_setting(name, default, minimum, maximum):
    try:
        value = int(os.environ.get(name, default))
    except ValueError:
        return default
    return min(maximum, max(minimum, value))


def _text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_text(x) for x in value)
    if isinstance(value, dict) and value.get("type") == "text":
        return str(value.get("text", ""))
    return ""


def source_evidence(session, messages=(), *, tools=SOURCE_TOOLS):
    """Pass observed tool output across the delegation boundary without summarizing numbers."""
    history = list(getattr(session, "history", [])) + list(messages)
    names = {}
    results = []
    for message in history if isinstance(history, list) else []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                names[block.get("id")] = block.get("name")
            elif block.get("type") == "tool_result" and not block.get("is_error"):
                name = names.get(block.get("tool_use_id"))
                if name in tools:
                    body = scrub_output(_text(block.get("content")))
                    if body:
                        results.append(f"Observed {name} result:\n{body}")
    kept, size, omitted = [], 0, 0
    for result in reversed(results):
        if size + len(result) <= MAX_SOURCE_CHARS:
            kept.append(result)
            size += len(result)
        else:
            omitted += 1
    evidence = "\n\n".join(reversed(kept))
    if omitted:
        evidence += f"\n{omitted} older/oversize source results omitted; do not assume they are complete."
    return evidence


def document(text):
    text = text.strip()
    fence = re.fullmatch(r"```(?:html)?\s*\n(.*?)\n```", text, re.S | re.I)
    if fence:
        text = fence.group(1).strip()
    if not re.match(r"(?:<!doctype\s+html[^>]*>\s*)?<html(?:\s|>)", text, re.I):
        raise ValueError("Return a complete HTML document, without commentary or escaped source.")
    if not re.search(r"</html>\s*$", text, re.I):
        raise ValueError("The HTML document must be complete through </html>.")
    return text


async def prepare(state):
    """Keep the existing source-access tools and let discovery settle the type."""
    from .discovery.engine import run_gathering_loop

    # Recalled source documentation lives in the calling conversation. Without
    # this handoff the delegate gets credential names but loses the API contract.
    state._pipeline_system = state.pipeline_system + "\n\nPreviously observed source documentation and results:\n" + source_evidence(state.session)
    if os.environ.get("ANTON_STATIC_CONNECTED_PROCEDURES") == "on":
        state._pipeline_system += "\n\n" + _connection_procedures(state.session)
    if (os.environ.get("ANTON_STATIC_BATCH_GATHER") == "on"
            and os.environ.get("ANTON_STATIC_PROFILE_FIRST") != "on"):
        state._pipeline_system += (
            "\nEfficient evidence collection: once the source contract is known, fetch all "
            "required tables, normalize records, calculate requested aggregates, validate "
            "them, and save the structured handoff in one scratchpad execution when practical. "
            "Then finish gathering; don't spend extra rounds printing the same data. Verify "
            "field names against the source schema. Never replace a missing measure or a "
            "failed join with zero. Check joined row coverage and reconcile grouped totals "
            "against the complete source. For risk or what-if analysis, quantify the relevant "
            "available alternatives against the current plan, with their scope explicit."
        )
    if os.environ.get("ANTON_STATIC_REQUEST_METRICS") == "on":
        state._pipeline_system += (
            "\nTranslate every requested business question into computed display metrics during "
            "this evidence collection. Save a compact request_metrics section in observed-data.json: "
            "each requested measure needs its scope/group, unit, actual value and derivation, or a "
            "specific unavailable reason. A dataset-wide total does not answer a question about a "
            "named segment, route, period or alternative: also calculate those requested groups. "
            "For what-if or before/after questions, query the applicable reference and intervention "
            "scenarios and calculate both on the same measures. Lack of an individual causal join "
            "does not prevent a clearly labelled scenario-level comparison. Keep reductions in "
            "exposure, gross revenue, cost and net profit distinct; never call revenue or avoided "
            "risk profit. Before declaring a requested outcome unavailable, inspect the documented "
            "tables/fields and relevant comparison scenarios that can actually answer it. Fetch only "
            "the source tables needed for the requested questions and views, not an entire catalog. "
            "Perform calculations in code against real rows; use no guessed figures or schema fields."
        )
    if os.environ.get("ANTON_STATIC_BATCH_FINISH") == "on":
        state._pipeline_system += (
            "\nFor a resolved request whose data can be collected and validated in one "
            "scratchpad execution, issue scratchpad and finish_gathering together in the "
            "same response, in that order. The cell must fetch the complete required data, "
            "assert the schema/joins/aggregates, and save observed-data.json. The finishing "
            "manifest can name the source, requested scope and data file without guessing "
            "unseen row counts. Tools run sequentially; completion is rejected if an "
            "earlier source execution failed or if source tools follow finish_gathering. "
            "When schema exploration, a decision or a user answer is still needed, inspect "
            "it first and use the ordinary gathering loop."
        )
    if os.environ.get("ANTON_STATIC_REQUEST_FIDELITY") == "on":
        state._pipeline_system += (
            "\nBefore finishing, reconcile the requested quantities with the computed handoff. "
            "Counts, sums, rates, peaks and incremental amounts are different measures: a count "
            "of periods satisfying a condition must count those periods, and an incremental "
            "amount must use the stated reference and scope. Retain every requested quantity "
            "with its exact unit, source scope and derivation. Validate them against the real "
            "source rows without substituting an adjacent measure."
        )
    if os.environ.get("ANTON_STATIC_METRIC_SCOPE") == "on":
        state._pipeline_system += (
            "\nResolve each requested measure's grain separately. A global total and a per-entity "
            "maximum can require different grains; do not force all measures into one grouping. "
            "Inspect the documented role of join keys, not just their names. Reconcile grouped "
            "counts and sums to the intended complete source population and account explicitly "
            "for unmatched rows. An unrelated or missing association does not make the actual "
            "population's total zero. Preserve requested global totals independently of optional "
            "grouped details, and retain unavailable group allocations as unavailable."
        )
    if os.environ.get("ANTON_STATIC_SOURCE_SEMANTICS") == "on":
        state._pipeline_system += (
            "\nResolve source semantics before filtering or asserting a population. Inspect the "
            "observed types and distinct categorical values of fields used in filters or joins; "
            "do not invent enum values or boolean spellings. If a meaning is undocumented, "
            "inspect representative related rows and preserve unresolved values explicitly. "
            "Use the requested or documented reporting horizon, as-of date and calendar. Raw "
            "fact tables can include history, future rows and non-boundary dates. Apply the "
            "actual date bounds before aggregating; distinct event dates are not a count of "
            "reporting periods. Build period buckets from the stated calendar when requested. "
            "Keep raw dates and the chosen bounds in the handoff. Validate assumptions against "
            "observed rows, and include observed values, counts or bounds in assertion failures "
            "so a repair can address the cause without repeatedly guessing."
        )
    if os.environ.get("ANTON_STATIC_STATE_SETS") == "on":
        state._pipeline_system += (
            "\nTranslate the user's business population into a predicate over all observed states. "
            "A lifecycle phrase such as not yet complete can cover several nonterminal source "
            "categories; it need not mean equality to one similarly named enum. Inspect documented "
            "state meanings and account for every observed category. Record the included, excluded "
            "and unresolved states and why, then reconcile the requested count or total against "
            "the rows in that population. Preserve uncertainty when source meanings do not justify "
            "a classification. Use an exact status equality only when the request or source rule "
            "actually specifies that exact status."
        )
    if os.environ.get("ANTON_STATIC_REFERENCE_METRICS") == "on":
        state._pipeline_system += (
            "\nRelative measures such as extra, additional, incremental, avoided or changed need "
            "a reference. Identify the reference from the request and source documentation. "
            "Calculate comparable absolute totals and reference-adjusted differences at the "
            "same scope, grain, period and units, retaining the component derivation. Costs "
            "already incurred in the reference are not additional costs of an alternative. "
            "Do not subtract twice from a source measure already defined as incremental. "
            "Preserve both totals and differences and label the requested measure correctly; "
            "state an unresolved reference explicitly rather than silently choosing one."
        )
    if os.environ.get("ANTON_STATIC_QUERY_CLIENT") == "on":
        state._pipeline_system += (
            "\nFor a connected source whose documented protocol uses form-body OAuth client credentials "
            "and Scenarios/QueryString requests with Columns/Rows responses, the host provides "
            "a standard-library client: `from anton.core.tools.generate_artifact.source_query "
            "import ScenarioRowsClient`; `client = ScenarioRowsClient.from_environment()`; "
            "`rows = client.query_rows(documented_query_path, scenario=scenario_object, "
            "query=query_string)`. rows is list[dict] with the original column names and "
            "values. The first argument is the query URL/path from the procedure, not the "
            "connection's name: for a documented example that posts to base_url + "
            "'/integration/V1/query', use '/integration/V1/query'. Relative paths without "
            "a leading slash and full URLs on that same source origin are accepted. Recreate "
            "the client in each execution rather than relying on notebook persistence. "
            "Supply one scenario mapping per query, with the documented fields; loop "
            "over scenarios when necessary. Use the exact injected DS_ prefix argument if "
            "several complete connections exist. The client handles auth, expiry, row-shape "
            "validation and truncation errors; an unexpected response never becomes zero rows. "
            "This client needs no third-party package. For other source protocols, use their "
            "documented client instead. Never print credentials or embed a client object."
        )
    if os.environ.get("ANTON_STATIC_GATHER_GEOCODING") == "on":
        state._pipeline_system += (
            "\nFor geographic views, resolve ONLY the actual places needed for the requested entities "
            "and position roles during this collection. Use the installed local helper: "
            "`from anton.core.tools.generate_artifact.geo_index import resolve_place`; "
            "`position = resolve_place(observed_place, source_country)` returns lat, lon, country, "
            "location, geonameid and GeoNames provenance. Country can be omitted only for a unique "
            "place. Catch ValueError to keep ambiguous/unmatched places explicitly unresolved; do "
            "not guess a candidate. Save each resolved position alongside its actual source entity "
            "id and location role in observed-data.json. A destination or headquarters coordinate "
            "must never stand in for a current position. Do not geocode arbitrary identifiers, "
            "country columns or descriptive labels. Retain source-supplied coordinates when they "
            "describe the requested position. Credit GeoNames in a page using this lookup."
        )
    if os.environ.get("ANTON_STATIC_GEO_CONTRACT") == "on":
        state._pipeline_system += (
            "\nGeocoding return contract: resolve_place returns a JSON mapping containing lat, "
            "lon, country, location, geonameid and source. Retain that returned mapping directly "
            "with the entity and position role; mapping access such as point['lat'] is supported. "
            "An optional country must describe this exact place/position, not the organization's "
            "home country or an origin/destination elsewhere. A place containing its own country "
            "can be resolved without an unrelated hint. Catch only ValueError for an unresolved "
            "place and preserve str(error); TypeError and AttributeError are programming errors "
            "to repair, not evidence that coordinates are unavailable."
        )
    elif os.environ.get("ANTON_STATIC_GEO_CAPABILITY") == "on":
        state._pipeline_system += (
            "\nGeographic capability: geonamescache is installed in Python. Prefer coordinates "
            "already supplied by the source. For named places lacking coordinates, use "
            "geonamescache.GeonamesCache().search_cities(place, contains_search=False), then "
            "disambiguate the returned name, countrycode, latitude and longitude using the "
            "source's country/region. Country name-to-code metadata is available from "
            "get_countries_by_names(). Never invent coordinates or silently select an ambiguous "
            "place. Print the resolved positions and provenance with the data; report unmatched "
            "places. Credit GeoNames in any page using this gazetteer."
        )
    if os.environ.get("ANTON_STATIC_DATA_BINDING") == "on":
        state._pipeline_system += (
            "\nStructured data handoff: save one canonical JSON document containing the "
            "complete relevant datasets and exact computed aggregates to "
            f"{state.artifact_path / 'observed-data.json'}. Use json.dumps(..., allow_nan=False) "
            "and UTF-8. Include clear source/scenario/time scope and units, but no credentials. "
            "Print only a short manifest of keys and row counts. The host reads this file "
            "directly and embeds the JSON; the console is capped at 10,000 characters, so "
            "printing a long dataset silently loses rows. Do not print raw and normalized "
            "copies of the same records. Source documentation remains instructions for "
            "access, not evidence for invented rows."
        )
    state._pipeline_tools = [tool for tool in state.pipeline_tools
                             if tool.get("name") in {"scratchpad", "web_fetch", "web_search", "ask_user", "finish_gathering"}]
    if os.environ.get("ANTON_STATIC_COHERENT_GATHER") == "on":
        state._pipeline_tools = copy.deepcopy(state._pipeline_tools)
        for tool in state._pipeline_tools:
            if tool["name"] == "finish_gathering":
                tool["description"] = (
                    "Complete evidence collection for this static artifact. With a documented source and "
                    "resolved scope, call scratchpad and this tool in the SAME response, in that order. "
                    "The cell must fetch, validate and save observed-data.json. The host executes the "
                    "cell first and rejects completion if it fails. Supply only a short summary, "
                    "artifact_type and source names; don't guess unseen results or draft a PRD. "
                    "When exploration or a user answer is needed, inspect it before completing."
                )
    from . import static_gathering
    if static_gathering.enabled(state):
        state._pipeline_tools = static_gathering.tools(state._pipeline_tools)
        state._pipeline_system += (
            "\nThis static collection supports one-call completion: call scratchpad with action exec, "
            "the data-fetch/validation code, and complete_static_artifact={summary,data_sources}. "
            "The code must fetch the complete requested data, assert its schema and calculations, "
            "and write observed-data.json at the exact artifact path above. After successful "
            "execution and fresh-file validation, the host completes collection automatically. "
            "No separate finish_gathering call is needed. Omit completion metadata for exploratory "
            "cells or when a user answer is needed; normal collection tools remain available."
        )
    if os.environ.get("ANTON_STATIC_REUSE_SNAPSHOT") == "on":
        from .snapshot_reuse import instructions
        state._pipeline_system += instructions(state.artifact_path, state.primary or "index.html")
    if os.environ.get("ANTON_STATIC_PROFILE_FIRST") == "on":
        state._pipeline_system += (
            "\nObserve before analyzing: for new or changed source scope, use two short stages. "
            "First execute only the necessary documented queries and cache their complete JSON "
            "record lists, keyed by descriptive scenario/table names, at "
            f"{state.artifact_path / 'source-records.json'}. Save with UTF-8 and allow_nan=False, "
            "without credentials. Print `profile_tables(tables)` from "
            "`anton.core.tools.generate_artifact.source_profile`; it returns bounded JSON text "
            "with real row counts, fields, types, missing/null counts, observed scalar values "
            "with counts, and numeric/date ranges. If it asks for fewer tables, print profiles "
            "for smaller groups. Do not finish collection or include complete_static_artifact "
            "in this first execution: inspect the returned profile before writing business "
            "predicates, joins or conclusions. Then execute analysis against the saved records, "
            "using the actual observed categories and documented field roles. Reconcile joins "
            "and totals; a missing association is not a zero population. Save the complete "
            "requested data and computed measures into observed-data.json and complete collection "
            "with the ordinary validated completion metadata. Reuse the same scratchpad name; "
            "read the cache file instead of fetching the same rows again during repairs. Fetch "
            "more only when missing data or a changed request actually requires it. A profile "
            "is evidence of shape and values, not a substitute for analysis or a final handoff. "
            "A purely presentational edit may instead use the validated same-artifact snapshot "
            "reuse procedure above when its scope and required data are still applicable."
        )
    gather_effort = os.environ.get("ANTON_STATIC_GATHER_EFFORT")
    if gather_effort:
        from anton.core.llm.reasoning import scoped_effort
        state.record("gathering", "effort_override", gather_effort)
        with scoped_effort(gather_effort):
            await run_gathering_loop(state)
    else:
        await run_gathering_loop(state)
    final_type = state.final_artifact_type or state.artifact_type
    if final_type != "html-app" or not state.gathering_complete or state.unverified_sources:
        state._pipeline_system, state._pipeline_tools = "", None
        return False
    if final_type != state.artifact_type:
        from anton.core.tools.tool_handlers import resolve_artifact_store
        store = resolve_artifact_store(state.session)
        if store is not None:
            store.update(state.slug, type=final_type, primary="index.html")
        state.primary = "index.html"
    state.settle_artifact_type(final_type)
    return True
