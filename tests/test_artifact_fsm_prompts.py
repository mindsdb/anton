from __future__ import annotations

import json
from pathlib import Path

from anton.core.tools.generate_artifact import prompts
from anton.core.tools.generate_artifact.state import GenState


def _state(**kw):
    base = dict(
        session=object(), artifact_type="fullstack-stateless-app",
        artifact_path=Path("/tmp/a"), slug="a", brief="Build X", is_fullstack=True,
    )
    base.update(kw)
    return GenState(**base)


def test_fsm_digraph_is_english_and_covers_nodes():
    g = prompts.FSM_DIGRAPH
    for node in [
        "is_data_enough", "define_required_data", "is_possible_to_fetch",
        "fetch_data_sample", "not_enough_data", "make_tech_spec",
        "is_fullstack", "make_api_spec", "generate_backend", "verify_backend",
        "generate_frontend", "verify_frontend", "run_app", "verify_fullstack",
    ]:
        assert node in g


def test_backend_rules_require_health_endpoint():
    assert "/api/health" in prompts._BACKEND_RULES


def test_decision_prompts_embed_the_graph_and_state():
    st = _state(data_notes="pad `a` cell 2 pulled 100 rows")
    system, user = prompts.build_required_data_prompt(st)
    assert "digraph" in system
    assert "Build X" in user
    assert "pad `a`" in user


def test_tech_spec_prompt_targets_spec_md():
    system, user = prompts.build_tech_spec_prompt(_state())
    assert "spec.md" in system or "spec.md" in user


def test_tech_spec_prompt_pins_the_stack():
    """The stack block lives in the step instruction now, so the hot path —
    which never saw the cold-start system prompt — gets it too."""
    system, user = prompts.build_tech_spec_prompt(_state())
    joined = system + user
    assert "FastAPI" in joined
    assert "Python >= 3.12" in joined
    assert "/api/*" in joined
    instruction = prompts.build_tech_spec_instruction(_state())
    assert prompts._TECH_SPEC_STACK in instruction
    assert "never mention a port number" in joined
    # The generator rules must state the target runtime too.
    assert "Python >= 3.12" in prompts._BACKEND_RULES


def test_fetch_data_prompts_exist():
    assert isinstance(prompts.build_fetch_data_system_prompt(Path("/tmp/a")), str)
    assert "scratchpad" in prompts.build_fetch_data_kickoff(_state()).lower()


def test_prompts_include_progress_journal():
    st = _state()
    _, user = prompts.build_required_data_prompt(st)
    assert "## Progress journal" not in user  # empty journal → no section
    st.record("data_check", "no", "need orders")
    _, user = prompts.build_required_data_prompt(st)
    assert "## Progress journal" in user
    assert "- data_check: no — need orders" in user


def test_backend_prompt_states_the_ds_env_var_convention():
    """Without the naming convention the generator guesses keys and fails _map_datasources."""
    for stateless in (True, False):
        system = prompts.build_backend_system_prompt(Path("/tmp/a"), stateless=stateless)
        assert "DS_<ENGINE>_<NAME>__<FIELD>" in system


_CATALOG = (
    "\n\n## Connected Data Sources\nEach connection has a Slug …\n"
    "\n### Slug: `mysql-df6b1140` — Label: (none)\nEngine: MySQL\nHost: h\n"
    "Credential env vars:\n - DS_MYSQL_DF6B1140__HOST\n"
    "\n### Slug: `snowflake-wh42` — Label: warehouse\nEngine: Snowflake\n"
    "Credential env vars:\n - DS_SNOWFLAKE_WH42__ACCOUNT\n"
)


def test_backend_prompt_embeds_the_catalog_entries_the_artifact_declared():
    """Distinctive names — `_BACKEND_RULES` itself quotes DS_POSTGRES_PROD_DB__*
    and DS_HUBSPOT_MAIN__* as examples, so an assertion on those passes
    vacuously."""
    system = prompts.build_backend_system_prompt(
        Path("/tmp/a"), stateless=True, datasource_context=_CATALOG,
        declared_sources=["sales rows from the warehouse (snowflake)"],
    )
    assert "DS_SNOWFLAKE_WH42__ACCOUNT" in system
    assert "DS_MYSQL_DF6B1140__HOST" not in system


def test_backend_prompt_replaces_the_catalog_with_a_note_when_no_source_is_declared():
    """Run 18: two databases and ten DS_* names went into a backend whose spec
    said "no external data" — noise and a temptation."""
    system = prompts.build_backend_system_prompt(
        Path("/tmp/a"), stateless=True, datasource_context=_CATALOG,
    )
    assert "None are used by this artifact" in system
    assert "DS_SNOWFLAKE_WH42__ACCOUNT" not in system
    assert "DS_MYSQL_DF6B1140__HOST" not in system


def test_datasource_section_keeps_the_whole_catalog_when_unsure():
    """Declared names are free text; a wrong drop costs a regeneration that
    the full catalog never does. Gathered cells reading `DS_*` count as a
    declaration the model forgot to make."""
    both = ("DS_MYSQL_DF6B1140__HOST", "DS_SNOWFLAKE_WH42__ACCOUNT")
    unsure = prompts._datasource_section(_CATALOG, ["the article at http://x"], "")
    assert all(name in unsure for name in both)
    forgot = prompts._datasource_section(_CATALOG, [], "cell: os.environ['DS_MYSQL_DF6B1140__HOST']")
    assert all(name in forgot for name in both)
    by_slug = prompts._datasource_section(_CATALOG, ["mysql-df6b1140 orders"], "")
    assert "DS_MYSQL_DF6B1140__HOST" in by_slug and "DS_SNOWFLAKE" not in by_slug
    assert prompts._datasource_section("", [], "") == ""


def test_fetch_prompt_embeds_the_datasource_catalog():
    catalog = "\n\n## Connected Data Sources\n- `hubspot-main` (hubspot) → DS_HUBSPOT_MAIN__ACCESS_TOKEN"
    system = prompts.build_fetch_data_system_prompt(
        Path("/tmp/a"), datasource_context=catalog
    )
    assert "DS_HUBSPOT_MAIN__ACCESS_TOKEN" in system


def test_prompt_builders_tolerate_missing_catalog():
    """The default is mandatory: existing callers pass no new argument."""
    assert isinstance(prompts.build_backend_system_prompt(Path("/tmp/a")), str)
    assert isinstance(prompts.build_fetch_data_system_prompt(Path("/tmp/a")), str)


def test_backend_template_includes_secrets_block_and_os_import():
    """The verifier requires a module-level SECRETS dict (verifiers.py:171)."""
    rules = prompts._BACKEND_RULES
    assert "import os" in rules
    assert "SECRETS = {" in rules
    assert "os.environ.get(" in rules


def test_backend_rules_forbid_hoisting_secrets_to_module_level():
    """The AST check at verifiers.py:183 punishes copying a secret at import time."""
    rules = prompts._BACKEND_RULES.lower()
    # Case-insensitive: the prompt spells the phrase in caps ("AT ITS POINT OF
    # USE"), and the contract lock also compares via _squash — two tests about the
    # same thing must not disagree over letter case.
    assert "point of use" in rules
    assert "module-level" in rules


def test_backend_rules_explain_the_two_run_modes():
    """The SECRETS rule is followed better when the reason is stated."""
    rules = prompts._BACKEND_RULES
    assert "overlays" in rules.lower()


def test_stateless_and_stateful_rules_are_mutually_exclusive():
    stateless = prompts.build_backend_system_prompt(Path("/tmp/a"), stateless=True)
    stateful = prompts.build_backend_system_prompt(Path("/tmp/a"), stateless=False)

    # stateless: no local storage at all, and no STATE SDK either
    assert "sqlite" in stateless.lower()
    assert "no local" in stateless.lower() or "must not persist" in stateless.lower()
    assert "state_manifest.json" not in stateless
    assert "Do NOT import `anton_state`" in stateless

    # stateful: durable state goes through the platform STATE store
    assert "STATE = None" in stateful
    assert "state_manifest.json" in stateful
    assert "open_store" in stateful
    # ...and the stateless branch's prohibition does not leak into it
    assert "assume read-only at runtime" not in stateful
    assert "Do NOT import `anton_state`" not in stateful


def test_shared_backend_rules_no_longer_hardcode_statelessness():
    """An unconditional STATELESS in _BACKEND_RULES broke fullstack-stateful-app."""
    rules = prompts._BACKEND_RULES
    assert "STATELESS:" not in rules
    assert "assume read-only at runtime" not in rules


def test_stateful_rules_carry_the_state_sdk_contract():
    """The STATE contract markers the verifier rules point at (contract lock)."""
    stateful = prompts.build_backend_system_prompt(Path("/tmp/a"), stateless=False)
    # module-level slot + point-of-use store construction
    assert "STATE = None" in stateful
    assert "POINT OF USE" in stateful
    # manifest: flat object, never a CreateTable shape, collections registry
    assert "FLAT JSON object" in stateful
    assert "collections" in stateful
    # never in requirements.txt
    assert "NEVER list `anton_state`" in stateful
    # no scan / no secondary indexes, atomics, no manual retries on mutations
    assert "NO `scan()`" in stateful
    assert "increment" in stateful
    assert "retry loop" in stateful
    # heavy/relational data belongs in an external DB
    assert "EXTERNAL database" in stateful


def test_stateful_block_says_when_local_state_survives():
    """The STATE store must be described as working both locally and deployed."""
    stateful = prompts.build_backend_system_prompt(Path("/tmp/a"), stateless=False)
    assert "Lambda" in stateful
    assert "locally" in stateful
    assert "SQLite" in stateful  # the local fallback driver


def test_api_spec_prompt_state_constraints_by_type():
    """Stateless and stateful each get their own persistence constraint block."""
    _, stateless_user = prompts.build_api_spec_prompt("ctx", stateless=True)
    _, stateful_user = prompts.build_api_spec_prompt("ctx", stateless=False)
    assert "## Stateless constraint" in stateless_user
    assert "## Durable state constraint" not in stateless_user
    assert "## Durable state constraint" in stateful_user
    assert "## Stateless constraint" not in stateful_user
    # key design guidance for the endpoint shapes
    assert "NO scan" in stateful_user
    assert "partition-key query" in stateful_user
    assert "atomic increment" in stateful_user


def test_api_spec_instruction_carries_the_rules_the_system_prompt_used_to():
    """On the hot path the node continues the shared history under the
    pipeline system prompt, so a system prompt of its own is never seen. The
    rules have to travel in the step message — the seventeenth live run got
    a fenced OpenAPI 3.0 document with tags and header schemas because they
    did not."""
    instruction = prompts.build_api_spec_instruction(stateless=True)
    for marker in (
        "no markdown fence",
        "OpenAPI 3.1",
        "`## Backend`",
        "add none, drop none, rename none",
        "spelled exactly",
        "ONLY place the request and response shapes",
        "no invented",
        "`description` only where",
        "/api/...",
        "Compact",
        "no `tags`",
        "## Stateless constraint",
        "Write the OpenAPI JSON document now.",
    ):
        assert marker in instruction, marker
    assert "## Durable state constraint" in prompts.build_api_spec_instruction(stateless=False)


def test_both_kickoffs_carry_the_contract_as_one_line_of_json():
    """`openapi.json` is indented for people; the twentieth live run put
    35.8 KB of it into each generator's kickoff where the compact form is
    11.7 KB. Text that is not JSON passes through unchanged."""
    pretty = json.dumps({"openapi": "3.1.0", "paths": {"/api/t": {"get": {"responses": {"200": {}}}}}}, indent=2)
    compact = json.dumps(json.loads(pretty))
    assert "\n" in pretty and "\n" not in compact
    for kickoff in (
        prompts.build_backend_kickoff("ctx", pretty),
        prompts.build_frontend_kickoff("ctx", pretty),
    ):
        assert compact in kickoff
        assert pretty not in kickoff
    assert "{not json" in prompts.build_backend_kickoff("ctx", "{not json")


def test_cold_and_hot_api_spec_paths_ask_for_the_same_document():
    """The cold start prepends the assembled context; the instruction after
    it is byte-for-byte what the hot path sends on its own."""
    system, user = prompts.build_api_spec_prompt("## Product requirements\nX", stateless=True)
    assert user.startswith("## Requirements\n## Product requirements\nX")
    assert user.endswith(prompts.build_api_spec_instruction(stateless=True))
    assert "`make_api_spec` node" in system


def test_tech_spec_stack_pins_the_state_store():
    """Without this the spec writer invents sqlite and the generators build it."""
    stack = prompts._TECH_SPEC_STACK
    assert "STATE store" in stack
    assert "fullstack-stateful-app" in stack
    assert "Do NOT propose sqlite" in stack
    assert "EXTERNAL" in stack


def test_stateful_task_demands_the_manifest_file():
    """Stateful asks for three files (incl. state_manifest.json), stateless for two."""
    stateful = prompts.build_backend_system_prompt(Path("/tmp/a"), stateless=False)
    stateless = prompts.build_backend_system_prompt(Path("/tmp/a"), stateless=True)
    assert "exactly three files" in stateful
    assert "state_manifest.json" in stateful
    assert "exactly two files" in stateless


def test_verifier_contract_reaches_both_frontend_prompts():
    """`_VERIFIER_CONTRACT` is quoted under `## Verifier contract` by the
    html-app builder and the fullstack frontend builder alike."""
    for rules in (
        prompts._VERIFIER_CONTRACT,
        prompts.build_subagent_system_prompt("html-app", Path("/tmp/a")),
        prompts.build_frontend_system_prompt(Path("/tmp/a")),
    ):
      for marker in (
        "explicit `<body>`",
        "absolute URL",
        "__antonCommentsLayer",
        "!important",
        "z-index",
        "stable `id`",
      ):
        assert marker in rules, marker


def test_design_rules_recommend_tailwind_but_keep_hand_written_css():
    """Tailwind via CDN is the recommended styling; a `<style>` block stays
    allowed, and no other library may be loaded. Both frontend prompts and
    the spec writer's stack block must agree, or the spec forbids what the
    frontend step is told to do."""
    rules = prompts._DESIGN_RULES
    assert "https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4" in rules
    assert "recommended" in rules
    assert "Hand-written CSS" in rules
    assert "ONLY external resources" in rules
    # Both CDNs need network at view time: an offline requirement wins over
    # the recommendation, and the spec writer knows it too.
    assert "OFFLINE" in rules
    assert "offline" in prompts._TECH_SPEC_STACK
    for text in (
        prompts.build_subagent_system_prompt("html-app", Path("/tmp/a")),
        prompts.build_frontend_system_prompt(Path("/tmp/a")),
        prompts._TECH_SPEC_STACK,
    ):
        assert "Tailwind" in text
        assert "ECharts" in text
    # The spec block no longer says "inline CSS" as if a framework were banned.
    assert "inline CSS" not in prompts._TECH_SPEC_STACK


# ── backend prompt: same skeleton as the two frontend prompts (2026-09-17) ──

def _backend_prompt(*, stateless: bool = True) -> str:
    return prompts.build_backend_system_prompt(
        Path("/tmp/artifact-prompt-probe"), stateless=stateless
    )


def test_backend_prompt_has_the_shared_skeleton_plus_its_own_rules():
    s = _backend_prompt()
    order = ["## Your task", "## What you receive", "## Workflow",
             "## Output protocol", "## Verifier contract", "## Backend rules",
             "## State rules", "## Tools"]
    positions = [s.index(h) for h in order]
    assert positions == sorted(positions), order
    assert "step is `generate_backend`" in s
    assert "one file per reply" in s
    assert "NOT part of the\njob" in s or "NOT part of the job" in s


def test_backend_prompt_carries_no_leftovers_from_the_role_block():
    s = _backend_prompt()
    for gone in (
        "SCRATCHPAD DISCIPLINE",
        "DATA INTO FILES",
        "HOW MUCH GOES IN ONE REPLY",
        "## Output folder",
        "/tmp/artifact-prompt-probe",
        "the brief's",
        "main agent",
        'write_file(path="index.html")',
        "</body></html>",
    ):
        assert gone not in s, gone
    # The python example replaces the html one in the output protocol.
    assert "uvicorn.run(app" in s.split("## Output protocol")[1].split("## Verifier")[0]


def test_backend_verifier_contract_names_what_verify_backend_checks():
    stateless = _backend_prompt(stateless=True)
    contract = stateless.split("## Verifier contract")[1].split("## Backend rules")[0]
    for marker in ("/api/health", 'Mangum(app, lifespan="off")', "SECRETS", "point of use",
                   "requirements.txt", "`anton_state`", "DS_*", "fails the step"):
        assert marker in contract, marker
    assert "`anton_state` is not imported" in contract
    stateful = _backend_prompt(stateless=False).split("## Verifier contract")[1].split("## Backend rules")[0]
    assert "STATE = None" in stateful and "state_manifest.json" in stateful


def test_backend_kickoff_does_not_open_with_a_scratchpad_invitation():
    kickoff = prompts.build_backend_kickoff("## Product requirements\nx", "{}")
    assert "confirm the schema/sample" not in kickoff
    assert "follow the workflow from your instructions" in kickoff
    assert 'write_file(path="backend.py")' in kickoff
    assert kickoff.index("## API Specification") < kickoff.index("follow the workflow")


def test_frontend_rules_pin_static_as_the_only_served_folder():
    assert "static/" in prompts._FRONTEND_RULES
    assert "only" in prompts._FRONTEND_RULES.lower()


def test_backend_rules_describe_requirements_line_format():
    rules = prompts._BACKEND_RULES
    assert "--index-url" in rules or "-r" in rules


def test_subagent_prompt_serves_html_app_only():
    """The only live caller is orchestrator.py:329, the non-fullstack branch."""
    html = prompts.build_subagent_system_prompt("html-app", Path("/tmp/a"))
    assert "single self-contained HTML file" in html or "ONE self-contained HTML" in html

    # The dead fullstack branches are gone: a third copy of the contract must not
    # exist and silently drift away from _BACKEND_RULES.
    for dead in ("fullstack-stateless-app", "fullstack-stateful-app"):
        out = prompts.build_subagent_system_prompt(dead, Path("/tmp/a"))
        assert "Unsupported artifact type" in out
        assert "backend.py" not in out


def test_html_prompt_pins_the_registered_primary():
    """Otherwise the model writes the default name while metadata promises another."""
    system = prompts.build_subagent_system_prompt(
        "html-app", Path("/tmp/a"), primary="report.html"
    )
    assert "report.html" in system


def test_html_prompt_falls_back_to_index_html():
    """primary is optional (Artifact.primary: str | None). The default was
    `dashboard.html` until 2026-09-16 — the tenth live run wrote a card game
    under that name."""
    for primary in (None, ""):
        system = prompts.build_subagent_system_prompt(
            "html-app", Path("/tmp/a"), primary=primary
        )
        assert "index.html" in system
        assert "dashboard.html" not in system


def test_size_rules_block_is_present_in_both_frontend_prompts():
    """Split writing is needed by html-app and the fullstack frontend alike;
    since 2026-09-17 both quote the one `_GEN_SIZE_RULES` block."""
    assert "mode=\"a\"" in prompts._GEN_SIZE_RULES
    for system in (
        prompts.build_subagent_system_prompt("html-app", Path("/tmp/a")),
        prompts.build_frontend_system_prompt(Path("/tmp/a")),
    ):
        assert "mode=\"a\"" in system
        assert "Split ONLY" in system


def test_role_no_longer_forbids_splitting_a_file():
    """HARD RULES outweigh any block below it — the old rule has to go."""
    role = prompts._ROLE
    assert "Do NOT split a single file across multiple calls" not in role
    assert "exactly once per file" not in role


def test_role_documents_the_mode_argument():
    """`content` is gone from the signature — the body travels as text now."""
    assert "write_file(path, mode" in prompts._ROLE
    assert "write_file(path, content" not in prompts._ROLE


def test_role_using_data_no_longer_demands_one_shot_embedding():
    """The USING DATA paragraph told the model to embed data in a single call."""
    role = prompts._ROLE
    assert "EMBED the real data into the output file" not in role
    assert "mode=\"a\"" in role


def test_tech_spec_prompt_requires_an_insight_list_for_html_app():
    system, user = prompts.build_tech_spec_prompt(
        _state(artifact_type="html-app", is_fullstack=False)
    )
    joined = system + user
    assert "insight" in joined.lower()
    assert "one line each" in joined.lower()


def test_fetch_prompt_tells_the_node_to_reuse_the_named_pad():
    """A fresh pad name gives an isolated environment — variables and imports are lost."""
    system = prompts.build_fetch_data_system_prompt(Path("/tmp/a"))
    low = system.lower()
    assert "only what is missing" in low or "only the missing" in low
    assert "same scratchpad" in low


def test_fetch_prompt_embeds_the_public_sources_catalog():
    system = prompts.build_fetch_data_system_prompt(
        Path("/tmp/a"), public_sources="PUBLIC DATA:\n- Google News RSS: ..."
    )
    assert "Google News RSS" in system


def test_fetch_prompt_tolerates_missing_public_sources():
    assert isinstance(prompts.build_fetch_data_system_prompt(Path("/tmp/a")), str)


def test_role_carries_the_scratchpad_discipline():
    """The rules from the main agent's system prompt never reached the generator."""
    role = prompts._ROLE
    for marker in (
        "clean namespace",   # nothing is pre-imported
        "120",               # the hard per-cell timeout
        "print(",            # output only via print
        "DS_",               # credentials arrive as env vars
        "data_vault",         # must not be read directly
        "change strategy",   # switch approach after a repeated failure
    ):
        assert marker in role, marker


def test_role_does_not_duplicate_the_exec_field_requirement():
    """`one_line_description` is already required in USING DATA — no second copy."""
    assert prompts._ROLE.count("one_line_description") == 1


def test_fetch_prompt_has_no_write_file_instructions():
    """The node writes no files — write_file instructions only get in its way."""
    system = prompts.build_fetch_data_system_prompt(Path("/tmp/a"))
    assert "Do NOT write any artifact files" in system
    assert "write_file" not in system
    assert "read_file" not in system
    assert "mode=\"a\"" not in system


def test_fetch_prompt_keeps_the_common_part():
    """The node still needs scratchpad, the discipline and finish."""
    system = prompts.build_fetch_data_system_prompt(Path("/tmp/a"))
    for marker in ("scratchpad(", "finish(", "clean namespace", "DS_"):
        assert marker in system, marker


def test_generator_prompts_still_get_the_write_part():
    for system in (
        prompts.build_subagent_system_prompt("html-app", Path("/tmp/a")),
        prompts.build_frontend_system_prompt(Path("/tmp/a")),
        prompts.build_backend_system_prompt(Path("/tmp/a"), stateless=True),
    ):
        assert "write_file" in system
        assert "finish(" in system


def test_role_is_the_composition_of_both_halves():
    """The _ROLE name is preserved: three stage-1c tests read it directly."""
    assert prompts._ROLE_COMMON in prompts._ROLE
    assert prompts._ROLE_WRITE in prompts._ROLE


def test_tech_spec_instruction_refuses_to_restate_a_prd_it_has():
    """A full spec next to a confirmed PRD is near-pure duplication (measured
    2026-08-27: 190 s / 13k output tokens restating a 20 KB PRD; seventh live
    run 2026-09-16: a 5.5 KB spec for a 2.1 KB PRD, five of eight sections a
    retelling), and `_spec_context` carries both into every generation
    prompt. The rule is in the INSTRUCTION, because the hot path never saw
    the cold-start system prompt where it used to live."""
    instruction = prompts.build_tech_spec_instruction(
        _state(artifact_type="html-app", is_fullstack=False, prd="# PRD\nGoal: x")
    )
    assert "Do not restate the PRD" in instruction
    assert "Implementation notes" in instruction
    assert "No acceptance criteria" in instruction
    assert "Do not call any tool" in instruction


def test_tech_spec_instruction_drops_the_no_restate_rule_without_a_prd():
    """Nothing to restate — the rule would only confuse."""
    instruction = prompts.build_tech_spec_instruction(
        _state(artifact_type="html-app", is_fullstack=False)
    )
    assert "Do not restate the PRD" not in instruction
    assert "Implementation notes" in instruction


def test_tech_spec_instruction_adds_a_backend_section_for_fullstack():
    """Fullstack specs feed the API design: they get a `## Backend` section
    on top of the shared rules, not a licence to retell the PRD."""
    fullstack = prompts.build_tech_spec_instruction(
        _state(artifact_type="fullstack-stateless-app", is_fullstack=True, prd="# PRD")
    )
    html = prompts.build_tech_spec_instruction(
        _state(artifact_type="html-app", is_fullstack=False, prd="# PRD")
    )
    assert "## Backend" in fullstack and "/api/*" in fullstack
    assert "## Backend" not in html
    assert "Do not restate the PRD" in fullstack
    # The section lists endpoints; the contract (shapes, fields, formats)
    # lives in openapi.json only — run 18 had the two disagree.
    assert "No request or response shapes" in fullstack
    assert "`openapi.json`" in fullstack


def test_tech_spec_instruction_leaves_the_frontends_design_rules_alone():
    """Seventh live run: the spec spent lines on palette, flip timing and a
    breakpoint the generator's own `_DESIGN_RULES` already decide."""
    instruction = prompts.build_tech_spec_instruction(_state(prd="# PRD"))
    assert "frontend's own design rules" in instruction
    assert "theme, fonts, chart library" in instruction


def test_cold_and_hot_tech_spec_paths_ask_for_the_same_document():
    """The cold-start prompt is the assembled context plus the very same
    instruction the hot path sends — one text, one set of rules."""
    st = _state(artifact_type="html-app", is_fullstack=False, prd="# PRD")
    _, user = prompts.build_tech_spec_prompt(st)
    assert user.endswith(prompts.build_tech_spec_instruction(st))


def test_role_says_verification_is_not_the_models_job():
    """Nine of twenty rounds of the 2026-08-27 live run went to self-checks the
    deterministic verifier repeats anyway — and the round budget died of it."""
    role = prompts._ROLE_WRITE
    assert "VERIFICATION IS NOT YOUR JOB" in role
    assert "finish" in role


def test_role_forbids_new_scratchpad_names():
    assert "NEVER create a scratchpad with a new name" in prompts._ROLE_COMMON


def test_the_size_rules_no_longer_state_a_character_limit():
    """The limit was derived from how long a silent connection survives, and
    with the body in the text there is no silence to survive.

    Keeping the number would make the model split files that fit — and under
    one-body-per-reply every needless split costs a whole round. The ceiling
    that remains is the reply's own output budget, which needs no number here.
    """
    d = prompts._GEN_SIZE_RULES
    assert "characters of `content`" not in d
    assert "HARD CHUNK LIMIT" not in d
    # The phrase that produced the opposite error: naming the first two chunks
    # as "where oversized calls fail" had the model keep exactly those small
    # and then send 19 819 characters in the one right after.
    assert "oversized calls fail" not in d
    assert "single body" in d


def test_the_size_guidance_is_a_number_the_model_can_check():
    """"When it fits in your output budget" is a condition the model cannot
    evaluate — it has no view of its remaining tokens.

    Measured 2026-09-15: asked that way, it followed the concrete splitting
    recipe underneath instead and cut a 33 687-character file that would have
    fit, costing a round. The figure is in CHARACTERS because that is what the
    model can compare against the file it is about to write, and it is quoted
    from the constant so the prompt and the budget cannot drift apart.
    """
    from anton.core.tools.generate_artifact.state import REPLY_BODY_CHARS

    d = prompts._GEN_SIZE_RULES
    assert f"{REPLY_BODY_CHARS:,} characters" in d
    assert "output budget" not in d


def test_splitting_is_described_as_the_exception_not_the_recipe():
    """The old text prescribed WHERE to split — head/style/body, then the data
    block, then the markup, then the scripts. That is a procedure the model can
    execute without ever weighing whether to split at all, and the one live
    split we measured fell exactly on the boundary it named.
    """
    d = prompts._GEN_SIZE_RULES

    # No structural recipe left.
    for gone in ("opening `<body>`", "then the scripts", "First part:"):
        assert gone not in d, gone

    # The whole-file case is stated before the split case.
    assert d.index("single body") < d.index("Split ONLY")


def test_the_markers_are_quoted_identically_on_every_surface():
    """One pair of constants, several surfaces the model reads.

    This replaces the chunk-limit lock for the same reason it existed: the
    protocol breaks silently if any surface drifts. A body written against a
    marker the parser does not recognise is a file that never lands.
    """
    from anton.core.tools.generate_artifact import engine, sub_tools

    begin, end = sub_tools.FILE_BEGIN_MARKER, sub_tools.FILE_END_MARKER
    for surface in (
        sub_tools.WRITE_FILE_SCHEMA["description"],
        prompts._ROLE,
        engine._CONTENT_ARG_MSG,
    ):
        assert begin in surface and end in surface

    # And the kickoffs, which are the last thing the model reads before writing.
    for kickoff in (
        prompts.build_user_kickoff("## Brief\nx"),
        prompts.build_frontend_kickoff("ctx", "{}"),
    ):
        assert begin in kickoff and end in kickoff


# ── `read_file`'s surfaces must agree about `full` ──────────────────────────
#
# The 2026-09-14 run found them disagreeing in the way that costs the most:
# the only place stating that `full=true` is expensive and must not be used to
# check finished work was the tool schema's `description`, while the system
# prompt showed `read_file(path)` as a one-argument call and separately
# promised the tail was "all you need". The model read all of them, believed
# the promise, discovered the tail says nothing about the middle of a file,
# and escalated to `full=true` anyway.

def _write_loop_prompts() -> dict[str, str]:
    """Every system prompt a `_run_loop` write round can be sent.

    All three, not just the html-app one: they share the FILE TOOLS and chunk
    sections, so a fix applied to one surface and not the others is exactly the
    divergence these tests exist to catch.
    """
    return {
        "html-app": prompts.build_subagent_system_prompt(
            "html-app", Path("/tmp/a"), primary="index.html"
        ),
        "frontend": prompts.build_frontend_system_prompt(Path("/tmp/a")),
        "backend": prompts.build_backend_system_prompt(Path("/tmp/a")),
    }


def test_the_system_prompt_names_the_full_parameter_at_all():
    """It did not, which is why the only warning about it lived in the tool
    schema — which the model reads as a listing, not as rules."""
    for name, text in _write_loop_prompts().items():
        assert "full=true" in text, name


def test_the_system_prompt_says_what_full_costs_and_when_not_to_use_it():
    for name, text in _write_loop_prompts().items():
        assert "ENTIRE file" in text, name
        assert "finish" in text, name


def test_no_surface_still_claims_the_tail_is_all_you_need():
    """Falsified by the run: the tail says nothing about the middle of a file,
    so promising it is sufficient is what sends the model looking for a way
    around it."""
    for name, text in _write_loop_prompts().items():
        assert "which is all you need" not in text, name


def test_the_write_result_is_advertised_as_the_cheaper_answer():
    """`write_file` reports lines now; the prompt has to say so, or the model
    spends a round re-learning what it was already told."""
    for name, text in _write_loop_prompts().items():
        assert "LINES" in text, name
        assert "without reading" in text, name


# ── html-app generator prompt: structure (2026-09-15 rewrite) ───────────────
#
# The prompt was reordered so the model reads the task and its done-criterion
# first, then what its input carries, then the workflow, and only then the
# mechanics. Each rule is stated once. These locks hold the order, the input
# headings (which must match what `_spec_context` renders) and the removals.

def _html_prompt(primary: str = "index.html") -> str:
    return prompts.build_subagent_system_prompt(
        "html-app", Path("/tmp/artifact-prompt-probe"), primary=primary
    )


def test_html_prompt_states_the_task_before_the_mechanics():
    s = _html_prompt()
    order = ["## Your task", "## What you receive", "## Workflow",
             "## Output protocol", "## Verifier contract", "## Tools"]
    positions = [s.index(h) for h in order]
    assert positions == sorted(positions), order


def test_html_prompt_names_the_input_sections_spec_context_renders():
    """The old text pointed at `## Data gathered so far`, a heading only the
    data-phase nodes ever see; the generator's kickoff renders `## Data`."""
    s = _html_prompt()
    for heading in (
        "`## Brief`",
        "`## Data`",
        "`### Sources read from the web`",
        "`## Technical specification`",
        "`## Progress journal`",
        prompts.PRD_SECTION_FOOTER,
    ):
        assert heading in s, heading
    assert "Data gathered so far" not in s


def test_html_prompt_carries_no_leftovers_from_other_builders():
    """Absolute folder path, fullstack wording and the two-tool-era 'main
    agent' all came from shared constants the html-app prompt no longer uses."""
    s = _html_prompt()
    assert "/tmp/artifact-prompt-probe" not in s
    assert "fullstack" not in s.lower()
    assert "main agent" not in s
    # The old DATA INTO FILES block told the model to append the data as its
    # own part — contradicting the whole-file-in-one-body default.
    assert "DATA INTO FILES" not in s
    assert "as its own\n  part" not in s


def test_html_prompt_states_each_rule_once():
    s = _html_prompt()
    assert s.count('name="viewport"') == 1
    assert s.count("VERIFICATION IS NOT YOUR JOB") == 0
    assert "NOT part of the job" in s


def test_html_prompt_pins_ui_language_to_the_prd():
    assert "language of the PRD" in _html_prompt()


def test_html_prompt_quotes_the_markers_and_the_size_from_the_constants():
    from anton.core.tools.generate_artifact import sub_tools
    from anton.core.tools.generate_artifact.state import REPLY_BODY_CHARS

    s = _html_prompt()
    assert sub_tools.FILE_BEGIN_MARKER in s and sub_tools.FILE_END_MARKER in s
    assert f"{REPLY_BODY_CHARS:,} characters" in s
    assert s.index("single body") < s.index("Split ONLY")


def test_design_rules_do_not_restate_the_contract():
    """The two halves are quoted under their own headings by both frontend
    builders, so a rule must live in exactly one of them."""
    assert 'name="viewport"' not in prompts._DESIGN_RULES
    for system in (
        prompts.build_subagent_system_prompt("html-app", Path("/tmp/a")),
        prompts.build_frontend_system_prompt(Path("/tmp/a")),
    ):
        assert "## Verifier contract\n" in system
        assert "## Design rules\n" in system


# ── fullstack frontend prompt: same skeleton as html-app (2026-09-17) ───────
#
# Until then `build_frontend_system_prompt` stacked the shared `_ROLE` on top
# of its own blocks. Two live runs (13 and 14) showed the html-app structure
# holding — one body, no self-check — so the fullstack frontend now uses the
# same blocks, with the fullstack-only material in the task paragraph and a
# `## Fullstack rules` section.

def _fullstack_prompt() -> str:
    return prompts.build_frontend_system_prompt(Path("/tmp/artifact-prompt-probe"))


def test_fullstack_prompt_has_the_html_app_skeleton_plus_its_own_rules():
    s = _fullstack_prompt()
    order = ["## Your task", "## What you receive", "## Workflow",
             "## Output protocol", "## Verifier contract", "## Fullstack rules",
             "## Design rules", "## Tools"]
    positions = [s.index(h) for h in order]
    assert positions == sorted(positions), order
    assert s.count("static/index.html") >= 3  # task, workflow, output protocol


def test_fullstack_prompt_carries_no_leftovers_from_the_role_block():
    """Scratchpad discipline, the html-app data recipe, a second copy of the
    write protocol and the absolute folder path all came from `_ROLE`."""
    s = _fullstack_prompt()
    for gone in (
        "SCRATCHPAD DISCIPLINE",
        "DATA INTO FILES",
        "HOW MUCH GOES IN ONE REPLY",
        "## Output folder",
        "/tmp/artifact-prompt-probe",
        "main agent",
        "VERIFICATION IS NOT YOUR JOB",
        "the brief's",
    ):
        assert gone not in s, gone
    assert "NOT part of the job" in s
    assert s.count('name="viewport"') == 1
    assert s.count("Split ONLY") == 1


def test_fullstack_prompt_names_its_input_sections_and_the_api_spec():
    s = _fullstack_prompt()
    for heading in (
        "`## Data`",
        "`## Technical specification`",
        "`## Progress journal`",
        "`## API Specification`",
        prompts.PRD_SECTION_FOOTER,
    ):
        assert heading in s, heading
    assert "language of the PRD" in s
    # The two fullstack-only verifier failures are announced as such.
    assert 'name="api-base"' in s
    assert "fails the step" in s


def test_fullstack_prompt_and_kickoff_do_not_invite_a_data_step():
    """The page fetches from the backend; the old kickoff opened with "Use the
    scratchpad tool to inspect a data sample", an invitation to spend a round
    on data the page will never embed."""
    s = _fullstack_prompt()
    assert "Embed no data" in s
    kickoff = prompts.build_frontend_kickoff("## Product requirements\nx", "{}")
    assert "inspect a data sample" not in kickoff
    assert "follow the workflow from your instructions" in kickoff
    assert kickoff.index("## API Specification") < kickoff.index("follow the workflow")
