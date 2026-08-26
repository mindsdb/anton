from __future__ import annotations

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
    system, user = prompts.build_data_enough_prompt(st)
    assert "digraph" in system
    assert "Build X" in user
    assert "pad `a`" in user


def test_tech_spec_prompt_targets_spec_md():
    system, user = prompts.build_tech_spec_prompt(_state())
    assert "spec.md" in system or "spec.md" in user


def test_tech_spec_prompt_pins_the_stack():
    system, _ = prompts.build_tech_spec_prompt(_state())
    assert "FastAPI" in system
    assert "Python >= 3.12" in system
    assert "/api/*" in system
    assert "never mention a port number" in system
    # The generator rules must state the target runtime too.
    assert "Python >= 3.12" in prompts._BACKEND_RULES


def test_fetch_data_prompts_exist():
    assert isinstance(prompts.build_fetch_data_system_prompt(Path("/tmp/a")), str)
    assert "scratchpad" in prompts.build_fetch_data_kickoff(_state()).lower()


def test_prompts_include_progress_journal():
    st = _state()
    _, user = prompts.build_data_enough_prompt(st)
    assert "## Progress journal" not in user  # empty journal → no section
    st.record("is_data_enough", "no", "need orders")
    _, user = prompts.build_data_enough_prompt(st)
    assert "## Progress journal" in user
    assert "- is_data_enough: no — need orders" in user


def test_backend_prompt_states_the_ds_env_var_convention():
    """Without the naming convention the generator guesses keys and fails _map_datasources."""
    for stateless in (True, False):
        system = prompts.build_backend_system_prompt(Path("/tmp/a"), stateless=stateless)
        assert "DS_<ENGINE>_<NAME>__<FIELD>" in system


def test_backend_prompt_embeds_the_datasource_catalog():
    catalog = "\n\n## Connected Data Sources\n- `postgres-prod_db` (postgres) → DS_POSTGRES_PROD_DB__HOST"
    system = prompts.build_backend_system_prompt(
        Path("/tmp/a"), stateless=True, datasource_context=catalog
    )
    assert "DS_POSTGRES_PROD_DB__HOST" in system


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


def test_visual_rules_carry_the_frontend_verifier_contract():
    """_VISUAL_RULES is shared by html-app and the fullstack frontend — the rules go there."""
    rules = prompts._VISUAL_RULES
    for marker in (
        "explicit `<body>`",
        "absolute URL",
        "resource reference",
        "__antonCommentsLayer",
        "!important",
        "z-index",
        "stable `id`",
    ):
        assert marker in rules, marker


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
    """Otherwise the model writes dashboard.html while metadata promises another name."""
    system = prompts.build_subagent_system_prompt(
        "html-app", Path("/tmp/a"), primary="report.html"
    )
    assert "report.html" in system


def test_html_prompt_falls_back_to_dashboard_html():
    """primary is optional (Artifact.primary: str | None)."""
    for primary in (None, ""):
        system = prompts.build_subagent_system_prompt(
            "html-app", Path("/tmp/a"), primary=primary
        )
        assert "dashboard.html" in system


def test_write_discipline_block_is_present_in_both_frontend_prompts():
    """Chunked writing is needed by html-app and the fullstack frontend alike."""
    assert "mode=\"a\"" in prompts._WRITE_DISCIPLINE
    for system in (
        prompts.build_subagent_system_prompt("html-app", Path("/tmp/a")),
        prompts.build_frontend_system_prompt(Path("/tmp/a")),
    ):
        assert "mode=\"a\"" in system
        assert "chunk" in system.lower()


def test_role_no_longer_forbids_splitting_a_file():
    """HARD RULES outweigh any block below it — the old rule has to go."""
    role = prompts._ROLE
    assert "Do NOT split a single file across multiple calls" not in role
    assert "exactly once per file" not in role


def test_role_documents_the_mode_argument():
    assert "write_file(path, content, mode" in prompts._ROLE


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


def test_data_enough_prompt_counts_inspected_cells_as_data():
    """Otherwise the inspection does not affect the verdict and task 1 is pointless."""
    st = _state(data_notes="### Cells the main agent already ran in: orders")
    system, _ = prompts.build_data_enough_prompt(st)
    low = system.lower()
    # `already` is unusable as an assert here — the current prompt already contains
    # "ALREADY enough data", so the test would have been green before the change.
    assert "already available" in low
    assert "regardless of who obtained it" in low


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
