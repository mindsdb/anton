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
