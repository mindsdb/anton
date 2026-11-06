"""Unit coverage for the artifact-generation FSM state module."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from anton.core.tools.generate_artifact.state import (
    DATA_LOOP_MAX,
    GEN_VERIFY_MAX_RETRIES,
    JOURNAL_DETAIL_MAX,
    RUNAPP_MAX_RETRIES,
    FetchVerdict,
    GenState,
    RequiredData,
    StepResult,
    VerifyResult,
    known_data_is_reference,
    known_data_usable,
)


def test_budgets_match_spec():
    assert DATA_LOOP_MAX == 3
    assert GEN_VERIFY_MAX_RETRIES == 1
    assert RUNAPP_MAX_RETRIES == 1


def test_verdict_models_validate():
    rd = RequiredData(items=[{"name": "orders", "where": "postgres", "why": "chart"}], reasoning="r")
    assert rd.items[0].name == "orders"
    fv = FetchVerdict(possible=False, reasoning="no source")
    assert fv.possible is False


def test_verify_result_ok_is_derived():
    assert VerifyResult(errors=[], warnings=["w"]).ok is True
    assert VerifyResult(errors=["e"], warnings=[]).ok is False


def test_genstate_defaults():
    st = GenState(
        session=object(),
        artifact_type="html-app",
        artifact_path=Path("/tmp/x"),
        slug="x",
        brief="b",
        is_fullstack=False,
    )
    assert st.data_iterations == 0
    assert st.data_notes == ""
    assert st.api_spec is None
    assert st.files_written == []
    assert st.trace == []
    assert st.error is None
    st.trace.append(StepResult(node="is_data_enough", outcome="yes"))
    assert st.trace[0].node == "is_data_enough"


def test_genstate_journal_is_compact_and_capped():
    st = GenState(
        session=object(),
        artifact_type="html-app",
        artifact_path=Path("/tmp/x"),
        slug="x",
        brief="b",
        is_fullstack=False,
    )
    assert st.journal() == ""
    st.record("is_data_enough", "no", "need orders\nand users")
    st.record("make_tech_spec", "done")
    st.record("fetch_data_sample", "done", "x" * (JOURNAL_DETAIL_MAX + 100))
    lines = st.journal().splitlines()
    assert lines[0] == "- is_data_enough: no — need orders and users"
    assert lines[1] == "- make_tech_spec: done"
    assert lines[2].startswith("- fetch_data_sample: done — ")
    assert lines[2].endswith("…")
    assert len(lines[2]) < JOURNAL_DETAIL_MAX + 50


def test_genstate_carries_the_discovery_fields():
    """One state object for all five phases: the shared message list cannot
    live in a second dataclass without being threaded across a boundary."""
    from pathlib import Path
    from types import SimpleNamespace

    from anton.core.tools.generate_artifact.state import GenState

    state = GenState(
        session=SimpleNamespace(),
        artifact_type="html-app",
        artifact_path=Path("/tmp/x"),
        slug="x",
        brief="",
        is_fullstack=False,
        user_request="build a dashboard",
        agent_understanding="a dashboard from an article",
    )
    assert state.messages == []
    assert state.qa_log == []
    assert state.declared_sources == []
    assert state.unverified_sources == []
    assert state.scratchpad_execs == []
    assert state.web_calls == []
    assert state.web_notes == ""
    assert state.gathering_complete is False
    assert state.final_artifact_type == ""
    assert state.call_changed is False
    assert state.spend is None

    state.record_qa("Which range?", "weekly")
    assert "weekly" in state.qa_log_markdown()


def test_winding_down_tolerates_a_state_built_without_a_guard():
    """`spend` is None for the bench harness and for hand-built test states,
    and neither should acquire budget behaviour just by existing."""
    from pathlib import Path
    from types import SimpleNamespace

    from anton.core.tools.generate_artifact.state import GenState

    state = GenState(
        session=SimpleNamespace(),
        artifact_type="html-app",
        artifact_path=Path("/tmp/x"),
        slug="x",
        brief="",
        is_fullstack=False,
    )
    assert state.winding_down() is False


def test_question_budget_reserves_slots_for_the_brief_phase():
    from types import SimpleNamespace

    from anton.core.interaction.elicit import MAX_QUESTIONS_PER_TURN
    from anton.core.tools.generate_artifact.state import (
        PHASE2_RESERVED_QUESTIONS,
        gathering_question_budget,
    )

    # Against the imported constants, not a hardcoded 8 - 3: a hardcoded
    # number here would silently stop reflecting reality the next time
    # MAX_QUESTIONS_PER_TURN changes.
    session = SimpleNamespace(question_count=0)
    assert gathering_question_budget(session) == (
        MAX_QUESTIONS_PER_TURN - PHASE2_RESERVED_QUESTIONS
    )
    assert PHASE2_RESERVED_QUESTIONS == 3

    session.question_count = 2
    assert gathering_question_budget(session) == (
        MAX_QUESTIONS_PER_TURN - 2 - PHASE2_RESERVED_QUESTIONS
    )

    session.question_count = MAX_QUESTIONS_PER_TURN
    assert gathering_question_budget(session) == 0


def test_is_fullstack_is_derived_from_the_artifact_type():
    """One fact, one field. A state whose type and flag disagree would send an
    html-app down the fullstack branch with nothing reporting it — the same
    class of silent drift the internal-file constants exist to prevent."""
    from pathlib import Path
    from types import SimpleNamespace

    from anton.core.tools.generate_artifact.state import GenState

    def _st(artifact_type: str, **over) -> GenState:
        return GenState(
            session=SimpleNamespace(),
            artifact_type=artifact_type,
            artifact_path=Path("/tmp/x"),
            slug="x",
            **over,
        )

    assert _st("html-app").is_fullstack is False
    assert _st("fullstack-stateless-app").is_fullstack is True
    assert _st("fullstack-stateful-app").is_fullstack is True
    # An explicit value still wins — existing call sites pass one.
    assert _st("html-app", is_fullstack=True).is_fullstack is True


def _gs(**kw) -> GenState:
    base = dict(session=object(), artifact_type="html-app", artifact_path=Path("/tmp/a"), slug="a")
    base.update(kw)
    return GenState(**base)


def test_known_data_stands_in_for_code_only_for_an_html_app_snapshot():
    assert known_data_usable(_gs(known_data="site,lat\nToronto,43.7")) is True
    assert known_data_usable(_gs(known_data="   ")) is False
    assert known_data_usable(_gs(known_data="x" * 50_000)) is True
    assert known_data_usable(
        _gs(artifact_type="fullstack-stateless-app", known_data="site,lat\nToronto,43.7")
    ) is False


def test_known_data_usable_reads_the_type_gathering_settled_on():
    st = _gs(known_data="site,lat\nToronto,43.7")
    st.final_artifact_type = "fullstack-stateful-app"
    assert known_data_usable(st) is False


def test_the_required_data_schema_says_an_empty_list_means_nothing_is_missing():
    schema = RequiredData.model_json_schema()
    assert "Empty when nothing is missing" in schema["properties"]["items"]["description"]


def _ref_state(known_data, pads=("supply_kpi", "main")):
    return GenState(
        session=SimpleNamespace(), artifact_type="html-app", artifact_path=Path("/tmp/x"),
        slug="s", known_data=known_data, pad_names=list(pads),
    )


@pytest.mark.parametrize("text, expected", [
    ("scratchpad `supply_kpi`, variable `frames` (dict of DataFrames)", True),
    ("data in scratchpad supply_kpi, variable frames", True),
    ("in pad `supply_kpi`", True),
    ("Live supply data metrics from scratchpad supply_kpi: Base Plan = 169 lines", True),
    ("the main KPI is on-time share", False),
    ("Base Plan = 169 lines, Supplier Delay = 40 late", False),
    ("scratchpad supply_kpi_old, variable x", False),
    # A pad this session does not list is still a reference: its cell fails
    # and the "not enough data" path takes over, instead of the text passing
    # as values.
    ("scratchpad `typo_pad`, variable `df` (DataFrame)", True),
])
def test_reference_detection(text, expected):
    assert known_data_is_reference(_ref_state(text)) is expected


def test_a_reference_cannot_stand_in_for_executed_code():
    assert known_data_usable(_ref_state("Base Plan = 169 lines")) is True
    assert known_data_usable(_ref_state("scratchpad `supply_kpi`, variable `frames`")) is False
