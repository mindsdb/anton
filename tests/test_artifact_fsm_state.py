"""Unit coverage for the artifact-generation FSM state module."""
from __future__ import annotations

from pathlib import Path

from anton.core.tools.generate_artifact.state import (
    DATA_LOOP_MAX,
    GEN_VERIFY_MAX_RETRIES,
    RUNAPP_MAX_RETRIES,
    DataVerdict,
    FetchVerdict,
    GenState,
    RequiredData,
    StepResult,
    VerifyResult,
)


def test_budgets_match_spec():
    assert DATA_LOOP_MAX == 3
    assert GEN_VERIFY_MAX_RETRIES == 1
    assert RUNAPP_MAX_RETRIES == 1


def test_verdict_models_validate():
    dv = DataVerdict(enough=True, reasoning="no data needed")
    assert dv.enough is True
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
