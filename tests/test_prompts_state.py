from anton.core.llm import prompts

T = prompts.BACKEND_GENERATION_PROMPT


def test_template_declares_state_slot():
    assert "STATE = None" in T


def test_template_shows_open_store_usage():
    assert "from anton_state import open_store" in T
    assert "state_manifest.json" in T
    assert "Path(__file__).resolve().parent" in T


def test_rules_position_state_for_light_and_external_db_for_heavy():
    combined = prompts.BACKEND_GENERATION_PROMPT + prompts.ARTIFACTS_PROMPT
    low = combined.lower()
    assert "state_manifest.json" in combined
    # light state → STATE; heavy/relational → external DB
    assert "external" in low and "relational" in low
