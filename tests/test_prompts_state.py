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


# --- shared-table API guidance (ENG-704) ---
def test_state_overlay_is_url_token_not_creds():
    # cloud overlay is now {url, token}, not the old STS {table, region, credentials}
    assert "url, token" in T
    assert "table, region" not in T


def test_store_api_mentions_collection_and_atomics():
    assert "Collection" in T
    assert "increment" in T
    assert "update" in T


def test_store_api_drops_gsi_index_kwarg():
    # the query() signature no longer offers an index= argument in v1
    assert "index=None" not in T


def test_rules_warn_against_manual_mutation_retries():
    assert "retry" in T.lower()
