"""State-store guidance (ENG-704).

The backend + STATE contract was moved out of BACKEND_GENERATION_PROMPT and into
the `build-fullstack-backend` built-in skill (recalled on demand). These tests
assert the STATE guidance survives in the skill body, which is where an agent
actually reads it.
"""

from pathlib import Path

import pytest

from anton.core.memory import skills as skills_mod
from anton.core.memory.skills import SkillStore

REAL_BUILTIN_ROOT = Path(skills_mod.__file__).parent / "builtin_skills"


@pytest.fixture(scope="module")
def T() -> str:
    """The build-fullstack-backend skill body (frontmatter stripped)."""
    store = SkillStore(root=REAL_BUILTIN_ROOT.parent / "_no_user_skills", builtin_root=REAL_BUILTIN_ROOT)
    skill = store.load("build-fullstack-backend")
    assert skill is not None and skill.provenance == "builtin"
    return skill.declarative_md


def test_template_declares_state_slot(T):
    assert "STATE = None" in T


def test_template_shows_open_store_usage(T):
    assert "from anton_state import open_store" in T
    assert "state_manifest.json" in T
    assert "Path(__file__).resolve().parent" in T


def test_rules_position_state_for_light_and_external_db_for_heavy(T):
    low = T.lower()
    assert "state_manifest.json" in T
    # light state → STATE; heavy/relational → external DB
    assert "external" in low and "relational" in low


# --- shared-table API guidance (ENG-704) ---
def test_state_overlay_is_url_token_not_creds(T):
    # cloud overlay is now {url, token}, not the old STS {table, region, credentials}
    assert "url, token" in T
    assert "table, region" not in T


def test_store_api_mentions_collection_and_atomics(T):
    assert "Collection" in T
    assert "increment" in T
    assert "update" in T


def test_store_api_drops_gsi_index_kwarg(T):
    # the query() signature no longer offers an index= argument in v1
    assert "index=None" not in T


def test_rules_warn_against_manual_mutation_retries(T):
    assert "retry" in T.lower()
