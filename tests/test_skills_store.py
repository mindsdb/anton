"""Storage-layer tests for `anton.core.memory.skills`."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from anton.core.memory.skills import (
    Skill,
    SkillStore,
    make_unique_label,
    slugify,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def store_root(tmp_path: Path) -> Path:
    return tmp_path / "skills"


@pytest.fixture()
def store(store_root: Path) -> SkillStore:
    return SkillStore(root=store_root)


def _make_skill(label: str = "csv-summary", **overrides) -> Skill:
    base = dict(
        label=label,
        name="CSV Summary",
        description="Load a CSV, infer schema, compute summary stats.",
        declarative_md="1. Load the CSV.\n2. Infer types.\n3. Print summary.\n",
        created_at="2026-04-10T12:00:00+00:00",
        provenance="manual",
    )
    base.update(overrides)
    return Skill(**base)


# ─────────────────────────────────────────────────────────────────────────────
# slugify / make_unique_label
# ─────────────────────────────────────────────────────────────────────────────


class TestSlugify:
    def test_simple_lowercase(self):
        assert slugify("CSV Summary") == "csv-summary"

    def test_strips_special_chars(self):
        assert slugify("Web Scraping (v2)!") == "web-scraping-v2"

    def test_collapses_hyphens(self):
        assert slugify("foo---bar") == "foo-bar"

    def test_trims_leading_trailing(self):
        assert slugify("--hello--") == "hello"

    def test_empty_input_falls_back(self):
        assert slugify("") == "skill"
        assert slugify("   ") == "skill"
        assert slugify("!@#$") == "skill"

    def test_hyphen_preserved(self):
        assert slugify("api-data-fetcher") == "api-data-fetcher"


class TestMakeUniqueLabel:
    def test_returns_base_when_unique(self, store: SkillStore):
        assert make_unique_label("Brand New", store) == "brand-new"

    def test_appends_number_on_collision(self, store: SkillStore):
        store.save(_make_skill(label="csv-summary"))
        assert make_unique_label("CSV Summary", store) == "csv-summary-2"

    def test_chains_numbers(self, store: SkillStore):
        store.save(_make_skill(label="csv-summary"))
        store.save(_make_skill(label="csv-summary-2"))
        store.save(_make_skill(label="csv-summary-3"))
        assert make_unique_label("CSV Summary", store) == "csv-summary-4"


# ─────────────────────────────────────────────────────────────────────────────
# Save / load round-trip
# ─────────────────────────────────────────────────────────────────────────────


class TestSaveLoadRoundtrip:
    def test_save_creates_directory_with_required_files(
        self, store: SkillStore, store_root: Path
    ):
        skill = _make_skill()
        path = store.save(skill)
        assert path == store_root / "csv-summary"
        assert (path / "SKILL.md").is_file()
        assert (path / "stats.json").is_file()

    def test_load_after_save_round_trip(self, store: SkillStore):
        original = _make_skill()
        store.save(original)
        loaded = store.load("csv-summary")
        assert loaded is not None
        assert loaded.label == original.label
        assert loaded.name == original.name
        assert loaded.description == original.description
        assert loaded.declarative_md == original.declarative_md
        assert loaded.provenance == "manual"

    def test_load_unknown_returns_none(self, store: SkillStore):
        assert store.load("does-not-exist") is None

    def test_load_with_corrupt_skill_md_returns_none(
        self, store: SkillStore, store_root: Path
    ):
        store.save(_make_skill())
        (store_root / "csv-summary" / "SKILL.md").write_text("not yaml: [\nbad")
        assert store.load("csv-summary") is None

    def test_save_does_not_wipe_existing_stats(
        self, store: SkillStore, store_root: Path
    ):
        # First save initializes stats.json with zeroes.
        store.save(_make_skill())
        # Simulate accumulated counters.
        store.increment_recommended("csv-summary", stage=1)
        store.increment_recommended("csv-summary", stage=1)
        # Saving again (e.g., editing the procedure) must NOT zero them.
        store.save(_make_skill(declarative_md="updated procedure"))
        loaded = store.load("csv-summary")
        assert loaded is not None
        assert loaded.stats.stage_1.recommended == 2
        assert loaded.declarative_md == "updated procedure"


# ─────────────────────────────────────────────────────────────────────────────
# list_all / list_summaries / delete
# ─────────────────────────────────────────────────────────────────────────────


class TestListing:
    def test_list_all_empty(self, store: SkillStore):
        assert store.list_all() == []

    def test_list_summaries_empty(self, store: SkillStore):
        assert store.list_summaries() == []

    def test_list_all_returns_sorted(self, store: SkillStore):
        store.save(_make_skill(label="zebra", name="Zebra"))
        store.save(_make_skill(label="alpha", name="Alpha"))
        store.save(_make_skill(label="mike", name="Mike"))
        labels = [s.label for s in store.list_all()]
        assert labels == ["alpha", "mike", "zebra"]

    def test_list_summaries_skips_malformed(
        self, store: SkillStore, store_root: Path
    ):
        store.save(_make_skill(label="good"))
        # Create a directory with no SKILL.md — should be skipped.
        (store_root / "broken").mkdir()
        summaries = store.list_summaries()
        assert [s["label"] for s in summaries] == ["good"]

    def test_list_summaries_lightweight_shape(self, store: SkillStore):
        store.save(_make_skill())
        summaries = store.list_summaries()
        assert len(summaries) == 1
        assert summaries[0] == {
            "label": "csv-summary",
            "name": "CSV Summary",
            "description": "Load a CSV, infer schema, compute summary stats.",
        }

    def test_list_summaries_parses_unchanged_frontmatter_once(
        self, store: SkillStore, yaml_parses: list[str]
    ):
        """The prompt lists skills every turn; an unchanged SKILL.md is parsed once."""
        store.save(_make_skill())
        first = store.list_summaries()
        second = store.list_summaries()

        assert first == second
        assert len(yaml_parses) == 1

    def test_list_summaries_sees_an_edited_frontmatter(self, store: SkillStore):
        store.save(_make_skill())
        assert store.list_summaries()[0]["description"].startswith("Load a CSV")

        store.save(_make_skill(description="Profile a CSV file."))
        assert store.list_summaries()[0]["description"] == "Profile a CSV file."

    def test_delete_removes_directory(self, store: SkillStore, store_root: Path):
        store.save(_make_skill())
        assert (store_root / "csv-summary").is_dir()
        assert store.delete("csv-summary") is True
        assert not (store_root / "csv-summary").exists()

    def test_delete_unknown_returns_false(self, store: SkillStore):
        assert store.delete("nope") is False


# ─────────────────────────────────────────────────────────────────────────────
# Stats increments
# ─────────────────────────────────────────────────────────────────────────────


class TestStatsIncrement:
    def test_increment_unknown_skill_is_noop(self, store: SkillStore):
        # Should not raise.
        store.increment_recommended("not-a-skill", stage=1)

    def test_increment_stage_1(self, store: SkillStore):
        store.save(_make_skill())
        store.increment_recommended("csv-summary", stage=1)
        loaded = store.load("csv-summary")
        assert loaded is not None
        assert loaded.stats.stage_1.recommended == 1
        assert loaded.stats.stage_1.last_used  # set to a timestamp
        assert loaded.stats.total_recalls == 1

    def test_increment_multiple_times(self, store: SkillStore):
        store.save(_make_skill())
        for _ in range(5):
            store.increment_recommended("csv-summary", stage=1)
        loaded = store.load("csv-summary")
        assert loaded is not None
        assert loaded.stats.stage_1.recommended == 5
        assert loaded.stats.total_recalls == 5

    def test_increment_per_stage_independent(self, store: SkillStore):
        store.save(_make_skill())
        store.increment_recommended("csv-summary", stage=1)
        store.increment_recommended("csv-summary", stage=1)
        store.increment_recommended("csv-summary", stage=2)
        loaded = store.load("csv-summary")
        assert loaded is not None
        assert loaded.stats.stage_1.recommended == 2
        assert loaded.stats.stage_2.recommended == 1
        assert loaded.stats.stage_3.recommended == 0
        assert loaded.stats.total_recalls == 3

    def test_invalid_stage_raises(self, store: SkillStore):
        store.save(_make_skill())
        with pytest.raises(ValueError):
            store.increment_recommended("csv-summary", stage=4)


# ─────────────────────────────────────────────────────────────────────────────
# closest_match — typo recovery
# ─────────────────────────────────────────────────────────────────────────────


class TestClosestMatch:
    def test_empty_store_returns_none(self, store: SkillStore):
        assert store.closest_match("anything") is None

    def test_exact_match(self, store: SkillStore):
        store.save(_make_skill(label="csv-summary"))
        assert store.closest_match("csv-summary") == "csv-summary"

    def test_typo_one_char(self, store: SkillStore):
        store.save(_make_skill(label="csv-summary"))
        assert store.closest_match("csv-sumary") == "csv-summary"

    def test_underscore_normalized_to_hyphen(self, store: SkillStore):
        store.save(_make_skill(label="web-scraping"))
        assert store.closest_match("web_scraping") == "web-scraping"

    def test_completely_unrelated_returns_none(self, store: SkillStore):
        store.save(_make_skill(label="csv-summary"))
        assert store.closest_match("xyzzy-quark") is None

    def test_picks_closer_of_two_candidates(self, store: SkillStore):
        store.save(_make_skill(label="csv-summary"))
        store.save(_make_skill(label="api-fetcher"))
        # Closer to csv-summary
        assert store.closest_match("csv-summery") == "csv-summary"
        # Closer to api-fetcher
        assert store.closest_match("api-fecher") == "api-fetcher"


# ─────────────────────────────────────────────────────────────────────────────
# Disk format sanity (catches accidental schema drift)
# ─────────────────────────────────────────────────────────────────────────────


class TestDiskFormat:
    def test_skill_md_has_expected_fields(self, store: SkillStore, store_root: Path):
        store.save(_make_skill())
        text = (store_root / "csv-summary" / "SKILL.md").read_text()
        assert text.startswith("---\n")
        assert "name: csv-summary" in text
        assert "description:" in text
        assert "display_name:" in text

    def test_stats_json_initial_shape(self, store: SkillStore, store_root: Path):
        store.save(_make_skill())
        stats = json.loads((store_root / "csv-summary" / "stats.json").read_text())
        assert stats["total_recalls"] == 0
        for stage_key in ("stage_1", "stage_2", "stage_3"):
            assert stage_key in stats
            assert stats[stage_key]["recommended"] == 0
            assert stats[stage_key]["used"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# applies_to: the mark that feeds a skill to artifact builds
# ─────────────────────────────────────────────────────────────────────────────


def _write_skill_md(store_root: Path, label: str, frontmatter: str) -> None:
    d = store_root / label
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {label}\ndescription: Chart rules.\n{frontmatter}---\nUse only Chart.js.\n",
        encoding="utf-8",
    )


class TestAppliesTo:
    @pytest.mark.parametrize(
        "frontmatter",
        [
            "applies_to: artifacts\n",
            "metadata:\n  applies_to: artifacts\n",
            "applies_to: [chat, artifacts]\n",
            "applies_to: chat, artifacts\n",
            "applies_to: Artifacts\n",
        ],
    )
    def test_a_marked_skill_is_for_artifacts(
        self, store: SkillStore, store_root: Path, frontmatter: str
    ):
        _write_skill_md(store_root, "chart-rules", frontmatter)
        skill = store.load("chart-rules")
        assert skill is not None
        assert skill.for_artifacts is True

    @pytest.mark.parametrize(
        "frontmatter", ["", "applies_to: chat\n", "applies_to: artifacts-preview\n"]
    )
    def test_an_unmarked_skill_is_not_for_artifacts(
        self, store: SkillStore, store_root: Path, frontmatter: str
    ):
        _write_skill_md(store_root, "chart-rules", frontmatter)
        skill = store.load("chart-rules")
        assert skill is not None
        assert skill.for_artifacts is False

    def test_save_keeps_the_mark(self, store: SkillStore):
        store.save(_make_skill(applies_to="artifacts"))
        loaded = store.load("csv-summary")
        assert loaded is not None
        assert loaded.applies_to == "artifacts"
        assert loaded.for_artifacts is True


class TestListForArtifacts:
    """Which marked skills an artifact build follows without a recall."""

    @pytest.fixture()
    def builtins(self, tmp_path: Path) -> Path:
        root = tmp_path / "builtins"
        _write_skill_md(root, "builtin-marked", "applies_to: artifacts\n")
        return root

    def test_a_cli_store_follows_its_own_marked_skills(self, tmp_path: Path, builtins: Path):
        root = tmp_path / "user"
        _write_skill_md(root, "mine-marked", "applies_to: artifacts\n")
        _write_skill_md(root, "mine-plain", "")
        store = SkillStore(root=root, builtin_root=builtins)

        assert [s.label for s in store.list_for_artifacts()] == ["builtin-marked", "mine-marked"]

    def test_a_cli_store_follows_a_skill_linked_into_its_library(
        self, tmp_path: Path, builtins: Path
    ):
        """A CLI user may keep skills in a dotfiles repo and link them in;
        nobody else writes that library."""
        elsewhere = tmp_path / "dotfiles"
        _write_skill_md(elsewhere, "linked-marked", "applies_to: artifacts\n")
        root = tmp_path / "user"
        root.mkdir()
        (root / "linked-marked").symlink_to(elsewhere / "linked-marked", target_is_directory=True)
        store = SkillStore(root=root, builtin_root=builtins)

        assert [s.label for s in store.list_for_artifacts()] == ["builtin-marked", "linked-marked"]

    def test_a_host_store_follows_what_it_trusted_when_built(self, tmp_path: Path, builtins: Path):
        """The host names its folders by path when the turn starts. A skill
        deleted during the turn and made again under its name by someone
        else is not what the build follows."""
        own_store = tmp_path / "own"
        _write_skill_md(own_store, "own-marked", "applies_to: artifacts\n")
        turn = tmp_path / "turn"
        turn.mkdir()
        (turn / "own-marked").symlink_to(own_store / "own-marked", target_is_directory=True)
        store = SkillStore(
            root=turn, builtin_root=builtins, extra_roots=[tmp_path / "managed"],
            artifact_roots=[own_store / "own-marked"],
        )

        shutil.rmtree(own_store / "own-marked")
        (own_store / "own-marked").mkdir()
        (own_store / "own-marked" / "SKILL.md").write_text(
            "---\nname: own-marked\ndescription: d\napplies_to: artifacts\n---\nSOMEONE-ELSES\n",
            encoding="utf-8",
        )

        followed = {s.label: s.declarative_md for s in store.list_for_artifacts()}
        assert followed["own-marked"].strip() == "Use only Chart.js."

    def test_a_host_store_with_no_named_roots_follows_only_the_builtins(
        self, tmp_path: Path, builtins: Path
    ):
        root, shared = tmp_path / "turn", tmp_path / "shared"
        _write_skill_md(root, "writable-marked", "applies_to: artifacts\n")
        _write_skill_md(shared, "shared-marked", "applies_to: artifacts\n")
        store = SkillStore(root=root, builtin_root=builtins, extra_roots=[shared])

        assert [s.label for s in store.list_for_artifacts()] == ["builtin-marked"]

    def test_only_folders_the_host_names_are_followed(self, tmp_path: Path, builtins: Path):
        """The turn's root links the asker's own skills; managed and shared
        skills come as host roots; the host names its own store and the
        managed folder only."""
        own_store, managed, shared = tmp_path / "own", tmp_path / "managed", tmp_path / "shared"
        _write_skill_md(own_store, "own-marked", "applies_to: artifacts\n")
        _write_skill_md(managed, "managed-marked", "applies_to: artifacts\n")
        _write_skill_md(shared, "shared-marked", "applies_to: artifacts\n")
        turn = tmp_path / "turn"
        turn.mkdir()
        (turn / "own-marked").symlink_to(own_store / "own-marked", target_is_directory=True)
        store = SkillStore(
            root=turn, builtin_root=builtins, extra_roots=[managed, shared],
            artifact_roots=[own_store, managed],
        )

        assert [s.label for s in store.list_for_artifacts()] == [
            "builtin-marked", "managed-marked", "own-marked",
        ]

    def test_a_real_folder_planted_in_the_turn_root_is_not_followed(
        self, tmp_path: Path, builtins: Path
    ):
        """Agent code can write the turn's root; only the host's links into a
        named folder count."""
        own_store = tmp_path / "own"
        _write_skill_md(own_store, "own-marked", "applies_to: artifacts\n")
        turn = tmp_path / "turn"
        _write_skill_md(turn, "planted", "applies_to: artifacts\n")
        store = SkillStore(
            root=turn, builtin_root=builtins, extra_roots=[tmp_path / "managed"],
            artifact_roots=[own_store],
        )

        assert [s.label for s in store.list_for_artifacts()] == ["builtin-marked"]

    def test_a_link_into_an_unnamed_folder_is_not_followed(self, tmp_path: Path, builtins: Path):
        other_store = tmp_path / "someone-else"
        _write_skill_md(other_store, "theirs", "applies_to: artifacts\n")
        turn = tmp_path / "turn"
        turn.mkdir()
        (turn / "theirs").symlink_to(other_store / "theirs", target_is_directory=True)
        store = SkillStore(
            root=turn, builtin_root=builtins, extra_roots=[tmp_path / "managed"],
            artifact_roots=[tmp_path / "own"],
        )

        assert [s.label for s in store.list_for_artifacts()] == ["builtin-marked"]
