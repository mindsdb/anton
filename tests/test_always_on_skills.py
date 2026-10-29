"""Org and team skills cowork marks always-on reach every turn's prompt (ENG-3787)."""

from __future__ import annotations

from anton.cloud_turn import session as cloud_session
from anton.core.memory import (
    ALWAYS_ON_SKILL_MAX_CHARS,
    ALWAYS_ON_TOTAL_MAX_CHARS,
    render_always_on_skills,
)
from tests.test_cloud_turn_session import _build

_HEADER = (
    "\n\n## Organization and team skills (always apply)\n"
    "These are how this organization works. Follow them on every task where "
    "they apply; they override general defaults but never the user's explicit "
    "instructions in this conversation."
)
_TRUNCATED = "\n[…truncated; the full skill is available on demand]"


def _skill_md(name: str, body: str) -> str:
    return f"---\nname: {name}\ndescription: d\n---\n{body}"


def test_format_is_exactly_the_contract():
    out = render_always_on_skills([
        {"name": "tone", "body": "Be brief."},
        {"name": "sql-style", "body": "Uppercase keywords.\nNo SELECT *."},
    ])
    assert out == (
        _HEADER
        + "\n\n### tone\nBe brief."
        + "\n\n### sql-style\nUppercase keywords.\nNo SELECT *."
    )


def test_no_skills_renders_nothing():
    assert render_always_on_skills([]) == ""


def test_a_long_skill_is_cut_at_a_line_boundary():
    line = "x" * 99
    body = "\n".join([line] * 100)  # 9999 chars, 100-char lines with the newline
    out = render_always_on_skills([{"name": "long", "body": body}])
    kept = out.split("### long\n", 1)[1]
    assert kept.endswith(_TRUNCATED)
    kept = kept.removesuffix(_TRUNCATED)
    assert len(kept) <= ALWAYS_ON_SKILL_MAX_CHARS
    assert kept == "\n".join([line] * 60)


def test_a_single_overlong_line_is_cut_at_the_cap():
    out = render_always_on_skills([{"name": "blob", "body": "y" * 7000}])
    assert out.endswith("### blob\n" + "y" * ALWAYS_ON_SKILL_MAX_CHARS + _TRUNCATED)


def test_skills_past_the_total_are_named_on_a_closing_line():
    full = "z" * ALWAYS_ON_SKILL_MAX_CHARS
    skills = [{"name": f"s{i}", "body": full} for i in range(4)]
    skills += [{"name": "late", "body": "short"}, {"name": "later", "body": "short"}]
    out = render_always_on_skills(skills)
    assert 4 * ALWAYS_ON_SKILL_MAX_CHARS == ALWAYS_ON_TOTAL_MAX_CHARS
    assert all(f"### s{i}\n" in out for i in range(4))
    # Past the total, a skill is omitted even when it is small.
    assert "### late" not in out and "### later" not in out
    assert out.endswith("\n\n(Also available on demand: late, later.)")


def test_only_always_on_skills_are_rendered():
    skills = {
        "tone": {"files": {"SKILL.md": _skill_md("tone", "\nBe brief.\n")}, "always_on": True},
        "recall-only": {"files": {"SKILL.md": _skill_md("recall-only", "Nope.")}},
        "off": {"files": {"SKILL.md": _skill_md("off", "Nope.")}, "always_on": False},
    }
    assert cloud_session._always_on_skills(skills) == [{"name": "tone", "body": "Be brief."}]


def test_name_falls_back_to_the_slug():
    skills = {"my-rules": {"files": {"SKILL.md": "---\ndescription: d\n---\nRule."}, "always_on": True}}
    assert cloud_session._always_on_skills(skills) == [{"name": "my-rules", "body": "Rule."}]


def test_malformed_entries_are_skipped():
    skills = {
        "Bad Slug": {"files": {"SKILL.md": _skill_md("x", "x")}, "always_on": True},
        "no-files": {"always_on": True},
        "no-skill-md": {"files": {"other.md": "x"}, "always_on": True},
        "not-text": {"files": {"SKILL.md": 3}, "always_on": True},
        "bad-yaml": {"files": {"SKILL.md": "---\n: [\n---\nbody"}, "always_on": True},
        "empty": {"files": {"SKILL.md": _skill_md("empty", "  \n")}, "always_on": True},
        "not-a-dict": ["x"],
        "good": {"files": {"SKILL.md": _skill_md("good", "Body.")}, "always_on": True},
    }
    assert cloud_session._always_on_skills(skills) == [{"name": "good", "body": "Body."}]
    assert cloud_session._always_on_skills(None) == []


def test_pod_suffix_carries_always_on_skills_and_still_stages_them(tmp_path, monkeypatch):
    monkeypatch.delenv(cloud_session._SKILLS_ROOT_ENV, raising=False)
    skills = {
        "tone": {"files": {"SKILL.md": _skill_md("tone", "Be brief.")}, "always_on": True},
        "recall-only": {"files": {"SKILL.md": _skill_md("recall-only", "Nope.")}},
    }
    _, cfg = _build(tmp_path, monkeypatch, skills=skills)
    assert cfg.system_prompt_context.suffix == (
        cloud_session._POD_PROMPT_SUFFIX + _HEADER + "\n\n### tone\nBe brief."
    )
    # Always-on skills stay recallable: _stage_skills still writes them.
    staged = cloud_session._stage_skills(skills)
    assert (staged / "tone" / "SKILL.md").is_file()


def test_pod_suffix_is_unchanged_without_always_on_skills(tmp_path, monkeypatch):
    monkeypatch.delenv(cloud_session._SKILLS_ROOT_ENV, raising=False)
    skills = {"recall-only": {"files": {"SKILL.md": _skill_md("recall-only", "Nope.")}}}
    _, cfg = _build(tmp_path, monkeypatch, skills=skills)
    assert cfg.system_prompt_context.suffix == cloud_session._POD_PROMPT_SUFFIX
