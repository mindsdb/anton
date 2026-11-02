import logging

import pytest

from anton.cloud_turn import session as cloud_session
from anton.cloud_turn.contract import parse_turn_skills
from tests.test_cloud_turn_session import _real_cloud_session


def _skill_md(name: str, body: str) -> str:
    return f"---\nname: {name}\ndescription: {name} procedure\n---\n{body}\n"


@pytest.fixture()
def org_mount(tmp_path_factory, monkeypatch, skills_tmp):
    """The org skills tree mounted in the pod, holding a skill the caller was
    not given (another member's private skill), with per-turn staging routed
    into pytest tmp."""
    mount = tmp_path_factory.mktemp("mnt") / "skills"
    (mount / "someone-elses").mkdir(parents=True)
    (mount / "someone-elses" / "SKILL.md").write_text(_skill_md("someone-elses", "Secret steps."))
    monkeypatch.setenv("ANTON_CLOUD_SKILLS_ROOT", str(mount))
    return mount


_SHARED = {"shared-with-me": {"files": {"SKILL.md": _skill_md("shared-with-me", "Allowed steps.")}}}


def test_skills_read_from_mount_only_without_a_skills_block(monkeypatch, tmp_path, caplog):
    """The rollout fallback: a request with no skills block, on a pod that
    has the mount, reads the mount, with a warning."""
    mount = tmp_path / "mnt" / "skills"
    (mount / "my-skill").mkdir(parents=True)
    (mount / "my-skill" / "SKILL.md").write_text("hello")
    monkeypatch.setenv("ANTON_CLOUD_SKILLS_ROOT", str(mount))

    with caplog.at_level(logging.WARNING, logger=cloud_session.logger.name):
        root = cloud_session._stage_skills(None)

    assert root == mount
    assert (root / "my-skill" / "SKILL.md").read_text() == "hello"
    warnings = [
        r for r in caplog.records
        if r.name == cloud_session.logger.name and r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1 and "no skills block" in warnings[0].getMessage()


def test_a_skills_block_replaces_the_mount(org_mount):
    root = cloud_session._stage_skills(parse_turn_skills(_SHARED))

    assert root != org_mount
    assert root.name.startswith(cloud_session._SKILLS_DIR_PREFIX)
    assert (root / "shared-with-me" / "SKILL.md").is_file()
    assert not (root / "someone-elses").exists()


def test_an_empty_skills_block_means_no_skills(org_mount):
    root = cloud_session._stage_skills(parse_turn_skills({}))
    assert root != org_mount
    assert list(root.iterdir()) == []


async def test_recall_sees_only_the_requests_skills(tmp_path, monkeypatch, org_mount):
    from anton.core.tools.recall_skill import handle_recall_skill

    session = _real_cloud_session(tmp_path, monkeypatch, skills=_SHARED)

    allowed = await handle_recall_skill(session, {"label": "shared-with-me"})
    assert allowed.ok is True and "Allowed steps." in allowed.content

    denied = await handle_recall_skill(session, {"label": "someone-elses"})
    text = denied if isinstance(denied, str) else denied.content
    assert "Secret steps." not in text
    assert text.count("someone-elses") == 1  # only echoing the label asked for

    store = session._skill_store
    labels = {s["label"] for s in store.list_summaries()}
    assert "shared-with-me" in labels and "someone-elses" not in labels
    assert store.load("someone-elses") is None
    assert store.dir_for("someone-elses") is None  # nor seedable as a draft


async def test_prompt_lists_only_the_requests_skills(tmp_path, monkeypatch, org_mount):
    session = _real_cloud_session(tmp_path, monkeypatch, skills=_SHARED)
    prompt = await session._build_system_prompt()
    assert "shared-with-me" in prompt
    assert "someone-elses" not in prompt


@pytest.mark.parametrize("block", ["x", ["shared-with-me"], 7])
async def test_a_malformed_skills_block_never_reads_the_mount(tmp_path, monkeypatch, org_mount, block):
    """Only an absent block takes the fallback, and the mount holds every
    member's skills, so a block that is present but not an object names none."""
    session = _real_cloud_session(tmp_path, monkeypatch, skills=block)
    store = session._skill_store
    assert store.root != org_mount
    assert store.load("someone-elses") is None
    assert "someone-elses" not in {s["label"] for s in store.list_summaries()}


async def test_without_a_skills_block_the_mount_is_still_read(tmp_path, monkeypatch, org_mount):
    from anton.core.tools.recall_skill import handle_recall_skill

    session = _real_cloud_session(tmp_path, monkeypatch)
    out = await handle_recall_skill(session, {"label": "someone-elses"})
    assert out.ok is True and "Secret steps." in out.content


def test_skills_fall_back_to_tmp_staging_without_env(monkeypatch, tmp_path):
    """Without the mount (desktop, CI), the request's skills stage into a
    per-turn folder."""
    monkeypatch.delenv("ANTON_CLOUD_SKILLS_ROOT", raising=False)

    root = cloud_session._stage_skills(
        parse_turn_skills({"my-skill": {"files": {"SKILL.md": "x"}}})
    )

    assert root.name.startswith(cloud_session._SKILLS_DIR_PREFIX)
    assert (root / "my-skill" / "SKILL.md").read_text() == "x"


def test_memory_read_from_mount_when_env_set(monkeypatch, tmp_path):
    mount = tmp_path / "mnt" / "memory" / "users" / "u1"
    mount.mkdir(parents=True)
    (mount / "profile.md").write_text("from efs")
    monkeypatch.setenv("ANTON_CLOUD_MEMORY_GLOBAL_ROOT", str(mount))

    global_dir = cloud_session._global_memory_dir({"global": {"profile": "from wire"}})

    assert global_dir == mount
    assert (global_dir / "profile.md").read_text() == "from efs"   # wire did not overwrite


def test_memory_falls_back_to_tmp_staging_without_env(monkeypatch):
    monkeypatch.delenv("ANTON_CLOUD_MEMORY_GLOBAL_ROOT", raising=False)

    global_dir = cloud_session._global_memory_dir({"global": {"profile": "from wire"}})

    assert (global_dir / "profile.md").read_text() == "from wire"
