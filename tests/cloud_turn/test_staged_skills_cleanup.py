"""A cloud turn's staged skills folder is gone once the turn ends (ENG-3787).

A pod lives across many turns, so a folder left behind keeps the skills an
earlier request carried readable from later turns' cells, after the sharer may
have taken them back. The turns here run through the real entrypoint and the
real session builder, with a stub turn body that records the staged folders
it sees.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

import anton.cloud_turn.__main__ as entry
import anton.core.llm.client as llm_client_mod
from anton.cloud_turn import session as cloud_session
from tests.test_cloud_turn_session import _mock_llm

PREFIX = cloud_session._SKILLS_DIR_PREFIX
SKILLS = {
    "team-steps": {
        "files": {"SKILL.md": "---\nname: team-steps\ndescription: team procedure\n---\nShared steps.\n"}
    }
}


def _raw(skills: dict | None = SKILLS) -> str:
    body = {"protocol_version": 1, "conversation_id": "c", "input": "hi"}
    if skills is not None:
        body["skills"] = skills
    return json.dumps(body)


def _staged(root: Path) -> list[Path]:
    return sorted(root.glob(f"{PREFIX}*"))


def _exited_pid() -> int:
    """The pid of a process that has already exited, like a killed turn's."""
    exited = subprocess.Popen([sys.executable, "-c", "pass"])
    exited.wait()
    return exited.pid


def _session_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == cloud_session.logger.name]


@pytest.fixture()
def offline_turn(skills_tmp, tmp_path, monkeypatch):
    """A workspace and an offline model for a turn run through the entrypoint,
    which stages in skills_tmp."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv(cloud_session._WORKSPACE_PATH_ENV, str(workspace))
    monkeypatch.delenv(cloud_session._SKILLS_ROOT_ENV, raising=False)
    monkeypatch.setattr(
        llm_client_mod.LLMClient, "from_settings",
        classmethod(lambda cls, settings: _mock_llm()),
    )


def _builder(root: Path, seen: list[dict[str, bool]], *, fail: bool = False):
    """The real builder, with a turn that records each staged folder it sees
    and whether the request's skill is readable in it."""

    def build(req, **kwargs):
        session = cloud_session.build_cloud_chat_session(req, **kwargs)

        async def turn_stream(user_input, **_):
            seen.append({
                folder.name: (folder / "team-steps" / "SKILL.md").is_file()
                for folder in _staged(root)
            })
            if fail:
                raise RuntimeError("the model call failed")
            return
            yield  # pragma: no cover - makes this an async generator

        session.turn_stream = turn_stream
        return session

    return build


async def test_each_turn_reads_only_its_own_folder_and_leaves_none(skills_tmp, offline_turn):
    seen: list[dict[str, bool]] = []
    for _ in range(2):
        events: list[dict] = []
        await entry.stream_turn(_raw(), events.append, session_builder=_builder(skills_tmp, seen))
        assert events[-1]["kind"] == "turn_completed"
        assert _staged(skills_tmp) == []

    first, second = seen
    assert list(first.values()) == list(second.values()) == [True]
    assert first.keys() != second.keys()


async def test_a_turn_that_raises_still_removes_its_folder(skills_tmp, offline_turn):
    seen: list[dict[str, bool]] = []
    events: list[dict] = []

    await entry.stream_turn(
        _raw(), events.append, session_builder=_builder(skills_tmp, seen, fail=True)
    )

    assert events[-1]["kind"] == "turn_failed"
    assert [list(s.values()) for s in seen] == [[True]]  # the turn did stage one
    assert _staged(skills_tmp) == []


async def test_a_build_that_raises_after_staging_still_removes_its_folder(
    skills_tmp, offline_turn, monkeypatch
):
    staged: list[Path] = []
    stage = cloud_session._stage_skills

    def recording_stage(skills):
        staged.append(stage(skills))
        return staged[-1]

    def no_model(cls, settings):
        raise RuntimeError("no model credential")

    monkeypatch.setattr(cloud_session, "_stage_skills", recording_stage)
    monkeypatch.setattr(llm_client_mod.LLMClient, "from_settings", classmethod(no_model))
    events: list[dict] = []

    await entry.stream_turn(_raw(), events.append)

    assert events[-1]["kind"] == "turn_failed"
    assert len(staged) == 1 and staged[0].name.startswith(PREFIX)
    assert _staged(skills_tmp) == []


async def test_the_org_mount_survives_a_turn_that_read_it(
    skills_tmp, offline_turn, tmp_path, monkeypatch
):
    """Only staged folders are deleted. An older sender's turn reads the mount
    as its skills root, and the mount is the org's store, not a copy."""
    mount = tmp_path / "mnt" / "skills"
    (mount / "org-skill").mkdir(parents=True)
    (mount / "org-skill" / "SKILL.md").write_text("Org steps.")
    monkeypatch.setenv(cloud_session._SKILLS_ROOT_ENV, str(mount))
    events: list[dict] = []

    await entry.stream_turn(
        _raw(skills=None), events.append, session_builder=_builder(skills_tmp, [])
    )

    assert events[-1]["kind"] == "turn_completed"
    assert (mount / "org-skill" / "SKILL.md").read_text() == "Org steps."


@pytest.mark.skipif(sys.platform == "win32", reason="the liveness probe is POSIX only")
def test_staging_removes_what_finished_turns_left_and_keeps_a_running_turns(skills_tmp):
    killed = skills_tmp / f"{PREFIX}{_exited_pid()}-a1b2c3d4"  # a turn the pod killed
    unnamed = skills_tmp / f"{PREFIX}a1b2c3d4"  # cell code made it: staging names the pid
    earlier = skills_tmp / f"{PREFIX}{os.getpid()}-e5f6g7h8"  # an earlier turn in this process
    running = skills_tmp / f"{PREFIX}{os.getppid()}-i9j0k1l2"  # a turn another process is running
    for folder in (killed, unnamed, earlier, running):
        (folder / "old-skill").mkdir(parents=True)
        (folder / "old-skill" / "SKILL.md").write_text("Old steps.")

    dest = cloud_session._stage_skills(SKILLS)

    assert _staged(skills_tmp) == sorted([dest, running, unnamed])
    assert (running / "old-skill" / "SKILL.md").is_file()
    assert (unnamed / "old-skill" / "SKILL.md").is_file()


#: A turn in another process: it stages, reports its folder, and runs until
#: its stdin closes.
_OTHER_PROCESS_TURN = """
import json, sys, tempfile
tempfile.tempdir = sys.argv[1]
from anton.cloud_turn import session
print(session._stage_skills(json.loads(sys.argv[2])), flush=True)
sys.stdin.read()
"""


@pytest.mark.skipif(sys.platform == "win32", reason="the liveness probe is POSIX only")
def test_a_turn_running_in_another_process_keeps_its_folder_until_it_exits(skills_tmp):
    """Both sides run the real staging: the folder name staging writes is
    what tells this process's cleanup that the other turn is still running."""
    other = subprocess.Popen(
        [sys.executable, "-c", _OTHER_PROCESS_TURN, str(skills_tmp), json.dumps(SKILLS)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        running = Path(other.stdout.readline().strip())
        mine = cloud_session._stage_skills(SKILLS)

        assert _staged(skills_tmp) == sorted([running, mine])
        assert (running / "team-steps" / "SKILL.md").is_file()
    finally:
        other.stdin.close()
        other.wait(timeout=60)
        other.stdout.close()

    again = cloud_session._stage_skills(SKILLS)

    assert _staged(skills_tmp) == [again]


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks and FIFOs are POSIX only here")
def test_planted_entries_are_unlinked_quietly_unless_a_running_process_names_them(
    skills_tmp, tmp_path, caplog
):
    """Cell code can plant any entry under the prefix. Staging makes only
    folders, so an entry that isn't one is unlinked, never followed, and logs
    nothing that would repeat on every later turn. One whose name carries
    another running process's pid stays, like that process's own folder."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("not a staged folder")
    (skills_tmp / f"{PREFIX}{_exited_pid()}-link").symlink_to(outside, target_is_directory=True)
    (skills_tmp / f"{PREFIX}file").write_text("planted")
    os.mkfifo(skills_tmp / f"{PREFIX}{os.getpid()}-fifo")
    running = skills_tmp / f"{PREFIX}{os.getppid()}-file"
    running.write_text("planted")
    caplog.set_level(logging.DEBUG, logger=cloud_session.logger.name)

    dest = cloud_session._stage_skills(SKILLS)

    assert _staged(skills_tmp) == sorted([dest, running])
    assert running.read_text() == "planted"
    assert (outside / "keep.txt").read_text() == "not a staged folder"
    assert _session_records(caplog) == []


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="a read-only folder refuses an unlink only on POSIX, and never for root",
)
def test_a_folder_whose_name_carries_no_pid_stays_and_logs_only_at_debug(skills_tmp, caplog):
    """Cell code can make such a folder and lock it. Removing it would fail
    and warn on every turn, so cleanup leaves it alone."""
    locked = skills_tmp / f"{PREFIX}x" / "locked"
    locked.mkdir(parents=True)
    (locked / "f").write_text("planted")
    caplog.set_level(logging.DEBUG, logger=cloud_session.logger.name)
    locked.chmod(0o555)
    try:
        cloud_session.remove_staged_skills()
        cloud_session.remove_staged_skills()
    finally:
        locked.chmod(0o755)

    assert (locked / "f").read_text() == "planted"
    assert [r.levelno for r in _session_records(caplog)] == [logging.DEBUG] * 2


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="a read-only folder refuses an unlink only on POSIX, and never for root",
)
def test_a_planted_entry_that_cannot_be_unlinked_logs_only_at_debug(skills_tmp, caplog):
    planted = skills_tmp / f"{PREFIX}file"
    planted.write_text("planted")
    caplog.set_level(logging.DEBUG, logger=cloud_session.logger.name)
    skills_tmp.chmod(0o555)
    try:
        cloud_session.remove_staged_skills()
    finally:
        skills_tmp.chmod(0o755)

    assert planted.is_file()
    assert [r.levelno for r in _session_records(caplog)] == [logging.DEBUG]


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="a read-only folder refuses an unlink only on POSIX, and never for root",
)
def test_a_staged_folder_that_cannot_be_removed_warns_on_every_call(skills_tmp, caplog):
    """Unlike a planted file, a folder may still hold skills, so it warns."""
    locked = skills_tmp / f"{PREFIX}{_exited_pid()}-a1b2c3d4" / "team-steps"
    locked.mkdir(parents=True)
    (locked / "SKILL.md").write_text("Shared steps.")
    caplog.set_level(logging.DEBUG, logger=cloud_session.logger.name)
    locked.chmod(0o555)
    try:
        cloud_session.remove_staged_skills()
        cloud_session.remove_staged_skills()
    finally:
        locked.chmod(0o755)

    assert (locked / "SKILL.md").is_file()
    assert [r.levelno for r in _session_records(caplog)] == [logging.WARNING] * 2
