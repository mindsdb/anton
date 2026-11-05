"""Tool reads in the host process: under a sandbox policy only the person's own files; without one, as before.

Covers handle_read_image, handle_read_text_file and generate_artifact's
resolve_attachments, which all read outside the sandbox.

The layout below copies a multi-person Cowork home: each person's space is
`<cowork>/<partition>/` with `projects/`, `files/<upload id>/` and
`sandbox/{home,tmp}`, beside one `shared-files/` folder.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import anton.core.backends.local as local
from anton.core.backends.manager import ScratchpadManager
from anton.core.backends.sandbox import CellSandbox
from anton.core.tools.generate_artifact.attachments import REFUSED_HIDDEN, REFUSED_OUTSIDE, resolve_attachments
from anton.core.tools.tool_handlers import handle_read_image, handle_read_text_file

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the sandbox policy is POSIX-only")

# A real 1x1 PNG (as in test_tool_verdict_migration.py).
_PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d4944415478da63f8ffff3f0005fe02fea735d18b00"
    "00000049454e44ae426082"
)


def _png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PNG_1X1)
    return path


@pytest.fixture
def cowork(tmp_path: Path) -> Path:
    return tmp_path / ".cowork"


@pytest.fixture
def bob(cowork: Path) -> Path:
    return cowork / "bob-partition"


@pytest.fixture
def alice(cowork: Path) -> Path:
    return cowork / "alice-partition"


@pytest.fixture
def project(bob: Path) -> Path:
    folder = bob / "projects" / "demo"
    folder.mkdir(parents=True)
    return folder


@pytest.fixture
def policy(cowork: Path, bob: Path, alice: Path, project: Path) -> CellSandbox:
    for folder in (bob / "sandbox" / "home", bob / "sandbox" / "tmp", cowork / "shared-files"):
        folder.mkdir(parents=True)
    attachment = bob / "files" / "this-conversation"
    attachment.mkdir(parents=True)
    linked_skill = bob / "skills" / "forecast"
    linked_skill.mkdir(parents=True)
    return CellSandbox(
        project_dir=project,
        home_dir=bob / "sandbox" / "home",
        tmp_dir=bob / "sandbox" / "tmp",
        shared_dir=cowork / "shared-files",
        read_only_dirs=(linked_skill, attachment),
        forbidden_roots=(cowork, bob, alice),
        other_partitions=(alice,),
        user_name="bob",
    )


def _session(project: Path | None, policy: CellSandbox | None):
    workspace = SimpleNamespace(base=project) if project is not None else None
    return SimpleNamespace(_workspace=workspace, _scratchpads=SimpleNamespace(cell_sandbox=policy))


async def _read(session, file_path: str):
    return await handle_read_image(session, {"file_path": file_path})


# ── Under a policy: the person's own files load ──────────────────────


@pytest.mark.parametrize(
    "where, asked",
    [
        ("projects/demo/chart.png", "chart.png"),
        ("projects/demo/chart.png", "{bob}/projects/demo/chart.png"),
        ("projects/demo/uploaded/logo.png", "uploaded/logo.png"),
        ("projects/demo/.anton/artifacts/deck/static/c.png", ".anton/artifacts/deck/static/c.png"),
        ("projects/demo/.anton/uploads/clipboard_1.png", ".anton/uploads/clipboard_1.png"),
        ("sandbox/home/plot.png", "{bob}/sandbox/home/plot.png"),
        ("sandbox/home/plot.png", "~/plot.png"),
        ("sandbox/tmp/tmpab12.png", "{bob}/sandbox/tmp/tmpab12.png"),
        ("files/this-conversation/photo.png", "{bob}/files/this-conversation/photo.png"),
        ("skills/forecast/diagram.png", "{bob}/skills/forecast/diagram.png"),
    ],
)
async def test_own_files_load(bob, project, policy, where, asked):
    _png(bob / where)
    out = await _read(_session(project, policy), asked.format(bob=bob))
    assert out.ok is True, out.content
    assert [b["type"] for b in out.content] == ["image", "text"]


async def test_the_shared_folder_loads(cowork, project, policy):
    _png(cowork / "shared-files" / "team.png")
    out = await _read(_session(project, policy), str(cowork / "shared-files" / "team.png"))
    assert out.ok is True


async def test_a_link_in_the_project_to_a_granted_folder_loads(bob, project, policy):
    _png(bob / "skills" / "forecast" / "diagram.png")
    (project / "skills").mkdir()
    (project / "skills" / "forecast").symlink_to(bob / "skills" / "forecast", target_is_directory=True)
    out = await _read(_session(project, policy), "skills/forecast/diagram.png")
    assert out.ok is True


# ── Under a policy: everything else is refused before it is looked at ──


@pytest.mark.parametrize(
    "asked",
    [
        "{alice}/files/upload-1/secret.png",
        "{alice}/projects/plan/chart.png",
        "{alice}/sandbox/home/plot.png",
        "{alice}/sandbox/tmp/tmpzz.png",
        # Matches read_text_file's `.cowork/files/` layout rule.
        "{cowork}/files/operator-upload/op.png",
        # Bob's own upload to another conversation: his cells cannot read it either.
        "{bob}/files/other-conversation/old.png",
        "../../../alice-partition/files/upload-1/secret.png",
    ],
)
async def test_other_files_are_denied(cowork, bob, alice, project, policy, asked):
    for folder in (alice, cowork / "files", bob / "files" / "other-conversation"):
        folder.mkdir(parents=True, exist_ok=True)
    _png(alice / "files" / "upload-1" / "secret.png")
    _png(cowork / "files" / "operator-upload" / "op.png")
    _png(bob / "files" / "other-conversation" / "old.png")
    _png(alice / "projects" / "plan" / "chart.png")
    _png(alice / "sandbox" / "home" / "plot.png")
    _png(alice / "sandbox" / "tmp" / "tmpzz.png")
    out = await _read(_session(project, policy), asked.format(cowork=cowork, bob=bob, alice=alice))
    assert out.ok is False
    assert out.reason == "access_denied"
    assert isinstance(out.content, str)
    assert "ask them to attach it" in out.content


async def test_a_denial_reads_the_same_whatever_is_at_the_path(alice, project, policy):
    """No existence oracle: present, absent, a folder and a non-image all get one reply."""
    asked = str(alice / "files" / "upload-1" / "secret.png")
    session = _session(project, policy)
    replies = []
    _png(Path(asked))
    replies.append((await _read(session, asked)).content)
    Path(asked).unlink()
    replies.append((await _read(session, asked)).content)
    Path(asked).mkdir()
    replies.append((await _read(session, asked)).content)
    Path(asked).rmdir()
    Path(asked).write_text("not a picture")
    replies.append((await _read(session, asked)).content)
    assert len(set(replies)) == 1
    assert "access denied" in replies[0]


async def test_a_link_in_the_project_to_another_persons_image_is_denied(alice, project, policy):
    secret = _png(alice / "files" / "upload-1" / "secret.png")
    (project / "innocent.png").symlink_to(secret)
    out = await _read(_session(project, policy), "innocent.png")
    assert out.reason == "access_denied"
    assert str(alice) not in out.content


async def test_tilde_is_the_cells_home_not_the_servers(alice, project, policy, monkeypatch):
    monkeypatch.setenv("HOME", str(alice.parent.parent))
    _png(alice / "files" / "upload-1" / "secret.png")
    out = await _read(_session(project, policy), f"~/.cowork/{alice.name}/files/upload-1/secret.png")
    # Inside bob's own sandbox HOME, where nothing is: not found, not alice's file.
    assert out.reason == "path_not_found"


async def test_the_policy_is_read_off_a_real_scratchpad_manager(alice, project, policy):
    _png(alice / "files" / "upload-1" / "secret.png")
    manager = ScratchpadManager(
        runtime_factory=local.local_scratchpad_runtime_factory,
        coding_provider="anthropic", coding_model="", coding_api_key="", coding_base_url="",
        workspace_path=project, session_id="conv-1", cell_sandbox=policy,
    )
    session = SimpleNamespace(_workspace=SimpleNamespace(base=project), _scratchpads=manager)
    out = await _read(session, str(alice / "files" / "upload-1" / "secret.png"))
    assert out.reason == "access_denied"


async def test_a_policy_of_the_wrong_type_fails_closed(project):
    session = SimpleNamespace(
        _workspace=SimpleNamespace(base=project),
        _scratchpads=SimpleNamespace(cell_sandbox={"project_dir": str(project)}),
    )
    with pytest.raises(TypeError):
        await _read(session, "chart.png")


# ── Without a policy (CLI, desktop, cloud pod): unchanged ────────────


async def test_without_a_policy_any_readable_path_loads(alice, project):
    """The CLI's dropped-file hint names ~/Desktop, /tmp, external volumes."""
    _png(alice / "files" / "upload-1" / "secret.png")
    out = await _read(_session(project, None), str(alice / "files" / "upload-1" / "secret.png"))
    assert out.ok is True


async def test_without_a_policy_tilde_is_the_process_home(tmp_path, project, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "me"))
    _png(tmp_path / "me" / "Desktop" / "shot.png")
    out = await _read(_session(project, None), "~/Desktop/shot.png")
    assert out.ok is True


async def test_without_a_policy_a_link_is_followed(tmp_path, project):
    target = _png(tmp_path / "elsewhere" / "shot.png")
    (project / "shot.png").symlink_to(target)
    out = await _read(_session(project, None), "shot.png")
    assert out.ok is True


async def test_without_a_workspace_a_relative_path_reads_from_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _png(tmp_path / "shot.png")
    out = await _read(SimpleNamespace(), "shot.png")
    assert out.ok is True


# ── Edges the critic named ───────────────────────────────────────────


def _with_shared_skill(policy: CellSandbox, folder: Path) -> CellSandbox:
    return policy.model_copy(update={"read_only_dirs": (*policy.read_only_dirs, folder)})


async def test_a_skill_another_person_shares_loads_and_nothing_beside_it(alice, project, policy):
    shared = alice / "skills" / "shared-one"
    _png(shared / "diagram.png")
    _png(alice / "skills" / "private-one" / "diagram.png")
    session = _session(project, _with_shared_skill(policy, shared))
    assert (await _read(session, str(shared / "diagram.png"))).ok is True
    beside = await _read(session, str(alice / "skills" / "private-one" / "diagram.png"))
    assert beside.reason == "access_denied"


async def test_a_grant_that_is_a_link_admits_nothing(alice, bob, project, policy):
    _png(alice / "files" / "upload-1" / "secret.png")
    link = bob / "skills" / "looks-mine"
    link.symlink_to(alice / "files" / "upload-1", target_is_directory=True)
    session = _session(project, _with_shared_skill(policy, link))
    out = await _read(session, str(link / "secret.png"))
    assert out.reason == "access_denied"


async def test_a_dot_dot_escape_names_only_the_path_as_given(alice, project, policy):
    _png(alice / "files" / "upload-1" / "secret.png")
    asked = "../../../alice-partition/files/upload-1/secret.png"
    out = await _read(_session(project, policy), asked)
    assert out.reason == "access_denied"
    assert asked in out.content
    assert str(alice) not in out.content


async def test_tilde_alone_is_the_cells_home(project, policy):
    out = await _read(_session(project, policy), "~")
    # The cell's HOME is a folder, not an image: past the policy, refused as no file.
    assert out.reason == "path_not_found"


async def test_tilde_slash_slash_stays_in_the_cells_home(project, policy):
    out = await _read(_session(project, policy), "~//etc/x.png")
    assert out.reason == "path_not_found"


async def test_another_users_home_is_denied(project, policy):
    out = await _read(_session(project, policy), "~root/x.png")
    assert out.reason in ("access_denied", "invalid_type")
    assert out.ok is False


@pytest.mark.parametrize("asked", ["~no-such-user-eng3428/x.png", "bad\x00name.png"])
async def test_without_a_policy_a_path_that_cannot_parse_is_an_error_not_a_crash(project, asked):
    out = await _read(_session(project, None), asked)
    assert out.ok is False
    assert out.reason in ("invalid_type", "path_not_found")


async def test_the_size_comes_from_the_bytes_read(project, policy):
    pil = pytest.importorskip("PIL.Image")
    image = pil.new("RGB", (2, 3))
    image.save(project / "two-by-three.png")
    out = await _read(_session(project, policy), "two-by-three.png")
    assert out.ok is True
    assert "2x3" in out.content[1]["text"]


# ── read_text_file under the same policy ─────────────────────────────


async def _read_text(session, path: str):
    return await handle_read_text_file(session, {"path": path})


def _text(path: Path, body: str = "a line\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


async def test_read_text_file_reads_this_conversations_upload(bob, project, policy):
    """v1.3.0 keeps uploads at <cowork>/<partition>/files/<id>/, which the layout rule misses."""
    upload = _text(bob / "files" / "this-conversation" / "notes.txt")
    out = await _read_text(_session(project, policy), str(upload))
    assert out.ok is True, out.content


@pytest.mark.parametrize(
    "asked",
    [
        "{alice}/files/upload-1/secret.txt",
        "{cowork}/files/operator-upload/op.txt",
        "{bob}/files/other-conversation/old.txt",
    ],
)
async def test_read_text_file_refuses_what_bobs_cells_cannot_read(cowork, bob, alice, project, policy, asked):
    _text(alice / "files" / "upload-1" / "secret.txt")
    _text(cowork / "files" / "operator-upload" / "op.txt")
    _text(bob / "files" / "other-conversation" / "old.txt")
    out = await _read_text(_session(project, policy), asked.format(cowork=cowork, bob=bob, alice=alice))
    assert out.reason == "access_denied"
    assert "ask them to attach it" in out.content


async def test_read_text_file_names_only_the_path_as_given_under_a_policy(alice, project, policy):
    secret = _text(alice / "files" / "upload-1" / "secret.txt")
    (project / "notes.txt").symlink_to(secret)
    out = await _read_text(_session(project, policy), "notes.txt")
    assert out.reason == "access_denied"
    assert str(alice) not in out.content


async def test_read_text_file_still_refuses_the_projects_dot_files(project, policy):
    _text(project / ".anton" / ".env", "KEY=value\n")
    out = await _read_text(_session(project, policy), ".anton/.env")
    assert out.reason == "access_denied"


async def test_read_text_file_without_a_policy_keeps_the_layout_rule(cowork, project):
    upload = _text(cowork / "files" / "operator-upload" / "op.txt")
    out = await _read_text(_session(project, None), str(upload))
    assert out.ok is True


# ── generate_artifact's attachments under the same policy ────────────


def test_attachments_keep_this_conversations_upload(bob, project, policy):
    upload = _png(bob / "files" / "this-conversation" / "logo.png")
    kept, dropped = resolve_attachments([str(upload)], workspace=project, policy=policy)
    assert [a.name for a in kept] == ["logo.png"], dropped


def test_attachments_drop_what_bobs_cells_cannot_read(cowork, alice, project, policy):
    theirs = _png(alice / "files" / "upload-1" / "secret.png")
    operator = _png(cowork / "files" / "operator-upload" / "op.png")
    kept, dropped = resolve_attachments([str(theirs), str(operator)], workspace=project, policy=policy)
    assert kept == []
    assert all(REFUSED_OUTSIDE in reason for reason in dropped) and len(dropped) == 2


def test_attachments_still_drop_the_projects_dot_files(project, policy):
    hidden = _png(project / ".anton" / "cache" / "x.png")
    kept, dropped = resolve_attachments([str(hidden)], workspace=project, policy=policy)
    assert kept == [] and REFUSED_HIDDEN in dropped[0]


def test_attachments_without_a_policy_keep_the_layout_rule(cowork, project):
    operator = _png(cowork / "files" / "operator-upload" / "op.png")
    kept, _ = resolve_attachments([str(operator)], workspace=project)
    assert [a.name for a in kept] == ["op.png"]

