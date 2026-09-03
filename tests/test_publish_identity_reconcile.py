"""ENG-1424: legacy project-vault publish keys are collapsed into the global vault.

Removing the writer of ``<project>/.anton/.env``'s MindsHub key was not enough on
its own. Every reader stayed: ``_ensure_workspace`` promotes the project vault into
``os.environ``, above every config file, where a stale key still reaches the
scratchpad subprocess and the ``ANTON_MINDS_API_KEY`` -> ``OPENAI_API_KEY``
fallback. Worse, the old code at least *refreshed* that copy on every successful
publish, so deleting the writer alone would have frozen the stale key forever.

``_reconcile_publish_identity`` runs once at boot, before any promotion, and
leaves exactly one copy — in ``~/.anton/.env``.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from anton.cli import _reconcile_publish_identity
from anton.workspace import Workspace


def _settings(workspace_path: Path):
    """A settings stand-in carrying only what reconciliation reads."""
    from types import SimpleNamespace

    return SimpleNamespace(workspace_path=str(workspace_path), artifacts_dir="artifacts")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start clean AND guarantee restoration.

    `Workspace.set_secret` writes `os.environ` as a side effect, and
    `monkeypatch.delenv(..., raising=False)` records nothing when the variable
    is absent — so it has nothing to undo, and a value the test body sets that
    way leaks for the rest of the session. Since `os.environ` outranks every
    `env_file`, a later test building `AntonSettings()` would resolve the leaked
    key. Touch the name first so monkeypatch has it on the undo list.
    """
    monkeypatch.setenv("ANTON_MINDS_API_KEY", "__sentinel__")
    monkeypatch.delenv("ANTON_MINDS_API_KEY", raising=False)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A PRISTINE home per test, overriding the suite-wide one.

    conftest's `_no_real_home` shares one isolated home for the whole session
    (a fresh one per test cost 230s of e2e subprocess first-runs), so other
    files leave keys in it — `test_publish_api_key.py` leaves `goodkey` in the
    global vault. Reconciliation branches on whether that vault is EMPTY, so
    these tests cannot use the shared one. Setting $HOME is enough: conftest's
    patched `Path.home()` defers to any test that names its own.
    """
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    return h


def test_project_key_is_moved_up_when_the_global_vault_is_empty(tmp_path, home):
    """The --folder case: the target's vault is not in the settings chain at all.

    ``_build_env_files`` is evaluated at import time against ``Path.cwd()``, so
    with ``anton -f <dir>`` the target's ``.anton/.env`` never enters the chain
    and its key is invisible to the session — while ``_ensure_workspace`` still
    promotes it into os.environ. Before reconciliation that mismatch made the
    publish tool refuse ("STOP: No Minds API key configured") for a user who
    could publish before.
    """
    project = tmp_path / "proj"
    project.mkdir()
    Workspace(project).set_secret("ANTON_MINDS_API_KEY", "PROJECT_ONLY_KEY")

    assert _reconcile_publish_identity(_settings(project)) is True

    assert Workspace(home).get_secret("ANTON_MINDS_API_KEY") == "PROJECT_ONLY_KEY"
    assert Workspace(project).get_secret("ANTON_MINDS_API_KEY") is None


def test_a_duplicate_project_copy_is_dropped_outright(tmp_path, home):
    """Same string in both vaults — the copy carries no information, so drop it."""
    project = tmp_path / "proj"
    project.mkdir()
    Workspace(project).set_secret("ANTON_MINDS_API_KEY", "SAME_KEY")
    Workspace(home).set_secret("ANTON_MINDS_API_KEY", "SAME_KEY")

    assert _reconcile_publish_identity(_settings(project)) is True

    assert Workspace(home).get_secret("ANTON_MINDS_API_KEY") == "SAME_KEY"
    assert Workspace(project).get_secret("ANTON_MINDS_API_KEY") is None
    # Nothing was lost, so nothing needs archiving.
    assert not (project / ".anton" / ".env.superseded").exists()


def test_a_divergent_project_key_is_archived_not_destroyed(tmp_path, home):
    """Two DIFFERENT keys: keep one identity, but never delete the other.

    Reconciliation cannot tell which of the two authenticates — reading a vault
    is a presence check. An earlier revision assumed the global one was current
    and deleted the project's outright, which is wrong in the one direction that
    costs the user an account (see the ticket-state test below).
    """
    project = tmp_path / "proj"
    project.mkdir()
    Workspace(project).set_secret("ANTON_MINDS_API_KEY", "PROJECT_KEY")
    Workspace(home).set_secret("ANTON_MINDS_API_KEY", "GLOBAL_KEY")

    assert _reconcile_publish_identity(_settings(project)) is True

    # One identity wins, so the promotion can no longer diverge...
    assert Workspace(home).get_secret("ANTON_MINDS_API_KEY") == "GLOBAL_KEY"
    assert Workspace(project).get_secret("ANTON_MINDS_API_KEY") is None
    # ...but the other is still on disk, and readable only by its owner.
    archive = project / ".anton" / ".env.superseded"
    assert "PROJECT_KEY" in archive.read_text()
    if sys.platform != "win32":
        assert stat.S_IMODE(archive.stat().st_mode) == 0o600


def test_the_ticket_s_measured_machine_state_loses_nothing(tmp_path, home):
    """ENG-1424's own evidence table, 2026-08-11, on the reporter's machine:

        <workspace>/.anton/.env   valid, account f1ec7bf69, 48 published reports
        ~/.anton/.env             401 — invalid (7-char, all letters)

    So `global vault is set` does NOT mean `global vault is good`, and the
    valid key is the project one. Deleting it unconditionally would strand the
    account owning those 48 reports — a worse version of the bug being fixed,
    and the same presence-vs-validity mistake the ticket criticises in
    `_has_api_key`.
    """
    project = tmp_path / "proj"
    project.mkdir()
    Workspace(project).set_secret("ANTON_MINDS_API_KEY", "mdb_valid_account_a")
    Workspace(home).set_secret("ANTON_MINDS_API_KEY", "nonsense")  # the 7-char 401

    _reconcile_publish_identity(_settings(project))

    on_disk = "".join(
        f.read_text() for f in (project / ".anton").iterdir() if f.is_file()
    )
    assert "mdb_valid_account_a" in on_disk, "the working key was destroyed"


def test_the_stale_key_can_no_longer_be_promoted(tmp_path, home):
    """The point of doing this BEFORE _ensure_workspace.

    os.environ outranks every config file, so a surviving project copy would
    still be the identity that scratchpad cells and any mid-turn settings
    rebuild saw — the same-session two-key split, just moved off the publish path.
    """
    project = tmp_path / "proj"
    project.mkdir()
    Workspace(project).set_secret("ANTON_MINDS_API_KEY", "OLD_ROTATED_KEY")
    Workspace(home).set_secret("ANTON_MINDS_API_KEY", "CURRENT_KEY")
    os.environ.pop("ANTON_MINDS_API_KEY", None)

    _reconcile_publish_identity(_settings(project))

    # What _ensure_workspace would promote, replayed.
    Workspace(project).apply_env_to_process()
    assert os.environ.get("ANTON_MINDS_API_KEY") != "OLD_ROTATED_KEY"


def test_reconciliation_is_idempotent_and_silent_when_there_is_nothing_to_do(tmp_path, home):
    project = tmp_path / "proj"
    project.mkdir()
    Workspace(project).set_secret("ANTON_MINDS_API_KEY", "PROJECT_ONLY_KEY")

    assert _reconcile_publish_identity(_settings(project)) is True
    # Second boot: nothing left to move, so no write and no re-resolve.
    assert _reconcile_publish_identity(_settings(project)) is False


def test_reconciliation_is_actually_wired_into_boot():
    """The unit tests above call the function directly, so none of them notices
    if the CALL SITE disappears — verified by mutation (stub the call out in
    `cli.main` and every other test in this file still passes).

    Checked structurally against the AST rather than by grepping the file for a
    string: a source-text match would also hit a comment or a docstring, and
    would break on a harmless reformat. This asserts the name is really
    referenced inside `main`, and that it is referenced BEFORE
    `_ensure_workspace` — reconciling after the promotion would leave the stale
    key already in os.environ, which is the whole thing being prevented.
    """
    import ast
    import inspect

    import anton.cli as cli_mod

    tree = ast.parse(inspect.getsource(cli_mod))
    main_fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "main"
    )
    # Keyed on lineno, NOT on position in the walk: ast.walk is breadth-first,
    # so list order is depth order. A version that left _ensure_workspace nested
    # inside `if ctx.invoked_subcommand is None:` and moved the reconcile call
    # after it at statement level would be yielded reconcile-first and pass,
    # while the stale key had already been promoted.
    called = {
        n.func.id: n.lineno
        for n in ast.walk(main_fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_reconcile_publish_identity" in called, (
        "cli.main no longer reconciles the publish identity at boot — legacy "
        "project-vault keys will be promoted into os.environ again"
    )
    assert "_ensure_workspace" in called
    assert called["_reconcile_publish_identity"] < called["_ensure_workspace"], (
        "reconciliation must run BEFORE _ensure_workspace promotes the project vault"
    )


def test_a_home_workspace_is_never_reconciled_against_itself(home):
    """`anton` run from $HOME: project vault and global vault are the same file.

    Without the guard this would read the key, delete it, and (since the global
    vault it just emptied is the same file) write it back — or lose it.
    """
    Workspace(home).set_secret("ANTON_MINDS_API_KEY", "HOME_KEY")

    assert _reconcile_publish_identity(_settings(home)) is False
    assert Workspace(home).get_secret("ANTON_MINDS_API_KEY") == "HOME_KEY"


def test_the_session_can_use_the_key_reconciliation_just_moved(tmp_path, home, monkeypatch):
    """Review finding: the migrated key was unusable for the whole session.

    The `.env` chain is built ONCE at import from the files that existed then
    (`config/settings.py`). Creating `~/.anton/.env` during reconciliation does
    not add it, and `remove_secret` pops the project copy out of `os.environ`,
    so the caller's re-resolve came back with NO key — immediately after the
    console said the key had been moved there. Self-healed only on next boot.
    """
    from anton.config.settings import AntonSettings

    project = tmp_path / "proj"
    project.mkdir()
    Workspace(project).set_secret("ANTON_MINDS_API_KEY", "PROJECT_KEY")
    # `set_secret` also writes os.environ. At a real boot the project vault is
    # only on DISK at this point — `_ensure_workspace` has not promoted it yet —
    # and leaving the variable set masks this bug entirely, because os.environ
    # outranks every env_file and would satisfy the assertion on its own.
    monkeypatch.delenv("ANTON_MINDS_API_KEY", raising=False)
    assert not (home / ".anton" / ".env").exists()  # precondition: not in the chain

    assert _reconcile_publish_identity(_settings(project)) is True

    assert AntonSettings().minds_api_key == "PROJECT_KEY"


def test_reconciliation_leaves_an_exported_key_alone(tmp_path, home, monkeypatch):
    """Review finding: `ANTON_MINDS_API_KEY=X anton` had X silently deleted.

    Reconciliation is a DISK migration, but `set_secret`/`remove_secret` both
    write `os.environ` as a side effect — and pydantic ranks `os.environ` above
    every env_file, so X is precisely the key that session was resolving.
    """
    from anton.config.settings import AntonSettings

    project = tmp_path / "proj"
    project.mkdir()
    Workspace(project).set_secret("ANTON_MINDS_API_KEY", "Y_PROJECT")
    Workspace(home).set_secret("ANTON_MINDS_API_KEY", "Z_GLOBAL")
    monkeypatch.setenv("ANTON_MINDS_API_KEY", "X_EXPORTED")

    _reconcile_publish_identity(_settings(project))

    assert os.environ.get("ANTON_MINDS_API_KEY") == "X_EXPORTED"
    assert AntonSettings().minds_api_key == "X_EXPORTED"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_a_symlinked_home_is_still_recognised_as_home(tmp_path, monkeypatch):
    """Review finding: a lexical Path comparison deleted the only key.

    `workspace_path` is canonical (`Path.cwd()` / `Path(folder).resolve()`)
    while `Path.home()` is `$HOME` verbatim, so on `/home -> /export/home` the
    guard missed. The same file was then addressed as two vaults, compared
    equal to itself, and took the "duplicate — drop it" branch.
    """
    real = tmp_path / "export" / "home" / "user"
    real.mkdir(parents=True)
    (tmp_path / "home").symlink_to("export/home")
    monkeypatch.setenv("HOME", str(tmp_path / "home" / "user"))
    Workspace(real).set_secret("ANTON_MINDS_API_KEY", "THE_ONLY_KEY")

    assert _reconcile_publish_identity(_settings(real)) is False
    assert Workspace(real).get_secret("ANTON_MINDS_API_KEY") == "THE_ONLY_KEY"
