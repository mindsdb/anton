"""The sandbox policy model, its environment allowlist, its launch call and its plumbing."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pydantic
import pytest

from anton.core.backends import sandbox
from anton.core.backends.local import LocalScratchpadRuntime, local_scratchpad_runtime_factory
from anton.core.backends.manager import ScratchpadManager
from anton.core.backends.sandbox import (
    SANDBOX_REFUSED_MARKER,
    CellSandbox,
    SandboxUnavailable,
    cell_sandbox_of,
    validate_pad_name,
)

# The policy's paths are POSIX paths; on Windows they are not absolute.
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the sandbox policy is POSIX-only")


@pytest.fixture
def policy(tmp_path: Path) -> CellSandbox:
    for name in ("project", "home", "tmp"):
        (tmp_path / name).mkdir()
    return CellSandbox(
        project_dir=tmp_path / "project",
        home_dir=tmp_path / "home",
        tmp_dir=tmp_path / "tmp",
        user_name="bob",
    )


def _fields(tmp_path: Path, **overrides) -> dict:
    fields = {
        "project_dir": tmp_path / "project",
        "home_dir": tmp_path / "home",
        "tmp_dir": tmp_path / "tmp",
        "user_name": "bob",
    }
    fields.update(overrides)
    return fields


def _read_payload(fd: int) -> dict:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while chunk := os.read(fd, 1 << 16):
        chunks.append(chunk)
    return json.loads(b"".join(chunks))


# ── The model ────────────────────────────────────────────────────────────────


def test_the_policy_is_frozen_and_takes_no_unknown_fields(policy, tmp_path):
    with pytest.raises(pydantic.ValidationError):
        policy.user_name = "alice"  # type: ignore[misc]
    with pytest.raises(pydantic.ValidationError, match="extra"):
        CellSandbox(**_fields(tmp_path), network="on")


@pytest.mark.parametrize(
    "root", ["/", "/home", "/opt", "/opt/cowork", "/tmp", "/var/tmp", "/dev", "/etc", "/proc", "/sys"]
)
@pytest.mark.parametrize("field", ["project_dir", "home_dir", "tmp_dir", "shared_dir"])
def test_no_grant_may_equal_a_denied_root(tmp_path, root, field):
    with pytest.raises(pydantic.ValidationError, match="equals or contains"):
        CellSandbox(**_fields(tmp_path, **{field: Path(root)}))


def test_no_grant_may_contain_the_servers_home(tmp_path):
    home = Path.home()
    with pytest.raises(pydantic.ValidationError, match="equals or contains"):
        CellSandbox(**_fields(tmp_path, read_only_dirs=(home.parent,)))
    with pytest.raises(pydantic.ValidationError, match="equals or contains"):
        CellSandbox(**_fields(tmp_path, home_dir=home))


def test_no_grant_may_equal_or_contain_a_host_forbidden_root(tmp_path):
    cowork_home = tmp_path / "cowork-home"
    alice = cowork_home / "partitions" / "alice"
    forbidden = (cowork_home, alice)
    with pytest.raises(pydantic.ValidationError, match="equals or contains"):
        CellSandbox(**_fields(tmp_path, forbidden_roots=forbidden, read_only_dirs=(cowork_home,)))
    with pytest.raises(pydantic.ValidationError, match="equals or contains"):
        CellSandbox(**_fields(tmp_path, forbidden_roots=forbidden, shared_dir=cowork_home / "partitions"))
    with pytest.raises(pydantic.ValidationError, match="equals or contains"):
        CellSandbox(**_fields(tmp_path, forbidden_roots=forbidden, read_only_files=(alice,)))
    # Below a forbidden root is fine: that is where a person's own space lives.
    bob = cowork_home / "partitions" / "bob"
    CellSandbox(
        **_fields(
            tmp_path,
            project_dir=bob / "project",
            home_dir=bob / "sandbox" / "home",
            tmp_dir=bob / "sandbox" / "tmp",
            forbidden_roots=forbidden,
        )
    )


@pytest.mark.parametrize("field", ["project_dir", "home_dir", "tmp_dir", "shared_dir"])
@pytest.mark.parametrize("where", ["equal", "inside"])
def test_no_writable_grant_may_sit_in_another_persons_partition(tmp_path, field, where):
    alice = tmp_path / "partitions" / "alice"
    grant = alice if where == "equal" else alice / "projects" / "p"
    with pytest.raises(pydantic.ValidationError, match="another person's partition|equals or contains"):
        CellSandbox(**_fields(tmp_path, other_partitions=(alice,), **{field: grant}))


def test_a_read_only_grant_inside_another_persons_partition_is_allowed(tmp_path):
    alice = tmp_path / "partitions" / "alice"
    shared_skill = alice / "skills" / "forecast"
    policy = CellSandbox(**_fields(tmp_path, other_partitions=(alice,), read_only_dirs=(shared_skill,)))
    assert policy.read_only_dirs == (shared_skill,)
    with pytest.raises(pydantic.ValidationError, match="equals or contains"):
        CellSandbox(**_fields(tmp_path, other_partitions=(alice,), read_only_dirs=(alice.parent,)))
    # Older hosts pass no partitions, and their policies still validate.
    assert CellSandbox(**_fields(tmp_path)).other_partitions == ()


def test_a_grant_through_a_link_to_a_denied_root_is_refused(tmp_path):
    link = tmp_path / "looks-private"
    link.symlink_to("/etc")
    with pytest.raises(pydantic.ValidationError, match="equals or contains"):
        CellSandbox(**_fields(tmp_path, read_only_dirs=(link,)))


@pytest.mark.parametrize("bad", ["relative/project", "/a/../b"])
def test_every_path_must_be_absolute_and_normalized(tmp_path, bad):
    with pytest.raises(pydantic.ValidationError, match="absolute, normalized"):
        CellSandbox(**_fields(tmp_path, project_dir=Path(bad)))


@pytest.mark.parametrize(
    ("first", "second"),
    [("project_dir", "home_dir"), ("home_dir", "tmp_dir"), ("tmp_dir", "project_dir")],
)
def test_the_private_folders_may_not_overlap(tmp_path, first, second):
    with pytest.raises(pydantic.ValidationError, match="overlap"):
        CellSandbox(**_fields(tmp_path, **{first: tmp_path / "x", second: tmp_path / "x" / "y"}))
    with pytest.raises(pydantic.ValidationError, match="overlap"):
        CellSandbox(**_fields(tmp_path, shared_dir=tmp_path / "project" / "shared"))


def test_the_user_name_is_required(tmp_path):
    with pytest.raises(pydantic.ValidationError):
        CellSandbox(**_fields(tmp_path, user_name=""))


# ── Resolving each path once ─────────────────────────────────────────────────


def _folded_loop_tree(base: Path) -> list[str]:
    """A link that loops back on itself before a ``..``, ``a/L -> L/../x``, and
    a link below the folder that ``..`` lands on, ``a/x/y -> elsewhere``. The
    kernel can't open ``a/L/y`` (ELOOP). realpath stops at the loop, keeps the
    rest as written and folds the ``..``, so it answers ``a/x/y`` without
    following ``y``. ``b/K`` is a plain self loop."""
    for folder in ("a/x/sub", "elsewhere", "b"):
        (base / folder).mkdir(parents=True)
    (base / "a" / "L").symlink_to("L/../x")
    (base / "a" / "x" / "y").symlink_to(base / "elsewhere", target_is_directory=True)
    (base / "b" / "K").symlink_to("K")
    return [str(base / "a" / "L"), str(base / "a" / "L" / "y"), str(base / "a" / "L" / "y" / "z")]


def _resolver_tree(tmp_path: Path) -> list[str]:
    """Paths through links in the middle and at the end, a relative link, a
    loop, a missing folder and a file used as a folder."""
    (tmp_path / "a" / "b" / "c").mkdir(parents=True)
    (tmp_path / "a" / "file.txt").write_text("x")
    (tmp_path / "d").mkdir()
    (tmp_path / "mid").symlink_to(tmp_path / "a", target_is_directory=True)
    (tmp_path / "end").symlink_to(tmp_path / "a" / "b", target_is_directory=True)
    (tmp_path / "d" / "relative").symlink_to("../a/b", target_is_directory=True)
    (tmp_path / "d" / "chain").symlink_to(tmp_path / "mid" / "b", target_is_directory=True)
    (tmp_path / "loop1").symlink_to(tmp_path / "loop2")
    (tmp_path / "loop2").symlink_to(tmp_path / "loop1")
    (tmp_path / "dangling").symlink_to(tmp_path / "gone")
    folded = _folded_loop_tree(tmp_path / "folded")
    return [
        *folded,
        str(tmp_path / "folded" / "b" / "K" / "q"),
        str(tmp_path) + "/deep" * 1000,
        "/",
        "//",
        "/etc",
        "/tmp",
        str(tmp_path),
        str(tmp_path / "a" / "b" / "c"),
        str(tmp_path / "mid"),
        str(tmp_path / "mid" / "b" / "c"),
        str(tmp_path / "end"),
        str(tmp_path / "end" / "c" / "missing"),
        str(tmp_path / "d" / "relative" / "c"),
        str(tmp_path / "d" / "chain" / "c"),
        str(tmp_path / "loop1"),
        str(tmp_path / "loop1" / "x"),
        str(tmp_path / "dangling" / "x"),
        str(tmp_path / "missing" / "deeper"),
        str(tmp_path / "a" / "file.txt" / "inner"),
        f"{tmp_path}/a//b",
        f"{tmp_path}/a/./b",
        f"{tmp_path}/mid/../a/b",
        f"/{tmp_path}",
        "relative/path",
    ]


def test_the_resolver_resolves_every_path_as_realpath_does(tmp_path):
    paths = _resolver_tree(tmp_path)
    shared = sandbox._PathResolver()
    for path in paths:
        assert shared(path) == os.path.realpath(path), path
        assert sandbox._PathResolver()(path) == os.path.realpath(path), path
    # A second ask returns what the first did, in any order.
    for path in reversed(paths):
        assert shared(Path(path)) == os.path.realpath(path), path


def test_the_resolver_raises_what_realpath_raises_for_a_nul_byte(tmp_path):
    with pytest.raises(ValueError) as expected:
        os.path.realpath(f"{tmp_path}/nul\0name")
    with pytest.raises(ValueError) as raised:
        sandbox._PathResolver()(f"{tmp_path}/nul\0name")
    assert str(raised.value) == str(expected.value)


def _pairwise_refusal(fields: dict) -> str | None:
    """What the pairwise check refused for ``fields``, with os.path.realpath
    run on every pair: the check this index replaced, kept as the reference."""
    model = CellSandbox.model_construct(**fields)
    for grant in model._grants():
        for root in model._denied_roots():
            if sandbox._overlaps(grant, root):
                return f"grant {grant} equals or contains {root}"
    private = model._writable_dirs()
    for grant in private:
        for partition in model.other_partitions:
            if sandbox._overlaps(partition, grant):
                return f"writable grant {grant} sits in another person's partition {partition}"
    return None


def test_a_link_that_loops_before_a_dotdot_is_judged_as_the_pairwise_check_judged_it(tmp_path, monkeypatch):
    _folded_loop_tree(tmp_path / "t")
    a, elsewhere = tmp_path / "t" / "a", tmp_path / "t" / "elsewhere"
    looped = a / "L" / "y"
    cases = [
        {"read_only_dirs": (a / "x",), "forbidden_roots": (looped,)},
        {"read_only_dirs": (looped,), "forbidden_roots": (elsewhere,)},
        {"read_only_dirs": (a / "x",), "forbidden_roots": (looped, a / "x" / "sub")},
        {"project_dir": looped / "proj", "other_partitions": (a / "x",)},
        {"project_dir": looped / "proj", "other_partitions": (elsewhere,)},
    ]
    for case in cases:
        fields = _fields(tmp_path, **case)
        expected = _pairwise_refusal(fields)
        if expected is None:
            CellSandbox(**fields)
            continue
        with pytest.raises(pydantic.ValidationError) as refused:
            CellSandbox(**fields)
        assert expected in str(refused.value), case

    policy = CellSandbox(**_fields(tmp_path, forbidden_roots=(looped,)))
    folder = str(a / "x")
    monkeypatch.setattr(sandbox, "_interpreter_dirs", lambda: [folder])
    expected = _first_overlap(folder, policy._denied_roots())
    assert expected == str(looped)
    with pytest.raises(SandboxUnavailable, match=f"the interpreter folder {folder} contains {expected}$"):
        policy.launch(["/venv/bin/python"], env={})


def test_a_path_a_thousand_folders_deep_is_checked_like_any_other(tmp_path, policy):
    deep = Path(str(tmp_path) + "/deep" * 1000)
    assert CellSandbox(**_fields(tmp_path, read_only_dirs=(deep,))).read_only_dirs == (deep,)
    assert _ca_rules(policy, {"SSL_CERT_FILE": str(deep / "ca.pem")}) == {}


def test_a_root_that_cannot_be_resolved_raises_only_where_the_pairwise_check_did(tmp_path, monkeypatch):
    nul_root = Path("/nul\0root")
    # A grant refused by a root listed before the unresolvable one is refused
    # for that root; a grant that isn't normalized is refused for that first.
    for case, expected in (
        ({"project_dir": Path("/tmp")}, "grant /tmp equals or contains /tmp"),
        ({"project_dir": Path("/x/../y")}, "'/x/../y' must be an absolute, normalized path"),
    ):
        with pytest.raises(pydantic.ValidationError) as refused:
            CellSandbox(**_fields(tmp_path, forbidden_roots=(nul_root,), **case))
        assert expected in str(refused.value), case
    # A relative HOME under a working folder that is gone can't be resolved either.
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    monkeypatch.setenv("HOME", "relative-home")
    with pytest.raises(pydantic.ValidationError, match="grant /tmp equals or contains /tmp"):
        CellSandbox(**_fields(tmp_path, project_dir=Path("/tmp")))


_PREFIX_CASES = ["", "/", "//", "/a", "/a/", "/ab", "/a/b", "/a/b/", "/a/bc", "//a", "//a/b", "/b"]


def test_the_index_answers_as_the_pairwise_check_does():
    for path in _PREFIX_CASES:
        holders = sandbox._holders(path)
        for root in _PREFIX_CASES:
            assert (root in holders) == sandbox._same_or_inside(path, root), (path, root)
    ordered = sorted(_PREFIX_CASES)
    for root in _PREFIX_CASES:
        found = sandbox._first_at_or_below(ordered, root)
        assert (found is not None) == any(sandbox._same_or_inside(p, root) for p in ordered), root
        assert found is None or sandbox._same_or_inside(found, root), (root, found)


def _people(tmp_path: Path, count: int) -> tuple[Path, tuple[Path, ...], tuple[Path, ...]]:
    """A cowork-home whose shared root holds ``count`` people, as the host passes them."""
    shared_root = tmp_path / "cowork-home" / "shared-root"
    people = [shared_root / f"person-{i:02d}" for i in range(count)]
    for person in people:
        (person / "sandbox").mkdir(parents=True, exist_ok=True)
    forbidden = (tmp_path / "cowork-home", shared_root, *people)
    return people[0], forbidden, tuple(people[1:])


def _person_fields(tmp_path: Path, count: int) -> dict:
    own, forbidden, others = _people(tmp_path, count)
    return _fields(
        tmp_path,
        project_dir=own / "project",
        home_dir=own / "sandbox" / "home",
        tmp_dir=own / "sandbox" / "tmp",
        forbidden_roots=forbidden,
        other_partitions=others,
    )


def test_each_more_person_costs_at_most_two_lstats_and_no_realpath_call(tmp_path, monkeypatch):
    # realpath lstats every folder of a path on each call, so counting lstat
    # catches a realpath call wherever it comes from. The pairwise check made
    # about 180 lstats per person.
    real_lstat, real_realpath = os.lstat, os.path.realpath
    lstats: list[str] = []
    realpaths: list[str] = []

    def counting_lstat(path, *args, **kwargs):
        lstats.append(str(path))
        return real_lstat(path, *args, **kwargs)

    def counting_realpath(path, *args, **kwargs):
        realpaths.append(str(path))
        return real_realpath(path, *args, **kwargs)

    counts = {}
    for count in (1, 53):
        fields = _person_fields(tmp_path / str(count), count)
        policy = CellSandbox(**fields)
        monkeypatch.setattr(sandbox.os, "lstat", counting_lstat)
        monkeypatch.setattr(sandbox.os.path, "realpath", counting_realpath)
        lstats.clear()
        realpaths.clear()
        CellSandbox(**fields)
        built = (len(lstats), len(realpaths))
        lstats.clear()
        realpaths.clear()
        policy.launch(["/venv/bin/python"], env={}).close()
        counts[count] = (built, (len(lstats), len(realpaths)))
        monkeypatch.undo()
    for (more_lstats, more_realpaths), (base_lstats, base_realpaths) in zip(counts[53], counts[1]):
        assert more_lstats - base_lstats <= 2 * 52
        assert more_realpaths == base_realpaths


def test_a_grant_that_reaches_another_persons_folder_through_a_link_is_refused(tmp_path):
    own, forbidden, others = _people(tmp_path, 2)
    alice = others[0]
    own_fields = dict(project_dir=own / "project", home_dir=own / "sandbox" / "home", tmp_dir=own / "sandbox" / "tmp")
    # A link to alice's folder, as a read-only grant and as the project.
    to_alice = tmp_path / "to-alice"
    to_alice.symlink_to(alice, target_is_directory=True)
    with pytest.raises(pydantic.ValidationError, match="equals or contains"):
        CellSandbox(**_fields(tmp_path, **own_fields, forbidden_roots=forbidden, other_partitions=others,
                              read_only_dirs=(to_alice,)))
    with pytest.raises(pydantic.ValidationError, match="another person's partition|equals or contains"):
        CellSandbox(**_fields(tmp_path, **{**own_fields, "project_dir": to_alice / "projects" / "p"},
                              forbidden_roots=forbidden, other_partitions=others))
    # A partition the host names through a linked entry, and a shared root
    # reached through a linked folder above it.
    elsewhere = tmp_path / "elsewhere" / "alice"
    elsewhere.mkdir(parents=True)
    linked_alice = own.parent / "linked-alice"
    linked_alice.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(pydantic.ValidationError, match="another person's partition|equals or contains"):
        CellSandbox(**_fields(tmp_path, **{**own_fields, "project_dir": elsewhere / "projects" / "p"},
                              other_partitions=(linked_alice,)))
    root_link = tmp_path / "root-link"
    root_link.symlink_to(own.parent, target_is_directory=True)
    with pytest.raises(pydantic.ValidationError, match="another person's partition|equals or contains"):
        CellSandbox(**_fields(tmp_path, **{**own_fields, "tmp_dir": alice / "sandbox" / "tmp"},
                              other_partitions=(root_link / alice.name,)))


def test_a_dotdot_path_into_another_persons_folder_is_refused(tmp_path):
    own, forbidden, others = _people(tmp_path, 2)
    sneaky = Path(f"{own}/../{others[0].name}")
    with pytest.raises(pydantic.ValidationError, match="absolute, normalized"):
        CellSandbox(**_person_fields(tmp_path, 2) | {"read_only_dirs": (sneaky,)})
    with pytest.raises(pydantic.ValidationError, match="absolute, normalized"):
        CellSandbox(**_person_fields(tmp_path / "again", 2) | {"other_partitions": (sneaky,)})


def _first_overlap(grant: Path | str, roots) -> str | None:
    """The root the pairwise loop names: the first, in order, that ``grant`` equals or contains."""
    return next((str(root) for root in roots if sandbox._overlaps(grant, root)), None)


def test_each_refusal_names_the_root_the_pairwise_loop_named(tmp_path, monkeypatch):
    fields = _person_fields(tmp_path, 5)
    policy = CellSandbox(**fields)
    roots = policy._denied_roots()
    cowork_home = tmp_path / "cowork-home"
    link = tmp_path / "to-cowork-home"
    link.symlink_to(cowork_home, target_is_directory=True)
    for grant in (cowork_home, cowork_home / "shared-root", link, tmp_path):
        expected = _first_overlap(grant, roots)
        assert expected is not None, grant
        with pytest.raises(pydantic.ValidationError) as refused:
            CellSandbox(**fields | {"read_only_dirs": (grant,)})
        assert f"grant {grant} equals or contains {expected}" in str(refused.value), grant
    # Roots listed out of sorted order: the first listed is named, as before.
    late, early = cowork_home / "z-listed-first", cowork_home / "a-listed-second"
    with pytest.raises(pydantic.ValidationError) as refused:
        CellSandbox(**_fields(tmp_path, forbidden_roots=(late, early), read_only_dirs=(cowork_home,)))
    assert f"grant {cowork_home} equals or contains {late}" in str(refused.value)
    # A writable grant in two partitions at once names the first listed, in
    # either order.
    _, _, others = _people(tmp_path / "nested", 3)
    outer, inner = others[0], others[0] / "inner"
    for listed in ((inner, outer), (outer, inner)):
        with pytest.raises(pydantic.ValidationError) as refused:
            CellSandbox(**_fields(tmp_path, project_dir=inner / "p", other_partitions=listed))
        assert f"sits in another person's partition {listed[0]}" in str(refused.value)

    for folder in (str(cowork_home), str(link), str(tmp_path / "unrelated")):
        monkeypatch.setattr(sandbox, "_interpreter_dirs", lambda folder=folder: [folder])
        expected = _first_overlap(folder, roots)
        if expected is None:
            policy.launch(["/venv/bin/python"], env={}).close()
            continue
        with pytest.raises(SandboxUnavailable, match=f"the interpreter folder {folder} contains {expected}$"):
            policy.launch(["/venv/bin/python"], env={})


# ── The environment ──────────────────────────────────────────────────────────

_SERVER_ENV = {
    "COWORK_AUTH_TOKEN": "tok-secret",
    "COWORK_TENANCY_MODE": "local",
    "DATABASE_URI": "postgresql://cowork:db-secret@postgres/cowork",
    "MINDS_API_KEY": "minds-secret",
    "ANTON_MINDS_API_KEY": "anton-minds-secret",
    "ANTON_UV_PATH": "/home/anton/.local/bin/uv",
    "LD_PRELOAD": "/x.so",
    "LD_LIBRARY_PATH": "/x",
    "HOME": "/home/anton",
    "TMPDIR": "/tmp",
    "UNLISTED": "x",
    "PYTHONHOME": "/x",
    "OPENAI_API_KEY": "sk-cell",
    "OPENAI_BASE_URL": "https://model-route/v1",
    "ANTHROPIC_API_KEY": "ak-secret",
    "ANTON_OPENAI_MODEL": "m",
    "ANTON_ANTHROPIC_API_KEY": "aak-secret",
    "ANTON_SCRATCHPAD_PERSIST_SESSION": "true",
    "ANTON_LANGFUSE_HEADERS": "1",
    "ANTON_MINDS_SSL_VERIFY": "false",
    "ANTON_CELL_INSTALL_TIMEOUT": "300",
    "DS_PG_MAIN__PASSWORD": "pg-secret",
    "PIP_INDEX_URL": "https://pypi.internal/simple",
    "PYTHONPATH": "/opt/venv/lib/python3.12/site-packages",
    "PYTHONUTF8": "1",
    "PATH": "/opt/venv/bin:/usr/bin",
    "LANG": "C.UTF-8",
    "NO_PROXY": "localhost",
    "HTTPS_PROXY": "http://proxy:3128",
    "SSL_CERT_FILE": "/etc/ssl/ca.pem",
    "REQUESTS_CA_BUNDLE": "/etc/ssl/ca.pem",
    "TZ": "UTC",
}


# The keys whose values are secrets: the token, the database URI, the model
# and connection keys.
_SECRET_KEYS = (
    "COWORK_AUTH_TOKEN", "DATABASE_URI", "MINDS_API_KEY", "ANTON_MINDS_API_KEY", "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY", "ANTON_ANTHROPIC_API_KEY", "DS_PG_MAIN__PASSWORD",
)


def test_the_child_environment_is_the_allowlist_plus_the_persons_folders(policy):
    env = policy.child_env(_SERVER_ENV)
    for key in (
        "COWORK_AUTH_TOKEN", "COWORK_TENANCY_MODE", "DATABASE_URI", "MINDS_API_KEY",
        "ANTON_MINDS_API_KEY", "ANTON_UV_PATH", "LD_PRELOAD", "LD_LIBRARY_PATH", "UNLISTED",
        "PYTHONHOME",
    ):
        assert key not in env, key
    for key in (
        "OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "ANTON_OPENAI_MODEL",
        "ANTON_ANTHROPIC_API_KEY", "ANTON_SCRATCHPAD_PERSIST_SESSION", "ANTON_LANGFUSE_HEADERS",
        "ANTON_MINDS_SSL_VERIFY", "ANTON_CELL_INSTALL_TIMEOUT", "DS_PG_MAIN__PASSWORD",
        "PIP_INDEX_URL", "PYTHONPATH", "PYTHONUTF8", "PATH", "LANG", "NO_PROXY", "HTTPS_PROXY",
        "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "TZ",
    ):
        assert env[key] == _SERVER_ENV[key], key
    home, tmp = str(policy.home_dir), str(policy.tmp_dir)
    assert {k: env[k] for k in ("HOME", "TMPDIR", "TMP", "TEMP", "USER", "LOGNAME")} == {
        "HOME": home, "TMPDIR": tmp, "TMP": tmp, "TEMP": tmp, "USER": "bob", "LOGNAME": "bob",
    }
    assert env["MPLCONFIGDIR"] == f"{home}/.config/matplotlib"
    assert env["XDG_CACHE_HOME"] == f"{home}/.cache"
    assert env["XDG_CONFIG_HOME"] == f"{home}/.config"
    assert env["XDG_DATA_HOME"] == f"{home}/.local/share"
    assert env["PIP_CACHE_DIR"] == f"{home}/.cache/pip"
    secrets = ("tok-secret", "db-secret", "minds-secret", "anton-minds-secret")
    assert not any(secret in value for value in env.values() for secret in secrets)


def test_workspace_env_keys_pass_unless_they_repoint_the_interpreter_or_folders(policy):
    overlay = {
        "MY_API_TOKEN": "fine",
        "LD_PRELOAD": "/evil.so",
        "PYTHONSTARTUP": "/evil.py",
        "HOME": "/evil",
        "TMPDIR": "/evil",
        "XDG_CONFIG_HOME": "/evil",
        "SSL_CERT_FILE": "/evil.pem",
        "REQUESTS_CA_BUNDLE": "/evil.pem",
        "PIP_CERT": "/evil.pem",
        "PIP_CACHE_DIR": "/evil",
        "MPLCONFIGDIR": "/evil",
        "COWORK_AUTH_TOKEN": "x",
        "DATABASE_URI": "x",
        "MINDS_API_KEY": "x",
        "ANTON_MINDS_API_KEY": "x",
        "ANTON_UV_PATH": "/evil/uv",
    }
    env = policy.child_env(overlay, overlay_keys=overlay)
    assert env["MY_API_TOKEN"] == "fine"
    assert not any(value.startswith("/evil") for value in env.values())
    for key in ("LD_PRELOAD", "PYTHONSTARTUP", "COWORK_AUTH_TOKEN", "DATABASE_URI", "MINDS_API_KEY",
                "ANTON_MINDS_API_KEY", "ANTON_UV_PATH", "SSL_CERT_FILE", "PIP_CERT"):
        assert key not in env, key
    assert sandbox.overlay_key_allowed("MY_API_TOKEN")
    for key in ("PATH", "PYTHONPATH", "LD_AUDIT", "XDG_DATA_HOME", "USER", "LOGNAME", "TEMP", "TMP",
                "CURL_CA_BUNDLE", "SSL_CERT_DIR"):
        assert not sandbox.overlay_key_allowed(key), key


def test_entries_execve_would_reject_are_dropped_and_logged(policy, caplog):
    overlay = {"": "x", "A=B": "y", "K\0": "z", "NUL_VALUE": "a\0b", "GOOD": "fine"}
    with caplog.at_level(logging.WARNING, logger=sandbox.__name__):
        env = policy.child_env(overlay, overlay_keys=overlay)
    assert env["GOOD"] == "fine"
    for key in ("", "A=B", "K\0", "NUL_VALUE"):
        assert key not in env, key
        assert f"dropped the environment entry {key!r}" in caplog.text, key
    assert "a\0b" not in caplog.text


# A mirror's password, a token in a username, a token in a query string and a
# proxy's password, none of which agent code may get.
_PIP_WITH_CREDENTIALS = {
    "PIP_INDEX_URL": "https://mirror-user:pw-secret@mirror.internal:8443/pypi/simple",
    "PIP_EXTRA_INDEX_URL": "https://tok-secret@extra.internal/simple  https://pypi.org/simple",
    "PIP_FIND_LINKS": "https://links.internal/wheels/?token=q-secret",
    "PIP_PROXY": "http://proxy-user:px-secret@proxy.internal:3128",
    "PIP_TRUSTED_HOST": "mirror.internal",
}


@pytest.fixture
def fresh_pip_log(monkeypatch) -> None:
    monkeypatch.setattr(sandbox, "_LOGGED_PIP_CREDENTIALS", set())


def _sandbox_warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == sandbox.__name__ and r.levelno == logging.WARNING]


@pytest.mark.parametrize("from_dotenv", [False, True], ids=["server", "dotenv"])
def test_pip_urls_reach_agent_code_without_userinfo_or_query(policy, from_dotenv, fresh_pip_log):
    overlay_keys = _PIP_WITH_CREDENTIALS if from_dotenv else ()
    env = policy.child_env(_PIP_WITH_CREDENTIALS, overlay_keys=overlay_keys)
    assert {key: env[key] for key in _PIP_WITH_CREDENTIALS} == {
        "PIP_INDEX_URL": "https://mirror.internal:8443/pypi/simple",
        "PIP_EXTRA_INDEX_URL": "https://extra.internal/simple https://pypi.org/simple",
        "PIP_FIND_LINKS": "https://links.internal/wheels/",
        "PIP_PROXY": "http://proxy.internal:3128",
        "PIP_TRUSTED_HOST": "mirror.internal",
    }
    assert not any("secret" in value for value in env.values())


def test_pip_settings_without_credentials_pass_exactly_as_set(policy, caplog, fresh_pip_log):
    plain = {
        "PIP_INDEX_URL": "https://pypi.internal/simple",
        "PIP_EXTRA_INDEX_URL": "https://a.internal/simple\n  https://b.internal:8080/simple/",
        "PIP_FIND_LINKS": "/wheels https://links.internal/wheels/pkg-1.0.tar.gz#sha256=abc",
        "PIP_CERT": "/etc/ssl/ca.pem",
        "PIP_NO_INPUT": "1",
    }
    with caplog.at_level(logging.WARNING, logger=sandbox.__name__):
        env = policy.child_env(plain)
    assert {key: env[key] for key in plain} == plain
    assert _sandbox_warnings(caplog) == []


def test_each_pip_variable_that_loses_credentials_is_logged_once_without_them(policy, caplog, fresh_pip_log):
    # The environment is built on every spawn.
    with caplog.at_level(logging.WARNING, logger=sandbox.__name__):
        policy.child_env(_PIP_WITH_CREDENTIALS)
        policy.child_env(_PIP_WITH_CREDENTIALS)
    records = _sandbox_warnings(caplog)
    assert sorted(record.args[0] for record in records) == [
        "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS", "PIP_INDEX_URL", "PIP_PROXY",
    ]
    assert not any("secret" in record.getMessage() for record in records)


def test_a_pip_variable_with_a_url_that_cannot_be_parsed_is_dropped(policy, caplog, fresh_pip_log):
    with caplog.at_level(logging.WARNING, logger=sandbox.__name__):
        env = policy.child_env(
            {"PIP_INDEX_URL": "https://mirror-user:pw-secret@[::1/simple", "PIP_NO_INPUT": "1"}
        )
    assert "PIP_INDEX_URL" not in env
    assert env["PIP_NO_INPUT"] == "1"
    assert [record.getMessage() for record in _sandbox_warnings(caplog)] == [
        "sandbox: dropped PIP_INDEX_URL, which holds a URL that cannot be parsed",
    ]


# ── The rules ────────────────────────────────────────────────────────────────


def test_the_rules_grant_the_system_and_the_persons_folders_only(policy, tmp_path):
    shared = tmp_path / "shared"
    skills = tmp_path / "skills"
    ca = tmp_path / "ca.pem"
    full = policy.model_copy(update={"shared_dir": shared, "read_only_dirs": (skills,), "read_only_files": (ca,)})
    private_ca = tmp_path / "private-ca.pem"
    private_ca.write_text("cert")
    rules = {rule.path: rule for rule in full.rules(host_env={"SSL_CERT_FILE": str(private_ca)})}

    for denied in ("/", "/opt", "/opt/cowork", "/home", "/tmp", "/var/tmp", str(Path.home())):
        assert denied not in rules, denied
    assert set(rules["/usr"].access) == {"execute", "read_file", "read_dir"}
    assert set(rules[sys.prefix].access) == {"execute", "read_file", "read_dir"}
    assert set(rules["/proc"].access) == {"read_file", "read_dir"}
    assert set(rules["/sys"].access) == {"read_file", "read_dir"}
    assert rules["/etc/passwd"].access == ("read_file",)
    assert rules[str(private_ca)].access == ("read_file",)
    assert set(rules["/dev/null"].access) == {"read_file", "write_file"}
    assert rules["/dev/urandom"].access == ("read_file",)
    assert "read_dir" not in rules["/dev/shm"].access
    assert {"make_reg", "remove_file", "truncate"} <= set(rules["/dev/shm"].access)

    project = rules[str(policy.project_dir)]
    home = rules[str(policy.home_dir)]
    tmp = rules[str(policy.tmp_dir)]
    assert project.required and home.required and tmp.required
    assert {"execute", "refer", "write_file", "truncate"} <= set(project.access)
    assert {"execute", "refer", "write_file"} <= set(home.access)
    assert {"execute", "refer", "write_file"} <= set(tmp.access)
    assert "make_sock" in tmp.access and "make_sock" not in project.access
    special = {"make_sym", "make_fifo", "make_char", "make_block"}
    for rule in (project, home, tmp, rules[str(shared)]):
        assert not special & set(rule.access), rule
    assert not {"refer", "execute", "make_sock"} & set(rules[str(shared)].access)
    assert not rules[str(shared)].required
    assert set(rules[str(skills)].access) == {"read_file", "read_dir"} and not rules[str(skills)].required
    assert rules[str(ca)].access == ("read_file",)
    person = {str(policy.project_dir), str(policy.home_dir), str(policy.tmp_dir), str(shared), str(skills), str(ca)}
    for path, rule in rules.items():
        # A person's folders refuse a link; system folders such as /bin and
        # /lib are links on Debian and keep following them.
        assert rule.refuse_links == (path in person), path


def _ca_rules(policy: CellSandbox, host_env: dict[str, str]) -> dict[str, tuple[str, ...]]:
    """The rules ``host_env``'s certificate variables add."""
    without = {rule.path for rule in policy.rules(host_env={})}
    return {rule.path: rule.access for rule in policy.rules(host_env=host_env) if rule.path not in without}


def test_every_ca_variable_gets_a_read_rule_for_its_file_or_folder(policy, tmp_path):
    bundle, curl, certs = tmp_path / "bundle.pem", tmp_path / "curl.pem", tmp_path / "certs"
    bundle.write_text("cert")
    curl.write_text("cert")
    certs.mkdir()
    host_env = {
        "SSL_CERT_FILE": str(bundle), "REQUESTS_CA_BUNDLE": str(bundle), "PIP_CERT": str(bundle),
        "CURL_CA_BUNDLE": str(curl), "SSL_CERT_DIR": str(certs),
    }
    assert _ca_rules(policy, host_env) == {
        str(bundle): ("read_file",), str(curl): ("read_file",), str(certs): ("read_file", "read_dir"),
    }


def test_a_ca_variable_naming_a_folder_or_a_denied_root_gets_no_rule(policy, tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(sandbox, "_LOGGED_CA_PROBLEMS", set())
    cowork_home = tmp_path / "cowork-home"
    cowork_home.mkdir()
    (cowork_home / ".env").write_text("COWORK_AUTH_TOKEN=x")
    guarded = policy.model_copy(update={"forbidden_roots": (cowork_home,)})
    host_env = {
        "REQUESTS_CA_BUNDLE": str(cowork_home),
        "SSL_CERT_FILE": str(tmp_path),
        "PIP_CERT": str(tmp_path / "missing.pem"),
        "CURL_CA_BUNDLE": "relative.pem",
        "SSL_CERT_DIR": "/etc",
    }
    with caplog.at_level(logging.WARNING, logger=sandbox.__name__):
        assert _ca_rules(guarded, host_env) == {}
    for name in host_env:
        assert f"no read rule for {name}=" in caplog.text, name

    # The rules are built on every spawn; each bad setting is logged once.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=sandbox.__name__):
        assert _ca_rules(guarded, host_env) == {}
    assert "no read rule" not in caplog.text


# ── The launch ───────────────────────────────────────────────────────────────


def test_launch_hands_the_launcher_its_policy_by_descriptor_never_in_argv(policy):
    target = ["/venv/bin/python", "-c", "print('ok')"]
    spawn = policy.launch(target, env=policy.child_env(_SERVER_ENV))
    try:
        fd, refusal_write = spawn.pass_fds
        assert spawn.argv == [
            sys.executable, "-I", "-S", str(sandbox.launcher_path()),
            "--policy-fd", str(fd), "--refusal-fd", str(refusal_write), "--", *target,
        ]
        assert spawn.env == {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"}
        argv_text = " ".join(spawn.argv)
        for key in _SECRET_KEYS:
            assert _SERVER_ENV[key] not in argv_text, key
        payload = _read_payload(fd)
        assert payload["version"] == sandbox.LAUNCHER_API_VERSION
        assert payload["env"]["OPENAI_API_KEY"] == "sk-cell"
        assert payload["env"]["HOME"] == str(policy.home_dir)
        assert "COWORK_AUTH_TOKEN" not in payload["env"]
        assert {
            "path": str(policy.project_dir), "access": list(sandbox._PROJECT_RIGHTS), "required": True,
            "refuse_links": True,
        } in payload["rules"]
    finally:
        spawn.close()
    for closed in (fd, refusal_write):
        with pytest.raises(OSError):
            os.fstat(closed)
    spawn.close()
    assert spawn.refusal() is None


def test_launch_refuses_an_environment_that_skipped_the_allowlist(policy):
    with pytest.raises(ValueError, match="child_env"):
        policy.launch(["/venv/bin/python"], env={"COWORK_AUTH_TOKEN": "tok"})
    with pytest.raises(ValueError, match="absolute"):
        policy.launch(["python"], env={})


def test_launch_refuses_an_interpreter_the_cell_could_write(policy, monkeypatch):
    monkeypatch.setattr(sandbox.sys, "executable", str(policy.project_dir / "bin" / "python"))
    with pytest.raises(SandboxUnavailable, match="writable grant"):
        policy.launch(["/venv/bin/python"], env={})


def test_launch_refuses_a_launcher_or_standard_library_the_cell_could_write(policy, monkeypatch):
    monkeypatch.setattr(sandbox, "launcher_path", lambda: policy.tmp_dir / "landlock_launcher.py")
    with pytest.raises(SandboxUnavailable, match="the launcher .* sits under the writable grant"):
        policy.launch(["/venv/bin/python"], env={})
    monkeypatch.undo()
    real_get_path = sandbox.sysconfig.get_path
    monkeypatch.setattr(
        sandbox.sysconfig, "get_path",
        lambda name, *a, **k: str(policy.home_dir / "lib") if name == "stdlib" else real_get_path(name, *a, **k),
    )
    with pytest.raises(SandboxUnavailable, match="the standard library .* sits under the writable grant"):
        policy.launch(["/venv/bin/python"], env={})


def test_launch_logs_an_optional_grant_reached_through_a_link(policy, tmp_path, caplog):
    real = tmp_path / "skills"
    real.mkdir()
    link = tmp_path / "skills-link"
    link.symlink_to(real, target_is_directory=True)
    linked = CellSandbox(**_fields(tmp_path, read_only_dirs=(link,)))
    with caplog.at_level(logging.WARNING, logger=sandbox.__name__):
        spawn = linked.launch(["/venv/bin/python"], env={})
    spawn.close()
    assert spawn.refusal() is None
    assert f"sandbox grant skipped, path is a link or sits below one: {link}" in caplog.text


def test_a_refusal_is_read_only_from_the_launchers_refusal_pipe(policy):
    line = f"{SANDBOX_REFUSED_MARKER} the kernel's Landlock ABI is 2, below 3 (Linux 6.2)"
    refused = policy.launch(["/venv/bin/python"], env={})
    os.write(refused.pass_fds[1], f"{line}\n".encode())
    refused.close()
    refusal = refused.refusal()
    assert isinstance(refusal, SandboxUnavailable)
    assert str(refusal) == f"SandboxUnavailable: {line}"
    assert refused.refusal() is refusal

    # A launcher that exec'd its target closed the pipe unwritten.
    started = policy.launch(["/venv/bin/python"], env={})
    started.close()
    assert started.refusal() is None

    # A process still running holds the write end: no refusal, and no wait.
    running = policy.launch(["/venv/bin/python"], env={})
    keep = os.dup(running.pass_fds[1])
    try:
        running.close()
        assert running.refusal() is None
    finally:
        os.close(keep)


def test_the_abi_probe_never_raises():
    abi = sandbox.landlock_abi()
    assert isinstance(abi, int) and abi >= 0
    if not sys.platform.startswith("linux"):
        assert abi == 0


def test_a_pools_policy_is_read_only_when_it_is_one(policy):
    assert cell_sandbox_of(SimpleNamespace(cell_sandbox=policy)) is policy
    assert cell_sandbox_of(SimpleNamespace(cell_sandbox=None)) is None
    assert cell_sandbox_of(SimpleNamespace()) is None
    assert cell_sandbox_of(None) is None


@pytest.mark.parametrize("value", [MagicMock(), {"project_dir": "/srv/p"}, "policy"])
def test_a_pool_whose_policy_is_not_a_policy_is_refused(policy, value):
    with pytest.raises(TypeError, match="not a CellSandbox"):
        cell_sandbox_of(SimpleNamespace(cell_sandbox=value))


# ── Pad names ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "", "/home/anton/.cowork", "../../..", "a/../../b", "x\0y", ".", "./", "...", "C:\\Users",
        "\\\\host\\share", "c:foo", "\\x", "a\\..\\b", ".\\.",
    ],
)
def test_a_pad_name_that_leaves_the_venvs_folder_is_refused(name):
    with pytest.raises(ValueError):
        validate_pad_name(name)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("a/b", "one folder name"),
        ("a\\b", "one folder name"),
        ("q:forecast", "drive or stream separator"),
        ("1:x", "drive or stream separator"),
        ("step:1", "drive or stream separator"),
        ("x.", "must not end with '.' or a space"),
        ("x/...", "one folder name"),
        ("main ", "must not end with '.' or a space"),
    ],
)
def test_a_pad_name_that_nests_or_aliases_another_pad_is_refused(name, message):
    with pytest.raises(ValueError, match=message):
        validate_pad_name(name)


@pytest.mark.parametrize("name", ["main", "my pad", "artifact-slug_1", "v1..2", ".hidden", "__datasource_test__"])
def test_ordinary_pad_names_pass(name):
    assert validate_pad_name(name) == name


# ── Plumbing: the policy is never dropped ────────────────────────────────────


def _factory_without_policy(*, name, cells, coding_provider, coding_model, coding_api_key,
                            coding_base_url, workspace_path, session_id=None):
    return SimpleNamespace(name=name, start=_noop)


async def _noop():
    return None


_MANAGER_ARGS = dict(coding_provider="anthropic", coding_model="", coding_api_key="", coding_base_url="")


def test_a_factory_that_cannot_take_the_policy_refuses_the_manager(policy):
    with pytest.raises(TypeError, match="cell_sandbox"):
        ScratchpadManager(runtime_factory=_factory_without_policy, cell_sandbox=policy, **_MANAGER_ARGS)


async def test_without_a_policy_an_older_factory_still_works():
    manager = ScratchpadManager(runtime_factory=_factory_without_policy, **_MANAGER_ARGS)
    assert manager.cell_sandbox is None
    pad = await manager.get_or_create("main")
    assert pad.name == "main"


async def test_the_manager_hands_the_policy_to_each_pad(policy):
    seen: list[dict] = []

    def factory(*, cell_sandbox=None, **kwargs):
        seen.append({**kwargs, **({"cell_sandbox": cell_sandbox} if cell_sandbox is not None else {})})
        return SimpleNamespace(start=_noop, cell_sandbox=cell_sandbox)

    with_policy = ScratchpadManager(runtime_factory=factory, cell_sandbox=policy, **_MANAGER_ARGS)
    await with_policy.get_or_create("main")
    without = ScratchpadManager(runtime_factory=factory, **_MANAGER_ARGS)
    await without.get_or_create("main")
    assert with_policy.cell_sandbox is policy
    assert seen[0]["cell_sandbox"] is policy
    assert "cell_sandbox" not in seen[1]


def test_a_factory_that_takes_only_bare_kwargs_refuses_the_manager(policy):
    def factory(**kwargs):
        return SimpleNamespace(start=_noop)

    with pytest.raises(TypeError, match="does not name a cell_sandbox parameter"):
        ScratchpadManager(runtime_factory=factory, cell_sandbox=policy, **_MANAGER_ARGS)
    # Without a policy the same factory still works.
    ScratchpadManager(runtime_factory=factory, **_MANAGER_ARGS)


async def test_a_runtime_that_drops_the_policy_is_never_started(policy):
    started: list[str] = []

    async def start():
        started.append("started")

    def factory(*, cell_sandbox=None, **kwargs):
        return SimpleNamespace(start=start, cell_sandbox=None)

    manager = ScratchpadManager(runtime_factory=factory, cell_sandbox=policy, **_MANAGER_ARGS)
    with pytest.raises(TypeError, match="does not run under the session's sandbox policy"):
        await manager.get_or_create("main")
    assert started == []
    assert manager.pads == {}


def test_the_local_factory_builds_a_locked_runtime(policy):
    pad = local_scratchpad_runtime_factory(
        name="main", cells=None, workspace_path=None, cell_sandbox=policy, **_MANAGER_ARGS
    )
    assert isinstance(pad, LocalScratchpadRuntime)
    assert pad.cell_sandbox is policy


def test_a_session_passes_its_policy_to_its_scratchpads(make_session, policy):
    assert make_session()._scratchpads.cell_sandbox is None
    assert make_session(cell_sandbox=policy)._scratchpads.cell_sandbox is policy


def test_a_session_refuses_a_factory_that_would_drop_its_policy(make_session, policy):
    with pytest.raises(TypeError, match="cell_sandbox"):
        make_session(cell_sandbox=policy, runtime_factory=_factory_without_policy)


def test_chat_session_config_defaults_to_no_sandbox():
    from anton.core.session import ChatSessionConfig

    assert ChatSessionConfig(llm_client=MagicMock()).cell_sandbox is None


def test_allows_read_is_the_grants_resolved_and_nothing_else(tmp_path):
    for name in ("project", "home", "tmp", "shared", "skill", "project-evil", "outside"):
        (tmp_path / name).mkdir()
    (tmp_path / "notes.txt").write_text("x")
    policy = CellSandbox(**_fields(
        tmp_path,
        shared_dir=tmp_path / "shared",
        read_only_dirs=(tmp_path / "skill",),
        read_only_files=(tmp_path / "notes.txt",),
    ))
    for granted in ("project", "home", "tmp", "shared", "skill"):
        assert policy.allows_read(tmp_path / granted / "a" / "b.png")
        assert policy.allows_read(tmp_path / granted)
    assert policy.allows_read(tmp_path / "notes.txt")
    # A sibling whose name starts with a grant's name is not inside it.
    assert not policy.allows_read(tmp_path / "project-evil" / "b.png")
    assert not policy.allows_read(tmp_path / "outside" / "b.png")
    # The path is resolved here, whatever the caller passed.
    (tmp_path / "project" / "link").symlink_to(tmp_path / "outside", target_is_directory=True)
    assert not policy.allows_read(tmp_path / "project" / "link" / "b.png")
    assert not policy.allows_read(tmp_path / "project" / ".." / "outside" / "b.png")
    # The system paths rules() adds are not grants.
    assert not policy.allows_read("/etc/hosts")
    assert not policy.allows_read("/usr/share")
