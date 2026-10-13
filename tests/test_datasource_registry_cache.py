"""DatasourceRegistry() reuses YAML parses without sharing objects or going stale.

A turn builds several registries, so each one re-parsing datasources.md was
most of a turn's setup CPU. These pin that a registry built from unchanged
files does no YAML parsing, that an edit is always seen, and that no registry
can change what another one returns.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from anton.core.datasources.datasource_registry import (
    DatasourceField,
    DatasourceRegistry,
    _parse_file,
)

_ACME = """\
## Acme

```yaml
engine: acme
display_name: {display}
name_from: [host, database]
fields:
  - {{ name: host, required: true, description: hostname }}
  - {{ name: password, required: true, secret: true }}
auth_method: choice
auth_methods:
  - name: token
    display: Token
    fields:
      - {{ name: token, required: true, secret: true }}
```
"""


@pytest.fixture()
def builtin_md(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "builtin.md"
    path.write_text(_ACME.format(display="Acme"), encoding="utf-8")
    monkeypatch.setattr(DatasourceRegistry, "_BUILTIN_PATH", path)
    return path


@pytest.fixture()
def user_md(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "user.md"
    monkeypatch.setattr(DatasourceRegistry, "_USER_PATH", path)
    return path


def _rewrite_keeping_stat(path: Path, text: str) -> None:
    """Rewrite `path` in place with same-size `text` and put its mtime back.

    A cache keyed on path, mtime and size would serve the old parse here.
    """
    before = path.stat()
    assert len(text.encode()) == before.st_size
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def test_unchanged_files_are_parsed_once(builtin_md, user_md, yaml_parses):
    user_md.write_text(
        _ACME.format(display="Custom").replace("engine: acme", "engine: custom"),
        encoding="utf-8",
    )
    first = DatasourceRegistry()
    second = DatasourceRegistry()

    assert len(yaml_parses) == 2  # one block per file, on the first registry only
    assert [e.engine for e in second.all_engines()] == ["acme", "custom"]
    assert first.get("custom").fields == second.get("custom").fields


def test_edited_user_file_is_seen_on_next_construction_and_reload(builtin_md, user_md):
    user_md.write_text(_ACME.format(display="Acme v1"), encoding="utf-8")
    registry = DatasourceRegistry()
    assert registry.get("acme").display_name == "Acme v1"

    _rewrite_keeping_stat(user_md, _ACME.format(display="Acme v2"))
    assert DatasourceRegistry().get("acme").display_name == "Acme v2"

    _rewrite_keeping_stat(user_md, _ACME.format(display="Acme v3"))
    registry.reload()
    assert registry.get("acme").display_name == "Acme v3"


def test_new_user_file_is_seen_on_next_construction(builtin_md, user_md):
    assert DatasourceRegistry().get("acme").custom is False

    user_md.write_text(_ACME.format(display="Acme (mine)"), encoding="utf-8")
    engine = DatasourceRegistry().get("acme")
    assert engine.display_name == "Acme (mine)"
    assert engine.custom is True


def test_validate_file_sees_an_edit(tmp_path):
    draft = tmp_path / "datasources.tmp"
    draft.write_text(_ACME.format(display="Draft 1"), encoding="utf-8")
    registry = DatasourceRegistry.__new__(DatasourceRegistry)
    assert registry.validate_file(draft)["acme"].display_name == "Draft 1"

    _rewrite_keeping_stat(draft, _ACME.format(display="Draft 2"))
    assert registry.validate_file(draft)["acme"].display_name == "Draft 2"


def test_mutating_one_registry_leaks_into_no_other(builtin_md, user_md):
    engine = DatasourceRegistry().get("acme")
    engine.display_name = "changed"
    engine.name_from.append("port")
    engine.fields[0].required = False
    engine.fields.append(DatasourceField(name="extra"))
    engine.auth_methods[0].fields[0].secret = False

    fresh = DatasourceRegistry().get("acme")
    assert fresh.display_name == "Acme"
    assert fresh.name_from == ["host", "database"]
    assert [(f.name, f.required) for f in fresh.fields] == [
        ("host", True),
        ("password", True),
    ]
    assert fresh.auth_methods[0].fields[0].secret is True


def test_custom_parse_does_not_make_builtin_fields_optional(tmp_path):
    """The same block read as a custom engine and as a built-in one."""
    path = tmp_path / "datasources.md"
    path.write_text(_ACME.format(display="Acme"), encoding="utf-8")

    custom = _parse_file(path, custom=True)["acme"]
    builtin = _parse_file(path)["acme"]

    assert [f.required for f in custom.fields] == [False, False]
    assert [f.required for f in builtin.fields] == [True, True]
    assert builtin.auth_methods[0].fields[0].required is True
    assert builtin.custom is False


def test_malformed_block_warns_on_every_construction(builtin_md, user_md, capsys):
    user_md.write_text(
        "## Broken\n\n```yaml\nengine: broken\nfields: [unclosed\n```\n",
        encoding="utf-8",
    )
    DatasourceRegistry()
    DatasourceRegistry()

    err = capsys.readouterr().err
    assert err.count("skipping malformed YAML block") == 2


@pytest.mark.parametrize(
    "value",
    ["1" * 5000, "2026-13-45", "[" * 2000 + "]" * 2000],
    ids=["integer-too-long", "date-out-of-range", "nested-too-deep"],
)
def test_a_block_yaml_cannot_construct_is_skipped(builtin_md, user_md, capsys, value):
    user_md.write_text(
        f"## Broken\n\n```yaml\nengine: broken\npip: {value}\n```\n",
        encoding="utf-8",
    )
    registry = DatasourceRegistry()

    assert registry.get("broken") is None
    assert registry.get("acme") is not None
    assert "skipping malformed YAML block" in capsys.readouterr().err
