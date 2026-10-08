"""Published bundles hold only artifact content from inside the artifact folder:
no owner-side files and nothing a symlink pulls in from outside it."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from anton.publisher import _zip_fullstack, _zip_html

ASSETS = ("style.css", "app.js", "img/logo.png", "img/bg.png")


def _names(zbytes: bytes) -> set[str]:
    with zipfile.ZipFile(io.BytesIO(zbytes)) as bundle:
        return set(bundle.namelist())


def _assert_no_leak(zbytes: bytes) -> None:
    with zipfile.ZipFile(io.BytesIO(zbytes)) as bundle:
        data = b"".join(bundle.read(name) for name in bundle.namelist())
    assert b"owner-secret" not in data
    assert b"OUTSIDE_SECRET" not in data


def _make_artifact(tmp_path: Path, content_dir: str = "") -> Path:
    """An artifact folder with normal assets, owner-side files and symlinks.

    Assets go under *content_dir* (``static`` for fullstack). Next to them:
    `.published.json`, a `.revisions/` entry, a symlink to a file outside the
    folder and a symlink aliasing `.published.json` under an innocent name.
    """
    artifact = tmp_path / "sales"
    content = artifact / content_dir
    (content / "img").mkdir(parents=True)
    for asset in ASSETS:
        (content / asset).write_bytes(b"asset")
    (artifact / ".published.json").write_text('{"access_password": "owner-secret"}')
    (artifact / ".revisions" / "entries").mkdir(parents=True)
    (artifact / ".revisions" / "entries" / "old.html").write_text("owner-secret")
    outside = tmp_path / "outside.env"
    outside.write_text("OUTSIDE_SECRET=1")
    (content / "data.txt").symlink_to(outside)
    (content / "settings.json").symlink_to(artifact / ".published.json")
    return artifact


def test_single_file_bundles_assets_but_not_owner_side_or_outside_files(tmp_path):
    artifact = _make_artifact(tmp_path)
    html = artifact / "index.html"
    html.write_text(
        '<link href="style.css"><script src="app.js"></script><img src="img/logo.png">'
        "<style>body { background: url('img/bg.png') }</style>"
        '<a href=".published.json">x</a><a href=".revisions/entries/old.html">x</a>'
        '<a href="data.txt">x</a><a href="settings.json">x</a>'
    )

    zbytes = _zip_html(html)

    assert _names(zbytes) == {"index.html", *ASSETS}
    _assert_no_leak(zbytes)


def test_single_file_skips_sibling_folder_with_same_name_prefix(tmp_path):
    artifact = tmp_path / "sales"
    artifact.mkdir()
    (artifact / "style.css").write_text("own")
    (tmp_path / "sales-old").mkdir()
    (tmp_path / "sales-old" / "style.css").write_text("sibling")
    html = artifact / "index.html"
    html.write_text('<link href="style.css"><link href="../sales-old/style.css">')

    assert _names(_zip_html(html)) == {"index.html", "style.css"}


def test_directory_bundles_assets_but_not_owner_side_or_outside_files(tmp_path):
    artifact = _make_artifact(tmp_path)
    (artifact / "index.html").write_text("<h1>Report</h1>")

    zbytes = _zip_html(artifact)

    assert _names(zbytes) == {"index.html", *ASSETS}
    _assert_no_leak(zbytes)


def test_fullstack_bundles_assets_but_not_owner_side_or_outside_files(tmp_path):
    artifact = _make_artifact(tmp_path, content_dir="static")
    (artifact / "backend.py").write_text("app = None\n")
    (artifact / "static" / ".published.json").write_text("owner-secret")
    (artifact / "static" / ".revisions").mkdir()
    (artifact / "static" / ".revisions" / "old.js").write_text("owner-secret")
    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE_SECRET=1")
    (artifact / "requirements.txt").symlink_to(outside)

    zbytes, included = _zip_fullstack(artifact)

    own = {name for name in _names(zbytes) if not name.startswith("anton_state/")}
    assert own == {"backend.py", *(f"static/{asset}" for asset in ASSETS)}
    assert set(included) >= own
    _assert_no_leak(zbytes)
