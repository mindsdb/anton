"""Each artifact's main page as the turn found it (`page_copies.TurnPages`):
what is kept, what is put back, and the lines that close the turn."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from anton.core.artifacts import ArtifactStore, render_check
from anton.core.artifacts import page_copies
from anton.core.artifacts.page_copies import TurnPages, main_page_name
from anton.core.artifacts.render_check import Comparison, ComparisonNotes
from anton.core.tools.tool_handlers import PageKey


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    """Copies are kept only with a render-check service to compare them."""
    monkeypatch.setenv("ANTON_RENDER_CHECK_URL", "http://127.0.0.1:9")


def _artifact(store: ArtifactStore, name: str = "Control tower", *, pages=None, primary=None, type="html-app") -> str:
    slug = store.create(name=name, description="d", type=type, primary=primary).slug
    for page, text in (pages or {"index.html": "<p>before</p>"}).items():
        path = store.root / slug / page
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode() if isinstance(text, str) else text)
    return slug


def _captured(tmp_path: Path, **artifact) -> tuple[ArtifactStore, str, PageKey, TurnPages]:
    store = ArtifactStore(tmp_path / "artifacts")
    slug = _artifact(store, **artifact)
    pages = TurnPages()
    pages.capture(store, cell_sandbox=None)
    return store, slug, PageKey(slug, "index.html"), pages


def _comparison(**fields) -> Comparison:
    """A comparison that covered the new page (`removal_checked: all`) unless FIELDS say otherwise."""
    return Comparison.model_validate(
        {"previous": "complete", "previous_status": "passed", "removal_checked": "all", **fields}
    )


_BROKE_MAP = {"pressed": "Map", "reason": "new_problem", "viewports": ["desktop"], "findings": []}
_BLANK_DETAILS = {"pressed": "Details", "reason": "no_view", "viewports": ["phone"], "findings": []}
_NOT_CHECKED = {"previous": "not_checked", "previous_status": None}
_BACK = "“Control tower”: the edit could not be made safely, so the page is back as it was before this turn."


# ── What the turn keeps ──────────────────────────────────────────────────────


def test_the_first_capture_keeps_each_main_page_as_the_turn_found_it(tmp_path):
    store, slug, key, pages = _captured(tmp_path)
    (store.root / slug / "index.html").write_text("<p>after</p>")
    new = _artifact(store, "New")

    pages.capture(store, cell_sandbox=None)

    assert pages.previous(key) == "<p>before</p>"
    assert pages.main_previous(slug) == (key, "<p>before</p>")
    assert pages.main_previous(new) is None


@pytest.mark.parametrize(("files", "primary", "main"), [
    (["a.html", "index.html", "report.html"], "report.html", "report.html"),
    (["a.html", "index.html"], "missing.html", "index.html"),
    (["b.html", "a.html"], None, "a.html"),
    (["data.csv"], None, None),
])
def test_the_main_page_is_the_primary_then_index_then_the_first_by_name(tmp_path, files, primary, main):
    for name in files:
        (tmp_path / name).write_text("x")
    assert main_page_name(tmp_path, primary) == main


def test_no_copy_of_a_fullstack_page_a_page_over_the_limit_or_one_not_utf8(tmp_path, monkeypatch):
    monkeypatch.setattr(render_check, "MAX_PAGE_BYTES", 64)
    store = ArtifactStore(tmp_path / "artifacts")
    app = _artifact(store, "App", type="fullstack-stateless-app")
    big = _artifact(store, "Big", pages={"index.html": "x" * 65})
    latin = _artifact(store, "Latin", pages={"index.html": b"\xff\xfe<p>"})
    ok = _artifact(store, "Ok")

    pages = TurnPages()
    pages.capture(store, cell_sandbox=None)

    assert [pages.main_previous(slug) is not None for slug in (app, big, latin, ok)] == [False, False, False, True]


@pytest.mark.skipif(sys.platform == "win32", reason="symbolic links need privileges on Windows")
def test_a_folder_without_metadata_is_skipped_without_a_warning(tmp_path, caplog):
    store = ArtifactStore(tmp_path / "artifacts")
    slug = _artifact(store)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "index.html").write_text("<p>other</p>")
    os.symlink(elsewhere, store.root / "linked")
    (store.root / "stray").mkdir()

    pages = TurnPages()
    with caplog.at_level("DEBUG"):
        pages.capture(store, cell_sandbox=None)

    assert not [record for record in caplog.records if record.levelname in ("WARNING", "ERROR")]
    assert pages.main_previous(slug) is not None


def test_no_copy_of_a_page_the_persons_cells_may_not_read(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    slug = _artifact(store)

    pages = TurnPages()
    pages.capture(store, cell_sandbox=SimpleNamespace(allows_read=lambda path: False))

    assert pages.main_previous(slug) is None


def test_past_the_turns_limit_the_newest_pages_keep_their_copies(tmp_path, monkeypatch):
    monkeypatch.setattr(page_copies, "MAX_TURN_COPY_BYTES", 10)
    store = ArtifactStore(tmp_path / "artifacts")
    old = _artifact(store, "Old", pages={"index.html": "<old/>"})
    new = _artifact(store, "New", pages={"index.html": "<new/>"})
    os.utime(store.root / old / "index.html", ns=(1_000_000_000, 1_000_000_000))

    pages = TurnPages()
    pages.capture(store, cell_sandbox=None)

    assert pages.main_previous(new) is not None
    assert pages.main_previous(old) is None


def test_a_page_that_cannot_fit_the_turns_limit_is_never_read(tmp_path, monkeypatch):
    monkeypatch.setattr(page_copies, "MAX_TURN_COPY_BYTES", 10)
    store = ArtifactStore(tmp_path / "artifacts")
    small = _artifact(store, "Small", pages={"index.html": "<p/>"})
    big = _artifact(store, "Big", pages={"index.html": "<p>too big</p>"})
    asked: list[Path] = []
    read_page = page_copies._read_page

    def recording(path, *, cell_sandbox):
        asked.append(path)
        return read_page(path, cell_sandbox=cell_sandbox)

    monkeypatch.setattr(page_copies, "_read_page", recording)

    pages = TurnPages()
    pages.capture(store, cell_sandbox=None)

    assert asked == [store.root / small / "index.html"]
    assert pages.main_previous(small) is not None and pages.main_previous(big) is None


def test_a_capture_that_failed_keeps_no_copies_later(tmp_path, monkeypatch):
    """A later capture would copy pages the turn already edited: no copies
    is safer than late ones."""
    store = ArtifactStore(tmp_path / "artifacts")
    slug = _artifact(store)
    pages = TurnPages()
    read_copies = pages._read_copies
    calls = []

    def failing_once(store, *, cell_sandbox):
        calls.append(None)
        if len(calls) == 1:
            raise OSError("disk went away")
        read_copies(store, cell_sandbox=cell_sandbox)

    monkeypatch.setattr(pages, "_read_copies", failing_once)

    with pytest.raises(OSError):
        pages.capture(store, cell_sandbox=None)
    (store.root / slug / "index.html").write_text("<p>edited</p>")
    pages.capture(store, cell_sandbox=None)

    assert pages.captured is True
    assert pages.main_previous(slug) is None


async def test_once_the_turn_has_its_copies_a_capture_does_nothing(tmp_path, monkeypatch, caplog):
    from anton.core.tools import tool_handlers

    store = ArtifactStore(tmp_path / "artifacts")
    _artifact(store)
    session = SimpleNamespace(_turn_pages=TurnPages(), _scratchpads=None)
    await tool_handlers.capture_turn_pages(session, store)
    assert session._turn_pages.captured is True

    def fail(*args, **kwargs):
        raise AssertionError("capture called again")

    monkeypatch.setattr(TurnPages, "capture", fail)
    await tool_handlers.capture_turn_pages(session, store)

    assert "keeping the turn's copies" not in caplog.text


def test_without_the_render_check_service_nothing_is_kept(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTON_RENDER_CHECK_URL")
    _, slug, _, pages = _captured(tmp_path)
    assert pages.main_previous(slug) is None


# ── Putting a page back ──────────────────────────────────────────────────────


def test_restore_puts_the_page_back_byte_for_byte(tmp_path):
    store, slug, key, pages = _captured(tmp_path, pages={"index.html": b"<p>a\r\nb</p>"})
    (store.root / slug / "index.html").write_text("<p>broken</p>")
    pages.record({key: _comparison(broken_views=[_BROKE_MAP])})

    assert pages.restore(key, store) is True

    assert (store.root / slug / "index.html").read_bytes() == b"<p>a\r\nb</p>"
    assert pages.end_lines(store, checks={}) == [_BACK]


@pytest.mark.skipif(sys.platform == "win32", reason="links below the root are refused where dir_fd exists")
def test_restore_never_replaces_a_link_at_the_page(tmp_path):
    store, slug, key, pages = _captured(tmp_path)
    outside = tmp_path / "outside.html"
    outside.write_text("theirs")
    page = store.root / slug / "index.html"
    page.unlink()
    page.symlink_to(outside)

    assert pages.restore(key, store) is False

    assert page.is_symlink() and outside.read_text() == "theirs"


def test_a_restore_that_cannot_write_says_so(tmp_path, caplog):
    store, slug, key, pages = _captured(tmp_path)
    shutil.rmtree(store.root / slug)

    assert pages.restore(key, store) is False
    assert "could not put" in caplog.text
    assert pages.end_lines(store, checks={}) == []


def test_restore_if_changed_writes_only_a_changed_or_missing_page(tmp_path):
    store, slug, key, pages = _captured(tmp_path)
    page = store.root / slug / "index.html"
    inode = page.stat().st_ino

    assert pages.restore_if_changed(key, store) is False
    assert page.stat().st_ino == inode
    page.unlink()
    assert pages.restore_if_changed(key, store) is True
    assert page.read_text() == "<p>before</p>"
    page.write_text("<p>half written</p>")
    assert pages.restore_if_changed(key, store) is True
    assert page.read_text() == "<p>before</p>"


def test_write_back_puts_a_mid_turn_page_back_without_marking_it_restored(tmp_path, caplog):
    store, slug, key, pages = _captured(tmp_path)
    page = store.root / slug / "index.html"
    page.write_text("<p>the edit</p>")
    pages.record({key: _comparison(removed_views=[{"pressed": "Globe"}], removed_controls=["Globe"])})
    run_start = pages.read_current(key, store)
    inode = page.stat().st_ino

    assert pages.write_back(key, store, run_start) is False
    assert page.stat().st_ino == inode
    page.write_text("<p>half written</p>")
    assert pages.write_back(key, store, run_start) is True

    assert page.read_text() == "<p>the edit</p>"
    assert pages.has_restored() is False
    assert pages.end_lines(store, checks={}) == ["“Control tower” no longer has “Globe”: this turn removed it."]
    shutil.rmtree(store.root / slug)
    assert pages.write_back(key, store, run_start) is False
    assert "could not put" in caplog.text


def test_the_page_on_disk_and_its_copy_read_byte_for_byte(tmp_path):
    store, slug, key, pages = _captured(tmp_path, pages={"index.html": b"<p>a\r\nb</p>"})
    assert pages.copy_bytes(key) == pages.read_current(key, store) == b"<p>a\r\nb</p>"
    assert pages.copy_bytes(PageKey("other", "index.html")) is None
    (store.root / slug / "index.html").unlink()
    assert pages.read_current(key, store) is None


@pytest.mark.skipif(sys.platform == "win32", reason="links below the root are refused where dir_fd exists")
def test_a_link_at_the_page_never_reads_as_the_page_the_turn_found(tmp_path):
    store, slug, key, pages = _captured(tmp_path)
    outside = tmp_path / "outside.html"
    outside.write_text("<p>before</p>")
    page = store.root / slug / "index.html"
    page.unlink()
    page.symlink_to(outside)

    assert pages.is_as_found(key, store) is False


@pytest.mark.parametrize(("latest", "earlier", "restore"), [
    (_comparison(), None, True),
    (_comparison(previous_status="failed"), None, False),
    (_comparison(previous_status="failed", broken_views=[_BROKE_MAP]), None, True),
    (_comparison(**_NOT_CHECKED), _comparison(previous_status="failed"), False),
    (_comparison(**_NOT_CHECKED), None, True),
    (None, None, True),
])
def test_should_restore_reads_the_turns_latest_comparison_that_says_something(tmp_path, latest, earlier, restore):
    _, _, key, pages = _captured(tmp_path)
    if earlier is not None:
        pages.record({key: earlier})
    assert pages.should_restore(key, latest) is restore


def test_a_page_without_a_copy_is_never_put_back():
    assert TurnPages().should_restore(PageKey("x", "index.html"), None) is False


def test_the_status_before_is_the_previous_pages_or_unverified():
    key = PageKey("x", "index.html")
    pages = TurnPages()
    assert pages.status_before(key, None) == "unverified"
    assert pages.status_before(key, _comparison(previous_status="failed")) == "failed"
    # The new page's check was cut short, but the previous page's status is known.
    assert pages.status_before(key, _comparison(previous_status="failed", removal_checked="none")) == "failed"
    pages.record({key: _comparison()})
    assert pages.status_before(key, _comparison(**_NOT_CHECKED)) == "passed"


def test_a_turn_knows_whether_it_put_a_page_back(tmp_path):
    store, _, key, pages = _captured(tmp_path)
    assert pages.has_restored() is False
    pages.restore(key, store)
    assert pages.has_restored() is True


# ── What the turn heard and what ends it ─────────────────────────────────────


@pytest.mark.parametrize("later", [
    _comparison(**_NOT_CHECKED),
    # The new page's check was cut short: the service compared nothing.
    _comparison(previous_status="failed", removal_checked="none"),
])
def test_a_comparison_that_says_nothing_keeps_the_earlier_one(tmp_path, later):
    store, _, key, pages = _captured(tmp_path)
    pages.record({key: _comparison(
        broken_views=[_BROKE_MAP], removed_views=[{"pressed": "Globe"}], removed_controls=["Globe"],
    )})

    pages.record({key: later})

    assert pages.end_lines(store, checks={key: "failed"}) == [
        "“Control tower”: “Map” drew before this turn and fails now.",
        "“Control tower” no longer has “Globe”: this turn removed it.",
    ]
    assert pages.should_restore(key, later) is True


def test_a_page_compared_after_it_was_put_back_is_no_longer_restored(tmp_path):
    store, _, key, pages = _captured(tmp_path)
    pages.restore(key, store)
    pages.record({key: _comparison()})
    assert pages.end_lines(store, checks={}) == []


def test_a_page_edited_after_it_was_put_back_is_no_longer_restored_without_a_comparison(tmp_path):
    store, _, key, pages = _captured(tmp_path)
    other = PageKey("other", "index.html")
    pages.restore(key, store)

    pages.record({}, edited=[other])
    assert pages.end_lines(store, checks={}) == [_BACK]
    pages.record({key: _comparison(**_NOT_CHECKED)}, edited=[key])

    assert pages.has_restored() is False
    assert pages.end_lines(store, checks={}) == []


def test_the_same_blank_and_removed_views_are_shown_once_and_broke_always():
    key = PageKey("x", "index.html")
    pages = TurnPages()
    comparison = _comparison(broken_views=[_BROKE_MAP, _BLANK_DETAILS], removed_views=[{"pressed": "Globe"}])
    notes = ComparisonNotes(broke=["“Map”"], blank=["“Details”"], removed=["“Globe”"], shrank=[])

    assert pages.notes_to_show(key, comparison) == notes
    assert pages.notes_to_show(key, comparison) == ComparisonNotes(broke=["“Map”"], blank=[], removed=[], shrank=[])
    changed = comparison.model_copy(update={"removed_controls": ["Routes"]})
    assert pages.notes_to_show(key, changed) == notes._replace(blank=[], removed=["“Globe”", "“Routes”"])


def test_a_comparison_that_says_nothing_shows_nothing_and_keeps_what_was_shown():
    key = PageKey("x", "index.html")
    pages = TurnPages()
    comparison = _comparison(broken_views=[_BLANK_DETAILS])
    pages.notes_to_show(key, comparison)

    assert pages.notes_to_show(key, None) == ComparisonNotes(broke=[], blank=[], removed=[], shrank=[])
    assert pages.notes_to_show(key, _comparison(**_NOT_CHECKED)) == ComparisonNotes(broke=[], blank=[], removed=[], shrank=[])
    assert pages.notes_to_show(key, comparison).blank == []


def test_after_a_restore_the_same_notes_are_shown_again(tmp_path):
    store, _, key, pages = _captured(tmp_path)
    comparison = _comparison(broken_views=[_BLANK_DETAILS])
    notes = ComparisonNotes(broke=[], blank=["“Details”"], removed=[], shrank=[])
    pages.notes_to_show(key, comparison)

    pages.restore(key, store)

    assert pages.notes_to_show(key, comparison) == notes


def test_the_turns_last_lines_name_what_it_broke_blanked_and_removed(tmp_path):
    store, _, key, pages = _captured(tmp_path)
    pages.record({key: _comparison(
        broken_views=[_BROKE_MAP, _BLANK_DETAILS],
        removed_views=[{"pressed": "Globe"}], removed_controls=["Globe", "Dark mode"],
    )})

    assert pages.end_lines(store, checks={key: "failed"}) == [
        "“Control tower”: “Map” drew before this turn and fails now.",
        "“Control tower”: “Details” showed a graphic before this turn and shows none now.",
        "“Control tower” no longer has “Globe” and “Dark mode”: this turn removed them.",
    ]
    # A later check that passed may have seen the map fixed.
    assert pages.end_lines(store, checks={key: "passed"})[0] == (
        "“Control tower”: “Details” showed a graphic before this turn and shows none now."
    )


_SHRUNK_PORTS = {
    "pressed": "Ports", "view": "svg#ports", "viewports": ["desktop"], "mark": "circle", "before": 24, "after": 3,
}


def test_a_shrunk_view_is_shown_once_by_its_state_and_name_whatever_its_counts():
    """A view still drawing its markers reads with a different count on each check: the same view
    is not named to the agent again for that."""
    key = PageKey("x", "index.html")
    pages = TurnPages()

    assert pages.notes_to_show(key, _comparison(shrunk_views=[_SHRUNK_PORTS])).shrank == [
        "“Ports” (svg#ports: 3 of 24 marks)",
    ]
    assert pages.notes_to_show(key, _comparison(shrunk_views=[{**_SHRUNK_PORTS, "after": 7}])).shrank == []
    another = {**_SHRUNK_PORTS, "pressed": "Lanes", "view": "table#lanes"}
    assert len(pages.notes_to_show(key, _comparison(shrunk_views=[_SHRUNK_PORTS, another])).shrank) == 2


def test_a_set_that_did_not_change_stays_quiet_when_another_does():
    """The agent heard the Ports shrink with the blank Details view; once Details draws again, the
    blank set changes, and the shrink, which didn't, isn't sent back to be fixed again."""
    key = PageKey("x", "index.html")
    pages = TurnPages()
    pages.notes_to_show(key, _comparison(broken_views=[_BLANK_DETAILS], shrunk_views=[_SHRUNK_PORTS]))

    notes = pages.notes_to_show(key, _comparison(shrunk_views=[_SHRUNK_PORTS]))

    assert notes == ComparisonNotes(broke=[], blank=[], removed=[], shrank=[])
    blanked_again = pages.notes_to_show(key, _comparison(broken_views=[_BLANK_DETAILS], shrunk_views=[_SHRUNK_PORTS]))
    assert (blanked_again.blank, blanked_again.shrank) == (["“Details”"], [])


def test_the_turns_last_lines_name_a_view_that_shrank_by_its_tab_even_when_the_page_passed(tmp_path):
    """The line is the person's: it names the tab they see, once however many of its views shrank,
    and neither the view's id nor its counts."""
    store, _, key, pages = _captured(tmp_path)
    legend = {**_SHRUNK_PORTS, "view": "svg#legend"}
    opens = {**_SHRUNK_PORTS, "pressed": None, "view": "table#lanes", "mark": "tr"}
    pages.record({key: _comparison(shrunk_views=[_SHRUNK_PORTS, legend, opens])})

    assert pages.end_lines(store, checks={key: "passed"}) == [
        "“Control tower”: “Ports” and the page as it opens show fewer than half of what they showed before this turn.",
    ]


def test_a_copy_is_kept_as_the_pages_bytes_not_as_wider_text(tmp_path):
    """One character past U+FFFF makes Python keep a whole str at four bytes a character, so the
    turn keeps the page's bytes and decodes them only for a request."""
    page = "<p>\U0001f6a2</p>" + "x" * 100_000
    store, _, key, pages = _captured(tmp_path, pages={"index.html": page})

    assert pages.previous(key) == page
    assert pages.copy_bytes(key) == page.encode("utf-8")
    assert sys.getsizeof(pages._copies[key]) < 2 * len(page.encode("utf-8"))
