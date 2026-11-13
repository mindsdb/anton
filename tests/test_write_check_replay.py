"""The write-check replay (scripts/write_check_replay.py) and the card budget
it holds this tree's write check to."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "write_check_replay.py"
BUDGET = ROOT / "tests" / "fixtures" / "write_check_replay_budget.yaml"

_spec = importlib.util.spec_from_file_location("write_check_replay_under_test", SCRIPT)
replay_tool = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = replay_tool
_spec.loader.exec_module(replay_tool)


# ── the budget ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def own_scorecard():
    return replay_tool.replay(ROOT, replay_tool.DEFAULT_CORPUS, commit="this tree")


def test_the_write_check_stays_within_its_card_budget(own_scorecard):
    """No real write the budget lists loses its card, and cards on cells that
    write nothing stay at or under the budget's ceiling."""
    budget = yaml.safe_load(BUDGET.read_text())
    carded = {r.key for r in own_scorecard.results if r.real_write and r.card}
    lost = sorted(set(budget["real_writes_carded"]) - carded)
    assert not lost, f"real writes that lost their card: {lost}"
    no_write = [r.key for r in own_scorecard.results if r.card and not r.real_write and not r.runs_unread]
    assert len(no_write) <= budget["no_write_cards_max"], (
        f"{len(no_write)} cards on cells that write nothing, over the budget of "
        f"{budget['no_write_cards_max']}: {no_write}")


def test_the_replay_never_crashes_the_check(own_scorecard):
    assert own_scorecard.crashes == 0, [r.key for r in own_scorecard.results if r.crashed]


def test_a_chat_with_no_cells_is_refused_by_the_corpus_model():
    with pytest.raises(ValidationError, match="cells"):
        replay_tool.Corpus.model_validate({"rule": "", "chats": [{"name": "empty", "style": "t", "cells": []}]})


def test_every_pair_has_a_hold_and_a_free_cell():
    corpus = replay_tool.Corpus.model_validate(yaml.safe_load(replay_tool.DEFAULT_CORPUS.read_text()))
    roles: dict[str, set[str]] = {}
    for chat in corpus.chats:
        for cell in chat.cells:
            assert (cell.pair is None) == (cell.role is None), f"{chat.name}: pair and role go together"
            if cell.pair:
                roles.setdefault(cell.pair, set()).add(cell.role)
                assert cell.real_write == (cell.role == "hold"), f"{chat.name}: a hold cell writes, a free one doesn't"
    assert roles and all(r == {"hold", "free"} for r in roles.values()), roles


# ── the replay against other trees (fake ones, in a subprocess) ─────────────

_COMMON = '''
from pydantic import BaseModel


class Writes(BaseModel):
    card: bool = False
    def needs_approval(self):
        return self.card
    def kinds(self):
        return ("unchecked",) if self.card else ()


def snapshot_project_files(*, root, label, taken_at):
    return None


def parse_http_write_rules(rule):
    return ()
'''

NEW_API = _COMMON + '''

class EarlierCell(BaseModel):
    code: str
    errored: bool = False
    ended_process: bool = False


def find_cell_writes(*, code, earlier, snapshots, http_rules=()):
    if "boom" in code:
        raise RuntimeError("the check failed")
    # Cards the first cell after a process ended, and any cell that says so.
    return Writes(card="card" in code or bool(earlier and earlier[-1].ended_process))
'''

OLD_API = _COMMON + '''

def find_cell_writes(*, code, earlier_code, snapshots, http_rules=(), earlier_errored=()):
    return Writes(card="card" in code or any("card" in c for c in earlier_code))
'''

CORPUS = {
    "rule": "POST https://erp.example.test/**",
    "chats": [
        {"name": "one", "style": "t", "cells": [
            {"code": "x = 1", "real_write": False},
            {"code": "print('card')", "real_write": True},
            {"code": "y = 2", "real_write": False, "turn_start": True},
            {"code": "boom()", "real_write": False},
        ]},
    ],
}


def _fake_tree(tmp_path: Path, source: str) -> Path:
    module = tmp_path / "tree" / "anton" / "core" / "utils"
    module.mkdir(parents=True)
    for pkg in (module.parents[2], module.parents[1], module.parents[0], module):
        (pkg / "__init__.py").touch()
    (module / "write_check.py").write_text(textwrap.dedent(source))
    return tmp_path / "tree"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(SCRIPT), *args], capture_output=True, text=True, timeout=120)


def test_a_turn_start_follows_a_cell_whose_process_ended_and_a_crash_is_a_card(tmp_path):
    tree = _fake_tree(tmp_path, NEW_API)
    corpus = tmp_path / "c.yaml"
    corpus.write_text(yaml.safe_dump(CORPUS))
    out = tmp_path / "card.json"
    done = _run("--tree", str(tree), "--corpus", str(corpus), "--commit", "fake", "--out", str(out))
    assert done.returncode == 0, done.stderr
    card = replay_tool.Scorecard.model_validate_json(out.read_text())
    assert [r.card for r in card.results] == [False, True, True, True]
    assert [r.turn for r in card.results] == [0, 0, 1, 1]
    assert card.results[3].crashed and card.results[3].kinds == ["unchecked"]
    assert (card.questions, card.signature, card.real_write.carded) == (2, "earlier", 1)


def test_an_older_tree_is_called_with_earlier_code(tmp_path):
    tree = _fake_tree(tmp_path, OLD_API)
    corpus = tmp_path / "c.yaml"
    corpus.write_text(yaml.safe_dump(CORPUS))
    out = tmp_path / "card.json"
    done = _run("--tree", str(tree), "--corpus", str(corpus), "--commit", "fake", "--out", str(out))
    assert done.returncode == 0, done.stderr
    card = replay_tool.Scorecard.model_validate_json(out.read_text())
    assert card.signature == "earlier_code"
    assert [r.card for r in card.results] == [False, True, True, True]


def test_a_tree_without_the_write_check_is_refused(tmp_path):
    (tmp_path / "not_anton").mkdir()
    done = _run("--tree", str(tmp_path / "not_anton"), "--corpus", str(replay_tool.DEFAULT_CORPUS))
    assert done.returncode == replay_tool.REFUSED, done.stderr
    assert "refused:" in done.stderr


def test_a_write_check_imported_from_elsewhere_is_refused(tmp_path, monkeypatch, capsys):
    # Once this process holds ROOT's write check, a replay here of any other
    # tree would score ROOT's check; the guard must refuse it.
    importlib.import_module("anton.core.utils.write_check")
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(SystemExit) as stopped:
        replay_tool.WriteCheck(_fake_tree(tmp_path, NEW_API).resolve(), code_root=False, ablations=[])
    assert stopped.value.code == replay_tool.REFUSED
    assert "refused: the write check came from" in capsys.readouterr().err


def test_code_root_is_refused_on_a_tree_that_cannot_take_it(tmp_path):
    tree = _fake_tree(tmp_path, NEW_API)
    corpus = tmp_path / "c.yaml"
    corpus.write_text(yaml.safe_dump(CORPUS))
    done = _run("--tree", str(tree), "--corpus", str(corpus), "--code-root")
    assert done.returncode == replay_tool.REFUSED and "code_root" in done.stderr


# ── the bar ─────────────────────────────────────────────────────────────────


def _card(results: list[tuple[bool, bool]], sha: str = "s") -> replay_tool.Scorecard:
    """A scorecard of (real_write, card) cells."""
    rows = [replay_tool.CellResult(chat="c", cell=i, turn=0, style="t", card=card, crashed=False,
                                   kinds=["sql"] if card else [], causes=[], real_write=rw, runs_unread=False,
                                   pair=None, role=None, secs=0.0) for i, (rw, card) in enumerate(results)]
    return replay_tool._score(rows, tree="t", commit="c", signature="earlier", code_root=False, ablations=[],
                              corpus_sha256=sha, questions=1)


def test_compare_fails_a_lost_hold_even_when_cards_fall():
    base = _card([(True, True), (False, True), (False, True)])
    head = _card([(True, False), (False, False), (False, False)])
    verdict = replay_tool.compare(base, head)
    assert not verdict.passed and verdict.lost_holds == ["c #0"]


def test_compare_fails_a_rise_in_cards_on_cells_that_write_nothing():
    base = _card([(True, True), (False, False), (False, False)])
    head = _card([(True, True), (False, True), (False, False)])
    verdict = replay_tool.compare(base, head)
    assert not verdict.passed and verdict.new_no_write_cards == ["c #1"]
    assert verdict.no_write_cards == (0, 1)


def test_compare_passes_fewer_cards_with_every_hold_kept():
    base = _card([(True, True), (False, True), (False, True)])
    head = _card([(True, True), (False, False), (False, True)])
    assert replay_tool.compare(base, head).passed


def test_compare_names_a_freed_unread_cell_without_failing_on_it():
    def card(unread_card: bool) -> replay_tool.Scorecard:
        rows = [replay_tool.CellResult(chat="c", cell=0, turn=0, style="t", card=unread_card, crashed=False,
                                       kinds=["unchecked"] if unread_card else [], causes=[], real_write=False,
                                       runs_unread=True, pair=None, role=None, secs=0.0)]
        return replay_tool._score(rows, tree="t", commit="c", signature="earlier", code_root=False, ablations=[],
                                  corpus_sha256="s", questions=1)

    verdict = replay_tool.compare(card(True), card(False))
    assert verdict.passed and verdict.freed_unread_cells == ["c #0"]


def test_compare_refuses_scorecards_from_different_corpora(capsys):
    with pytest.raises(SystemExit) as stopped:
        replay_tool.compare(_card([(True, True)], sha="a"), _card([(True, True)], sha="b"))
    assert stopped.value.code == replay_tool.REFUSED
    assert "different corpora" in capsys.readouterr().err
