"""Replay a corpus of analysis chats through one anton tree's write check, and
score how many approval cards it raises on cells that write and on cells that
don't.

Each chat runs in its own temporary project folder holding the chat's files,
with the folder as the working directory and one project snapshot taken as if
the turn began after the files were there. Every cell is checked with the
chat's earlier cells, as `hold_unapproved_write` checks it: a `turn_start`
cell follows a cell whose process ended, since the host closes the pad at
every turn end, and a check that raises counts as a card, since the product
asks when it can't clear a cell.

The tree may be any anton commit: the replay reads `find_cell_writes`'
signature and calls the `earlier=` form or the older `earlier_code=` one. It
imports the write check from `--tree` only, and exits 2 when Python imported
it from anywhere else.

With `--compare <scorecard.json>`, the run is held to the bar a release
check uses: it fails (exit 1) when a real write the base carded loses its
card, or when cards on cells that write nothing rise above the base's.

Usage:
  python -I scripts/write_check_replay.py --tree <anton tree> [--corpus <yaml>]
      [--out <scorecard.json>] [--compare <base scorecard.json>] [--code-root]
      [--commit <sha>] [--ablate <module function>=<empty value> ...]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS = HERE.parent / "tests" / "fixtures" / "write_check_replay_corpus.yaml"
KINDS = ("unchecked", "sql", "sql_file", "http", "files")
FILE_ACTIONS = frozenset({"delete", "rename", "move", "overwrite", "append"})
EMPTY_VALUES = {"list": list, "tuple": tuple, "none": lambda: None, "false": lambda: False}


# ── corpus ──────────────────────────────────────────────────────────────────


class CorpusCell(BaseModel):
    code: str
    real_write: bool
    why: str = ""
    runs_unread: bool = False
    turn_start: bool = False
    errored: bool = False
    pair: str | None = None
    role: Literal["hold", "free"] | None = None


class GeneratedFile(BaseModel):
    gen: Literal["pdftext", "log", "sqlseed", "csvbig", "jsonbig"]
    bytes: int


class FolderSpec(BaseModel):
    dir: Literal[True]


class Chat(BaseModel):
    name: str
    style: str
    files: dict[str, str | GeneratedFile | FolderSpec] = Field(default_factory=dict)
    tmp_files: dict[str, str | GeneratedFile | FolderSpec] = Field(default_factory=dict)
    cells: list[CorpusCell]


class Corpus(BaseModel):
    rule: str
    chats: list[Chat]


# ── scorecard ───────────────────────────────────────────────────────────────


class CellResult(BaseModel):
    chat: str
    cell: int
    turn: int
    style: str
    card: bool
    crashed: bool
    kinds: list[str]
    causes: list[str]
    real_write: bool
    runs_unread: bool
    pair: str | None
    role: Literal["hold", "free"] | None
    secs: float

    @property
    def key(self) -> str:
        return f"{self.chat} #{self.cell}"


class Tally(BaseModel):
    cells: int
    carded: int


class PairResult(BaseModel):
    pair: str
    holds: Tally
    frees: Tally


class Scorecard(BaseModel):
    tree: str
    commit: str
    signature: Literal["earlier", "earlier_code"]
    code_root: bool
    ablations: list[str]
    corpus_sha256: str
    cells: int
    questions: int
    cards: int
    crashes: int
    per_100_cells: float
    per_100_questions: float
    by_kind_per_100_cells: dict[str, float]
    real_write: Tally
    runs_unread: Tally
    no_write: Tally
    causes: dict[str, dict[str, int]]
    pairs: list[PairResult]
    results: list[CellResult]


# ── generated project files ─────────────────────────────────────────────────

_WORDS = ("supply plan demand forecast safety stock lead time supplier capacity constraint "
          "allocation order fill rate backlog inventory horizon scenario sandbox exception "
          "planner region site part bill material routing calendar shipment receipt").split()


def _generated_text(kind: str, size: int, seed: int) -> str:
    """About `size` bytes of text in the shape of `kind`, the same for one seed."""
    rnd = random.Random(seed)
    lines: list[str] = []
    n = i = 0
    while n < size:
        i += 1
        if kind == "pdftext":
            if i % 40 == 1:
                line = f"\n{i // 40 + 1}.{rnd.randint(1, 9)} {' '.join(rnd.choices(_WORDS, k=4)).title()}\n"
            else:
                line = " ".join(rnd.choices(_WORDS, k=rnd.randint(8, 16))).capitalize() + "."
        elif kind == "log":
            lvl = rnd.choices(["INFO", "WARN", "ERROR", "DEBUG"], weights=[70, 15, 5, 10])[0]
            line = (f"2026-11-0{rnd.randint(1, 9)}T{rnd.randint(0, 23):02d}:{rnd.randint(0, 59):02d}:"
                    f"{rnd.randint(0, 59):02d}Z {lvl} planner.worker[{rnd.randint(1, 8)}] "
                    f"{' '.join(rnd.choices(_WORDS, k=6))} id={rnd.randint(1000, 99999)}")
        elif kind == "sqlseed":
            line = (f"INSERT INTO staging.part_master (part_id, site, lead_time_days) VALUES "
                    f"('P{rnd.randint(10000, 99999)}', 'S{rnd.randint(1, 40):02d}', {rnd.randint(1, 90)});")
        elif kind == "csvbig":
            if i == 1:
                line = "order_id,site,part_id,qty,order_date,status"
            else:
                line = (f"{100000 + i},S{rnd.randint(1, 40):02d},P{rnd.randint(10000, 99999)},{rnd.randint(1, 500)},"
                        f"2026-{rnd.randint(1, 10):02d}-{rnd.randint(1, 28):02d},"
                        f"{rnd.choice(['open', 'shipped', 'late', 'cancelled'])}")
        else:  # jsonbig: a JSON array, one object per line
            line = json.dumps({"part": f"P{rnd.randint(10000, 99999)}", "site": f"S{rnd.randint(1, 40):02d}",
                               "on_hand": rnd.randint(0, 900), "notes": " ".join(rnd.choices(_WORDS, k=8))}) + ","
        lines.append(line)
        n += len(line) + 1
    text = "\n".join(lines)
    if kind == "jsonbig":
        text = "[\n" + text.rstrip(",") + "\n]"
    return text + "\n"


def _write_files(root: Path, files: dict, seed: int, subst) -> None:
    for rel, spec in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(spec, str):
            path.write_text(subst(spec))
        elif isinstance(spec, GeneratedFile):
            path.write_text(_generated_text(spec.gen, spec.bytes, seed))
        else:
            path.mkdir(parents=True, exist_ok=True)


# ── the write check under test ──────────────────────────────────────────────


class WriteCheck:
    """One tree's write check, called the way its own signature takes it."""

    def __init__(self, tree: Path, *, code_root: bool, ablations: list[str]) -> None:
        sys.path.insert(0, str(tree))
        wc = importlib.import_module("anton.core.utils.write_check")
        imported = Path(wc.__file__).resolve()
        if not imported.is_relative_to(tree):
            raise SystemExit(f"refused: the write check came from {imported}, not from --tree {tree}")
        for spec in ablations:
            name, _, empty = spec.partition("=")
            if not hasattr(wc, name) or empty not in EMPTY_VALUES:
                raise SystemExit(f"refused: can't ablate {spec!r} on this tree")
            make = EMPTY_VALUES[empty]
            setattr(wc, name, lambda *a, _make=make, **k: _make())
        self.wc = wc
        params = inspect.signature(wc.find_cell_writes).parameters
        self.signature: Literal["earlier", "earlier_code"] = "earlier" if "earlier" in params else "earlier_code"
        if code_root and "code_root" not in params:
            raise SystemExit("refused: --code-root needs a tree whose find_cell_writes takes code_root")
        self.code_root = code_root
        self.earlier_fields = set(getattr(wc, "EarlierCell").model_fields) if self.signature == "earlier" else set()

    def check(self, *, code: str, earlier: list[dict], snapshot, rules, root: Path):
        if self.signature == "earlier_code":
            return self.wc.find_cell_writes(
                code=code, earlier_code=tuple(e["code"] for e in earlier),
                earlier_errored=tuple(e["errored"] for e in earlier), snapshots=(snapshot,), http_rules=rules)
        cells = [self.wc.EarlierCell(**{k: v for k, v in e.items() if k in self.earlier_fields}) for e in earlier]
        kwargs = {"code": code, "earlier": cells, "snapshots": (snapshot,), "http_rules": rules}
        if self.code_root:
            kwargs["code_root"] = root
        return self.wc.find_cell_writes(**kwargs)


def _bucket(kinds) -> list[str]:
    """The scorecard's kinds: file actions fold into `files`."""
    return sorted({"files" if k in FILE_ACTIONS else k for k in kinds})


def _replay_chat(check: WriteCheck, chat: Chat, rules, seed: int, first_turn: int) -> list[CellResult]:
    results: list[CellResult] = []
    with tempfile.TemporaryDirectory() as r, tempfile.TemporaryDirectory() as t:
        root, tmp = Path(r).resolve(), Path(t).resolve()

        def subst(text: str) -> str:
            return text.replace("{ROOT}", str(root)).replace("{TMP}", str(tmp))

        _write_files(root, chat.files, seed, subst)
        _write_files(tmp, chat.tmp_files, seed + 1, subst)
        cwd = os.getcwd()
        os.chdir(root)
        try:
            snapshot = check.wc.snapshot_project_files(root=root, label="project", taken_at=time.time() + 60)
            earlier: list[dict] = []
            turn = first_turn
            for i, cell in enumerate(chat.cells):
                if cell.turn_start and earlier:
                    turn += 1
                    earlier[-1]["ended_process"] = True
                code = subst(cell.code)
                started = time.time()
                try:
                    writes = check.check(code=code, earlier=earlier, snapshot=snapshot, rules=rules, root=root)
                    card, crashed = bool(writes.needs_approval()), False
                    kinds = _bucket(writes.kinds()) if card else []
                    causes = sorted({c.kind for c in getattr(writes, "causes", ())}) if card else []
                    recorded = dict(getattr(writes, "code_files", {}) or {})
                except Exception:  # the product asks when the check fails
                    card, crashed, kinds, causes, recorded = True, True, ["unchecked"], ["crash"], {}
                results.append(CellResult(
                    chat=chat.name, cell=i, turn=turn, style=chat.style, card=card, crashed=crashed, kinds=kinds,
                    causes=causes, real_write=cell.real_write, runs_unread=cell.runs_unread, pair=cell.pair,
                    role=cell.role, secs=round(time.time() - started, 3)))
                entry = {"code": code, "errored": cell.errored, "ended_process": False}
                if recorded:
                    entry["code_files"] = recorded
                earlier.append(entry)
        finally:
            os.chdir(cwd)
    return results


def _commit_of(tree: Path, given: str | None) -> str:
    if given:
        return given
    done = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else "unknown"


def replay(tree: Path, corpus_path: Path, *, code_root: bool = False, commit: str | None = None,
           ablations: list[str] | None = None) -> Scorecard:
    tree = tree.resolve()
    raw = corpus_path.read_bytes()
    corpus = Corpus.model_validate(yaml.safe_load(raw))
    check = WriteCheck(tree, code_root=code_root, ablations=ablations or [])
    rules = check.wc.parse_http_write_rules(corpus.rule)
    results: list[CellResult] = []
    questions = 0
    for k, chat in enumerate(corpus.chats):
        chat_results = _replay_chat(check, chat, rules, seed=1000 + k, first_turn=questions)
        results.extend(chat_results)
        questions = chat_results[-1].turn + 1
    return _score(results, tree=str(tree), commit=_commit_of(tree, commit), signature=check.signature,
                  code_root=code_root, ablations=ablations or [], corpus_sha256=hashlib.sha256(raw).hexdigest(),
                  questions=questions)


def _tally(results: list[CellResult]) -> Tally:
    return Tally(cells=len(results), carded=sum(r.card for r in results))


def _score(results: list[CellResult], *, questions: int, **identity) -> Scorecard:
    n = len(results)
    cards = [r for r in results if r.card]
    by_kind = Counter(k for r in cards for k in r.kinds)
    causes: dict[str, dict[str, int]] = {}
    for r in cards:
        group = "real_write" if r.real_write else "runs_unread" if r.runs_unread else "no_write"
        for cause in r.causes or ["(none named)"]:
            causes.setdefault(cause, {"real_write": 0, "runs_unread": 0, "no_write": 0})[group] += 1
    pair_ids = sorted({r.pair for r in results if r.pair})
    pairs = [PairResult(pair=p, holds=_tally([r for r in results if r.pair == p and r.role == "hold"]),
                        frees=_tally([r for r in results if r.pair == p and r.role == "free"])) for p in pair_ids]
    return Scorecard(
        **identity, cells=n, questions=questions, cards=len(cards), crashes=sum(r.crashed for r in results),
        per_100_cells=round(100 * len(cards) / n, 1), per_100_questions=round(100 * len(cards) / questions, 1),
        by_kind_per_100_cells={k: round(100 * by_kind.get(k, 0) / n, 1) for k in KINDS},
        real_write=_tally([r for r in results if r.real_write]),
        runs_unread=_tally([r for r in results if r.runs_unread and not r.real_write]),
        no_write=_tally([r for r in results if not r.real_write and not r.runs_unread]),
        causes=causes, pairs=pairs, results=results)


# ── the bar ─────────────────────────────────────────────────────────────────


class Verdict(BaseModel):
    passed: bool
    lost_holds: list[str]
    new_no_write_cards: list[str]
    no_write_cards: tuple[int, int]


def compare(base: Scorecard, head: Scorecard) -> Verdict:
    """Hold `head` to `base`: no real write `base` carded may lose its card,
    and cards on cells that write nothing may not rise. Both must come from
    the same corpus, or the comparison means nothing."""
    if base.corpus_sha256 != head.corpus_sha256:
        raise SystemExit("refused: the two scorecards replayed different corpora")
    base_by_key = {r.key: r for r in base.results}
    lost = [r.key for r in head.results if r.real_write and not r.card and base_by_key[r.key].card]
    new = [r.key for r in head.results
           if r.card and not r.real_write and not r.runs_unread and not base_by_key[r.key].card]
    counts = (base.no_write.carded, head.no_write.carded)
    return Verdict(passed=not lost and counts[1] <= counts[0], lost_holds=lost, new_no_write_cards=new,
                   no_write_cards=counts)


def _summary(card: Scorecard) -> str:
    lines = [
        f"{card.commit[:12]} ({card.signature}{', code_root' if card.code_root else ''}"
        f"{', ablated ' + ' '.join(card.ablations) if card.ablations else ''}): "
        f"{card.cards} cards in {card.cells} cells and {card.questions} questions = "
        f"{card.per_100_cells} per 100 cells, {card.per_100_questions} per 100 questions; crashes {card.crashes}",
        f"  by kind per 100 cells: {card.by_kind_per_100_cells}",
        f"  real writes carded {card.real_write.carded}/{card.real_write.cells}; "
        f"cells running unread code carded {card.runs_unread.carded}/{card.runs_unread.cells}; "
        f"cells that write nothing carded {card.no_write.carded}/{card.no_write.cells}",
    ]
    for p in card.pairs:
        ok = p.holds.carded == p.holds.cells and p.frees.carded == 0
        lines.append(f"  pair {p.pair}: holds {p.holds.carded}/{p.holds.cells} carded, "
                     f"frees {p.frees.carded}/{p.frees.cells} carded{'' if ok else '  <-'}")
    if card.causes:
        lines.append("  causes (real write / unread code / no write): " + "; ".join(
            f"{c} {v['real_write']}/{v['runs_unread']}/{v['no_write']}" for c, v in sorted(card.causes.items())))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", type=Path, default=HERE.parent, help="the anton tree whose write check runs")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, help="where to write the scorecard JSON")
    ap.add_argument("--compare", type=Path, help="a base scorecard to hold this run to")
    ap.add_argument("--code-root", action="store_true", help="read helpers a cell runs by path, as a local pad does")
    ap.add_argument("--commit", help="the tree's commit, when it isn't a git checkout")
    ap.add_argument("--ablate", action="append", default=[], metavar="FUNCTION=EMPTY",
                    help=f"switch one write_check function off; EMPTY is one of {', '.join(EMPTY_VALUES)}")
    ns = ap.parse_args(argv)
    card = replay(ns.tree, ns.corpus, code_root=ns.code_root, commit=ns.commit, ablations=ns.ablate)
    if ns.out:
        ns.out.write_text(card.model_dump_json(indent=1))
    print(_summary(card))
    if not ns.compare:
        return 0
    verdict = compare(Scorecard.model_validate_json(ns.compare.read_text()), card)
    print(f"against {ns.compare.name}: cards on cells that write nothing {verdict.no_write_cards[0]} -> "
          f"{verdict.no_write_cards[1]}; {len(verdict.lost_holds)} real writes lost their card")
    for key in verdict.lost_holds:
        print(f"  lost hold: {key}")
    for key in verdict.new_no_write_cards:
        print(f"  new card on a cell that writes nothing: {key}")
    print("PASS" if verdict.passed else "FAIL")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
