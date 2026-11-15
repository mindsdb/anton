"""Tests for the artifact editor's patch engine (ENG-4207).

Most cases are ported one for one from yolocoder's ``internal/repo`` tests
(compact_test.go, patch_test.go, anchor_repair_test.go,
anchor_contract_test.go, noop_test.go, and the apply cases of repo_test.go);
each Go test name is kept in the docstring or the test name. Expectations
differ from the Go originals only where this port decides differently on
purpose, and those cases say so:

- a created file ends with a newline (Go wrote it without one);
- ``@+path`` on an existing file rewrites it (Mochi), where yolocoder refused;
- a block matching in several places is never placed at its first match,
  however long, unless ordered placement resolves it.
"""

from __future__ import annotations

import pytest

from anton.core.artifact_editor.patch import (
    ApplyResult,
    FileEdit,
    Hunk,
    PatchParseError,
    _clip,
    _is_compact,
    _lines_of,
    _similarity,
    _without_stray_removals,
    apply_patch,
    parse_patch,
)


def run(files: dict[str, str], patch: str) -> tuple[ApplyResult, dict[str, str]]:
    """Apply ``patch`` to an in-memory project; return the result and the
    project as it would be after writing ``result.written``."""
    result = apply_patch(parse_patch(patch), files.get)
    return result, {**files, **result.written}


def ok(files: dict[str, str], patch: str) -> dict[str, str]:
    result, after = run(files, patch)
    assert result.ok, result.feedback()
    return after


def diagnoses(result: ApplyResult):
    return [one for group in result.failed.values() for one in group]


# ---------------------------------------------------------------------------
# compact_test.go
# ---------------------------------------------------------------------------


def test_compact_applies_the_documented_example():
    after = ok(
        {
            "src/app.py": "def start():\n    server.run(config)\n    return server\n",
            "src/config.py": 'PORT = 8000\nHOST = "localhost"\n',
        },
        "@src/app.py\n"
        " def start():\n"
        "-    server.run(config)\n"
        "+    server.run(config, debug=True)\n"
        "     return server\n"
        "\n"
        "@src/config.py\n"
        "-PORT = 8000\n"
        "+PORT = 8080\n",
    )
    assert "debug=True" in after["src/app.py"]
    assert "PORT = 8080" in after["src/config.py"]
    assert 'HOST = "localhost"' in after["src/config.py"]


def test_compact_creates_a_file_from_literal_content():
    # Literal content: nothing stripped, and the blank line inside it is part
    # of the file. (Go wrote it without a final newline; created files here
    # end with one.)
    result, after = run(
        {"keep.txt": "x\n"}, "@+docs/notes.md\n# Notes\n\n- first\n- second\n"
    )
    assert result.ok
    assert after["docs/notes.md"] == "# Notes\n\n- first\n- second\n"


def test_compact_created_content_may_begin_with_diff_characters():
    after = ok(
        {"keep.txt": "x\n"}, "@+README.md\n- a bullet\n+ not an addition\n  indented\n"
    )
    assert after["README.md"] == "- a bullet\n+ not an addition\n  indented\n"


def test_compact_created_content_survives_at_rules():
    css = '@import url("x.css");\n\n@media (max-width: 520px) {\n  body { color: red; }\n}\n'
    result, after = run({"keep.txt": "x\n"}, "@+styles/app.css\n" + css)
    assert result.ok
    assert after["styles/app.css"] == css
    assert set(result.written) == {"styles/app.css"}


def test_compact_takes_several_edits_to_one_file():
    after = ok(
        {"a.ts": "const first = 1;\nconst middle = 2;\nconst last = 3;\n"},
        "@a.ts\n-const first = 1;\n+const first = 10;\n\n-const last = 3;\n+const last = 30;\n",
    )
    assert after["a.ts"] == "const first = 10;\nconst middle = 2;\nconst last = 30;\n"


def test_compact_tolerates_a_context_line_that_lost_its_space():
    after = ok(
        {"a.ts": "function go() {\n  const x = 1;\n}\n"},
        "@a.ts\nfunction go() {\n-  const x = 1;\n+  const x = 2;\n}\n",
    )
    assert "const x = 2" in after["a.ts"]


def test_compact_refuses_delete_and_move_by_name():
    with pytest.raises(PatchParseError, match="deletes gone.txt"):
        parse_patch("@-gone.txt\n")
    with pytest.raises(PatchParseError, match="moves gone.txt"):
        parse_patch("@>gone.txt\nelsewhere.txt\n")


def test_compact_failures_carry_the_same_diagnostics():
    result, _ = run(
        {"a.ts": "const board = useState(empty());\n", "b.ts": "const other = 1;\n"},
        "@a.ts\n-const board = useState(() => empty());\n+const board = useState(fresh());\n\n"
        "@b.ts\n-const other = 99;\n+const other = 2;\n",
    )
    assert not result.ok
    text = result.feedback()
    assert text.startswith("2 edits in this patch could not be placed.")
    assert "() => empty()" in text


@pytest.mark.parametrize(
    ("patch", "compact"),
    [
        ("@src/app.py\n-a\n+b\n", True),
        ("@+src/new.py\nprint(1)\n", True),
        ("\n\n@src/app.py\n-a\n+b\n", True),
        ("--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n", False),
        ("diff --git a/x b/x\n--- a/x\n+++ b/x\n@@\n-a\n+b\n", False),
        ("*** Begin Patch\n*** Update File: x\n-a\n+b\n*** End Patch\n", False),
        ("@@ -1,3 +1,3 @@\n-a\n+b\n", False),
        ("", False),
    ],
)
def test_is_compact_patch_tells_the_formats_apart(patch, compact):
    assert _is_compact(_lines_of(patch)) is compact


def test_compact_and_unified_still_coexist():
    after = ok(
        {"a.ts": "const x = 1;\n"},
        "--- a/a.ts\n+++ b/a.ts\n@@ -1 +1 @@\n-const x = 1;\n+const x = 2;\n",
    )
    assert after["a.ts"] == "const x = 2;\n"


def test_a_single_line_edit_still_gets_a_near_miss():
    result, _ = run(
        {"a.ts": "const board = useState(empty());\nconst other = 1;\n"},
        "@a.ts\n-const board = useState(() => empty());\n+const board = useState(fresh());\n",
    )
    [one] = diagnoses(result)
    assert one.kind == "near_miss"
    assert 'expected: "const board = useState(() => empty());"' in one.message
    assert 'in file:  "const board = useState(empty());"' in one.message


def test_no_near_miss_when_nothing_resembles():
    result, _ = run(
        {"a.ts": 'import fs from "fs";\n'},
        "@a.ts\n-const completelyDifferent = compute(alpha, beta);\n+const x = 1;\n",
    )
    [one] = diagnoses(result)
    assert one.kind == "not_found"
    assert "in file:" not in result.feedback()


@pytest.mark.parametrize(
    ("a", "b", "low", "high"),
    [
        ("const x = 1;", "const x = 1;", 1, 1),
        ("server.run(config)", "server.run(config, debug=True)", 0.5, 1),
        ('import fs from "fs";', "export default function App() {", 0, 0.4),
        ("", "", 1, 1),
    ],
)
def test_similarity_scores_a_transcription_slip(a, b, low, high):
    assert low <= _similarity(a, b) <= high


def test_two_edits_under_one_header_with_no_blank_line_between():
    after = ok(
        {
            "App.tsx": "const a = 1;\n"
            "          <header><div><h1>tictacos</h1></div></header>\n"
            '          <section className="lobby-card">\n'
            "          <header><div><p>ROOM</p><h1>tictacos</h1></div></header>\n"
            "const b = 2;\n"
        },
        "@App.tsx\n"
        "-          <header><div><h1>tictacos</h1></div></header>\n"
        "+          <header><div><h1>tacos tacos</h1></div></header>\n"
        "-          <header><div><p>ROOM</p><h1>tictacos</h1></div></header>\n"
        "+          <header><div><p>ROOM</p><h1>tacos tacos</h1></div></header>\n",
    )
    got = after["App.tsx"]
    assert "tictacos" not in got
    assert got.count("tacos tacos") == 2
    assert '<section className="lobby-card">' in got


def test_a_contiguous_replacement_is_not_split():
    after = ok(
        {"a.ts": "keep;\nfirst;\nsecond;\nkeep2;\n"},
        "@a.ts\n-first;\n-second;\n+only;\n",
    )
    assert after["a.ts"] == "keep;\nonly;\nkeep2;\n"


def test_context_still_holds_a_hunk_together():
    after = ok(
        {"a.ts": "one;\nmiddle;\ntwo;\n"},
        "@a.ts\n-one;\n+ONE;\n middle;\n-two;\n+TWO;\n",
    )
    assert after["a.ts"] == "ONE;\nmiddle;\nTWO;\n"


def test_bunched_edits_are_split_and_placed():
    app = (
        'type Language = "en" | "ru";\n'
        'import x from "y";\n'
        "\n"
        "function load() {\n"
        '  return saved === "ru" ? saved : "en";\n'
        "}\n"
        "\n"
        "const translations = {\n"
        '  pt: { createRoom: "Criar sala" },\n'
        "};\n"
    )
    result, after = run(
        {"App.tsx": app},
        "@App.tsx\n"
        '-type Language = "en" | "ru";\n'
        '+type Language = "en" | "ru" | "pl";\n'
        '-  return saved === "ru" ? saved : "en";\n'
        '+  return saved === "ru" || saved === "pl" ? saved : "en";\n'
        '   pt: { createRoom: "Criar sala" },\n'
        '+  pl: { createRoom: "Utwórz pokój" },\n',
    )
    assert result.ok, result.feedback()
    got = after["App.tsx"]
    for want in (
        'type Language = "en" | "ru" | "pl";',
        'saved === "ru" || saved === "pl" ? saved : "en";',
        'pl: { createRoom: "Utwórz pokój" },',
        'import x from "y";',
        'pt: { createRoom: "Criar sala" },',
    ):
        assert want in got
    assert "split_by_anchor" in result.repairs


def test_an_ambiguous_anchor_is_reported_rather_than_split():
    result, _ = run(
        {"a.ts": "call();\nfiller\ncall();\nmore\nother();\n"},
        "@a.ts\n call();\n-other();\n+changed();\n",
    )
    assert not result.ok
    assert "a.ts" in result.failed
    assert result.feedback().startswith("a.ts:")


def test_a_near_miss_never_reports_a_line_as_differing_from_itself():
    result, _ = run(
        {"a.ts": "const kept = 1;\nfiller\nconst kept = 1;\nmore\nconst other = 2;\n"},
        "@a.ts\n const kept = 1;\n-const other = 2;\n+merged;\n",
    )
    assert not result.ok
    for line in result.feedback().splitlines():
        assert not (line.strip().startswith("expected:") and "const kept" in line)


def test_a_patch_that_changes_nothing_is_refused():
    result, after = run(
        {"a.ts": "const x = 1;\nconst y = 2;\n"},
        "@a.ts\n const x = 1;\n const y = 2;\n",
    )
    assert result.noop and not result.ok
    assert "changes nothing" in result.feedback()
    assert result.written == {}
    assert after["a.ts"] == "const x = 1;\nconst y = 2;\n"


def test_at_at_separates_edits_the_way_models_write_them():
    source = (
        'export type TetrominoId = "i" | "o";\n'
        'export type Phase = "idle" | "playing";\n'
        "filler\n"
        "  private sfx: Sfx;\n"
        "  private onHud: (h: Hud) => void;\n"
        "more filler\n"
        "  setMouse(x: number | null) {\n"
        "    this.mouseX = x;\n"
        "  }\n"
    )
    after = ok(
        {"tetrapong.ts": source},
        "@tetrapong.ts\n"
        ' export type TetrominoId = "i" | "o";\n'
        '+export type Language = "en" | "es";\n'
        ' export type Phase = "idle" | "playing";\n'
        "@@\n"
        "   private onHud: (h: Hud) => void;\n"
        '+  private language: Language = "en";\n'
        "@@\n"
        "   setMouse(x: number | null) {\n"
        "     this.mouseX = x;\n"
        "   }\n"
        "+\n"
        "+  setLanguage(language: Language) {\n"
        "+    this.language = language;\n"
        "+  }\n",
    )
    got = after["tetrapong.ts"]
    for want in (
        'export type Language = "en" | "es";',
        'private language: Language = "en";',
        "setLanguage(language: Language) {",
    ):
        assert want in got
    assert "@@" not in got


def test_apply_patch_markers_are_separators_too():
    after = ok(
        {"a.ts": "const x = 1;\nconst y = 2;\n"},
        "@a.ts\n-const x = 1;\n+const x = 9;\n*** End Patch\n",
    )
    assert after["a.ts"] == "const x = 9;\nconst y = 2;\n"


def test_a_line_that_really_starts_with_at_at_is_still_content():
    after = ok(
        {"notes.md": "intro\n@@ this is content @@\ntrailer\n"},
        "@notes.md\n @@ this is content @@\n-trailer\n+footer\n",
    )
    assert after["notes.md"] == "intro\n@@ this is content @@\nfooter\n"


def test_a_bare_row_of_stars_separates_edits():
    after = ok(
        {"a.ts": "const x = 1;\nfiller\nconst y = 2;\n"},
        "@a.ts\n-const x = 1;\n+const x = 9;\n***\n-const y = 2;\n+const y = 8;\n",
    )
    assert after["a.ts"] == "const x = 9;\nfiller\nconst y = 8;\n"


def test_stars_inside_a_file_are_still_content():
    after = ok(
        {"notes.md": "intro\n***bold***\ntrailer\n"},
        "@notes.md\n ***bold***\n-trailer\n+footer\n",
    )
    assert after["notes.md"] == "intro\n***bold***\nfooter\n"


def test_a_stray_marker_in_a_hunk_is_named():
    result, _ = run(
        {"a.ts": "const first = 1;\nconst second = 2;\nconst third = 3;\n"},
        "@a.ts\n const first = 1;\n~~~~\n-const second = 2;\n+const second = 9;\n",
    )
    [one] = diagnoses(result)
    assert one.kind == "absent_lines"
    assert "nowhere in the file" in one.message
    assert "~~~~" in one.message
    assert "a blank line does that" in one.message


def test_a_transcription_slip_still_gets_the_near_miss():
    result, _ = run(
        {"a.ts": "const a = 1;\nconst board = useState(empty());\nconst c = 3;\n"},
        "@a.ts\n const a = 1;\n-const board = useState(() => empty());\n+const board = useState(fresh());\n const c = 3;\n",
    )
    [one] = diagnoses(result)
    assert "nowhere in the file" not in one.message
    assert "useState(empty())" in one.message


def test_a_file_header_behind_a_marker_is_still_a_file_header():
    after = ok(
        {
            "App.tsx": "const app = 1;\n",
            "tetrapong.ts": 'export type Language = "en" | "es";\n',
        },
        "@App.tsx\n-const app = 1;\n+const app = 2;\n"
        '*** @tetrapong.ts\n-export type Language = "en" | "es";\n+export type Language = "en" | "es" | "de";\n',
    )
    assert after["App.tsx"] == "const app = 2;\n"
    assert '"de"' in after["tetrapong.ts"]


def test_a_path_behind_a_marker_with_no_at_sign_works_too():
    after = ok(
        {"a.ts": "const a = 1;\n", "b.ts": "const b = 1;\n"},
        "@a.ts\n-const a = 1;\n+const a = 2;\n*** b.ts\n-const b = 1;\n+const b = 2;\n",
    )
    assert after["b.ts"] == "const b = 2;\n"


def test_end_patch_and_bare_stars_are_still_separators():
    result, after = run(
        {"a.ts": "const x = 1;\nfiller\nconst y = 2;\n"},
        "@a.ts\n-const x = 1;\n+const x = 9;\n***\n-const y = 2;\n+const y = 8;\n*** End Patch\n",
    )
    assert result.ok
    assert after["a.ts"] == "const x = 9;\nfiller\nconst y = 8;\n"
    assert set(result.written) == {"a.ts"}


def test_edits_under_the_wrong_header_say_where_they_belong():
    files = {
        "App.tsx": "const app = 1;\n",
        "tetrapong.ts": 'export type Language = "en" | "es";\n',
    }
    result, after = run(
        files,
        "@App.tsx\n-const app = 1;\n+const app = 2;\n\n"
        '-export type Language = "en" | "es";\n+export type Language = "en" | "es" | "de";\n'
        "\n@tetrapong.ts\n const unrelated = 1;\n",
    )
    [one] = diagnoses(result)
    assert one.kind == "misrouted"
    assert "these lines are in tetrapong.ts, not in App.tsx" in one.message
    assert "header naming tetrapong.ts" in one.message
    assert after["App.tsx"] == "const app = 1;\n"


def test_created_file_strips_diff_prefixes_and_trailer():
    after = ok(
        {"keep.txt": "x\n"},
        '@+game/doom.ts\n+export type TetrominoId = "i" | "o";\n+export const VIEW_W = 480;\n*** End Patch\n',
    )
    assert (
        after["game/doom.ts"]
        == 'export type TetrominoId = "i" | "o";\nexport const VIEW_W = 480;\n'
    )


def test_created_file_keeps_mixed_plus_lines_and_bare_stars():
    after = ok(
        {"keep.txt": "x\n"}, "@+notes.md\n# Title\n\n+ a bullet\n\n***\n\ntext\n"
    )
    assert after["notes.md"] == "# Title\n\n+ a bullet\n\n***\n\ntext\n"


def test_garbage_after_the_end_marker_is_not_part_of_the_patch():
    after = ok(
        {"a.ts": "const x = 1;\n"},
        "@a.ts\n-const x = 1;\n+const x = 2;\n*** End Patch\n♀♀♀♀♀♀json pandurog? of k not valid? \n}ર્ય?\n",
    )
    assert after["a.ts"] == "const x = 2;\n"


def test_several_blocks_still_run_on_past_an_end_marker():
    after = ok(
        {"a.ts": "const x = 1;\nfiller\nconst y = 2;\n"},
        "@a.ts\n-const x = 1;\n+const x = 9;\n*** End Patch\n@a.ts\n-const y = 2;\n+const y = 8;\n*** End Patch\n",
    )
    assert after["a.ts"] == "const x = 9;\nfiller\nconst y = 8;\n"


LONG_JSX = (
    '          <header><div><p className="eyebrow">ROOM 7</p><h1>tictataco</h1>'
    '<p className="subtitle">your turn</p></div><div><button className="reset"'
    ' onClick={() => reset()}>{t("leave")}</button></div></header>'
)


def test_a_line_truncated_mid_way_still_places():
    cut = (
        '          <header><div><p className="eyebrow">ROOM 7</p><h1>tictataco</h1>'
        '<p className="subtitle">your turn</p></div><div><button className="reset"'
    )
    fixed = cut.replace("tictataco", "tictacos", 1)
    result, after = run(
        {"App.tsx": "const before = 1;\n" + LONG_JSX + "\nconst after = 2;\n"},
        "@App.tsx\n-" + cut + "\n+" + fixed + "\n",
    )
    assert result.ok, result.feedback()
    assert (
        after["App.tsx"]
        == "const before = 1;\n"
        + LONG_JSX.replace("tictataco", "tictacos", 1)
        + "\nconst after = 2;\n"
    )
    assert "truncated_line" in result.repairs


def test_a_short_prefix_is_not_treated_as_a_truncation():
    result, _ = run({"a.ts": "const x = 1;\nconst y = 2;\n"}, "@a.ts\n-const\n+let\n")
    assert not result.ok


def test_a_truncation_must_be_unambiguous():
    long = "        this is a long line of configuration that goes on for a while"
    result, _ = run(
        {"a.ts": long + " and ends here;\n" + long + " and ends differently;\n"},
        "@a.ts\n-" + long + "\n+" + long + " changed\n",
    )
    assert not result.ok


def test_a_pure_deletion_is_never_truncated():
    line = "        const somethingVeryLongIndeed = configure({ alpha: 1, beta: 2 });"
    result, after = run({"a.ts": line + " // keep me\n"}, "@a.ts\n-" + line + "\n")
    assert not result.ok
    assert "keep me" in after["a.ts"]


@pytest.mark.parametrize(
    "header",
    [
        "*** Update File: frontend/src/App.tsx",
        "@@ frontend/src/App.tsx",
        "@@ frontend/src/App.tsx @@",
    ],
)
def test_a_borrowed_header_mid_patch_still_names_its_file(header):
    after = ok(
        {
            "backend/index.ts": "const app = express();\n",
            "frontend/src/App.tsx": "export default function App() {\n  return null;\n}\n",
        },
        "@backend/index.ts\n-const app = express();\n+const app = express(); // rooms\n\n"
        + header
        + "\n-  return null;\n+  return <Game />;\n",
    )
    assert "<Game />" in after["frontend/src/App.tsx"]


def test_an_apply_patch_function_hint_is_not_a_file():
    after = ok(
        {"app.ts": "function render() {\n  return 1;\n}\n"},
        "@app.ts\n@@ render\n-  return 1;\n+  return 2;\n",
    )
    assert after["app.ts"] == "function render() {\n  return 2;\n}\n"


def test_a_blank_line_inside_added_code_stays_in_it():
    after = ok(
        {"App.tsx": "export default function App() {\n  return null;\n}\n"},
        "@App.tsx\n-export default function App() {\n-  return null;\n-}\n"
        '+import { useState } from "react";\n\n+export default function App() {\n\n\n+  return <Game />;\n+}\n',
    )
    assert (
        after["App.tsx"]
        == 'import { useState } from "react";\n\nexport default function App() {\n\n\n  return <Game />;\n}\n'
    )


def test_a_blank_line_before_an_anchored_insertion_still_separates():
    after = ok(
        {"a.ts": "one\ntwo\nthree\nfour\n"},
        "@a.ts\n one\n+one-and-a-half\n\n+three-minus\n three\n",
    )
    assert after["a.ts"] == "one\none-and-a-half\ntwo\nthree-minus\nthree\nfour\n"


# ---------------------------------------------------------------------------
# patch_test.go
# ---------------------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Tec-Tac-Tris</title>
  <style>
    h1 { margin:0; }
  </style>
</head>
<body>
  <main class="wrap">
    <header>
      <div><h1>Tec-Tac-Tris</h1><p class="tagline">Drop a piece.</p></div>
      <button id="restart">New game</button>
    </header>
  </main>
</body>
</html>
"""


def patched_page(patch: str) -> str:
    return ok({"index.html": PAGE}, patch)["index.html"]


def test_apply_hunk_with_no_trailing_context():
    got = patched_page(
        "diff --git a/index.html b/index.html\n--- a/index.html\n+++ b/index.html\n@@ -2,7 +2,7 @@\n"
        ' <html lang="en">\n <head>\n'
        '   <meta charset="UTF-8" />\n'
        '   <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n'
        "-  <title>Tec-Tac-Tris</title>\n"
        "+  <title>TeIC-TAC-TRoIS</title>\n"
    )
    assert "<title>TeIC-TAC-TRoIS</title>" in got
    assert '<button id="restart">New game</button>' in got


def test_apply_hunk_with_wrong_line_numbers_and_counts():
    got = patched_page(
        "--- a/index.html\n+++ b/index.html\n@@ -900,42 +900,42 @@\n"
        "-  <title>Tec-Tac-Tris</title>\n+  <title>TeIC-TAC-TRoIS</title>\n   <style>\n"
    )
    assert "<title>TeIC-TAC-TRoIS</title>" in got


def test_apply_multiple_hunks_in_one_file():
    got = patched_page(
        "--- a/index.html\n+++ b/index.html\n@@ -5,1 +5,1 @@\n"
        "-  <title>Tec-Tac-Tris</title>\n+  <title>TeIC-TAC-TRoIS</title>\n"
        "@@ -65,1 +65,1 @@\n"
        '-      <div><h1>Tec-Tac-Tris</h1><p class="tagline">Drop a piece.</p></div>\n'
        '+      <div><h1>TeIC-TAC-TRoIS</h1><p class="tagline">Drop a piece.</p></div>\n'
    )
    assert "Tec-Tac-Tris" not in got
    assert got.count("TeIC-TAC-TRoIS") == 2


def test_apply_is_refused_when_the_hunk_is_ambiguous():
    result, after = run(
        {"a.txt": "x\nx\nx\n"}, "--- a/a.txt\n+++ b/a.txt\n@@ -1,1 +1,1 @@\n-x\n+y\n"
    )
    [one] = diagnoses(result)
    assert one.kind == "ambiguous"
    assert "ambiguous" in one.message
    assert after["a.txt"] == "x\nx\nx\n"


def test_ambiguous_hunk_error_points_at_each_match():
    original = (
        "function a() {\n  setSession(x);\n}\nfunction b() {\n  setSession(x);\n}\n"
    )
    result, _ = run(
        {"a.txt": original},
        "--- a/a.txt\n+++ b/a.txt\n@@ -1,1 +1,1 @@\n-  setSession(x);\n+  setSession(y);\n",
    )
    text = result.feedback()
    for want in ("at line 2", "at line 5", "function a() {", "function b() {", "> "):
        assert want in text


def test_apply_leaves_files_alone_when_a_hunk_cannot_be_placed():
    result, after = run(
        {"index.html": PAGE},
        "--- a/index.html\n+++ b/index.html\n@@ -1,1 +1,1 @@\n"
        "-  <title>Something Else Entirely</title>\n+  <title>TeIC-TAC-TRoIS</title>\n",
    )
    assert not result.ok
    assert result.written == {}
    assert after["index.html"] == PAGE


def test_apply_recovers_from_a_spelled_out_html_entity():
    after = ok(
        {"index.html": "<p>Salt &amp; pepper</p>\n<p>unrelated</p>\n"},
        "--- a/index.html\n+++ b/index.html\n@@ -1,1 +1,1 @@\n-<p>Salt & pepper</p>\n+<p>Salt & vinegar</p>\n",
    )
    assert "Salt & vinegar" in after["index.html"]
    assert "unrelated" in after["index.html"]


def test_apply_failure_reports_the_closest_matching_line():
    result, _ = run(
        {"index.html": PAGE},
        "--- a/index.html\n+++ b/index.html\n@@ -1,1 +1,1 @@\n"
        '   <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n'
        "-  <title>Tec-Tac-Tris!</title>\n+  <title>New Title</title>\n",
    )
    text = result.feedback()
    assert "closest match is at line" in text
    assert '"  <title>Tec-Tac-Tris!</title>"' in text
    assert '"  <title>Tec-Tac-Tris</title>"' in text


def test_apply_failure_omits_near_miss_when_nothing_is_close():
    result, _ = run(
        {"index.html": PAGE},
        "--- a/index.html\n+++ b/index.html\n@@ -1,1 +1,1 @@\n"
        "-totally unrelated content that appears nowhere in the file at all\n+replacement\n",
    )
    assert not result.ok
    assert "closest match" not in result.feedback()


def test_parse_patch_reads_paths_and_hunks():
    [edit] = parse_patch(
        "diff --git a/x.go b/x.go\nindex 111..222 100644\n--- a/x.go\n+++ b/x.go\n@@ -1,2 +1,2 @@\n a\n-b\n+c\n"
    )
    assert edit.path == "x.go" and not edit.create
    [hunk] = edit.hunks
    assert hunk.before == ("a", "b")
    assert hunk.after == ("a", "c")


def test_parse_patch_accepts_a_minus_header_with_no_plus_header():
    [edit] = parse_patch("--- frontend/src/App.tsx\n@@\n-old\n+new\n")
    assert edit.path == "frontend/src/App.tsx"


def test_apply_recovers_when_the_patch_has_only_a_minus_header():
    got = patched_page(
        "--- index.html\n@@\n-  <title>Tec-Tac-Tris</title>\n+  <title>TeIC-TAC-TRoIS</title>\n"
    )
    assert "<title>TeIC-TAC-TRoIS</title>" in got


def test_a_well_formed_unified_diff_applies():
    got = patched_page(
        "--- a/index.html\n+++ b/index.html\n@@ -5,3 +5,3 @@\n"
        '   <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n'
        "-  <title>Tec-Tac-Tris</title>\n+  <title>TeIC-TAC-TRoIS</title>\n   <style>\n"
    )
    assert "<title>TeIC-TAC-TRoIS</title>" in got


def test_apply_openai_apply_patch_format():
    got = patched_page(
        "*** Begin Patch\n*** Update File: index.html\n@@\n"
        '   <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n'
        "-  <title>Tec-Tac-Tris</title>\n+  <title>bambaloba</title>\n   <style>\n"
        "@@\n"
        '-      <div><h1>Tec-Tac-Tris</h1><p class="tagline">Drop a piece.</p></div>\n'
        '+      <div><h1>bambaloba</h1><p class="tagline">Drop a piece.</p></div>\n'
        "*** End Patch"
    )
    assert "Tec-Tac-Tris" not in got
    assert got.count("bambaloba") == 2
    assert '<button id="restart">New game</button>' in got


def test_parse_patch_reads_apply_patch_headers():
    [edit] = parse_patch(
        "*** Begin Patch\n*** Update File: a/b/c.go\n@@\n ctx\n-old\n+new\n*** End Patch"
    )
    assert edit.path == "b/c.go"
    [hunk] = edit.hunks
    assert hunk.before == ("ctx", "old")


def test_apply_patch_format_can_add_a_file():
    result, after = run(
        {},
        "*** Begin Patch\n*** Add File: notes/new.txt\n@@\n+hello\n+world\n*** End Patch",
    )
    assert result.ok
    assert after["notes/new.txt"] == "hello\nworld\n"


def test_apply_refuses_to_delete_files():
    with pytest.raises(PatchParseError, match="does not do"):
        parse_patch("*** Begin Patch\n*** Delete File: gone.txt\n*** End Patch")


def test_apply_several_patch_blocks_for_one_file():
    got = patched_page(
        "*** Begin Patch\n*** Update File: index.html\n@@\n"
        "-  <title>Tec-Tac-Tris</title>\n+  <title>one</title>\n   <style>\n*** End Patch\n"
        "*** Begin Patch\n*** Update File: index.html\n@@\n"
        '-      <div><h1>Tec-Tac-Tris</h1><p class="tagline">Drop a piece.</p></div>\n'
        '+      <div><h1>two</h1><p class="tagline">Drop a piece.</p></div>\n'
        "*** End Patch"
    )
    assert "<title>one</title>" in got
    assert "<h1>two</h1>" in got
    assert "Tec-Tac-Tris" not in got


@pytest.mark.parametrize(
    ("patch", "path", "create"),
    [
        (
            "*** Begin Patch\n*** Update File: a.go\n@@\n-x\n+y\n*** End Patch",
            "a.go",
            False,
        ),
        ("*** Update File: a.go\n@@\n-x\n+y", "a.go", False),
        ("  *** Begin Patch\n*** Add File: new.txt\n@@\n+hi", "new.txt", True),
        (
            "diff --git a/a.go b/a.go\n--- a/a.go\n+++ b/a.go\n@@ -1 +1 @@\n-x\n+y",
            "a.go",
            False,
        ),
        ("--- a/a.go\n+++ b/a.go\n@@ -1 +1 @@\n-x\n+y", "a.go", False),
    ],
)
def test_is_apply_patch_format(patch, path, create):
    # Go's isApplyPatchFormat chose between its own applier and git apply;
    # with no git here, what matters is that each shape reads as its file.
    [edit] = parse_patch(patch)
    assert (edit.path, edit.create) == (path, create)


def test_add_file_with_no_hunk_header():
    result, after = run(
        {},
        "*** Begin Patch\n"
        "*** Add File: package.json\n"
        "+{\n"
        '+  "name": "super-tic-tac-tris",\n'
        '+  "scripts": { "start": "node server.js" }\n'
        "+}\n"
        "*** Add File: server.js\n"
        "+const express = require('express');\n"
        "+const app = express();\n"
        "*** Add File: launch.sh\n"
        "+#!/bin/sh\n"
        "+node server.js\n"
        "*** End Patch",
    )
    assert result.ok
    assert '"name": "super-tic-tac-tris"' in after["package.json"]
    assert "const express = require('express');" in after["server.js"]
    assert "node server.js" in after["launch.sh"]


def test_update_file_with_a_hunk_header_is_unaffected():
    got = patched_page(
        "*** Begin Patch\n*** Update File: index.html\n@@\n"
        "-  <title>Tec-Tac-Tris</title>\n+  <title>kept</title>\n   <style>\n*** End Patch"
    )
    assert "<title>kept</title>" in got
    assert '<button id="restart">New game</button>' in got


def test_explain_names_the_file_and_the_line_that_differs():
    result, _ = run(
        {"app.tsx": "const a = 1;\nconst board = useState(empty());\nconst c = 3;\n"},
        "--- a/app.tsx\n+++ b/app.tsx\n@@\n const a = 1;\n-const board = useState(() => empty());\n"
        "+const board = useState(fresh());\n const c = 3;\n",
    )
    text = result.feedback()
    assert text.startswith("app.tsx: could not find this hunk's lines")
    assert "() => empty()" in text
    assert "useState(empty())" in text


def test_a_long_line_is_clipped():
    assert len(_clip("x" * 300)) <= 113


THREE_FILES = {
    "app.tsx": "const a = 1;\nconst board = useState(empty());\nconst c = 3;\n",
    "main.css": ":root {\n  --bg: #b7c98a;\n}\n",
    "dup.ts": "call();\nother();\ncall();\n",
}


def test_one_pass_reports_every_edit_that_cannot_be_placed():
    result, _ = run(
        THREE_FILES,
        "--- a/app.tsx\n+++ b/app.tsx\n@@\n const a = 1;\n-const board = useState(() => empty());\n"
        "+const board = useState(fresh());\n const c = 3;\n"
        "--- a/main.css\n+++ b/main.css\n@@\n-  --bg: #001933;\n+  --bg: #123456;\n"
        "--- a/dup.ts\n+++ b/dup.ts\n@@\n-call();\n+called();\n",
    )
    assert len(diagnoses(result)) == 3
    assert set(result.failed) == {"app.tsx", "main.css", "dup.ts"}
    text = result.feedback()
    assert text.startswith(
        "3 edits in this patch could not be placed. Fix every one of them, all in one apply_diff call:"
    )
    for want in ("[1]", "[2]", "[3]"):
        assert want in text


def test_only_files_whose_edits_all_place_are_written():
    result, after = run(
        THREE_FILES,
        "--- a/app.tsx\n+++ b/app.tsx\n@@\n const a = 1;\n-const board = useState(empty());\n"
        "+const board = useState(fresh());\n const c = 3;\n"
        "--- a/main.css\n+++ b/main.css\n@@\n-  --bg: #001933;\n+  --bg: #123456;\n",
    )
    assert list(result.written) == ["app.tsx"]
    assert list(result.failed) == ["main.css"]
    assert "fresh()" in after["app.tsx"]
    assert after["main.css"] == THREE_FILES["main.css"]
    assert "Every edit to app.tsx placed and was applied" in result.feedback()


def test_a_file_with_any_failed_edit_is_not_written():
    result, after = run({"a.ts": "one\ntwo\n"}, "@a.ts\n-one\n+uno\n\n-three\n+tres\n")
    assert result.written == {}
    assert after["a.ts"] == "one\ntwo\n"
    [one] = diagnoses(result)
    assert one.hunk_index == 1


def test_resending_a_landed_edit_is_not_applied_twice():
    patch = "@a.ts\n anchor\n+inserted\n\n@b.ts\n-old\n+new\n"
    files = ok({"a.ts": "anchor\nend\n", "b.ts": "old\n"}, patch)
    result, files = run(files, patch + "\n@a.ts\n-end\n+fin\n")
    assert result.ok, result.feedback()
    assert files["a.ts"] == "anchor\ninserted\nfin\n"
    assert files["b.ts"] == "new\n"
    assert result.repairs.count("already_in_place") == 2


def test_a_single_failure_still_reads_as_one():
    result, _ = run(
        THREE_FILES,
        "--- a/main.css\n+++ b/main.css\n@@\n-  --bg: #001933;\n+  --bg: #123456;\n",
    )
    text = result.feedback()
    assert "edits in this patch could not be placed" not in text
    assert text.startswith("main.css: ")


@pytest.mark.parametrize(
    "patch",
    [
        (
            '@App.tsx\n-import React from "react";\n-\n export default function App() {\n-  return (\n'
            "-    <div>hi</div>\n-  );\n+  return <Game />;\n }\n"
        ),
        (
            "@App.tsx\n export default function App() {\n-  return (\n-    <div>hi</div>\n-  );\n"
            "+  return <Game />;\n }\n-export const unused = 1;\n"
        ),
    ],
    ids=["leading", "trailing"],
)
def test_a_removed_line_that_is_not_in_the_file_is_dropped(patch):
    app = "export default function App() {\n  return (\n    <div>hi</div>\n  );\n}\n"
    result, after = run({"App.tsx": app}, patch)
    assert result.ok, result.feedback()
    assert (
        after["App.tsx"] == "export default function App() {\n  return <Game />;\n}\n"
    )
    assert "stray_removals" in result.repairs


def test_a_removed_line_elsewhere_in_the_file_is_not_dropped():
    lines = 'import x from "x";\n\nconst a = 1;\nconst b = 2;\n'.split("\n")
    real = Hunk.from_ops(
        [("-", 'import x from "x";'), (" ", "const b = 2;"), ("+", "const c = 3;")]
    )
    assert _without_stray_removals(lines, real) is None
    stray = Hunk.from_ops(
        [("-", 'import y from "y";'), (" ", "const b = 2;"), ("+", "const c = 3;")]
    )
    repaired = _without_stray_removals(lines, stray)
    assert repaired is not None and len(repaired.before) == 1


def test_edits_in_a_file_are_placed_in_order_after_their_anchor():
    backend = "\n".join(
        [
            'app.post("/api/rooms", (_req, res) => {',
            "  const room: Room = { code, game: newGame() };",
            '  res.json({ player: "X" });',
            "});",
            "",
            'app.post("/api/rooms/:code/join", (req, res) => {',
            '  res.json({ player: "O" });',
            "});",
            "",
            'app.get("/api/rooms/:code", (req, res) => {',
            "  res.json({ state: 1 });",
            "});",
            "",
        ]
    )
    after = ok(
        {"index.ts": backend},
        '@index.ts\n app.post("/api/rooms", (_req, res) => {\n@@\n'
        "-  const room: Room = { code, game: newGame() };\n"
        "+  const room: Room = { code, game: newGame(), solo: false };\n@@\n"
        ' });\n+\n+app.post("/api/solo", (_req, res) => {\n+  res.json({ player: "X" });\n+});\n',
    )
    got = after["index.ts"]
    rooms = got.index('res.json({ player: "X" });\n});')
    solo = got.index('app.post("/api/solo"')
    join = got.index('app.post("/api/rooms/:code/join"')
    assert rooms < solo < join


def test_an_anchor_only_edit_points_at_the_next_one():
    app = (
        '  const applyRoom = () => {\n    setError("");\n  };\n\n'
        "  const createRoom = async () => {\n    setBusy(true);\n  };\n\n"
        "  const joinRoom = async () => {\n    setBusy(false);\n  };\n"
    )
    after = ok(
        {"App.tsx": app},
        "@App.tsx\n   const createRoom = async () => {\n@@\n   };\n+\n+  const startSolo = async () => {};\n",
    )
    assert "    setBusy(true);\n  };\n\n  const startSolo" in after["App.tsx"]


def test_a_repeated_line_with_no_anchor_is_still_ambiguous():
    result, _ = run({"a.ts": "  };\n  };\n"}, "@a.ts\n   };\n+  x();\n")
    assert "too ambiguous" in result.feedback()


def test_a_piece_of_a_long_line_is_replaced_in_place():
    line = (
        '  return <main className="lobby"><h1>Title</h1><button className="primary-button full" '
        "onClick={createRoom} disabled={busy}>Create room</button><p>After</p></main>;"
    )
    result, after = run(
        {"App.tsx": line + "\n"},
        '@App.tsx\n-<button className="primary-button full" onClick={createRoom} disabled={busy}>Create room</button>\n'
        '+<button className="primary-button full" onClick={createRoom} disabled={busy}>Create room</button>\n'
        '+<button className="secondary-button full" onClick={startSolo} disabled={busy}>Play solo</button>\n',
    )
    assert result.ok, result.feedback()
    got = after["App.tsx"]
    assert '<h1>Title</h1><button className="primary-button full"' in got
    assert "Play solo</button><p>After</p></main>;" in got
    assert "within_line" in result.repairs


# ---------------------------------------------------------------------------
# anchor_repair_test.go
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("repair", "want"),
    [
        ("+added\n   anchor\tvalue", "start\nadded\n  anchor\tvalue\nend\n"),
        ("   anchor\tvalue\n+added", "start\n  anchor\tvalue\nadded\nend\n"),
        ("-  anchor\tvalue\n+added", "start\nadded\nend\n"),
    ],
    ids=["before", "after", "replacement"],
)
def test_anchor_repair_writes_the_companion_then_places_repair(repair, want):
    original = "start\n  anchor\tvalue\nend\n"
    files = {"target.txt": original, "companion.txt": "old\n"}
    companion = "@companion.txt\n-old\n+new\n\n"
    result, files = run(files, companion + "@target.txt\n+added\n")
    [one] = diagnoses(result)
    assert (one.path, one.kind) == ("target.txt", "no_anchor_insert")
    assert one.message.startswith("a hunk has no context to place it by. ")
    # The advice describes the rule without inventing source for the repair.
    assert "current file" in one.message and "anchor\tvalue" not in one.message
    assert files["target.txt"] == original
    assert files["companion.txt"] == "new\n" and list(result.written) == [
        "companion.txt"
    ]

    result, files = run(files, companion + "@target.txt\n" + repair + "\n")
    assert result.ok, result.feedback()
    assert files["target.txt"] == want
    assert files["companion.txt"] == "new\n"


@pytest.mark.parametrize(
    ("files", "patch"),
    [
        ({"target.txt": ""}, "@target.txt\n+added\n"),
        ({"target.txt": " \t\n\t\n"}, "@target.txt\n+added\n"),
        ({}, "@+target.txt\nadded\n"),
    ],
    ids=["empty", "whitespace", "new"],
)
def test_anchor_repair_preserves_empty_and_new_file_insertion(files, patch):
    after = ok(files, patch)
    assert after["target.txt"] == "added\n"


def test_create_header_on_a_nonempty_file_rewrites_it():
    # Deliberately different from yolocoder (which refused): "@+" is the whole
    # literal content, so on an existing file it is a rewrite (as in Mochi).
    # Whether a path may be rewritten is the workspace's call, not the patch's.
    result, after = run({"target.txt": "existing\n"}, "@+target.txt\nreplacement\n")
    assert result.ok
    assert after["target.txt"] == "replacement\n"


# ---------------------------------------------------------------------------
# anchor_contract_test.go
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("before", "patch", "want"),
    [
        (
            "enabled = false\nkeep = true\n",
            "@settings.txt\n-enabled = false\n+enabled = true\n",
            "enabled = true\nkeep = true\n",
        ),
        (
            "[first]\nvalue = 1\n[second]\nvalue = 1\n",
            "@settings.txt\n [second]\n value = 1\n+extra = 2\n",
            "[first]\nvalue = 1\n[second]\nvalue = 1\nextra = 2\n",
        ),
        (
            "old\n\nremaining\n",
            "@settings.txt\n-old\n-\n-remaining\n+new\n+contents\n",
            "new\ncontents\n",
        ),
    ],
    ids=[
        "removed line anchors replacement",
        "unchanged context anchors insertion",
        "whole file removes current contents",
    ],
)
def test_compact_anchor_contract_edits(before, patch, want):
    assert ok({"settings.txt": before}, patch)["settings.txt"] == want


ANCHOR_PREFIX = "@other.txt\n-old\n+changed\n@+notes.txt\nNotes\n+ literal plus\n"


@pytest.mark.parametrize(
    ("before", "refused", "repaired", "want"),
    [
        (
            "first\nlast\n",
            "@settings.txt\n+inserted\n",
            "@settings.txt\n first\n+inserted\n",
            "first\ninserted\nlast\n",
        ),
        (
            "first\nvalue\nlast\nvalue\n",
            "@settings.txt\n value\n+inserted\n",
            "@settings.txt\n last\n value\n+inserted\n",
            "first\nvalue\nlast\nvalue\ninserted\n",
        ),
    ],
    ids=["insertion lacks context", "insertion anchor repeats"],
)
def test_compact_anchor_contract_refusal_and_repair_are_atomic(
    before, refused, repaired, want
):
    files = {"settings.txt": before, "other.txt": "old\n"}
    result, files = run(files, ANCHOR_PREFIX + refused)
    [one] = diagnoses(result)
    assert one.path == "settings.txt"
    assert files["settings.txt"] == before
    assert files["other.txt"] == "changed\n"
    assert list(result.written) == ["other.txt", "notes.txt"]

    result, files = run(files, ANCHOR_PREFIX + repaired)
    assert result.ok, result.feedback()
    assert files == {
        "settings.txt": want,
        "other.txt": "changed\n",
        "notes.txt": "Notes\n+ literal plus\n",
    }


def test_compact_anchor_contract_create_rewrites_with_its_companions():
    # The Go table's third row ("creation cannot replace nonempty file") is
    # a rewrite here; see test_create_header_on_a_nonempty_file_rewrites_it.
    result, files = run(
        {"settings.txt": "first\nlast\n", "other.txt": "old\n"},
        ANCHOR_PREFIX + "@+settings.txt\nreplacement\n",
    )
    assert result.ok
    assert files == {
        "settings.txt": "replacement\n",
        "other.txt": "changed\n",
        "notes.txt": "Notes\n+ literal plus\n",
    }


# ---------------------------------------------------------------------------
# noop_test.go
# ---------------------------------------------------------------------------


def test_a_context_only_hunk_does_not_kill_the_patch():
    css = "\n".join(
        [
            ".game-intro-screen {",
            "  image-rendering: pixelated;",
            "}",
            ".game-intro-screen__panel {",
            "  box-shadow: 6px 6px 0 #071f07;",
            "}",
            ".other {",
            "}",
            ".game-intro-screen__panel {",
            "  color: #e0f8a0;",
            "}",
            "",
        ]
    )
    patch = "\n".join(
        [
            "@index.css",
            " .game-intro-screen {",
            "@@",
            "  image-rendering: pixelated;",
            "+ text-shadow: 2px 0 0 rgba(15, 56, 15, .35);",
            "@@",
            " .game-intro-screen__panel {",
            "@@",
            "  box-shadow: 6px 6px 0 #071f07;",
            "+ image-rendering: pixelated;",
            "",
        ]
    )
    got = ok({"index.css": css}, patch)["index.css"]
    assert "text-shadow: 2px 0 0" in got
    assert "image-rendering: pixelated;\n}" in got


def test_a_patch_of_only_context_is_not_counted_as_applied():
    result, _ = run({"a.css": ".a {\n}\n"}, "@a.css\n .a {\n }\n")
    assert result.noop and not result.ok


# ---------------------------------------------------------------------------
# repo_test.go (the apply cases; git itself is not used here)
# ---------------------------------------------------------------------------


def test_unified_new_file_from_dev_null():
    patch = "diff --git a/hello.txt b/hello.txt\nnew file mode 100644\n--- /dev/null\n+++ b/hello.txt\n@@ -0,0 +1 @@\n+hello\n"
    [edit] = parse_patch(patch)
    assert edit.create
    assert ok({}, patch)["hello.txt"] == "hello\n"


def test_apply_tolerates_a_miscounted_hunk_header():
    patch = "diff --git a/file.txt b/file.txt\n--- a/file.txt\n+++ b/file.txt\n@@ -1,3 +1,3 @@\n a\n b\n+new line\n c\n"
    assert ok({"file.txt": "a\nb\nc\n"}, patch)["file.txt"] == "a\nb\nnew line\nc\n"


def test_apply_unified_diff():
    patch = "diff --git a/file.txt b/file.txt\n--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new\n"
    assert ok({"file.txt": "old\n"}, patch)["file.txt"] == "new\n"


# ---------------------------------------------------------------------------
# Beyond the Go suite: Mochi's additions, the ticket's decisions, and the
# line-ending contract.
# ---------------------------------------------------------------------------


def test_crlf_files_keep_crlf():
    files = {"a.html": "<p>one</p>\r\n<p>two</p>\r\n"}
    after = ok(files, "@a.html\n-<p>one</p>\n+<p>uno</p>\n+<p>extra</p>\n")
    assert after["a.html"] == "<p>uno</p>\r\n<p>extra</p>\r\n<p>two</p>\r\n"


def test_crlf_in_the_patch_itself_is_trimmed():
    after = ok(
        {"a.ts": "const x = 1;\n"}, "@a.ts\r\n-const x = 1;\r\n+const x = 2;\r\n"
    )
    assert after["a.ts"] == "const x = 2;\n"


def test_crlf_file_rewritten_with_create_stays_crlf():
    after = ok({"a.txt": "a\r\nb\r\n"}, "@+a.txt\nx\ny\n")
    assert after["a.txt"] == "x\r\ny\r\n"


@pytest.mark.parametrize("trailing", ["", "\n"])
def test_trailing_newline_state_is_kept(trailing):
    after = ok({"a.ts": "one\ntwo" + trailing}, "@a.ts\n-two\n+three\n")
    assert after["a.ts"] == "one\nthree" + trailing


def test_last_line_without_newline_can_be_replaced_with_several():
    after = ok({"a.ts": "one\ntwo"}, "@a.ts\n one\n-two\n+2\n+3\n")
    assert after["a.ts"] == "one\n2\n3"


def test_no_newline_marker_is_ignored():
    unified = "--- a/a.ts\n+++ b/a.ts\n@@\n-two\n\\ No newline at end of file\n+three\n\\ No newline at end of file\n"
    assert ok({"a.ts": "one\ntwo"}, unified)["a.ts"] == "one\nthree"
    compact = "@a.ts\n-two\n\\ No newline at end of file\n+three\n"
    assert ok({"a.ts": "one\ntwo"}, compact)["a.ts"] == "one\nthree"


def test_whitespace_runs_squeezed_is_the_last_match_tier():
    # Mochi's fourth tier: aligned columns in a table counted wrong.
    files = {
        "README.md": "| name   | value |\n|--------|-------|\n| alpha  | 1     |\n"
    }
    after = ok(files, "@README.md\n-| alpha | 1 |\n+| alpha  | 2     |\n")
    assert "| alpha  | 2     |" in after["README.md"]


def test_an_ambiguous_three_line_block_is_not_placed_at_its_first_match():
    # The ticket's decision, different from both ports: they placed a block
    # of 3+ lines at its first match. Two identical cards are exactly where
    # editing the wrong one would go unnoticed.
    card = '<div class="card">\n  <h2>Plan</h2>\n  <p>$10</p>\n</div>\n'
    files = {"index.html": "<main>\n" + card + "<hr>\n" + card + "</main>\n"}
    result, after = run(
        files,
        '@index.html\n <div class="card">\n   <h2>Plan</h2>\n-  <p>$10</p>\n+  <p>$12</p>\n',
    )
    [one] = diagnoses(result)
    assert one.kind == "ambiguous"
    assert "appear 2 times" in one.message
    assert "at line 2:" in one.message and "at line 7:" in one.message
    # The window widened until the two candidates read differently: one is
    # preceded by "<main>", the other by "<hr>".
    assert "    <main>" in one.message and "    <hr>" in one.message
    assert '  > <div class="card">' in one.message
    assert after == files


def test_ordered_placement_still_resolves_a_repeated_block():
    card = '<div class="card">\n  <h2>Plan</h2>\n  <p>$10</p>\n</div>\n'
    files = {"index.html": "<main>\n" + card + "<hr id=second>\n" + card + "</main>\n"}
    after = ok(
        files,
        '@index.html\n-<hr id=second>\n+<hr id="second">\n\n'
        ' <div class="card">\n   <h2>Plan</h2>\n-  <p>$10</p>\n+  <p>$12</p>\n',
    )
    assert after["index.html"].count("$10") == 1
    assert after["index.html"].index("$12") > after["index.html"].index(
        '<hr id="second">'
    )


def test_disambiguate_window_is_capped_at_six_lines():
    block = "same\n" * 1
    filler = "x\n" * 10
    files = {"a.txt": filler + block + filler + block + filler}
    result, _ = run(files, "@a.txt\n-same\n+other\n")
    [one] = diagnoses(result)
    # Never distinct (all "x" around), so the window stops at 6 lines each side.
    first = one.message.split("at line 11:")[1].split("at line")[0]
    assert first.count("\n    x") == 12


def test_editing_a_missing_file_says_to_create_it():
    result, _ = run({"a.ts": "x\n"}, "@b.ts\n-x\n+y\n")
    [one] = diagnoses(result)
    assert (one.kind, one.hunk_index) == ("no_such_file", -1)
    assert "use @+b.ts" in one.message


def test_added_only_patch_covering_most_of_the_file_is_a_rewrite():
    # Mochi: a whole new version written as "+" lines, split by its own
    # blank lines into several context-free edits.
    files = {"a.css": "body { color: red; }\n\nh1 { margin: 0; }\n"}
    result, after = run(
        files,
        "@a.css\n+body { color: blue; }\n@@\n+h1 { margin: 1px; }\n+p { margin: 0; }\n",
    )
    assert result.ok, result.feedback()
    # The edits were separated where the file had blank lines; they come back.
    assert (
        after["a.css"]
        == "body { color: blue; }\n\nh1 { margin: 1px; }\np { margin: 0; }\n"
    )
    assert "whole_file_rewrite" in result.repairs


def test_added_only_patch_split_by_blank_lines_rewrites_with_them():
    files = {"a.css": "body { color: red; }\n\nh1 { margin: 0; }\n"}
    result, after = run(
        files,
        "@a.css\n+body { color: blue; }\n\n+h1 { margin: 1px; }\n\n+p { margin: 0; }\n",
    )
    assert result.ok
    # The blank lines between "+" runs are kept as part of the addition.
    assert (
        after["a.css"]
        == "body { color: blue; }\n\nh1 { margin: 1px; }\n\np { margin: 0; }\n"
    )


def test_a_short_anchorless_insertion_is_not_a_rewrite():
    result, _ = run(
        {"a.ts": "first\nsecond\nthird\nfourth\n"}, "@a.ts\n+inserted line here\n"
    )
    [one] = diagnoses(result)
    assert one.kind == "no_anchor_insert"


def test_scattered_lines_name_where_each_line_is():
    files = {"a.ts": "alpha();\nbeta();\ngamma();\ndelta();\n"}
    # Two lines that are each unique but not adjacent, and a third that
    # repeats one of them, so the hunk cannot be split.
    result, _ = run(
        {"a.ts": files["a.ts"] + "alpha();\n"},
        "@a.ts\n alpha();\n-delta();\n+omega();\n",
    )
    [one] = diagnoses(result)
    assert one.kind == "scattered"
    assert "line 1: alpha();" in one.message
    assert "line 4: delta();" in one.message


@pytest.mark.parametrize("trailer", ["@END", "*** END ***", "EOF", "END", "@eof"])
def test_bare_end_markers_end_the_patch_and_junk_after_is_dropped(trailer):
    after = ok(
        {"a.ts": "const x = 1;\n"},
        f"@a.ts\n-const x = 1;\n+const x = 2;\n{trailer}\nrandom junk ✓\n",
    )
    assert after["a.ts"] == "const x = 2;\n"


def test_end_marker_is_not_a_created_file_and_not_its_content():
    result, after = run({}, "@+run.sh\necho hi\n@END\n")
    assert result.ok
    assert after == {"run.sh": "echo hi\n"}


def test_a_heredoc_eof_inside_a_created_file_is_content():
    script = "cat <<EOF\nhello\nEOF\necho done\n"
    after = ok({}, "@+run.sh\n" + script)
    assert after["run.sh"] == script


def test_a_lowercase_end_context_line_is_not_a_terminator():
    after = ok({"a.rb": "def go\n  1\nend\n"}, "@a.rb\n def go\n-  1\n+  2\nend\n")
    assert after["a.rb"] == "def go\n  2\nend\n"


def test_a_removed_sql_comment_in_a_unified_hunk_is_not_a_header():
    files = {"q.sql": "-- old note\nselect 1;\n"}
    after = ok(
        files,
        "--- a/q.sql\n+++ b/q.sql\n@@ -1,2 +1,2 @@\n--- old note\n+-- new note\n select 1;\n",
    )
    assert after["q.sql"] == "-- new note\nselect 1;\n"


def test_unified_deletion_is_refused_by_name():
    with pytest.raises(PatchParseError, match="deletes gone.txt"):
        parse_patch("--- a/gone.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n")


def test_apply_patch_move_is_refused_by_name():
    with pytest.raises(PatchParseError, match="moves a.ts"):
        parse_patch(
            "*** Begin Patch\n*** Update File: a.ts\n*** Move to: b.ts\n@@\n-x\n+y\n*** End Patch"
        )


@pytest.mark.parametrize("patch", ["just some prose\n", "-x\n+y\n", ""])
def test_a_patch_with_no_file_header_is_refused(patch):
    with pytest.raises(PatchParseError, match="no file headers"):
        parse_patch(patch)


def test_a_fenced_patch_is_read():
    after = ok(
        {"a.ts": "const x = 1;\n"},
        "```diff\n@a.ts\n-const x = 1;\n+const x = 2;\n```\n",
    )
    assert after["a.ts"] == "const x = 2;\n"


def test_empty_hunks_are_dropped():
    [edit] = parse_patch(
        "*** Begin Patch\n*** Update File: a.ts\n@@\n@@\n-x\n+y\n*** End Patch"
    )
    assert len(edit.hunks) == 1
    [edit] = parse_patch("@a.ts\n\n\n-x\n+y\n\n\n")
    assert len(edit.hunks) == 1


def test_a_header_with_no_edits_changes_nothing():
    result, _ = run({"a.ts": "x\n"}, "@a.ts\n")
    assert result.noop


def test_multiple_failures_carry_their_hunk_index():
    result, _ = run(
        {"a.ts": "one\ntwo\nthree\n"}, "@a.ts\n-uno\n+1\n\n-two\n+2\n\n-tres\n+3\n"
    )
    assert [one.hunk_index for one in diagnoses(result)] == [0, 2]


def test_misrouted_lines_in_a_new_file_of_the_patch_are_not_claimed():
    # Misrouting compares against the files as they were; a file the patch
    # creates had no lines to steal.
    result, _ = run(
        {"a.ts": "x\n"}, "@+b.ts\nconst q = 1;\n\n@a.ts\n-const q = 1;\n+const q = 2;\n"
    )
    assert "a.ts" in result.failed
    assert diagnoses(result)[0].kind != "misrouted"


def test_hunk_constructed_without_ops_still_places():
    plain = [FileEdit("a.ts", False, (Hunk(before=("x",), after=("y",)),))]
    result = apply_patch(plain, {"a.ts": "x\n"}.get)
    assert result.written == {"a.ts": "y\n"}
