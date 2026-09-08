"""Coverage for the deterministic patch engine.

These are pure-string tests: no filesystem, no LLM, no anton. That is the
point of the module they cover — the valuable part of yolo mode is a set
of functions over strings, and it should be provable in milliseconds.

The cases are the real failures seen while building yolocoder, not
invented ones. Where a test looks oddly specific, it is because a model
did exactly that.
"""

from __future__ import annotations

import pytest

from anton.core.yolo.patch import (
    PatchError,
    apply_hunks,
    is_apply_patch_format,
    locate,
    parse_patch,
)

# ─── The headline behaviour: numbers are ignored ────────────────────────


def test_wrong_line_numbers_do_not_matter():
    """The whole reason this module exists.

    Every number in this header is a lie — wrong start, wrong counts. The
    content is right, so it applies.
    """
    content = "one\ntwo\nthree\nfour\n"
    patch = (
        "--- a/f.txt\n"
        "+++ b/f.txt\n"
        "@@ -99,44 +1201,7 @@\n"
        " two\n"
        "-three\n"
        "+THREE\n"
        " four\n"
    )
    [file_patch] = parse_patch(patch)
    assert apply_hunks(content, file_patch.hunks) == "one\ntwo\nTHREE\nfour\n"


def test_a_hunk_with_no_trailing_context_still_applies():
    """git rejects this outright: a hunk ending on its last change asserts
    the file ends there. Models write it constantly."""
    content = "header\ntitle: old\nfooter\n"
    patch = "--- a/f\n+++ b/f\n@@\n header\n-title: old\n+title: new\n"
    [file_patch] = parse_patch(patch)
    assert apply_hunks(content, file_patch.hunks) == "header\ntitle: new\nfooter\n"


def test_indentation_drift_is_tolerated():
    """A model reproducing a file by eye gets the characters right and the
    leading whitespace slightly wrong. The exact pass fails, the stripped
    pass finds it."""
    content = "def f():\n        return 1\n"
    patch = "--- a/f\n+++ b/f\n@@\n-    return 1\n+    return 2\n"
    [file_patch] = parse_patch(patch)
    assert "return 2" in apply_hunks(content, file_patch.hunks)


# ─── Refusing rather than guessing ──────────────────────────────────────


def test_an_unfindable_hunk_is_an_error_not_a_no_op():
    content = "alpha\nbeta\n"
    patch = "--- a/f\n+++ b/f\n@@\n-gamma\n+delta\n"
    [file_patch] = parse_patch(patch)
    with pytest.raises(PatchError, match="could not find"):
        apply_hunks(content, file_patch.hunks)


def test_a_short_ambiguous_hunk_is_refused():
    """Picking the first match is how an edit lands in the wrong function
    and reports success."""
    lines = ["}", "", "}", "", "}"]
    with pytest.raises(PatchError, match="ambiguous"):
        locate(lines, ["}"])


def test_a_long_repeated_block_is_placed_at_the_first_match():
    """Ambiguity only applies to blocks short enough to be a coincidence.
    Three identical lines in a row is a real location."""
    lines = ["a", "b", "c", "x", "a", "b", "c"]
    assert locate(lines, ["a", "b", "c"]) == 0


def test_a_hunk_with_no_context_cannot_be_placed():
    content = "some\nexisting\ncontent\n"
    with pytest.raises(PatchError, match="no context"):
        apply_hunks(content, parse_patch("--- a/f\n+++ b/f\n@@\n+new\n")[0].hunks)


def test_a_pure_insertion_into_an_empty_file_is_unambiguous():
    """There is only one place it can go."""
    [file_patch] = parse_patch("--- a/f\n+++ b/f\n@@\n+hello\n+world\n")
    assert apply_hunks("", file_patch.hunks) == "hello\nworld"


# ─── apply_patch (V4A) dialect ──────────────────────────────────────────


def test_add_file_creates_a_file_with_no_at_header():
    """'*** Add File:' is followed straight by its + lines. Waiting for an
    '@@' reads the whole file as noise and creates nothing."""
    patch = (
        "*** Begin Patch\n"
        "*** Add File: greet.py\n"
        "+def hi():\n"
        '+    return "hi"\n'
        "*** End Patch\n"
    )
    [file_patch] = parse_patch(patch)
    assert file_patch.path == "greet.py"
    assert apply_hunks("", file_patch.hunks) == 'def hi():\n    return "hi"'


def test_update_file_drops_the_empty_hunk_its_header_creates():
    patch = (
        "*** Begin Patch\n"
        "*** Update File: f.txt\n"
        "@@\n"
        " keep\n"
        "-old\n"
        "+new\n"
        "*** End Patch\n"
    )
    [file_patch] = parse_patch(patch)
    assert len(file_patch.hunks) == 1
    assert apply_hunks("keep\nold\n", file_patch.hunks) == "keep\nnew\n"


def test_the_format_is_detected_without_being_confused_by_a_unified_diff():
    assert is_apply_patch_format("*** Begin Patch\n*** Add File: a\n+x\n")
    assert not is_apply_patch_format("--- a/f\n+++ b/f\n@@\n-x\n+y\n")
    assert not is_apply_patch_format("diff --git a/f b/f\n--- a/f\n")


def test_deleting_a_file_is_refused_loudly():
    with pytest.raises(PatchError, match="deletes"):
        parse_patch("*** Begin Patch\n*** Delete File: important.txt\n*** End Patch\n")


# ─── Parsing edge cases that cost real debugging time ───────────────────


def test_a_trailing_blank_line_is_not_context():
    """It comes from the patch ending in a newline. Counted as an empty
    context line it makes every final hunk unmatchable."""
    [file_patch] = parse_patch("--- a/f\n+++ b/f\n@@\n keep\n-old\n+new\n\n\n")
    assert file_patch.hunks[0].before == ["keep", "old"]


def test_an_empty_line_inside_a_hunk_is_context():
    content = "top\n\nbottom\n"
    [file_patch] = parse_patch("--- a/f\n+++ b/f\n@@\n top\n\n-bottom\n+BOTTOM\n")
    assert apply_hunks(content, file_patch.hunks) == "top\n\nBOTTOM\n"


def test_dev_null_headers_do_not_become_a_file_named_dev_null():
    patch = "--- /dev/null\n+++ b/new.txt\n@@\n+created\n"
    [file_patch] = parse_patch(patch)
    assert file_patch.path == "new.txt"


def test_ab_prefixes_are_stripped():
    [file_patch] = parse_patch("--- a/src/x.py\n+++ b/src/x.py\n@@\n-a\n+b\n")
    assert file_patch.path == "src/x.py"


def test_a_patch_with_no_file_header_is_an_error():
    with pytest.raises(PatchError, match="no file headers"):
        parse_patch("just some prose the model wrote\n")


def test_a_hunk_before_any_file_header_is_an_error():
    with pytest.raises(PatchError, match="before any file header"):
        parse_patch("@@\n-a\n+b\n")


def test_several_hunks_in_one_file_compose():
    content = "one\ntwo\nthree\nfour\nfive\n"
    patch = (
        "--- a/f\n+++ b/f\n"
        "@@\n one\n-two\n+TWO\n"
        "@@\n four\n-five\n+FIVE\n"
    )
    [file_patch] = parse_patch(patch)
    assert apply_hunks(content, file_patch.hunks) == "one\nTWO\nthree\nfour\nFIVE\n"


def test_a_no_newline_marker_is_ignored():
    patch = "--- a/f\n+++ b/f\n@@\n-old\n\\ No newline at end of file\n+new\n"
    [file_patch] = parse_patch(patch)
    assert apply_hunks("old", file_patch.hunks) == "new"
