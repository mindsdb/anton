"""Wire protocol constants shared between LocalScratchpadRuntime and scratchpad_boot.py.

These delimiter strings must be identical on both sides of the subprocess pipe.
Neither side should redefine them — import from here.
"""

CELL_DELIM = "__ANTON_CELL_END__"
RESULT_START = "__ANTON_RESULT__"
RESULT_END = "__ANTON_RESULT_END__"
PROGRESS_MARKER = "__ANTON_PROGRESS__"


def heal_surrogate_source(code: str) -> str:
    """Return ``code`` with any lone surrogates resolved to valid UTF-8.

    Cell source can arrive carrying lone surrogates (``\\udcXX``): a non-ASCII
    Windows path byte (e.g. from ``Área de Trabalho`` or an emoji filename) is
    surrogate-escaped upstream on a non-UTF-8 host and passed through the
    scratchpad's lenient ``surrogateescape`` stdin. ``compile()`` strict-encodes
    the source to UTF-8 and rejects a lone surrogate ("surrogates not allowed"),
    crashing the cell before any user code runs (ENG-981). UTF-8 mode never
    makes ``compile()`` lenient, so we must clean the source ourselves.

    Strategy: re-encode via ``surrogateescape`` to recover the original bytes,
    then decode as UTF-8. When the surrogates are the byte-escaped halves of a
    real multibyte character (the common path case) this reassembles it exactly
    — so the cell not only compiles but references the *correct* path. A
    genuinely lone surrogate (not a valid byte sequence, or outside the
    escapable range) falls back to the replacement char so ``compile()`` at
    least succeeds instead of crashing. A no-op for clean source.
    """
    if not any("\ud800" <= ch <= "\udfff" for ch in code):
        return code
    try:
        return code.encode("utf-8", "surrogateescape").decode("utf-8")
    except UnicodeError:
        # High/unpaired surrogates outside the surrogateescape range: scrub.
        return code.encode("utf-8", "surrogatepass").decode("utf-8", "replace")
