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
    then decode as UTF-8 with ``errors="replace"``. When the surrogates are the
    byte-escaped halves of a real multibyte character (the common path case)
    this reassembles it exactly — so the cell not only compiles but references
    the *correct* path. ``errors="replace"`` (rather than a strict decode with a
    full-scrub fallback) matters for a **mixed** cell: a recoverable path and an
    unrelated lone byte in the same source are handled in one pass, so the
    recoverable character isn't lost just because a stray byte sits next to it;
    the stray byte becomes U+FFFD (a genuinely-lone byte can't be recovered, but
    ``compile()`` succeeds instead of crashing). A no-op for clean source.
    """
    if not any("\ud800" <= ch <= "\udfff" for ch in code):
        return code
    try:
        return code.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
    except UnicodeEncodeError:
        # High/unpaired surrogates outside surrogateescape's DC80..DCFF range —
        # the escape codec can't map them at all; surrogatepass can, and the
        # replace-decode then scrubs them. This branch can't raise.
        return code.encode("utf-8", "surrogatepass").decode("utf-8", "replace")
