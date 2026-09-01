"""Wire protocol constants shared between LocalScratchpadRuntime and scratchpad_boot.py.

These delimiter strings must be identical on both sides of the subprocess pipe.
Neither side should redefine them — import from here.
"""

CELL_DELIM = "__ANTON_CELL_END__"
RESULT_START = "__ANTON_RESULT__"
RESULT_END = "__ANTON_RESULT_END__"
PROGRESS_MARKER = "__ANTON_PROGRESS__"
HEARTBEAT_MARKER = "__ANTON_HEARTBEAT__"
STDOUT_CHUNK_MARKER = "__ANTON_STDOUT_CHUNK__"
# Dead: the worker no longer auto-installs on a missing import, so nothing
# emits these anymore. local.py still parses for them defensively.
INSTALL_START_MARKER = "__ANTON_INSTALL_START__"
INSTALL_END_MARKER = "__ANTON_INSTALL_END__"

# The hint the worker prepends to a ModuleNotFoundError traceback (ENG-1635).
# Lives here so the session side (nudge tests, and anything that needs the
# exact shape) shares one definition with the boot script. Deliberately does
# NOT tell the model to declare the failed name in 'packages' — that would
# route the same hallucinated package through the surviving install path one
# turn later, which is the attack this ticket closes.
MISSING_MODULE_HINT = (
    "'{name}' is not installed, and imports never install anything. Check "
    "the import itself first — a missing module is often a wrong or invented "
    "name, not a missing package. Only a real PyPI distribution this task "
    "genuinely needs should ever be installed.\n"
)


def heal_surrogate_source(code: str) -> str:
    """Return ``code`` with any lone surrogates resolved to valid UTF-8.

    Cell source can arrive carrying lone surrogates (``\\udcXX``): a non-ASCII
    Windows path byte (e.g. from ``Área de Trabalho`` or an emoji filename) is
    surrogate-escaped upstream on a non-UTF-8 host and passed through the
    scratchpad's lenient ``surrogateescape`` stdin. ``compile()`` strict-encodes
    the source to UTF-8 and rejects a lone surrogate ("surrogates not allowed"),
    crashing the cell before any user code runs (ENG-981). UTF-8 mode never
    makes ``compile()`` lenient, so we must clean the source ourselves.

    Strategy, in two steps so a single unrecoverable surrogate can't take down
    the recoverable ones elsewhere in the same cell (mixed-cell data loss):

    1. ``surrogateescape`` only maps the ``DC80..DCFF`` byte-escape range. Any
       *other* surrogate (a high/unpaired one) would make the ``surrogateescape``
       encode raise and force the whole cell down a lossy path — so replace those
       up front with U+FFFD.
    2. Re-encode the rest via ``surrogateescape`` to recover the original bytes,
       then decode UTF-8 with ``errors="replace"``. Byte-escaped halves of a real
       multibyte character (the common path case) reassemble exactly — the cell
       compiles *and* references the correct path — while any leftover stray byte
       becomes U+FFFD in the same pass. Never raises. A no-op for clean source.
    """
    if not any("\ud800" <= ch <= "\udfff" for ch in code):
        return code
    # Step 1: scrub surrogates outside the escapable DC80..DCFF range.
    code = "".join(
        ch if not ("\ud800" <= ch <= "\udfff") or ("\udc80" <= ch <= "\udcff")
        else "�"
        for ch in code
    )
    # Step 2: recover DC80..DCFF byte-escapes; replace anything still invalid.
    return code.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
