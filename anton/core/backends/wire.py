"""Wire protocol constants shared between LocalScratchpadRuntime and scratchpad_boot.py.

These delimiter strings must be identical on both sides of the subprocess pipe.
Neither side should redefine them — import from here.
"""

CELL_DELIM = "__ANTON_CELL_END__"
RESULT_START = "__ANTON_RESULT__"
RESULT_END = "__ANTON_RESULT_END__"
PROGRESS_MARKER = "__ANTON_PROGRESS__"
