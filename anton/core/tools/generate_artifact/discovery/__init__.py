"""Discovery phases of the artifact pipeline: gather -> brief -> PRD.

Formerly the standalone `generate_prd` tool. There is no entry point here any
more — `generate_artifact.generate` runs all five phases — and the boundary
these phases hand over is no longer a file the next tool re-reads: it is
`GenState` in memory on the hot path, and `discovery.json` on a cold start.

Layout: `engine` (phase A, the gathering loop), `brief` (phase B),
`prd` (phase C), `checkpoint` (the persisted boundary), `notes` (the
deterministic renderers phase E reads), `orchestrator` (the sequencer).
"""
