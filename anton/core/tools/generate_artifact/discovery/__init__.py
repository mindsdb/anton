"""Discovery phases of the artifact pipeline: gather -> brief -> PRD.

Formerly the standalone `generate_prd` tool. The phases still share one
message list, but the boundary they hand over is no longer a file the next
tool re-reads — it is `GenState` in memory on the hot path, and
`discovery.json` on a cold start (see the design document, sections 2.4-2.6).
"""
