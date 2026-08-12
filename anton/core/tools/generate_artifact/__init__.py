"""Deterministic generation FSM behind the ``generate_artifact`` tool.

Public entry point: ``generate(session, artifact_type, artifact_path,
context, slug)``. The outer tool handler validates input and reads artifact
metadata; everything below this surface is provider-agnostic. ``generate``
builds a ``GenState`` and hands off to ``orchestrator.run``, which walks the
graph nodes (data-sufficiency loop → tech spec → api spec → backend/frontend
generation with verification → launch & health check). Nodes reach real data
via the `scratchpad` sub-tool, guided by the brief's ``## Data`` section.
"""

from .engine import generate, MAX_ROUNDS

__all__ = ["generate", "MAX_ROUNDS"]
