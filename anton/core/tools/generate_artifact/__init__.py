"""Deterministic generation FSM behind the ``generate_artifact`` tool.

Public entry point: ``generate(session, slug, artifact_path, artifact_type,
user_request, agent_understanding, known_data, user_preferences)``. The outer
tool handler validates input and reads artifact metadata; everything below
this surface is provider-agnostic. ``generate`` decides where a repeat call
is entitled to resume (``discovery/checkpoint.py``), builds a ``GenState``
and hands off to ``orchestrator.run``, which walks five phases: gather →
brief & confirm → PRD → specs → generation with verification, then launch and
health check for a fullstack app.

Phases A-D share one message history; it is dropped once the specs are
written, so what the generation phase gets is assembled from the state —
``spec.md`` plus the rendered ``data_notes`` / ``web_notes`` channels
(``discovery/notes.py``). ``discovery.json`` carries that same boundary
across process restarts.
"""

from .engine import generate, MAX_ROUNDS

__all__ = ["generate", "MAX_ROUNDS"]
