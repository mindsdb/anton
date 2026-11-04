"""Chimera: a main agent that delegates bounded legwork to a cheaper sidekick.

The main agent (the planning role) calls the `delegate` tool (`tools.py`). The
tool runs a `SubAgent` loop on the sidekick role (`subagent.py`) with its own
persistent messages (`state.py`) and a restricted tool set, watched by Jev
(`policy.py`, `jev_questions.py`), and returns a `SidekickReport` (`report.py`)
for the main agent to review. Everything is off unless `ChimeraConfig.enabled`
and the client has a sidekick model (`config.py`).

Modules are imported directly rather than re-exported here, so the LLM client
can read `config` without pulling the tool layer in.
"""
