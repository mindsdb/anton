"""Inner generator used by the experimental ``generate_artifact`` tool.

Public entry point: ``generate(session, artifact_type, artifact_path,
context)``. The outer tool handler validates input and reads artifact
metadata; everything below this surface is provider-agnostic. The
sub-generator reaches real data via the `scratchpad` sub-tool, guided by the
brief's free-form ``## Data`` section.
"""

from .engine import generate, MAX_ROUNDS

__all__ = ["generate", "MAX_ROUNDS"]
