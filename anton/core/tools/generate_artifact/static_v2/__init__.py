"""Replacement static pipeline, isolated behind an evaluation rollout switch."""
from .collect import collect
from .compose import compose


async def run(state):
    from ..progress import StepCounter
    state.step_counter = StepCounter(["gathering", "generate_frontend", "verify_frontend"])
    evidence = await collect(state)
    if isinstance(evidence, str):
        return evidence
    return await compose(state, evidence)
