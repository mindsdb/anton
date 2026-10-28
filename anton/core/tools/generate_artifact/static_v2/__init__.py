"""Replacement static pipeline, isolated behind an evaluation rollout switch."""
from .collect import collect
from .compose import compose
from .content_review import EvidenceRequest


async def run(state):
    from ..progress import StepCounter
    state.step_counter = StepCounter(["gathering", "generate_frontend", "verify_frontend"])
    feedback = ''
    for _ in range(2):
        evidence = await collect(state, feedback=feedback)
        if isinstance(evidence, str):
            return evidence
        result = await compose(state, evidence)
        if not isinstance(result,EvidenceRequest):
            return result
        feedback = '\n'.join(result.issues)
        state.record('static_evidence','refresh_requested',feedback)
    return 'Static artifact still needs source evidence; the previous artifact is preserved. ' + feedback
