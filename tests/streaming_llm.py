"""A fake `session._llm` for the artifact pipeline.

Discovery streams its model calls. Most of its tests were written against
`plan` / `code` mocks, so this double replays whatever those return as a
stream: `plan_stream` and `code_stream` look the mocks up at call time, so a
test that reassigns `.plan` after building the session still drives them.
"""
from __future__ import annotations

from types import SimpleNamespace

from anton.core.llm.provider import StreamComplete

STREAM_ONLY_KWARGS = ("max_tokens", "wait_note")


class StreamingLLM(SimpleNamespace):
    def __init__(self, budget: int = 8192, **kw):
        super().__init__(budget=budget, stream_calls=[], **kw)

    def stream_budget(self, role: str = "planning") -> int:
        return self.budget

    def plan_stream(self, **kw):
        return self._replay("planning", lambda: self.plan, kw)

    def code_stream(self, **kw):
        return self._replay("coding", lambda: self.code, kw)

    async def _replay(self, role, call, kw):
        self.stream_calls.append({"role": role, **kw, "messages": list(kw.get("messages", []))})
        forwarded = {k: v for k, v in kw.items() if k not in STREAM_ONLY_KWARGS}
        response = await call()(**forwarded)
        yield StreamComplete(response=response)
