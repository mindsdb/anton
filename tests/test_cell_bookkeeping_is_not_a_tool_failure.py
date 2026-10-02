"""A cell that ran must not be reported as a failed tool call because a step
after it raised.

Explainability bookkeeping runs over a cell's code and output once the cell
has finished. An exception from it used to escape into the tool dispatch and
reach the model as "Tool 'scratchpad' failed: <exc>", so the agent debugged a
tool that had worked. These drive a real LocalScratchpadRuntime through
`turn_stream`, the path desktop and web run.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.conftest import make_mock_llm

from anton.core.backends.local import LocalScratchpadRuntime
from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage
from anton.core.session import ChatSession, ChatSessionConfig, _VerifierVerdict

_DEFAULTS = dict(
    coding_provider="anthropic",
    coding_model="",
    coding_api_key="",
    coding_base_url="",
)

# Scraping a tunnel hostname: the regex source looks like a URL whose netloc
# has an unmatched `[`.
_TUNNEL_SCRAPE = (
    "import re\n"
    "output = 'your url is https://abc-def.trycloudflare.com ok'\n"
    "m = re.search(r'https://[a-z0-9-]+\\.trycloudflare\\.com', output)\n"
    "print('tunnel', m.group(0))\n"
)


def _exec_response(code: str) -> LLMResponse:
    return LLMResponse(
        content="running",
        tool_calls=[ToolCall(id="tc_cell", name="scratchpad",
                             input={"action": "exec", "name": "main", "code": code})],
        usage=Usage(input_tokens=1, output_tokens=1),
        stop_reason="tool_use",
    )


async def _run_turn(tmp_path, code: str):
    workspace = MagicMock(base=tmp_path)
    workspace.artifacts_dir = tmp_path / "artifacts"
    llm = make_mock_llm()
    llm.generate_object_code = AsyncMock(
        return_value=_VerifierVerdict(status="COMPLETE", reason="done")
    )
    responses = iter([_exec_response(code)])
    seen_by_model: list[str] = []

    def plan_stream(**kwargs):
        seen_by_model.append(str(kwargs.get("messages")))

        async def gen():
            yield StreamComplete(response=next(responses, LLMResponse(
                content="done", tool_calls=[], usage=Usage(input_tokens=1, output_tokens=1),
                stop_reason="end_turn",
            )))
        return gen()

    llm.plan_stream = plan_stream
    session = ChatSession(ChatSessionConfig(llm_client=llm, workspace=workspace))
    pad = LocalScratchpadRuntime(name="main", **_DEFAULTS)
    await pad.start()
    session._scratchpads.get_or_create = AsyncMock(return_value=pad)
    try:
        with patch("anton.analytics.send_event") as sent:
            async for _ in session.turn_stream("open the tunnel"):
                pass
    finally:
        await pad.close()
        await session.close()

    rows = [c.kwargs for c in sent.call_args_list if c.args[1] == "tool_completed"]
    return rows, seen_by_model[1] if len(seen_by_model) > 1 else ""


async def test_a_url_shaped_regex_in_the_cell_is_reported_as_success(tmp_path):
    rows, next_prompt = await _run_turn(tmp_path, _TUNNEL_SCRAPE)

    assert len(rows) == 1
    assert rows[0]["ok"] == "true"
    assert "tunnel https://abc-def.trycloudflare.com" in next_prompt
    assert "Tool 'scratchpad' failed" not in next_prompt
    assert "Invalid IPv6 URL" not in next_prompt


@pytest.mark.parametrize("method", ["add_sources_from_text", "add_scratchpad_step"])
async def test_any_bookkeeping_exception_leaves_the_cell_successful(tmp_path, method):
    with patch(
        f"anton.explainability.ExplainabilityCollector.{method}",
        side_effect=RuntimeError("bookkeeping broke"),
    ):
        rows, next_prompt = await _run_turn(tmp_path, "print('hello from the cell')")

    assert rows[0]["ok"] == "true"
    assert "hello from the cell" in next_prompt
    assert "bookkeeping broke" not in next_prompt
