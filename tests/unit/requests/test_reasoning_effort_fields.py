"""Tests for the reasoning-effort request fields on both inbound schemas."""

from minds.requests.chat_completions_request import ChatCompletionsRequest
from minds.requests.responses_request import ResponsesRequest

_MESSAGES = [{"role": "user", "content": "hi"}]


class TestChatCompletionsReasoningEffort:
    def test_parses_reasoning_effort(self):
        req = ChatCompletionsRequest(model="latest:opus", messages=_MESSAGES, reasoning_effort="xhigh")
        assert req.reasoning_effort == "xhigh"

    def test_defaults_to_none(self):
        req = ChatCompletionsRequest(model="latest:opus", messages=_MESSAGES)
        assert req.reasoning_effort is None


class TestResponsesReasoningParam:
    def test_parses_reasoning_effort(self):
        req = ResponsesRequest(model="latest:gpt", input="hi", reasoning={"effort": "high"})
        assert req.reasoning.effort == "high"

    def test_defaults_to_none(self):
        req = ResponsesRequest(model="latest:gpt", input="hi")
        assert req.reasoning is None

    def test_ignores_other_reasoning_subfields(self):
        # OpenAI's Responses reasoning object has other keys (e.g. summary);
        # they must not break parsing.
        req = ResponsesRequest(model="latest:gpt", input="hi", reasoning={"effort": "low", "summary": "auto"})
        assert req.reasoning.effort == "low"
