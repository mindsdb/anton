"""pytest fixtures for the eval harness — creates real LLM clients."""

from __future__ import annotations

import os

import pytest

from anton.llm.anthropic import AnthropicProvider
from anton.llm.client import LLMClient

from evals.runner import EvalRunner


@pytest.fixture(scope="session")
def anthropic_api_key() -> str:
    """Resolve the Anthropic API key from environment."""
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
        "ANTON_ANTHROPIC_API_KEY"
    )
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set — skipping evals")
    return key


@pytest.fixture(scope="session")
def eval_llm_client(anthropic_api_key: str) -> LLMClient:
    """Create a real LLM client wired to Anthropic for eval runs."""
    provider = AnthropicProvider(api_key=anthropic_api_key)
    return LLMClient(
        planning_provider=provider,
        planning_model="claude-sonnet-4-6",
        coding_provider=provider,
        coding_model="claude-haiku-4-5-20251001",
    )


@pytest.fixture(scope="session")
def eval_runner(eval_llm_client: LLMClient, anthropic_api_key: str) -> EvalRunner:
    """Create an EvalRunner with real LLM and scratchpad execution."""
    return EvalRunner(
        eval_llm_client,
        coding_provider="anthropic",
        coding_api_key=anthropic_api_key,
    )
