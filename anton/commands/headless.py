"""Headless single-shot prompt execution."""
from __future__ import annotations

import asyncio
import json as _json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from anton.chat_session import build_runtime_context
from anton.data_vault import DataVault
from anton.datasource_registry import DatasourceRegistry
from anton.llm.provider import (
    StreamComplete,
    StreamTextDelta,
    StreamToolUseDelta,
    StreamToolUseEnd,
    StreamToolUseStart,
)
from anton.utils.datasources import register_secret_vars

if TYPE_CHECKING:
    from rich.console import Console

    from anton.config.settings import AntonSettings


def run_headless(
    console: Console, settings: AntonSettings, *, prompt: str, output_format: str = "text"
) -> None:
    """Run a single prompt in headless mode and exit."""
    if not prompt:
        print("Error: headless mode requires a prompt via --prompt or --stdin", file=sys.stderr)
        raise SystemExit(1)

    asyncio.run(_headless(console, settings, prompt=prompt, output_format=output_format))


async def _headless(
    console: Console, settings: AntonSettings, *, prompt: str, output_format: str = "text"
) -> None:
    """Execute a single prompt without interactive elements."""
    try:
        from anton.context.self_awareness import SelfAwarenessContext
        from anton.llm.client import LLMClient
        from anton.memory.cortex import Cortex
        from anton.workspace import Workspace

        llm_client = LLMClient.from_settings(settings)

        self_awareness = SelfAwarenessContext(Path(settings.context_dir))
        workspace = Workspace(settings.workspace_path)
        workspace.apply_env_to_process()

        # Inject datasource env vars
        dv = DataVault()
        dreg = DatasourceRegistry()
        for conn in dv.list_connections():
            dv.inject_env(conn["engine"], conn["name"])
            edef = dreg.get(conn["engine"])
            if edef is not None:
                register_secret_vars(edef, engine=conn["engine"], name=conn["name"])
        del dv, dreg

        global_memory_dir = Path.home() / ".anton" / "memory"
        project_memory_dir = settings.workspace_path / ".anton" / "memory"

        cortex = Cortex(
            global_dir=global_memory_dir,
            project_dir=project_memory_dir,
            mode=settings.memory_mode,
            llm_client=llm_client,
        )

        from anton.memory.episodes import EpisodicMemory

        episodes_dir = settings.workspace_path / ".anton" / "episodes"
        episodic = EpisodicMemory(episodes_dir, enabled=settings.episodic_memory)
        if episodic.enabled:
            episodic.start_session()

        from anton.memory.history_store import HistoryStore

        history_store = HistoryStore(episodes_dir)
        current_session_id = episodic._session_id if episodic.enabled else None

        from anton.chat import ChatSession

        runtime_context = build_runtime_context(settings)
        coding_api_key = (
            settings.anthropic_api_key
            if settings.coding_provider == "anthropic"
            else settings.openai_api_key
        ) or ""

        session = ChatSession(
            llm_client,
            self_awareness=self_awareness,
            cortex=cortex,
            episodic=episodic,
            runtime_context=runtime_context,
            workspace=workspace,
            console=None,
            coding_provider=settings.coding_provider,
            coding_api_key=coding_api_key,
            coding_base_url=settings.openai_base_url or "",
            history_store=history_store,
            session_id=current_session_id,
            proactive_dashboards=False,
        )

        # Execute single turn
        response_text = ""
        tool_calls: list[dict] = []
        usage_data: dict = {}

        async for event in session.turn_stream(prompt):
            if isinstance(event, StreamTextDelta):
                response_text += event.text
            elif isinstance(event, StreamToolUseStart):
                tool_calls.append({"name": event.name, "id": event.id, "input": {}})
            elif isinstance(event, StreamToolUseDelta):
                if tool_calls:
                    last = tool_calls[-1]
                    last.setdefault("_raw_input", "")
                    last["_raw_input"] += event.json_delta
            elif isinstance(event, StreamToolUseEnd):
                if tool_calls:
                    last = tool_calls[-1]
                    raw = last.pop("_raw_input", "{}")
                    try:
                        last["input"] = _json.loads(raw)
                    except _json.JSONDecodeError:
                        last["input"] = raw
            elif isinstance(event, StreamComplete):
                usage_data = {
                    "input_tokens": event.response.usage.input_tokens,
                    "output_tokens": event.response.usage.output_tokens,
                }

        # Output
        if output_format == "json":
            result = {
                "response": response_text,
                "tool_calls": [{"name": tc["name"], "input": tc.get("input", {})} for tc in tool_calls],
                "usage": usage_data,
            }
            print(_json.dumps(result))
        else:
            print(response_text)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
