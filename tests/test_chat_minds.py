"""Chat integration tests for the minds tool."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.chat import MINDS_TOOL, ChatSession
from anton.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )


def _minds_response(
    text: str, action: str, tool_id: str = "tc_minds_1", **kwargs
) -> LLMResponse:
    tc_input: dict = {"action": action, **kwargs}
    return LLMResponse(
        content=text,
        tool_calls=[
            ToolCall(id=tool_id, name="minds", input=tc_input),
        ],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="tool_use",
    )


class TestMindsToolDefinition:
    def test_tool_name(self):
        assert MINDS_TOOL["name"] == "minds"

    def test_tool_has_four_actions(self):
        actions = MINDS_TOOL["input_schema"]["properties"]["action"]["enum"]
        assert actions == ["ask", "data", "export", "catalog"]

    def test_tool_properties(self):
        props = MINDS_TOOL["input_schema"]["properties"]
        assert "action" in props
        assert "mind" in props
        assert "question" in props
        assert "datasource" in props
        assert "limit" in props
        assert "offset" in props

    def test_only_action_required(self):
        assert MINDS_TOOL["input_schema"]["required"] == ["action"]


class TestMindsToolInBuildTools:
    async def test_minds_tool_appears_when_configured(self):
        """minds tool should appear in _build_tools() when api key is set."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))
        mock_run = AsyncMock()

        session = ChatSession(
            mock_llm, mock_run, minds_api_key="test-key"
        )
        try:
            await session.turn("hello")
            call_kwargs = mock_llm.plan.call_args
            tools = call_kwargs.kwargs.get("tools", [])
            tool_names = [t["name"] for t in tools]
            assert "minds" in tool_names
        finally:
            await session.close()

    async def test_minds_tool_absent_when_not_configured(self):
        """minds tool should NOT appear when no api key is set."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        try:
            await session.turn("hello")
            call_kwargs = mock_llm.plan.call_args
            tools = call_kwargs.kwargs.get("tools", [])
            tool_names = [t["name"] for t in tools]
            assert "minds" not in tool_names
        finally:
            await session.close()

    async def test_minds_tool_shows_default_mind(self):
        """minds tool description should mention default mind when set."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))
        mock_run = AsyncMock()

        with patch.dict(os.environ, {"MINDS_DEFAULT_MIND": "my_sales_mind"}):
            session = ChatSession(
                mock_llm, mock_run, minds_api_key="test-key"
            )
            try:
                await session.turn("hello")
                call_kwargs = mock_llm.plan.call_args
                tools = call_kwargs.kwargs.get("tools", [])
                minds_tools = [t for t in tools if t["name"] == "minds"]
                assert len(minds_tools) == 1
                assert "my_sales_mind" in minds_tools[0]["description"]
            finally:
                await session.close()


class TestMindsAskDispatch:
    async def test_ask_action_dispatches(self):
        """ask action calls MindsClient.ask() and returns result."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _minds_response(
                    "Let me check.",
                    "ask",
                    mind="sales",
                    question="top customers?",
                ),
                _text_response("The top customer is Acme."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(
            mock_llm, mock_run, minds_api_key="test-key"
        )
        try:
            with patch.object(
                session._minds, "ask", new_callable=AsyncMock, return_value="Acme Corp is #1."
            ) as mock_ask:
                reply = await session.turn("who are our top customers?")

            mock_ask.assert_awaited_once_with("top customers?", "sales")

            # Check tool result in history
            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            assert len(tool_result_msgs) == 1
            result_content = tool_result_msgs[0]["content"][0]["content"]
            assert "Acme Corp" in result_content
        finally:
            await session.close()

    async def test_ask_without_question_returns_error(self):
        """ask action without question returns an error message."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _minds_response("Checking.", "ask", mind="sales"),
                _text_response("I need a question."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(
            mock_llm, mock_run, minds_api_key="test-key"
        )
        try:
            await session.turn("query minds")

            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            result_content = tool_result_msgs[0]["content"][0]["content"]
            assert "question" in result_content.lower()
        finally:
            await session.close()

    async def test_ask_falls_back_to_default_mind(self):
        """ask action uses MINDS_DEFAULT_MIND when mind not specified."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _minds_response("Checking.", "ask", question="revenue?"),
                _text_response("Here's the revenue."),
            ]
        )
        mock_run = AsyncMock()

        with patch.dict(os.environ, {"MINDS_DEFAULT_MIND": "default_mind"}):
            session = ChatSession(
                mock_llm, mock_run, minds_api_key="test-key"
            )
            try:
                with patch.object(
                    session._minds, "ask", new_callable=AsyncMock, return_value="$1M"
                ) as mock_ask:
                    await session.turn("show revenue")

                mock_ask.assert_awaited_once_with("revenue?", "default_mind")
            finally:
                await session.close()


class TestMindsDataDispatch:
    async def test_data_action_dispatches(self):
        """data action calls MindsClient.data() and returns result."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _minds_response("Fetching data.", "data", limit=50),
                _text_response("Here's the table."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(
            mock_llm, mock_run, minds_api_key="test-key"
        )
        try:
            table_md = "| name | revenue |\n| --- | --- |\n| Acme | 1M |"
            with patch.object(
                session._minds, "data", new_callable=AsyncMock, return_value=table_md
            ) as mock_data:
                await session.turn("get the data")

            mock_data.assert_awaited_once_with(limit=50, offset=0)

            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            result_content = tool_result_msgs[0]["content"][0]["content"]
            assert "Acme" in result_content
        finally:
            await session.close()

    async def test_data_without_prior_ask_returns_error(self):
        """data action returns error when no prior ask has been made."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _minds_response("Fetching.", "data"),
                _text_response("Need to ask first."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(
            mock_llm, mock_run, minds_api_key="test-key"
        )
        try:
            # _minds.data() will raise ValueError since no prior ask
            await session.turn("get data")

            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            result_content = tool_result_msgs[0]["content"][0]["content"]
            assert "No prior ask" in result_content
        finally:
            await session.close()


class TestMindsExportDispatch:
    async def test_export_action_dispatches(self):
        """export action calls MindsClient.export() and returns CSV."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _minds_response("Exporting.", "export"),
                _text_response("Here's the CSV data."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(
            mock_llm, mock_run, minds_api_key="test-key"
        )
        try:
            csv_text = "name,revenue\nAcme,1000000\nGlobex,750000\n"
            with patch.object(
                session._minds, "export", new_callable=AsyncMock, return_value=csv_text
            ) as mock_export:
                await session.turn("export the data")

            mock_export.assert_awaited_once()

            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            result_content = tool_result_msgs[0]["content"][0]["content"]
            assert "name,revenue" in result_content
            assert "Acme" in result_content
        finally:
            await session.close()

    async def test_export_without_prior_ask_returns_error(self):
        """export action returns error when no prior ask has been made."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _minds_response("Exporting.", "export"),
                _text_response("Need to ask first."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(
            mock_llm, mock_run, minds_api_key="test-key"
        )
        try:
            # _minds.export() will raise ValueError since no prior ask
            await session.turn("export data")

            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            result_content = tool_result_msgs[0]["content"][0]["content"]
            assert "No prior ask" in result_content
        finally:
            await session.close()


class TestMindsCatalogDispatch:
    async def test_catalog_action_dispatches(self):
        """catalog action calls MindsClient.catalog() and returns result."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _minds_response("Discovering.", "catalog", datasource="my_db"),
                _text_response("Here are the tables."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(
            mock_llm, mock_run, minds_api_key="test-key"
        )
        try:
            catalog_text = "## customers\n  - id (integer)\n  - name (varchar)"
            with patch.object(
                session._minds, "catalog", new_callable=AsyncMock, return_value=catalog_text
            ) as mock_catalog:
                await session.turn("show tables")

            mock_catalog.assert_awaited_once_with("my_db")
        finally:
            await session.close()

    async def test_catalog_without_datasource_returns_error(self):
        """catalog action without datasource returns an error."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _minds_response("Looking.", "catalog"),
                _text_response("I need a datasource name."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(
            mock_llm, mock_run, minds_api_key="test-key"
        )
        try:
            await session.turn("show catalog")

            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            result_content = tool_result_msgs[0]["content"][0]["content"]
            assert "datasource" in result_content.lower()
        finally:
            await session.close()


class TestMindsNotConfigured:
    async def test_minds_tool_call_without_config_returns_error(self):
        """If minds is called but not configured, return helpful error."""
        mock_llm = AsyncMock()
        # Simulate a situation where the tool is somehow called without config
        mock_llm.plan = AsyncMock(
            side_effect=[
                _minds_response("Let me check.", "ask", question="test", mind="m"),
                _text_response("Minds is not set up."),
            ]
        )
        mock_run = AsyncMock()

        # No minds_api_key — _minds is None
        session = ChatSession(mock_llm, mock_run)
        try:
            await session.turn("query data")

            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            result_content = tool_result_msgs[0]["content"][0]["content"]
            assert "not configured" in result_content.lower()
        finally:
            await session.close()


class _FakeAsyncIter:
    """Wraps items into an async iterator for mocking plan_stream."""

    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class TestMindsStreaming:
    async def test_minds_ask_in_streaming_path(self):
        """minds ask works through turn_stream()."""
        mock_llm = AsyncMock()

        call_count = 0

        def fake_plan_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _FakeAsyncIter([
                    StreamComplete(
                        response=_minds_response(
                            "Asking.", "ask", mind="sales", question="top customers?"
                        )
                    )
                ])
            return _FakeAsyncIter([
                StreamComplete(response=_text_response("Acme is #1."))
            ])

        mock_llm.plan_stream = fake_plan_stream
        mock_run = AsyncMock()

        session = ChatSession(
            mock_llm, mock_run, minds_api_key="test-key"
        )
        try:
            with patch.object(
                session._minds, "ask", new_callable=AsyncMock, return_value="Acme Corp is #1."
            ):
                events = []
                async for event in session.turn_stream("who are top customers?"):
                    events.append(event)

            assert any(isinstance(e, StreamComplete) for e in events)

            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            assert len(tool_result_msgs) == 1
            result_content = tool_result_msgs[0]["content"][0]["content"]
            assert "Acme Corp" in result_content
        finally:
            await session.close()


class TestMindsKnowledgeInjection:
    async def test_connected_mind_appears_in_system_prompt(self, tmp_path):
        """When .anton/minds/X.md exists, its content is injected into the system prompt."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))
        mock_run = AsyncMock()

        workspace = MagicMock()
        workspace.base = tmp_path
        workspace.build_anton_md_context.return_value = ""

        minds_dir = tmp_path / ".anton" / "minds"
        minds_dir.mkdir(parents=True)
        (minds_dir / "sales.md").write_text("This mind has sales data with customers and orders.")

        session = ChatSession(
            mock_llm, mock_run,
            workspace=workspace,
            minds_api_key="test-key",
        )
        try:
            await session.turn("hello")
            call_kwargs = mock_llm.plan.call_args
            system = call_kwargs.kwargs.get("system", "")
            assert "## Connected Minds" in system
            assert "### sales" in system
            assert "sales data with customers and orders" in system
        finally:
            await session.close()

    async def test_no_minds_dir_means_no_injection(self):
        """When there's no .anton/minds directory, no mind sections are injected."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run, minds_api_key="test-key")
        try:
            await session.turn("hello")
            call_kwargs = mock_llm.plan.call_args
            system = call_kwargs.kwargs.get("system", "")
            # The "## Connected Minds" section header should NOT appear
            assert "## Connected Minds" not in system
        finally:
            await session.close()

    async def test_empty_minds_dir_means_no_injection(self, tmp_path):
        """When .anton/minds exists but is empty, nothing is injected."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))
        mock_run = AsyncMock()

        workspace = MagicMock()
        workspace.base = tmp_path
        workspace.build_anton_md_context.return_value = ""

        (tmp_path / ".anton" / "minds").mkdir(parents=True)

        session = ChatSession(
            mock_llm, mock_run,
            workspace=workspace,
            minds_api_key="test-key",
        )
        try:
            await session.turn("hello")
            call_kwargs = mock_llm.plan.call_args
            system = call_kwargs.kwargs.get("system", "")
            assert "## Connected Minds" not in system
        finally:
            await session.close()


class TestMindsConnect:
    async def test_connect_writes_llm_summary(self, tmp_path):
        """connect fetches mind info, catalogs datasources, gets LLM summary, writes file."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("This mind provides sales analytics."))
        mock_run = AsyncMock()

        workspace = MagicMock()
        workspace.base = tmp_path
        workspace.build_anton_md_context.return_value = ""

        session = ChatSession(
            mock_llm, mock_run,
            workspace=workspace,
            minds_api_key="test-key",
        )
        console = MagicMock()

        with patch.object(
            session._minds, "get_mind", new_callable=AsyncMock,
            return_value={"name": "sales", "datasources": ["sales_db"]},
        ), patch.object(
            session._minds, "catalog", new_callable=AsyncMock,
            return_value="## customers\n  - id (integer)\n  - name (varchar)",
        ):
            await session._handle_minds_connect("sales", console)

        md_file = tmp_path / ".anton" / "minds" / "sales.md"
        assert md_file.exists()
        content = md_file.read_text()
        assert "sales analytics" in content

        await session.close()

    async def test_connect_fallback_on_llm_failure(self, tmp_path):
        """When LLM fails, raw catalog is stored as fallback."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(side_effect=Exception("LLM unavailable"))
        mock_run = AsyncMock()

        workspace = MagicMock()
        workspace.base = tmp_path
        workspace.build_anton_md_context.return_value = ""

        session = ChatSession(
            mock_llm, mock_run,
            workspace=workspace,
            minds_api_key="test-key",
        )
        console = MagicMock()

        with patch.object(
            session._minds, "get_mind", new_callable=AsyncMock,
            return_value={"name": "sales", "datasources": ["db1"]},
        ), patch.object(
            session._minds, "catalog", new_callable=AsyncMock,
            return_value="## orders\n  - id (integer)",
        ):
            await session._handle_minds_connect("sales", console)

        md_file = tmp_path / ".anton" / "minds" / "sales.md"
        assert md_file.exists()
        content = md_file.read_text()
        assert "orders" in content

        await session.close()

    async def test_connect_uses_table_list_from_get_mind_when_catalog_fails(self, tmp_path):
        """When catalog 404s but get_mind has tables, use those."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("This mind has order and car data."))
        mock_run = AsyncMock()

        workspace = MagicMock()
        workspace.base = tmp_path
        workspace.build_anton_md_context.return_value = ""

        session = ChatSession(
            mock_llm, mock_run,
            workspace=workspace,
            minds_api_key="test-key",
        )
        console = MagicMock()

        import httpx
        catalog_error = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock(status_code=404),
        )

        # get_mind returns datasource dict with tables (like the real API)
        with patch.object(
            session._minds, "get_mind", new_callable=AsyncMock,
            return_value={
                "name": "test",
                "datasources": [
                    {"name": "demo_db", "tables": ["orders", "used_car_price", "customers"]},
                ],
            },
        ), patch.object(
            session._minds, "catalog", new_callable=AsyncMock,
            side_effect=catalog_error,
        ):
            await session._handle_minds_connect("test", console)

        # Verify file was written — LLM got the table list
        md_file = tmp_path / ".anton" / "minds" / "test.md"
        assert md_file.exists()
        content = md_file.read_text()
        assert "order" in content.lower()

        # The LLM prompt should have received the table names
        plan_call = mock_llm.plan.call_args
        prompt_msg = plan_call.kwargs["messages"][0]["content"]
        assert "orders" in prompt_msg
        assert "used_car_price" in prompt_msg

        await session.close()

    async def test_connect_falls_back_to_ask_when_no_tables_in_metadata(self, tmp_path):
        """When catalog 404s AND get_mind has no tables, ask the mind directly."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("This mind has order tracking data."))
        mock_run = AsyncMock()

        workspace = MagicMock()
        workspace.base = tmp_path
        workspace.build_anton_md_context.return_value = ""

        session = ChatSession(
            mock_llm, mock_run,
            workspace=workspace,
            minds_api_key="test-key",
        )
        console = MagicMock()

        import httpx
        catalog_error = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock(status_code=404),
        )

        # Datasource is a plain string (no tables metadata)
        with patch.object(
            session._minds, "get_mind", new_callable=AsyncMock,
            return_value={"name": "test", "datasources": ["demo_db"]},
        ), patch.object(
            session._minds, "catalog", new_callable=AsyncMock,
            side_effect=catalog_error,
        ), patch.object(
            session._minds, "ask", new_callable=AsyncMock,
            return_value="I have access to: orders (id, customer_id, total)",
        ) as mock_ask:
            await session._handle_minds_connect("test", console)

        # Verify it fell back to asking the mind
        mock_ask.assert_awaited_once()

        md_file = tmp_path / ".anton" / "minds" / "test.md"
        assert md_file.exists()

        await session.close()

    async def test_connect_rejects_invalid_name(self, tmp_path):
        """Invalid mind names are rejected."""
        mock_llm = AsyncMock()
        mock_run = AsyncMock()

        workspace = MagicMock()
        workspace.base = tmp_path

        session = ChatSession(
            mock_llm, mock_run,
            workspace=workspace,
            minds_api_key="test-key",
        )
        console = MagicMock()

        await session._handle_minds_connect("bad/name", console)
        console.print.assert_called()
        assert not (tmp_path / ".anton" / "minds").exists()

        await session.close()

    async def test_connect_without_minds_configured(self, tmp_path):
        """Connect fails gracefully when minds not configured."""
        mock_llm = AsyncMock()
        mock_run = AsyncMock()

        workspace = MagicMock()
        workspace.base = tmp_path

        session = ChatSession(mock_llm, mock_run, workspace=workspace)
        console = MagicMock()

        await session._handle_minds_connect("sales", console)
        call_args = [str(c) for c in console.print.call_args_list]
        assert any("not configured" in s.lower() or "setup" in s.lower() for s in call_args)

        await session.close()


class TestMindsDisconnect:
    def test_disconnect_removes_file(self, tmp_path):
        """Disconnect removes the mind's knowledge file."""
        mock_llm = AsyncMock()
        mock_run = AsyncMock()

        workspace = MagicMock()
        workspace.base = tmp_path

        minds_dir = tmp_path / ".anton" / "minds"
        minds_dir.mkdir(parents=True)
        md_file = minds_dir / "sales.md"
        md_file.write_text("Sales mind knowledge.")

        session = ChatSession(
            mock_llm, mock_run,
            workspace=workspace,
            minds_api_key="test-key",
        )
        console = MagicMock()

        session._handle_minds_disconnect("sales", console)
        assert not md_file.exists()

    def test_disconnect_warns_when_not_connected(self, tmp_path):
        """Disconnect warns when the mind isn't connected."""
        mock_llm = AsyncMock()
        mock_run = AsyncMock()

        workspace = MagicMock()
        workspace.base = tmp_path

        (tmp_path / ".anton" / "minds").mkdir(parents=True)

        session = ChatSession(
            mock_llm, mock_run,
            workspace=workspace,
            minds_api_key="test-key",
        )
        console = MagicMock()

        session._handle_minds_disconnect("nonexistent", console)
        call_args = [str(c) for c in console.print.call_args_list]
        assert any("not connected" in s.lower() for s in call_args)
