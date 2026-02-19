from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from anton.core.agent import Agent, _slugify
from anton.core.executor import ExecutionResult, StepResult
from anton.core.planner import Plan, PlanStep
from anton.events.types import Phase, StatusUpdate
from anton.llm.provider import LLMResponse, ToolCall, Usage
from anton.skill.base import SkillResult


class TestAgent:
    async def test_run_happy_path(self, mock_channel, skill_registry):
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            return_value=LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc_1",
                        name="create_plan",
                        input={
                            "reasoning": "Simple task",
                            "steps": [
                                {
                                    "skill_name": "list_files",
                                    "description": "List files",
                                    "parameters": {"pattern": "*"},
                                }
                            ],
                            "estimated_time_seconds": 1.0,
                        },
                    )
                ],
                usage=Usage(),
                stop_reason="tool_use",
            )
        )

        agent = Agent(
            channel=mock_channel,
            llm_client=mock_llm,
            registry=skill_registry,
        )
        # Subscribe directly to the bus to avoid relay timing issues
        test_queue = agent._bus.subscribe()
        await agent.run("list all files")

        events = []
        while not test_queue.empty():
            events.append(test_queue.get_nowait())

        event_types = [type(e).__name__ for e in events]
        assert "StatusUpdate" in event_types
        assert "TaskComplete" in event_types
        mock_llm.plan.assert_awaited_once()

    async def test_run_exception_publishes_task_failed(self, mock_channel, skill_registry):
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(side_effect=RuntimeError("LLM down"))

        agent = Agent(
            channel=mock_channel,
            llm_client=mock_llm,
            registry=skill_registry,
        )
        # Subscribe directly to the bus to avoid relay timing issues
        test_queue = agent._bus.subscribe()
        await agent.run("fail task")

        events = []
        while not test_queue.empty():
            events.append(test_queue.get_nowait())

        event_types = [type(e).__name__ for e in events]
        assert "TaskFailed" in event_types


class TestBuildSummary:
    def test_build_summary_basic(self, mock_channel, skill_registry):
        mock_llm = AsyncMock()
        agent = Agent(
            channel=mock_channel,
            llm_client=mock_llm,
            registry=skill_registry,
        )

        plan = Plan(
            reasoning="Test reasoning",
            steps=[PlanStep(skill_name="read_file", description="read", parameters={})],
        )
        exec_result = ExecutionResult(
            step_results=[
                StepResult(
                    step_index=0,
                    skill_name="read_file",
                    result=SkillResult(output="file content"),
                    duration_seconds=0.5,
                )
            ],
            total_duration_seconds=0.5,
        )

        summary = agent._build_summary(plan, exec_result)
        assert "0.5s" in summary
        assert "Test reasoning" in summary
        assert "read_file" in summary
        assert "file content" in summary

    def test_build_summary_with_error(self, mock_channel, skill_registry):
        mock_llm = AsyncMock()
        agent = Agent(
            channel=mock_channel,
            llm_client=mock_llm,
            registry=skill_registry,
        )

        plan = Plan(reasoning="r", steps=[])
        exec_result = ExecutionResult(
            step_results=[
                StepResult(
                    step_index=0,
                    skill_name="bad_skill",
                    result=SkillResult(output=None, metadata={"error": "Skill not found"}),
                    duration_seconds=0,
                )
            ],
            total_duration_seconds=0.1,
        )

        summary = agent._build_summary(plan, exec_result)
        assert "ERROR" in summary
        assert "Skill not found" in summary

    def test_build_summary_truncates_long_output(self, mock_channel, skill_registry):
        mock_llm = AsyncMock()
        agent = Agent(
            channel=mock_channel,
            llm_client=mock_llm,
            registry=skill_registry,
        )

        plan = Plan(reasoning="r", steps=[])
        long_output = "x" * 300
        exec_result = ExecutionResult(
            step_results=[
                StepResult(
                    step_index=0,
                    skill_name="read_file",
                    result=SkillResult(output=long_output),
                    duration_seconds=0.1,
                )
            ],
            total_duration_seconds=0.1,
        )

        summary = agent._build_summary(plan, exec_result)
        assert "..." in summary


def _plan_response_with_steps(
    steps: list[dict],
    skills_to_create: list[str] | None = None,
) -> LLMResponse:
    """Helper to create a plan LLMResponse with given steps."""
    tool_input: dict = {
        "reasoning": "test",
        "steps": steps,
        "estimated_time_seconds": 1.0,
    }
    if skills_to_create:
        tool_input["skills_to_create"] = skills_to_create
    return LLMResponse(
        content="",
        tool_calls=[
            ToolCall(
                id="tc_1",
                name="create_plan",
                input=tool_input,
            )
        ],
        usage=Usage(),
        stop_reason="tool_use",
    )


VALID_SKILL_CODE = '''\
```python
from anton.skill.base import SkillResult, skill


@skill("count_lines", "Count lines in a file")
async def count_lines(path: str) -> SkillResult:
    with open(path) as f:
        lines = f.readlines()
    return SkillResult(output=len(lines), metadata={"path": path})
```
'''


class TestSkillBuilding:
    async def test_unknown_steps_trigger_building_and_replan(
        self, mock_channel, skill_registry, tmp_path: Path
    ):
        """When plan has unknown steps, agent builds skills and re-plans."""
        # First plan call returns an unknown step
        first_plan = _plan_response_with_steps([
            {
                "skill_name": "unknown",
                "description": "count lines",
                "parameters": {"path": "test.txt"},
            }
        ])
        # Second plan call (re-plan) returns a known step
        second_plan = _plan_response_with_steps([
            {
                "skill_name": "list_files",
                "description": "List files",
                "parameters": {"pattern": "*"},
            }
        ])

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(side_effect=[first_plan, second_plan])
        # code() returns valid skill code for the builder
        mock_llm.code = AsyncMock(
            return_value=LLMResponse(
                content=VALID_SKILL_CODE,
                tool_calls=[],
                usage=Usage(),
                stop_reason="end_turn",
            )
        )

        agent = Agent(
            channel=mock_channel,
            llm_client=mock_llm,
            registry=skill_registry,
            user_skills_dir=tmp_path,
        )
        test_queue = agent._bus.subscribe()
        await agent.run("count lines in a file")

        events = []
        while not test_queue.empty():
            events.append(test_queue.get_nowait())

        # Verify skill building events were published
        building_events = [
            e for e in events
            if isinstance(e, StatusUpdate) and e.phase == Phase.SKILL_BUILDING
        ]
        assert len(building_events) >= 1

        # plan() called twice (initial + re-plan), code() called once
        assert mock_llm.plan.await_count == 2
        mock_llm.code.assert_awaited_once()

        # Task completed successfully
        event_types = [type(e).__name__ for e in events]
        assert "TaskComplete" in event_types

    async def test_no_unknown_steps_skips_building(self, mock_channel, skill_registry):
        """When plan has no unknown steps, code() is never called."""
        plan_response = _plan_response_with_steps([
            {
                "skill_name": "list_files",
                "description": "List files",
                "parameters": {"pattern": "*"},
            }
        ])

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=plan_response)
        mock_llm.code = AsyncMock()

        agent = Agent(
            channel=mock_channel,
            llm_client=mock_llm,
            registry=skill_registry,
        )
        await agent.run("list all files")

        mock_llm.plan.assert_awaited_once()
        mock_llm.code.assert_not_awaited()


    async def test_skills_to_create_triggers_building(
        self, mock_channel, skill_registry, tmp_path: Path
    ):
        """When plan has skills_to_create, agent builds them even without unknown steps."""
        # First plan has skills_to_create but all steps are "unknown"
        first_plan = _plan_response_with_steps(
            steps=[
                {
                    "skill_name": "unknown",
                    "description": "count lines",
                    "parameters": {"path": "test.txt"},
                }
            ],
            skills_to_create=["count lines"],
        )
        # Second plan (re-plan) uses a known built-in skill
        second_plan = _plan_response_with_steps([
            {
                "skill_name": "list_files",
                "description": "List files",
                "parameters": {"pattern": "*"},
            }
        ])

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(side_effect=[first_plan, second_plan])
        mock_llm.code = AsyncMock(
            return_value=LLMResponse(
                content=VALID_SKILL_CODE,
                tool_calls=[],
                usage=Usage(),
                stop_reason="end_turn",
            )
        )

        agent = Agent(
            channel=mock_channel,
            llm_client=mock_llm,
            registry=skill_registry,
            user_skills_dir=tmp_path,
        )
        test_queue = agent._bus.subscribe()
        await agent.run("count lines in a file")

        events = []
        while not test_queue.empty():
            events.append(test_queue.get_nowait())

        # plan() called twice, code() called once (deduped via seen set)
        assert mock_llm.plan.await_count == 2
        mock_llm.code.assert_awaited_once()

        event_types = [type(e).__name__ for e in events]
        assert "TaskComplete" in event_types


class TestEstimatorIntegration:
    async def test_estimator_used_for_eta_after_first_run(self, mock_channel, skill_registry):
        """After first run records durations, second run uses estimator ETA."""
        plan_response = _plan_response_with_steps([
            {
                "skill_name": "list_files",
                "description": "List files",
                "parameters": {"pattern": "*"},
            }
        ])

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=plan_response)

        agent = Agent(
            channel=mock_channel,
            llm_client=mock_llm,
            registry=skill_registry,
        )

        # First run — records durations
        await agent.run("list files")
        assert agent._estimator.estimate("list_files") is not None

        # Second run — estimator provides ETA
        test_queue = agent._bus.subscribe()
        await agent.run("list files again")

        events = []
        while not test_queue.empty():
            events.append(test_queue.get_nowait())

        # Find the "Plan ready" event — should have estimator-based ETA
        plan_ready = [
            e for e in events
            if isinstance(e, StatusUpdate)
            and e.phase == Phase.PLANNING
            and "Plan ready" in e.message
        ]
        assert len(plan_ready) == 1
        assert plan_ready[0].eta_seconds is not None


class TestMemoryIntegration:
    async def test_memory_enabled_run_logs_transcript(self, mock_channel, skill_registry, tmp_path):
        """When memory is enabled, transcript entries are logged."""
        from anton.memory.learnings import LearningStore
        from anton.memory.store import SessionStore

        memory = SessionStore(tmp_path)
        learnings = LearningStore(tmp_path)

        plan_response = _plan_response_with_steps([
            {
                "skill_name": "list_files",
                "description": "List files",
                "parameters": {"pattern": "*"},
            }
        ])

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=plan_response)
        # code() for learning extraction — return empty learnings
        mock_llm.code = AsyncMock(
            return_value=LLMResponse(
                content="[]",
                tool_calls=[],
                usage=Usage(),
                stop_reason="end_turn",
            )
        )

        agent = Agent(
            channel=mock_channel,
            llm_client=mock_llm,
            registry=skill_registry,
            memory=memory,
            learnings=learnings,
        )
        test_queue = agent._bus.subscribe()
        await agent.run("list files")

        events = []
        while not test_queue.empty():
            events.append(test_queue.get_nowait())

        event_types = [type(e).__name__ for e in events]
        assert "TaskComplete" in event_types

        # Verify transcript was written
        sessions = memory.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["status"] == "completed"

        transcript = memory.get_transcript(sessions[0]["id"])
        types = [e["type"] for e in transcript]
        assert "task" in types
        assert "plan" in types
        assert "step" in types
        assert "complete" in types

    async def test_memory_disabled_run_works(self, mock_channel, skill_registry):
        """When memory is None, agent runs without errors (backward-compatible)."""
        plan_response = _plan_response_with_steps([
            {
                "skill_name": "list_files",
                "description": "List files",
                "parameters": {"pattern": "*"},
            }
        ])

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=plan_response)

        agent = Agent(
            channel=mock_channel,
            llm_client=mock_llm,
            registry=skill_registry,
            memory=None,
            learnings=None,
        )
        test_queue = agent._bus.subscribe()
        await agent.run("list files")

        events = []
        while not test_queue.empty():
            events.append(test_queue.get_nowait())

        event_types = [type(e).__name__ for e in events]
        assert "TaskComplete" in event_types
        # No memory recall event since memory is disabled
        memory_recall_events = [
            e for e in events
            if isinstance(e, StatusUpdate) and e.phase == Phase.MEMORY_RECALL
        ]
        assert len(memory_recall_events) == 0

    async def test_learnings_extracted_after_completion(self, mock_channel, skill_registry, tmp_path):
        """After successful completion, learnings are extracted via LLM."""
        from anton.memory.learnings import LearningStore
        from anton.memory.store import SessionStore

        memory = SessionStore(tmp_path)
        learnings = LearningStore(tmp_path)

        plan_response = _plan_response_with_steps([
            {
                "skill_name": "list_files",
                "description": "List files",
                "parameters": {"pattern": "*"},
            }
        ])

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=plan_response)
        # Return a learning from the LLM
        mock_llm.code = AsyncMock(
            return_value=LLMResponse(
                content='[{"topic": "file_listing", "content": "Use list_files for glob patterns", "summary": "list_files supports globs"}]',
                tool_calls=[],
                usage=Usage(),
                stop_reason="end_turn",
            )
        )

        agent = Agent(
            channel=mock_channel,
            llm_client=mock_llm,
            registry=skill_registry,
            memory=memory,
            learnings=learnings,
        )
        await agent.run("list files")

        # Verify learnings were recorded
        all_learnings = learnings.list_all()
        assert len(all_learnings) == 1
        assert all_learnings[0]["topic"] == "file_listing"

    async def test_failed_task_records_failure_in_session(self, mock_channel, skill_registry, tmp_path):
        """When task fails, session is marked as failed."""
        from anton.memory.store import SessionStore

        memory = SessionStore(tmp_path)

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(side_effect=RuntimeError("LLM down"))

        agent = Agent(
            channel=mock_channel,
            llm_client=mock_llm,
            registry=skill_registry,
            memory=memory,
        )
        await agent.run("fail task")

        sessions = memory.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["status"] == "failed"


class TestSlugify:
    def test_simple(self):
        assert _slugify("Count lines in a file") == "count_lines_in_a_file"

    def test_special_chars(self):
        assert _slugify("Read a file's content!") == "read_a_files_content"

    def test_truncates(self):
        result = _slugify("one two three four five six seven eight")
        parts = result.split("_")
        assert len(parts) <= 6
