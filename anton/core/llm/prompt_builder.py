from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .identity import DeploymentIdentity, build_runtime_identity_section
from .prompts import (
    ARTIFACTS_PROMPT,
    BASE_VISUALIZATIONS_PROMPT,
    BACKEND_GENERATION_PROMPT,
    CHAT_SYSTEM_PROMPT,
    CONVERSATION_DISCIPLINE_ACT_FIRST,
    CONVERSATION_DISCIPLINE_ASK_FIRST,
    NO_DIRECT_REPORTS_ARTIFACTS_PROMPT,
    NO_DIRECT_REPORTS_STATIC_ARTIFACTS_PROMPT,
    NO_DIRECT_REPORTS_VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT,
    NO_DIRECT_REPORTS_VISUALIZATIONS_MARKDOWN_OUTPUT_FORMAT_PROMPT,
    NO_WEB_CHAT_SYSTEM_PROMPT,
    NO_WEB_CONVERSATION_DISCIPLINE_ACT_FIRST,
    STATIC_ARTIFACTS_PROMPT,
    VISUALIZATIONS_MARKDOWN_OUTPUT_FORMAT_PROMPT,
    VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT,
)

if TYPE_CHECKING:
    from anton.core.memory.skills import SkillStore
    from anton.core.tools.tool_defs import ToolDef


# Boundary between the prompt every session shares and the per-project /
# per-conversation part. The OpenAI Responses transport sends the shared part as
# its own message with an explicit cache breakpoint, so it is cached across
# projects and conversations; other transports send the prompt as one string.
SESSION_CONTEXT_MARKER = "\n\n# SESSION CONTEXT (this project and conversation)\n"

# The build-html-dashboard skill's pointer to the DIRECT REPORTS path. Matched
# rather than cut with `cut_spans`: a skill description is data a project can
# override, and a missing note must not break the prompt.
DIRECT_REPORTS_SKILL_NOTE = re.compile(
    r"\s*NOT needed to change the content of a report.*?DIRECT REPORTS\)\.", re.DOTALL
)


@dataclass(frozen=True)
class SystemPromptContext:
    """Bundled prompt-injection points for the system prompt.

    Three levels with increasing importance (later = stronger influence):
      1. ``prefix``  — prepended before the base prompt
      2. ``runtime_context`` — the CONFIGURED LLM block of the RUNTIME IDENTITY
         section (provider + model ids for code the agent writes). The
         *serving* model is not part of this context: the session derives it
         per turn from what the provider reported (ENG-1638).
      3. ``suffix``  — appended after all other sections
    """

    runtime_context: str = ""
    prefix: str = ""
    suffix: str = ""


class ChatSystemPromptBuilder:
    """
    Build Anton's chat system prompt from core components.
    """

    def _build_tool_prompts_section(self, tool_defs: list["ToolDef"] | None) -> str:
        """Build an optional system-prompt section from `ToolDef.prompt`."""
        if not tool_defs:
            return ""

        chunks: list[str] = []
        for tool in tool_defs:
            prompt = getattr(tool, "prompt", None)

            if not prompt:
                continue

            prompt_text = str(prompt).strip()
            if not prompt_text:
                continue

            chunks.append(prompt_text)

        if not chunks:
            return ""

        return "\n\n".join(chunks)

    def _build_procedural_memory_section(
        self,
        skill_store: "SkillStore | None",
        *,
        direct_generation: bool = False,
        fullstack_offered: bool = True,
        direct_reports: bool = True,
        web_offered: bool = True,
    ) -> str:
        """Build the '## Procedural memory' section listing available skills.

        Lists each skill as `- label: description` (one line) plus a short
        instruction telling the LLM to call `recall_skill(label)` to load
        the full procedure. Returns an empty string if no store is wired
        or no skills are saved — the caller skips the section entirely.
        ``fullstack_offered`` False leaves out `build-fullstack-backend`,
        whose line offers the fullstack types the tools leave out.
        ``web_offered`` False leaves out `public-data-sources`, a catalog of
        web endpoints nothing here can reach.
        """
        if skill_store is None:
            return ""
        try:
            summaries = skill_store.list_summaries()
        except Exception:
            return ""
        if not summaries:
            return ""

        instructions = (
                "These are reusable procedures you've previously refined for "
                "recurring tasks. When the user's request matches one of "
                "them, call `recall_skill(label)` to load the full step-by-"
                "step procedure into your context. You may recall multiple "
                "skills if the task spans several. If none apply, proceed "
                "with normal reasoning."
        )
        if direct_generation:
            instructions = (
                "These are reusable procedures. Call recall_skill(label) when a procedure "
                "matches. For an HTML artifact delegated to generate_artifact, pass what "
                "a recalled source procedure says about access in known_data instead of "
                "querying the same source beforehand."
            )
        lines: list[str] = ["", "", "## Procedural memory (skills available)", "", instructions, ""]
        for s in summaries:
            label = s.get("label", "")
            when = s.get("description", "").strip()
            if not direct_reports:
                when = DIRECT_REPORTS_SKILL_NOTE.sub("", when).strip()
            if not label or (not fullstack_offered and label == "build-fullstack-backend"):
                continue
            if not web_offered and label == "public-data-sources":
                continue
            if when:
                lines.append(f"- `{label}` — {when}")
            else:
                lines.append(f"- `{label}`")
        return "\n".join(lines)

    def _build_visualizations_section(
        self,
        *,
        proactive_dashboards: bool,
        output_dir: str,
        direct_generation: bool = False,
        direct_reports: bool = True,
    ) -> str:
        if direct_generation:
            from .static_artifact_prompt import visualizations

            return visualizations(proactive=proactive_dashboards)
        if proactive_dashboards:
            template = (VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT if direct_reports
                        else NO_DIRECT_REPORTS_VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT)
        else:
            template = (VISUALIZATIONS_MARKDOWN_OUTPUT_FORMAT_PROMPT if direct_reports
                        else NO_DIRECT_REPORTS_VISUALIZATIONS_MARKDOWN_OUTPUT_FORMAT_PROMPT)
        output_format = template.format(output_dir=output_dir)
        return BASE_VISUALIZATIONS_PROMPT.format(output_format=output_format)

    def build(
        self,
        *,
        conversation_started: str,
        system_prompt_context: SystemPromptContext,
        proactive_dashboards: bool,
        output_dir: str,
        act_first: bool = True,
        tool_defs: list["ToolDef"] | None = None,
        memory_context: str = "",
        project_context: str = "",
        self_awareness_context: str = "",
        datasource_context: str = "",
        skill_store: "SkillStore | None" = None,
        workspace_context: str = "",
        runtime_identity_lines: list[str] | None = None,
        product_lines: list[str] | None = None,
        deployment_identity: DeploymentIdentity | None = None,
        fullstack_offered: bool = True,
        web_offered: bool = True,
        direct_reports: bool = True,
    ) -> str:
        """``fullstack_offered`` False (`live_app_slots.fullstack_offered`)
        leaves out the fullstack types, the backend section and the backend
        skill's line: nothing on this deployment would start the app's
        backend. ``web_offered`` False (`wire.web_search_off`) swaps the
        internet sections for one that says the web is out of reach."""
        direct_generation = (os.environ.get("ANTON_STATIC_SINGLE_CALL") == "on"
                             and any(getattr(tool, "name", "") == "generate_artifact"
                                     and "name" in tool.input_schema.get("properties", {})
                                     for tool in tool_defs or []))
        visualizations_section = self._build_visualizations_section(
            proactive_dashboards=proactive_dashboards,
            output_dir=output_dir,
            direct_generation=direct_generation,
            direct_reports=direct_reports,
        )
        if fullstack_offered:
            artifacts_section = ARTIFACTS_PROMPT if direct_reports else NO_DIRECT_REPORTS_ARTIFACTS_PROMPT
        else:
            artifacts_section = (STATIC_ARTIFACTS_PROMPT if direct_reports
                                 else NO_DIRECT_REPORTS_STATIC_ARTIFACTS_PROMPT)
        # The single-call path has its own artifact instructions, which have no direct report path
        if direct_generation:
            from .static_artifact_prompt import ARTIFACTS

            artifacts_section = ARTIFACTS

        prompt = ""

        prefix = system_prompt_context.prefix.strip()
        if prefix:
            prompt += f"{prefix}\n\n"

        if not act_first:
            conversation_discipline = CONVERSATION_DISCIPLINE_ASK_FIRST
        elif web_offered:
            conversation_discipline = CONVERSATION_DISCIPLINE_ACT_FIRST
        else:
            conversation_discipline = NO_WEB_CONVERSATION_DISCIPLINE_ACT_FIRST

        # ENG-1638: two questions, two sub-blocks. `runtime_identity_lines` say
        # what is serving THIS conversation (from the provider's response, or the
        # cannot-verify fallback); `runtime_context` says what code the agent
        # writes should call (configuration). Never one block for both.
        runtime_identity_section = build_runtime_identity_section(
            identity_lines=list(runtime_identity_lines or []),
            configured_block=system_prompt_context.runtime_context,
            product_block=list(product_lines) if product_lines else None,
        )
        template = CHAT_SYSTEM_PROMPT if web_offered else NO_WEB_CHAT_SYSTEM_PROMPT
        prompt += template.format(
            # PRODUCT carries the configured name. An explicit deployment must
            # not also introduce itself as Anton in the prompt's opening line.
            identity_lead="You are an AI coworker" if deployment_identity is not None else "You are Anton",
            runtime_identity_section=runtime_identity_section,
            artifacts_section=artifacts_section,
            visualizations_section=visualizations_section,
            conversation_discipline=conversation_discipline,
            conversation_started=conversation_started,
        )

        # Short pointer only — the full backend/fullstack contract lives in the
        # built-in `build-fullstack-backend` skill (recalled on demand).
        if fullstack_offered:
            prompt += "\n\n" + BACKEND_GENERATION_PROMPT

        tool_prompts = self._build_tool_prompts_section(tool_defs)
        if tool_prompts:
            prompt += tool_prompts

        # Static note (cache-stable): the current time is no longer in the
        # system prompt — each message carries its send time as a bracketed
        # prefix, so the model reads "now" from the latest message. Keeping the
        # clock out of the prefix lets it stay byte-identical across turns.
        prompt += (
            "\n\n(Messages are prefixed with the time they were sent; that "
            "bracketed timestamp is metadata, not part of the message text.)"
        )

        # Saved skills change rarely (the built-ins plus the user's own), so
        # they stay in the shared part ahead of anything project-specific.
        procedural_memory = self._build_procedural_memory_section(
            skill_store, direct_generation=direct_generation, fullstack_offered=fullstack_offered,
            direct_reports=direct_reports, web_offered=web_offered,
        )
        if procedural_memory:
            prompt += procedural_memory

        # Everything below depends on the project or the conversation: stable
        # per-session content first, the volatile tail last, so within a
        # session the prefix still stays cache-stable across turns.
        session = ""
        if project_context:
            session += project_context
        if self_awareness_context:
            session += self_awareness_context
        if datasource_context:
            session += datasource_context

        suffix = system_prompt_context.suffix.strip()
        if suffix:
            session += f"\n\n{suffix}"

        # Volatile tail — LAST so everything above can be cached. Only the
        # relevance-filtered memory snapshot remains here; it changes every turn.
        if memory_context:
            session += memory_context
        # Workspace discovery (ENG-578) is volatile too — pads and files
        # change between turns, so it must never sit in the cached prefix.
        if workspace_context:
            session += workspace_context

        if session.strip():
            prompt += SESSION_CONTEXT_MARKER + session.lstrip("\n")
        return prompt


__all__ = ["ChatSystemPromptBuilder", "SystemPromptContext", "SESSION_CONTEXT_MARKER"]
