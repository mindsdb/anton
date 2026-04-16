from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .prompts import (
    BASE_VISUALIZATIONS_PROMPT,
    CHAT_SYSTEM_PROMPT,
    VISUALIZATIONS_MARKDOWN_OUTPUT_FORMAT_PROMPT,
    VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT,
)

if TYPE_CHECKING:
    from anton.core.memory.skills import SkillStore
    from anton.core.tools.tool_defs import ToolDef


@dataclass(frozen=True)
class SystemPromptContext:
    """Bundled prompt-injection points for the system prompt.

    Four levels with increasing importance (later = stronger influence):
      1. ``prefix``  — prepended before the base prompt
      2. ``runtime_context`` — interpolated into the RUNTIME IDENTITY section
      3. ``output_context`` — free-text instructions on where to
         store generated resources (visualizations, HTML files, data exports)
      4. ``suffix``  — appended after all other sections
    """

    runtime_context: str = ""
    prefix: str = ""
    suffix: str = ""
    output_context: str = ""


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
        self, skill_store: "SkillStore | None"
    ) -> str:
        """Build the '## Procedural memory' section listing available skills.

        Lists each skill as `- label: when_to_use` (one line) plus a short
        instruction telling the LLM to call `recall_skill(label)` to load
        the full procedure. Returns an empty string if no store is wired
        or no skills are saved — the caller skips the section entirely.
        """
        if skill_store is None:
            return ""
        try:
            summaries = skill_store.list_summaries()
        except Exception:
            return ""
        if not summaries:
            return ""

        lines: list[str] = [
            "",
            "",
            "## Procedural memory (skills available)",
            "",
            (
                "These are reusable procedures you've previously refined for "
                "recurring tasks. When the user's request matches one of "
                "them, call `recall_skill(label)` to load the full step-by-"
                "step procedure into your context. You may recall multiple "
                "skills if the task spans several. If none apply, proceed "
                "with normal reasoning."
            ),
            "",
        ]
        for s in summaries:
            label = s.get("label", "")
            when = s.get("when_to_use", "").strip()
            if not label:
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
        output_context: str,
    ) -> str:
        visualizations_output_format_prompt = (
            VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT
            if proactive_dashboards
            else VISUALIZATIONS_MARKDOWN_OUTPUT_FORMAT_PROMPT
        )
        output_format = visualizations_output_format_prompt.format(
            output_context=output_context,
        )
        return BASE_VISUALIZATIONS_PROMPT.format(output_format=output_format)

    def build(
        self,
        *,
        current_datetime: str,
        system_prompt_context: SystemPromptContext,
        proactive_dashboards: bool,
        tool_defs: list["ToolDef"] | None = None,
        memory_context: str = "",
        project_context: str = "",
        self_awareness_context: str = "",
        datasource_context: str = "",
        skill_store: "SkillStore | None" = None,
    ) -> str:
        visualizations_section = self._build_visualizations_section(
            proactive_dashboards=proactive_dashboards,
            output_context=system_prompt_context.output_context,
        )

        prompt = ""

        prefix = system_prompt_context.prefix.strip()
        if prefix:
            prompt += f"{prefix}\n\n"

        prompt += CHAT_SYSTEM_PROMPT.format(
            runtime_context=system_prompt_context.runtime_context,
            visualizations_section=visualizations_section,
            current_datetime=current_datetime,
        )

        tool_prompts = self._build_tool_prompts_section(tool_defs)
        if tool_prompts:
            prompt += tool_prompts

        if memory_context:
            prompt += memory_context
        if project_context:
            prompt += project_context
        if self_awareness_context:
            prompt += self_awareness_context
        if datasource_context:
            prompt += datasource_context

        procedural_memory = self._build_procedural_memory_section(skill_store)
        if procedural_memory:
            prompt += procedural_memory

        suffix = system_prompt_context.suffix.strip()
        if suffix:
            prompt += f"\n\n{suffix}"

        return prompt


__all__ = ["ChatSystemPromptBuilder", "SystemPromptContext"]
