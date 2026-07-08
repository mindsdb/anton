"""Built-in skills — big prompt tutorials loaded on demand (ENG-648).

Before this module, ~57% of every request's system prompt was two
always-included tutorials (backend/fullstack app generation and the
HTML-dashboard build discipline) plus a public-data-source catalog —
paid on every call of every tool round, even for "what's 2+2".

They now ship as built-in skills: one-line summaries appear in the
'## Procedural memory' section of the system prompt (see
``ChatSystemPromptBuilder._build_procedural_memory_section``) and the
full text loads through the same two paths user skills use:

  • the model calls ``recall_skill(label)`` when it recognizes the task, or
  • the router names the label at delegation time and the session preloads
    it (``ChatSession._inject_recalled_skills``).

Built-ins resolve BEFORE the on-disk ``SkillStore`` in ``recall_skill``,
and their labels are reserved: a user skill with the same label is
shadowed. Content stays in ``prompts.py`` — this module only wraps it
with labels, when-to-use descriptions, and deferred ``str.format``
rendering (the templates carry ``{{ }}``-escaped braces exactly like
they did when the prompt builder formatted them inline).
"""

from __future__ import annotations

from dataclasses import dataclass

from anton.core.llm.prompts import (
    BACKEND_GENERATION_PROMPT,
    PUBLIC_DATA_SOURCES_PROMPT,
    VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT,
)


@dataclass(frozen=True)
class BuiltinSkill:
    label: str
    # When-to-use line, shown in the system prompt's procedural memory
    # section and in the router's skill list.
    description: str
    # Prompt template; rendered with `output_dir` at recall time so the
    # doubled-brace escapes collapse exactly as they did when the prompt
    # builder inlined these sections.
    template: str

    def render(self, *, output_dir: str = ".anton/output") -> str:
        return self.template.format(output_dir=output_dir)

    def format_response(self, *, output_dir: str = ".anton/output") -> str:
        """Same payload shape as ``recall_skill``'s store-backed response."""
        return (
            f"# Skill: {self.label} (built-in)\n"
            f"\n"
            f"{self.description}\n"
            f"\n"
            f"## Procedure\n"
            f"\n"
            f"{self.render(output_dir=output_dir).strip()}"
        )


BUILTIN_SKILLS: dict[str, BuiltinSkill] = {
    s.label: s
    for s in (
        BuiltinSkill(
            label="html-dashboards",
            description=(
                "REQUIRED before building any HTML dashboard, chart page, "
                "report, infographic, or other browser-based visualization — "
                "the full build discipline (cell structure, artifact "
                "registration, ECharts setup, responsive layout, JS-string "
                "safety)."
            ),
            template=VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT,
        ),
        BuiltinSkill(
            label="backend-apps",
            description=(
                "REQUIRED before building any backend service, REST API, or "
                "fullstack web app — the full FastAPI/Mangum contract "
                "(artifact types, file layout, secrets, launch and "
                "deployment rules)."
            ),
            template=BACKEND_GENERATION_PROMPT,
        ),
        BuiltinSkill(
            label="public-data-sources",
            description=(
                "Catalog of free public data endpoints and URL patterns "
                "(news RSS, financial, economic, social) — recall before "
                "fetching public news, market, or world data."
            ),
            template=PUBLIC_DATA_SOURCES_PROMPT,
        ),
    )
}


def get_builtin_skill(label: str) -> BuiltinSkill | None:
    return BUILTIN_SKILLS.get((label or "").strip())


def builtin_skill_summaries() -> list[dict]:
    """Same shape as ``SkillStore.list_summaries()`` for listing/routing."""
    return [
        {"label": s.label, "description": s.description}
        for s in BUILTIN_SKILLS.values()
    ]


__all__ = [
    "BUILTIN_SKILLS",
    "BuiltinSkill",
    "builtin_skill_summaries",
    "get_builtin_skill",
]
