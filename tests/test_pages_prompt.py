"""The page guidance stays consistent with the code it documents.

Two texts describe pages. PAGES in the ARTIFACTS prompt tells the main agent
to route every `html-app` through `generate_artifact`, whose page builder
(`kit.builder.build_page`) carries the kit reference (`KIT_REFERENCE`, inside
`BUILDER_PROMPT`). These tests pin that each text documents calls and
parameters that exist, not its wording.
"""
import inspect
import re

from anton.core.artifacts import kit
from anton.core.artifacts.kit.builder import BUILDER_PROMPT, KIT_REFERENCE
from anton.core.llm.prompts import ARTIFACTS_PROMPT, STATIC_ARTIFACTS_PROMPT
from anton.core.tools.tool_defs import (
    GENERATE_ARTIFACT_TOOL,
    STATIC_GENERATE_ARTIFACT_TOOL,
)

PAGES_HEADER = "PAGES (every `html-app`):"


def _pages_block(prompt: str = ARTIFACTS_PROMPT) -> str:
    return prompt.split(PAGES_HEADER, 1)[1].split("WORKFLOW:", 1)[0]


def _reference() -> str:
    return KIT_REFERENCE.split("```", 2)[1]


def _top_level(inner: str) -> str:
    """`inner` with everything nested in brackets blanked out."""
    out, depth = [], 0
    for ch in inner:
        depth += {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}.get(ch, 0)
        out.append(ch if depth == 0 else " ")
    return "".join(out)


def _calls(text: str, owners: str = "page|kit|deck"):
    """(owner, name, keyword args) for every `<owner>.name(...)` call in `text`."""
    flat = re.sub(r"\s+", " ", text)
    for m in re.finditer(rf"\b({owners})\.(\w+)\(", flat):
        depth, i = 1, m.end()
        while i < len(flat) and depth:
            depth += {"(": 1, ")": -1}.get(flat[i], 0)
            i += 1
        inner = flat[m.end():i - 1]
        yield m.group(1), m.group(2), re.findall(r"(\w+)=", _top_level(inner))


# ── The kit reference the page builder reads ────────────────────────────────


def test_the_reference_is_part_of_the_builder_prompt():
    assert KIT_REFERENCE in BUILDER_PROMPT
    assert "from anton.core.artifacts import kit" in _reference()


def test_every_documented_call_exists_with_the_documented_parameters():
    seen = 0
    for owner, name, kwargs in _calls(_reference()):
        target = {"page": kit.Page, "deck": kit.Deck, "kit": kit}[owner]
        assert hasattr(target, name), f"{owner}.{name} is documented but does not exist"
        fn = getattr(target, name)
        if not callable(fn):
            continue
        params = inspect.signature(fn).parameters
        takes_kwargs = any(p.kind is p.VAR_KEYWORD for p in params.values())
        for arg in kwargs:
            assert arg in params or takes_kwargs, f"{owner}.{name} has no parameter {arg}"
        seen += 1
    assert seen >= 20


def test_documented_chart_kinds_are_the_kit_chart_kinds():
    from anton.core.artifacts.kit.page import CHART_KINDS

    line = next(x for x in _reference().splitlines() if x.lstrip().startswith("# kind:"))
    kinds = re.findall(r"\b([a-z]+)\b", re.sub(r"\(.*?\)", "", line.split(":", 1)[1]))
    assert kinds and set(kinds) <= set(CHART_KINDS)


def test_documented_formats_are_the_kit_formats():
    from anton.core.artifacts.kit.page import FORMATS

    lines = _reference().splitlines()
    at = next(i for i, x in enumerate(lines) if x.lstrip().startswith("# formats:"))
    text = lines[at].split(":", 1)[1] + " " + lines[at + 1]
    names = re.findall(r"\b([a-z_]+(?::\d+)?)\b(?:\s*\(|,|$)", re.sub(r"\(.*?\)", "(", text))
    assert len(names) >= 10
    for name in names:
        assert name.split(":")[0] in FORMATS, name


# ── PAGES in the main agent's prompt ────────────────────────────────────────


def test_pages_route_every_page_through_generate_artifact():
    for prompt in (ARTIFACTS_PROMPT, STATIC_ARTIFACTS_PROMPT):
        block = _pages_block(prompt)
        assert "generate_artifact(" in block
        assert "artifact kit" in block


def test_the_main_prompt_does_not_carry_the_kit_reference():
    """The builder reads the reference; the main agent only routes."""
    for prompt in (ARTIFACTS_PROMPT, STATIC_ARTIFACTS_PROMPT):
        assert not list(_calls(prompt)), "a page./kit./deck. call leaked into the main prompt"
        assert "kit.Page(" not in prompt


def test_the_documented_generate_artifact_call_matches_the_tool_schema():
    """Every argument PAGES passes exists in the schema, and every required
    one is passed."""
    props = GENERATE_ARTIFACT_TOOL.input_schema["properties"]
    required = set(GENERATE_ARTIFACT_TOOL.input_schema["required"])
    flat = re.sub(r"\s+", " ", _pages_block())
    m = re.search(r"generate_artifact\((.*?)\)`", flat)
    assert m, "PAGES documents no generate_artifact(...) call"
    placeholders_out = re.sub(r"<[^>]*>", "", m.group(1))  # `<what they asked for: a, b>`
    args = {a.split("=")[0].strip() for a in _top_level(placeholders_out).split(",") if a.strip()}
    assert args <= set(props), args - set(props)
    assert required <= args, required - args
    assert "name" in args and "slug" not in required


def test_the_static_prompt_never_offers_fullstack_generation():
    for text in (STATIC_ARTIFACTS_PROMPT, STATIC_GENERATE_ARTIFACT_TOOL.description,
                 STATIC_GENERATE_ARTIFACT_TOOL.prompt or ""):
        for word in ("fullstack-stateless-app", "fullstack-stateful-app", "launch_backend", "backend.py"):
            assert word not in text, word
    assert "prd.md" not in STATIC_ARTIFACTS_PROMPT


def test_the_block_stays_short():
    """The reference moved to the builder; PAGES only routes."""
    assert len(_pages_block()) < 3000
