"""The PAGES guidance stays consistent with the artifact kit it documents."""
import inspect
import re

from anton.core.artifacts import kit
from anton.core.llm.prompts import ARTIFACTS_PROMPT, STATIC_ARTIFACTS_PROMPT
from anton.core.tools.tool_defs import CREATE_ARTIFACT_TOOL, GENERATE_ARTIFACT_TOOL


def _pages_block() -> str:
    return ARTIFACTS_PROMPT.split("PAGES (every `html-app`):", 1)[1].split("WORKFLOW:", 1)[0]


def _reference() -> str:
    block = _pages_block()
    return block.split("```", 2)[1]


def _calls(text: str):
    """(owner, name, keyword args) for every `page.x(...)`, `kit.x(...)` and `deck.x(...)` call."""
    flat = re.sub(r"\s+", " ", text)
    for m in re.finditer(r"\b(page|kit|deck)\.(\w+)\(", flat):
        depth, i = 1, m.end()
        while i < len(flat) and depth:
            depth += {"(": 1, ")": -1}.get(flat[i], 0)
            i += 1
        inner = flat[m.end():i - 1]
        top, depth = [], 0
        for ch in inner:  # keyword names at the call's own level only
            depth += {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}.get(ch, 0)
            top.append(ch if depth == 0 else " ")
        yield m.group(1), m.group(2), re.findall(r"(\w+)=", "".join(top))


def test_every_documented_call_exists_with_the_documented_parameters():
    block = _pages_block()
    assert "from anton.core.artifacts import kit" in block
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


def test_documented_formats_are_the_kit_formats():
    from anton.core.artifacts.kit.page import FORMATS
    line = next(x for x in _reference().splitlines() if x.lstrip().startswith("# formats:"))
    following = _reference().split(line, 1)[1].splitlines()[0]
    for name in re.findall(r"\b([a-z_]+)\b(?:\s*\(|,|$)", line.split(":", 1)[1] + " " + following):
        if name in ("e", "g", "nov"):
            continue
        assert name.split(":")[0] in FORMATS, name


def test_pages_never_go_to_the_generator():
    for text in (ARTIFACTS_PROMPT, STATIC_ARTIFACTS_PROMPT, CREATE_ARTIFACT_TOOL.description):
        assert "artifact kit" in text
    assert "html-app" in GENERATE_ARTIFACT_TOOL.description and "not built here" in GENERATE_ARTIFACT_TOOL.description
    assert "generate_artifact" not in STATIC_ARTIFACTS_PROMPT


def test_the_block_stays_reasonable():
    assert len(_pages_block()) < 9000
