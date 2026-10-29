"""Assemble a page: the spec (with its data) first, then the stylesheet and only the runtime
modules the page uses. One file, no build step; libraries load from cdn.jsdelivr.net at view time."""

from __future__ import annotations

import html as _html
import json
import re
from functools import lru_cache
from pathlib import Path

RUNTIME = Path(__file__).resolve().parent / "runtime"
MODULES = {  # block type -> runtime module
    "table": "table.js", "chart": "chart.js", "echarts": "chart.js", "map": "geo.js", "globe": "geo.js",
    "network": "flow.js", "flow": "flow.js", "timeline": "timeline.js",
}
ORDER = ["core.js", "blocks.js", "table.js", "chart.js", "geo.js", "flow.js", "timeline.js", "deck.js"]


@lru_cache(maxsize=None)
def _asset(name: str) -> str:
    text = (RUNTIME / name).read_text(encoding="utf-8")
    # Comments help maintainers, not readers of the page.
    if name.endswith(".js"):
        text = re.sub(r"(?m)^\s*/\*[\s\S]*?\*/\s*\n", "", text)
    elif name.endswith(".css"):
        text = re.sub(r"/\*[\s\S]*?\*/", "", text)
    return text.strip() + "\n"


def _types(spec: dict) -> set[str]:
    found: set[str] = set()

    def walk(blocks):
        for b in blocks or []:
            found.add(b.get("type"))
            for key in ("children", "blocks", "right"):
                if isinstance(b.get(key), list):
                    walk(b[key])
            for t in b.get("tabs") or []:
                walk(t.get("children"))
    walk(spec.get("blocks"))
    walk(spec.get("slides"))
    return found


def _script_json(value) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return text.replace("</", "<\\/").replace("<!--", "<\\!--")


def render_html(spec: dict, custom: dict | None = None) -> str:
    custom = custom or {}
    types = _types(spec)
    modules = {"core.js", "blocks.js"} | {MODULES[t] for t in types if t in MODULES}
    if spec.get("mode") == "deck":
        modules.add("deck.js")
    js = "".join(_asset(m) for m in ORDER if m in modules)
    lang = _html.escape(str(spec.get("lang") or "en"))
    theme = spec.get("theme") if spec.get("theme") in ("light", "dark") else "light"
    title = _html.escape(str(spec.get("title") or "Report"))
    custom_css = "".join(f'<style data-ak-custom="{_html.escape(bid)}">{c["css"]}</style>\n'
                         for bid, c in custom.items() if c.get("css"))
    custom_js = "".join(
        f'<script data-ak-custom="{_html.escape(bid)}">AK.custom["{bid}"]=function(el,AK){{\n'
        + re.sub(r"</script", r"<\\/script", c["js"], flags=re.I) + "\n};</script>\n"
        for bid, c in custom.items() if c.get("js"))
    return (
        "<!doctype html>\n"
        f'<html lang="{lang}" data-ak-theme="{theme}">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n"
        f'<script type="application/json" id="ak-spec">{_script_json(spec)}</script>\n'
        f"<style>\n{_asset('ak.css')}</style>\n{custom_css}"
        "</head>\n<body>\n<div id=\"ak-root\"></div>\n"
        f"<script>\n{js}</script>\n{custom_js}<script>AK.boot();</script>\n</body>\n</html>\n"
    )
