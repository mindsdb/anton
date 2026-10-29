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


def _md_text(md: str) -> str:
    """Markdown-lite to plain escaped text with line breaks kept as paragraphs."""
    text = re.sub(r"\*\*|`|^#+\s*", "", str(md or ""), flags=re.M)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return "".join(f"<p>{_html.escape(p.strip())}</p>" for p in re.split(r"\n\s*\n", text) if p.strip())


def static_html(spec: dict, limit: int = 200) -> str:
    """The page as plain HTML (headings, text, KPI values, tables), for a reader without JavaScript
    or before it runs. The runtime replaces it with the interactive page."""
    from .data import fmt

    cur = spec.get("currency") or "USD"
    data = spec.get("data") or {}
    esc = lambda v: _html.escape("" if v is None else str(v))  # noqa: E731
    out: list[str] = []

    def table(head: list, rows: list) -> str:
        th = "".join(f"<th>{esc(h)}</th>" for h in head)
        tb = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows[:limit])
        return f'<table><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table>'

    def label(b, f):
        return ((b.get("labels") or {}).get(f) or (spec.get("labels") or {}).get(f) or str(f))

    def form(b, f):
        return (b.get("formats") or {}).get(f) or (spec.get("formats") or {}).get(f)

    def walk(blocks):
        for b in blocks or []:
            t = b.get("type")
            title = f"<h3>{esc(b['title'])}</h3>" if b.get("title") else ""
            if t == "section":
                out.append(f"<section>{'<h2>' + esc(b['title']) + '</h2>' if b.get('title') else ''}")
                walk(b.get("children"))
                out.append("</section>")
            elif t == "grid":
                walk(b.get("children"))
            elif t == "tabs":
                for tab in b.get("tabs") or []:
                    out.append(f"<h3>{esc(tab['label'])}</h3>")
                    walk(tab.get("children"))
            elif t in ("text", "callout"):
                out.append(title + _md_text(b.get("md") or ""))
            elif t == "list":
                out.append(title + "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in b.get("items") or []) + "</ul>")
            elif t == "kpis":
                out.append("<ul>" + "".join(
                    f"<li>{esc(k.get('label'))}: <strong>{esc(k['value'] if isinstance(k.get('value'), str) and not k.get('format') else fmt(k.get('value'), k.get('format'), cur))}</strong></li>"
                    for k in b.get("items") or []) + "</ul>")
            elif t == "compare":
                rows = [[r.get("label")] + [fmt(v, r.get("format"), cur) if not isinstance(v, str) else v for v in r.get("values") or []]
                        for r in b.get("rows") or []]
                out.append(title + table([b.get("metric_label") or ""] + list(b.get("columns") or []), rows))
            elif t == "table":
                rws = data.get(b.get("data"), [])
                cols = [c.get("field") for c in (b.get("columns") or [])] or (list(rws[0].keys()) if rws else [])
                fmts = {c.get("field"): c.get("format") for c in (b.get("columns") or [])}
                labels = {c.get("field"): c.get("label") for c in (b.get("columns") or [])}
                rows = [[fmt(r.get(c), fmts.get(c) or form(b, c), cur) if not isinstance(r.get(c), str) else r.get(c) for c in cols] for r in rws]
                out.append(title + table([labels.get(c) or label(b, c) for c in cols], rows))
            elif t in ("chart", "echarts", "map", "globe", "network", "timeline", "image", "html"):
                if b.get("title"):
                    out.append(f"<figure><figcaption>{esc(b['title'])}</figcaption></figure>")
            for key in ("blocks", "right"):
                if isinstance(b.get(key), list) and t not in ("section", "grid", "tabs"):
                    walk(b[key])

    if spec.get("title"):
        out.append(f"<h1>{esc(spec['title'])}</h1>")
    if spec.get("subtitle"):
        out.append(_md_text(spec["subtitle"]))
    walk(spec.get("blocks"))
    for slide in spec.get("slides") or []:
        out.append(f"<h2>{esc(slide.get('title') or '')}</h2>")
        walk(slide.get("blocks"))
        walk(slide.get("right"))
    return '<div class="ak-static">' + "".join(out) + "</div>"


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
        f"<style>\n{_asset('ak.css')}</style>\n{custom_css}"
        f'<script type="application/json" id="ak-spec">{_script_json(spec)}</script>\n'
        "</head>\n<body>\n<div id=\"ak-root\">" + static_html(spec) + "</div>\n"
        f"<script>\n{js}</script>\n{custom_js}<script>AK.boot();</script>\n</body>\n</html>\n"
    )
