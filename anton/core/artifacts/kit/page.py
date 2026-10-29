"""Page and Deck builders: compose blocks over real data, then save one self-contained HTML file."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from .data import fields as _fields
from .data import fmt as _fmt
from .data import is_number, records

THEMES = ("light", "dark", "auto")
WIDTHS = ("narrow", "normal", "wide", "full", "mobile")
CHART_KINDS = ("bar", "hbar", "line", "area", "combo", "scatter", "pie", "donut", "heatmap", "waterfall")
TONES = ("good", "bad", "warn", "info", "accent", "neutral")
FORMATS = ("auto", "text", "int", "number", "percent", "fraction", "currency", "currency_compact", "compact",
           "signed", "days", "hours", "date", "date_short", "date_iso", "month", "datetime", "weekday")


class KitError(ValueError):
    """A mistake in how a page is put together, with what to change."""


def _check_format(fmt: str | None, where: str) -> None:
    if fmt is None:
        return
    base = str(fmt).split(":")[0]
    if base not in FORMATS:
        raise KitError(f"{where}: unknown format {fmt!r}. Use one of: {', '.join(FORMATS)} (optionally ':N' decimals).")


def _need(rows: list[dict], names: Iterable[str | None], where: str) -> None:
    have = _fields(rows)
    missing = [n for n in names if n is not None and rows and n not in have]
    if missing:
        raise KitError(f"{where}: no field {', '.join(map(repr, missing))} in the data. Fields: {', '.join(have)}.")


# ---------------------------------------------------------------------------------------------
# Item helpers (plain dicts the blocks take)
# ---------------------------------------------------------------------------------------------

def kpi(label: str, value: Any, format: str | None = None, *, sub: str | None = None, delta: Any = None,
        delta_format: str | None = None, delta_label: str | None = None, better: str | None = None,
        tone: str | None = None, spark: list | None = None, color_value: bool = False) -> dict:
    """One KPI card. `better` ("up"/"down") colours the delta; `tone` marks the card good/bad/warn/accent."""
    _check_format(format, f"kpi {label!r}")
    from .data import _scalar
    return {k: v for k, v in dict(label=label, value=_scalar(value), format=format, sub=sub, delta=_scalar(delta),
                                   delta_format=delta_format, delta_label=delta_label, better=better, tone=tone,
                                   spark=[_scalar(x) for x in spark] if spark is not None else None,
                                   color_value=color_value or None).items() if v is not None}


def metric(label: str, values: list, format: str | None = None, *, better: str | None = None, note: str | None = None,
           diff_format: str | None = None, show_pct: bool = False, tones: list | None = None) -> dict:
    """One row of `compare`: a metric's value in each column, in column order."""
    _check_format(format, f"metric {label!r}")
    from .data import _scalar
    return {k: v for k, v in dict(label=label, values=[_scalar(x) for x in values], format=format, better=better,
                                   note=note, diff_format=diff_format, show_pct=show_pct or None,
                                   tones=tones).items() if v is not None}


def col(field: str, label: str | None = None, format: str | None = None, *, pill: bool = False,
        tones: dict | None = None, bar: Any = False, align: str | None = None, wrap: bool = False) -> dict:
    """A table column. `pill` shows the value as a status pill (`tones` maps values to good/bad/warn/info);
    `bar` adds an inline bar for numbers; `wrap` gives long text room."""
    _check_format(format, f"column {field!r}")
    return {k: v for k, v in dict(field=field, label=label, format=format, pill=pill or None, tones=tones,
                                   bar=bar or None, align=align, wrap=wrap or None).items() if v is not None}


def points(data: Any, lat: str = "lat", lon: str = "lon", *, label: str | None = None, sublabel: str | None = None,
           color: str | None = None, colors: dict | None = None, size: str | None = None, radius: float | None = None,
           shape: str | None = None, tooltip: list | None = None, key: str | None = None, link: str | None = None,
           pulse: Any = None, name: str | None = None, label_always: bool | None = None,
           labels: dict | None = None, formats: dict | None = None) -> dict:
    """A map/globe layer of markers. `color` is a field (categories get palette colours, `colors` overrides)
    or a colour/tone; `shape` circle|square|diamond|triangle; `pulse` True or a field whose truthy rows pulse."""
    return {"type": "points", "data": data, "lat": lat, "lon": lon, "label": label, "sublabel": sublabel,
            "color": color, "colors": colors, "size": size, "radius": radius, "shape": shape, "tooltip": tooltip,
            "key": key, "link": link, "pulse": pulse, "name": name, "label_always": label_always,
            "labels": labels, "formats": formats}


def routes(data: Any, from_lat: str, from_lon: str, to_lat: str, to_lon: str, *, via_lat: str | None = None,
           via_lon: str | None = None, color: str | None = None, colors: dict | None = None, label: str | None = None,
           width: str | None = None, animate: bool = False, key: str | None = None, link: str | None = None,
           tooltip: list | None = None, name: str | None = None, remaining_label: str | None = None,
           labels: dict | None = None, formats: dict | None = None) -> dict:
    """A layer of great-circle routes from one point to another. With `via_*` (e.g. where a shipment is
    now) the leg before it is solid and the leg after it dashed."""
    return {"type": "routes", "data": data, "from_lat": from_lat, "from_lon": from_lon, "to_lat": to_lat,
            "to_lon": to_lon, "via_lat": via_lat, "via_lon": via_lon, "color": color, "colors": colors,
            "label": label, "width": width, "animate": animate, "key": key, "link": link, "tooltip": tooltip,
            "name": name, "remaining_label": remaining_label, "labels": labels, "formats": formats}


def regions(data: Any, country: str, value: str, *, format: str | None = None, color: str | None = None,
            name: str | None = None) -> dict:
    """Countries shaded by a value (country names in English, or ISO numeric codes)."""
    return {"type": "regions", "data": data, "country": country, "value": value, "format": format,
            "color": color, "name": name}


def bars(data: Any, row: str, start: str, end: str, *, label: str | None = None, color: str | None = None,
         colors: dict | None = None, tone: str | None = None, name: str | None = None, tooltip: list | None = None,
         dashed: bool = False, overlay: bool = False, labels: dict | None = None, formats: dict | None = None) -> dict:
    """A timeline layer of spans (start..end dates) per row. Each bars layer gets its own lane in a row,
    so two layers compare plans; `overlay` draws on the previous lane instead."""
    return {"data": data, "row": row, "start": start, "end": end, "label": label, "color": color, "colors": colors,
            "tone": tone, "name": name, "tooltip": tooltip, "dashed": dashed or None, "overlay": overlay or None,
            "labels": labels, "formats": formats}


def milestones(data: Any, row: str, date: str, *, label: str | None = None, color: str | None = None,
               colors: dict | None = None, tone: str | None = None, name: str | None = None, shape: str = "circle",
               tooltip: list | None = None, show_labels: bool = False, detail_fields: list | None = None,
               labels: dict | None = None, formats: dict | None = None) -> dict:
    """A timeline layer of dated events per row. Hover/tap shows `label`, the date and `tooltip` fields."""
    return {"data": data, "row": row, "date": date, "label": label, "color": color, "colors": colors, "tone": tone,
            "name": name, "shape": shape, "tooltip": tooltip, "show_labels": show_labels or None,
            "detail_fields": detail_fields, "labels": labels, "formats": formats}


def marker(date: Any, label: str | None = None, tone: str = "neutral") -> dict:
    """A dashed vertical line across a timeline (e.g. "As of", "Deadline")."""
    from .data import _scalar
    return {"date": _scalar(date), "label": label, "tone": tone}


def rows(data: Any, key: str, label: str | None = None, sublabel: str | None = None) -> dict:
    """Timeline row order and labels (optional; rows otherwise come from the layers)."""
    return {"data": data, "key": key, "label": label, "sublabel": sublabel}


# ---------------------------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------------------------

class _Tabs:
    def __init__(self, page: "Page", block: dict):
        self._page, self.block = page, block

    def __getitem__(self, which):
        tabs = self.block["tabs"]
        if isinstance(which, int):
            tab = tabs[which]
        else:
            match = [t for t in tabs if t["label"] == which]
            if not match:
                raise KitError(f"no tab {which!r}; tabs are {[t['label'] for t in tabs]}")
            tab = match[0]
        return self._page._into(tab["children"])


class Page:
    """A report or dashboard page. Add blocks in reading order; containers are context managers.

        page = kit.Page("Title", "One-line takeaway", lang="en", badges=["Scenario X"])
        page.kpis([kit.kpi("Revenue", 1.2e6, "currency_compact")])
        with page.grid(2):
            page.chart("bar", df, x="month", y="revenue", title="Revenue by month", format="currency")
            page.table(df, title="Detail")
        page.save(artifact_path)
    """

    mode = "page"

    def __init__(self, title: str = "", subtitle: str | None = None, *, lang: str = "en", theme: str = "light",
                 badges: list | None = None, meta: list | None = None, width: str = "normal", currency: str = "USD",
                 accent: str | None = None, motion: bool = True, sources: list | None = None,
                 notes: list | None = None, labels: dict | None = None, formats: dict | None = None,
                 palette: list | None = None, state: dict | None = None):
        if theme not in THEMES:
            raise KitError(f"theme must be one of {THEMES}")
        if width not in WIDTHS:
            raise KitError(f"width must be one of {WIDTHS}")
        for f, v in (formats or {}).items():
            _check_format(v, f"formats[{f!r}]")
        self.spec: dict = {"v": 1, "mode": self.mode, "title": title, "subtitle": subtitle, "lang": lang,
                           "theme": theme, "badges": badges or [], "meta": meta or [], "width": width,
                           "currency": currency, "accent": accent, "motion": motion, "sources": sources or [],
                           "notes": notes or [], "labels": labels or {}, "formats": formats or {},
                           "palette": palette, "state": state or {}, "blocks": [], "data": {}}
        self._custom: dict[str, dict] = {}
        self._targets: list[list] = [self.spec["blocks"]]
        self._cursor: list | None = None  # [container, index] while inserting after/before a block
        self.warnings: list[str] = []

    # ---- internals ----
    def _next_id(self) -> str:
        used = {b.get("id") for b in self._walk()}
        n = len(used) + 1
        while f"b{n}" in used:
            n += 1
        return f"b{n}"

    def _walk(self, blocks: list | None = None):
        for b in (self._roots() if blocks is None else blocks):
            yield b
            for key in ("children", "blocks", "right"):
                if isinstance(b.get(key), list):
                    yield from self._walk(b[key])
            for t in b.get("tabs") or []:
                yield from self._walk(t["children"])

    def _roots(self) -> list:
        return self.spec["blocks"]

    def _add(self, block: dict) -> dict:
        block = {k: v for k, v in block.items() if v is not None}
        block["id"] = block.get("id") or self._next_id()
        if self._cursor is not None:
            container, index = self._cursor
            container.insert(index, block)
            self._cursor[1] += 1
        else:
            self._targets[-1].append(block)
        return block

    @contextmanager
    def _into(self, children: list):
        saved = self._cursor
        self._cursor = None
        self._targets.append(children)
        try:
            yield self
        finally:
            self._targets.pop()
            self._cursor = saved

    def _dataset(self, data: Any, where: str) -> tuple[str, list[dict]]:
        if isinstance(data, str) and data in self.spec["data"]:
            return data, self.spec["data"][data]
        rows = records(data)
        if not rows:
            self.warnings.append(f"{where}: the data is empty, so this block shows nothing.")
        blob = json.dumps(rows, sort_keys=True, default=str).encode()
        digest = hashlib.sha1(blob).hexdigest()[:8]
        for name, existing in self.spec["data"].items():
            if name.endswith(digest) and existing == rows:
                return name, existing
        name = f"d{len(self.spec['data']) + 1}_{digest}"
        self.spec["data"][name] = rows
        return name, rows

    # ---- containers ----
    @contextmanager
    def section(self, title: str | None = None, subtitle: str | None = None, *, id: str | None = None):
        """A titled group of blocks."""
        block = self._add({"type": "section", "title": title, "subtitle": subtitle, "children": [], "id": id})
        with self._into(block["children"]):
            yield block

    @contextmanager
    def grid(self, columns: int = 2, *, widths: list | None = None, id: str | None = None):
        """Side-by-side blocks (each block added inside is one cell). Collapses to one column on phones.
        `widths` gives relative column widths, e.g. [2, 1]."""
        if widths and len(widths) != columns:
            raise KitError("grid: widths needs one number per column")
        block = self._add({"type": "grid", "columns": columns, "widths": widths, "children": [], "id": id})
        with self._into(block["children"]):
            yield block

    def tabs(self, labels: list[str], *, style: str = "tabs", state: str | None = None, label: str | None = None,
             initial: int = 0, id: str | None = None) -> _Tabs:
        """Tabs (style "tabs") or a segmented switch (style "segmented", e.g. a scenario toggle).
        Fill each with `with t["Label"]:`. Tab groups sharing `state` switch together."""
        if style not in ("tabs", "segmented"):
            raise KitError("tabs: style is 'tabs' or 'segmented'")
        block = self._add({"type": "tabs", "style": style, "state": state, "label": label, "initial": initial or None,
                           "tabs": [{"label": str(x), "children": []} for x in labels], "id": id})
        return _Tabs(self, block)

    # ---- text ----
    def text(self, md: str, *, style: str | None = None, card: bool = False, title: str | None = None,
             id: str | None = None) -> dict:
        """Markdown-lite text: paragraphs, **bold**, *italic*, `code`, [links](https://...), lists, ## headings.
        style="lead" for a larger intro paragraph."""
        return self._add({"type": "text", "md": md, "style": style, "card": card or None, "title": title, "id": id})

    def callout(self, md: str, *, title: str | None = None, tone: str = "info", icon: str | None = None,
                id: str | None = None) -> dict:
        """A highlighted message: the key finding, a recommendation, a caveat. tone good|bad|warn|info|accent|neutral."""
        if tone not in TONES:
            raise KitError(f"callout tone must be one of {TONES}")
        return self._add({"type": "callout", "md": md, "title": title, "tone": tone, "icon": icon, "id": id})

    def list(self, items: list[str], *, ordered: bool = False, title: str | None = None, id: str | None = None) -> dict:
        return self._add({"type": "list", "items": [str(i) for i in items], "ordered": ordered or None,
                          "title": title, "id": id})

    def divider(self) -> dict:
        return self._add({"type": "divider"})

    # ---- figures ----
    def kpis(self, items: list[dict], *, columns: int | None = None, id: str | None = None) -> dict:
        """A row of KPI cards (use kit.kpi for each)."""
        for it in items:
            if "label" not in it or "value" not in it:
                raise KitError("kpis: each item needs a label and a value (use kit.kpi(label, value, format))")
        return self._add({"type": "kpis", "items": list(items), "columns": columns, "id": id})

    def compare(self, columns: list[str], rows: list[dict], *, title: str | None = None, subtitle: str | None = None,
                metric_label: str = "", show_diff: bool | None = None, diff_label: str | None = None,
                note: str | None = None, id: str | None = None) -> dict:
        """Metrics side by side across options/scenarios/periods (use kit.metric per row). With two
        columns a change column (second minus first) is added; `better` on a metric colours it."""
        for r in rows:
            if len(r.get("values", [])) != len(columns):
                raise KitError(f"compare: metric {r.get('label')!r} has {len(r.get('values', []))} values for {len(columns)} columns")
        return self._add({"type": "compare", "columns": [str(c) for c in columns], "rows": list(rows), "title": title,
                          "subtitle": subtitle, "metric_label": metric_label, "show_diff": show_diff,
                          "diff_label": diff_label, "note": note, "id": id})

    def chart(self, kind: str, data: Any, x: str | None = None, y: str | list | None = None, *,
              series: str | None = None, title: str | None = None, subtitle: str | None = None,
              note: str | None = None, format: str | None = None, x_format: str | None = None,
              labels: dict | None = None, formats: dict | None = None, stack: bool = False,
              horizontal: bool = False, sort: str | None = None, show_values: bool | None = None,
              ref_lines: list | None = None, threshold: dict | None = None, point_tones: Any = None,
              y2: str | list | None = None, y2_format: str | None = None, y2_kind: str | None = None,
              y2_label: str | None = None, colors: dict | None = None, height: int | None = None,
              y_min: float | None = None, y_max: float | None = None, x_label: str | None = None,
              y_label: str | None = None, smooth: bool | None = None, legend: bool | None = None,
              link: str | None = None, key: str | None = None, size: str | None = None, label: str | None = None,
              value: str | None = None, y_category: str | None = None, totals: list | None = None,
              dashed: Any = None, area: bool | None = None, show_data: bool | None = None,
              center_label: str | None = None, id: str | None = None) -> dict:
        """A chart. kind: bar | hbar | line | area | combo (bars y + line y2) | scatter | pie | donut |
        heatmap (x, y_category, value) | waterfall (x, y, totals=[names of total bars]).
        Wide data: y=["a","b"]; long data: y="value", series="group". Values are drawn as given (pre-aggregate)."""
        if kind not in CHART_KINDS:
            raise KitError(f"chart kind {kind!r} is not one of {CHART_KINDS}; use page.echarts(option) for others")
        if kind == "combo":
            kind = "bar"
            if not y2:
                raise KitError("combo chart: give y (bars) and y2 (line)")
        _check_format(format, f"chart {title!r}")
        _check_format(x_format, f"chart {title!r}")
        _check_format(y2_format, f"chart {title!r}")
        name, rws = self._dataset(data, f"chart {title!r}")
        need = [x, series, size, label, value, y_category] + ([y] if isinstance(y, str) else list(y or []))
        need += [y2] if isinstance(y2, str) else list(y2 or [])
        if isinstance(point_tones, str):
            need.append(point_tones)
        _need(rws, need, f"chart {title!r}")
        if x is None:
            raise KitError("chart: x is required")
        if kind != "heatmap" and y is None:
            raise KitError("chart: y is required")
        ylist = [y] if isinstance(y, str) else list(y or [])
        for f in ylist:
            bad = [r.get(f) for r in rws if r.get(f) is not None and not is_number(r.get(f))]
            if bad:
                raise KitError(f"chart {title!r}: y field {f!r} must be numeric; got e.g. {bad[0]!r}")
        if series and kind in ("bar", "hbar", "line", "area"):
            seen = set()
            for r in rws:
                k = (r.get(x), r.get(series))
                if k in seen:
                    self.warnings.append(f"chart {title!r}: several rows share {x}={k[0]!r}, {series}={k[1]!r}; they are summed. Aggregate first if that is not intended.")
                    break
                seen.add(k)
        return self._add({"type": "chart", "kind": kind, "data": name, "x": x, "y": y, "series": series,
                          "title": title, "subtitle": subtitle, "note": note, "format": format, "x_format": x_format,
                          "labels": labels, "formats": formats, "stack": stack or None, "horizontal": horizontal or None,
                          "sort": sort, "show_values": show_values, "ref_lines": ref_lines, "threshold": threshold,
                          "point_tones": point_tones, "y2": y2, "y2_format": y2_format, "y2_kind": y2_kind,
                          "y2_label": y2_label, "colors": colors, "height": height, "y_min": y_min, "y_max": y_max,
                          "x_label": x_label, "y_label": y_label, "smooth": smooth, "legend": legend, "link": link,
                          "key": key, "size": size, "label": label, "value": value, "y_category": y_category,
                          "totals": totals, "dashed": dashed, "area": area, "show_data": show_data,
                          "center_label": center_label, "id": id})

    def echarts(self, option: dict, *, title: str | None = None, subtitle: str | None = None, note: str | None = None,
                height: int | None = None, id: str | None = None) -> dict:
        """Any ECharts option (radar, treemap, sunburst, funnel, gauge, calendar, boxplot, parallel, ...).
        JSON only: use ECharts template strings for formatters, not functions."""
        try:
            json.dumps(option)
        except TypeError as e:
            raise KitError(f"echarts option must be JSON (no functions): {e}") from None
        return self._add({"type": "echarts", "option": option, "title": title, "subtitle": subtitle, "note": note,
                          "height": height, "id": id})

    def table(self, data: Any, columns: list | None = None, *, title: str | None = None, subtitle: str | None = None,
              note: str | None = None, search: bool | None = None, filters: list | None = None, sort: Any = None,
              key: str | None = None, link: str | None = None, detail: Any = None, detail_fields: list | None = None,
              detail_title: str | None = None, labels: dict | None = None, formats: dict | None = None,
              totals: dict | None = None, totals_label: str | None = None, row_tone: dict | None = None,
              max_height: int | None = None, mobile: str | None = None, caption: str | None = None,
              empty_text: str | None = None, sortable: bool = True, page_size: int | None = None,
              id: str | None = None) -> dict:
        """A table. columns: field names or kit.col(...). search/filters (fields for select filters), sort
        ("field" or {"field","desc"}), detail ("drawer" | "inline" | False: what a row click opens), key+link
        (row click selects that key in every block sharing the link), totals {"field": "sum"|"mean"|"count"},
        row_tone {"field": f, "map": {value: tone}}."""
        name, rws = self._dataset(data, f"table {title!r}")
        cols = [c if isinstance(c, dict) else {"field": c} for c in (columns or [])]
        flt = [f if isinstance(f, str) else f.get("field") for f in (filters or [])]
        _need(rws, [c["field"] for c in cols] + flt + [key] + list(detail_fields or []) + list((totals or {}).keys()),
              f"table {title!r}")
        if link and not key:
            raise KitError(f"table {title!r}: link needs key (the field other blocks select by)")
        if detail not in (None, True, False, "drawer", "inline"):
            raise KitError("table detail is 'drawer', 'inline', True or False")
        if len(rws) > 2000:
            self.warnings.append(f"table {title!r}: {len(rws)} rows; consider summarising or filtering")
        return self._add({"type": "table", "data": name, "columns": cols or None, "title": title, "subtitle": subtitle,
                          "note": note, "search": search, "filters": filters, "sort": sort, "key": key, "link": link,
                          "detail": detail, "detail_fields": detail_fields, "detail_title": detail_title,
                          "labels": labels, "formats": formats, "totals": totals, "totals_label": totals_label,
                          "row_tone": row_tone, "max_height": max_height, "mobile": mobile, "caption": caption,
                          "empty_text": empty_text, "sortable": None if sortable else False,
                          "page_size": page_size, "id": id})

    def _geo_layers(self, layers: tuple, where: str) -> list[dict]:
        out = []
        for L in layers:
            if not isinstance(L, dict) or L.get("type") not in ("points", "routes", "regions"):
                raise KitError(f"{where}: layers are kit.points(...), kit.routes(...) or kit.regions(...)")
            L = {k: v for k, v in L.items() if v is not None}
            rws = records(L["data"]) if not isinstance(L["data"], str) else self.spec["data"][L["data"]]
            label = L.get("label") or L.get("key")
            if L["type"] == "points":
                _need(rws, [L["lat"], L["lon"], L.get("label"), L.get("sublabel"), L.get("key"), L.get("size")]
                      + list(L.get("tooltip") or []), where)
                coords = (("lat", L["lat"]), ("lon", L["lon"]))
            elif L["type"] == "routes":
                keys = ["from_lat", "from_lon", "to_lat", "to_lon"] + (["via_lat", "via_lon"] if L.get("via_lat") else [])
                _need(rws, [L[k] for k in keys] + [L.get("label"), L.get("key"), L.get("width")]
                      + list(L.get("tooltip") or []), where)
                coords = tuple((k, L[k]) for k in keys)
            else:
                _need(rws, [L["country"], L["value"]], where)
                coords = ()
            if isinstance(L.get("color"), str) and rws and L["color"] in _fields(rws):
                pass
            elif L.get("color") and L["type"] != "regions":
                L["color_value"] = L.pop("color")
            if coords:
                # Coordinates go first in each row as named lat/lon pairs, so they read as places.
                fixed, missing = [], []
                for r in rws:
                    vals = {}
                    for std, src in coords:
                        v = r.get(src)
                        vals[std] = float(round(float(v), 5)) if is_number(v) else None
                    if any(v is None for v in vals.values()):
                        missing.append(str(r.get(label, "?")) if label else "?")
                        continue
                    bad = [k for k, v in vals.items() if (k.endswith("lat") and abs(v) > 90) or (k.endswith("lon") and abs(v) > 180)]
                    if bad:
                        raise KitError(f"{where}: {bad[0]} out of range in row {r.get(label, r)} (latitude ±90, longitude ±180); lat/lon swapped?")
                    rest = {k: v for k, v in r.items() if k not in {s for _, s in coords}}
                    fixed.append({**vals, **rest})
                if missing:
                    self.warnings.append(f"{where}: {len(missing)} row(s) without coordinates are not drawn: {', '.join(missing[:8])}")
                for std, src in coords:
                    L[std] = std
                rws = fixed
            name, _ = self._dataset(rws, where)
            L["data"] = name
            out.append(L)
        return out

    def map(self, *layers: dict, title: str | None = None, subtitle: str | None = None, note: str | None = None,
            height: int | None = None, fit: str | None = None, projection: str | None = None,
            legend: bool | None = None, detail: str | None = None, center_lon: float | None = None,
            id: str | None = None) -> dict:
        """A world/region map with country shapes. Layers: kit.points, kit.routes, kit.regions. Zooms to the
        data (fit="world" for the whole world); projection natural|equal_earth|mercator."""
        L = self._geo_layers(layers, f"map {title!r}")
        return self._add({"type": "map", "layers": L, "title": title, "subtitle": subtitle, "note": note,
                          "height": height, "fit": fit, "projection": projection, "legend": legend,
                          "detail": detail, "center_lon": center_lon, "id": id})

    def globe(self, *layers: dict, title: str | None = None, subtitle: str | None = None, note: str | None = None,
              height: int | None = None, rotate: Any = None, speed: float | None = None, legend: bool | None = None,
              center_lon: float | None = None, center_lat: float | None = None, id: str | None = None) -> dict:
        """A 3D globe (orthographic, drag to turn). rotate: "spin" (continuous), "sway" (gentle, for regional
        data) or False; default picks by how spread the data is. Routes arc above the surface."""
        L = self._geo_layers(layers, f"globe {title!r}")
        return self._add({"type": "globe", "layers": L, "title": title, "subtitle": subtitle, "note": note,
                          "height": height, "rotate": rotate, "speed": speed, "legend": legend,
                          "center_lon": center_lon, "center_lat": center_lat, "id": id})

    def network(self, links: Any, *, source: str = "source", target: str = "target", value: str | None = None,
                nodes: Any = None, id_field: str = "id", label: str | None = None, group: str | None = None,
                tiers: list | None = None, style: str = "graph", title: str | None = None,
                subtitle: str | None = None, note: str | None = None, value_format: str | None = None,
                link_label: Any = None, link_color: str | None = None, link_colors: dict | None = None,
                node_colors: dict | None = None, dashed: Any = None, tooltip: list | None = None,
                node_tooltip: list | None = None, link: str | None = None, highlight: str | None = None,
                height: int | None = None, arrows: bool = True, show_totals: bool = True,
                labels: dict | None = None, formats: dict | None = None, id: str | None = None) -> dict:
        """A network diagram from link rows (source, target, value). `nodes` (optional) gives node rows with
        `id_field`, `label`, `group`; `tiers` orders groups left to right (e.g. ["Supplier","Plant","DC"]).
        style "graph" (boxes and arrows, value labels on links) or "sankey" (bands sized by value).
        `link_color` colours links by a field (e.g. lane kind) with `link_colors` {value: colour/tone};
        `dashed` True or a field marking dashed links; `link_label` a field to label links with (False: none)."""
        if style not in ("graph", "sankey"):
            raise KitError("network style is 'graph' or 'sankey'")
        _check_format(value_format, f"network {title!r}")
        lname, lrows = self._dataset(links, f"network {title!r} links")
        _need(lrows, [source, target, value, link_color, link_label if isinstance(link_label, str) else None]
              + list(tooltip or []) + ([dashed] if isinstance(dashed, str) else []), f"network {title!r} links")
        nspec: dict = {"id": id_field}
        if nodes is not None:
            nname, nrows = self._dataset(nodes, f"network {title!r} nodes")
            _need(nrows, [id_field, label, group] + list(node_tooltip or []), f"network {title!r} nodes")
            nspec.update(data=nname, label=label, group=group, tooltip=node_tooltip, colors=node_colors)
            ids = {str(r.get(id_field)) for r in nrows}
            loose = sorted({str(r[k]) for r in lrows for k in (source, target) if str(r.get(k)) not in ids})
            if loose:
                self.warnings.append(f"network {title!r}: link ends not in nodes (drawn with their id as label): {', '.join(loose[:8])}")
        if tiers and not group:
            raise KitError("network: tiers order the node `group` values; give nodes=... and group=...")
        lspec = {"data": lname, "source": source, "target": target, "value": value, "label": link_label,
                 "color": link_color, "colors": link_colors, "dashed": dashed, "tooltip": tooltip,
                 "labels": labels, "formats": formats}
        return self._add({"type": "network", "style": style, "nodes": {k: v for k, v in nspec.items() if v is not None},
                          "links": {k: v for k, v in lspec.items() if v is not None}, "tiers": tiers, "title": title,
                          "subtitle": subtitle, "note": note, "value_format": value_format, "link": link,
                          "highlight": highlight, "height": height, "arrows": None if arrows else False,
                          "show_totals": None if show_totals else False, "labels": labels, "id": id})

    def flow(self, links: Any, **kw) -> dict:
        """A sankey: network(..., style="sankey")."""
        kw.setdefault("style", "sankey")
        return self.network(links, **kw)

    def timeline(self, *, bars: list | dict | None = None, milestones: list | dict | None = None,
                 markers: list | None = None, rows: dict | None = None, title: str | None = None,
                 subtitle: str | None = None, note: str | None = None, link: str | None = None,
                 range: list | None = None, row_height: int | None = None, detail: bool = True,
                 legend: bool | None = None, id: str | None = None) -> dict:
        """A timeline / Gantt chart. bars: kit.bars(...) layers (one lane each, so plans compare);
        milestones: kit.milestones(...) layers; markers: kit.marker(date, label, tone); rows: kit.rows(...)."""
        def prep(layer, kind):
            layer = {k: v for k, v in layer.items() if v is not None}
            name, rws = self._dataset(layer["data"], f"timeline {title!r} {kind}")
            want = [layer.get("row"), layer.get("start"), layer.get("end"), layer.get("date"), layer.get("label")]
            want += list(layer.get("tooltip") or []) + list(layer.get("detail_fields") or [])
            if isinstance(layer.get("color"), str) and rws and layer["color"] in _fields(rws):
                want.append(layer["color"])
            elif layer.get("color"):
                layer["color_value"] = layer.pop("color")
            _need(rws, want, f"timeline {title!r} {kind}")
            layer["data"] = name
            return layer
        bl = [prep(b, "bars") for b in ([bars] if isinstance(bars, dict) else (bars or []))]
        ml = [prep(m, "milestones") for m in ([milestones] if isinstance(milestones, dict) else (milestones or []))]
        rs = None
        if rows:
            rname, rrows = self._dataset(rows["data"], f"timeline {title!r} rows")
            _need(rrows, [rows.get("key"), rows.get("label"), rows.get("sublabel")], f"timeline {title!r} rows")
            rs = {k: v for k, v in {**rows, "data": rname}.items() if v is not None}
        if not bl and not ml:
            raise KitError("timeline needs bars or milestones")
        return self._add({"type": "timeline", "bars": bl or None, "milestones": ml or None, "markers": markers,
                          "rows": rs, "title": title, "subtitle": subtitle, "note": note, "link": link,
                          "range": range, "row_height": row_height, "detail": None if detail else False,
                          "legend": legend, "id": id})

    def image(self, src: Any, *, title: str | None = None, subtitle: str | None = None, alt: str | None = None,
              note: str | None = None, card: bool = True, id: str | None = None) -> dict:
        """An image: a file path, PNG/JPEG/SVG bytes, a data: URI, or a matplotlib figure (embedded)."""
        uri = _image_uri(src)
        return self._add({"type": "image", "src": uri, "title": title, "subtitle": subtitle, "alt": alt,
                          "note": note, "card": None if card else False, "id": id})

    def html(self, html: str, *, js: str | None = None, css: str | None = None, title: str | None = None,
             subtitle: str | None = None, note: str | None = None, libs: list | None = None,
             view: str | None = None, card: bool = True, id: str | None = None) -> dict:
        """Custom markup for anything the blocks don't cover. `js` is the body of function(el, AK): `el` is
        this block's element; AK gives AK.data(name), AK.fmt(v, format), AK.palette(), AK.css('--ak-accent'),
        AK.select(link, key), AK.onSelect(link, fn), AK.tip(html, event). `libs`: ["echarts","d3","topojson"]
        are loaded first. `view` names what it draws (map, chart, globe, network, timeline)."""
        block = self._add({"type": "html", "html": html, "title": title, "subtitle": subtitle, "note": note,
                           "libs": libs, "view": view, "card": None if card else False, "id": id})
        self._custom[block["id"]] = {"js": js, "css": css}
        return block

    # ---- data access for custom blocks ----
    def dataset(self, data: Any, name: str | None = None) -> str:
        """Embed rows for custom code (AK.data(name)); returns the dataset name."""
        dname, rws = self._dataset(data, "dataset")
        if name and name != dname:
            self.spec["data"][name] = self.spec["data"].pop(dname)
            return name
        return dname

    # ---- editing a saved page ----
    def block(self, id: str) -> dict:
        for b in self._walk():
            if b.get("id") == id:
                return b
        raise KitError(f"no block {id!r}; see page.outline()")

    def _locate(self, id: str) -> tuple[list, int]:
        def search(blocks: list):
            for i, b in enumerate(blocks):
                if b.get("id") == id:
                    return blocks, i
                for key in ("children", "blocks", "right"):
                    if isinstance(b.get(key), list):
                        hit = search(b[key])
                        if hit:
                            return hit
                for t in b.get("tabs") or []:
                    hit = search(t["children"])
                    if hit:
                        return hit
            return None
        hit = search(self._roots())
        if not hit:
            raise KitError(f"no block {id!r}; see page.outline()")
        return hit

    @contextmanager
    def after(self, id: str):
        """New blocks go right after block `id` (see outline())."""
        container, i = self._locate(id)
        saved = self._cursor
        self._cursor = [container, i + 1]
        try:
            yield self
        finally:
            self._cursor = saved

    @contextmanager
    def before(self, id: str):
        container, i = self._locate(id)
        saved = self._cursor
        self._cursor = [container, i]
        try:
            yield self
        finally:
            self._cursor = saved

    @contextmanager
    def inside(self, id: str):
        """New blocks go at the end of container block `id` (a section, grid or slide)."""
        b = self.block(id)
        target = b.get("children") if isinstance(b.get("children"), list) else b.get("blocks")
        if target is None:
            raise KitError(f"block {id!r} is not a container")
        with self._into(target):
            yield self

    def remove(self, id: str) -> dict:
        container, i = self._locate(id)
        return container.pop(i)

    def update(self, id: str, **changes) -> dict:
        """Change a block's settings in place (e.g. title=..., kind="line", height=420)."""
        b = self.block(id)
        for k, v in changes.items():
            if v is None:
                b.pop(k, None)
            else:
                b[k] = v
        return b

    def set(self, **changes) -> "Page":
        """Change page settings: title, subtitle, theme, badges, meta, width, accent, lang, sources, notes ..."""
        for k, v in changes.items():
            if k in ("blocks", "data", "slides"):
                raise KitError("use the block methods to change content")
            self.spec[k] = v
        return self

    def outline(self) -> str:
        """The block tree with ids, for edits: page.after(id), page.update(id, ...), page.remove(id)."""
        lines: list[str] = []

        def walk(blocks: list, depth: int):
            for b in blocks:
                lines.append("  " * depth + _describe(b, self.spec))
                for key in ("children", "blocks", "right"):
                    if isinstance(b.get(key), list):
                        walk(b[key], depth + 1)
                for t in b.get("tabs") or []:
                    lines.append("  " * (depth + 1) + f"[tab {t['label']!r}]")
                    walk(t["children"], depth + 2)
        if self.mode == "deck":
            for i, s in enumerate(self.spec["slides"], 1):
                lines.append(f"{s['id']} slide {i} ({s.get('layout', 'content')}): {s.get('title', '')!r}")
                walk(s.get("blocks") or [], 1)
                if s.get("right"):
                    lines.append("  [right]")
                    walk(s["right"], 2)
        else:
            walk(self.spec["blocks"], 0)
        head = f"{self.mode} {self.spec.get('title')!r} (theme {self.spec.get('theme')}, lang {self.spec.get('lang')})"
        return "\n".join([head] + lines)

    # ---- output ----
    def html_text(self) -> str:
        from .render import render_html
        return render_html(self.spec, self._custom)

    def save(self, path: Any, *, quiet: bool = False) -> str:
        """Write the page (a folder gets index.html). Prints what the page shows and any warnings; read them."""
        p = Path(path)
        if p.suffix.lower() not in (".html", ".htm"):
            p = p / "index.html"
        p.parent.mkdir(parents=True, exist_ok=True)
        text = self.html_text()
        p.write_text(text, encoding="utf-8")
        if not quiet:
            print(f"Saved {p} ({len(text.encode()) // 1024} KB)")
            print(self.outline())
            if self.warnings:
                print("WARNINGS (fix these before you answer):")
                for w in dict.fromkeys(self.warnings):
                    print("  - " + w)
        return str(p)


def _describe(b: dict, spec: dict) -> str:
    t = b.get("type")
    title = f" {b['title']!r}" if b.get("title") else ""
    data = spec.get("data", {})
    n = lambda name: len(data.get(name, [])) if isinstance(name, str) else 0  # noqa: E731
    if t == "kpis":
        items = "; ".join(f"{k['label']}={_fmt(k.get('value'), k.get('format'), spec.get('currency', 'USD'))}" for k in b["items"])
        return f"{b['id']} kpis: {items}"
    if t == "compare":
        rows = "; ".join(f"{r['label']}: " + " | ".join(_fmt(v, r.get('format'), spec.get('currency', 'USD')) for v in r["values"]) for r in b["rows"])
        return f"{b['id']} compare{title} [{' | '.join(b['columns'])}] {rows}"
    if t == "chart":
        return f"{b['id']} chart {b['kind']}{title}: {n(b['data'])} rows, x={b.get('x')}, y={b.get('y')}" + (f", series={b['series']}" if b.get("series") else "")
    if t == "table":
        cols = [c["field"] for c in (b.get("columns") or [])] or _fields(data.get(b["data"], []))
        return f"{b['id']} table{title}: {n(b['data'])} rows x {len(cols)} columns ({', '.join(cols)})"
    if t in ("map", "globe"):
        parts = [f"{L['type']} {n(L['data'])}" + (f" labelled by {L['label']}" if L.get("label") else "") for L in b["layers"]]
        return f"{b['id']} {t}{title}: " + ", ".join(parts)
    if t == "network":
        return f"{b['id']} network ({b.get('style')}){title}: {n(b['links']['data'])} links"
    if t == "timeline":
        parts = [f"bars {n(x['data'])}" for x in b.get("bars") or []] + [f"milestones {n(x['data'])}" for x in b.get("milestones") or []]
        return f"{b['id']} timeline{title}: " + ", ".join(parts)
    if t in ("text", "callout"):
        txt = re.sub(r"\s+", " ", b.get("md", ""))[:90]
        return f"{b['id']} {t}: {txt}"
    if t == "tabs":
        return f"{b['id']} tabs ({b.get('style')}): " + " | ".join(x["label"] for x in b["tabs"])
    if t == "grid":
        return f"{b['id']} grid {b.get('columns')} columns"
    return f"{b.get('id')} {t}{title}"


def _image_uri(src: Any) -> str:
    if hasattr(src, "savefig"):  # matplotlib figure
        import io
        buf = io.BytesIO()
        src.savefig(buf, format="png", dpi=144, bbox_inches="tight")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    if isinstance(src, (bytes, bytearray)):
        data = bytes(src)
        mime = "image/png" if data[:4] == b"\x89PNG" else "image/svg+xml" if data.lstrip()[:4] in (b"<svg", b"<?xm") else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(data).decode()
    s = str(src)
    if s.startswith("data:"):
        return s
    if re.match(r"https?://", s):
        raise KitError("image: embed the image (path or bytes); the preview cannot load images from other sites")
    p = Path(s)
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml",
            ".gif": "image/gif", ".webp": "image/webp"}.get(p.suffix.lower(), "image/png")
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


# ---------------------------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------------------------

class Deck(Page):
    """A slide deck. Each `with deck.slide(...)` holds blocks; layouts: title, section, content,
    big_number (value + caption), split (blocks left, `with deck.right():` right)."""

    mode = "deck"

    def __init__(self, title: str = "", subtitle: str | None = None, **kw):
        super().__init__(title, subtitle, **kw)
        self.spec["slides"] = []
        self.spec.pop("blocks", None)
        self._targets = []
        self._slide: dict | None = None

    def _roots(self) -> list:
        out = []
        for s in self.spec.get("slides", []):
            out.append(s)
        return out

    def _next_id(self) -> str:
        used = {b.get("id") for b in self._walk()}
        n = len(used) + 1
        while f"b{n}" in used:
            n += 1
        return f"b{n}"

    def _add(self, block: dict) -> dict:
        if not self._targets and self._cursor is None:
            raise KitError("deck: add blocks inside `with deck.slide(...):`")
        return super()._add(block)

    @contextmanager
    def slide(self, title: str | None = None, subtitle: str | None = None, *, layout: str = "content",
              value: Any = None, value_text: str | None = None, format: str | None = None,
              caption: str | None = None, eyebrow: str | None = None, notes: str | None = None, id: str | None = None):
        if layout not in ("title", "section", "content", "big_number", "split"):
            raise KitError("slide layout is title | section | content | big_number | split")
        from .data import _scalar
        s = {k: v for k, v in dict(type="slide", title=title, subtitle=subtitle, layout=layout, value=_scalar(value),
                                   value_text=value_text, format=format, caption=caption, eyebrow=eyebrow,
                                   notes=notes, blocks=[], right=[] if layout == "split" else None).items() if v is not None}
        s["id"] = id or f"s{len(self.spec['slides']) + 1}"
        self.spec["slides"].append(s)
        self._slide = s
        with self._into(s["blocks"]):
            yield s
        self._slide = None

    @contextmanager
    def right(self):
        if not self._slide or self._slide.get("layout") != "split":
            raise KitError("deck.right() is for the right column of a split slide")
        with self._into(self._slide["right"]):
            yield self._slide


# ---------------------------------------------------------------------------------------------
# Loading a saved page for edits
# ---------------------------------------------------------------------------------------------

_SPEC_RE = re.compile(r'<script type="application/json" id="ak-spec">(.*?)</script>', re.S)
_CUSTOM_RE = re.compile(r'<script data-ak-custom="([^"]+)">AK\.custom\[[^\]]+\]=function\(el,AK\)\{\n(.*?)\n\};</script>', re.S)
_CSS_RE = re.compile(r'<style data-ak-custom="([^"]+)">(.*?)</style>', re.S)


def load(path: Any) -> Page:
    """Open a page saved by the kit, to change it and save it again. Its data comes with it."""
    p = Path(path)
    if p.is_dir():
        p = p / "index.html"
    text = p.read_text(encoding="utf-8")
    m = _SPEC_RE.search(text)
    if not m:
        raise KitError(f"{p} was not made with the kit (no page spec); edit its HTML directly instead")
    spec = json.loads(m.group(1).replace("<\\/", "</"))
    cls = Deck if spec.get("mode") == "deck" else Page
    page = cls.__new__(cls)
    page.spec = spec
    page._custom = {}
    page.warnings = []
    page._cursor = None
    page._targets = [] if cls is Deck else [spec["blocks"]]
    if cls is Deck:
        page._slide = None
    for bid, js in _CUSTOM_RE.findall(text):
        page._custom.setdefault(bid, {})["js"] = js.replace("<\\/script", "</script")
    for bid, css in _CSS_RE.findall(text):
        page._custom.setdefault(bid, {})["css"] = css
    return page


def copy_page(page: Page) -> Page:
    clone = copy.deepcopy(page)
    return clone
