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
           "signed", "days", "hours", "date", "date_short", "date_iso", "month", "datetime", "datetime_short",
           "time", "weekday")


_COORD_FIELD = re.compile(r"(^|[_\s.])(lat|lon|lng|latitude|longitude)$|[a-z0-9](Lat|Lon|Lng|Latitude|Longitude)$", re.I)


def _is_coord_field(name: str) -> bool:
    return bool(_COORD_FIELD.search(str(name)))


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

def _unit_clash(unit: str | None, format: str | None) -> str | None:
    """A unit the format already writes ("%" with percent, "$" with currency) is dropped."""
    f = str(format or "").split(":")[0]
    if unit and ((f in ("percent", "fraction") and unit.strip() == "%") or
                 (f.startswith("currency") and unit.strip() in ("$", "€", "£", "¥", "USD", "EUR", "GBP", "JPY"))):
        return None
    return unit


def kpi(label: str, value: Any, format: str | None = None, *, sub: str | None = None, delta: Any = None,
        delta_format: str | None = None, delta_label: str | None = None, better: str | None = None,
        tone: str | None = None, spark: list | None = None, color_value: bool = False, unit: str | None = None) -> dict:
    """One KPI card. `better` ("up"/"down") colours the delta; `tone` marks the card good/bad/warn/accent;
    `unit` ("km", "kg", "hours") follows the value."""
    _check_format(format, f"kpi {label!r}")
    from .data import _scalar
    card = {k: v for k, v in dict(label=label, format=format, sub=sub, delta=_scalar(delta),
                                  delta_format=delta_format, delta_label=delta_label, better=better, tone=tone,
                                  spark=[_scalar(x) for x in spark] if spark is not None else None,
                                  color_value=color_value or None, unit=_unit_clash(unit, format)).items() if v is not None}
    # A missing figure (None, NaN) stays on the card as "—"; the page warns about it on save.
    return {"label": label, "value": _scalar(value), **card}


def metric(label: str, values: list, format: str | None = None, *, better: str | None = None, note: str | None = None,
           diff_format: str | None = None, show_pct: bool = False, tones: list | None = None,
           unit: str | None = None) -> dict:
    """One row of `compare`: a metric's value in each column, in column order; `unit` follows its label."""
    _check_format(format, f"metric {label!r}")
    from .data import _scalar
    return {k: v for k, v in dict(label=label, values=[_scalar(x) for x in values], format=format, better=better,
                                   note=note, diff_format=diff_format, show_pct=show_pct or None,
                                   tones=tones, unit=_unit_clash(unit, format)).items() if v is not None}


def col(field: str, label: str | None = None, format: str | None = None, *, pill: bool = False,
        tones: dict | None = None, bar: Any = False, align: str | None = None, wrap: bool = False,
        unit: str | None = None) -> dict:
    """A table column. `pill` shows the value as a status pill (`tones` maps values to good/bad/warn/info);
    `bar` adds an inline bar for numbers; `wrap` gives long text room; `unit` goes in the header."""
    _check_format(format, f"column {field!r}")
    return {k: v for k, v in dict(field=field, label=label, format=format, pill=pill or None, tones=tones,
                                   bar=bar or None, align=align, wrap=wrap or None,
                                   unit=_unit_clash(unit, format)).items() if v is not None}


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
    """A layer of great-circle routes from one point to another. With `via_*` (e.g. where a vehicle is
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
               tooltip: list | None = None, show_labels: bool | None = None, detail_fields: list | None = None,
               labels: dict | None = None, formats: dict | None = None) -> dict:
    """A timeline layer of dated events per row. `label` is written beside each milestone where it fits
    (show_labels=False: only in its tooltip); hover/tap shows it with the date and `tooltip` fields."""
    return {"data": data, "row": row, "date": date, "label": label, "color": color, "colors": colors, "tone": tone,
            "name": name, "shape": shape, "tooltip": tooltip, "show_labels": show_labels,
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
        self.checks: list[str] = []  # things to confirm, not errors

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
            self.warnings.append(f"{where}: the data is empty, so this block shows nothing; a filter or join may have "
                                 "matched nothing (check its values against the data).")
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
            if not isinstance(it, dict) or "label" not in it:
                raise KitError("kpis: each item needs a label and a value (use kit.kpi(label, value, format))")
            if it.get("value") is None:
                self.warnings.append(f"kpi {it['label']!r} has no value (None or NaN), shown as '—': compute it, "
                                     "or leave the card out and say why.")
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
              horizontal: bool | None = None, sort: str | None = None, show_values: bool | None = None,
              ref_lines: list | None = None, threshold: dict | None = None, point_tones: Any = None,
              y2: str | list | None = None, y2_format: str | None = None, y2_kind: str | None = None,
              y2_label: str | None = None, colors: dict | None = None, height: int | None = None,
              y_min: float | None = None, y_max: float | None = None, x_label: str | None = None,
              y_label: str | None = None, smooth: bool | None = None, legend: bool | None = None,
              link: str | None = None, key: str | None = None, size: str | None = None, label: str | None = None,
              value: str | None = None, y_category: str | None = None, totals: list | None = None,
              dashed: Any = None, area: bool | None = None, show_data: bool | None = None,
              center_label: str | None = None, unit: str | None = None, y2_unit: str | None = None,
              id: str | None = None) -> dict:
        """A chart. `unit` ("km", "kg", "hours") labels the value axis, values and tooltips; a y2 in
        another `y2_unit` gets its own axis.
        kind: bar | hbar | line | area | combo (bars y + line y2) | scatter | pie | donut |
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
                          "labels": labels, "formats": formats, "stack": stack or None, "horizontal": horizontal,
                          "sort": sort, "show_values": show_values, "ref_lines": ref_lines, "threshold": threshold,
                          "point_tones": point_tones, "y2": y2, "y2_format": y2_format, "y2_kind": y2_kind,
                          "y2_label": y2_label, "colors": colors, "height": height, "y_min": y_min, "y_max": y_max,
                          "x_label": x_label, "y_label": y_label, "smooth": smooth, "legend": legend, "link": link,
                          "key": key, "size": size, "label": label, "value": value, "y_category": y_category,
                          "totals": totals, "dashed": dashed, "area": area, "show_data": show_data,
                          "center_label": center_label, "unit": _unit_clash(unit, format),
                          "y2_unit": _unit_clash(y2_unit, y2_format), "id": id})

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
        rws0 = records(data) if not (isinstance(data, str) and data in self.spec["data"]) else self.spec["data"][data]
        cols = [c if isinstance(c, dict) else {"field": c} for c in (columns or [])]
        wanted = {c["field"] for c in cols} | set(detail_fields or [])
        if cols and any(_is_coord_field(k) and k not in wanted for k in _fields(rws0)):
            # Coordinates are for maps: a table embeds only those it shows.
            rws0 = [{k: v for k, v in r.items() if k in wanted or not _is_coord_field(k)} for r in rws0]
        name, rws = self._dataset(rws0, f"table {title!r}")
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
                # One shape for every layer whatever the source columns are called: a point's
                # {lat, lon}, a route's from/via/to objects. Coordinate columns the layer does not
                # draw are dropped (a smaller page).
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
                    rest = {k: v for k, v in r.items() if k not in {s for _, s in coords} and not _is_coord_field(k)}
                    if L["type"] == "points":
                        fixed.append({"lat": vals["lat"], "lon": vals["lon"], **rest})
                    else:
                        ends = {e: {"lat": vals[e + "_lat"], "lon": vals[e + "_lon"]}
                                for e in ("from", "via", "to") if e + "_lat" in vals}
                        fixed.append({**ends, **rest})
                if missing:
                    self.warnings.append(f"{where}: {len(missing)} row(s) without coordinates are not drawn: {', '.join(missing[:8])}")
                if L["type"] == "points" and len(fixed) >= 3:
                    # Different items on one spot draw on top of each other: either they really share a
                    # place, or shared coordinates (a destination, a site) were joined instead of each
                    # row's own.
                    from collections import Counter
                    spot, n = Counter((p["lat"], p["lon"]) for p in fixed).most_common(1)[0]
                    ident = label  # the layer's label field, else its key
                    names = {p.get(ident) for p in fixed if (p["lat"], p["lon"]) == spot} if ident else set()
                    if n >= 3 and n > len(fixed) / 2 and (not ident or len(names) > 1):
                        self.warnings.append(f"{where}: {n} of {len(fixed)} points are at the same spot {spot} and hide "
                                             "each other. If they really share a place, show one point per place with "
                                             "the count (and the items in its tooltip or a table); otherwise use each "
                                             "row's own coordinates.")
                if L["type"] == "points":
                    L["lat"], L["lon"] = "lat", "lon"
                else:
                    for e in ("from", "via", "to"):
                        L.pop(e + "_lat", None)
                        L.pop(e + "_lon", None)
                    L["ends"] = [e for e in ("from", "via", "to") if any(k.startswith(e + "_") for k, _ in coords)]
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
                controls: bool | None = None, labels: dict | None = None, formats: dict | None = None,
                id: str | None = None) -> dict:
        """A network diagram from link rows (source, target, value). `nodes` (optional) gives node rows with
        `id_field`, `label`, `group`; `tiers` orders groups left to right (e.g. ["Source","Hub","Destination"]);
        a tier may hold several groups (["Supplier", "Subcontractor"]).
        style "graph" (boxes and arrows, value labels on links) or "sankey" (bands sized by value).
        `link_color` colours links by a field (e.g. link type) with `link_colors` {value: colour/tone};
        `dashed` True or a field marking dashed links; `link_label` a field to label links with (False: none).
        Clicking (or Enter on) a node highlights its links; `controls=True` adds a row of buttons that do the
        same, for pages where highlighting is a main use."""
        if style not in ("graph", "sankey"):
            raise KitError("network style is 'graph' or 'sankey'")
        _check_format(value_format, f"network {title!r}")
        lname, lrows = self._dataset(links, f"network {title!r} links")
        _need(lrows, [source, target, value, link_color, link_label if isinstance(link_label, str) else None]
              + list(tooltip or []) + ([dashed] if isinstance(dashed, str) else []), f"network {title!r} links")
        if value:
            def numeric(v):
                try:
                    float(str(v).replace(",", ""))
                    return True
                except ValueError:
                    return False
            text = [r.get(value) for r in lrows if r.get(value) is not None and not is_number(r.get(value))
                    and not numeric(r.get(value))]
            if text:
                raise KitError(f"network {title!r}: value={value!r} holds text (e.g. {str(text[0])[:40]!r}); value is "
                               f"each link's number (its width and label). For text on the links use "
                               f"link_label={value!r} with a numeric value field.")
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
        if tiers:
            # Each tier is a group or a list of groups sharing a column.
            tiers = [[str(g) for g in t] if isinstance(t, (list, tuple, set)) else [str(t)]
                     for t in ([tiers] if isinstance(tiers, str) else tiers)]
            have = {str(r.get(group)) for r in nrows if r.get(group) is not None} if nodes is not None else set()
            named = {g for t in tiers for g in t}
            if have and not have & named:
                self.warnings.append(f"network {title!r}: tiers name groups no node has ({', '.join(sorted(named)[:6])}); "
                                     f"the node groups are {', '.join(sorted(have)[:8])}.")
            elif have - named:
                self.warnings.append(f"network {title!r}: groups not in tiers go in a last column: "
                                     f"{', '.join(sorted(have - named)[:8])}.")
        lspec = {"data": lname, "source": source, "target": target, "value": value, "label": link_label,
                 "color": link_color, "colors": link_colors, "dashed": dashed, "tooltip": tooltip,
                 "labels": labels, "formats": formats}
        return self._add({"type": "network", "style": style, "nodes": {k: v for k, v in nspec.items() if v is not None},
                          "links": {k: v for k, v in lspec.items() if v is not None}, "tiers": tiers, "title": title,
                          "subtitle": subtitle, "note": note, "value_format": value_format, "link": link,
                          "highlight": highlight, "height": height, "arrows": None if arrows else False,
                          "controls": controls,
                          "show_totals": None if show_totals else False, "labels": labels, "id": id})

    def flow(self, links: Any, **kw) -> dict:
        """A sankey: network(..., style="sankey")."""
        kw.setdefault("style", "sankey")
        return self.network(links, **kw)

    def timeline(self, *, bars: list | dict | None = None, milestones: list | dict | None = None,
                 markers: list | None = None, rows: dict | None = None, title: str | None = None,
                 subtitle: str | None = None, note: str | None = None, link: str | None = None,
                 range: list | None = None, row_height: int | None = None, detail: bool = True,
                 legend: bool | None = None, milestone_list: bool | None = None, id: str | None = None) -> dict:
        """A timeline / Gantt chart. bars: kit.bars(...) layers (one lane each, so plans compare);
        milestones: kit.milestones(...) layers; markers: kit.marker(date, label, tone); rows: kit.rows(...).
        Milestones are also listed as text under the chart: collapsed by default, open with
        milestone_list=True (when the request asks to see every date), none with milestone_list=False."""
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
                          "range": range, "row_height": row_height, "detail": None if detail else False, "list": milestone_list,
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
        if re.search(r"\bfetch\s*\(|XMLHttpRequest|WebSocket\s*\(|EventSource\s*\(", (js or "") + (html or "")):
            raise KitError("page.html: a page must not load data when it is viewed (the preview cannot reach any "
                           "service). Fetch in the scratchpad and embed: page.dataset(rows, 'name') + AK.data('name').")
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
    def beside(self, id: str, *, widths: list | None = None):
        """New blocks go next to block `id`: it and they share a row (a grid), the block itself
        unchanged. On phones they stack."""
        container, i = self._locate(id)
        block = container[i]
        grid = {"type": "grid", "columns": 2, "children": [block], "id": self._next_id()}
        if widths:
            grid["widths"] = widths
        container[i] = grid
        with self._into(grid["children"]):
            yield self
        grid["columns"] = max(1, len(grid["children"]))
        if widths and len(widths) != grid["columns"]:
            grid.pop("widths")

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
    def _check_empty(self) -> None:
        """Warn about a chart whose values are all zero, or a network whose links are mostly zero: almost
        always a filter or join that matched nothing (empty data is warned about when it is added)."""
        data = self.spec.get("data", {})
        for b in self._walk():
            t, name = b.get("type"), b.get("data")
            where = f"{b.get('id')} {t} {b.get('title')!r}"
            if t == "chart" and isinstance(name, str) and data.get(name):
                ys = [b["y"]] if isinstance(b.get("y"), str) else list(b.get("y") or [])
                vals = [r.get(f) for r in data[name] for f in ys if is_number(r.get(f))]
                if vals and not any(vals):
                    self.warnings.append(f"{where}: every value is 0; check the filter or join that produced it.")
            if t == "network" and isinstance((b.get("links") or {}).get("data"), str):
                rows, vf = data.get(b["links"]["data"], []), b["links"].get("value")
                if rows and vf and all(is_number(r.get(vf)) for r in rows):
                    zero = [r for r in rows if not r.get(vf)]
                    if len(zero) > len(rows) / 2:
                        self.warnings.append(f"{where}: {len(zero)} of {len(rows)} links have a value of 0. If they "
                                             "really carry nothing, leave them out or say so on the page; otherwise "
                                             "check the filter or join that produced them.")

    def _check_percents(self) -> None:
        """Warn when values shown as `percent` (percent units, 12.5 -> 12.5%) look like fractions."""
        data = self.spec.get("data", {})

        def looks_fractional(vals):
            nums = [v for v in vals if is_number(v)]
            return len(nums) >= 1 and all(0 <= v <= 1 for v in nums) and any(0 < v < 1 for v in nums)

        def check(where, fmt, vals):
            if fmt and str(fmt).split(":")[0] == "percent" and looks_fractional(vals):
                self.warnings.append(f"{where}: values are between 0 and 1 but the format is 'percent' (which "
                                     "expects 12.5 for 12.5%); use format='fraction' or multiply by 100.")
        for b in self._walk():
            t = b.get("type")
            if t == "kpis":
                for k in b["items"]:
                    check(f"kpi {k.get('label')!r}", k.get("format"), [k.get("value")])
            elif t == "chart":
                rows = data.get(b["data"], [])
                for f in ([b["y"]] if isinstance(b.get("y"), str) else list(b.get("y") or [])):
                    check(f"chart {b.get('title')!r}", (b.get("formats") or {}).get(f) or b.get("format")
                          or self.spec.get("formats", {}).get(f), [r.get(f) for r in rows])
            elif t == "table":
                rows = data.get(b["data"], [])
                for c in b.get("columns") or []:
                    check(f"table {b.get('title')!r} column {c['field']!r}", c.get("format")
                          or (b.get("formats") or {}).get(c["field"]) or self.spec.get("formats", {}).get(c["field"]),
                          [r.get(c["field"]) for r in rows])
            elif t == "compare":
                for r in b["rows"]:
                    check(f"compare row {r.get('label')!r}", r.get("format"), r.get("values") or [])

    # Numbers as the page's language writes them: "1,570,600.5" (en) or "1.570.600,5" / "1 570 600,5".
    _NUM_EN = re.compile(r"(?<![\w.\-/])([$€£¥]?)\s?(\d{1,3}(?:[,\u202f\u00a0]\d{3})+|\d+)(?:\.(\d+))?\s?(%|[KkMmBb]\b|bn\b)?(?![\w\-/])")
    _NUM_EU = re.compile(r"(?<![\w,\-/])([$€£¥]?)\s?(\d{1,3}(?:[. \u202f\u00a0]\d{3})+|\d+)(?:,(\d+))?\s?(%|[KkMmBb]\b|Mio\b|Md\b)?(?![\w\-/])")
    _DATES_IN_TEXT = re.compile(
        r"\d{4}-\d{2}(-\d{2})?([T ]\d{2}:\d{2}(:\d{2})?)?|\d{1,2}:\d{2}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?!\d)(?:,\s*\d{4})?|"
        r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?(?:\s+\d{4})?|\b(?:19|20)\d{2}\b|\b[QH][1-4]\b|\bFY\s?\d{2,4}\b", re.I)
    # A span of time ("13 weeks", "24-hour", "7 días") describes a period, not a figure.
    _SPANS_IN_TEXT = re.compile(
        r"\b\d+(?:[.,]\d+)?[\s\-]*(?:\w+\s+)?(?:hours?|hourly|hrs?|h|minutes?|mins?|days?|daily|weeks?|weekly|months?|monthly|years?|quarters?|"
        r"horas?|d[ií]as?|semanas?|meses|mes|a[ñn]os?|heures?|jours?|semaines?|mois|ans?|ann[ée]es?|"
        r"stunden?|tage?n?|wochen?|monate?n?|jahre?n?|ore|giorni|settimane|mesi|anni)\b", re.I)

    def _check_prose_numbers(self) -> None:
        """Warn about numbers the page's words state that match nothing among its figures and data
        (a typo, a stale or invented figure, or one computed and never shown), so they get checked."""
        data = self.spec.get("data", {})
        known: list[float] = []
        for rows in data.values():
            known.append(float(len(rows)))
            for f in _fields(rows):
                vals = [r.get(f) for r in rows if is_number(r.get(f))]
                if vals:
                    total = float(sum(vals))
                    known += [float(v) for v in vals] + [total, total / len(vals), max(vals) - min(vals)]
                    if total:
                        known += [float(v) / total * 100 for v in vals]  # each value's share of the total
                else:
                    from collections import Counter
                    counts = Counter(str(r.get(f)) for r in rows if r.get(f) not in (None, ""))
                    known += [float(c) for c in counts.values()] + [float(len(counts))]
                    known += [c / len(rows) * 100 for c in counts.values()]  # each category's share of rows
        texts: list[str] = [str(self.spec.get(k) or "") for k in ("title", "subtitle")]
        texts += [str(x) for x in (self.spec.get("notes") or []) + (self.spec.get("meta") or [])]
        kpi_values: list[float] = []
        for b in self._walk():
            t = b.get("type")
            if t == "chart":
                for line in ([b["threshold"]] if isinstance(b.get("threshold"), dict) else []) + list(b.get("ref_lines") or []):
                    if isinstance(line, dict) and is_number(line.get("value")):
                        known.append(float(line["value"]))
            if t == "kpis":
                for k in b["items"]:
                    if is_number(k.get("value")):
                        kpi_values.append(float(k["value"]))
                    for key in ("value", "delta"):
                        if is_number(k.get(key)):
                            known.append(float(k[key]))
                    if isinstance(k.get("value"), str):
                        texts.append(k["value"])
                    texts.append(str(k.get("sub") or ""))
            elif t == "compare":
                for r in b.get("rows") or []:
                    vals = [float(v) for v in r.get("values") or [] if is_number(v)]
                    known += vals + [b2 - a for a in vals for b2 in vals] + [(b2 - a) / a * 100 for a in vals for b2 in vals if a]
            if t in ("text", "callout"):
                texts.append(str(b.get("md") or ""))
            elif t == "list":
                texts += [str(i) for i in b.get("items") or []]
            texts += [str(b.get(k) or "") for k in ("title", "subtitle", "note")]
        # Changes between headline figures ("saves 13 lines", "falls 32%")
        known += [b2 - a for a in kpi_values for b2 in kpi_values] + [(b2 - a) / a * 100 for a in kpi_values for b2 in kpi_values if a]
        known += [abs(v) for v in known]
        unsupported: list[tuple[str, str]] = []

        def parse(m):
            whole, dec, suffix = m.group(2), m.group(3) or "", (m.group(4) or "").lower()
            value = float(re.sub(r"[,. \u202f\u00a0]", "", whole) + ("." + dec if dec else ""))
            mult = {"k": 1e3, "m": 1e6, "mio": 1e6, "b": 1e9, "bn": 1e9, "md": 1e9}.get(suffix, 1.0)
            return value, dec, suffix, mult

        def supported(value, dec, suffix, mult):
            half = 0.5 * 10 ** -len(dec) * mult  # what the text's rounding allows
            targets = [(value * mult, half)] + ([(value / 100, half / 100)] if suffix == "%" else [])
            return any(abs(t - k) <= max(tol, abs(k) * 0.002) for t, tol in targets for k in known)

        # The page's language decides how numbers are written; English-style numbers on another
        # page (and the reverse) are read both ways before they count as unsupported.
        eu = str(self.spec.get("lang") or "en")[:2] != "en"
        first, other = (self._NUM_EU, self._NUM_EN) if eu else (self._NUM_EN, self._NUM_EU)
        for text in texts:
            blank = lambda m: " " * len(m.group(0))  # noqa: E731  (keeps positions, to quote the sentence)
            clean = self._SPANS_IN_TEXT.sub(blank, self._DATES_IN_TEXT.sub(blank, text))
            for m in first.finditer(clean):
                value, dec, suffix, mult = parse(m)
                if value * mult < 10 and not dec:
                    continue  # small counts read as words ("3 sites") more than as figures
                if supported(value, dec, suffix, mult):
                    continue
                alt = [parse(a) for a in other.finditer(clean) if a.start() <= m.start() and a.end() >= m.end() - 1]
                if any(supported(*a) for a in alt):
                    continue
                # The sentence it stands in, so it can be checked where it is.
                dot, line = text.rfind(". ", 0, m.start()), text.rfind("\n", 0, m.start())
                start = max(dot + 2 if dot >= 0 else 0, line + 1 if line >= 0 else 0)
                end = min([i for i in (text.find(". ", m.end()), text.find("\n", m.end())) if i >= 0] or [len(text)])
                unsupported.append((m.group(0).strip(), re.sub(r"\*\*|`", "", text[start:end + 1]).strip()[:160]))
        for num, sentence in dict.fromkeys(unsupported):
            self.checks.append(f"{num} in \"{sentence}\"")

    def _check_field_names_in_text(self) -> None:
        """Warn when the page's words use a data field's raw name (signup_date, order.total) instead of
        words a reader knows. A CamelCase name counts only when the page gives that field a label (so
        brand-like names such as YouTube stay alone); names in backticks are meant as written."""
        data = self.spec.get("data", {})
        labels = dict(self.spec.get("labels") or {})
        for b in self._walk():
            labels.update(b.get("labels") or {})
            for c in b.get("columns") or [] if b.get("type") == "table" else []:  # a grid's columns is a count
                if isinstance(c, dict) and c.get("label"):
                    labels.setdefault(c["field"], c["label"])
        names = {}
        for f in {f for rows in data.values() for f in _fields(rows)}:
            if len(f) < 5:
                continue
            if "_" in f or "." in f:
                names[f] = labels.get(f)
            elif re.search(r"[a-z][A-Z]", f) and f in labels:
                names[f] = labels[f]
        if not names:
            return
        texts = [str(self.spec.get("subtitle") or "")]
        for b in self._walk():
            if b.get("type") in ("text", "callout"):
                texts.append(str(b.get("md") or ""))
            elif b.get("type") == "list":
                texts += [str(i) for i in b.get("items") or []]
        texts = [re.sub(r"`[^`]*`", " ", t) for t in texts]
        found = sorted({n for n in names for t in texts if re.search(r"(?<![\w.])" + re.escape(n) + r"(?![\w])", t)})
        if found:
            said = ", ".join(f"{n} (\"{names[n]}\")" if names[n] else n for n in found[:6])
            self.warnings.append(f"the page's text uses data field names: {said}; write them as words in the page's "
                                 "language.")

    def _check_details(self) -> None:
        """On a page in another language, row details show fields the table does not under their own
        names: warn about those without a label, so they get one (or detail_fields picks the fields)."""
        lang = str(self.spec.get("lang") or "en")[:2]
        if lang == "en":
            return
        data, page_labels = self.spec.get("data", {}), self.spec.get("labels") or {}
        for b in self._walk():
            if b.get("type") != "table" or b.get("detail") in (None, False) or b.get("detail_fields"):
                continue
            rows = data.get(b["data"], [])
            shown = {c["field"] for c in b.get("columns") or []}
            labels = {**page_labels, **(b.get("labels") or {})}
            bare = [f for f in _fields(rows) if f not in shown and f not in labels and not _is_coord_field(f)]
            if bare:
                self.warnings.append(f"table {b.get('title')!r}: its row detail also shows {', '.join(bare[:8])} under these "
                                     f"field names on a '{lang}' page; give them labels (labels={{field: label}}, "
                                     "values translated in pandas) or choose detail_fields=[...].")

    def _check_times(self) -> None:
        """Warn when a date-only format would hide the times of day the values have."""
        data = self.spec.get("data", {})
        date_only = ("date", "date_short", "date_iso", "month", "weekday")
        timed = re.compile(r"^\d{4}-\d{2}-\d{2}[T ](\d{2}):(\d{2})")

        def check(where, fmt, vals):
            if not fmt or str(fmt).split(":")[0] not in date_only:
                return
            for v in vals:
                m = timed.match(str(v)) if isinstance(v, str) else None
                if m and (m.group(1) != "00" or m.group(2) != "00"):
                    self.warnings.append(f"{where}: the values have times of day but format={fmt!r} shows only the "
                                         "date; use 'datetime', 'datetime_short' or 'time' (or no format).")
                    return
        for b in self._walk():
            t = b.get("type")
            if t == "kpis":
                for k in b["items"]:
                    check(f"kpi {k.get('label')!r}", k.get("format"), [k.get("value")])
            elif t == "chart":
                rows = data.get(b["data"], [])
                check(f"chart {b.get('title')!r} x", b.get("x_format") or (b.get("formats") or {}).get(b.get("x"))
                      or self.spec.get("formats", {}).get(b.get("x")), [r.get(b.get("x")) for r in rows])
            elif t == "table":
                rows = data.get(b["data"], [])
                for c in b.get("columns") or []:
                    check(f"table {b.get('title')!r} column {c['field']!r}", c.get("format")
                          or (b.get("formats") or {}).get(c["field"]) or self.spec.get("formats", {}).get(c["field"]),
                          [r.get(c["field"]) for r in rows])

    def _autolink(self) -> None:
        """Tables and maps, globes or timelines that show the same records (the same ids) are linked:
        clicking a row highlights that record in the other views. Explicit links are left alone."""
        data = self.spec.get("data", {})

        def ids(rows, field):
            vals = [r.get(field) for r in rows]
            return {str(v) for v in vals if isinstance(v, str) and v.strip()}

        tables = [b for b in self._walk() if b.get("type") == "table" and not b.get("link")]
        views = []
        for b in self._walk():
            if b.get("type") in ("map", "globe"):
                for L in b.get("layers") or []:
                    if L.get("type") in ("points", "routes") and not L.get("link"):
                        views.append(("layer", L, data.get(L["data"], [])))
            elif b.get("type") == "timeline" and not b.get("link"):
                rows = [r for layer in (b.get("bars") or []) + (b.get("milestones") or []) for r in data.get(layer["data"], [])]
                field = next((layer["row"] for layer in (b.get("bars") or []) + (b.get("milestones") or [])), None)
                if field:
                    views.append(("timeline", b, [{"__row": r.get(field)} for r in rows]))
        n = 0
        for t in tables:
            rows = data.get(t["data"], [])
            if not rows:
                continue
            fields = t.get("key") and [t["key"]] or [c["field"] for c in (t.get("columns") or [])] or _fields(rows)
            for f in fields:
                keys = ids(rows, f)
                if len(keys) < 2 or len(keys) != len(rows):
                    continue
                matched = []
                for kind, target, vrows in views:
                    cand = ["__row"] if kind == "timeline" else _fields(vrows)
                    for g in cand:
                        vals = ids(vrows, g)
                        if vals and len(vals & keys) >= max(2, 0.8 * len(vals)):
                            matched.append((kind, target, g))
                            break
                if matched:
                    n += 1
                    name = f"auto{n}"
                    t["key"], t["link"] = f, name
                    for kind, target, g in matched:
                        if kind == "timeline":
                            target["link"] = name
                        else:
                            target["key"], target["link"] = g, name
                    views = [v for v in views if all(v[1] is not m[1] for m in matched)]
                    break

    def html_text(self) -> str:
        from .render import render_html
        self._autolink()
        # The checks below describe the page as it is now: a page saved again after a fix drops what
        # the last save found (warnings from building blocks stay).
        last = getattr(self, "_save_warnings", [])
        self.warnings = [w for w in self.warnings if w not in last]
        start, self.checks = len(self.warnings), []
        self._check_empty()
        self._check_percents()
        self._check_times()
        self._check_details()
        self._check_field_names_in_text()
        self._check_prose_numbers()
        self._save_warnings = self.warnings[start:]
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
            if self.checks:
                print("CHECK (not an error): numbers in the text that no figure or data on the page shows. Confirm "
                      "each against the value your code computed and change the text only where it differs; "
                      "numbers the user gave (a target, a limit) are fine:")
                for c in dict.fromkeys(self.checks):
                    print("  - " + c)
        return str(p)


def _describe(b: dict, spec: dict) -> str:
    t = b.get("type")
    title = f" {b['title']!r}" if b.get("title") else ""
    data = spec.get("data", {})
    n = lambda name: len(data.get(name, [])) if isinstance(name, str) else 0  # noqa: E731
    if t == "kpis":
        items = "; ".join(f"{k['label']}={_fmt(k.get('value'), k.get('format'), spec.get('currency', 'USD'))}"
                          + (f" {k['unit']}" if k.get("unit") else "") for k in b["items"])
        return f"{b['id']} kpis: {items}"
    if t == "compare":
        rows = "; ".join(f"{r['label']}: " + " | ".join(_fmt(v, r.get('format'), spec.get('currency', 'USD')) for v in r["values"]) for r in b["rows"])
        return f"{b['id']} compare{title} [{' | '.join(b['columns'])}] {rows}"
    if t == "chart":
        rows = data.get(b["data"], [])
        stats = []
        for f in ([b["y"]] if isinstance(b.get("y"), str) else list(b.get("y") or [])):
            vals = [r.get(f) for r in rows if is_number(r.get(f))]
            if vals:
                fm = (b.get("formats") or {}).get(f) or b.get("format")
                cur = spec.get("currency", "USD")
                # Rates and shares do not add up: their mean says more than a sum.
                rate = str(fm or "").split(":")[0] in ("percent", "fraction")
                agg = f"mean {_fmt(sum(vals) / len(vals), fm, cur)}" if rate else f"sum {_fmt(sum(vals), fm, cur)}"
                stats.append(f"{f}: {agg}, min {_fmt(min(vals), fm, cur)}, max {_fmt(max(vals), fm, cur)}")
        return (f"{b['id']} chart {b['kind']}{title}: {n(b['data'])} rows, x={b.get('x')}, y={b.get('y')}"
                + (f", series={b['series']}" if b.get("series") else "") + (" [" + "; ".join(stats) + "]" if stats else ""))
    if t == "table":
        rows = data.get(b["data"], [])
        cols = [c["field"] for c in (b.get("columns") or [])] or _fields(rows)
        first = ("; first row: " + ", ".join(f"{c}={rows[0].get(c)!r}" for c in cols[:8])) if rows else ""
        return f"{b['id']} table{title}: {n(b['data'])} rows x {len(cols)} columns ({', '.join(cols)}){first}"
    if t in ("map", "globe"):
        parts = []
        for L in b["layers"]:
            rows = data.get(L["data"], [])
            part = f"{L['type']} {len(rows)}"
            if L.get("label") and rows:
                sample = ", ".join(str(r.get(L["label"])) + (f" ({r.get(L['sublabel'])})" if L.get("sublabel") else "") for r in rows[:6])
                part += f" labelled {sample}{' ...' if len(rows) > 6 else ''}"
            parts.append(part)
        return f"{b['id']} {t}{title}: " + "; ".join(parts)
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
    page.checks = []
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
