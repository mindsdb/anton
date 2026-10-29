"""Turn the agent's data (DataFrames, lists of dicts, dicts of lists) into JSON-safe rows."""

from __future__ import annotations

import datetime as _dt
import decimal
import math
import re
from typing import Any


def _scalar(v: Any) -> Any:
    """A JSON-safe value: NaN and NaT become None, numpy scalars become Python ones, dates ISO text."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(v, str):
        return v
    if isinstance(v, decimal.Decimal):
        return float(v)
    # pandas / numpy without importing them
    mod = type(v).__module__
    if mod.startswith("numpy"):
        if hasattr(v, "item"):
            return _scalar(v.item())
    if mod.startswith("pandas"):
        name = type(v).__name__
        if name in ("NaTType", "NAType"):
            return None
        if name == "Timestamp":
            return _date_text(v.to_pydatetime())
        if name == "Timedelta":
            return v.total_seconds() / 86400.0
        if name == "Period":
            return str(v)
    if isinstance(v, _dt.datetime):
        return _date_text(v)
    if isinstance(v, _dt.date):
        return v.isoformat()
    if isinstance(v, (list, tuple)):
        return [_scalar(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _scalar(x) for k, x in v.items()}
    return str(v)


def _date_text(v: _dt.datetime) -> str:
    if v.hour == 0 and v.minute == 0 and v.second == 0 and v.microsecond == 0:
        return v.date().isoformat()
    return v.replace(microsecond=0, tzinfo=None).isoformat(sep=" ")


_DAY_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_WITH_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$")


def _same_shape_times(rows: list[dict]) -> list[dict]:
    """A column of timestamps keeps one shape: when some have a time of day, the midnight ones
    (written as a bare date) get theirs too, so the column reads as times throughout."""
    if not rows:
        return rows
    for f in fields(rows):
        vals = [r.get(f) for r in rows if isinstance(r.get(f), str)]
        if vals and any(_WITH_TIME.match(v) for v in vals) and any(_DAY_ONLY.match(v) for v in vals):
            for r in rows:
                v = r.get(f)
                if isinstance(v, str) and _DAY_ONLY.match(v):
                    r[f] = v + " 00:00:00"
    return rows


def records(data: Any) -> list[dict]:
    """Rows as a list of dicts from a DataFrame, a list of dicts/tuples, or a dict of columns."""
    return _same_shape_times(_records(data))


def _records(data: Any) -> list[dict]:
    if data is None:
        return []
    if hasattr(data, "to_dict") and hasattr(data, "columns"):  # DataFrame
        frame = data
        if any(n is not None for n in (getattr(frame.index, "names", None) or [])):
            frame = frame.reset_index()  # a named index (e.g. after groupby) is a column the page needs
        rows = frame.to_dict(orient="records")
        return [{str(k): _scalar(v) for k, v in r.items()} for r in rows]
    if hasattr(data, "to_dict") and hasattr(data, "index") and not hasattr(data, "columns"):  # Series
        name = getattr(data, "name", None) or "value"
        idx = getattr(data.index, "name", None) or "key"
        return [{str(idx): _scalar(k), str(name): _scalar(v)} for k, v in data.items()]
    if isinstance(data, dict):
        cols = list(data.keys())
        if cols and all(isinstance(data[c], (list, tuple)) for c in cols):
            n = max(len(data[c]) for c in cols)
            return [{str(c): _scalar(data[c][i] if i < len(data[c]) else None) for c in cols} for i in range(n)]
        return [{str(k): _scalar(v) for k, v in data.items()}]
    if isinstance(data, (list, tuple)):
        out = []
        for r in data:
            if isinstance(r, dict):
                out.append({str(k): _scalar(v) for k, v in r.items()})
            elif hasattr(r, "_asdict"):
                out.append({str(k): _scalar(v) for k, v in r._asdict().items()})
            else:
                raise TypeError("rows must be dicts (or a DataFrame); got " + type(r).__name__)
        return out
    raise TypeError("data must be a DataFrame, a list of dicts or a dict of columns; got " + type(data).__name__)


def fields(rows: list[dict]) -> list[str]:
    seen: dict[str, None] = {}
    for r in rows[:500]:
        for k in r:
            seen.setdefault(k, None)
    return list(seen)


def is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def fmt(v: Any, format: str | None = None, currency: str = "USD") -> str:
    """An approximate Python rendering of the page's number formats, for the build summary."""
    if v is None or v == "":
        return "—"
    f = (format or "auto").split(":")[0]
    if not is_number(v):
        return str(v)
    sym = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}.get(currency, currency + " ")
    if f == "percent":
        return f"{v:,.1f}%".replace(".0%", "%")
    if f == "fraction":
        return f"{v * 100:,.1f}%".replace(".0%", "%")
    if f in ("currency_compact", "compact"):
        a = abs(v)
        for div, unit in ((1e9, "B"), (1e6, "M"), (1e4 / 10, "K")):
            if a >= max(div, 1e4 if unit == "K" else div):
                s = f"{v / div:.3g}{unit}"
                return (sym + s) if f == "currency_compact" else s
        return (sym if f == "currency_compact" else "") + f"{v:,.0f}"
    if f == "currency":
        return f"{sym}{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{sym}{v:,.2f}"
    if isinstance(v, float) and not v.is_integer():
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return f"{int(v):,}"


def profile(data: Any, name: str = "data", max_values: int = 12) -> str:
    """A compact profile of a table: row count, then per column its type, distinct values with counts
    (for few-valued columns) or range and total (for numbers) and span (for dates). Printed and returned.
    Use it before computing, to see which statuses, categories and periods the data really holds."""
    from collections import Counter

    rows = records(data)
    lines = [f"{name}: {len(rows)} rows"]
    # Pages show clock times as written; a timezone-aware column says which zone that is.
    zones = {str(c): str(getattr(t, "tz", None)) for c, t in getattr(data, "dtypes", {}).items()
             if getattr(t, "tz", None) is not None} if hasattr(data, "dtypes") else {}
    for f in fields(rows):
        vals = [r.get(f) for r in rows]
        present = [v for v in vals if v is not None and v != ""]
        missing = len(vals) - len(present)
        tail = f", {missing} empty" if missing else ""
        if present and all(is_number(v) for v in present):
            lines.append(f"  {f}: number, min {min(present):,.6g}, max {max(present):,.6g}, sum {sum(present):,.6g}{tail}")
            continue
        texts = [str(v) for v in present]
        if texts and all(len(t) >= 10 and t[:4].isdigit() and t[4] == "-" for t in texts):
            timed = any(_WITH_TIME.match(t) and t[11:16] != "00:00" for t in texts)
            kind, cut = ("datetime (with times of day)", 16) if timed else ("date", 10)
            zone = f", clock times in {zones[f]}" if f in zones else ""
            lines.append(f"  {f}: {kind}, {min(texts)[:cut]} .. {max(texts)[:cut]}, {len(set(texts))} distinct{zone}{tail}")
            continue
        counts = Counter(texts)
        if len(counts) <= max_values:
            shown = ", ".join(f"{k!r} x{n}" for k, n in counts.most_common())
            lines.append(f"  {f}: {len(counts)} values: {shown}{tail}")
        else:
            sample = ", ".join(repr(k) for k, _ in counts.most_common(4))
            lines.append(f"  {f}: {len(counts)} distinct, e.g. {sample}{tail}")
    out = "\n".join(lines)
    print(out)
    return out


def _num(v: float) -> str:
    return f"{v:,.6g}" if abs(v) < 1e6 else f"{v:,.0f}"


def _delta(a: float, b: float) -> str:
    d = b - a
    return f"{_num(a)} -> {_num(b)} ({'+' if d > 0 else ''}{_num(d)})"


_ID_NAME = re.compile(r"(^|[_\s.])(id|code|key|no|number|line|seq|row|rank|index)$|[a-z](Id|ID|Code|Key|Line|Seq)$", re.I)
_RATE_NAME = re.compile(r"pct|percent|rate|ratio|share|util|margin", re.I)
_COORD_NAME = re.compile(r"(^|[_\s.])(lat|lon|lng|latitude|longitude)$|[a-z0-9](Lat|Lon|Lng|Latitude|Longitude)$", re.I)


def _same(a: Any, b: Any) -> bool:
    if is_number(a) and is_number(b):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    return a == b


def diff(before: Any, after: Any, names: tuple[str, str] = ("before", "after"), *, max_values: int = 12,
         max_lines: int = 40) -> str:
    """What changes between two versions of the same data: two scenarios, two periods, a plan with
    and without an option. Pass two tables, or two dicts of tables {name: table} with the same names.
    For each table it prints only what differs, largest changes first: the row count; each amount
    column's total (rates and percentages: mean and max; ids, codes, years and coordinates are not
    added up); for each few-valued column, the count of each value that changed and, for that value,
    the totals of the amount columns (e.g. Status 'Overdue': 3 -> 11 rows, Amount 2,400 -> 9,150); and,
    when rows share an id (or a pair of ids), how many changed in which columns and how many are only
    on one side. Printed and returned. On a large source, diff aggregates from queries (counts and
    totals by status) rather than whole tables."""
    from collections import Counter, defaultdict

    a_tabs = before if isinstance(before, dict) and not hasattr(before, "columns") else {"data": before}
    b_tabs = after if isinstance(after, dict) and not hasattr(after, "columns") else {"data": after}
    lines = [f"diff {names[0]} -> {names[1]}"]
    for name in list(dict.fromkeys(list(a_tabs) + list(b_tabs))):
        ra, rb = records(a_tabs.get(name)), records(b_tabs.get(name))
        both = ra + rb
        cols = list(dict.fromkeys(fields(ra) + fields(rb)))
        out: list[tuple[float, str]] = []  # (size of change, line)
        skipped: list[str] = []
        if len(ra) != len(rb):
            out.append((float("inf"), f"  rows: {_delta(len(ra), len(rb))}"))
        nums = [f for f in cols if any(is_number(r.get(f)) for r in both)
                and all(is_number(r.get(f)) or r.get(f) in (None, "") for r in both)]

        def id_like(f):
            vals = [r.get(f) for r in both if r.get(f) not in (None, "")]
            years = vals and all(isinstance(v, int) and 1900 <= v <= 2100 for v in vals)
            return bool(_ID_NAME.search(f) or _COORD_NAME.search(f) or years)
        amounts = [f for f in nums if not id_like(f)]

        def rate(f):
            vals = [r.get(f) for r in both if is_number(r.get(f))]
            return bool(_RATE_NAME.search(f)) and vals and min(vals) >= 0 and max(vals) <= 100
        for f in amounts:
            va = [r.get(f) for r in ra if is_number(r.get(f))]
            vb = [r.get(f) for r in rb if is_number(r.get(f))]
            if rate(f):
                ma = sum(va) / len(va) if va else 0.0
                mb = sum(vb) / len(vb) if vb else 0.0
                if not math.isclose(ma, mb, rel_tol=1e-9, abs_tol=1e-9) or (va and vb and max(va) != max(vb)):
                    out.append((abs(mb - ma), f"  {f} mean: {_delta(ma, mb)}, max {_num(max(va or [0]))} -> {_num(max(vb or [0]))}"))
                continue
            sa, sb = float(sum(va)), float(sum(vb))
            if not math.isclose(sa, sb, rel_tol=1e-9, abs_tol=1e-9):
                out.append((abs(sb - sa) / (abs(sa) or 1.0), f"  {f} total: {_delta(sa, sb)}"))
        for f in cols:
            if f in nums:
                continue
            ca = Counter(str(r.get(f)) for r in ra if r.get(f) not in (None, ""))
            cb = Counter(str(r.get(f)) for r in rb if r.get(f) not in (None, ""))
            values = set(ca) | set(cb)
            if len(values) > max_values:
                if any(ca[v] != cb[v] for v in values):
                    skipped.append(f"{f} ({len(values)} values)")
                continue
            for v in sorted(values, key=lambda x: -abs(cb[x] - ca[x])):
                if ca[v] == cb[v]:
                    continue
                block = [f"  {f} = {v!r}: rows {_delta(ca[v], cb[v])}"]
                for g in amounts:
                    if rate(g):
                        continue
                    sa = float(sum(r.get(g) or 0 for r in ra if str(r.get(f)) == v))
                    sb = float(sum(r.get(g) or 0 for r in rb if str(r.get(f)) == v))
                    if not math.isclose(sa, sb, rel_tol=1e-9, abs_tol=1e-9):
                        block.append(f"    {g} where {f} = {v!r}: {_delta(sa, sb)}")
                out.append((abs(cb[v] - ca[v]) / max(1, ca[v]) + 1, "\n".join(block)))
        # Rows matched by an id (or a pair of ids): which columns changed, which rows are on one side.
        idcols = [f for f in cols if _ID_NAME.search(f)]
        keys = [(f,) for f in idcols] + [(f, g) for i, f in enumerate(idcols) for g in idcols[i + 1:]] + \
               [(f, g) for f in idcols for g in cols if g not in idcols and g.lower() in ("version", "period")]
        for key in keys:
            ka = [tuple(r.get(k) for k in key) for r in ra]
            kb = [tuple(r.get(k) for k in key) for r in rb]
            if ra and rb and len(set(ka)) == len(ka) and len(set(kb)) == len(kb):
                by_b = dict(zip(kb, rb))
                changed: dict[str, int] = defaultdict(int)
                n_rows = 0
                for k, r in zip(ka, ra):
                    other = by_b.get(k)
                    if other is None:
                        continue
                    diffs = [f for f in cols if f not in key and not _same(r.get(f), other.get(f))]
                    if diffs:
                        n_rows += 1
                        for f in diffs:
                            changed[f] += 1
                only_a, only_b = set(ka) - set(kb), set(kb) - set(ka)
                label = "+".join(key)
                if n_rows:
                    detail = ", ".join(f"{f} {n}" for f, n in sorted(changed.items(), key=lambda x: -x[1])[:8])
                    out.append((0.5, f"  {n_rows} of {len(ra)} rows changed (by {label}): {detail}"))
                for side, ks in ((names[0], only_a), (names[1], only_b)):
                    if ks:
                        some = ", ".join("/".join(str(x) for x in k) for k in sorted(ks, key=str)[:5])
                        out.append((0.5, f"  {len(ks)} rows only in {side} (by {label}): {some}"))
                break
        out.sort(key=lambda x: -x[0])
        shown = [line for _, line in out]
        text_lines = "\n".join(shown).split("\n")
        if len(text_lines) > max_lines:
            skipped.append(f"{len(text_lines) - max_lines} more lines")
            text_lines = text_lines[:max_lines]
        lines.append(f"{name}:" + ("" if shown else " no changes"))
        lines += [ln for ln in text_lines if ln]
        if skipped:
            lines.append("  not shown: " + "; ".join(skipped))
    text = "\n".join(lines)
    print(text)
    return text
