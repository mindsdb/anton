"""Artifact kit: build report pages, dashboards, maps and slide decks from data, in the scratchpad.

    from anton.core.artifacts import kit
    page = kit.Page("Title", "The one-line takeaway", lang="en", badges=["Scope"])
    page.kpis([kit.kpi("Revenue", 1250000, "currency_compact", delta=0.08, delta_format="fraction", better="up")])
    with page.grid(2):
        page.chart("line", df, x="month", y="revenue", title="Revenue by month", format="currency")
        page.map(kit.points(sites, label="name", color="type"), title="Sites")
    page.table(df, ["month", kit.col("revenue", "Revenue", "currency")], title="Detail")
    page.save(artifact_path)          # writes index.html; prints the outline and any warnings

The page is one self-contained HTML file: the data is embedded, the design system and the needed
renderers are inlined, and charting/map libraries load from cdn.jsdelivr.net. It adapts to phones,
works in light and dark themes, and every view is pre-tested, so pages render the same every time.
Values are drawn from the data you pass in; nothing is re-typed. `kit.load(path)` reopens a saved
page for edits (`outline()`, `after(id)`, `update(id, ...)`, `remove(id)`).
"""

from .data import profile
from .geocode import geocode, geocode_many
from .page import (Deck, KitError, Page, bars, col, kpi, load, marker, metric, milestones, points, regions,
                   routes, rows)

__all__ = [
    "Page", "Deck", "load", "KitError", "kpi", "metric", "col", "points", "routes", "regions", "bars",
    "milestones", "marker", "rows", "geocode", "geocode_many", "profile",
]
