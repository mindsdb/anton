"""Versioned composition contract. New visuals can use a custom HTML block."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, localcontext
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .views import View
from .labels import Label, LabelText
from .data import BinaryOp, DataError, Evidence, Model, Name, Query, calculate, execute, number


class Format(Model):
    decimals: int | None = Field(default=None, ge=0, le=18, description='Omit for automatic precision, up to six fractional places; set explicitly for fixed precision.')
    prefix: str = Field(default="", max_length=24)
    suffix: str = Field(default="", max_length=24)
    percent: Literal['none', 'ratio', 'points'] = Field(default='none', description='ratio scales a fraction by 100; points preserves a value already in percentage points; none is not a percentage.')
    missing: str = "Not available"

    @model_validator(mode='after')
    def one_percentage_symbol(self):
        if self.percent != 'none' and '%' in self.prefix + self.suffix:
            raise ValueError('Percentage formats already append %; remove it from prefix/suffix')
        return self

    def display(self, value) -> str:
        if value is None:
            return self.missing
        if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
            value = number(value)
            with localcontext() as context:
                context.prec = 4096
                value *= 100 if self.percent == 'ratio' else 1
                if self.decimals is None and value and abs(value) < Decimal('0.000001'):
                    mantissa, exponent = f'{value:.5E}'.split('E')
                    shown = mantissa.rstrip('0').rstrip('.') + 'e' + str(int(exponent))
                    return f"{self.prefix}{shown}{'%' if self.percent != 'none' else ''}{self.suffix}"
                places = self.decimals if self.decimals is not None else min(6, max(0, -value.normalize().as_tuple().exponent))
                value = value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
            return f"{self.prefix}{value:,.{places}f}{'%' if self.percent != 'none' else ''}{self.suffix}"
        return f"{self.prefix}{value}{self.suffix}"


class QueryFact(Model):
    type: Literal["query"] = "query"
    query: Query | Name
    field: str
    format: Format = Field(default_factory=Format)


class Calculation(Model):
    type: Literal["calculation"]
    op: BinaryOp
    inputs: list[Name] = Field(min_length=2, max_length=2)
    format: Format = Field(default_factory=Format)


Fact = Annotated[QueryFact | Calculation, Field(discriminator="type")]


class Prose(Model):
    type: Literal["prose"]
    text: str


class Callout(Model):
    type: Literal["callout"]
    title: str
    text: str
    tone: Literal["insight", "caution", "neutral"] = "insight"


class Metric(Model):
    type: Literal["metric"]
    label: str
    fact: Name
    detail: str = ""


class Column(Model):
    field: str
    label: str
    format: Format | None = None


class Table(Model):
    type: Literal["table"]
    title: str
    query: Query | Name
    columns: list[Column] = Field(min_length=1, max_length=30)
    caption: str = ""


class Series(Model):
    field: str
    label: str
    format: Format = Field(default_factory=Format)


class Chart(Model):
    type: Literal["chart"]
    title: str
    kind: Literal["bar", "line", "scatter"] = "bar"
    query: Query | Name
    x: str
    series: list[Series] = Field(min_length=1, max_length=8)
    x_label: str = ""
    y_label: str = ""
    caption: str = ""

    @model_validator(mode='after')
    def shared_axis_units(self):
        units={series.format.percent != 'none' for series in self.series}
        if len(units)>1:
            raise ValueError('A shared chart axis needs compatible units; use separate charts for percentages and plain numeric values')
        return self


class Html(Model):
    """Author-controlled fragment; runs under the host artifact sandbox/CSP."""
    type: Literal["html"]
    html: str


SimpleBlock = Annotated[Prose | Callout | Metric | Table | Chart | Html, Field(discriminator="type")]


class TabPane(Model):
    label: str = Field(min_length=1, max_length=120)
    layout: Literal['flow', 'split', 'metrics'] = 'flow'
    blocks: list[SimpleBlock] = Field(min_length=1, max_length=40)


class Tabs(Model):
    type: Literal['tabs']
    title: str = Field(min_length=1, max_length=200)
    panes: list[TabPane] = Field(min_length=2, max_length=12)
    initial: int = Field(default=0, ge=0)

    @model_validator(mode='after')
    def choices(self):
        if self.initial >= len(self.panes):
            raise ValueError('Initial tab must name an existing pane')
        if len({pane.label for pane in self.panes}) != len(self.panes):
            raise ValueError('Tab labels must be distinct')
        return self


Block = Annotated[Prose | Callout | Metric | Table | Chart | Html | Tabs, Field(discriminator="type")]


class Section(Model):
    id: Name
    title: str = ""
    eyebrow: str = ""
    summary: str = ""
    layout: Literal["flow", "split", "metrics"] = "flow"
    blocks: list[Block] = Field(min_length=1, max_length=60)
    notes: str = ""


class Theme(Model):
    mode: Literal["light", "dark"] = "light"
    accent: str = Field(default="#12685c", pattern=r"^#[0-9a-fA-F]{6}$")
    typography: Literal["sans", "editorial"] = "sans"
    density: Literal["comfortable", "compact"] = "comfortable"


class Document(Model):
    version: Literal[1] = 1
    kind: Literal["report", "slides"] = "report"
    title: str = Field(min_length=1, max_length=300)
    subtitle: str = ""
    eyebrow: str = ""
    language: str = Field(default="en", pattern=r"^[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})*$")
    labels: dict[Label, LabelText] = Field(default_factory=dict)
    theme: Theme = Field(default_factory=Theme)
    facts: dict[Name, Fact] = Field(default_factory=dict, max_length=200)
    queries: dict[Name, Query] = Field(default_factory=dict, max_length=100)
    bindings: dict[Name, Query] = Field(default_factory=dict, max_length=100)
    views: list[View] = Field(default_factory=list, max_length=100)
    sections: list[Section] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_sections(self):
        if len({s.id for s in self.sections}) != len(self.sections):
            raise ValueError("Section ids must be unique")
        return self



def resolve_query(document: Document, evidence: Evidence, query: Query | str, fields: list[str]) -> Query:
    if isinstance(query, Query):
        return query
    if query in document.queries:
        return document.queries[query]
    if query in document.bindings:
        return document.bindings[query]
    if query in evidence.datasets:
        return Query(dataset=query, columns=list(dict.fromkeys(fields)))
    raise DataError(f"Unknown dataset or named query: {query}")


def resolve_facts(document: Document, evidence: Evidence) -> dict:
    """Calculations are a dependency graph, with missingness propagated."""
    values, visiting = {}, set()

    def resolve(name):
        if name in values:
            return values[name]
        if name not in document.facts:
            raise DataError(f"Unknown fact: {name}")
        if name in visiting:
            raise DataError(f"Cyclic fact: {name}")
        visiting.add(name)
        fact = document.facts[name]
        if isinstance(fact, QueryFact):
            rows = execute(evidence, resolve_query(document, evidence, fact.query, [fact.field]))
            if len(rows) != 1 or fact.field not in rows[0]:
                raise DataError(f"Fact {name} must select one row containing {fact.field}")
            value = rows[0][fact.field]
        else:
            value = calculate(fact.op, *(resolve(i) for i in fact.inputs))
        visiting.remove(name)
        values[name] = value
        return value

    for name in document.facts:
        resolve(name)
    return values
