"""Versioned composition contract. New visuals can use a custom HTML block."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .data import DataError, Evidence, Model, Name, Query, execute, number


class Format(Model):
    decimals: int = Field(default=0, ge=0, le=8)
    prefix: str = Field(default="", max_length=24)
    suffix: str = Field(default="", max_length=24)
    percent: bool = False
    missing: str = "Not available"

    def display(self, value) -> str:
        if value is None:
            return self.missing
        if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
            value = number(value) * (100 if self.percent else 1)
            value = value.quantize(Decimal(1).scaleb(-self.decimals), rounding=ROUND_HALF_UP)
            return f"{self.prefix}{value:,.{self.decimals}f}{'%' if self.percent else ''}{self.suffix}"
        return f"{self.prefix}{value}{self.suffix}"


class QueryFact(Model):
    type: Literal["query"] = "query"
    query: Query
    field: str
    format: Format = Field(default_factory=Format)


class Calculation(Model):
    type: Literal["calculation"]
    op: Literal["add", "subtract", "multiply", "divide"]
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
    query: Query
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
    query: Query
    x: str
    series: list[Series] = Field(min_length=1, max_length=8)
    x_label: str = ""
    y_label: str = ""
    caption: str = ""


class Html(Model):
    """Author-controlled fragment; runs under the host artifact sandbox/CSP."""
    type: Literal["html"]
    html: str


Block = Annotated[Prose | Callout | Metric | Table | Chart | Html, Field(discriminator="type")]


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
    theme: Theme = Field(default_factory=Theme)
    facts: dict[Name, Fact] = Field(default_factory=dict, max_length=200)
    bindings: dict[Name, Query] = Field(default_factory=dict, max_length=100)
    sections: list[Section] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_sections(self):
        if any(s.id in {"content", "outline", "title", "sources", "artifact-bindings"} for s in self.sections):
            raise ValueError("Section id is reserved by the document")
        if len({s.id for s in self.sections}) != len(self.sections):
            raise ValueError("Section ids must be unique")
        return self


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
            rows = execute(evidence, fact.query)
            if len(rows) != 1 or fact.field not in rows[0]:
                raise DataError(f"Fact {name} must select one row containing {fact.field}")
            value = rows[0][fact.field]
        else:
            a, b = (resolve(i) for i in fact.inputs)
            if a is None or b is None:
                value = None
            else:
                a, b = number(a), number(b)
                if fact.op == "divide":
                    value = a / b if b else None
                elif fact.op == "add":
                    value = a + b
                elif fact.op == "subtract":
                    value = a - b
                else:
                    value = a * b
        visiting.remove(name)
        values[name] = value
        return value

    for name in document.facts:
        resolve(name)
    return values
