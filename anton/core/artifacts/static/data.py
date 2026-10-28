"""Small, deterministic relational queries over explicitly supplied evidence.

No model-written arithmetic, implicit coercion, sampling, or missing-as-zero.
The acquisition layer owns access; this module never opens a source or network.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
from hashlib import sha256
import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

Scalar = StrictStr | StrictInt | StrictFloat | StrictBool | Decimal | None
Name = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,79}$")]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Source(Model):
    label: str = Field(min_length=1)
    url: str = ""
    scope: str = ""


class Dataset(Model):
    source: Source
    rows: list[dict[str, Scalar]] = Field(max_length=100_000)

    @model_validator(mode="after")
    def finite(self):
        if any((isinstance(v, Decimal) and not v.is_finite()) or
               (isinstance(v, float) and not math.isfinite(v))
               for r in self.rows for v in r.values()):
            raise ValueError("Evidence contains a non-finite number")
        return self


class Evidence(Model):
    datasets: dict[Name, Dataset] = Field(default_factory=dict, max_length=100)

    def digest(self) -> str:
        return sha256(exact_json(self.model_dump()).encode()).hexdigest()


class Filter(Model):
    field: str
    op: Literal["eq", "ne", "lt", "le", "gt", "ge", "in"] = "eq"
    value: Scalar | list[Scalar]


class Measure(Model):
    name: Name
    op: Literal["count", "sum", "mean", "min", "max"]
    field: str | None = None
    missing: Literal["propagate", "skip"] = "propagate"

    @model_validator(mode="after")
    def field_required(self):
        if self.op != "count" and not self.field:
            raise ValueError(f"{self.op} needs a field")
        return self


class Order(Model):
    field: str
    descending: bool = False


BinaryOp = Literal['add', 'subtract', 'multiply', 'divide']


class QueryCalculation(Model):
    name: Name
    op: BinaryOp
    fields: list[str] = Field(min_length=2, max_length=2)


class Query(Model):
    dataset: Name
    columns: list[str] = Field(default_factory=list)
    where: list[Filter] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    measures: list[Measure] = Field(default_factory=list)
    order_by: list[Order] = Field(default_factory=list)
    calculations: list[QueryCalculation] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def shape(self):
        names = self.group_by + [m.name for m in self.measures] + [c.name for c in self.calculations]
        if len(names) != len(set(names)):
            raise ValueError("Query output names must be unique")
        if self.measures and self.columns:
            raise ValueError("Use group_by and measures, or columns")
        if self.group_by and not self.measures:
            raise ValueError("group_by needs measures")
        if not self.measures and not self.columns:
            raise ValueError("Explicit columns are required for a row query")
        return self


class DataError(ValueError):
    """A binding cannot be resolved without inventing information."""


def number(value) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise DataError(f"Expected a number, got {type(value).__name__}")
    result = Decimal(str(value))
    if not result.is_finite() or abs(result.adjusted()) > 1000:
        raise DataError("Expected a finite number with a decimal exponent between -1000 and 1000")
    return result


def calculate(op: BinaryOp, a, b):
    if a is None or b is None:
        return None
    a, b = number(a), number(b)
    with localcontext() as context:
        context.prec = 4096
        if op == 'divide':
            return a / b if b else None
        if op == 'add':
            return a + b
        if op == 'subtract':
            return a - b
        if op == 'multiply':
            return a * b
        raise DataError(f'Unknown calculation: {op}')


def _matches(row: dict, condition: Filter) -> bool:
    actual, wanted = row[condition.field], condition.value
    if condition.op == "in":
        if not isinstance(wanted, list):
            raise DataError("An in filter requires a list")
        return any(actual == v and (isinstance(actual, bool) == isinstance(v, bool)) for v in wanted)
    if isinstance(wanted, list):
        raise DataError("Only an in filter accepts a list")
    if condition.op in ("eq", "ne"):
        equal = actual == wanted and (isinstance(actual, bool) == isinstance(wanted, bool))
        return equal if condition.op == "eq" else not equal
    if actual is None or wanted is None:
        return False
    if isinstance(actual, str) and isinstance(wanted, str):
        left, right = actual, wanted
    else:
        left, right = number(actual), number(wanted)
    return {"lt": left < right, "le": left <= right, "gt": left > right, "ge": left >= right}[condition.op]


def _aggregate(rows: list[dict], measure: Measure):
    values = [r[measure.field] for r in rows if r[measure.field] is not None] if measure.field else rows
    if measure.op == "count":
        return len(values)
    if measure.missing == "propagate" and any(r[measure.field] is None for r in rows):
        return None
    values = [number(v) for v in values]
    if not values:
        return None
    if measure.op in ("sum", "mean"):
        with localcontext() as context:
            context.prec = 4096
            total = sum(values, Decimal(0))
            return total if measure.op == "sum" else total / len(values)
    return min(values) if measure.op == "min" else max(values)


def execute(evidence: Evidence, query: Query) -> list[dict]:
    """Resolve all rows, validating field coverage before applying any filter."""
    if query.dataset not in evidence.datasets:
        raise DataError(f"Unknown dataset: {query.dataset}")
    source = evidence.datasets[query.dataset].rows
    calculated = {c.name for c in query.calculations}
    required = set(query.columns) - calculated
    required.update(query.group_by + [w.field for w in query.where])
    required.update(m.field for m in query.measures if m.field)
    available = set(query.group_by + [m.name for m in query.measures]) if query.measures else {key for row in source for key in row}
    produced = set()
    for calculation in query.calculations:
        if calculation.name in available:
            raise DataError(f'Calculation would overwrite field: {calculation.name}')
        inputs = set(calculation.fields)
        if (inputs & calculated) - produced:
            raise DataError('Calculations may reference only source/output fields or earlier calculations')
        if query.measures:
            if missing := inputs - available:
                raise DataError(f'Calculation needs query output fields: {sorted(missing)}')
        else:
            required.update(inputs - produced)
        available.add(calculation.name)
        produced.add(calculation.name)
    for index, row in enumerate(source):
        missing = required - row.keys()
        if missing:
            raise DataError(f"{query.dataset} row {index}: missing fields {sorted(missing)}")
    rows = [r for r in source if all(_matches(r, w) for w in query.where)]
    if query.measures:
        groups = {}
        for row in rows:
            key = tuple(("number", number(row[k])) if isinstance(row[k], (int, float, Decimal)) and not isinstance(row[k], bool) else (type(row[k]).__name__, row[k]) for k in query.group_by)
            groups.setdefault(key, []).append(row)
        if not query.group_by and not groups:
            groups[()] = []
        rows = [{field: group[0][field] for field in query.group_by} |
                {m.name: _aggregate(group, m) for m in query.measures} for key, group in groups.items()]
    if query.calculations:
        rows = [dict(row) for row in rows]
        for row in rows:
            for calculation in query.calculations:
                row[calculation.name] = calculate(calculation.op, *(row[field] for field in calculation.fields))
    if not query.measures:
        rows = [{k: r[k] for k in query.columns} for r in rows]
    output_fields = set(query.group_by + [m.name for m in query.measures] + list(calculated) if query.measures else query.columns)
    for order in reversed(query.order_by):
        if order.field not in output_fields:
            raise DataError(f"Cannot sort by unselected field: {order.field}")
        present = [r for r in rows if r[order.field] is not None]
        try:
            present.sort(key=lambda r: r[order.field], reverse=order.descending)
        except TypeError as exc:
            raise DataError(f"Cannot sort mixed types in {order.field}") from exc
        rows = present + [r for r in rows if r[order.field] is None]
    return rows


def json_value(value):
    """JSON for a public binding; Decimal stays exact as a decimal string."""
    def encode(item):
        if isinstance(item, Decimal):
            return str(item)
        raise TypeError(f"Not JSON serializable: {type(item).__name__}")
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=encode)


def exact_json(value):
    """Host/authoring JSON: Decimal stays a number without float conversion.

    Browser bindings use json_value instead, where decimal strings prevent
    JavaScript's numeric parser from losing precision.
    """
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise DataError('JSON requires finite numbers')
        return str(value)
    if isinstance(value, dict):
        return '{' + ','.join(json.dumps(key, ensure_ascii=False) + ':' + exact_json(item) for key, item in value.items()) + '}'
    if isinstance(value, list):
        return '[' + ','.join(exact_json(item) for item in value) + ']'
    return json.dumps(value, ensure_ascii=False, allow_nan=False)
