"""Executable minimum render contracts for author-controlled HTML views."""
from typing import Literal

from pydantic import Field

from .data import DataError, Model, Name


class View(Model):
    selector: str = Field(min_length=1, max_length=240)
    binding: Name
    minimum: Literal['rows', 'one'] = 'one'


def checks(bindings, views, *, covered=()):
    missing = bindings.keys() - {v.binding for v in views} - set(covered)
    if missing:
        raise DataError(f'Custom bindings need a render check: {sorted(missing)}')
    result = []
    for view in views:
        if view.binding not in bindings:
            raise DataError(f'Unknown view binding: {view.binding}')
        result.append({'selector': view.selector, 'minimum': len(bindings[view.binding]) if view.minimum == 'rows' else 1})
    return result


def validate_ids(html):
    from html.parser import HTMLParser

    class Ids(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = set()

        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key == 'id' and value:
                    if value in self.ids:
                        raise DataError(f'Duplicate HTML id: {value}')
                    self.ids.add(value)

    Ids().feed(html)
