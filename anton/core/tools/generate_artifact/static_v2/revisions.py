"""Transactional edits to the latest complete composition, before normal validation."""
import copy
from typing import Any, Literal

from pydantic import Field, StrictInt, StrictStr, model_validator

from anton.core.artifacts.static.data import Model


class Change(Model):
    op: Literal['add', 'replace', 'remove'] = 'replace'
    path: list[StrictStr | StrictInt] = Field(min_length=1, max_length=20)
    value: Any = None


class TextEdit(Model):
    find: str = Field(min_length=1, max_length=20000)
    replace: str = Field(max_length=20000)


class Revision(Model):
    changes: list[Change] = Field(default_factory=list, max_length=40)
    text_edits: list[TextEdit] = Field(default_factory=list, max_length=40)

    @model_validator(mode='after')
    def nonempty(self):
        if not self.changes and not self.text_edits:
            raise ValueError('A revision needs at least one change or text edit')
        if any(change.op != 'remove' and 'value' not in change.model_fields_set for change in self.changes):
            raise ValueError('add and replace require an explicit value (which may be null)')
        return self


TOOL = {'name': 'revise_static_artifact', 'description': 'Revise the latest complete candidate, or the saved composition on entry, without rewriting it. changes apply sequentially to its structured object: paths start with document (compiled mode) or html/bindings/views (custom mode), using string object keys and integer array indices. add inserts an array item or a new object key; replace/remove require an existing target. Mode changes are forbidden. For custom HTML, text_edits replace exact text in html sequentially, and each find must occur exactly once. A failed edit changes nothing. The resulting complete composition goes through the same compilation, browser and content review as a full compose call; it is not published directly. Use for small edits and repairs, preserving all unrelated content.', 'input_schema': Revision.model_json_schema()}


def _target(container, key, *, insert=False):
    if isinstance(container, list):
        if type(key) is not int or key < 0 or key >= len(container) + int(insert):
            raise ValueError('Array paths need an in-range, non-negative integer index')
    elif isinstance(container, dict):
        if not isinstance(key, str):
            raise ValueError('Object paths need string keys')
        if not insert and key not in container:
            raise ValueError(f'Revision path does not exist: {key}')
    else:
        raise ValueError('Revision path crosses a scalar value')


def apply(base, revision: Revision):
    result = copy.deepcopy(base)
    allowed = {'document'} if result['mode'] == 'document' else {'html', 'bindings', 'views'}
    for change in revision.changes:
        if change.path[0] not in allowed:
            raise ValueError('Revision cannot change composition mode or unknown root fields')
        container = result
        for key in change.path[:-1]:
            _target(container, key)
            container = container[key]
        key = change.path[-1]
        _target(container, key, insert=change.op == 'add')
        if change.op == 'remove':
            del container[key]
        elif change.op == 'add':
            if isinstance(container, list):
                container.insert(key, copy.deepcopy(change.value))
            else:
                if key in container:
                    raise ValueError('add requires a new object key; use replace for an existing key')
                container[key] = copy.deepcopy(change.value)
        else:
            container[key] = copy.deepcopy(change.value)
    if revision.text_edits and result['mode'] != 'html':
        raise ValueError('text_edits apply only to custom HTML; use document field paths for compiled content')
    for edit in revision.text_edits:
        if result['html'].count(edit.find) != 1:
            raise ValueError('Each HTML find must match exactly once; include enough unique surrounding text')
        result['html'] = result['html'].replace(edit.find, edit.replace, 1)
    return result
