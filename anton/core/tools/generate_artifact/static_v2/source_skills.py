"""Full source procedures across a delegation boundary, with artifact trust rules."""
from __future__ import annotations

import inspect
import json

from anton.core.tools.registry import ToolOutcome


def catalog(session) -> str:
    store = getattr(session, '_skill_store', None)
    if store is None:
        return ''
    summaries = store.list_summaries()
    if inspect.iscoroutine(summaries):
        summaries.close()
        return ''
    if not isinstance(summaries, list):
        return ''
    return json.dumps([{'label': row['label'], 'description': row.get('description', '')}
                       for row in summaries if isinstance(row, dict) and isinstance(row.get('label'), str)], ensure_ascii=False)


async def recall(session, arguments):
    label = arguments.get('label')
    store = getattr(session, '_skill_store', None)
    if not isinstance(label, str) or not label.strip() or store is None:
        return ToolOutcome(content='Source procedure lookup needs an exact label and a trusted skill store.', ok=False)
    skill = store.load_for_artifacts(label.strip())
    if inspect.iscoroutine(skill):
        skill.close()
        skill = None
    body = getattr(skill, 'declarative_md', None)
    if not isinstance(body, str) or not body.strip():
        return ToolOutcome(content='That exact source procedure is unavailable in the artifact trust scope. Do not guess its API or substitute another source.', ok=False)
    # A parent recall's "already loaded" stub is not the procedure. The
    # delegate receives the exact trusted body even when the parent read it.
    return ToolOutcome(content=body, ok=True)
