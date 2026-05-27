"""Resolve ``data_refs`` — variable lookups against running scratchpads.

Each ``{scratchpad, variable}`` pair is materialised into:
  - a sidecar file under ``<artifact>/_gen_data/<variable>.{pkl,json}``
    that the sub-agent can ``open()`` to access the raw value;
  - a short textual summary embedded into the sub-agent's prompt so it
    can reason about shape and types without re-loading the object.

Extraction runs inside the source scratchpad subprocess via ``pad.execute``
and goes through a ``__OK_DILL__<b64>`` marker (with a ``__OK_JSON__`` fallback
for non-picklable objects). The variable name is validated upfront as a
Python identifier so we never inject untrusted strings into the exec body.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anton.chat_session import ChatSession


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_EXTRACT_TEMPLATE = """\
import sys, json, traceback, base64
try:
    import dill as _pickle
    _SCHEME = '__OK_DILL__'
except ImportError:
    import pickle as _pickle
    _SCHEME = '__OK_PICKLE__'
try:
    _v = {var}
except NameError:
    print('__MISSING__')
    sys.exit(0)
try:
    print(_SCHEME + base64.b64encode(_pickle.dumps(_v)).decode())
except Exception:
    try:
        print('__OK_JSON__' + json.dumps(_v, default=str))
    except Exception:
        print('__FAIL__' + traceback.format_exc()[-800:])
"""


def _summarize(value: Any, max_chars: int = 800) -> str:
    """Build a short, type-aware text summary for the sub-agent prompt."""
    try:
        import pandas as pd  # type: ignore

        if isinstance(value, pd.DataFrame):
            head = value.head(5).to_string()
            return (
                f"pandas.DataFrame shape={value.shape}\n"
                f"columns={list(value.columns)}\n"
                f"dtypes={ {c: str(t) for c, t in value.dtypes.items()} }\n"
                f"head(5):\n{head}"
            )[:max_chars]
        if isinstance(value, pd.Series):
            return (
                f"pandas.Series shape={value.shape} dtype={value.dtype}\n"
                f"head(5):\n{value.head(5).to_string()}"
            )[:max_chars]
    except ImportError:
        pass

    if isinstance(value, dict):
        keys_preview = list(value.keys())[:20]
        return (
            f"dict({len(value)} keys)\n"
            f"sample_keys={keys_preview}\n"
            f"repr={repr(value)[:max_chars]}"
        )[:max_chars]
    if isinstance(value, list):
        sample = repr(value[:5])
        return (
            f"list({len(value)} items)\nsample={sample[:max_chars]}"
        )[:max_chars]
    return f"{type(value).__name__}: {repr(value)[:max_chars]}"


async def resolve_refs(
    session: "ChatSession",
    refs: list[dict],
    sidecar_dir: Path,
) -> list[dict] | str:
    """Extract each ref into a sidecar file + summary.

    Returns the list of resolved dicts on success or a single error
    string on the first failure (no partial results — keeps the
    contract simple for the main LLM).
    """
    if not refs:
        return []

    sidecar_dir.mkdir(parents=True, exist_ok=True)
    resolved: list[dict] = []

    for ref in refs:
        scratchpad = ref["scratchpad"]
        variable = ref["variable"]
        if not _IDENT_RE.match(variable):
            return (
                f"Error: variable name `{variable}` is not a valid Python "
                "identifier (must match [A-Za-z_][A-Za-z0-9_]*)."
            )

        pad = await session._scratchpads.get_or_create(scratchpad)
        code = _EXTRACT_TEMPLATE.format(var=variable)
        cell = await pad.execute(
            code,
            description=f"generate_artifact: extract `{variable}`",
            estimated_seconds=15,
        )

        stdout = (cell.stdout or "").strip()
        err = (cell.error or "").strip()
        if err:
            return (
                f"Error: extraction failed for `{variable}` in scratchpad "
                f"`{scratchpad}`: {err[:300]}"
            )

        marker_line: str | None = None
        for line in stdout.splitlines():
            if (
                line.startswith("__OK_DILL__")
                or line.startswith("__OK_PICKLE__")
                or line.startswith("__OK_JSON__")
                or line.startswith("__MISSING__")
                or line.startswith("__FAIL__")
            ):
                marker_line = line
                break

        if marker_line is None:
            return (
                f"Error: extractor produced no marker for `{variable}` in "
                f"scratchpad `{scratchpad}`. stdout head: {stdout[:300]!r}"
            )

        if marker_line.startswith("__MISSING__"):
            return (
                f"Error: variable `{variable}` not found in scratchpad "
                f"`{scratchpad}`."
            )
        if marker_line.startswith("__FAIL__"):
            return (
                f"Error: extraction raised inside scratchpad `{scratchpad}` "
                f"for `{variable}`: {marker_line[len('__FAIL__'):][:400]}"
            )

        if marker_line.startswith("__OK_DILL__") or marker_line.startswith(
            "__OK_PICKLE__"
        ):
            scheme = "dill" if marker_line.startswith("__OK_DILL__") else "pickle"
            tag_len = len("__OK_DILL__") if scheme == "dill" else len("__OK_PICKLE__")
            payload_b64 = marker_line[tag_len:]
            try:
                raw_bytes = base64.b64decode(payload_b64)
            except Exception as exc:
                return f"Error: base64 decode failed for `{variable}`: {exc}"

            value: Any
            try:
                if scheme == "dill":
                    import dill as _pickle  # type: ignore
                else:
                    import pickle as _pickle
                value = _pickle.loads(raw_bytes)
            except Exception as exc:
                return f"Error: {scheme} decode failed for `{variable}`: {exc}"

            sidecar_path = sidecar_dir / f"{variable}.pkl"
            sidecar_path.write_bytes(raw_bytes)
            resolved.append(
                {
                    "scratchpad": scratchpad,
                    "variable": variable,
                    "format": scheme,
                    "sidecar_path": str(sidecar_path),
                    "summary": _summarize(value),
                }
            )
            continue

        # __OK_JSON__ — value isn't picklable but is JSON-serialisable.
        payload_json = marker_line[len("__OK_JSON__"):]
        sidecar_path = sidecar_dir / f"{variable}.json"
        sidecar_path.write_text(payload_json, encoding="utf-8")
        resolved.append(
            {
                "scratchpad": scratchpad,
                "variable": variable,
                "format": "json",
                "sidecar_path": str(sidecar_path),
                "summary": (
                    "(json-only; not picklable) "
                    + payload_json[:800]
                ),
            }
        )

    return resolved
