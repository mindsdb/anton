from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import yaml


@dataclass
class DatasourceField:
    name: str
    required: bool = True
    secret: bool = False
    description: str = ""
    default: str = ""


@dataclass
class AuthMethod:
    name: str
    display: str
    fields: list[DatasourceField] = field(default_factory=list)


@dataclass
class DatasourceEngine:
    engine: str
    display_name: str
    pip: str = ""
    name_from: Union[str, list[str]] = ""
    fields: list[DatasourceField] = field(default_factory=list)
    # "choice" means the user must pick from auth_methods before collecting fields
    # empty string means no choice, just collect fields from the top-level "fields" list
    auth_method: str = ""
    auth_methods: list[AuthMethod] = field(default_factory=list)
    test_snippet: str = ""
    popular: bool = False
    # True for engines defined in ~/.anton/datasources.md
    custom: bool = False


# Matches a level-2 heading followed by a ```yaml fenced block.
_YAML_BLOCK_RE = re.compile(
    r"^##\s+(.+?)\s*$\n(.*?)^```yaml\n(.*?)^```",
    re.MULTILINE | re.DOTALL,
)


def _parse_fields(raw: list) -> list[DatasourceField]:
    result: list[DatasourceField] = []
    for f in raw or []:
        if not isinstance(f, dict):
            continue
        result.append(
            DatasourceField(
                name=f.get("name", ""),
                required=bool(f.get("required", True)),
                secret=bool(f.get("secret", False)),
                description=f.get("description", ""),
                default=str(f.get("default", "")),
            )
        )
    return result


def _parse_file(
    path: Path, *, custom: bool = False
) -> dict[str, DatasourceEngine]:
    """Extract engine definitions from a datasources.md file."""
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    engines: dict[str, DatasourceEngine] = {}

    for match in _YAML_BLOCK_RE.finditer(text):
        yaml_text = match.group(3)
        try:
            data = yaml.safe_load(yaml_text)
        except yaml.YAMLError as exc:
            import sys

            print(
                f"[anton] Warning: skipping malformed YAML block in {path}: {exc}",
                file=sys.stderr,
            )
            continue
        if not isinstance(data, dict) or "engine" not in data:
            continue

        raw_auth_methods = data.get("auth_methods", []) or []
        auth_methods: list[AuthMethod] = []
        for am in raw_auth_methods:
            if not isinstance(am, dict):
                continue
            auth_methods.append(
                AuthMethod(
                    name=am.get("name", ""),
                    display=am.get("display", am.get("name", "")),
                    fields=_parse_fields(am.get("fields", [])),
                )
            )

        engine_slug = str(data["engine"])
        engines[engine_slug] = DatasourceEngine(
            engine=engine_slug,
            display_name=str(data.get("display_name", engine_slug)),
            pip=str(data.get("pip", "")),
            name_from=data.get("name_from", ""),
            fields=_parse_fields(data.get("fields", [])),
            auth_method=str(data.get("auth_method", "")),
            auth_methods=auth_methods,
            test_snippet=str(data.get("test_snippet", "")),
            popular=bool(data.get("popular", False)),
            custom=custom,
        )

    return engines


class DatasourceRegistry:
    """Parsed registry of all available data source engines."""

    _BUILTIN_PATH: Path = Path(__file__).resolve().parent / "datasources.md"
    _USER_PATH: Path = Path("~/.anton/datasources.md").expanduser()

    def __init__(self) -> None:
        self._engines: dict[str, DatasourceEngine] = {}
        self._load()

    def _load(self) -> None:
        self._engines = _parse_file(self._BUILTIN_PATH)
        for slug, engine in _parse_file(
            self._USER_PATH, custom=True
        ).items():
            self._engines[slug] = engine

    def reload(self) -> None:
        """Reload datasource definitions from disk."""
        self._load()

    def validate_file(self, path: Path) -> dict[str, DatasourceEngine]:
        """Parse a datasources.md file and return its engine definitions."""
        return _parse_file(path)

    def get(self, engine_slug: str) -> DatasourceEngine | None:
        return self._engines.get(engine_slug)

    def find_by_name(self, display_name: str) -> DatasourceEngine | None:
        """Case-insensitive match on display_name or engine slug."""
        needle = display_name.strip().lower()
        for engine in self._engines.values():
            if engine.display_name.lower() == needle or engine.engine.lower() == needle:
                return engine
        matches = [
            e
            for e in self._engines.values()
            if needle in e.display_name.lower() or needle in e.engine.lower()
        ]
        return matches[0] if len(matches) == 1 else None

    def fuzzy_find(self, text: str) -> list[DatasourceEngine]:
        """Return engines whose name/slug closely matches *text* (fuzzy, for typo tolerance)."""

        def _normalize(s: str) -> str:
            return re.sub(r"[\s\-_]", "", s).lower()

        needle = _normalize(text)
        # Build a map from normalized key → engine (display_name takes priority)
        key_to_engine: dict[str, DatasourceEngine] = {}
        for engine in self._engines.values():
            key_to_engine[_normalize(engine.display_name)] = engine
            # Don't overwrite display_name key with slug key
            slug_key = _normalize(engine.engine)
            if slug_key not in key_to_engine:
                key_to_engine[slug_key] = engine

        close_keys = difflib.get_close_matches(
            needle, key_to_engine.keys(), n=3, cutoff=0.6
        )
        # Deduplicate while preserving order
        seen: set[str] = set()
        results: list[DatasourceEngine] = []
        for k in close_keys:
            eng = key_to_engine[k]
            if eng.engine not in seen:
                seen.add(eng.engine)
                results.append(eng)
        return results

    def all_engines(self) -> list[DatasourceEngine]:
        return sorted(self._engines.values(), key=lambda e: e.display_name)

    def derive_name(
        self, engine_def: DatasourceEngine, credentials: dict[str, str]
    ) -> str:
        """Derive a default connection name from name_from field(s)."""
        name_from = engine_def.name_from
        if not name_from:
            return ""
        if isinstance(name_from, str):
            return credentials.get(name_from, "")
        parts = [credentials.get(f, "") for f in name_from if credentials.get(f)]
        return "_".join(parts)
