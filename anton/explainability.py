from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExplainabilityQuery:
    datasource: str
    sql: str
    engine: str | None = None
    status: str = "ok"
    error_message: str | None = None

    def to_dict(self) -> dict:
        return {
            "datasource": self.datasource,
            "sql": self.sql,
            "engine": self.engine,
            "status": self.status,
            "error_message": self.error_message,
        }


@dataclass
class ExplainabilityRecord:
    turn: int
    created_at: str
    user_message: str
    answer_text: str
    summary: str
    data_sources: list[dict] = field(default_factory=list)
    sql_queries: list[dict] = field(default_factory=list)
    scratchpad_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "created_at": self.created_at,
            "user_message": self.user_message,
            "answer_text": self.answer_text,
            "summary": self.summary,
            "data_sources": self.data_sources,
            "sql_queries": self.sql_queries,
            "scratchpad_steps": self.scratchpad_steps,
        }


class ExplainabilityStore:
    def __init__(self, workspace_path: Path) -> None:
        self._dir = workspace_path / ".anton" / "explainability"

    def save(self, record: ExplainabilityRecord) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n"
        latest = self._dir / "latest.json"
        latest.write_text(payload, encoding="utf-8")
        turn_file = self._dir / f"turn-{record.turn:04d}.json"
        turn_file.write_text(payload, encoding="utf-8")

    def load_latest(self) -> ExplainabilityRecord | None:
        latest = self._dir / "latest.json"
        if not latest.is_file():
            return None
        try:
            payload = json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            return None
        try:
            return ExplainabilityRecord(
                turn=int(payload.get("turn", 0)),
                created_at=str(payload.get("created_at", "")),
                user_message=str(payload.get("user_message", "")),
                answer_text=str(payload.get("answer_text", "")),
                summary=str(payload.get("summary", "")),
                data_sources=list(payload.get("data_sources", [])),
                sql_queries=list(payload.get("sql_queries", [])),
                scratchpad_steps=list(payload.get("scratchpad_steps", [])),
            )
        except Exception:
            return None


class ExplainabilityCollector:
    def __init__(self, store: ExplainabilityStore, *, turn: int, user_message: str) -> None:
        self._store = store
        self._turn = turn
        self._user_message = user_message
        self._created_at = _utc_now()
        self._scratchpad_steps: list[str] = []
        self._queries: list[ExplainabilityQuery] = []
        self._sources: list[dict[str, str | None]] = []

    def add_scratchpad_step(self, description: str) -> None:
        cleaned = (description or "").strip()
        if cleaned and cleaned not in self._scratchpad_steps:
            self._scratchpad_steps.append(cleaned)

    def add_query(
        self,
        *,
        datasource: str,
        sql: str,
        engine: str | None = None,
        status: str = "ok",
        error_message: str | None = None,
    ) -> None:
        cleaned_sql = (sql or "").strip()
        cleaned_ds = (datasource or "").strip() or "Unknown datasource"
        if not cleaned_sql:
            return
        entry = ExplainabilityQuery(
            datasource=cleaned_ds,
            sql=cleaned_sql,
            engine=(engine or "").strip() or None,
            status=status,
            error_message=(error_message or "").strip() or None,
        )
        if any(
            existing.datasource == entry.datasource
            and existing.sql == entry.sql
            and existing.status == entry.status
            for existing in self._queries
        ):
            return
        self._queries.append(entry)
        self.add_source(name=cleaned_ds, engine=(engine or "").strip() or None)

    def add_source(self, *, name: str, engine: str | None = None) -> None:
        cleaned_name = (name or "").strip()
        if not cleaned_name:
            return
        entry = {"name": cleaned_name, "engine": (engine or "").strip() or None}
        if entry not in self._sources:
            self._sources.append(entry)

    def add_sources_from_text(self, *texts: str) -> None:
        for text in texts:
            if not text:
                continue
            for source in _extract_sources_from_text(text):
                self.add_source(name=source, engine=None)

    def add_inferred_queries_from_code(self, code: str) -> None:
        if self._queries:
            return
        sql_statements = _extract_sql_from_code(code)
        datasource_names = _extract_datasource_names_from_code(code)
        datasource = datasource_names[0] if datasource_names else "connected datasource"
        for sql in sql_statements:
            self.add_query(
                datasource=datasource,
                sql=sql,
                engine=None,
                status="ok",
                error_message=None,
            )

    def finalize(self, answer_text: str) -> ExplainabilityRecord:
        data_sources: list[dict] = []
        seen_sources: set[tuple[str, str | None]] = set()
        for source in self._sources:
            key = (str(source.get("name", "")), source.get("engine"))
            if key in seen_sources:
                continue
            seen_sources.add(key)
            data_sources.append({"name": key[0], "engine": key[1]})

        summary = self._build_summary(answer_text, data_sources)
        record = ExplainabilityRecord(
            turn=self._turn,
            created_at=self._created_at,
            user_message=self._user_message,
            answer_text=answer_text.strip(),
            summary=summary,
            data_sources=data_sources,
            sql_queries=[query.to_dict() for query in self._queries],
            scratchpad_steps=list(self._scratchpad_steps),
        )
        if self._store is not None:
            self._store.save(record)
        return record

    def _build_summary(self, answer_text: str, data_sources: list[dict]) -> str:
        if self._queries:
            source_names = ", ".join(source["name"] for source in data_sources[:3])
            query_count = len(self._queries)
            step_text = ""
            if self._scratchpad_steps:
                lead = self._scratchpad_steps[0].rstrip(".")
                step_text = f" I used the scratchpad to {lead.lower()}."
            return (
                f"I queried {source_names} with {query_count} SQL "
                f"{'statement' if query_count == 1 else 'statements'} to gather the data behind this answer."
                f"{step_text}"
            )
        if data_sources:
            source_names = ", ".join(source["name"] for source in data_sources[:3])
            if self._scratchpad_steps:
                lead = self._scratchpad_steps[0].rstrip(".").lower()
                return (
                    f"I gathered information from {source_names} and used the scratchpad to "
                    f"{lead} before drafting the answer."
                )
            return f"I gathered information from {source_names} before drafting the answer."
        if self._scratchpad_steps:
            primary_step = self._scratchpad_steps[0].rstrip(".").lower()
            return f"I used the scratchpad to {primary_step} before drafting the answer."
        if answer_text.strip():
            return "I answered directly from the conversation context without querying a datasource or generating SQL."
        return "No explainability details were captured for this answer."


_URL_RE = re.compile(r"https?://[^\s)\"'>]+")
_SQL_LITERAL_RE = re.compile(
    r"(?P<quote>'''|\"\"\"|'|\")(?P<body>.*?)(?P=quote)",
    re.DOTALL,
)
_DS_PREFIX_RE = re.compile(r"\b(DS_[A-Z0-9_]+)__[A-Z0-9_]+\b")


def _extract_sources_from_text(text: str) -> list[str]:
    sources: list[str] = []
    for match in _URL_RE.findall(text):
        parsed = urlparse(match)
        host = (parsed.hostname or "").lower()
        host = host.removeprefix("www.")
        if not host:
            continue
        if host not in sources:
            sources.append(host)
    return sources


def _looks_like_sql(text: str) -> bool:
    normalized = " ".join(text.strip().split()).upper()
    if len(normalized) < 12:
        return False
    starters = ("SELECT ", "WITH ", "INSERT ", "UPDATE ", "DELETE ", "SHOW ", "DESCRIBE ")
    if not normalized.startswith(starters):
        return False
    return any(keyword in normalized for keyword in (" FROM ", " JOIN ", " INTO ", " TABLE ", "SELECT "))


def _extract_sql_from_code(code: str) -> list[str]:
    sql_statements: list[str] = []
    for match in _SQL_LITERAL_RE.finditer(code or ""):
        body = match.group("body").strip()
        if not _looks_like_sql(body):
            continue
        cleaned = "\n".join(line.rstrip() for line in body.splitlines()).strip()
        if cleaned and cleaned not in sql_statements:
            sql_statements.append(cleaned)
    return sql_statements


def _extract_datasource_names_from_code(code: str) -> list[str]:
    names: list[str] = []
    for prefix in _DS_PREFIX_RE.findall(code or ""):
        slug = prefix.removeprefix("DS_").lower().replace("_", "-")
        if slug not in names:
            names.append(slug)
    return names
