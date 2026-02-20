"""MindsDB (Minds) HTTP client for natural language data access.

Translates natural language questions into SQL via MindsDB's REST API.
Data stays in MindsDB — only results come back.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class MindsClient:
    """Stateful HTTP client for the Minds REST API.

    Tracks conversation_id and message_id from the last ask() so that
    data() can fetch tabular results without the caller managing IDs.
    """

    api_key: str
    base_url: str = "https://mdb.ai"
    _last_conversation_id: str | None = field(default=None, repr=False)
    _last_message_id: str | None = field(default=None, repr=False)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def ask(
        self,
        question: str,
        mind: str,
        conversation_id: str | None = None,
    ) -> str:
        """Ask a natural language question to a mind.

        POSTs to /api/v1/responses and stores the returned conversation_id
        and message_id for subsequent data() calls.

        Returns the text answer from the mind.
        """
        import httpx

        payload: dict = {
            "input": question,
            "model": mind,
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id

        async with httpx.AsyncClient(base_url=self.base_url, timeout=60, follow_redirects=True) as client:
            resp = await client.post(
                "/api/v1/responses",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        # Store IDs for subsequent data() calls
        self._last_conversation_id = data.get("conversation_id")
        self._last_message_id = data.get("id")

        # Extract text output
        output_parts: list[str] = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        output_parts.append(content.get("text", ""))
        return "\n".join(output_parts) if output_parts else data.get("output_text", "")

    async def data(self, limit: int = 100, offset: int = 0) -> str:
        """Fetch raw tabular results from the last ask() call.

        GETs /api/v1/conversations/{conv_id}/items/{msg_id}/result and
        formats the response as a markdown table.

        Raises ValueError if no prior ask() has been made.
        """
        if not self._last_conversation_id or not self._last_message_id:
            raise ValueError(
                "No prior ask() call — use 'ask' first to get an answer, "
                "then 'data' to fetch raw results."
            )

        import httpx

        url = (
            f"/api/v1/conversations/{self._last_conversation_id}"
            f"/items/{self._last_message_id}/result"
        )
        params: dict = {}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset

        async with httpx.AsyncClient(base_url=self.base_url, timeout=60, follow_redirects=True) as client:
            resp = await client.get(url, headers=self._headers(), params=params)
            resp.raise_for_status()
            data = resp.json()

        return self._format_table(data)

    async def export(self) -> str:
        """Export the full result set from the last ask() as CSV.

        GETs /api/v1/conversations/{conv_id}/items/{msg_id}/export and
        returns the raw CSV text.

        Raises ValueError if no prior ask() has been made.
        """
        if not self._last_conversation_id or not self._last_message_id:
            raise ValueError(
                "No prior ask() call — use 'ask' first to get an answer, "
                "then 'export' to fetch CSV results."
            )

        import httpx

        url = (
            f"/api/v1/conversations/{self._last_conversation_id}"
            f"/items/{self._last_message_id}/export"
        )

        async with httpx.AsyncClient(base_url=self.base_url, timeout=60, follow_redirects=True) as client:
            resp = await client.get(url, headers=self._headers())
            resp.raise_for_status()
            return resp.text

    async def catalog(self, datasource: str) -> str:
        """Discover tables and columns for a datasource.

        GETs /api/v1/datasources/{datasource}/catalog and returns a
        formatted listing of tables and their columns.
        """
        import httpx

        async with httpx.AsyncClient(base_url=self.base_url, timeout=60, follow_redirects=True) as client:
            resp = await client.get(
                f"/api/v1/datasources/{datasource}/catalog",
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        return self._format_catalog(data)

    @staticmethod
    def _format_table(data: dict) -> str:
        """Format a result payload as a markdown table."""
        columns: list[str] = data.get("column_names", [])
        rows: list[list] = data.get("data", [])

        if not columns:
            return "No data returned."

        # Header
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        lines = [header, separator]

        for row in rows:
            cells = [str(cell) if cell is not None else "" for cell in row]
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    @staticmethod
    def _format_catalog(data: dict | list) -> str:
        """Format a catalog payload as a readable listing."""
        # The API may return a list of tables or a dict with a tables key
        tables: list[dict] = []
        if isinstance(data, list):
            tables = data
        elif isinstance(data, dict):
            tables = data.get("tables", data.get("items", []))

        if not tables:
            return "No tables found."

        lines: list[str] = []
        for table in tables:
            name = table.get("name", table.get("table_name", "unknown"))
            lines.append(f"## {name}")
            columns = table.get("columns", [])
            if columns:
                for col in columns:
                    col_name = col.get("name", col.get("column_name", "?"))
                    col_type = col.get("type", col.get("data_type", ""))
                    type_str = f" ({col_type})" if col_type else ""
                    lines.append(f"  - {col_name}{type_str}")
            else:
                lines.append("  (no column info)")
            lines.append("")

        return "\n".join(lines)


@dataclass
class SyncMindsClient:
    """Sync wrapper around MindsClient for use in scratchpad code."""

    api_key: str
    base_url: str = "https://mdb.ai"
    _client: MindsClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = MindsClient(api_key=self.api_key, base_url=self.base_url)

    def ask(self, question: str, mind: str, conversation_id: str | None = None) -> str:
        """Ask a natural language question (sync). Returns text answer."""
        return asyncio.run(self._client.ask(question, mind, conversation_id))

    def data(self, limit: int = 100, offset: int = 0) -> str:
        """Fetch raw tabular results as a markdown table (sync)."""
        return asyncio.run(self._client.data(limit=limit, offset=offset))

    def export(self) -> str:
        """Export the full result set as CSV (sync)."""
        return asyncio.run(self._client.export())

    def catalog(self, datasource: str) -> str:
        """Discover tables and columns for a datasource (sync)."""
        return asyncio.run(self._client.catalog(datasource))
