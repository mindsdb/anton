"""MindsDB query client for running native queries and discovering data catalogs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class MindsQueryClient:
    """REST client to run native queries via a MindsDB Mind and retrieve results as DataFrames.

    Public surface:
      - get_data_catalog()         discover available datasources, tables, columns
      - run_native_query_df()      run a native query against a datasource

    Internal/private:
      - _run_query_df()            (used by run_native_query_df)
    """

    mindsserver_url: str
    api_key: str
    mind_name: str
    timeout_s: float = 60.0
    verify_ssl: bool = True
    progress_fn: Callable[..., object] | None = field(default=None, repr=False)

    # ── HTTP helpers ─────────────────────────────────────────────

    def _client(self):  # -> httpx.Client
        import httpx
        return httpx.Client(
            base_url=self.mindsserver_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            follow_redirects=True,
            timeout=self.timeout_s,
            verify=self.verify_ssl,
        )

    @staticmethod
    def _normalize_sql_for_match(sql: str) -> str:
        return " ".join(sql.strip().strip(";").split()).lower()

    # ── Datasources + Data Catalog ───────────────────────────────

    def _list_datasources(self, client: httpx.Client) -> list[dict[str, Any]]:
        r = client.get("/api/v1/datasources")
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected datasources payload: {type(data)}")
        return data

    def _get_datasource_catalog(
        self,
        client: httpx.Client,
        datasource_name: str,
        *,
        mind_filter: str | None = None,
    ) -> dict[str, Any]:
        params = {}
        if mind_filter:
            params["mind"] = mind_filter
        r = client.get(f"/api/v1/datasources/{datasource_name}/catalog", params=params)
        if r.status_code == 404:
            return {
                "datasource": {"name": datasource_name},
                "tables": [],
                "status": {"overall_status": "NOT_FOUND", "message": "Catalog not found for datasource"},
            }
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected catalog payload: {type(payload)}")
        return payload

    def _get_mind_details(self, client: httpx.Client) -> dict[str, Any]:
        """Fetch Mind metadata from GET /api/v1/minds/{mind_name}."""
        r = client.get(f"/api/v1/minds/{self.mind_name}")
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected mind details payload: {type(payload)}")
        return payload

    def get_data_catalog(self) -> dict[str, Any]:
        """Return a mind data catalog: datasources + tables/columns metadata + mind prompts.

        Shape::

            {
              "mind": "<mind_name>",
              "system_prompt": "...",
              "prompt_template": "...",
              "datasources": [
                 {
                   "name": "...",
                   "engine": "...",
                   "description": "...",
                   "tables_allowlist": [...],
                   "catalog": { ... }
                 },
                 ...
              ]
            }
        """
        with self._client() as client:
            # Fetch mind-level details (system_prompt, prompt_template, datasources)
            mind_details = self._get_mind_details(client)
            params = mind_details.get("parameters") or {}
            system_prompt = params.get("system_prompt", "")
            prompt_template = params.get("prompt_template", "")

            # Only include datasources actually attached to this mind.
            # The mind details response has a "datasources" list with the
            # attached names — use it as a filter so the LLM doesn't try
            # to query datasources the mind can't reach.
            mind_ds_list = mind_details.get("datasources") or []
            mind_ds_names: set[str] = set()
            for entry in mind_ds_list:
                if isinstance(entry, dict):
                    n = entry.get("name")
                elif isinstance(entry, str):
                    n = entry
                else:
                    continue
                if n:
                    mind_ds_names.add(n)

            all_datasources = self._list_datasources(client)

            # If the mind has an explicit datasource list, filter to only those.
            # Otherwise fall back to all (for minds without explicit attachments).
            if mind_ds_names:
                datasources = [
                    ds for ds in all_datasources
                    if isinstance(ds, dict) and ds.get("name") in mind_ds_names
                ]
            else:
                datasources = all_datasources

            out: list[dict[str, Any]] = []
            for ds in datasources:
                if not isinstance(ds, dict):
                    continue
                name = ds.get("name")
                if not name:
                    continue
                engine = ds.get("engine")
                description = ds.get("description")
                tables_allowlist = ds.get("tables") or []
                catalog = self._get_datasource_catalog(client, name, mind_filter=self.mind_name)
                if isinstance(catalog, dict):
                    cat_ds = catalog.get("datasource") or {}
                    if isinstance(cat_ds, dict) and not cat_ds.get("engine") and engine:
                        cat_ds["engine"] = engine
                        catalog["datasource"] = cat_ds
                out.append(
                    {
                        "name": name,
                        "engine": engine,
                        "description": description,
                        "tables_allowlist": tables_allowlist,
                        "catalog": catalog,
                    }
                )
            result: dict[str, Any] = {"mind": self.mind_name, "datasources": out}
            if system_prompt:
                result["system_prompt"] = system_prompt
            if prompt_template:
                result["prompt_template"] = prompt_template
            return result

    # ── Conversations / exports (internal) ───────────────────────

    def _list_conversations(self, client: httpx.Client) -> list[dict[str, Any]]:
        r = client.get("/api/v1/conversations/")
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected conversations payload: {type(data)}")
        return data

    def _list_items(self, client: httpx.Client, conversation_id: str) -> list[dict[str, Any]]:
        r = client.get(f"/api/v1/conversations/{conversation_id}/items/")
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload, list):
            return payload
        keys = list(payload.keys()) if isinstance(payload, dict) else None
        raise RuntimeError(f"Unexpected items payload: {type(payload)} keys={keys}")

    def _export_csv_text(self, client: httpx.Client, conversation_id: str, item_id: str) -> str:
        r = client.get(f"/api/v1/conversations/{conversation_id}/items/{item_id}/export")
        r.raise_for_status()
        return r.text

    def _conversations_for_mind(self, conversations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for c in conversations:
            if not isinstance(c, dict):
                continue
            if c.get("metadata", {}).get("model_name") == self.mind_name:
                out.append(c)
        return out

    def _pick_latest_conversation_id(self, client: httpx.Client) -> str:
        convs = self._conversations_for_mind(self._list_conversations(client))
        if not convs:
            raise RuntimeError(f"No conversations found for mind {self.mind_name!r} after running query")

        def _ts(c: dict[str, Any]) -> str:
            return c.get("created_at") or ""

        return sorted(convs, key=_ts, reverse=True)[0]["id"]

    def _find_existing_item_for_sql(
        self,
        client: httpx.Client,
        sql: str,
        *,
        conversation_id: str | None = None,
        search_limit_conversations: int = 25,
    ) -> dict[str, str] | None:
        target = self._normalize_sql_for_match(sql)
        convs = self._conversations_for_mind(self._list_conversations(client))
        if conversation_id is not None:
            convs = [c for c in convs if c.get("id") == conversation_id]
        else:
            convs = convs[:search_limit_conversations]
        for c in convs:
            cid = c.get("id")
            if not cid:
                continue
            items = self._list_items(client, cid)
            for it in reversed(items):
                if not isinstance(it, dict) or it.get("role") != "assistant":
                    continue
                sq = it.get("sql_query")
                if not sq or sq == "None":
                    continue
                if self._normalize_sql_for_match(str(sq)) == target:
                    iid = it.get("id")
                    if iid:
                        return {"conversation_id": cid, "item_id": iid}
        return None

    def _find_exportable_item_in_conversation(
        self,
        client: httpx.Client,
        conversation_id: str,
        target_sql: str,
    ) -> str | None:
        items = self._list_items(client, conversation_id)
        for it in reversed(items):
            if not isinstance(it, dict) or it.get("role") != "assistant":
                continue
            sq = it.get("sql_query")
            if not sq or sq == "None":
                continue
            if self._normalize_sql_for_match(str(sq)) == target_sql:
                return it.get("id")
        for it in reversed(items):
            if not isinstance(it, dict) or it.get("role") != "assistant":
                continue
            sq = it.get("sql_query")
            if sq and sq != "None" and it.get("id"):
                return it["id"]
        return None

    # ── Query execution (private) ────────────────────────────────

    def _run_query_df(
        self,
        sql: str,
        *,
        conversation_id: str | None = None,
        reuse_existing: bool = True,
        poll_s: float = 0.25,
        max_wait_s: float = 300.0,
    ) -> pd.DataFrame:
        """Execute a SQL string via the Mind and return a DataFrame."""
        from io import StringIO
        import pandas as pd
        with self._client() as client:
            if reuse_existing:
                found = self._find_existing_item_for_sql(client, sql, conversation_id=conversation_id)
                if found:
                    csv_text = self._export_csv_text(client, found["conversation_id"], found["item_id"])
                    return pd.read_csv(StringIO(csv_text))
            if self.progress_fn:
                self.progress_fn("submitting query...")
            r = client.post(
                "/api/v1/responses/",
                json={
                    "model": self.mind_name,
                    "input": sql,
                    "tool_query": True,
                    "stream": False,
                },
            )
            r.raise_for_status()
            conv_id = conversation_id or self._pick_latest_conversation_id(client)
            target = self._normalize_sql_for_match(sql)
            deadline = time.time() + max_wait_s
            start = time.time()
            last_items: list[dict[str, Any]] | None = None
            while time.time() < deadline:
                last_items = self._list_items(client, conv_id)
                candidate_id = self._find_exportable_item_in_conversation(client, conv_id, target)
                if candidate_id:
                    csv_text = self._export_csv_text(client, conv_id, candidate_id)
                    return pd.read_csv(StringIO(csv_text))
                if self.progress_fn:
                    elapsed = int(time.time() - start)
                    self.progress_fn(f"waiting for query results... {elapsed}s")
                time.sleep(poll_s)
            raise RuntimeError(
                f"Timed out waiting for exportable assistant item in conversation {conv_id}. "
                f"Last items seen: {last_items}"
            )

    # ── Public: native queries ───────────────────────────────────

    def run_native_query_df(
        self,
        native_query: str,
        datasource_name: str,
        *,
        conversation_id: str | None = None,
        reuse_existing: bool = True,
        poll_s: float = 0.25,
        max_wait_s: float = 300.0,
    ) -> pd.DataFrame:
        """Run a native query against a datasource and return the result as a DataFrame.

        Wraps native_query as::

            SELECT * FROM <datasource_name>('<native_query>');
        """
        nq = native_query.strip().rstrip(";")
        nq_escaped = nq.replace("'", "''")
        wrapped = f"SELECT * FROM {datasource_name}('{nq_escaped}')"
        return self._run_query_df(
            wrapped,
            conversation_id=conversation_id,
            reuse_existing=reuse_existing,
            poll_s=poll_s,
            max_wait_s=max_wait_s,
        )
