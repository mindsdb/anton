from __future__ import annotations

import json
from unittest.mock import MagicMock

from anton.commands.ui import handle_explain
from anton.explainability import ExplainabilityCollector, ExplainabilityStore


def test_explainability_store_persists_latest_and_turn_file(tmp_path):
    store = ExplainabilityStore(tmp_path)
    collector = ExplainabilityCollector(store, turn=3, user_message="How did revenue change?")
    collector.add_scratchpad_step("Query monthly revenue")
    collector.add_query(
        datasource="warehouse.orders",
        sql="SELECT month, revenue FROM revenue_by_month",
        engine="postgres",
    )

    record = collector.finalize("Revenue increased 12% month over month.")

    latest = tmp_path / ".anton" / "explainability" / "latest.json"
    turn_file = tmp_path / ".anton" / "explainability" / "turn-0003.json"

    assert latest.is_file()
    assert turn_file.is_file()

    latest_payload = json.loads(latest.read_text())
    assert latest_payload["turn"] == 3
    assert latest_payload["sql_queries"][0]["datasource"] == "warehouse.orders"
    assert "queried warehouse.orders" in latest_payload["summary"].lower()
    assert record.summary == latest_payload["summary"]


def test_explainability_sql_shape_includes_datasources_and_queries(tmp_path):
    store = ExplainabilityStore(tmp_path)
    collector = ExplainabilityCollector(store, turn=4, user_message="What was monthly revenue?")
    collector.add_scratchpad_step("Query monthly revenue")
    collector.add_query(
        datasource="finance.monthly_revenue",
        sql="SELECT month, revenue FROM monthly_revenue ORDER BY month DESC",
        engine="snowflake",
    )

    record = collector.finalize("Revenue rose in March.")

    assert record.data_sources == [{"name": "finance.monthly_revenue", "engine": "snowflake"}]
    assert record.sql_queries == [
        {
            "datasource": "finance.monthly_revenue",
            "sql": "SELECT month, revenue FROM monthly_revenue ORDER BY month DESC",
            "engine": "snowflake",
            "status": "ok",
            "error_message": None,
        }
    ]
    assert "sql statement" in record.summary.lower()


def test_explainability_summary_without_queries_is_direct_answer(tmp_path):
    store = ExplainabilityStore(tmp_path)
    collector = ExplainabilityCollector(store, turn=1, user_message="What is Anton?")

    record = collector.finalize("Anton is MindsDB's autonomous AI coworker.")

    assert record.sql_queries == []
    assert (
        record.summary
        == "I answered directly from the conversation context without querying a datasource or generating SQL."
    )


def test_explainability_extracts_non_sql_sources_from_text(tmp_path):
    store = ExplainabilityStore(tmp_path)
    collector = ExplainabilityCollector(store, turn=2, user_message="Compare green coffee prices")
    collector.add_scratchpad_step("Fetch green coffee bean prices and compute roasting cost comparison")
    collector.add_sources_from_text(
        'See https://www.happymugcoffee.com/collections/green-coffee and https://burmancoffee.com/'
    )

    record = collector.finalize("Home roasting is much cheaper.")

    source_names = {source["name"] for source in record.data_sources}
    assert source_names == {"happymugcoffee.com", "burmancoffee.com"}
    assert "gathered information from" in record.summary.lower()


def test_handle_explain_prints_sections_for_latest_record(tmp_path):
    store = ExplainabilityStore(tmp_path)
    collector = ExplainabilityCollector(store, turn=5, user_message="What was revenue?")
    collector.add_scratchpad_step("Query monthly revenue")
    collector.add_query(
        datasource="finance.monthly_revenue",
        sql="SELECT month, revenue FROM monthly_revenue",
        engine="postgres",
    )
    collector.finalize("Revenue rose.")

    console = MagicMock()
    handle_explain(console, tmp_path)

    rendered = "\n".join(
        str(call.args[0]) for call in console.print.call_args_list if call.args
    )
    assert "Explain This Answer" in rendered
    assert "Summary" in rendered
    assert "Data Sources Used" in rendered
    assert "Generated SQL" in rendered
    assert "finance.monthly_revenue" in rendered
    assert "SELECT month, revenue FROM monthly_revenue" in rendered


def test_explainability_infers_sql_and_datasource_from_scratchpad_code(tmp_path):
    store = ExplainabilityStore(tmp_path)
    collector = ExplainabilityCollector(store, turn=6, user_message="Average revenue")
    collector.add_scratchpad_step("Average annual revenue over last 10 years in the dataset")
    collector.add_inferred_queries_from_code(
        """
import os
sql = \"\"\"
SELECT EXTRACT(YEAR FROM sale_date) AS year, AVG(revenue) AS avg_revenue
FROM sales
GROUP BY 1
ORDER BY 1
\"\"\"
host = os.environ["DS_POSTGRES_PROD_DB__HOST"]
cur.execute(sql)
"""
    )

    record = collector.finalize("Average revenue is stable.")

    assert record.data_sources == [{"name": "postgres-prod-db", "engine": None}]
    assert len(record.sql_queries) == 1
    assert "SELECT EXTRACT(YEAR FROM sale_date)" in record.sql_queries[0]["sql"]
    assert record.sql_queries[0]["datasource"] == "postgres-prod-db"
