"""What `find_cell_writes` holds, and what it lets through.

A missed write runs with no approval. A false hold costs an unattended answer
the full card timeout and then the answer itself. So each rule here has cells
it must hold and cells, often the same call shape, it must not.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from anton.core.backends.base import Cell
from anton.core.utils import scratchpad
from anton.core.utils.scratchpad import _card_prompt, _code_span, _fence, _refusal
from anton.core.utils.write_check import (
    CallWrite,
    CellWrites,
    FileChange,
    HttpWrite,
    HttpWriteRule,
    find_cell_writes,
    glob_match,
    parse_http_write_rules,
    snapshot_project_files,
    write_statements,
)

PG = (
    "import os, psycopg2\n"
    "conn = psycopg2.connect(host=os.environ['DS_POSTGRES_QA__HOST'])\n"
    "cur = conn.cursor()\n"
)
# A proposed fix as an f-string, as a cell builds one per supplier.
UPD = "f\"UPDATE suppliers SET lead_time_days = {d} WHERE supplier_id = '{s}'\""


@pytest.fixture
def snap(tmp_path: Path, monkeypatch):
    for rel in ("report.pdf", "stock.csv", "summary.md", "output/old.png",
                ".anton/artifacts/dash/index.html", ".anton/memory/notes.md",
                ".git/index", "docs/a.pdf"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    monkeypatch.chdir(tmp_path)
    return snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)


def _held(code: str, snap, *, earlier=()) -> bool:
    return find_cell_writes(code=code, earlier_code=earlier, snapshots=(snap,)).needs_approval()


def _statements(code: str, snap, *, earlier=()) -> list[str]:
    return [w.statement for w in find_cell_writes(code=code, earlier_code=earlier, snapshots=(snap,)).sql]


def _card(code: str, snap, *, earlier=()) -> str:
    return _card_prompt(find_cell_writes(code=code, earlier_code=earlier, snapshots=(snap,)))


def _fenced(card: str) -> list[str]:
    """The text of each code fence on a card, in order."""
    blocks: list[str] = []
    fence: str | None = None
    body: list[str] = []
    for line in card.splitlines():
        if fence is None and re.fullmatch(r"`{3,}|~{3,}", line):
            fence, body = line, []
        elif line == fence:
            blocks.append("\n".join(body))
            fence = None
        elif fence is not None:
            body.append(line)
    return blocks


# ── SQL statements ────────────────────────────────────────────────────────

@pytest.mark.parametrize("sql", [
    "UPDATE suppliers SET lead_time_days = 21 WHERE supplier_id = 'S005'",
    "update suppliers set a = 1",
    "  /* fix */ UPDATE s AS x SET a = 1",
    "UPDATE ONLY s SET a = 1",
    "DELETE FROM suppliers WHERE id = 5",
    "DELETE suppliers WHERE id = 5",
    "INSERT INTO t VALUES (1)",
    "INSERT IGNORE INTO t VALUES (1)",
    "MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN UPDATE SET a = s.a",
    "REPLACE INTO t VALUES (1)",
    "CREATE TABLE copy_of AS SELECT * FROM suppliers",
    "CREATE OR REPLACE VIEW v AS SELECT 1",
    "CREATE UNIQUE INDEX i ON t (a)",
    "ALTER TABLE t ADD COLUMN c int",
    "DROP TABLE IF EXISTS t",
    "TRUNCATE TABLE suppliers",
    "TRUNCATE suppliers RESTART IDENTITY",
    "GRANT SELECT ON t TO bob",
    "REVOKE SELECT ON t FROM bob",
    "WITH moved AS (DELETE FROM a RETURNING *) INSERT INTO b SELECT * FROM moved",
    "WITH s AS (SELECT 1) UPDATE t SET a = 1",
    "SELECT a INTO new_table FROM t",
    "COPY t FROM STDIN",
    "SELECT 1; DELETE FROM t",
    "BEGIN; UPDATE t SET a = 1; COMMIT;",
    "UPDATE {} SET {} = %s",
    # TRUNCATE on qualified, quoted, listed and IF EXISTS names.
    "TRUNCATE TABLE public.suppliers",
    "TRUNCATE public.suppliers",
    "TRUNCATE TABLE analytics.public.suppliers",
    'TRUNCATE TABLE "Suppliers"',
    "TRUNCATE TABLE `p.d.suppliers`",
    "TRUNCATE TABLE [dbo].[suppliers]",
    "TRUNCATE TABLE a, b",
    "TRUNCATE TABLE IF EXISTS suppliers",
    "TRUNCATE TABLE public.suppliers RESTART IDENTITY CASCADE",
    "TRUNCATE TABLE _x_._x_",
    # Dialect forms of the verbs above.
    "CREATE OR ALTER VIEW v AS SELECT 1",
    "CREATE OR ALTER PROCEDURE p AS SELECT 1",
    "CREATE OR REPLACE FORCE VIEW v AS SELECT 1 FROM dual",
    "CREATE BITMAP INDEX i ON t (a)",
    "CREATE DYNAMIC TABLE d TARGET_LAG = '1 hour' WAREHOUSE = w AS SELECT 1",
    "CREATE OR REPLACE DYNAMIC TABLE d AS SELECT 1",
    "CREATE ICEBERG TABLE t (a int)",
    "CREATE FOREIGN TABLE f (a int) SERVER s",
    "CREATE STAGE s",
    "DROP DYNAMIC TABLE d",
    "DROP EXTERNAL TABLE e",
    "ALTER EXTERNAL TABLE e REFRESH",
    "WITH s AS (SELECT * FROM t) SELECT * INTO backup FROM s",
    "# fix lead times\nUPDATE t SET a = 1",
    "DELETE suppliers",
    "DELETE FROM suppliers",
    "delete from suppliers",
    "DELETE TOP (10) suppliers",
    "DELETE TOP (10) FROM suppliers",
    "DELETE a, b FROM a JOIN b ON a.id = b.id",
    "DELETE LOW_PRIORITY QUICK FROM suppliers",
    "INSERT t VALUES (1)",
    "INSERT OVERWRITE TABLE t SELECT * FROM s",
    "INSERT ALL INTO t1 VALUES (1) INTO t2 VALUES (2) SELECT * FROM dual",
    "MERGE t USING s ON t.id = s.id WHEN MATCHED THEN DELETE",
    "GRANT analyst TO bob",
    "GRANT CREATE SESSION TO bob",
    "GRANT SELECT, INSERT ON db.* TO 'bob'@'%'",
    "UPDATE t SET arr[1] = 2",
    "UPDATE t JOIN s ON t.id = s.id SET t.a = s.a",
    # Writes outside the verbs above.
    "EXPLAIN ANALYZE UPDATE t SET a = 1",
    "EXPLAIN (ANALYZE, BUFFERS) DELETE FROM t",
    "LOAD DATA INFILE '/tmp/x.csv' INTO TABLE t",
    "LOAD DATA LOCAL INFILE 'x.csv' INTO TABLE t",
    "RENAME TABLE a TO b",
    "REPLACE t VALUES (1)",
    "REPLACE LOW_PRIORITY INTO t VALUES (1)",
    "UPSERT INTO t VALUES (1)",
    "DO $$ BEGIN UPDATE t SET a = 1; END $$",
    "COMMENT ON TABLE t IS 'supplier master'",
    "REFRESH MATERIALIZED VIEW mv",
    "CALL postgres_execute('pg', 'UPDATE suppliers SET a = 1')",
    "EXEC sp_executesql N'DELETE FROM t WHERE id = 1'",
    "EXECUTE IMMEDIATE 'TRUNCATE TABLE t'",
    # T-SQL runs a write on its own line with no `;` before it.
    "SELECT * FROM t\nDELETE FROM t WHERE id = 1",
    "SET NOCOUNT ON\nDELETE FROM t WHERE id = 1",
    "/* c */ # note\nUPDATE t SET a = 1",
    # SQL's own shape in sentence case, from a template, or with its rows apart.
    "Drop table suppliers",
    "Delete from orders where qty = 0",
    "Grant select on orders to analysts",
    "Create table t (a int)",
    "INSERT INTO t (a, b) VALUES",
    "DROP TABLE $t",
    "CREATE INDEX ON t (a)",
    "CREATE USER 'bob'@'%' IDENTIFIED BY 'pw'",
    "COPY INTO @s FROM t",
    "SELECT * INTO OUTFILE '/tmp/x' FROM t",
    "SELECT id INTO",
    "DELETE FROM t WHERE x = 1.",
    "MERGE INTO t USING s \ue0000\ue001",
    # Stored procedures and side effects: a procedure can change anything.
    "CALL refresh_totals()",
    "CALL refresh_totals",
    "call refresh_totals(1, 2)",
    "EXEC dbo.usp_rebuild_index",
    "EXEC usp_update_prices @region = 'EMEA'",
    "EXECUTE my_plan(1, 2)",
    "EXEC @status = dbo.usp_run",
    "EXEC (@sql)",
    # SELECT of a side-effecting function.
    "SELECT setval('orders_id_seq', 1000)",
    "select nextval('orders_id_seq')",
    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle'",
    "SELECT pg_cancel_backend(123)",
    "SELECT pg_reload_conf()",
    "SELECT lo_unlink(12345)",
    "SELECT lo_import('/etc/passwd')",
    "SELECT dblink_exec('conn', 'DELETE FROM t')",
    "SELECT pg_advisory_lock(1)",
    "SELECT set_config('statement_timeout', '0', false)",
    "SELECT GET_LOCK('lock1', 10)",
    "SELECT my_seq.NEXTVAL FROM dual",
    "WITH s AS (SELECT setval('q', 1)) SELECT * FROM s",
    "SELECT count(*) FROM (SELECT pg_cancel_backend(1)) x",
    "SELECT * FROM t INTO OUTFILE '/tmp/dump.csv'",
    "SELECT * INTO DUMPFILE '/tmp/one.bin' FROM t",
    "COPY t TO '/tmp/out.csv'",
    "COPY (SELECT * FROM t) TO PROGRAM 'gzip > /tmp/t.gz'",
])
def test_write_statements(sql):
    assert write_statements(sql)


def test_each_write_in_one_string_is_its_own_statement():
    assert write_statements("UPDATE suppliers SET a = 1 WHERE id = 5; DROP TABLE suppliers_audit") == (
        "UPDATE suppliers SET a = 1 WHERE id = 5", "DROP TABLE suppliers_audit",
    )
    assert write_statements("BEGIN; UPDATE t SET a = 1 WHERE id = 'S005'; DELETE FROM orders; COMMIT;") == (
        "UPDATE t SET a = 1 WHERE id = 'S005'", "DELETE FROM orders",
    )


@pytest.mark.parametrize("text", [
    "SELECT * FROM suppliers",
    "select max(lead_time) from s -- UPDATE in a comment",
    "SELECT 'DELETE FROM x' AS s",
    "SELECT * FROM t FOR UPDATE",
    "WITH s AS (SELECT 1) SELECT * FROM s FOR UPDATE",
    "WITH update_x AS (SELECT 1) SELECT * FROM update_x",
    "CREATE TEMP TABLE scratch AS SELECT 1",
    "CREATE TEMPORARY TABLE scratch AS SELECT 1",
    "CREATE LOCAL TEMP TABLE scratch AS SELECT 1",
    "COPY (SELECT * FROM t) TO STDOUT",
    "ALTER SESSION SET TIMEZONE = 'UTC'",
    "SHOW TABLES",
    "Update: lead times refreshed",
    "Delete the duplicate rows first.",
    "Create a table of totals per site",
    "Drop rows with missing quantity",
    "Insert chart below",
    "Truncate the text to 100 characters",
    "Update reorder points and set safety stock to 14 days for S005.",
    "S005 is 9 days late; update its lead time and set a new reorder point.",
    "Grant planners access to the dashboard.",
    "Revoke access from former staff",
    "Insert into the next review the values from Table 2.",
    "Delete from plan: orders where qty = 0",
    "Truncate labels.",
    "Replace the old chart with the new one",
    "Load data from the CSV",
    "Rename table columns to snake case",
    "Comment on the results",
    "Do this first",
    "With the new data, update the chart",
    "Copy the file from the share",
    "EXPLAIN UPDATE t SET a = 1",
    "SELECT a INTO @total FROM t",
    "SELECT * INTO #scratch FROM t",
    "DROP TEMPORARY TABLE scratch",
    "Truncate labels",
    "Delete everything",
    "Delete from backlog",
    "INSERT INTO #scratch VALUES (1)",
    "CALL postgres_execute('pg', 'SELECT 1')",
    # A report's sentences, labels and titles with a write verb first.
    "Merge the two supplier lists using the supplier ID.",
    "Merge duplicate SKUs using the master list",
    "Delete orders where the quantity is zero.",
    "Delete duplicates where SKU and date match.",
    "Copy data from the March export before the review.",
    "Create user accounts for the two new planners.",
    "Create view of late orders by region for the ops team.",
    "Create table of contents",
    "Create table summaries for each sheet in the workbook.",
    "Create role-based dashboards for each site.",
    "Grant access to Finance",
    "Revoke access from contractors",
    "Select suppliers into the watchlist for Q3.",
    "Insert the values into the Q3 template.",
    "Drop index funds below 2% weight",
    "Insert the table of lead times below the summary.",
    "Replace the table with a bar chart.",
    "Replace the default reorder point with 14 days of cover.",
    "Create stage gates for new suppliers.",
    "Replace missing values",
    "Replace nulls with zeros",
    "Select the folder to save into",
    # Lines of a paragraph, Markdown, an email and a page's JavaScript.
    "Supplier S005 is 9 days late on average.\nDelete orders where the quantity is zero.\nReview the plan on Friday.",
    "## Next steps\n\nDelete orders where the quantity is zero.\n\nReview S005.",
    "Hi Dana,\nDelete rows where qty is 0 before Friday.\nThanks",
    "<script>\nfunction reset() {\n  delete params.page;\n}\n</script>",
    # A plain read of ordinary functions stays free.
    "SELECT count(*), now(), sum(qty) FROM orders",
    "SELECT max(lead_time), avg(price) FROM suppliers",
    "SELECT nextval FROM settings",
    "COPY (SELECT * FROM t) TO STDOUT",
    # A capitalized verb, or a lowercase one not before `name(`, reads as prose.
    "Call the supplier about the late order",
    "Call refresh when the data lands",
    "execute the migration plan next week",
    "exec summary of the quarter",
    "Execute order 66",
    "CALL VOLUME BY HOUR",
    "EXEC summary attached",
    "Copy the chart to the deck",
    # A question mark, a heading's colon or a template's braces aren't a
    # driver's placeholder or ODBC's escape in prose, and "Begin" starts a sentence.
    "EXEC summary ?",
    "CALL ME: 555-0100",
    "{call me}",
    "Begin the review on Monday",
    "BEGIN HERE",
    "begin with the totals",
])
def test_reads_and_prose(text):
    assert write_statements(text) == ()


PYODBC = "import os, pyodbc\nconn = pyodbc.connect(os.environ['DSN'])\ncur = conn.cursor()\n"


@pytest.mark.parametrize("sql", [
    "EXEC p ?",
    "EXEC dbo.purge ?, ?",
    "exec purge ?",
    "EXEC p :1",
    "EXECUTE purge :days",
    "CALL purge ?",
    "{CALL p}",
    "{call p(?)}",
    "{CALL dbo.purge (?)}",
    "{? = CALL f(?)}",
    "BEGIN p; END;",
    "BEGIN pkg.proc(:1); END;",
    "begin purge_old(30); end;",
    "BEGIN DELETE FROM t WHERE id = 1; END;",
], ids=["exec-qmark", "exec-qmarks", "exec-lowercase-qmark", "exec-numbered", "execute-named", "call-qmark",
        "odbc-call", "odbc-call-group", "odbc-call-spaced-group", "odbc-function", "oracle-block",
        "oracle-package-block", "oracle-block-lowercase", "oracle-block-delete"])
def test_a_procedure_call_a_driver_sends_is_held(sql, snap):
    """pyodbc's `EXEC p ?`, Oracle's `:1`, ODBC's `{CALL p}` escape and an
    Oracle block `BEGIN p; END;` each run a procedure, which can change anything."""
    assert write_statements(sql, prose=False)
    assert _held(PYODBC + f"cur.execute({sql!r}, (1,))", snap)


@pytest.mark.parametrize("sql", [
    "BEGIN",
    "BEGIN;",
    "BEGIN TRANSACTION",
    "begin transaction;",
    "BEGIN TRAN",
    "BEGIN WORK",
    "BEGIN IMMEDIATE TRANSACTION",
    "BEGIN ISOLATION LEVEL SERIALIZABLE",
    "BEGIN READ ONLY",
    "BEGIN NULL; END;",
    "BEGIN; SELECT * FROM t; COMMIT;",
], ids=["begin", "begin-semicolon", "begin-transaction", "begin-transaction-lowercase", "begin-tran",
        "begin-work", "sqlite-immediate", "isolation-level", "read-only", "empty-block", "read-in-a-transaction"])
def test_a_transaction_start_is_no_write(sql, snap):
    assert write_statements(sql, prose=False) == ()
    assert not _held(PYODBC + f"cur.execute({sql!r})", snap)


def test_a_dash_comment_ends_at_a_carriage_return():
    """Postgres ends a `--` comment at a carriage return too, so what follows one runs."""
    assert write_statements("SELECT 1 -- note\r; DELETE FROM t") == ("DELETE FROM t",)


@pytest.mark.parametrize("text", [
    "SELECT 1\n# 1; DELETE FROM t",
    "# it's fine\nDELETE FROM t",
], ids=["hash-as-code", "hash-as-comment"])
def test_a_line_that_starts_with_hash_counts_as_code_and_as_a_comment(text):
    """Postgres reads `#` as XOR and MySQL as a comment: a write in either reading counts."""
    assert write_statements(text)


# ── SQL in cells ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("code", [
    PG + "cur.execute('UPDATE suppliers SET lead_time_days = 21 WHERE supplier_id = %s', ('S005',))",
    PG + 'cur.execute("""\n  UPDATE suppliers\n  SET lead_time_days = %s\n  WHERE supplier_id = %s\n""", (21, "S005"))',
    PG + "t = 'suppliers'\ncur.execute(f'DELETE FROM {t} WHERE id = 3')",
    PG + "tbl = 'suppliers'\ncur.execute('UPDATE ' + tbl + ' SET a = 1')",
    PG + "q = 'UPDATE suppliers SET a = 1'\ncur.execute(q)",
    PG + "from psycopg2 import sql\ncur.execute(sql.SQL('UPDATE {} SET {} = %s').format(sql.Identifier('s'), sql.Identifier('a')), (1,))",
    PG + "from sqlalchemy import text\nwith engine.begin() as c:\n    c.execute(text('DELETE FROM suppliers WHERE id = 1'))",
    PG + "def run(q):\n    cur.execute(q)\nrun('DROP TABLE suppliers')",
    PG + "import pandas as pd\ndf.to_sql('suppliers', engine, if_exists='replace')",
    PG + "stmts = ['UPDATE a SET x = 1', 'UPDATE b SET y = 2']\nfor s in stmts:\n    cur.execute(s)",
    "import os, duckdb\ncon = duckdb.connect()\ncon.execute(\"ATTACH 'postgres:...' AS pg\")\ncon.execute('UPDATE pg.suppliers SET a = 1')",
    "import duckdb\ncon = duckdb.connect('md:prod')\ncon.execute('DELETE FROM suppliers')",
    "from sqlalchemy import create_engine, text\ne = create_engine('duckdb:///md:prod')\nwith e.begin() as c:\n    c.execute(text('DELETE FROM suppliers'))",
    "import duckdb\ncon = duckdb.connect()\ncon.execute(\"(ATTACH 'postgres:dbname=prod' AS pg)\")\ncon.execute('UPDATE pg.suppliers SET a = 1')",
    PG + "s, t = 'public', 'suppliers'\ncur.execute(f'TRUNCATE TABLE {s}.{t}')",
    "from google.cloud import bigquery\nclient = bigquery.Client()\nclient.query('TRUNCATE TABLE `p.d.suppliers`').result()",
    PG + "sql = 'UPDATE suppliers '\nsql += 'SET lead_time_days = 21 '\nsql += \"WHERE supplier_id = 'S005'\"\ncur.execute(sql)",
    PG + "cur.execute(' '.join(['UPDATE suppliers', 'SET a = 1', 'WHERE id = 5']))",
    PG + "parts = ['UPDATE orders', \"SET status = 'cancelled'\"]\nparts.append(f'WHERE id = {order_id}')\ncur.execute(' '.join(parts))",
    PG + "override = None\nsql = override or 'DELETE FROM suppliers WHERE stale'\ncur.execute(sql)",
    PG + "class Q:\n    FIX = 'DELETE FROM t'\ncur.execute(Q.FIX)",
    PG + "cur.execute(b\"UPDATE orders SET status = 'x' WHERE id = 42\")",
    PG + "cur.execute(\"{} orders SET status = 'x' WHERE id = {}\".format('UPDATE', 42))",
    PG + "cur.execute('%s' % \"UPDATE orders SET status = 'x' WHERE id = 42\")",
    PG + "exec(\"cur.execute('UPDATE orders SET status = 1 WHERE id = 42')\")",
    PG + "q = 'UPDATE orders SET status = 1 WHERE id = 42'\neval('cur.execute(q)')",
    PG + "q = 'UPDATE orders SET status = 1 WHERE id = 42'\nexec('cur.execute(' + repr(q) + ')')",
    PG + "html = '<p>' + str(cur.execute('DELETE FROM t')) + '</p>'",
    PG + "stmts = [f\"UPDATE t SET a = {d} WHERE id = '{s}'\" for s, d in fixes]\nfor i, q in enumerate(stmts):\n    cur.execute(q)",
    PG + "m = {s: f\"UPDATE t SET a = {d} WHERE id = '{s}'\" for s, d in fixes}\nfor k, v in m.items():\n    cur.execute(v)",
    PG + "stmts = ['DELETE FROM a; DELETE FROM b']\n[cur.execute(y) for x in stmts for y in x.split(';')]",
    PG + "stmts = [f\"UPDATE t SET a = {d} WHERE id = '{s}'\" for s, d in fixes]\nlist(map(cur.execute, stmts))",
    PG + "import pandas as pd\ns = pd.Series([f\"UPDATE t SET a = {d}\" for d in days])\ns.apply(cur.execute)",
    PG + "import pandas as pd\ndf['sql'] = df.apply(lambda r: f\"UPDATE t SET a = {r.a}\", axis=1)\nfor q in df['sql']:\n    cur.execute(q)",
    PG + "def build(s, d):\n    return f\"UPDATE t SET a = {d} WHERE id = '{s}'\"\nprint(build('S5', 2))\ncur.execute(build('S6', 3))",
    PG + "class R:\n    def __init__(self):\n        self.sql = 'DELETE FROM t'\ncur.execute(R().sql)",
    PG + "a = {}\nb = a\nb['q'] = 'DELETE FROM t'\ncur.execute(a['q'])",
    PG + "sql = 'UPDATE t SET '\nfor k, v in changes.items():\n    sql += f'{k} = {v!r}, '\n"
         "sql = sql.rstrip(', ') + ' WHERE id = 5'\ncur.execute(sql)",
    PG + "q = 'DELETE FROM t'\ndef go():\n    cur.execute(q)\ngo()\nq = 'SELECT 1'",
], ids=["literal", "multi-line", "f-string", "concat", "named", "psycopg2-sql", "sqlalchemy-text",
        "pad-helper", "to_sql", "list-loop", "duckdb-attach", "motherduck", "motherduck-sqlalchemy",
        "attach-in-parens", "truncate-f-string-schema", "bigquery-backticks", "plus-equals",
        "join-literals", "join-appended-list", "or-default", "class-attribute", "bytes",
        "format-verb-argument", "percent-whole-statement", "exec-literal", "eval-name", "exec-repr",
        "write-inside-a-concatenation", "enumerate-loop", "items-loop", "nested-comprehension",
        "map-execute", "series-apply-execute", "apply-column-run", "same-cell-builder-run",
        "attribute-of-self", "entry-of-an-alias", "rebinding-built-on-itself", "helper-run-before-rebinding"])
def test_cells_that_write_a_database(code, snap):
    assert _held(code, snap)


@pytest.mark.parametrize("code", [
    "c = make()\nc.execute('UPDATE t SET a = 1')",
    "import psycopg2\nc = psycopg2.connect('host=db.example.com dbname=prod')\nc.cursor().execute('UPDATE t SET a = 1')",
    "import os\nfrom sqlalchemy import create_engine, text\ne = create_engine(os.environ['DATABASE_URL'])\n"
    "with e.begin() as c:\n    c.execute(text('DELETE FROM t'))",
], ids=["unknown-handle", "literal-credentials", "non-ds-variable"])
def test_a_write_is_held_with_no_saved_connection(code, snap):
    """Every conversation is checked: only an in-process handle is exempt."""
    assert _held(code, snap)


def test_each_statement_in_a_string_with_a_write_is_its_own_finding(snap):
    assert _statements(PG + 'cur.execute("UPDATE suppliers SET a = 1 WHERE id = 5; DROP TABLE suppliers_audit")', snap) == [
        "UPDATE suppliers SET a = 1 WHERE id = 5", "DROP TABLE suppliers_audit",
    ]
    script = "BEGIN;\nUPDATE suppliers SET a = 2 WHERE id = 5;\nDELETE FROM purchase_orders;\nCOMMIT;"
    assert _statements(PG + f"cur.execute('''{script}''')", snap) == [
        "BEGIN", "UPDATE suppliers SET a = 2 WHERE id = 5", "DELETE FROM purchase_orders", "COMMIT",
    ]


def test_a_join_that_widens_a_write_is_quoted_whole(snap):
    code = (PG + "parts = [\"DELETE FROM orders WHERE status = 'test'\"]\n"
            "parts.append(\"OR created_at < '2026-01-01'\")\ncur.execute(' '.join(parts))")
    assert _statements(code, snap) == ["DELETE FROM orders WHERE status = 'test' OR created_at < '2026-01-01'"]
    built = PG + "sql = 'DELETE FROM orders WHERE id = 42'\nsql += ' OR 1 = 1'\ncur.execute(sql)"
    assert _statements(built, snap) == ["DELETE FROM orders WHERE id = 42 OR 1 = 1"]


def test_a_format_fills_in_its_literal_arguments(snap):
    code = PG + "cur.execute(\"{} WHERE id = 42\".format(\"UPDATE orders SET status = 'x'\"))"
    assert _statements(code, snap) == ["UPDATE orders SET status = 'x' WHERE id = 42"]
    percent = PG + "cur.execute('DELETE FROM %s WHERE id = %d' % ('orders', 7))"
    assert _statements(percent, snap) == ["DELETE FROM orders WHERE id = 7"]


SA = (
    "import os, pandas as pd\nfrom sqlalchemy import create_engine, text\n"
    "engine = create_engine(os.environ['DS_POSTGRES_QA__URL'])\n"
)
PREVIEW = (
    "fix = \"UPDATE suppliers SET lead_time_days = 21 WHERE supplier_id = 'S005'\"\n"
    "stmts = ['UPDATE a SET x = 1', 'DELETE FROM b WHERE id = 2']\n"
    "print('Will run:', fix, stmts)\n"
)


@pytest.mark.parametrize("code", [
    "cur.execute(fix)",
    "with engine.begin() as conn:\n    conn.execute(text(fix))",
    "pd.read_sql(fix, engine)",
    "cur.execute(fix.format())",
    "cur.execute(fix.strip())",
    "cur.execute(fix + ' RETURNING *')",
    "cur.execute(f'{fix}')",
    "for s in stmts:\n    cur.execute(s)",
    "with engine.begin() as conn:\n    for s in stmts:\n        conn.execute(text(s))",
    "cur.execute(stmts[0])",
    "[cur.execute(s) for s in stmts]",
    "run(fix)",
], ids=["bare", "text", "read_sql", "format", "strip", "plus", "f-string", "loop", "loop-text",
        "index", "comprehension", "earlier-helper"])
def test_a_statement_from_an_earlier_cell_run_here(code, snap):
    """The preview cell only prints; this cell runs what it built."""
    earlier = (PG + SA + PREVIEW + "def run(q):\n    cur.execute(q)\n",)
    assert not _held(earlier[0], snap)
    assert _held(code, snap, earlier=earlier)


def test_an_earlier_helper_runs_a_global_this_cell_sets(snap):
    earlier = (PG + "def apply():\n    cur.execute(Q)\n    conn.commit()\n",)
    assert _held("Q = 'DELETE FROM suppliers WHERE id = 5'\napply()", snap, earlier=earlier)


def test_a_local_handle_inside_a_helper_does_not_hide_the_connection(snap):
    helper = (
        "import duckdb\ndef load_csv(path):\n    conn = duckdb.connect()\n"
        "    return conn.execute(f\"SELECT * FROM read_csv('{path}')\").df()\nstock = load_csv('stock.csv')\n"
    )
    pg = "import os, psycopg2\nconn = psycopg2.connect(host=os.environ['DS_POSTGRES_QA__HOST'])\n"
    update = "conn.cursor().execute(\"UPDATE suppliers SET lead_time_days = 21 WHERE supplier_id = 'S005'\")"
    assert _held(update, snap, earlier=(pg, helper))
    assert _held(helper + update, snap, earlier=(pg,))
    assert _held(update, snap, earlier=(pg, "import duckdb\nif False:\n    conn = duckdb.connect()\n"))
    assert _held(update, snap, earlier=(pg, "import duckdb\nif ready:\n    conn = duckdb.connect()\n"))
    assert _held(update, snap, earlier=(pg, "import duckdb\nr = [conn.execute('SELECT 1') for conn in [duckdb.connect()]]\n"))
    assert not _held(update, snap, earlier=(pg, "import duckdb\nconn = duckdb.connect()\n"))


def test_a_connection_from_an_earlier_cell(snap):
    assert _held("cur.execute('UPDATE t SET a = 1')", snap, earlier=(PG,))


@pytest.mark.parametrize("name", ["log", "logger"])
def test_a_cursor_with_a_logger_name_still_runs_sql(snap, name):
    """A logger-named root reads as logging, but `execute` on it runs SQL."""
    code = PG + f"{name} = conn.cursor()\n{name}.execute('UPDATE suppliers SET a = 1 WHERE id = 5')"
    assert _held(code, snap)
    assert not _held(PG + f"import logging\n{name} = logging.getLogger()\n{name}.info('UPDATE t SET a = 1')", snap)


@pytest.mark.parametrize("code", [
    PG + "cur.execute('SELECT * FROM suppliers')",
    PG + "sql = 'UPDATE suppliers SET a = 1'\nprint(sql)",
    PG + "import logging\nlogging.info('DELETE FROM t WHERE a = 1')",
    PG + "from pathlib import Path\nPath('fix.sql').write_text('UPDATE t SET a = 1')",
    PG + '"""DELETE FROM suppliers WHERE 1=1"""\ncur.execute("SELECT 1")',
    PG + "unused = 'UPDATE t SET a = 1'",
    PG + "import duckdb\ncon = duckdb.connect()\ncon.execute(\"CREATE TABLE stock AS SELECT * FROM read_csv('stock.csv')\")",
    PG + "import duckdb\nduckdb.sql('CREATE TABLE t AS SELECT 1')",
    PG + "import duckdb as ddb\nddb.sql('CREATE OR REPLACE TABLE t AS SELECT 1')",
    PG + "import sqlite3\nc = sqlite3.connect(':memory:')\nc.execute('CREATE TABLE t (a int)')\nc.execute('INSERT INTO t VALUES (1)')",
    PG + "import sqlite3\nwith sqlite3.connect('scratch.db') as c:\n    c.execute('DELETE FROM t')",
    PG + "from sqlalchemy import create_engine, text\ne = create_engine('sqlite:///:memory:')\nwith e.begin() as c:\n    c.execute(text('CREATE TABLE t (a int)'))",
    PG + "import sqlite3\ncon = sqlite3.connect(':memory:')\ndef q(s):\n    return con.execute(s).fetchall()\nq('CREATE TABLE t (a int)')",
    PG + "import sqlite3\nc = sqlite3.connect(':memory:')\nfor s in ['CREATE TABLE a (x int)', 'INSERT INTO a VALUES (1)']:\n    c.execute(s)",
    PG + "import sqlite3, pandas as pd\nc = sqlite3.connect(':memory:')\ndf.to_sql('t', c)",
    PG + "print('Update the chart and set the colors')",
    PG + "updates = [f\"UPDATE suppliers SET lead_time_days = {r.new} WHERE supplier_id = '{r.id}'\" for r in rows]\n"
         "print('\\n'.join(updates))",
    PG + "from pathlib import Path\nupdates = [f\"UPDATE suppliers SET a = {n} WHERE id = '{i}'\" for n, i in rows]\n"
         "Path('fix.sql').write_text(';\\n'.join(updates))",
    PG + "fixes = ['UPDATE a SET x = 1', 'UPDATE b SET y = 2']\nfor f in fixes:\n    print(f)",
    PG + "fixes = []\nfixes.append('UPDATE a SET x = 1')\nprint('\\n'.join(fixes))",
    PG + "import json\njson.dump({'sql': 'UPDATE a SET x = 1'}, open('fix.json', 'w'))",
    PG + "print(repr('DELETE FROM t WHERE id = 1'))",
    PG + "from reportlab.platypus import Paragraph\nstory = []\n"
         "story.append(Paragraph('Update reorder points and set safety stock to 14 days for S005.', style))",
    PG + "recs = ['Update the lead time for S005 and set a review date.', 'Expedite PO-1182.']\n"
         "story = [Paragraph(r, style) for r in recs]",
    PG + "pdf.multi_cell(0, 6, 'Update the safety stock and set alerts for late suppliers.')",
    PG + "pdf.cell(0, 6, 'S005 is 9 days late; update its lead time and set a new reorder point.')",
    PG + "pdf.cell(0, 6, 'Grant planners access to the dashboard.')",
    PG + "doc.add_paragraph('Insert into the next review the values from Table 2.')",
    PG + "import matplotlib.pyplot as plt\nplt.title('Delete from plan: orders where qty = 0')",
    "import duckdb\nnote = 'Attach the PDF to the email'\nprint(note)\ncon = duckdb.connect()\n"
    "con.execute(\"CREATE TABLE stock AS SELECT * FROM read_csv('stock.csv')\")",
    "import duckdb\ncon = duckdb.connect()\ncon.execute(\"CREATE TABLE stock AS SELECT * FROM read_csv('stock.csv')\")",
    "from sqlalchemy import create_engine, text\ne = create_engine('duckdb:///:memory:')\n"
    "with e.begin() as c:\n    c.execute(text('CREATE TABLE t AS SELECT 1'))",
    "from IPython.display import Markdown, display\nsql = 'UPDATE suppliers SET a = 1 WHERE id = 5'\n"
    "display(Markdown(f'```sql\\n{sql}\\n```'))",
    "from rich.console import Console\nConsole().print('DELETE FROM t WHERE id = 1')",
    # SQL only printed or saved, through loops, maps and helpers.
    PG + f"stmts = [{UPD} for s, d in fixes]\nfor i, s in enumerate(stmts, 1):\n    print(f'{{i}}. {{s}};')",
    PG + f"fixes_sql = {{s: {UPD} for s, d in fixes}}\nfor sid, sql in fixes_sql.items():\n    print(sid, sql)",
    PG + f"stmts = [{UPD} for s, d in fixes]\nfor sid, sql in zip(ids, stmts):\n    print(sid, sql)",
    PG + "stmts = list(map(lambda r: f\"UPDATE suppliers SET a = {r[1]} WHERE id = '{r[0]}'\", fixes))\n"
         "print('\\n'.join(stmts))",
    PG + f"def build_fix(s, d):\n    return {UPD}\nprint(build_fix('S005', 21))",
    PG + f"import pandas as pd\nstmts = [{UPD} for s, d in fixes]\n"
         "pd.DataFrame({'statement': stmts}).to_csv('proposed_fixes.csv', index=False)",
    PG + "import csv\nwith open('fixes.csv', 'w', newline='') as f:\n    w = csv.writer(f)\n"
         f"    for s, d in fixes:\n        w.writerow([s, {UPD}])",
    PG + "import pandas as pd\ndf = pd.DataFrame(fixes, columns=['s', 'd'])\n"
         "df['fix_sql'] = df.apply(lambda r: f\"UPDATE suppliers SET a = {r.d} WHERE id = '{r.s}'\", axis=1)\n"
         "df.to_csv('fixes.csv', index=False)",
    PG + "import textwrap\nsql = 'UPDATE suppliers SET a = 1 WHERE id = 5'\nprint(textwrap.indent(sql, '    '))",
    PG + "import html\nsql = \"UPDATE suppliers SET a = 1 WHERE id = 'S005'\"\n"
         "open('report.html', 'w').write('<pre>' + html.escape(sql) + '</pre>')",
    PG + "q = 'SELECT * FROM orders'\nq += \" WHERE id = 'A1'\"\nprint(q)\nstmts = []\n"
         "stmts.append('UPDATE t SET a = 1 WHERE id = 2')\nprint(stmts)\nstmts = ['SELECT 1']\nfor s in stmts:\n    cur.execute(s)",
    # Report prose in a conversation with no data connection.
    "story.append(Paragraph('Merge the two supplier lists using the supplier ID.', style))",
    "doc.add_paragraph('Insert the table of lead times below the summary.')",
    "slide.placeholders[1].text = 'Merge duplicate SKUs using the master list'",
    "ax.set_title('Drop index funds below 2% weight')",
    "text = '''Supplier S005 is 9 days late on average.\nDelete orders where the quantity is zero.\n"
    "Review the plan on Friday.'''\nstory.append(Paragraph(text, style))",
    "import markdown\nmd = '## Next steps\\n\\nDelete orders where the quantity is zero.\\n\\nReview S005.'\n"
    "open('r.html', 'w').write(markdown.markdown(md))",
    "from email.message import EmailMessage\nm = EmailMessage()\nm.set_content('Hi Dana,\\nDelete rows where qty is 0 before Friday.\\nThanks')",
], ids=["select", "printed", "logged", "written-to-file", "docstring", "unused-name", "duckdb-csv",
        "duckdb-module", "duckdb-alias", "sqlite-memory", "sqlite-file", "sqlalchemy-sqlite",
        "local-helper", "local-loop", "local-to_sql", "prose", "list-comprehension-printed",
        "list-comprehension-saved", "named-list-printed", "appended-then-printed", "json-dump",
        "repr-printed", "paragraph-prose", "paragraph-list-prose", "multi-cell-prose",
        "semicolon-prose", "grant-prose", "insert-prose", "delete-prose", "attach-prose",
        "duckdb-csv-unconnected", "duckdb-sqlalchemy", "markdown-shown", "console-print",
        "enumerate-printed", "dict-items-printed", "zip-printed", "map-lambda-printed",
        "same-cell-builder-printed", "dataframe-to-csv", "csv-writerow", "apply-column-to-csv",
        "textwrap-indent-printed", "html-escape-written", "name-bound-again-runs-a-select",
        "merge-paragraph-unconnected", "insert-paragraph-unconnected", "slide-text-unconnected",
        "chart-title-unconnected", "multi-line-paragraph", "markdown-page", "email-body"])
def test_cells_that_change_no_database(code, snap):
    assert not _held(code, snap)


PAGE = "<script>\\nfunction reset() {\\n  delete params.page;\\n}\\n</script>"


@pytest.mark.parametrize("code", [
    f"from jinja2 import Template\nopen('late.html', 'w').write(Template('{PAGE}<h1>{{{{ t }}}}</h1>').render(t='Late'))",
    f"from weasyprint import HTML\nHTML(string='{PAGE}').write_pdf('r.pdf')",
    f"from bs4 import BeautifulSoup\nsoup = BeautifulSoup('{PAGE}', 'html.parser')",
    f"async def render():\n    await page.set_content('{PAGE}')\n    await page.pdf(path='r.pdf')",
], ids=["jinja2", "weasyprint", "beautifulsoup", "playwright"])
def test_an_html_page_whose_script_deletes_a_key_raises_no_card(code, snap):
    """JavaScript's `delete params.page` on a line of its own is not SQL."""
    assert not _held(code, snap)


DUCK_CSV = "import duckdb\ncon = duckdb.connect()\ncon.execute(\"CREATE TABLE stock AS SELECT * FROM read_csv('stock.csv')\")"


@pytest.mark.parametrize("note", [
    "Attach", "attach", "Attach: Q3 supplier report.pdf", "\nattach(handler);\n", "Attach 2 files",
    "Attach 'report.pdf' and send it to Dana.", "attach (optional) the raw export", "Attach - the PDF",
    "Attach #3", "Attach @dana", "Attach {name} to the ticket", "attach $FILE to it",
])
def test_the_word_attach_in_the_pad_keeps_duckdb_local(note, snap):
    assert not _held(DUCK_CSV, snap, earlier=(f"note = {note!r}\nprint(note)",))


@pytest.mark.parametrize("attach, held", [
    ("ATTACH ':memory:' AS m", False),
    ("ATTACH 'other.duckdb' AS o", False),
    ("ATTACH 'local.db' AS l (TYPE sqlite)", False),
    ("ATTACH DATABASE 'data/scratch.db' AS s", False),
    ("ATTACH 'sqlite:local.db' AS l", False),
    ("ATTACH 'postgres:dbname=prod' AS pg", True),
    ("ATTACH 'dbname=prod host=db' AS pg (TYPE postgres)", True),
    ("ATTACH '' AS my (TYPE mysql)", True),
    ("attach 'md:prod' as m", True),
    ("ATTACH 's3://bucket/x.duckdb' AS r", True),
])
def test_attaching_a_database_in_the_process_keeps_the_handle_local(attach, held, snap):
    """An in-memory database, a local file or a SQLite one opens in the pad's
    own process, as `duckdb.connect` on one does."""
    code = f"import duckdb\ncon = duckdb.connect()\ncon.execute({attach!r})\ncon.execute('DELETE FROM x.t')"
    assert _held(code, snap) is held
    built = "import duckdb, os\ncon = duckdb.connect()\ncon.execute(f\"ATTACH '{os.environ['DB']}' AS x\")\n"
    assert _held(built + "con.execute('DELETE FROM x.t')", snap), "a target picked at run time counts"


# ── Where a string goes decides how it reads ──────────────────────────────
#
# A SQL runner runs a statement whatever its case or closing period. A
# report, a template or a function the check can't follow may get English,
# so only there does sentence case or a period make a string prose.

@pytest.mark.parametrize("sql", [
    "delete suppliers",
    "Delete suppliers where id = 7",
    "Drop view v",
    "Drop index i",
    "Drop database d",
    "(SELECT 1)\nDELETE FROM suppliers",
    "Delete from suppliers",
    "Truncate suppliers",
    "Update suppliers set lead_time_days = 21 where id = 'S005'.",
])
def test_a_runner_runs_a_statement_whatever_its_case(sql, snap):
    """SQL Server and Oracle run `delete suppliers`; each of these runs as written."""
    assert write_statements(sql) == (), "as prose, it reads as English"
    assert write_statements(sql, prose=False)
    assert _held(PG + f"cur.execute({sql!r})", snap)


@pytest.mark.parametrize("code, earlier", [
    (PG + "q = 'Delete from suppliers'\ncur.execute(q)", ()),
    ("cur.execute(q)", (PG + "q = 'Truncate suppliers'\nprint(q)",)),
    (PG + "from sqlalchemy import text\nwith engine.begin() as c:\n    c.execute(text('Drop view v'))", ()),
    (PG + "import pandas as pd\npd.read_sql('Delete suppliers where id = 7', engine)", ()),
    (PG + "cur.execute(query='Delete from suppliers')", ()),
    (PG + "list(map(cur.execute, ['Delete from a', 'Truncate b']))", ()),
    (PG + "for s in ['Delete from a', 'Truncate b']:\n    cur.execute(s)", ()),
    (PG + "q = 'Delete from suppliers'\ndoc.add_paragraph(q)\nif ready:\n    cur.execute(q)", ()),
    (PG + "def run(q):\n    cur.execute(q)\nrun('Delete from suppliers')", ()),
    ("run('Delete from suppliers')", (PG + "def run(q):\n    cur.execute(q)\n",)),
], ids=["name", "earlier-cell-name", "sqlalchemy-text", "read_sql", "query-keyword", "map-execute",
        "loop-execute", "report-then-execute", "pad-helper", "earlier-helper"])
def test_a_sentence_case_statement_a_runner_gets_is_held(code, earlier, snap):
    """A use that runs the string decides, even after one that only shows it."""
    assert _held(code, snap, earlier=earlier)


def test_a_sentence_case_attach_a_runner_gets_reaches_outside_the_process(snap):
    duck = "import duckdb\ncon = duckdb.connect()\n"
    attach = "\"Attach 'postgres:dbname=x host=y' AS pg\""
    assert _held(duck + f"con.execute({attach})\ncon.execute('DELETE FROM pg.suppliers')", snap)
    assert _held(duck + f"q = {attach}\ncon.execute(q)\ncon.execute('DELETE FROM pg.suppliers')", snap)
    assert not _held(duck + f"pdf.cell(0, 6, {attach})\ncon.execute('DELETE FROM t')", snap), (
        "the same words in a report keep the handle local"
    )
    assert not _held(duck + "con.execute(\"Attach ':memory:' AS m\")\ncon.execute('DELETE FROM m.t')", snap)


@pytest.mark.parametrize("code", [
    PG + "story.append(Paragraph('Merge the two supplier lists using the supplier ID.', style))",
    PG + "doc.add_paragraph('Insert the table of lead times below the summary.')",
    PG + "story.append(Paragraph('Delete orders where the quantity is zero.', style))",
    PG + "story.append(Paragraph('Truncate labels', style))",
    PG + "pdf.cell(0, 6, 'Truncate labels')",
    PG + "q = 'Delete orders where the quantity is zero.'\nstory.append(Paragraph(q, style))",
    PG + f"from weasyprint import HTML\nHTML(string='{PAGE}').write_pdf('r.pdf')",
    PG + "import markdown\nmd = '## Next steps\\n\\nDelete orders where the quantity is zero.'\n"
         "open('r.html', 'w').write(markdown.markdown(md))",
    PG + "from email.message import EmailMessage\nm = EmailMessage()\n"
         "m.set_content('Hi Dana,\\nDelete rows where qty is 0 before Friday.\\nThanks')",
    PG + "cur.execute('SELECT * FROM notes WHERE body = %s', ('Delete orders where the quantity is zero.',))",
], ids=["merge-paragraph", "insert-add-paragraph", "delete-paragraph", "truncate-paragraph", "truncate-cell",
        "named-paragraph", "html-script", "markdown", "email", "execute-parameter"])
def test_prose_a_report_gets_stays_free_beside_a_connection(code, snap):
    """Sentence case and a closing period still read as English where the
    string goes to a report, a page, an email or an execute's parameters."""
    assert not _held(code, snap)


def test_a_statement_name_bound_again_shows_only_what_runs(snap):
    code = PG + "q = 'SELECT * FROM orders'\nq += \" WHERE id = 'A1'\"\ncur.execute(q)\nq = 'DELETE FROM orders'\ncur.execute(q)"
    assert set(_statements(code, snap)) == {"DELETE FROM orders"}


def test_a_long_concatenation_does_not_exhaust_recursion(snap):
    code = "html = " + " + ".join(f"'<td>{i}</td>'" for i in range(3000)) + "\nprint(html)"
    assert not _held(code, snap)


def test_a_concatenation_is_read_once(snap, monkeypatch):
    """Each link of a `+` chain is read once, not once per enclosing link."""
    from anton.core.utils import write_check

    calls = 0
    read = write_check._sql_text

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return read(*args, **kwargs)

    monkeypatch.setattr(write_check, "_sql_text", counting)
    terms = 2000
    code = "html = " + " + ".join(f"'<td>{i}</td>'" for i in range(terms)) + "\nprint(html)"
    assert not _held(code, snap)
    assert calls < 2 * terms


@pytest.mark.parametrize("item", ["UPDATE t SET a = {i} WHERE id = {i}", "Delete orders where the quantity is {i}."])
def test_each_read_of_a_list_is_read_once(item, snap, monkeypatch):
    """Each value appended to a list asks what every read of the list holds,
    so each read is answered once per check, not once per value."""
    from anton.core.utils import write_check

    calls = 0
    bindings = write_check._Reader.bindings_of

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return bindings(*args, **kwargs)

    monkeypatch.setattr(write_check._Reader, "bindings_of", counting)
    items = 300
    code = "recs = []\n" + "".join(f"recs.append({item.format(i=i)!r})\n" for i in range(items))
    assert not _held(code + "for r in recs:\n    print(r)\n", snap)
    assert calls < 10 * items


def test_an_earlier_cell_is_parsed_once_across_checks(snap, monkeypatch):
    """Every check reads all the earlier cells, but parses each one only the first time."""
    from anton.core.utils import write_check

    monkeypatch.setattr(write_check, "_EARLIER_CELLS", write_check._EarlierCells(budget=1_000_000))
    parsed: list[str] = []
    parse = write_check._parse

    def counting(code: str):
        parsed.append(code)
        return parse(code)

    monkeypatch.setattr(write_check, "_parse", counting)
    earlier = tuple(f"from os import remove as rm{i}" for i in range(50))
    assert _held("rm0('report.pdf')", snap, earlier=earlier)
    assert len(parsed) == 51
    parsed.clear()
    assert _held("rm7('report.pdf')", snap, earlier=earlier)
    assert parsed == ["rm7('report.pdf')"], "a later check parses only its own cell"


def test_the_earlier_cell_cache_keys_a_cell_by_its_place_and_keeps_to_its_budget():
    from anton.core.utils.write_check import _EarlierCells

    cache = _EarlierCells(budget=12, idle=0.0)
    first = cache.get(code="a = 1", index=0)
    assert cache.get(code="a = 1", index=0) is first
    moved = cache.get(code="a = 1", index=1)
    assert moved is not first and (first.index, moved.index) == (0, 1)
    # 15 characters against a budget of 12, every cell idle: the cell read
    # least recently goes.
    cache.get(code="b = 2", index=2)
    assert cache.get(code="a = 1", index=0) is not first
    assert cache.get(code="def (", index=0) is None, "a cell that doesn't parse is skipped"


def test_a_history_past_the_cache_budget_keeps_the_cells_it_cached(snap, monkeypatch):
    """Each check reads the cells in order, so the cells it cached stay
    while checks still read them, rather than each new one pushing out the
    one the next check reads first."""
    from anton.core.utils import write_check

    earlier = tuple(f"from os import remove as rm{i:02d}" for i in range(50))
    monkeypatch.setattr(write_check, "_EARLIER_CELLS", write_check._EarlierCells(budget=len(earlier[0]) * 20))
    parsed: list[str] = []
    parse = write_check._parse

    def counting(code: str):
        parsed.append(code)
        return parse(code)

    monkeypatch.setattr(write_check, "_parse", counting)
    assert _held("rm00('report.pdf')", snap, earlier=earlier)
    first = len(parsed)
    parsed.clear()
    assert _held("rm07('report.pdf')", snap, earlier=earlier)
    assert first == 51 and len(parsed) == 31, "the 20 cells that fit stay cached"


# ── Files ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("code", [
    "import os\nos.remove('report.pdf')",
    "import os\nos.unlink('./report.pdf')",
    "import os\nos.rename('report.pdf', 'old.pdf')",
    "import shutil\nshutil.move('report.pdf', 'archive/report.pdf')",
    "from pathlib import Path\nPath('report.pdf').unlink()",
    "from pathlib import Path\nPath('report.pdf').rename('old.pdf')",
    "from pathlib import Path\npdf = next(Path('.').glob('*.pdf'))\npdf.rename(pdf.with_name('old.pdf'))",
    "from pathlib import Path\nsorted(Path('.').glob('*.pdf'))[0].rename('old.pdf')",
    "from pathlib import Path\n[p for p in Path('.').glob('*.pdf')][0].replace('old.pdf')",
    "from pathlib import Path\nPath('report.pdf').rename(target='old.pdf')",
    "import glob, os\nfor f in glob.glob('*.pdf'):\n    os.remove(f)",
    "import glob, os\nlist(map(os.remove, glob.glob('*.pdf')))",
    "import os\nfor f in os.listdir('.'):\n    os.remove(f)",
    "import os\nos.remove(os.path.abspath('report.pdf'))",
    "import shutil\nshutil.rmtree('output')",
    "import shutil\nshutil.rmtree('.')",
    "import os, shutil\nshutil.rmtree(os.getcwd())",
    "import shutil\nshutil.rmtree('.git')",
    "import shutil\nshutil.rmtree('.anton/artifacts/dash')",
    "import os\nos.rmdir('output')",
    "import subprocess\nsubprocess.run(['rm', '-f', 'report.pdf'])",
    "import os\nos.system('mv report.pdf old.pdf')",
    "import os as o\no.remove('report.pdf')",
    "from shutil import rmtree\nrmtree('output')",
    "import os\nos.remove(os.environ['SOME_PATH'])",
    "import os\nos.chdir('docs')\nos.remove('a.pdf')",
    "import os\ndef drop(p):\n    os.remove(p)\ndrop('report.pdf')",
    "import os\nos.remove('.anton/memory/notes.md')",
    "import os\ndrop = lambda p: os.remove(p)\ndrop('report.pdf')",
    "import os\ndef drop(p='report.pdf'):\n    os.remove(p)\ndrop()",
], ids=["os.remove", "os.unlink", "os.rename", "shutil.move", "Path.unlink", "Path.rename",
        "next-glob", "sorted-glob", "comprehension", "target-kw", "glob-loop", "map", "listdir",
        "absolute", "rmtree-folder", "rmtree-root", "rmtree-cwd", "rmtree-git", "artifact", "rmdir",
        "subprocess-rm", "os.system-mv", "aliased-os", "from-import", "env-path", "after-chdir",
        "helper-param", "anton-host-files", "lambda-param", "helper-default"])
def test_cells_that_delete_or_rename_existing_files(code, snap):
    assert _held(code, snap)


@pytest.mark.parametrize("code", [
    "import pandas as pd\ndf = pd.read_csv('stock.csv')\ndf = df.rename(columns={'a': 'b'})",
    "import pandas as pd\ndf = pd.read_csv('stock.csv')\ns = df['qty'].rename('quantity')",
    "import pandas as pd\ndf = pd.read_csv('stock.csv')\ndf['site'] = df['site'].replace({'A': 'B'})",
    "import pandas as pd\ndf = pd.read_csv('stock.csv')\ndf = df.rename(str.lower, axis=1)",
    "name = 'report.pdf'.replace('.pdf', '.txt')",
    "open('summary2.md.tmp', 'w').write('x')\nimport os\nos.replace('summary2.md.tmp', 'summary2.md')",
    "import tempfile, os\ntmp = tempfile.NamedTemporaryFile(delete=False)\nos.remove(tmp.name)",
    "import tempfile, os\nfd, p = tempfile.mkstemp()\nos.remove(p)",
    "import tempfile, os, shutil\nwith tempfile.TemporaryDirectory() as d:\n    os.remove(os.path.join(d, 'x'))",
    "import os\nos.remove('/tmp/scratch.csv')",
    "import os\nos.remove(os.path.join('output', 'chart_tmp.png'))",
    "from pathlib import Path\n(Path('output') / 'chart_tmp.png').unlink()",
    "import shutil\nshutil.move('chart.png', 'output')",
    "import os\nos.remove('chart.png')",
    "import os\ndef write_atomic(path, text):\n    tmp = path + '.tmp'\n    open(tmp, 'w').write(text)\n    os.replace(tmp, path)\nwrite_atomic('lead_times.md', 's')",
    "import os\nfor f in os.listdir('pages_tmp'):\n    os.remove(os.path.join('pages_tmp', f))",
    "open('new_summary.md', 'w').write('x')",
    "from pathlib import Path\np = Path.cwd() / ('beat-' + TOKEN + '.json')\ntmp = p.with_suffix('.tmp')\ntmp.write_text('x')\ntmp.replace(p)",
    "import os\nown = 'move-in-' + str(os.getpid())\nopen(own, 'w').close()\nfor path in (own, own + '.moved'):\n    os.remove(path)",
    "import os\nown = []\nfor i in range(3):\n    own.append(f'part_{i}.csv')\n    open(own[-1], 'w').write('x')\nfor p in own:\n    os.remove(p)",
    "import os\nos.makedirs('.anton/skill_drafts/new', exist_ok=True)\n"
    "open('.anton/skill_drafts/new/SKILL.md.tmp', 'w').write('x')\n"
    "os.replace('.anton/skill_drafts/new/SKILL.md.tmp', '.anton/skill_drafts/new/SKILL.md')",
    "import os\nos.makedirs('.anton/artifacts/new', exist_ok=True)\n"
    "open('.anton/artifacts/new/data.json.tmp', 'w').write('x')\n"
    "os.replace('.anton/artifacts/new/data.json.tmp', '.anton/artifacts/new/data.json')",
    "import os\nos.system('rm -f part_0.csv part_1.csv > summary.md 2>&1')",
    "import subprocess\nsubprocess.run(['ls', '-la'])\nsubprocess.run(f'echo {TOKEN}', shell=True)",
    "from pathlib import Path\nfor p in Path('.').rglob('*.partial'):\n    p.unlink()",
    "import os\ndef add(acc, name):\n    acc.append(name)\nown = []\nadd(own, 'part_0.csv')\n"
    "open('part_0.csv', 'w').write('x')\nfor p in own:\n    os.remove(p)",
    "import subprocess, sys\nsubprocess.run('ls -la'.split())\nsubprocess.run(['bash'], input='echo hi', text=True)\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install', 'x'])\nsubprocess.run(['bash', 'run.sh'])",
], ids=["df-rename-columns", "series-rename", "series-replace", "df-rename-callable", "str-replace",
        "temp-renamed-into-place", "tempfile-name", "mkstemp", "tempdir", "outside-project",
        "new-file-in-existing-folder", "new-path-in-existing-folder", "move-into-folder",
        "new-file", "helper-temp-file", "cleanup-new-folder", "new-summary",
        "with-suffix-temp", "tuple-loop-cleanup", "own-list-cleanup", "own-skill-draft",
        "artifact-temp-renamed", "shell-rm-new-files", "other-commands", "rglob-own-files",
        "own-list-filled-by-a-helper", "commands-that-delete-nothing"])
def test_cells_that_change_no_existing_file(code, snap):
    assert not _held(code, snap)


@pytest.mark.parametrize("code", [
    "open('summary.md', 'w').write('new')",
    "import os\nos.remove('summary.md')\nopen('summary.md', 'w').write('new')",
    "from pathlib import Path\nout = Path('summary.md')\nout.unlink(missing_ok=True)\nout.write_text('new')",
    "import os\nopen('summary.md.tmp', 'w').write('x')\nos.replace('summary.md.tmp', 'summary.md')",
    "import os\nif os.path.exists('summary.md'):\n    os.remove('summary.md')\nwith open('summary.md', 'w') as f:\n    f.write('new')",
    "import os\nos.remove('summary.md')\nos.replace('summary.md.tmp', 'summary.md')",
], ids=["open-w", "delete-then-write", "unlink-then-write-text", "temp-renamed-onto-it", "exists-remove-write",
        "delete-then-replace"])
def test_an_overwrite_of_a_file_the_turn_began_with_is_held_and_named_once(code, snap):
    """A delete that a write always follows is the overwrite, and the write
    after it names the same file, so the card names it once."""
    writes = find_cell_writes(code=code, earlier_code=(), snapshots=(snap,))
    assert writes.needs_approval()
    assert [(c.action, c.paths) for c in writes.held_files] == [("overwrite", ("summary.md",))]
    assert "This step would overwrite:\n\n```\nsummary.md\n```" in _card_prompt(writes)


# ── Overwrites ────────────────────────────────────────────────────────────

@pytest.fixture
def office(tmp_path: Path, monkeypatch):
    """A project of the files an analyst's turn rewrites."""
    for rel in ("stock.csv", "orders.parquet", "budget.xlsx", "memo.docx", "deck.pptx", "chart.png",
                "log.txt", "warehouse.db", "analytics.duckdb", "docs/a.pdf",
                ".anton/artifacts/dash/index.html", ".anton/skill_drafts/weekly/SKILL.md"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    monkeypatch.chdir(tmp_path)
    return snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)


def _changes(code: str, snapshot, *, earlier=()) -> list[tuple[str, tuple[str, ...], bool]]:
    writes = find_cell_writes(code=code, earlier_code=earlier, snapshots=(snapshot,))
    return [(c.action, c.paths, c.held) for c in writes.files]


PD = "import pandas as pd\ndf = pd.read_csv('stock.csv')\n"


@pytest.mark.parametrize("code, path", [
    ("open('stock.csv', 'w').write('x')", "stock.csv"),
    ("open('stock.csv', 'w+').write('x')", "stock.csv"),
    ("open('stock.csv', 'wb').write(b'x')", "stock.csv"),
    ("open('stock.csv', 'r+').write('x')", "stock.csv"),
    ("open('stock.csv', mode='rb+').write(b'x')", "stock.csv"),
    ("open(file='stock.csv', mode='w').write('x')", "stock.csv"),
    ("with open('stock.csv', 'w', newline='') as f:\n    f.write('x')", "stock.csv"),
    ("from pathlib import Path\nPath('stock.csv').write_text('x')", "stock.csv"),
    ("from pathlib import Path\nPath('chart.png').write_bytes(b'x')", "chart.png"),
    ("from pathlib import Path\nwith Path('stock.csv').open('w') as f:\n    f.write('x')", "stock.csv"),
    (PD + "df.to_csv('stock.csv', index=False)", "stock.csv"),
    (PD + "df.to_csv(path_or_buf='stock.csv')", "stock.csv"),
    (PD + "df.to_excel('budget.xlsx', sheet_name='Q3')", "budget.xlsx"),
    (PD + "df.to_parquet('orders.parquet')", "orders.parquet"),
    (PD + "df.to_json('log.txt', orient='records')", "log.txt"),
    (PD + "df.to_feather('orders.parquet')", "orders.parquet"),
    (PD + "df.to_pickle('log.txt')", "log.txt"),
    (PD + "df.to_html('log.txt')", "log.txt"),
    (PD + "with pd.ExcelWriter('budget.xlsx') as w:\n    df.to_excel(w, sheet_name='A')", "budget.xlsx"),
    ("import matplotlib.pyplot as plt\nplt.savefig('chart.png', dpi=150)", "chart.png"),
    ("import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nfig.savefig(fname='chart.png')", "chart.png"),
    ("import shutil\nshutil.copy('docs/a.pdf', 'chart.png')", "chart.png"),
    ("import shutil\nshutil.copy2('docs/a.pdf', 'chart.png')", "chart.png"),
    ("import shutil\nshutil.copyfile('docs/a.pdf', 'chart.png')", "chart.png"),
    ("import os\nos.replace('new.csv', 'stock.csv')", "stock.csv"),
    ("import os\nos.rename('new.csv', 'stock.csv')", "stock.csv"),
    ("import shutil\nshutil.move('new.csv', 'stock.csv')", "stock.csv"),
    ("import json\njson.dump({'a': 1}, open('log.txt', 'w'))", "log.txt"),
    ("import pickle\nwith open('log.txt', 'wb') as f:\n    pickle.dump({'a': 1}, f)", "log.txt"),
    ("import openpyxl\nwb = openpyxl.load_workbook('budget.xlsx')\nwb['A']['B2'] = 7\nwb.save('budget.xlsx')", "budget.xlsx"),
    ("from docx import Document\nd = Document('memo.docx')\nd.add_paragraph('x')\nd.save('memo.docx')", "memo.docx"),
    ("from pptx import Presentation\np = Presentation('deck.pptx')\np.save('deck.pptx')", "deck.pptx"),
    ("import glob, pandas as pd\nfor f in glob.glob('*.csv'):\n    pd.read_csv(f).dropna().to_csv(f, index=False)",
     "stock.csv"),
    ("import os\nos.chdir('docs')\nopen('a.pdf', 'wb').write(b'x')", "docs/a.pdf"),
], ids=["w", "w+", "wb", "r+", "rb+-keyword", "file-keyword", "with-open", "write_text", "write_bytes",
        "path-open", "to_csv", "to_csv-keyword", "to_excel", "to_parquet", "to_json", "to_feather", "to_pickle",
        "to_html", "excel-writer", "savefig", "fig-savefig-keyword", "copy", "copy2", "copyfile", "replace-onto",
        "rename-onto", "move-onto", "json-dump", "pickle-dump",
        "openpyxl-save", "docx-save", "pptx-save", "glob-rewrite-in-place", "after-chdir"])
def test_every_write_that_replaces_a_file_the_turn_began_with_is_held(code, path, office):
    writes = find_cell_writes(code=code, earlier_code=(), snapshots=(office,))
    assert ("overwrite", (path,)) in [(c.action, c.paths) for c in writes.held_files]
    assert f"This step would overwrite:\n\n```\n{path}\n```" in _card_prompt(writes)


@pytest.mark.parametrize("code", [
    "open('log.txt', 'a').write('x')",
    "open('log.txt', 'a+').write('x')",
    "open('log.txt', 'ab').write(b'x')",
    "from pathlib import Path\nwith Path('log.txt').open('a') as f:\n    f.write('x')",
    PD.replace("stock.csv", "log.txt") + "df.to_csv('log.txt', mode='a', header=False)",
    PD + "with pd.ExcelWriter('budget.xlsx', mode='a', if_sheet_exists='new') as w:\n    df.to_excel(w)",
], ids=["a", "a+", "ab", "path-open-a", "to_csv-append", "excel-writer-append"])
def test_an_append_is_reported_but_not_held(code, office):
    writes = find_cell_writes(code=code, earlier_code=(), snapshots=(office,))
    assert not writes.needs_approval()
    assert [(c.action, c.held) for c in writes.files] == [("append", False)]


def test_an_exclusive_create_is_no_overwrite(office):
    """`x` fails on a file that exists, so it can't replace one."""
    assert _changes("open('stock.csv', 'x').write('x')", office) == []


@pytest.mark.parametrize("code, path", [
    ("open('.anton/artifacts/dash/index.html', 'w').write('<h1>v2</h1>')", ".anton/artifacts/dash/index.html"),
    ("from pathlib import Path\nPath('.anton/skill_drafts/weekly/SKILL.md').write_text('v2')",
     ".anton/skill_drafts/weekly/SKILL.md"),
    ("import os\nopen('.anton/artifacts/dash/index.html.tmp', 'w').write('v2')\n"
     "os.replace('.anton/artifacts/dash/index.html.tmp', '.anton/artifacts/dash/index.html')",
     ".anton/artifacts/dash/index.html"),
], ids=["artifact", "skill-draft", "artifact-temp-renamed-into-place"])
def test_an_overwrite_under_the_agents_folders_is_not_held(code, path, office):
    assert _changes(code, office) == [("overwrite", (path,), False)]


def test_the_clis_output_folder_is_one_of_the_agents_folders(tmp_path, monkeypatch):
    """`.anton/output` holds what the CLI makes when it has no artifacts: a
    rewrite there is the agent's own, and a delete is still held."""
    (tmp_path / ".anton" / "output").mkdir(parents=True)
    (tmp_path / ".anton" / "output" / "q3.html").write_text("x")
    monkeypatch.chdir(tmp_path)
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    assert _changes("open('.anton/output/q3.html', 'w').write('v2')", snap) == [
        ("overwrite", (".anton/output/q3.html",), False)
    ]
    assert _files("import os\nos.remove('.anton/output/q3.html')", (snap,)) == [("delete", (".anton/output/q3.html",))]


@pytest.mark.parametrize("earlier, code, path", [
    ("open('log.txt', 'w').write('first')", "open('log.txt', 'w').write('second')", "log.txt"),
    ("import matplotlib.pyplot as plt\nplt.savefig('chart.png')", "import matplotlib.pyplot as plt\nplt.savefig('chart.png')",
     "chart.png"),
    ("def save(df, name):\n    df.to_csv(name, index=False)\nsave(df, 'stock.csv')",
     PD + "df.to_csv('stock.csv', index=False)", "stock.csv"),
    ("import os\nopen('stock.csv.tmp', 'w').write('x')\nos.replace('stock.csv.tmp', 'stock.csv')",
     "import os\nopen('stock.csv.tmp', 'w').write('y')\nos.replace('stock.csv.tmp', 'stock.csv')", "stock.csv"),
    ("import sqlite3\nc = sqlite3.connect('warehouse.db')\nc.execute('CREATE TABLE t (a int)')",
     "import sqlite3\nc = sqlite3.connect('warehouse.db')\nc.execute('INSERT INTO t VALUES (1)')", "warehouse.db"),
], ids=["open", "savefig", "earlier-helper-called-earlier", "temp-renamed-into-place", "database"])
def test_rewriting_a_file_an_earlier_cell_wrote_is_not_held(earlier, code, path, office):
    """The earlier cell ran in an earlier turn, so the file is in this turn's snapshot."""
    assert _changes(code, office, earlier=(earlier,)) == [("overwrite", (path,), False)]
    assert _changes(code, office) == [("overwrite", (path,), True)], "with no earlier cell it is held"


@pytest.mark.parametrize("earlier", [
    "import pandas as pd\ndf = pd.read_csv('stock.csv')",
    "open('stock.csv', 'a').write('x')",
    "def save(df, name):\n    df.to_csv(name, index=False)",
], ids=["only-read", "only-appended", "helper-not-called-earlier"])
def test_a_file_an_earlier_cell_did_not_write_is_still_held(earlier, office):
    """The last: the earlier cells' writes are read against the earlier
    cells alone, so this cell's own call doesn't make the file the agent's."""
    code = "save(df, 'stock.csv')" if "def save" in earlier else PD + "df.to_csv('stock.csv')"
    assert ("overwrite", ("stock.csv",), True) in _changes(code, office, earlier=(earlier,))


@pytest.mark.parametrize("earlier", [
    ("def save():\n    df.to_csv('stock.csv')",),
    ("save = lambda: df.to_csv('stock.csv')",),
    ("def save():\n    df.to_csv('stock.csv')\ndef report():\n    save()",),
    ("if False:\n    df.to_csv('stock.csv')",),
    ("if 0:\n    df.to_csv('stock.csv')",),
    ("while False:\n    df.to_csv('stock.csv')",),
    ("if True:\n    pass\nelse:\n    df.to_csv('stock.csv')",),
    ("def save():\n    df.to_csv('log.txt')\nsave()", "def save():\n    df.to_csv('stock.csv')"),
], ids=["uncalled-def", "uncalled-lambda", "called-only-by-an-uncalled-def", "if-false", "if-0", "while-false",
        "else-of-if-true", "def-redefined-after-the-call"])
def test_a_write_that_never_ran_in_an_earlier_cell_leaves_the_file_held(earlier, office):
    """The earlier cells are read as code, not as what ran: a write in a
    function no cell calls, or in a branch that never runs, never touched
    the user's stock.csv, so rewriting it still waits for approval."""
    assert _changes(PD + "df.to_csv('stock.csv')", office, earlier=earlier) == [("overwrite", ("stock.csv",), True)]


@pytest.mark.parametrize("earlier", [
    ("def save():\n    df.to_csv('stock.csv')\nsave()",),
    ("def save():\n    df.to_csv('stock.csv')", "save()"),
    ("save = lambda: df.to_csv('stock.csv')\nsave()",),
    ("def save():\n    df.to_csv('stock.csv')\ndef report():\n    save()\nreport()",),
    ("def save():\n    df.to_csv('stock.csv')\nif ready:\n    save()",),
    ("class Store:\n    def save(self):\n        df.to_csv('stock.csv')\nStore().save()",),
], ids=["called-in-its-cell", "called-in-a-later-cell", "called-lambda", "called-through-another-function",
        "called-under-an-if", "called-method"])
def test_a_write_in_a_function_an_earlier_cell_called_makes_the_file_the_agents(earlier, office):
    assert _changes(PD + "df.to_csv('stock.csv')", office, earlier=earlier) == [("overwrite", ("stock.csv",), False)]


@pytest.mark.parametrize("errored, held", [((True,), True), ((False,), False), ((), False)],
                         ids=["stopped-on-an-error", "ran-to-its-end", "no-entry"])
def test_a_write_in_an_earlier_cell_that_stopped_on_an_error_makes_no_file_the_agents(errored, held, office):
    """The earlier cell may have stopped before its write ran, as `1 / 0`
    does here, so rewriting the user's stock.csv still waits for approval."""
    earlier = ("x = 1 / 0\nopen('stock.csv', 'w').write('z')",)
    writes = find_cell_writes(code="open('stock.csv', 'w').write('y')", earlier_code=earlier, snapshots=(office,),
                              earlier_errored=errored)
    assert [(c.action, c.paths, c.held) for c in writes.files] == [("overwrite", ("stock.csv",), held)]


def test_an_errored_cell_lines_up_with_its_code_past_one_that_does_not_parse(office):
    """A cell that doesn't parse is dropped, and the flags still name the right cells."""
    earlier = ("def (", "open('log.txt', 'w').write('z')", "x = 1 / 0\nopen('stock.csv', 'w').write('z')")
    code = "open('log.txt', 'w').write('y')\nopen('stock.csv', 'w').write('y')"
    writes = find_cell_writes(code=code, earlier_code=earlier, snapshots=(office,),
                              earlier_errored=(False, False, True))
    assert [(c.paths, c.held) for c in writes.files] == [(("log.txt",), False), (("stock.csv",), True)]


async def test_the_gate_passes_each_earlier_cells_error_to_the_check(office, monkeypatch):
    """`hold_unapproved_write` reads the errors off the pad's cells; a
    compacted cell kept no error, so it counts as one that ran."""
    asked: list[CellWrites] = []

    async def cancel(_session, writes):
        asked.append(writes)
        return "cancelled"

    monkeypatch.setattr(scratchpad, "_ask_approval", cancel)
    session = SimpleNamespace(_write_gate=scratchpad.WriteGate(snapshots=(office,)))
    wrote = "x = 1 / 0\nopen('stock.csv', 'w').write('z')"
    failed = Cell(code=wrote, stdout="", stderr="", error="ZeroDivisionError: division by zero")
    rewrite = "open('stock.csv', 'w').write('y')"
    pad = SimpleNamespace(compacted_code=(), cells=[failed])
    assert await scratchpad.hold_unapproved_write(session, pad=pad, code=rewrite) is not None
    assert [f.paths for f in asked[0].held_files] == [("stock.csv",)]
    ran = SimpleNamespace(compacted_code=(wrote,), cells=[failed])
    assert await scratchpad.hold_unapproved_write(session, pad=ran, code=rewrite) is None
    assert len(asked) == 1


@pytest.mark.parametrize("earlier", [
    PD + "from datetime import date\ndf.to_csv(f'exports/orders_{date.today()}.csv', index=False)",
    "import pandas as pd\nfor f in ['exports/orders', 'log']:\n    pd.DataFrame().to_csv(f + suffix)",
], ids=["dated-export", "name-with-a-run-time-suffix"])
def test_an_earlier_write_to_a_name_picked_at_run_time_makes_no_older_file_the_agents(earlier, tmp_path, monkeypatch):
    """The snapshot holds what the turn began with, not what the conversation
    began with, so an older day's export the earlier name could match may
    be the user's: rewriting it still waits for approval."""
    (tmp_path / "exports").mkdir()
    (tmp_path / "exports/orders_2026-10-01.csv").write_text("x")
    monkeypatch.chdir(tmp_path)
    snapshot = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    code = PD + "df.to_csv('exports/orders_2026-10-01.csv', index=False)"
    assert _changes(code, snapshot, earlier=(earlier,)) == [("overwrite", ("exports/orders_2026-10-01.csv",), True)]


@pytest.mark.parametrize("code", [
    "open('new_report.md', 'w').write('x')",
    "import matplotlib.pyplot as plt\nplt.savefig('charts/sales_by_site.png')",
    PD + "df.to_excel('stock_summary.xlsx')",
    "import os\nopen('report.md.tmp', 'w').write('x')\nos.replace('report.md.tmp', 'report.md')",
    "open(os.environ['OUT'], 'w').write('x')",
    "import shutil\nshutil.copy('stock.csv', 'docs')",
], ids=["new-file", "new-chart", "new-workbook", "temp-renamed-into-a-new-file", "root-the-check-cant-place",
        "copy-into-a-folder"])
def test_a_write_to_a_file_the_turn_did_not_begin_with_is_no_change(code, office):
    assert _changes(code, office) == []


@pytest.mark.parametrize("code", [
    PD + "from datetime import date\ndf.to_csv(f'exports/orders_{date.today():%Y-%m-%d}.csv', index=False)",
    "import matplotlib.pyplot as plt\nfor site, g in df.groupby('site'):\n    plt.savefig(f'charts/stock_{site}.png')",
    "out = 'exports/orders_'\nout += stamp\nopen(out + '.csv', 'w').write('x')",
    "import os\nopen(os.path.join('exports', name), 'w').write('x')",
], ids=["dated-export", "chart-per-site", "built-up-name", "joined-name"])
def test_a_write_to_a_name_the_code_picks_when_it_runs_counts_as_a_new_file(code, tmp_path, monkeypatch):
    """Older files the name could match are there, and none is held: a dated
    export or a chart per site is a new file."""
    for rel in ("exports/orders_2026-10-01.csv", "charts/stock_A.png"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    monkeypatch.chdir(tmp_path)
    snapshot = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    assert _changes(code, snapshot) == []


@pytest.mark.parametrize("code, path", [
    ("import os, pandas as pd\nfor name in os.listdir('docs'):\n    pd.DataFrame().to_csv(os.path.join('docs', name))",
     "docs/a.pdf"),
    ("from pathlib import Path\nfor p in Path('docs').iterdir():\n    p.write_text('x')", "docs/a.pdf"),
    ("import os\nos.remove(f'stock{suffix}.csv')", "stock.csv"),
], ids=["listdir-rewrite", "iterdir-rewrite", "delete-with-a-run-time-part"])
def test_a_listing_still_names_files_that_exist_and_a_delete_any_it_could_name(code, path, office):
    """A listing's wildcard names files the folder holds, so rewriting them
    is held; a delete with a part picked at run time still is too."""
    assert any(held and path in paths for _, paths, held in _changes(code, office))


# ── SQLite and DuckDB files ───────────────────────────────────────────────

@pytest.mark.parametrize("code, path", [
    ("import sqlite3\nc = sqlite3.connect('warehouse.db')\nc.execute('DELETE FROM orders WHERE qty = 0')\nc.commit()",
     "warehouse.db"),
    ("import sqlite3\nwith sqlite3.connect('warehouse.db') as c:\n    c.cursor().execute('UPDATE t SET a = 1')",
     "warehouse.db"),
    ("import sqlite3\ncon = sqlite3.connect(database='warehouse.db')\ncur = con.cursor()\n"
     "cur.executemany('INSERT INTO t VALUES (?)', rows)", "warehouse.db"),
    ("import sqlite3\nc = sqlite3.connect('warehouse.db')\nc.executescript('DROP TABLE t; CREATE TABLE t (a int);')",
     "warehouse.db"),
    ("import duckdb\ncon = duckdb.connect('analytics.duckdb')\ncon.execute(\"CREATE TABLE s AS SELECT * FROM 'stock.csv'\")",
     "analytics.duckdb"),
    ("import duckdb\ncon = duckdb.connect('analytics.duckdb')\ncon.sql('DELETE FROM s WHERE qty < 0')",
     "analytics.duckdb"),
    ("import duckdb\nq = 'UPDATE s SET qty = 0'\nduckdb.connect('analytics.duckdb').execute(q)", "analytics.duckdb"),
    ("import duckdb\ncon = duckdb.connect('analytics.duckdb')\ndef run(s):\n    con.execute(s)\nrun('DELETE FROM s')",
     "analytics.duckdb"),
    ("import pandas as pd, sqlite3\nc = sqlite3.connect('warehouse.db')\ndf.to_sql('stock', c, if_exists='replace')",
     "warehouse.db"),
    ("from sqlalchemy import create_engine, text\ne = create_engine('sqlite:///warehouse.db')\n"
     "with e.begin() as c:\n    c.execute(text('DELETE FROM orders'))", "warehouse.db"),
], ids=["sqlite-delete", "with-cursor", "keyword-executemany", "executescript", "duckdb-create", "duckdb-sql",
        "statement-in-a-name", "local-helper", "to_sql", "sqlalchemy-sqlite-url"])
def test_a_write_to_a_database_file_the_turn_began_with_is_held_as_its_overwrite(code, path, office):
    writes = find_cell_writes(code=code, earlier_code=(), snapshots=(office,))
    assert [(c.action, c.paths) for c in writes.held_files] == [("overwrite", (path,))]
    assert writes.sql == (), "the SQL check still lets an in-process handle through"


@pytest.mark.parametrize("code", [
    "import sqlite3\nc = sqlite3.connect('warehouse.db')\nprint(c.execute('SELECT * FROM orders').fetchall())",
    "import sqlite3\nc = sqlite3.connect(':memory:')\nc.execute('CREATE TABLE t (a int)')",
    "import sqlite3\nc = sqlite3.connect('scratch.db')\nc.execute('CREATE TABLE t (a int)')",
    "import duckdb\ncon = duckdb.connect()\ncon.execute(\"CREATE TABLE s AS SELECT * FROM 'stock.csv'\")",
    "import duckdb\ncon = duckdb.connect('analytics.duckdb', read_only=True)\ncon.execute('CREATE TABLE x AS SELECT 1')",
    "import sqlite3\nc = sqlite3.connect('file:warehouse.db?mode=ro', uri=True)\nc.execute('DELETE FROM t')",
    "import duckdb\nduckdb.sql(\"CREATE TABLE s AS SELECT * FROM 'stock.csv'\")",
], ids=["select", "memory", "new-file", "duckdb-memory", "duckdb-read-only", "sqlite-read-only-uri", "duckdb-module"])
def test_a_database_handle_that_writes_no_file_the_turn_began_with_is_free(code, office):
    assert not find_cell_writes(code=code, earlier_code=(), snapshots=(office,)).needs_approval()


@pytest.fixture
def databases(tmp_path: Path, monkeypatch):
    """A project that began the turn with a DuckDB and a SQLite file."""
    for rel in ("stock.csv", "scratch.duckdb", "scratch.db"):
        (tmp_path / rel).write_text("x")
    monkeypatch.chdir(tmp_path)
    return snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)


DUCK_AGAIN = "import duckdb\nduckdb.connect('scratch.duckdb').execute('INSERT INTO t VALUES (1)')"
SQLITE_AGAIN = "import sqlite3\nsqlite3.connect('scratch.db').execute('INSERT INTO t VALUES (1)')"


@pytest.mark.parametrize("earlier, code, path", [
    ("import duckdb\ncon = duckdb.connect('scratch.duckdb'); "
     "con.execute(\"CREATE TABLE t AS SELECT * FROM read_csv('stock.csv')\")", DUCK_AGAIN, "scratch.duckdb"),
    ("import sqlite3\ncon = sqlite3.connect('scratch.db'); con.execute('CREATE TABLE t (a int)')",
     SQLITE_AGAIN, "scratch.db"),
], ids=["duckdb", "sqlite"])
def test_a_database_file_an_earlier_turn_made_takes_no_card_when_written_again(earlier, code, path, databases):
    """The earlier cell connected to the file and wrote to it in an earlier
    turn, so this turn began with it."""
    assert _changes(code, databases, earlier=(earlier,)) == [("overwrite", (path,), False)]


@pytest.mark.parametrize("engine, path, code", [
    ("duckdb", "scratch.duckdb", DUCK_AGAIN),
    ("sqlite3", "scratch.db", SQLITE_AGAIN),
], ids=["duckdb", "sqlite"])
@pytest.mark.parametrize("earlier", [
    "",
    "con = {engine}.connect('{path}')\nprint(con.execute('SELECT count(*) FROM t').fetchall())",
    "con = {engine}.connect('other_{path}')\ncon.execute('CREATE TABLE t (a int)')",
    "con = {engine}.connect('{path}')\nif False:\n    con.execute('CREATE TABLE t (a int)')",
    "def load():\n    {engine}.connect('{path}').execute('CREATE TABLE t (a int)')",
], ids=["no-earlier-cell", "only-read", "wrote-another-file", "write-under-if-false", "write-in-an-uncalled-def"])
def test_a_database_file_no_earlier_cell_wrote_is_still_held(earlier, engine, path, code, databases):
    """No earlier cell connected to the file and wrote to it, so the turn
    began with the user's file and the user's data is in it: a write still
    waits."""
    cells = (f"import {engine}\n" + earlier.format(engine=engine, path=path),) if earlier else ()
    assert _changes(code, databases, earlier=cells) == [("overwrite", (path,), True)]


# ── HTTP ──────────────────────────────────────────────────────────────────

def _requests(code: str, *, earlier=(), rules=()) -> list[tuple[str, str]]:
    writes = find_cell_writes(code=code, earlier_code=earlier, snapshots=(), http_rules=rules)
    return [(w.method, w.url) for w in writes.http]


URL = "https://erp.example.com/api/orders/42"


@pytest.mark.parametrize("code, method", [
    (f"import requests\nrequests.put({URL!r}, json={{'qty': 5}})", "PUT"),
    (f"import requests\nrequests.patch({URL!r}, json={{'qty': 5}})", "PATCH"),
    (f"import requests\nrequests.delete({URL!r})", "DELETE"),
    (f"import requests\nrequests.request('PUT', {URL!r}, json={{}})", "PUT"),
    (f"import requests\nrequests.request(method='delete', url={URL!r})", "DELETE"),
    (f"from requests import put\nput({URL!r})", "PUT"),
    (f"import requests\ns = requests.Session()\ns.headers['X'] = 'y'\ns.patch({URL!r})", "PATCH"),
    (f"import requests\nwith requests.Session() as s:\n    s.delete({URL!r})", "DELETE"),
    (f"import requests\nrequests.Session().request('PUT', {URL!r})", "PUT"),
    (f"import httpx\nhttpx.put({URL!r})", "PUT"),
    (f"import httpx\nc = httpx.Client(timeout=10)\nc.delete({URL!r})", "DELETE"),
    (f"import httpx\nasync with httpx.AsyncClient() as c:\n    await c.patch({URL!r})", "PATCH"),
    (f"import httpx\nwith httpx.stream('PUT', {URL!r}) as r:\n    pass", "PUT"),
    (f"import httpx\nreq = httpx.Request('DELETE', {URL!r})", "DELETE"),
    (f"import urllib.request\nreq = urllib.request.Request({URL!r}, data=b'{{}}', method='PUT')\n"
     "urllib.request.urlopen(req)", "PUT"),
    (f"from urllib.request import Request, urlopen\nurlopen(Request({URL!r}, method='DELETE'))", "DELETE"),
    ("import requests\ndef make():\n    s = requests.Session()\n    return s\napi = make()\n"
     f"api.put({URL!r})", "PUT"),
    (f"import requests\nfor m in ['GET', 'DELETE']:\n    requests.request(m, {URL!r})", "DELETE"),
], ids=["put", "patch", "delete", "request", "request-keywords", "from-import", "session", "with-session",
        "session-request", "httpx", "httpx-client", "async-client", "httpx-stream", "httpx-request",
        "urllib-request", "urllib-request-inline", "session-from-a-helper", "method-from-a-loop"])
def test_a_put_patch_or_delete_is_held(code, method):
    assert _requests(code) == [(method, URL)]


@pytest.mark.parametrize("code", [
    f"import requests\nr = requests.get({URL!r}, params={{'site': 'A'}})",
    f"import requests\nr = requests.post({URL!r}, json={{'query': 'stock by site'}})",
    f"import requests\nrequests.head({URL!r})",
    f"import requests\nrequests.request('GET', {URL!r})",
    f"import httpx\nwith httpx.Client() as c:\n    c.post({URL!r}, json={{}})",
    f"import urllib.request\nurllib.request.urlopen({URL!r})",
    f"import urllib.request\nurllib.request.urlopen({URL!r}, data=b'q=1')",
    "cache = {}\ncache.get('a')\nqueue.delete('job-1')\nredis.put('k', 'v')",
    "from sqlalchemy.orm import Session\nwith Session(engine) as s:\n    s.delete(row)",
    f"import requests\ndef call(method, url):\n    return requests.request(method, url)\ncall('GET', {URL!r})",
    "import requests\ndef call(method, url):\n    return requests.request(method, url)",
], ids=["get", "post", "head", "request-get", "httpx-post", "urlopen", "urlopen-data", "other-objects",
        "orm-session", "helper-called-with-get", "helper-never-called"])
def test_a_read_or_a_plain_post_is_no_write(code):
    assert _requests(code) == []


def test_the_card_names_the_method_the_url_and_the_call():
    code = ("import requests\nBASE = 'https://erp.example.com/api'\n"
            "for r in rows:\n    requests.put(f'{BASE}/orders/{r.id}', json={'qty': r.qty})")
    writes = find_cell_writes(code=code, earlier_code=(), snapshots=())
    card = _card_prompt(writes)
    assert "This step would send PUT to:" in card
    assert _fenced(card) == ["https://erp.example.com/api/orders/{r.id}",
                             "requests.put(f'{BASE}/orders/{r.id}', json={'qty': r.qty})"]
    assert "A part in braces, `{...}`, is the code that fills in that value when the step runs." in card


def test_a_method_the_code_picks_when_it_runs_is_held_and_said_in_words():
    """The card used to read "This step would send `{row['verb']}` to:", a
    placeholder standing where the method would. It says in words that the
    code decides the method, and shows that code."""
    writes = find_cell_writes(code=f"import requests\nrequests.request(row['verb'], {URL!r})",
                              earlier_code=(), snapshots=())
    assert [(w.method, w.url) for w in writes.http] == [("{row['verb']}", URL)]
    card = _card_prompt(writes)
    assert "This step would send a request to:" in card
    assert "The code decides the request's method when it runs, from `row['verb']`." in card
    assert "`{row['verb']}`" not in card and "send `{" not in card


_API_HELPER = (
    "import requests\nBASE = 'https://erp.example.com/api'\n"
    "def api(method, path):\n    return requests.request(method, BASE + path)",
)


def test_a_helper_parameter_is_shown_as_what_the_call_passes():
    """`def api(method, path)` called with a value the code picks: the card
    names the call's `verb`, the code the person wrote in this step, not the
    helper's own `{method}`."""
    writes = find_cell_writes(code="for verb in verbs:\n    api(verb, '/orders/42')",
                              earlier_code=_API_HELPER, snapshots=())
    card = _card_prompt(writes)
    assert "The code decides the request's method when it runs, from `verb`." in card
    assert "{method}" not in card and "`method`" not in card
    assert _fenced(card) == [URL, "api(verb, '/orders/42')"]


def test_an_address_the_code_picks_whole_is_said_in_words():
    """A URL that is nothing but run-time code used to show as a bare
    `{u}` fence, as if it were the address."""
    writes = find_cell_writes(code="import requests\nfor u in urls:\n    requests.delete(u)",
                              earlier_code=(), snapshots=())
    card = _card_prompt(writes)
    assert "This step would send DELETE to an address the code decides when it runs, from `u`:" in card
    assert _fenced(card) == ["requests.delete(u)"]
    assert "{u}" not in card
    assert "A part in braces" not in card


def test_the_refusal_says_a_run_time_method_in_words_too():
    writes = find_cell_writes(code=f"import requests\nrequests.request(row['verb'], {URL!r})",
                              earlier_code=(), snapshots=())
    text = _refusal(writes, "cancelled").content
    assert f"send a request to `{URL}` (the code decides its method when it runs, from `row['verb']`)" in text
    assert "send {row" not in text


def test_a_helper_that_reassigns_its_parameter_names_its_own_variable():
    """`api(method, path)` that sets `method = 'DELETE'` for some paths: the
    caller's `verb` does not decide the method alone, so the card must not
    say it does. A person who knows `verbs` holds only GET would approve a
    DELETE."""
    helper = (
        "import requests\nBASE = 'https://erp.example.com/api'\n"
        "def api(method, path):\n    if path.endswith('/42'):\n        method = 'DELETE'\n"
        "    return requests.request(method, BASE + path)",
    )
    writes = find_cell_writes(code="for verb in verbs:\n    api(verb, '/orders/42')",
                              earlier_code=helper, snapshots=())
    card = _card_prompt(writes)
    assert "The code decides the request's method when it runs, from `method`." in card
    assert "from `verb`" not in card


def test_a_written_out_method_cannot_hide_a_held_request_from_the_card():
    """A method is any text an HTTP library sends. Written into the prose as
    is, `<!--` and `-->` made Cowork drop the DELETE between them from the
    card the person approves."""
    rules = parse_http_write_rules("* https://erp.example.com/api/**")
    code = (
        "import requests\nU = 'https://erp.example.com/api/ping'\n"
        "requests.request('<!--', U)\n"
        "requests.delete('https://erp.example.com/api/orders/1')\n"
        "requests.request('-->', U)\n"
    )
    writes = find_cell_writes(code=code, earlier_code=(), snapshots=(), http_rules=rules)
    assert [w.method for w in writes.http] == ["<!--", "DELETE", "-->"]

    card = _card_prompt(writes)
    assert "<!--" not in card and "-->" not in card
    assert "This step would send DELETE to:" in re.sub(r"<!--[\s\S]*?-->", "", card)
    assert "This step would send `<\u200b!--` to:" in card
    refusal = _refusal(writes, "cancelled").content
    assert "<!--" not in refusal and "-->" not in refusal

@pytest.mark.parametrize("code,earlier,held", [
    (f"import requests\nrequests.request(row['verb'], {URL!r})", (),
     [(2, f"requests.request(row['verb'], {URL!r})", URL)]),
    ("import requests\nBASE = 'https://erp.example.com/api'\ndef api(method, path):\n"
     "    return requests.request(method, BASE + path)\nfor verb, path in jobs:\n    api(verb, path)", (),
     [(4, "return requests.request(method, BASE + path)", "https://erp.example.com/api{path}")]),
    ("import requests\ndef call(method, url):\n    return requests.request(method, url)\nfor v, u in jobs:\n    call(v, u)",
     (), [(3, "return requests.request(method, url)", "{url}")]),
    ("import requests\nfor u in urls:\n    requests.delete(u)", (), [(3, "requests.delete(u)", "{u}")]),
    ("for v in verbs:\n    api(v, '/orders/42')", _API_HELPER, [(2, "api(v, '/orders/42')", URL)]),
    ("api(job['m'], job['p'])", _API_HELPER,
     [(1, "api(job['m'], job['p'])", "https://erp.example.com/api{job['p']}")]),
    ("api('PUT', '/orders/42')", _API_HELPER, [(1, "api('PUT', '/orders/42')", URL)]),
    ("api('GET', '/orders/42')", _API_HELPER, []),
    ("api('POST', '/orders')", _API_HELPER, []),
    ("import requests\nrequests.get('https://erp.example.com/api/orders')", (), []),
], ids=["method-from-data", "helper-in-the-cell", "helper-whole-url", "url-from-a-loop",
        "helper-called-with-a-variable", "helper-called-with-data", "helper-called-with-put",
        "helper-called-with-get", "helper-called-with-post", "plain-get"])
def test_the_card_wording_holds_the_same_requests_as_before(code, earlier, held):
    """Presentation only: each cell holds exactly the requests, lines and
    addresses the check held before the card said run-time parts in words
    (values read from the check at anton v2.26.10.19.3)."""
    writes = find_cell_writes(code=code, earlier_code=earlier, snapshots=())
    assert [(w.line, w.call, w.url) for w in writes.http] == held
    assert writes.kinds() == (("http",) if held else ())


def test_a_client_base_url_is_joined_to_a_relative_path():
    code = "import httpx\nc = httpx.Client(base_url='https://erp.example.com/api/')\nc.delete('/orders/42')"
    assert _requests(code) == [("DELETE", URL)]


def test_a_request_in_an_earlier_function_is_held_at_the_call():
    earlier = ("import requests\ndef cancel(order_id):\n"
               "    requests.delete(f'https://erp.example.com/api/orders/{order_id}')",)
    writes = find_cell_writes(code="cancel(42)", earlier_code=earlier, snapshots=())
    assert [(w.method, w.url, w.line, w.call) for w in writes.http] == [("DELETE", URL, 1, "cancel(42)")]


def test_a_token_in_the_url_is_scrubbed():
    token = "ya29." + "a" * 30
    writes = find_cell_writes(code=f"import requests\nrequests.delete('https://x.example.com/a?access_token={token}')",
                              earlier_code=(), snapshots=())
    assert token not in writes.http[0].url and token not in _card_prompt(writes)


def test_a_url_with_a_character_that_does_not_show_as_itself_is_named_with_escapes():
    """As a path is: a text direction override in the URL can't reorder what the card shows."""
    writes = find_cell_writes(code="import requests\nrequests.put('https://x.example.com/a\\u202eb')",
                              earlier_code=(), snapshots=())
    card = _card_prompt(writes)
    assert "'https://x.example.com/a\\u202eb'" in card and "A name in quotes is written with escapes" in card
    assert "`'https://x.example.com/a\\u202eb'`" in _refusal(writes, "cancelled").content


# ── HTTP endpoints the operator lists ─────────────────────────────────────

MAESTRO = "POST https://maestro.example.com/api/*/orders/*"


@pytest.mark.parametrize("url, held", [
    ("'https://maestro.example.com/api/v2/orders/9'", True),
    ("'https://MAESTRO.example.com/api/v2/orders/9?dry_run=0'", True),
    ("f'https://maestro.example.com/api/{version}/orders/{order_id}'", True),
    ("f'{MAESTRO_BASE}/orders/{order_id}'", True),
    ("url", True),
    ("'https://maestro.example.com/api/v2/orders'", False),
    ("'https://maestro.example.com/api/v2/orders/9/lines'", False),
    ("'https://maestro.example.com/api/v2/forecasts/9'", False),
    ("f'https://other.example.com/api/{version}/orders/{order_id}'", False),
], ids=["literal", "case-and-query", "run-time-segments", "run-time-base", "run-time-whole-url",
        "too-short", "star-stays-in-one-segment", "other-path", "other-host"])
def test_a_post_to_a_listed_endpoint_is_held(url, held):
    rules = parse_http_write_rules(MAESTRO)
    code = f"import requests\nrequests.post({url}, json=order)"
    assert bool(_requests(code, rules=rules)) is held
    assert _requests(code) == [], "with no rule a POST is free"


def test_a_rule_names_its_method_or_any():
    rules = parse_http_write_rules("GET https://maestro.example.com/api/**\n* https://hooks.example.com/**")
    assert _requests("import requests\nrequests.get('https://maestro.example.com/api/v2/orders/9/lines')",
                     rules=rules) == [("GET", "https://maestro.example.com/api/v2/orders/9/lines")]
    assert _requests("import requests\nrequests.post('https://maestro.example.com/api/v2/orders')", rules=rules) == []
    assert _requests("import requests\nrequests.head('https://hooks.example.com/x')", rules=rules) == [
        ("HEAD", "https://hooks.example.com/x")
    ]


def test_a_rule_matches_the_url_without_its_query_string():
    rules = parse_http_write_rules("POST https://maestro.example.com/api/jobs")
    url = "https://maestro.example.com/api/jobs?run=1&next=/home"
    assert _requests(f"import requests\nrequests.post({url!r})", rules=rules) == [("POST", url)]


def test_http_write_rules_parse_entries_and_skip_bad_ones_with_a_warning(caplog):
    text = f"{MAESTRO}\n  , delete https://x.example.com/* ,\nhttps://no-method.example.com/*\nPUTT https://x/*\nGET a b"
    with caplog.at_level(logging.WARNING, logger="anton.core.utils.write_check"):
        rules = parse_http_write_rules(text)
    assert rules == (
        HttpWriteRule(method="POST", url_glob="https://maestro.example.com/api/*/orders/*"),
        HttpWriteRule(method="DELETE", url_glob="https://x.example.com/*"),
    )
    warned = [r.getMessage() for r in caplog.records
              if r.name == "anton.core.utils.write_check" and r.levelno == logging.WARNING]
    assert len(warned) == 3
    for entry, message in zip(("https://no-method.example.com/*", "PUTT https://x/*", "GET a b"), warned, strict=True):
        assert message.startswith(f"write approval: skipped the HTTP write entry {entry!r}")
    assert parse_http_write_rules("") == ()


@pytest.mark.parametrize("code", [
    "import os\nif empty:\n    os.remove('stock.csv')\nelse:\n    open('stock.csv', 'w').write('x')",
    "open('summary.md', 'w').write('x')\nimport os\nos.remove('summary.md')",
    "from pathlib import Path\nPath('summary.md').write_text('')\nPath('summary.md').unlink()",
    "import os\ndef later():\n    open('stock.csv', 'w').write('x')\nos.remove('stock.csv')",
    "import os\ndef f():\n    os.remove('stock.csv')\n    return\n    open('stock.csv', 'w').write('x')\nf()",
    "import os\nos.remove('stock.csv')\nif refresh:\n    open('stock.csv', 'w').write('x')",
    "import os, sys\nos.remove('stock.csv')\nsys.exit(0)\ndf.to_csv('stock.csv')",
    "import os\nos.remove('stock.csv')\nos._exit(0)\ndf.to_csv('stock.csv')",
    "import os\nos.remove('stock.csv')\nquit()\ndf.to_csv('stock.csv')",
    "import os\nos.remove('stock.csv')\nassert False\ndf.to_csv('stock.csv')",
], ids=["if-else", "write-then-delete", "write-text-then-unlink", "write-in-uncalled-function",
        "write-after-return", "write-under-if", "write-after-sys-exit", "write-after-os-exit",
        "write-after-quit", "write-after-assert"])
def test_a_delete_counts_as_an_overwrite_only_when_a_write_always_follows(code, snap):
    writes = find_cell_writes(code=code, earlier_code=(), snapshots=(snap,))
    assert writes.needs_approval()
    assert "delete" in {c.action for c in writes.held_files}


def test_cards_name_the_files(snap):
    writes = find_cell_writes(code="import glob, os\nfor f in glob.glob('*.pdf'):\n    os.remove(f)",
                              earlier_code=(), snapshots=(snap,))
    assert writes.held_files[0].paths == ("report.pdf",), "a root glob doesn't reach docs/"
    picked = find_cell_writes(code="import os\nos.remove(os.environ['X'])",
                              earlier_code=(), snapshots=(snap,))
    assert picked.held_files[0].paths == () and "os.environ['X']" in picked.held_files[0].call
    whole = find_cell_writes(code="import shutil\nshutil.rmtree('.')",
                             earlier_code=(), snapshots=(snap,))
    assert whole.held_files[0].whole_project


def test_a_name_resolves_to_its_latest_binding(snap):
    earlier = ("path = 'report.pdf'\nprint(open(path).read()[:10])",)
    code = "import os\npath = 'lead_times.md.tmp'\nopen(path, 'w').write('x')\nos.replace(path, 'lead_times.md')"
    assert not _held(code, snap, earlier=earlier)
    assert _held("import os\nos.remove(path)", snap, earlier=earlier)


@pytest.mark.parametrize("earlier, code, expected", [
    ("from pathlib import Path\nout = Path('output')\nout /= 'charts'\nout.mkdir(parents=True, exist_ok=True)",
     "from pathlib import Path\nout = Path('report.pdf')\nout.unlink()", ("report.pdf",)),
    ("fname = 'export_' + 'q3'\nfname += '.csv'", "import os\nfname = 'stock.csv'\nos.remove(fname)", ("stock.csv",)),
    ("from pathlib import Path\nd = Path('.')\nd /= 'output'",
     "import shutil\nfrom pathlib import Path\nd = Path('docs')\nshutil.rmtree(d)", ("docs",)),
    ("def deeper():\n    global out\n    out /= 'charts'",
     "from pathlib import Path\nout = Path('report.pdf')\nout.unlink()", ("report.pdf",)),
], ids=["path-join", "text-add", "folder-join", "join-in-an-uncalled-function"])
def test_an_addition_before_a_name_is_bound_again_does_not_change_it(earlier, code, expected, snap):
    assert _files(code, (snap,), earlier=(earlier,)) == [("delete", expected)]
    same_cell = "import os\np = 'stock'\np += '.csv'\nprint(open(p).read())\np = 'report.pdf'\nos.remove(p)"
    assert "report.pdf" in _files(same_cell, (snap,))[0][1]


def test_a_list_name_bound_again_drops_what_was_added_before(snap):
    """An append made before the name was bound again grew the old list."""
    own = "files = ['tmp_a.csv']\nopen(files[0], 'w').write('x')\nfor f in files:\n    os.remove(f)"
    assert not _held("import os\n" + own, snap, earlier=("files = []\nfiles.append('stock.csv')",))
    assert not _held("import os\nfiles = []\nfiles.append('stock.csv')\nprint(files)\n" + own, snap)
    assert _held("import os\n" + own.replace("for f", "files.append('stock.csv')\nfor f"), snap)
    assert _held("import os\nfor f in files:\n    os.remove(f)", snap,
                 earlier=("files = ['tmp_a.csv']", "files.append('stock.csv')"))
    assert _held("import os\nfiles = []\nadd('stock.csv')\nfor f in files:\n    os.remove(f)", snap,
                 earlier=("def add(p):\n    files.append(p)",)), "a function's append runs when it is called"
    assert _held("import os\np = 'stock.csv'\np = p + ''\nos.remove(p)", snap)


def test_an_empty_project_holds_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    empty = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time())
    assert not _held("import os\nos.remove(os.environ['X'])", empty)


def test_a_cell_that_does_not_parse_holds_nothing(snap):
    assert not _held("def broken(:\n", snap)


def test_a_lone_surrogate_still_parses(snap):
    assert _held("import os\nos.remove('report.pdf')  # \udcff", snap)


def test_an_earlier_cell_that_does_not_parse_is_skipped(snap):
    assert _held("cur.execute('UPDATE t SET a = 1')", snap, earlier=("def broken(:", PG))


def _files(code: str, snapshots, *, earlier=()) -> list[tuple[str, tuple[str, ...]]]:
    writes = find_cell_writes(code=code, earlier_code=earlier, snapshots=snapshots)
    return [(c.action, c.paths) for c in writes.held_files]


# ── Functions an earlier cell defined ─────────────────────────────────────

WIPE = "import os\ndef wipe(p):\n    os.remove(p)"


@pytest.mark.parametrize("earlier, code, expected", [
    ((WIPE,), "wipe('report.pdf')", [("delete", ("report.pdf",))]),
    ((WIPE,), "wipe(p='report.pdf')", [("delete", ("report.pdf",))]),
    (("import os\nwipe = lambda p: os.remove(p)",), "wipe('report.pdf')", [("delete", ("report.pdf",))]),
    (("import shutil\ndef stash(p):\n    shutil.move(p, 'archive/')",), "stash('report.pdf')",
     [("move", ("report.pdf",))]),
    (("import shutil\ndef clean():\n    shutil.rmtree('output')",), "clean()", [("delete", ("output",))]),
    ((WIPE, "def outer(q):\n    wipe(q)"), "outer('stock.csv')", [("delete", ("stock.csv",))]),
    ((WIPE,), "list(map(wipe, files))", [("delete", ())]),
    ((WIPE, "wipe('scratch_tmp.csv')"), "wipe('summary.md')", [("delete", ("summary.md",))]),
    (("import shutil\nclass Reset:\n    def go(self):\n        shutil.rmtree('output')",), "Reset().go()",
     [("delete", ("output",))]),
    (("import os\nclass Cleaner:\n    def run(self, p):\n        os.remove(p)\ncleaner = Cleaner()",),
     "cleaner.run('report.pdf')", [("delete", ("report.pdf",))]),
    (("import os\nclass Base:\n    def run(self, p):\n        self.drop(p)\n    def drop(self, p):\n        os.remove(p)\n"
      "class Cleaner(Base):\n    pass",), "Cleaner().run('report.pdf')", [("delete", ("report.pdf",))]),
    (("import os\nops = {'rm': lambda p: os.remove(p)}",), "ops['rm']('report.pdf')", [("delete", ("report.pdf",))]),
    (("import os\nops = [lambda p: os.remove(p)]",), "ops[0]('report.pdf')", [("delete", ("report.pdf",))]),
    (("import os, functools\nrm_report = functools.partial(os.remove, 'report.pdf')",), "rm_report()",
     [("delete", ("report.pdf",))]),
    ((WIPE, "import functools\nrm_report = functools.partial(wipe, 'report.pdf')"), "rm_report()",
     [("delete", ("report.pdf",))]),
    (("import os\ndef make():\n    def w(p):\n        os.remove(p)\n    return w\nwipe = make()",), "wipe('report.pdf')",
     [("delete", ("report.pdf",))]),
    (("import os\nops = dict(rm=lambda p: os.remove(p))",), "ops['rm']('report.pdf')", [("delete", ("report.pdf",))]),
    (("import os\nclass Base:\n    def run(self, p):\n        os.remove(p)\nclass Child(Base):\n    def run(self, p):\n"
      "        super().run(p)",), "Child().run('report.pdf')", [("delete", ("report.pdf",))]),
    (("import os\nclass Tools:\n    rm = staticmethod(lambda p: os.remove(p))",), "Tools.rm('report.pdf')",
     [("delete", ("report.pdf",))]),
], ids=["def", "keyword", "lambda", "move", "no-params", "transitive", "passed-as-a-value", "new-call-only",
        "method-on-new-instance", "method-on-instance", "inherited-method-through-self", "dict-of-lambdas",
        "list-of-lambdas", "partial", "partial-of-helper", "closure-factory", "dict-call-of-lambdas",
        "method-through-super", "lambda-in-a-class-body"])
def test_a_function_an_earlier_cell_defined_is_read_at_each_call(earlier, code, expected, snap):
    writes = find_cell_writes(code=code, earlier_code=earlier, snapshots=(snap,))
    assert [(c.action, c.paths) for c in writes.held_files] == expected
    assert {(c.line, c.call) for c in writes.held_files} == {(1, code)}, "reported at the call"


def test_a_helper_defined_in_an_empty_project_is_held_where_it_is_later_called(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    empty = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time())
    assert not _held(WIPE + "\nwipe('scratch.csv')", empty)
    (tmp_path / "report.pdf").write_text("x")
    later = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    assert _held("wipe('report.pdf')", later, earlier=(WIPE + "\nwipe('scratch.csv')",))


@pytest.mark.parametrize("earlier, code", [
    ("import os\ndef save(p):\n    tmp = p + '.tmp'\n    open(tmp, 'w').write('x')\n    os.replace(tmp, p)",
     "save('lead_times.md')"),
    (WIPE, "wipe('scratch_tmp.csv')"),
    ("def q(t):\n    return f'UPDATE {t} SET a = 1'", "print(q('suppliers'))"),
    ("import os\nclass Cleaner:\n    def run(self, p):\n        os.remove(p)\ncleaner = Cleaner()",
     "cleaner.run('scratch_tmp.csv')"),
    ("class Thing:\n    def go(self):\n        return 1", "Thing().go()"),
], ids=["helper-temp-file", "helper-on-new-file", "statement-only-printed", "method-on-new-file",
        "method-that-deletes-nothing"])
def test_an_earlier_function_on_the_turns_own_files_raises_no_card(earlier, code, snap):
    assert not _held(code, snap, earlier=(earlier,))


@pytest.mark.parametrize("earlier, code, statement", [
    ("def bump():\n    cur.execute(\"UPDATE suppliers SET lead = 21 WHERE id = 'S005'\")", "bump()",
     "UPDATE suppliers SET lead = 21 WHERE id = 'S005'"),
    ("def bump(i):\n    cur.execute(f\"UPDATE suppliers SET lead = 21 WHERE id = '{i}'\")", "bump('S005')",
     "UPDATE suppliers SET lead = 21 WHERE id = 'S005'"),
    ("def q(t):\n    return 'UPDATE ' + t + ' SET a = 1'", "cur.execute(q('suppliers'))",
     "UPDATE suppliers SET a = 1"),
], ids=["literal-body", "parameter-filled-in", "returned-to-execute"])
def test_a_statement_inside_an_earlier_function_is_held_at_the_call(earlier, code, statement, snap):
    writes = find_cell_writes(code=code, earlier_code=(PG, earlier), snapshots=(snap,))
    assert [(w.statement, w.line) for w in writes.sql] == [(statement, 1)]


# ── Lists of paths, shell commands, lookups ───────────────────────────────

@pytest.mark.parametrize("code, earlier", [
    ("import os\nold = []\nfor name in os.listdir('.'):\n    if name.endswith('.pdf'):\n        old.append(name)\n"
     "for f in old:\n    os.remove(f)", ()),
    ("import glob, os\nold = []\nold += glob.glob('*.pdf')\nfor f in old:\n    os.remove(f)", ()),
    ("import glob, os\nold = []\nold.extend(glob.glob('*.pdf'))\nfor f in old:\n    os.remove(f)", ()),
    ("import glob, os\nold = ()\nold += tuple(glob.glob('*.pdf'))\nfor f in old:\n    os.remove(f)", ()),
    ("import os\n[os.remove(x) for x in old]", ("old = []\nold.append('report.pdf')",)),
    ("import os\nfs = ['new.txt']\nfs.append('report.pdf')\nfor f in fs:\n    os.remove(f)", ()),
    ("import os\nfs = set()\nfs.add('report.pdf')\nfor f in fs:\n    os.remove(f)", ()),
    ("import glob, os\nold = [] + glob.glob('*.pdf')\nfor f in old:\n    os.remove(f)", ()),
    ("import os\ndef collect(acc):\n    acc.append('report.pdf')\nold = []\ncollect(old)\nfor f in old:\n    os.remove(f)",
     ()),
    ("import os\nold = []\ncollect(old)\nfor f in old:\n    os.remove(f)", ("def collect(acc):\n    acc.append('report.pdf')",)),
    ("import os\na = []\nb = a\nb.append('report.pdf')\nfor f in a:\n    os.remove(f)", ()),
    ("import os\nold = []\nold[:] = ['report.pdf']\nfor f in old:\n    os.remove(f)", ()),
    ("import os\nold = ['x']\nold[0] = 'report.pdf'\nfor f in old:\n    os.remove(f)", ()),
    ("import os\nold = [[]]\nold[0].append('report.pdf')\nfor f in old[0]:\n    os.remove(f)", ()),
    ("import os\nold = []\ncollect()\nfor f in old:\n    os.remove(f)", ("def collect():\n    old.append('report.pdf')",)),
    ("import os\nold = []\nColl().fill(old)\nfor f in old:\n    os.remove(f)",
     ("class Coll:\n    def fill(self, acc):\n        acc.append('report.pdf')",)),
], ids=["append-from-listdir", "plus-equals-glob", "extend", "tuple", "built-in-an-earlier-cell",
        "literal-grown", "set-add", "list-plus-list", "filled-by-a-helper", "filled-by-an-earlier-helper",
        "filled-through-another-name", "slice-assigned", "item-assigned", "inner-list-grown",
        "global-filled-by-an-earlier-helper", "filled-by-a-method"])
def test_paths_gathered_into_a_list_are_held(code, earlier, snap):
    """The filter in the first shape isn't read, so its card names every file
    the listing gives."""
    found = _files(code, (snap,), earlier=earlier)
    assert len(found) == 1 and found[0][0] == "delete" and "report.pdf" in found[0][1]


@pytest.mark.parametrize("code, expected", [
    ("import os\ncmd = 'rm report.pdf'\nos.system(cmd)", [("delete", ("report.pdf",))]),
    ("import subprocess\nargs = ['rm', 'report.pdf']\nsubprocess.run(args)", [("delete", ("report.pdf",))]),
    ("import os\ncmd = 'rm '\ncmd += 'report.pdf'\nos.system(cmd)", [("delete", ("report.pdf",))]),
    ("import os\nos.system(f'rm {name}')", [("delete", ())]),
    ("import os\nos.system('rm ' + name)", [("delete", ())]),
    ("import os\nos.system('rm {}'.format(name))", [("delete", ())]),
    ("import os\nos.system('rm %s' % name)", [("delete", ())]),
    ("import os\nroot = '.'\nos.system(f'rm -rf {root}')", [("delete", ())]),
    ("import subprocess\nsubprocess.run(['sh', '-c', 'rm report.pdf'])", [("delete", ("report.pdf",))]),
    ("import subprocess\nsubprocess.run('bash -lc \"rm report.pdf\"', shell=True)", [("delete", ("report.pdf",))]),
    ("import subprocess\nsubprocess.getoutput('rm report.pdf')", [("delete", ("report.pdf",))]),
    ("import subprocess\nsubprocess.getstatusoutput('mv report.pdf x.pdf')", [("move", ("report.pdf",))]),
    ("import os\nos.system('mv -t archive report.pdf')", [("move", ("report.pdf",))]),
    ("import os\nos.system('mv --target-directory=/tmp/out report.pdf')", [("move", ("report.pdf",))]),
    ("import os\nos.system('sudo rm -f report.pdf')", [("delete", ("report.pdf",))]),
    ("import os\nos.system('ls *.pdf | xargs rm')", [("delete", ())]),
    ("import subprocess\nsubprocess.run(['find', '.', '-name', '*.pdf', '-delete'])", [("delete", ())]),
    ("import os\nos.system('find . -name \"*.pdf\" -exec rm {} \\\\;')", [("delete", ())]),
    ("import os\nos.system('git clean -fdx')", [("delete", ())]),
    ("import os\nos.spawnlp(os.P_WAIT, 'rm', 'rm', 'report.pdf')", [("delete", ("report.pdf",))]),
    ("import os\nos.system('rm $OLD')", [("delete", ())]),
    ("import os\nos.system('if [ -f report.pdf ]; then rm report.pdf; fi')", [("delete", ("report.pdf",))]),
    ("import os\nos.system('for f in *.pdf; do rm \"$f\"; done')", [("delete", ())]),
    ("import os\ndef run(cmd):\n    os.system(cmd)\nrun('rm report.pdf')", [("delete", ("report.pdf",))]),
    ("exec(\"import os\\nos.remove('report.pdf')\")", [("delete", ("report.pdf",))]),
    ("import subprocess\nsubprocess.run('rm report.pdf'.split())", [("delete", ("report.pdf",))]),
    ("import subprocess, shlex\ncmd = 'rm -f report.pdf'\nsubprocess.run(shlex.split(cmd))", [("delete", ("report.pdf",))]),
    ("import subprocess\nnames = ['report.pdf']\nsubprocess.run(['rm'] + names)", [("delete", ("report.pdf",))]),
    ("import subprocess, glob\nargs = ['rm', '-f'] + glob.glob('*.pdf')\nsubprocess.run(args)", [("delete", ())]),
    ("import os\nos.system('mv -fvt /tmp/out report.pdf summary.md')", [("move", ("report.pdf", "summary.md"))]),
    ("import os\nos.system('timeout -s KILL 5 rm report.pdf')", [("delete", ("report.pdf",))]),
    ("import os\nos.system('stdbuf -o0 rm report.pdf')", [("delete", ("report.pdf",))]),
    ("import os\nos.system('ionice -c 3 rm report.pdf')", [("delete", ("report.pdf",))]),
    ("import os\nos.system('busybox rm report.pdf')", [("delete", ("report.pdf",))]),
    ("import os\nos.system('echo rm report.pdf | sh')", [("delete", ())]),
    ("import subprocess\nsubprocess.run(['sh'], input='rm report.pdf', text=True)", [("delete", ("report.pdf",))]),
    ("import os\nos.system('$(which rm) report.pdf')", [("delete", ())]),
], ids=["named-command", "named-words", "built-up-command", "f-string", "plus", "format", "percent",
        "f-string-root", "sh-c", "bash-lc", "getoutput", "getstatusoutput", "mv-t", "mv-target-directory",
        "sudo", "xargs", "find-delete", "find-exec-rm", "git-clean", "spawnlp", "shell-variable",
        "if-then", "for-do", "helper-command", "exec-code", "split-text", "shlex-split", "list-plus-list",
        "list-plus-glob", "mv-flags-then-t", "timeout", "stdbuf", "ionice", "busybox", "piped-into-sh",
        "sh-input", "command-substitution"])
def test_shell_deletes_and_moves_in_every_form(code, expected, snap):
    assert _files(code, (snap,)) == expected


@pytest.mark.parametrize("code, expected", [
    ("import os\ngetattr(os, 'remove')('report.pdf')", [("delete", ("report.pdf",))]),
    ("import shutil\ngetattr(shutil, 'rmtree')('output')", [("delete", ("output",))]),
    ("import os\nfn = getattr(os, 'remove')\nfn('report.pdf')", [("delete", ())]),
    ("__import__('os').remove('report.pdf')", [("delete", ("report.pdf",))]),
    ("from pathlib import Path\nPath.unlink(Path('report.pdf'))", [("delete", ("report.pdf",))]),
    ("from pathlib import Path\nPath.rename(Path('report.pdf'), 'x.pdf')", [("rename", ("report.pdf",))]),
    ("from pathlib import Path\nfor p in Path('.').glob('*.pdf'):\n    Path.unlink(p)", [("delete", ("report.pdf",))]),
    ("from pathlib import Path\nlist(map(Path.unlink, Path('.').glob('*.pdf')))", [("delete", ())]),
    ("import pathlib\npathlib.Path.rmdir(pathlib.Path('output'))", [("delete", ("output",))]),
    ("import os\ngetattr(os, 'remove', None)('report.pdf')", [("delete", ("report.pdf",))]),
    ("import os\nfn = 'remove'\ngetattr(os, fn)('report.pdf')", [("delete", ("report.pdf",))]),
    ("import os\ngetattr(os, 'rem' + 'ove')('report.pdf')", [("delete", ("report.pdf",))]),
    ("import os\nvars(os)['remove']('report.pdf')", [("delete", ("report.pdf",))]),
    ("import os\nos.__dict__['remove']('report.pdf')", [("delete", ("report.pdf",))]),
    ("import os\ngetattr(os, os.environ['FN'])('report.pdf')", [("delete", ())]),
    ("import os\ngetattr(os, 'rem' + suffix)('report.pdf')", [("delete", ())]),
    ("import os\nfor n in ['exists']:\n    print(getattr(os.path, n)('report.pdf'))", []),
], ids=["getattr-os", "getattr-shutil", "getattr-value", "dunder-import", "Path.unlink-unbound",
        "Path.rename-unbound", "Path.unlink-loop", "map-Path.unlink", "Path.rmdir-unbound", "getattr-default",
        "getattr-named-by-a-name", "getattr-named-by-a-sum", "vars", "dunder-dict", "getattr-unread-name",
        "getattr-named-at-run-time", "getattr-on-os-path"])
def test_a_delete_reached_by_name_or_unbound_is_held(code, expected, snap):
    assert _files(code, (snap,)) == expected


@pytest.mark.parametrize("code", [
    "import os\nfn = 'replace'\ngetattr(os, fn)('new.tmp', 'stock.csv')",
    "import shutil\nfn = 'copy'\ngetattr(shutil, fn)('new.csv', 'stock.csv')",
    "import shutil\nvars(shutil)['copyfile']('new.csv', 'stock.csv')",
    "from pathlib import Path\nfn = 'replace'\ngetattr(Path('new.tmp'), fn)('stock.csv')",
], ids=["os-replace", "shutil-copy", "vars-copyfile", "path-replace"])
def test_a_move_or_copy_reached_by_name_overwrites_what_it_lands_on(code, snap):
    """As the plain call does: the file it lands on, one the turn began with, is replaced."""
    assert _files(code, (snap,)) == [("overwrite", ("stock.csv",))]


def test_a_listing_names_files_before_folders_and_host_folders_last(snap):
    unlinked = _files("from pathlib import Path\nfor p in Path('.').iterdir():\n    p.unlink()", (snap,))
    assert unlinked == [("delete", ("report.pdf", "stock.csv", "summary.md"))], "unlink can't remove a folder"
    removed = _files("import os, shutil\nfor p in os.listdir('.'):\n    shutil.rmtree(p)", (snap,))
    assert removed == [("delete", ("report.pdf", "stock.csv", "summary.md", "docs", "output", ".anton", ".git"))]


@pytest.mark.parametrize("code", [
    "import os\nos.chdir('docs')\nos.remove('../report.pdf')",
    "import os\nos.remove('../{project}/report.pdf')",
    "import contextlib, os\nwith contextlib.chdir('docs'):\n    os.remove('a.pdf')",
    "from contextlib import chdir\nimport os\nwith chdir('docs'):\n    os.remove('a.pdf')",
    "import os\nos.system('cd docs && rm a.pdf')",
    "import subprocess\nsubprocess.run(['rm', 'a.pdf'], cwd='docs')",
], ids=["parent-after-chdir", "out-and-back-in", "contextlib-chdir", "from-contextlib", "shell-cd", "cwd-keyword"])
def test_a_relative_path_counts_from_every_folder_the_pad_changes_into(code, snap):
    assert _held(code.replace("{project}", snap.root.name), snap)


# ── The shared folder, links, case and names that look like globs ──────────

@pytest.fixture
def shared(tmp_path_factory):
    root = tmp_path_factory.mktemp("shared")
    (root / "forecast.xlsx").write_text("x")
    (root / "team").mkdir()
    (root / "team" / "plan.docx").write_text("x")
    return snapshot_project_files(root=root, label="shared", taken_at=time.time() + 60)


@pytest.mark.parametrize("code, action", [
    ("import shutil\nshutil.rmtree({root!r})", "delete"),
    ("import shutil\nshutil.rmtree({root!r} + '/')", "delete"),
    ("import os\nos.system('rm -rf {root}')", "delete"),
    ("import shutil\nshutil.move({root!r}, '/tmp/elsewhere')", "move"),
    ("import os\nos.rename({root!r}, {root!r} + '.old')", "rename"),
], ids=["rmtree", "rmtree-slash", "rm-rf", "move", "rename"])
def test_the_shared_folder_itself_is_held_whole(code, action, snap, shared):
    writes = find_cell_writes(code=code.format(root=str(shared.root)), earlier_code=(), snapshots=(snap, shared))
    assert writes.needs_approval()
    held = writes.held_files[0]
    assert (held.action, held.whole_shared, held.whole_project) == (action, True, False)


@pytest.mark.parametrize("code", [
    "import os\nos.chdir({root!r})\nos.remove('forecast.xlsx')",
    "import subprocess\nsubprocess.run(['rm', 'forecast.xlsx'], cwd={root!r})",
    "import os\nos.system('cd {root} && rm forecast.xlsx')",
], ids=["chdir", "cwd-keyword", "shell-cd"])
def test_a_relative_path_after_changing_into_the_shared_folder_is_held(code, snap, shared):
    found = _files(code.format(root=str(shared.root)), (snap, shared))
    assert found == [("delete", ("shared folder: forecast.xlsx",))]


def test_a_relative_path_after_a_chdir_in_an_earlier_cell_reaches_the_shared_folder(snap, shared):
    found = _files("import os\nos.remove('forecast.xlsx')", (snap, shared),
                   earlier=(f"import os\nos.chdir({str(shared.root)!r})",))
    assert found == [("delete", ("shared folder: forecast.xlsx",))]


def test_a_path_spelled_through_a_link_is_held(tmp_path):
    real = tmp_path / "real"
    (real / "proj").mkdir(parents=True)
    (real / "proj" / "report.pdf").write_text("x")
    (tmp_path / "link").symlink_to(real)
    snap = snapshot_project_files(root=real / "proj", label="project", taken_at=time.time() + 60)
    linked = tmp_path / "link" / "proj"
    assert _files(f"import os\nos.remove({str(linked / 'report.pdf')!r})", (snap,)) == [("delete", ("report.pdf",))]
    whole = find_cell_writes(code=f"import shutil\nshutil.rmtree({str(linked)!r})", earlier_code=(), snapshots=(snap,))
    assert whole.held_files[0].whole_project
    given = snapshot_project_files(root=linked, label="project", taken_at=time.time() + 60)
    assert given.given_root == linked
    assert _files(f"import os\nos.remove({str(linked / 'report.pdf')!r})", (given,)) == [("delete", ("report.pdf",))]


def test_a_link_inside_the_project_is_its_own_entry(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "kept.csv").write_text("x")
    (tmp_path / "shortcut").symlink_to(tmp_path / "data")
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    found = _files(f"import os\nos.remove({str(tmp_path / 'shortcut')!r})", (snap,))
    assert found == [("delete", ("shortcut",))]


@pytest.mark.parametrize("call", [
    "os.remove({p!r})", "Path({p!r}).unlink()", "os.rename({p!r}, 'x.xlsx')", "shutil.move({p!r}, 'x.xlsx')",
    "subprocess.run(['rm', {p!r}])", "os.remove(os.path.join(os.getcwd(), {p!r}))",
])
@pytest.mark.parametrize("name", ["Q1 report [draft].xlsx", "data[1].csv"])
def test_a_name_with_glob_characters_matches_itself(call, name, tmp_path, monkeypatch):
    (tmp_path / name).write_text("x")
    monkeypatch.chdir(tmp_path)
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    code = "import os, shutil, subprocess\nfrom pathlib import Path\n" + call.format(p=name)
    assert [paths for _, paths in _files(code, (snap,))] == [(name,)]


def test_a_path_in_another_case_is_held_where_the_filesystem_ignores_case(tmp_path, monkeypatch):
    """Read as a filesystem that ignores case, as APFS does, whatever this
    one does: CI's is case-sensitive."""
    for rel in ("report.pdf", "stock.csv", "docs/a.pdf"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("anton.core.utils.write_check._folds_case", lambda _folder: True)
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    assert snap.casefold
    assert _files("import os\nos.remove('REPORT.PDF')", (snap,)) == [("delete", ("report.pdf",))]
    assert _files("import shutil\nshutil.rmtree('Docs')", (snap,)) == [("delete", ("docs",))]
    assert _files(f"import os\nos.remove({str(snap.root).upper() + '/Stock.CSV'!r})", (snap,)) == [
        ("delete", ("stock.csv",))
    ]


# ── The card ──────────────────────────────────────────────────────────────
#
# One Approve runs the whole cell, so the card shows every statement and
# every file whole, as code. Cowork renders the card as Markdown.

def test_the_card_text_holds_the_whole_statement_and_scrubs_it(snap):
    token = "ya29." + "a" * 30
    code = PG + (
        'cur.execute("""UPDATE suppliers\n   SET lead_time_days = 21, note = \'' + token + "'\n"
        '   WHERE supplier_id = %s""", ("' + token + '",))'
    )
    write = find_cell_writes(code=code, earlier_code=(), snapshots=(snap,)).sql[0]
    assert write.statement == (
        "UPDATE suppliers\nSET lead_time_days = 21, note = '[REDACTED_TOKEN]'\nWHERE supplier_id = %s"
    )
    assert write.values == "('[REDACTED_TOKEN]',)"
    assert token not in _card(code, snap)


def test_every_write_in_one_string_is_on_the_card(snap):
    card = _card(PG + 'cur.execute("UPDATE suppliers SET a = 1 WHERE id = 5; DROP TABLE suppliers_audit")', snap)
    assert _fenced(card) == ["UPDATE suppliers SET a = 1 WHERE id = 5", "DROP TABLE suppliers_audit"]


def test_every_write_in_a_cell_is_on_the_card(snap):
    deletes = [f"DELETE FROM orders WHERE region = 'r{i}'" for i in range(4)]
    code = PG + "".join(f'cur.execute("{d}")\n' for d in deletes) + "cur.execute('DROP TABLE customers')"
    card = _card(code, snap)
    assert _fenced(card) == [*deletes, "DROP TABLE customers"]
    assert "more" not in card


def test_every_file_a_delete_names_is_on_the_card(tmp_path, monkeypatch):
    names = [f"report_{i}.csv" for i in range(7)]
    for name in names:
        (tmp_path / name).write_text("x")
    monkeypatch.chdir(tmp_path)
    snapshot = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    card = _card("import glob, os\nfor f in glob.glob('report_*.csv'):\n    os.remove(f)", snapshot)
    assert _fenced(card) == ["\n".join(names)]
    assert "more" not in card


def test_a_long_statement_and_its_values_are_on_the_card_whole(snap):
    sets = ", ".join(f"col_{i:03d} = 'value_{i:03d}'" for i in range(30))
    statement = f"UPDATE customers SET {sets} WHERE region = 'EMEA' AND active"
    assert len(statement) > 700
    assert _fenced(_card(PG + f'cur.execute("{statement}")', snap)) == [statement]
    note = "b" * 250
    code = PG + f"cur.execute('UPDATE orders SET note = %s, price = %s WHERE id = %s', ('{note}', 0, 7))"
    assert _fenced(_card(code, snap)) == [
        f"UPDATE orders SET note = %s, price = %s WHERE id = %s\nwith values ('{note}', 0, 7)"
    ]


def test_the_card_keeps_the_statements_own_lines(snap):
    """A `--` comment keeps to its own line, so the OR after it still reads
    as SQL, and a literal keeps its spaces."""
    card = _card(PG + 'cur.execute("""DELETE FROM t WHERE id = 5 -- one row\n    OR 1 = 1""")', snap)
    assert _fenced(card) == ["DELETE FROM t WHERE id = 5 -- one row\nOR 1 = 1"]
    spaced = _card(PG + "cur.execute(\"UPDATE t SET a = 1\\n -- keep\\n WHERE b = 'x  y'\")", snap)
    assert _fenced(spaced) == ["UPDATE t SET a = 1\n-- keep\nWHERE b = 'x  y'"]
    indented = PG + 'def go():\n    cur.execute("""\n        UPDATE t\n        SET a = 1\n          + 2\n    """)\ngo()'
    assert _fenced(_card(indented, snap)) == ["UPDATE t\nSET a = 1\n  + 2"]


def test_file_names_and_the_quoted_code_reach_the_card_as_code(tmp_path, monkeypatch):
    names = ["src/__init__.py", "![x](http:leak.example.com).png", "<details open>x.txt", "a```b.txt"]
    for name in names:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x")
    monkeypatch.chdir(tmp_path)
    snapshot = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    for name in names:
        card = _card(f"import os\nos.remove({name!r})", snapshot)
        assert _fenced(card) == [name]
        assert f"This step would delete {name}" not in card
    assert "~~~\na```b.txt\n~~~" in _card("import os\nos.remove('a```b.txt')", snapshot), (
        "a name with three backticks in a row goes in a tilde fence"
    )
    picked = _card("import os\nos.remove(os.path.join(os.path.dirname(__file__), 'old.csv'))", snapshot)
    assert _fenced(picked) == ["os.remove(os.path.join(os.path.dirname(__file__), 'old.csv'))"]


def test_a_call_split_across_lines_is_quoted_whole(snap):
    call = "shutil.rmtree(\n    os.environ['TARGET'],\n    ignore_errors=True,\n)"
    writes = find_cell_writes(code="import os, shutil\n" + call, earlier_code=(), snapshots=(snap,))
    assert writes.held_files[0].call == call
    assert _fenced(_card_prompt(writes)) == [call]
    nested = "import os\nfor d in os.environ['DIRS'].split():\n    os.remove(\n        d)"
    assert find_cell_writes(code=nested, earlier_code=(), snapshots=(snap,)).held_files[0].call == "os.remove(\n    d)"
    helper = find_cell_writes(code="wipe(\n    os.environ['X'],\n)", earlier_code=(WIPE,), snapshots=(snap,))
    assert helper.held_files[0].call == "wipe(\n    os.environ['X'],\n)", "reported at the whole call"


def test_html_comment_marks_on_the_card_are_split(snap):
    """Cowork drops `<!-- ... -->` from the card before it reads the Markdown,
    fences included, and everything after an unclosed `<!--`."""
    card = _card(PG + "cur.execute(\"DELETE FROM orders WHERE note = '<!--' OR 1=1 OR note = '-->' AND id = 5\")", snap)
    assert "<!--" not in card and "-->" not in card
    assert _fenced(card) == ["DELETE FROM orders WHERE note = '<\u200b!--' OR 1=1 OR note = '--\u200b>' AND id = 5"]
    unclosed = _card(PG + "cur.execute(\"INSERT INTO t (body) VALUES ('<!-- header'); DROP TABLE customers\")", snap)
    assert "<!--" not in unclosed
    assert _fenced(unclosed) == ["INSERT INTO t (body) VALUES ('<\u200b!-- header')", "DROP TABLE customers"]


def test_a_part_picked_at_run_time_shows_the_code_that_fills_it(snap):
    code = PG + (
        "for r in rows:\n"
        "    cur.execute(f\"UPDATE suppliers SET lead_time_days = {r.new} WHERE supplier_id = '{r.id}'\")"
    )
    card = _card(code, snap)
    assert _fenced(card) == ["UPDATE suppliers SET lead_time_days = {r.new} WHERE supplier_id = '{r.id}'"]
    assert "A part in braces, `{...}`, is the code that fills in that value when the step runs." in card
    assert "_x_" not in card
    assert _fenced(_card(PG + "cur.execute('DROP TABLE ' + t)", snap)) == ["DROP TABLE {t}"]
    joined = _card(PG + "cur.execute(f\"DELETE FROM orders WHERE id IN ({','.join(map(str, ids))})\")", snap)
    assert _fenced(joined) == ["DELETE FROM orders WHERE id IN ({','.join(map(str, ids))})"]
    assert "braces" not in _card(PG + "cur.execute('DROP TABLE t')", snap)


@pytest.mark.parametrize("tail", [
    "CALL purge_all()",
    "COPY t TO PROGRAM 'rm -rf ~'",
    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity",
])
def test_every_statement_in_a_string_with_a_write_is_on_the_card(tail, snap):
    """One Approve runs the whole string, so a statement the check doesn't
    hold for still runs, and the card shows it."""
    text = f"UPDATE t SET a=1 WHERE id=5; {tail}"
    assert _fenced(_card(PG + f'cur.execute("{text}")', snap)) == ["UPDATE t SET a=1 WHERE id=5", tail]
    earlier = _card(PG + "cur.execute(q)", snap, earlier=(f'q = "{text}"',))
    assert _fenced(earlier) == ["UPDATE t SET a=1 WHERE id=5", tail]


@pytest.mark.parametrize("code, first", [
    ("cur.execute(f'UPDATE t SET a = 1 WHERE id = 5; {extra_sql}')", "UPDATE t SET a = 1 WHERE id = 5"),
    ("cur.execute('UPDATE t SET a = 1 WHERE id = 5; ' + extra_sql)", "UPDATE t SET a = 1 WHERE id = 5"),
    ("stmts = ['DELETE FROM a WHERE id = 1']\nstmts.append(extra_sql)\ncur.execute('; '.join(stmts))",
     "DELETE FROM a WHERE id = 1"),
], ids=["f-string", "concat", "join"])
def test_a_statement_picked_at_run_time_after_a_semicolon_is_on_the_card(code, first, snap):
    card = _card(PG + code, snap)
    assert _fenced(card) == [first, "{extra_sql}"]
    assert "A part in braces, `{...}`, is the code that fills in that value when the step runs." in card


@pytest.mark.parametrize("write, lead, overwritten", [
    ("open('report.pdf', 'w').write('gone')", "This step would overwrite:", "report.pdf"),
    ("Path('summary.md').write_text('')", "This step would overwrite:", "summary.md"),
    ("open('.anton/artifacts/dash/index.html', 'w').write('v2')", "This step would also overwrite:",
     ".anton/artifacts/dash/index.html"),
    ("open('summary.md', 'a').write('more')", "This step would also append to:", "summary.md"),
], ids=["held-open", "held-write-text", "agents-own-file", "append"])
def test_an_overwrite_beside_a_held_delete_is_on_the_card(write, lead, overwritten, snap):
    """A held overwrite is named as any held change is. An append, or an
    overwrite of the agent's own file, raises no card on its own, but the
    Approve runs it too, so the card names it as well."""
    card = _card(f"import os\nfrom pathlib import Path\nos.remove('stock.csv')\n{write}", snap)
    assert "This step would delete:" in card and lead in card
    assert _fenced(card) == ["stock.csv", overwritten]


def test_the_card_quotes_a_statement_through_its_end(snap):
    """Nothing after the last word the check reads is cut: a trailing comment
    stays, a carriage return ends a `--` comment as Postgres ends it, a `#`
    line stays, and a literal keeps its line breaks and spaces."""
    def statement(code: str) -> str:
        return find_cell_writes(code=PG + code, earlier_code=(), snapshots=(snap,)).sql[0].statement

    assert statement('cur.execute("DELETE FROM t WHERE id = 5 -- one row\\rOR 1 = 1")') == (
        "DELETE FROM t WHERE id = 5 -- one row\nOR 1 = 1"
    )
    assert statement('cur.execute("UPDATE t SET a = 1 WHERE id = 5 -- and more")') == (
        "UPDATE t SET a = 1 WHERE id = 5 -- and more"
    )
    assert statement('cur.execute("""DELETE FROM t WHERE id = 5\n#0 OR TRUE""")') == (
        "DELETE FROM t WHERE id = 5\n#0 OR TRUE"
    )
    assert statement("cur.execute(\"UPDATE t SET body = 'line1   \\n      indented' WHERE id = 1\")") == (
        "UPDATE t SET body = 'line1   \n      indented' WHERE id = 1"
    )


@pytest.mark.parametrize("code, fenced", [
    ("sid = 'S005'\ncur.execute(\"DELETE FROM suppliers WHERE id = '%s'\" % sid)",
     "DELETE FROM suppliers WHERE id = '{sid}'"),
    ("cur.execute('DELETE FROM %s WHERE id = %s' % p)", "DELETE FROM {p[0]} WHERE id = {p[1]}"),
    ("cur.execute('DELETE FROM {} WHERE id = 1'.format(tbl))", "DELETE FROM {tbl} WHERE id = 1"),
    ("cur.execute('DELETE FROM t WHERE id = {!r}'.format('x'))", "DELETE FROM t WHERE id = {'x'}"),
    ("from psycopg2 import sql\n"
     "cur.execute(sql.SQL('DELETE FROM {} WHERE id = %s').format(sql.Identifier(tbl)), (5,))",
     "DELETE FROM {sql.Identifier(tbl)} WHERE id = %s\nwith values (5,)"),
], ids=["percent-name", "percent-tuple-name", "format-name", "format-conversion", "psycopg2-sql"])
def test_a_format_shows_the_code_that_fills_each_field(code, fenced, snap):
    card = _card(PG + code, snap)
    assert _fenced(card) == [fenced]
    assert "A part in braces" in card


@pytest.mark.parametrize("code, values", [
    ("cur.execute(query='DELETE FROM t WHERE id = %s OR region = %s', vars=(5, 'all'))", "(5, 'all')"),
    ("c.execute(text('DELETE FROM t WHERE id = :id OR region = :r'), parameters={'id': 5, 'r': 'all'})",
     "{'id': 5, 'r': 'all'}"),
    ("params = (5, 'all')\ncur.execute('DELETE FROM t WHERE id = %s OR region = %s', params)", "(5, 'all')"),
    ("params = [5]\nparams.append('all')\ncur.execute('DELETE FROM t WHERE id = %s OR region = %s', params)",
     "{params}"),
], ids=["vars-keyword", "parameters-keyword", "tuple-name", "list-name"])
def test_the_values_a_statement_runs_with_are_on_the_card(code, values, snap):
    code = PG + "from sqlalchemy import text\n" + code
    assert find_cell_writes(code=code, earlier_code=(), snapshots=(snap,)).sql[0].values == values


def test_a_literal_that_looks_like_a_placeholder_is_quoted_as_written(snap):
    card = _card(PG + "cur.execute(f\"UPDATE t SET tag = 'a_x_0_b' WHERE id = {i}\")", snap)
    assert _fenced(card) == ["UPDATE t SET tag = 'a_x_0_b' WHERE id = {i}"]


def test_a_token_value_after_a_line_break_is_scrubbed(snap):
    token = "abcdefghijklmnopqrstuvwxyz0123456789"
    for key in ("access_token=\\n", "refresh_token\\n", "access_token\\n=\\n", "client_secret:\\n\\t"):
        card = _card(PG + f"cur.execute(\"UPDATE cfg SET v = '{key}{token}' WHERE id = 1\")", snap)
        assert token not in card and "[REDACTED_TOKEN]" in card, key


@pytest.mark.parametrize("sql, shown", [
    ("UPDATE cfg SET v = 'access_token=\\n{t}' WHERE id = 1", "UPDATE cfg SET v = 'access_token=\n[REDACTED_TOKEN]' WHERE id = 1"),
    ("UPDATE cfg SET v = 'access_token={t}' WHERE id = 1", "UPDATE cfg SET v = 'access_token=[REDACTED_TOKEN]' WHERE id = 1"),
    ("UPDATE cfg SET access_token = '{t}' WHERE id = 1", "UPDATE cfg SET access_token = '[REDACTED_TOKEN]' WHERE id = 1"),
    ("UPDATE cfg SET v = $$access_token=\\n{t}$$ WHERE id = 1",
     "UPDATE cfg SET v = $$access_token=\n[REDACTED_TOKEN]$$ WHERE id = 1"),
    ("UPDATE cfg SET v = 1 /* access_token={t} */ WHERE id = 1",
     "UPDATE cfg SET v = 1 /* access_token=[REDACTED_TOKEN]*/ WHERE id = 1"),
], ids=["next-line", "same-line", "key-before-the-literal", "dollar-quoted", "comment"])
def test_a_scrubbed_token_leaves_the_where_on_the_card(sql, shown, snap):
    """The scrubber hides only the token: the WHERE after it, which says
    which rows change, stays on the card."""
    token = "abcdefghijklmnopqrstuvwxyz0123456789"
    card = _card(PG + f"cur.execute(\"{sql.format(t=token)}\")", snap)
    assert token not in card
    assert _fenced(card) == [shown]


@pytest.mark.parametrize("sql", [
    "UPDATE users SET access_token = NULL WHERE tenant_id = 42",
    "UPDATE sessions SET access_token = NULL WHERE expires_at < now()",
    "UPDATE t SET refresh_token = NULL WHERE id = 1",
    "UPDATE sessions SET client_secret = NULL WHERE app_id = 3",
    "UPDATE cfg SET client_secret = DEFAULT WHERE id = 1",
    "DELETE FROM tokens WHERE access_token = abcdefghijklmnopqrstu OR 1 = 1",
], ids=["null", "null-before-a-call", "refresh-token", "client-secret", "default", "compared-to-a-column"])
def test_an_unquoted_token_column_keeps_the_where_on_the_card(sql, snap):
    """A token column set to NULL or DEFAULT, or compared to another column,
    has no secret to hide: the card quotes the statement whole, its WHERE
    and an `OR 1 = 1` that widens it included."""
    card = _card(PG + f'cur.execute("{sql}")', snap)
    assert _fenced(card) == [sql]


def test_a_token_in_a_dict_the_address_is_built_from_is_scrubbed_on_the_card():
    """The run-time code for an address is unparsed, with a space after each
    dict colon, so the token stands as a word of its own there."""
    token = "abcdefghijklmnopqrstuvwxyz0123456789"
    writes = find_cell_writes(code=f"import requests\nrequests.delete(make_url({{'access_token': '{token}'}}))",
                              earlier_code=(), snapshots=())
    [request] = writes.http
    assert request.url_code == "make_url({'access_token': '[REDACTED_TOKEN]'})"
    assert token not in _card_prompt(writes)


@pytest.mark.parametrize("values", [
    "('access_token=\\n{t}', 1)",
    "{{'v': 'refresh_token\\n{t}', 'id': 1}}",
], ids=["tuple", "dict"])
def test_a_token_in_the_values_is_scrubbed_on_the_card(values, snap):
    """The values line is unparsed code, where a line break shows as an
    escape; the token is scrubbed in the value itself, before that."""
    token = "abcdefghijklmnopqrstuvwxyz0123456789"
    code = PG + f"cur.execute('UPDATE cfg SET v = %(v)s WHERE id = %(id)s', {values.format(t=token)})"
    card = _card(code, snap)
    assert token not in card and "WHERE id" in card


def test_a_name_that_does_not_show_as_itself_is_escaped(tmp_path, monkeypatch):
    """A line break or a direction override in a name would misdraw it, so it
    shows as Python writes it, in quotes; so does a name that starts with a
    quote, which would read as one of those."""
    for name in ("a\nb.csv", "rlo\u202efdp.exe", "'quoted'.csv", "plain.csv"):
        (tmp_path / name).write_text("x")
    monkeypatch.chdir(tmp_path)
    snapshot = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    writes = find_cell_writes(code="import os\nfor f in os.listdir('.'):\n    os.remove(f)",
                              earlier_code=(), snapshots=(snapshot,))
    card = _card_prompt(writes)
    [listed] = _fenced(card)
    assert sorted(listed.split("\n")) == sorted(["'a\\nb.csv'", "'rlo\\u202efdp.exe'", "\"'quoted'.csv\"", "plain.csv"])
    assert "A name in quotes is written with escapes" in card
    refusal = _refusal(writes, "cancelled").content
    assert "`'a\\nb.csv'`" in refusal and "a\nb.csv" not in refusal and "\u202e" not in refusal


# The code Cowork's math pass skips (`_MD_CODE_REGION` in cowork's
# MarkdownContent.jsx). It rewrites a `$` or `\(` anywhere else, fences included.
_COWORK_CODE = re.compile(r"(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)")


def test_cowork_s_math_pass_skips_every_fence_on_the_card(tmp_path, monkeypatch, snap):
    sql = _card(PG + "cur.execute(\"UPDATE t SET note = '```' WHERE id = 1; "
                     "CREATE FUNCTION f() RETURNS void AS $body$ DELETE FROM audit $body$ LANGUAGE sql\")", snap)
    assert "$" not in _COWORK_CODE.sub("", sql)
    for name in ("a```b.txt", "b $x$.csv", "c \\(y\\).csv", "d~~~e```f $z$.csv"):
        (tmp_path / name).write_text("x")
    monkeypatch.chdir(tmp_path)
    snapshot = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    card = _card("import os\nfor f in os.listdir('.'):\n    os.remove(f)", snapshot)
    outside = _COWORK_CODE.sub("", card)
    assert "$" not in outside and "\\(" not in outside
    assert card.count("~~~") == 2, "a run of three tildes in a name can't end the tilde fence"


@pytest.mark.parametrize("action", ["delete", "overwrite"])
def test_the_refusal_names_the_first_paths_and_the_card_names_them_all(action):
    paths = tuple(f"many/f_{i:05d}.csv" for i in range(25))
    writes = CellWrites(files=(FileChange(action=action, paths=paths, call="os.remove(f)", line=3),))
    refusal = _refusal(writes, "cancelled").content
    assert "`many/f_00019.csv` and 5 more." in refusal and "f_00020" not in refusal
    assert _fenced(_card_prompt(writes)) == ["\n".join(paths)]


def test_the_refusal_names_the_first_requests_and_data_changes_and_the_card_names_them_all():
    urls = [f"https://erp.example.com/api/s/{i:05d}" for i in range(25)]
    requests = CellWrites(http=tuple(
        HttpWrite(method="PUT", url=url, call=f"requests.put({url!r})", line=i) for i, url in enumerate(urls)
    ))
    refusal = _refusal(requests, "cancelled").content
    assert "`https://erp.example.com/api/s/00019`; send 5 more such requests." in refusal
    assert "s/00020" not in refusal and all(url in _fenced(_card_prompt(requests)) for url in urls)
    calls = CellWrites(calls=tuple(CallWrite(via="redis", call=f"r.set('k{i:05d}', 1)", line=i) for i in range(25)))
    refusal = _refusal(calls, "cancelled").content
    assert "`r.set('k00019', 1)`; make 5 more such data changes." in refusal and "k00020" not in refusal
    assert _card_prompt(calls).count("This step would change data in Redis:") == 25


def test_fences_and_code_spans_take_linear_time():
    """The fence length used to grow one backtick per scan of the text."""
    started = time.perf_counter()
    _fence("`" * 200_000)
    _code_span("`" * 200_000)
    assert time.perf_counter() - started < 1


# ── Snapshots ─────────────────────────────────────────────────────────────

def test_the_snapshot_keeps_artifacts_and_skips_the_hosts_own_folders(snap):
    assert ".anton/artifacts/dash/index.html" in snap.files
    assert ".anton/memory/notes.md" not in snap.files
    assert ".anton" in snap.dirs
    assert ".git" in snap.unwalked and ".git/index" not in snap.files
    assert snap.holds(rel=".git/index"), "a path under an unwalked folder counts as existing"
    assert ".anton/memory" in snap.unwalked and snap.holds(rel=".anton/memory/notes.md")


def test_every_host_file_under_dot_anton_counts_as_existing(tmp_path, monkeypatch):
    for rel in (".anton/anton.md", ".anton/.env", ".anton/memory/rules.md",
                ".anton/scratchpad-venvs/main/bin/python", ".anton/skill_drafts/weekly/SKILL.md"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    monkeypatch.chdir(tmp_path)
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    assert {".anton/anton.md", ".anton/.env", ".anton/skill_drafts/weekly/SKILL.md"} <= snap.files
    assert snap.unwalked == {".anton/memory", ".anton/scratchpad-venvs"}, "venvs aren't walked"
    for code in ("import os\nos.remove('.anton/anton.md')",
                 "import os\nos.rename('.anton/.env', '.anton/env.bak')",
                 "import shutil\nshutil.rmtree('.anton/memory')",
                 "import shutil\nshutil.rmtree('.anton/skill_drafts/weekly')"):
        assert _held(code, snap), code


def test_a_temp_file_renamed_into_place_under_anton_output_raises_no_card(tmp_path, monkeypatch):
    """The CLI saves what it makes under .anton/output: the walk records it."""
    (tmp_path / ".anton/output").mkdir(parents=True)
    (tmp_path / ".anton/output/old.html").write_text("x")
    monkeypatch.chdir(tmp_path)
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    assert ".anton/output/old.html" in snap.files and ".anton/output" not in snap.unwalked
    for code in ("import os\nopen('.anton/output/r.tmp', 'w').write('x')\n"
                 "os.replace('.anton/output/r.tmp', '.anton/output/r.html')",
                 "import os\nopen('.anton/output/r.tmp', 'w').write('x')\nos.remove('.anton/output/r.tmp')"):
        assert not _held(code, snap), code
    assert _held("import os\nos.remove('.anton/output/old.html')", snap)


def test_a_glob_into_a_folder_the_walk_did_not_record_still_matches(snap, tmp_path):
    assert _files("import glob, os\nfor f in glob.glob('.git/*'):\n    os.remove(f)", (snap,)) == [
        ("delete", (".git/index",))
    ]
    assert _files("from pathlib import Path\nfor p in Path('.').rglob('index'):\n    p.unlink()", (snap,)) == [
        ("delete", (".git/index",))
    ]
    for d in ("d1", "d2"):
        (tmp_path / "big" / d).mkdir(parents=True)
        for i in range(10):
            (tmp_path / "big" / d / f"r{i}.pdf").write_text("x")
    capped = snapshot_project_files(root=tmp_path / "big", label="project", taken_at=time.time() + 60, cap=5)
    assert not capped.complete
    pattern = str(tmp_path / "big" / "d2" / "*.pdf")
    found = _files(f"import glob, os\nfor f in glob.glob({pattern!r}):\n    os.remove(f)", (capped,))
    assert found and len(found[0][1]) == 10


def test_a_check_reads_each_folder_once_per_pattern(snap, monkeypatch):
    """Loops that share a name each read every earlier loop's pattern too;
    the disk is read once per pattern and folder in one check."""
    from anton.core.utils.write_check import ProjectSnapshot

    reads: list[tuple[str, str]] = []
    on_disk = ProjectSnapshot._on_disk

    def counting(self, *, pattern: str, start: str, older: bool, made=None) -> list[str]:
        reads.append((pattern, start))
        return on_disk(self, pattern=pattern, start=start, older=older, made=made)

    monkeypatch.setattr(ProjectSnapshot, "_on_disk", counting)
    code = "from pathlib import Path\n" + "\n".join(
        f"for f in Path('.').rglob('*.ext{i}'):\n    f.unlink()" for i in range(6)
    )
    assert not _held(code, snap)
    assert reads and len(reads) == len(set(reads)) == 6 * len(snap.unwalked)


def test_ordinary_projects_stay_under_the_cap(tmp_path):
    """Above the old 20,000-entry cap the walk stayed set-based, so a file
    the turn touched before deleting it still counts as existing."""
    for i in range(20_050):
        os.close(os.open(tmp_path / f"f{i}", os.O_CREAT | os.O_WRONLY))
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time())
    assert snap.complete
    os.utime(tmp_path / "f20049", (time.time() + 5, time.time() + 5))
    assert snap.holds(rel="f20049")


def test_past_the_cap_an_older_file_still_counts(tmp_path):
    for i in range(30):
        (tmp_path / f"f{i}.csv").write_text("x")
    capped = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60, cap=10)
    assert not capped.complete
    assert all(capped.holds(rel=f"f{i}.csv") for i in range(30))
    assert not capped.holds(rel="missing.csv")


def test_past_the_cap_only_a_file_the_pads_cells_made_is_new(tmp_path, monkeypatch):
    """A file the turn changed counts as existing unless a cell's write left
    it new: an append or a touch keeps what it held."""
    names = [f"d{d}/f{i}.csv" for d in range(6) for i in range(5)]
    for rel in names:
        (tmp_path / rel).parent.mkdir(exist_ok=True)
        (tmp_path / rel).write_text("x")
    taken_at = time.time()
    capped = snapshot_project_files(root=tmp_path, label="project", taken_at=taken_at, cap=10)
    assert not capped.complete
    unrecorded = [rel for rel in names if rel not in capped.files]
    folder = unrecorded[0].split("/")[0]
    old, touched = [rel for rel in unrecorded if rel.startswith(folder + "/")][:2]
    new = f"{folder}/new.csv"
    os.utime(tmp_path / old, (taken_at - 60, taken_at - 60))
    (tmp_path / new).write_text("x")
    for rel in (touched, new):
        os.utime(tmp_path / rel, (taken_at + 5, taken_at + 5))
    monkeypatch.chdir(tmp_path)
    assert capped.holds(rel=old)
    assert capped.holds(rel=touched), "with no cells read, a path the walk didn't record counts"
    made = (f"open({new!r}, 'w').write('x')",)
    assert not _held(f"import os\nos.remove({new!r})", capped, earlier=made), "the turn's own file gets no card"
    for earlier in (f"open({touched!r}, 'a').write('x')", f"import os\nos.utime({touched!r})"):
        assert _files(f"import os\nos.remove({touched!r})", (capped,), earlier=(earlier,)) == [
            ("delete", (touched,))
        ], earlier
    listed = _files(f"import glob, os\nfor f in glob.glob({folder + '/*.csv'!r}):\n    os.remove(f)", (capped,),
                    earlier=(*made, f"open({touched!r}, 'a').write('x')"))
    assert touched in listed[0][1] and new not in listed[0][1]


def test_past_the_cap_a_file_a_cell_wrote_under_a_name_picked_at_run_time_is_new(tmp_path, monkeypatch):
    """A part of the name the code picks when it runs names any file there,
    as a `*` does."""
    names = [f"d{d}/f{i}.csv" for d in range(6) for i in range(5)]
    for rel in names:
        (tmp_path / rel).parent.mkdir(exist_ok=True)
        (tmp_path / rel).write_text("x")
    taken_at = time.time()
    capped = snapshot_project_files(root=tmp_path, label="project", taken_at=taken_at, cap=10)
    folder = next(rel for rel in names if rel not in capped.files).split("/")[0]
    new = f"{folder}/new.csv"
    (tmp_path / new).write_text("x")
    os.utime(tmp_path / new, (taken_at + 5, taken_at + 5))
    monkeypatch.chdir(tmp_path)
    remove = f"import os\nos.remove({new!r})"
    assert _held(remove, capped), "with no cells read, a path the walk didn't record counts"
    assert not _held(remove, capped, earlier=(f"open(f'{folder}/{{name}}.csv', 'w').write('x')",))


def test_past_the_cap_an_overwrite_of_a_file_the_turn_changed_is_held_unless_a_cell_made_it(tmp_path, monkeypatch):
    """The cap rule deletes follow: a file the walk didn't record that the
    turn changed counts as one it began with, unless a cell's write left it
    new. An append keeps what the file held."""
    names = [f"d{d}/f{i}.csv" for d in range(6) for i in range(5)]
    for rel in names:
        (tmp_path / rel).parent.mkdir(exist_ok=True)
        (tmp_path / rel).write_text("x")
    taken_at = time.time()
    capped = snapshot_project_files(root=tmp_path, label="project", taken_at=taken_at, cap=10)
    folder = next(rel for rel in names if rel not in capped.files).split("/")[0]
    touched = next(rel for rel in names if rel.startswith(folder + "/") and rel not in capped.files)
    new = f"{folder}/new.csv"
    (tmp_path / new).write_text("x")
    for rel in (touched, new):
        os.utime(tmp_path / rel, (taken_at + 5, taken_at + 5))
    monkeypatch.chdir(tmp_path)
    appended = (f"open({touched!r}, 'a').write('x')",)
    assert _changes(f"open({touched!r}, 'w').write('y')", capped, earlier=appended) == [
        ("overwrite", (touched,), True)
    ]
    made = (f"open({new!r}, 'w').write('x')",)
    assert not _held(f"import pandas as pd\npd.DataFrame().to_csv({new!r})", capped, earlier=made)


def test_an_unreadable_folder_fails_the_snapshot_closed(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "a.csv").write_text("x")
    os.chmod(locked, 0)
    try:
        failed = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time())
    finally:
        os.chmod(locked, 0o755)
    if os.geteuid() == 0:
        pytest.skip("root reads every folder")
    assert failed.failed and failed.holds(rel="anything")


def test_a_missing_root_fails_closed(tmp_path):
    gone = snapshot_project_files(root=tmp_path / "nope", label="project", taken_at=time.time())
    assert gone.failed


def test_glob_match_keeps_star_within_a_folder():
    assert glob_match("report.pdf", "*.pdf")
    assert not glob_match("docs/a.pdf", "*.pdf")
    assert glob_match("docs/a.pdf", "**/*.pdf")
    assert glob_match("a.pdf", "**/*.pdf")


# ── SQL hidden in a decoded literal ─────────────────────────────────────────
#
# A write encoded in base64, base32, base16, hex or rot13 and decoded at run
# time is decoded now and checked like any SQL text.

# "UPDATE suppliers SET a = 1 WHERE id = 5" in each encoding.
_HIDDEN = "UPDATE suppliers SET a = 1 WHERE id = 5"
_B64 = "VVBEQVRFIHN1cHBsaWVycyBTRVQgYSA9IDEgV0hFUkUgaWQgPSA1"
_B32 = "KVIEIQKUIUQHG5LQOBWGSZLSOMQFGRKUEBQSAPJAGEQFOSCFKJCSA2LEEA6SANI="
_B16 = "55504441544520737570706C69657273205345542061203D2031205748455245206964203D2035"
_HEX = "55504441544520737570706c69657273205345542061203d2031205748455245206964203d2035"
_ROT13 = "HCQNGR fhccyvref FRG n = 1 JURER vq = 5"


@pytest.mark.parametrize("code", [
    PG + f"import base64\ncur.execute(base64.b64decode('{_B64}').decode())",
    PG + f"import base64\ncur.execute(base64.b64decode(b'{_B64}'))",
    PG + f"import base64\ncur.execute(base64.urlsafe_b64decode('{_B64}').decode())",
    PG + f"import base64\ncur.execute(base64.b32decode('{_B32}').decode())",
    PG + f"import base64\ncur.execute(base64.b16decode('{_B16}').decode())",
    PG + f"cur.execute(bytes.fromhex('{_HEX}').decode())",
    PG + f"import codecs\ncur.execute(codecs.decode('{_HEX}', 'hex').decode())",
    PG + f"import codecs\ncur.execute(codecs.decode('{_ROT13}', 'rot13'))",
], ids=["b64", "b64-bytes", "urlsafe-b64", "b32", "b16", "bytes-fromhex", "codecs-hex", "codecs-rot13"])
def test_a_write_hidden_in_a_decoded_literal_is_checked(code, snap):
    assert _held(code, snap)


def test_a_decoded_literal_is_quoted_on_the_card(snap):
    code = PG + f"import base64\ncur.execute(base64.b64decode('{_B64}').decode())"
    assert _statements(code, snap) == ["UPDATE suppliers SET a = 1 WHERE id = 5"]
    assert _fenced(_card(code, snap)) == ["UPDATE suppliers SET a = 1 WHERE id = 5"]


@pytest.mark.parametrize("code", [
    # A decoded non-SQL payload stays free.
    PG + "import base64\nmsg = base64.b64decode('aGVsbG8gd29ybGQ=').decode()\nprint(msg)",
    # An unknown encoding isn't read.
    PG + "import codecs\ncur.execute(codecs.decode('a2V5', 'base85'))",
    # A decoded statement only printed stays free.
    PG + f"import base64\nprint(base64.b64decode('{_B64}').decode())",
])
def test_a_decoded_literal_that_is_no_write_stays_free(code, snap):
    assert not _held(code, snap)


# ── SQL read from a file ────────────────────────────────────────────────────

@pytest.fixture
def sql_files(tmp_path: Path, monkeypatch):
    (tmp_path / "fix.sql").write_text(
        "UPDATE suppliers SET lead_time_days = 21 WHERE supplier_id = 'S005';\nDELETE FROM orders WHERE id = 3;"
    )
    (tmp_path / "read.sql").write_text("SELECT name FROM suppliers ORDER BY 1")
    (tmp_path / "big.sql").write_text("UPDATE t SET a = 1;\n" * 80_000)
    assert (tmp_path / "big.sql").stat().st_size > 1_000_000
    monkeypatch.chdir(tmp_path)
    return snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)


@pytest.mark.parametrize("code", [
    PG + "cur.execute(open('fix.sql').read())",
    PG + "from pathlib import Path\ncur.execute(Path('fix.sql').read_text())",
    PG + "sql = open('fix.sql').read()\ncur.execute(sql)",
], ids=["open-read", "path-read_text", "named"])
def test_sql_read_from_a_file_is_checked(code, sql_files):
    assert _held(code, sql_files)
    assert _statements(code, sql_files) == [
        "UPDATE suppliers SET lead_time_days = 21 WHERE supplier_id = 'S005'",
        "DELETE FROM orders WHERE id = 3",
    ]


def test_a_read_only_sql_file_raises_no_card(sql_files):
    assert not _held(PG + "cur.execute(open('read.sql').read())", sql_files)
    assert not _held(PG + "from pathlib import Path\ncur.execute(Path('missing.sql').read_text())", sql_files)


def test_a_sql_file_past_the_cap_is_held_unchecked(sql_files):
    writes = find_cell_writes(code=PG + "cur.execute(open('big.sql').read())",
                              earlier_code=(), snapshots=(sql_files,))
    assert writes.needs_approval()
    assert [c.via for c in writes.calls] == ["sql_file"]
    assert writes.sql == ()
    card = _card_prompt(writes)
    assert "This step would run a SQL file too big to check:" in card
    assert "open('big.sql').read()" in card


# ── Writes with no SQL text: procedures, SQLAlchemy, non-SQL stores ─────────

@pytest.mark.parametrize("code, via", [
    (PG + "cur.callproc('refresh_totals', [1, 2])", "procedure"),
    ("from sqlalchemy import insert\nwith engine.begin() as c:\n    c.execute(insert(users), rows)", "sqlalchemy"),
    ("from sqlalchemy import update\nsession.execute(update(User).where(User.id == 1).values(a = 2))", "sqlalchemy"),
    ("conn.execute(suppliers.delete())", "sqlalchemy"),
    ("from sqlalchemy import delete\nstmt = delete(User)\nconn.execute(stmt)", "sqlalchemy"),
    ("users.drop(engine)", "sqlalchemy"),
    ("users.create(bind=engine)", "sqlalchemy"),
    ("Base.metadata.drop_all(engine)", "sqlalchemy"),
    ("meta.create_all(engine)", "sqlalchemy"),
    ("session.query(User).filter(User.id == 1).delete()", "sqlalchemy"),
    ("session.query(User).update({'a': 1})", "sqlalchemy"),
    ("session.add(User(id = 1))\nsession.commit()", "sqlalchemy"),
    ("session.add_all(objs)\nsession.flush()", "sqlalchemy"),
    ("s = Session()\ns.merge(obj)\ns.commit()", "sqlalchemy"),
    ("session.bulk_save_objects(objs)\nsession.commit()", "sqlalchemy"),
    ("db.users.insert_one({'a': 1})", "mongodb"),
    ("coll.update_many({}, {'$set': {'a': 1}})", "mongodb"),
    ("db.users.drop_collection()", "mongodb"),
    ("client.drop_database('prod')", "mongodb"),
    ("import pymongo\nclient = pymongo.MongoClient()\ndb = client['prod']\ncol = db['users']\ncol.drop()", "mongodb"),
    ("table.put_item(Item = {'id': '1'})", "dynamodb"),
    ("table.batch_write_item(RequestItems = {})", "dynamodb"),
    ("import boto3\nclient = boto3.client('dynamodb')\nclient.delete_table(TableName = 't')", "dynamodb"),
    ("import redis\nr = redis.Redis()\nr.set('k', 'v')", "redis"),
    ("import redis\nr = redis.from_url('redis://x')\nr.delete('k')", "redis"),
    ("from redis import StrictRedis\nr = StrictRedis()\nr.flushall()", "redis"),
    ("es.delete_by_query(index = 'i', body = {})", "elasticsearch"),
    ("es.indices.delete(index = 'i')", "elasticsearch"),
    ("from elasticsearch import Elasticsearch\nes = Elasticsearch()\nes.index(index = 'i', document = {})", "elasticsearch"),
], ids=["callproc", "sa-insert", "sa-update-session", "sa-table-delete", "sa-named-construct",
        "sa-table-drop", "sa-table-create", "sa-drop-all", "sa-create-all", "sa-query-delete",
        "sa-query-update", "sa-add-commit", "sa-add-all-flush", "sa-merge-commit", "sa-bulk-save",
        "mongo-insert-one", "mongo-update-many", "mongo-drop-collection", "mongo-drop-database",
        "mongo-collection-drop", "dynamo-put-item", "dynamo-batch-write", "dynamo-delete-table",
        "redis-set", "redis-from-url-delete", "redis-flushall", "es-delete-by-query",
        "es-indices-delete", "es-index"])
def test_a_write_with_no_sql_text_is_held(code, via, snap):
    writes = find_cell_writes(code=code, earlier_code=(), snapshots=(snap,))
    assert writes.needs_approval()
    assert [c.via for c in writes.calls] == [via]


@pytest.mark.parametrize("code", [
    # Read-only work on the same libraries stays free.
    "import pandas as pd\ndf = pd.read_csv('stock.csv')\ndf = df.drop(columns=['a'])",
    "import pandas as pd\ndf = pd.read_csv('stock.csv')\ndf.set_index('site')",
    "list(db.users.find({'a': 1}))",
    "list(coll.aggregate([{'$match': {}}]))",
    "from sqlalchemy import select\nconn.execute(select(User).where(User.id == 1))",
    "r.get('k')",
    "import redis\nr = redis.Redis()\nr.get('k')",
    "es.search(index = 'i', body = {})",
    # Generic method names on an unknown receiver stay free (pandas, dicts, sets).
    "obj.set('a', 1)",
    "obj.delete()",
    "obj.update({'a': 1})",
    "d = {}\nd.update({'a': 1})",
    "s = set()\ns.update([1, 2])",
    "df.insert(0, 'col', vals)",
    # callproc on an in-process handle stays free.
    "import sqlite3\nc = sqlite3.connect(':memory:')\nc.callproc('p')",
    # A session add with no commit or flush in the cell stays free.
    "session.add(User(id = 1))",
    # An openpyxl worksheet's insert_rows changes only the workbook in memory.
    "import openpyxl\nwb = openpyxl.load_workbook('a.xlsx')\nwb.active.insert_rows(2)",
], ids=["pandas-drop", "pandas-set-index", "pymongo-find", "pymongo-aggregate", "sqlalchemy-select",
        "redis-get-unknown", "redis-get-client", "es-search", "unknown-set", "unknown-delete",
        "unknown-update", "dict-update", "set-update", "df-insert",
        "callproc-local", "session-add-no-commit", "openpyxl-insert-rows"])
def test_read_only_calls_on_the_same_libraries_stay_free(code, snap):
    assert not _held(code, snap)


def test_a_sqlalchemy_write_quotes_the_code_line_on_the_card(snap):
    writes = find_cell_writes(code="conn.execute(suppliers.delete())", earlier_code=(), snapshots=(snap,))
    card = _card_prompt(writes)
    assert "This step would change data through SQLAlchemy:" in card
    assert _fenced(card) == ["conn.execute(suppliers.delete())"]


def test_a_store_write_quotes_the_code_line_on_the_card(snap):
    writes = find_cell_writes(code="db.users.insert_one({'id': 1})", earlier_code=(), snapshots=(snap,))
    card = _card_prompt(writes)
    assert "This step would change data in MongoDB:" in card
    assert _fenced(card) == ["db.users.insert_one({'id': 1})"]


BQ = "from google.cloud import bigquery\nclient = bigquery.Client(project='p')\n"


@pytest.mark.parametrize("code, via", [
    (BQ + "client.delete_table('p.d.orders', not_found_ok=True)", "bigquery"),
    (BQ + "client.create_table('p.d.orders')", "bigquery"),
    ("from google.cloud.bigquery import Client\nc = Client()\nc.delete_table('p.d.orders')", "bigquery"),
    (BQ + "client.insert_rows(table, rows)", "bigquery"),
    ("client.insert_rows_json('p.d.orders', rows)", "bigquery"),
    ("client.delete_dataset('p.d', delete_contents=True)", "bigquery"),
    (BQ + "config = bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE')\n"
     "client.load_table_from_dataframe(df, 'p.d.orders', job_config=config).result()", "bigquery"),
    ("import boto3\nddb = boto3.resource('dynamodb')\nddb.create_table(TableName = 't', KeySchema = [])", "dynamodb"),
    ("import boto3\ns = boto3.Session()\nddb = s.client(service_name = 'dynamodb')\nddb.delete_table(TableName = 't')",
     "dynamodb"),
    ("client.delete_table(TableName = 't')", "data_store"),
    ("client.delete_table('p.d.orders')", "data_store"),
], ids=["bq-delete-table", "bq-create-table", "bq-client-imported", "bq-insert-rows", "bq-insert-rows-json",
        "bq-delete-dataset", "bq-load-dataframe", "dynamo-resource-create-table", "dynamo-session-client",
        "untraced-keywords", "untraced-positional"])
def test_a_table_call_is_named_for_the_client_it_traces_to(code, via, snap):
    """DynamoDB's and BigQuery's clients both have `delete_table` and
    `create_table`: the card names the store the receiver traces to, and
    neither for one the check can't trace."""
    writes = find_cell_writes(code=code, earlier_code=(), snapshots=(snap,))
    assert writes.needs_approval()
    assert [c.via for c in writes.calls] == [via]


def test_a_bigquery_table_delete_names_bigquery_on_the_card(snap):
    card = _card(BQ + "client.delete_table('p.d.orders')", snap)
    assert "This step would change data in BigQuery:" in card
    assert "DynamoDB" not in card
    assert "This step would change data in a data store:" in _card("client.delete_table('p.d.orders')", snap)


def test_a_call_write_inside_an_earlier_function_is_held_at_the_call(snap):
    earlier = ("def wipe(coll):\n    coll.delete_many({})",)
    writes = find_cell_writes(code="wipe(users)", earlier_code=earlier, snapshots=(snap,))
    assert [(c.via, c.line, c.call) for c in writes.calls] == [("mongodb", 1, "wipe(users)")]
