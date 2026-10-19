"""What `find_cell_writes` holds, and what it lets through.

A missed write runs with no approval. A false hold costs an unattended answer
the full card timeout and then the answer itself. So each rule here has cells
it must hold and cells, often the same call shape, it must not.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pytest

from anton.core.utils.scratchpad import _card_prompt, _code_span, _fence, _refusal
from anton.core.utils.write_check import (
    CellWrites,
    FileChange,
    find_cell_writes,
    glob_match,
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
])
def test_reads_and_prose(text):
    assert write_statements(text) == ()


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
])
def test_an_overwrite_is_reported_not_held(code, snap):
    writes = find_cell_writes(code=code, earlier_code=(), snapshots=(snap,))
    assert not writes.needs_approval()
    assert [c.action for c in writes.files] and {c.action for c in writes.files} == {"overwrite"}


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
    ("import os\nfor n in ['exists']:\n    print(getattr(os.path, n)('report.pdf'))", []),
], ids=["getattr-os", "getattr-shutil", "getattr-value", "dunder-import", "Path.unlink-unbound",
        "Path.rename-unbound", "Path.unlink-loop", "map-Path.unlink", "Path.rmdir-unbound", "getattr-default",
        "getattr-named-by-a-name", "getattr-named-by-a-sum", "vars", "dunder-dict", "getattr-unread-name",
        "getattr-on-os-path"])
def test_a_delete_reached_by_name_or_unbound_is_held(code, expected, snap):
    assert _files(code, (snap,)) == expected


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


@pytest.mark.parametrize("write, overwritten", [
    ("open('report.pdf', 'w').write('gone')", "report.pdf"),
    ("Path('summary.md').write_text('')", "summary.md"),
])
def test_an_overwrite_beside_a_held_delete_is_on_the_card(write, overwritten, snap):
    card = _card(f"import os\nfrom pathlib import Path\nos.remove('stock.csv')\n{write}", snap)
    assert "This step would delete:" in card and "This step would also overwrite:" in card
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


def test_the_refusal_names_the_first_paths_and_the_card_names_them_all():
    paths = tuple(f"many/f_{i:05d}.csv" for i in range(25))
    writes = CellWrites(files=(FileChange(action="delete", paths=paths, call="os.remove(f)", line=3),))
    refusal = _refusal(writes, "cancelled").content
    assert "`many/f_00019.csv` and 5 more." in refusal and "f_00020" not in refusal
    assert _fenced(_card_prompt(writes)) == ["\n".join(paths)]


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
