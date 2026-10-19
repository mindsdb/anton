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

from anton.core.utils.scratchpad import _card_prompt
from anton.core.utils.write_check import (
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
        if fence is None and re.fullmatch(r"`{3,}", line):
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
])
def test_reads_and_prose(text):
    assert write_statements(text) == ()


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
], ids=["literal", "multi-line", "f-string", "concat", "named", "psycopg2-sql", "sqlalchemy-text",
        "pad-helper", "to_sql", "list-loop", "duckdb-attach", "motherduck", "motherduck-sqlalchemy",
        "attach-in-parens", "truncate-f-string-schema", "bigquery-backticks", "plus-equals",
        "join-literals", "join-appended-list", "or-default", "class-attribute", "bytes",
        "format-verb-argument", "percent-whole-statement", "exec-literal", "eval-name", "exec-repr",
        "write-inside-a-concatenation"])
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


def test_each_write_in_one_string_is_its_own_finding(snap):
    assert _statements(PG + 'cur.execute("UPDATE suppliers SET a = 1 WHERE id = 5; DROP TABLE suppliers_audit")', snap) == [
        "UPDATE suppliers SET a = 1 WHERE id = 5", "DROP TABLE suppliers_audit",
    ]
    script = "BEGIN;\nUPDATE suppliers SET a = 2 WHERE id = 5;\nDELETE FROM purchase_orders;\nCOMMIT;"
    assert _statements(PG + f"cur.execute('''{script}''')", snap) == [
        "UPDATE suppliers SET a = 2 WHERE id = 5", "DELETE FROM purchase_orders",
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
], ids=["select", "printed", "logged", "written-to-file", "docstring", "unused-name", "duckdb-csv",
        "duckdb-module", "duckdb-alias", "sqlite-memory", "sqlite-file", "sqlalchemy-sqlite",
        "local-helper", "local-loop", "local-to_sql", "prose", "list-comprehension-printed",
        "list-comprehension-saved", "named-list-printed", "appended-then-printed", "json-dump",
        "repr-printed", "paragraph-prose", "paragraph-list-prose", "multi-cell-prose",
        "semicolon-prose", "grant-prose", "insert-prose", "delete-prose", "attach-prose",
        "duckdb-csv-unconnected", "duckdb-sqlalchemy", "markdown-shown", "console-print"])
def test_cells_that_change_no_database(code, snap):
    assert not _held(code, snap)


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

    cache = _EarlierCells(budget=12)
    first = cache.get(code="a = 1", index=0)
    assert cache.get(code="a = 1", index=0) is first
    moved = cache.get(code="a = 1", index=1)
    assert moved is not first and (first.index, moved.index) == (0, 1)
    # 15 characters against a budget of 12: the cell read least recently goes.
    cache.get(code="b = 2", index=2)
    assert cache.get(code="a = 1", index=0) is not first
    assert cache.get(code="def (", index=0) is None, "a cell that doesn't parse is skipped"


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
], ids=["df-rename-columns", "series-rename", "series-replace", "df-rename-callable", "str-replace",
        "temp-renamed-into-place", "tempfile-name", "mkstemp", "tempdir", "outside-project",
        "new-file-in-existing-folder", "new-path-in-existing-folder", "move-into-folder",
        "new-file", "helper-temp-file", "cleanup-new-folder", "new-summary",
        "with-suffix-temp", "tuple-loop-cleanup", "own-list-cleanup", "own-skill-draft",
        "artifact-temp-renamed", "shell-rm-new-files", "other-commands", "rglob-own-files"])
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
], ids=["if-else", "write-then-delete", "write-text-then-unlink", "write-in-uncalled-function",
        "write-after-return", "write-under-if"])
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
], ids=["def", "keyword", "lambda", "move", "no-params", "transitive", "passed-as-a-value", "new-call-only"])
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
], ids=["helper-temp-file", "helper-on-new-file", "statement-only-printed"])
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
], ids=["append-from-listdir", "plus-equals-glob", "extend", "tuple", "built-in-an-earlier-cell",
        "literal-grown", "set-add", "list-plus-list"])
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
], ids=["named-command", "named-words", "built-up-command", "f-string", "plus", "format", "percent",
        "f-string-root", "sh-c", "bash-lc", "getoutput", "getstatusoutput", "mv-t", "mv-target-directory",
        "sudo", "xargs", "find-delete", "find-exec-rm", "git-clean", "spawnlp", "shell-variable",
        "if-then", "for-do", "helper-command", "exec-code"])
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
], ids=["getattr-os", "getattr-shutil", "getattr-value", "dunder-import", "Path.unlink-unbound",
        "Path.rename-unbound", "Path.unlink-loop", "map-Path.unlink", "Path.rmdir-unbound"])
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


def test_a_path_in_another_case_is_held_where_the_filesystem_ignores_case(snap):
    if not snap.casefold:
        pytest.skip("this filesystem tells case apart")
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
    assert "````\na```b.txt\n````" in _card("import os\nos.remove('a```b.txt')", snapshot), (
        "the fence is longer than any backtick run in the name"
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


def test_past_the_cap_a_file_the_turn_wrote_is_new(tmp_path):
    for i in range(30):
        (tmp_path / f"f{i}.csv").write_text("x")
    taken_at = time.time()
    capped = snapshot_project_files(root=tmp_path, label="project", taken_at=taken_at, cap=10)
    os.utime(tmp_path / "f0.csv", (taken_at - 60, taken_at - 60))
    (tmp_path / "new.csv").write_text("x")
    os.utime(tmp_path / "new.csv", (taken_at + 5, taken_at + 5))
    assert not capped.complete
    assert capped.holds(rel="f0.csv")
    assert not capped.holds(rel="new.csv"), "the turn's own file gets no card"


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
