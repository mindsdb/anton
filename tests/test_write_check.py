"""What `find_cell_writes` holds, and what it lets through.

A missed write runs with no approval. A false hold costs an unattended answer
the full card timeout and then the answer itself. So each rule here has cells
it must hold and cells, often the same call shape, it must not.
"""

from __future__ import annotations

import ast
import errno
import logging
import os
import re
import shlex
import textwrap
import time
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from anton.core.backends.base import Cell, CompactedCell
from anton.core.backends.wire import NOTHING_KEPT
from anton.core.utils import scratchpad
from anton.core.utils.scratchpad import _card_prompt, _code_span, _fence, _refusal
from anton.core.utils.write_check import (
    CallWrite,
    CellWrites,
    EarlierCell,
    FileChange,
    HttpWrite,
    HttpWriteRule,
    ProjectSnapshot,
    UncheckedCause,
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


def _earlier(
    codes: Iterable[str], *, errored: Sequence[bool] = (), stopped_at: Sequence[int | None] = (),
    ended: Sequence[bool] = (),
) -> list[EarlierCell]:
    """The earlier cells `codes` give, oldest first: each that `errored`
    marks stopped on an error, at the line `stopped_at` gives beside it,
    and each that `ended` marks one whose process ended. A list left empty
    says each cell ran to its end in a process that lives on."""
    codes = list(codes)
    errored = list(errored) or [False] * len(codes)
    stopped_at = list(stopped_at) or [None] * len(codes)
    ended = list(ended) or [False] * len(codes)
    return [EarlierCell(code=code, errored=failed, stopped_at=line, ended_process=gone)
            for code, failed, line, gone in zip(codes, errored, stopped_at, ended, strict=True)]


def _held(code: str, snap, *, earlier=()) -> bool:
    return find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,)).needs_approval()


def _parses(code: str) -> bool:
    """Whether this Python's parser reads `code`; 3.11 and 3.12.0 give up on very long expressions."""
    try:
        ast.parse(code)
    except RecursionError:
        return False
    return True


def _statements(code: str, snap, *, earlier=()) -> list[str]:
    return [w.statement for w in find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,)).sql]


def _card(code: str, snap, *, earlier=()) -> str:
    return _card_prompt(find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,)))


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


class _EarlierRow(EarlierCell):
    """An earlier cell a fixture row gives as a mapping, for one that stopped
    on an error or at a line. `extra="forbid"` so a typo'd key fails the run."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class _UnreadCase(BaseModel):
    """One row of the unread-code fixture. `extra="forbid"` so a typo'd key
    (`kind:` for `kinds:`) fails the run instead of being ignored, and `card`
    is one of the three the test knows. Each `earlier` entry is a cell's
    code, for one that ran to its end, or an earlier cell with its outcome,
    and `earlier_cells` reads them as the earlier cells `find_cell_writes`
    takes."""

    model_config = ConfigDict(extra="forbid")

    name: str
    code: str
    card: Literal["unchecked", "read", "none"]
    earlier: list[str | _EarlierRow] = []
    files: dict[str, str] = {}
    empty_project: bool = False
    kinds: list[str] = []
    not_kinds: list[str] = []

    def earlier_cells(self) -> list[EarlierCell]:
        """The row's earlier cells, oldest first."""
        return [EarlierCell(code=cell) if isinstance(cell, str) else cell for cell in self.earlier]


_UNREAD_CODE = yaml.safe_load(
    (Path(__file__).parent / "fixtures" / "write_check_unread_code.yaml").read_text())
_UNREAD_CODE_RULE = _UNREAD_CODE["rule"]
_UNREAD_CODE_CASES = [_UnreadCase.model_validate(case) for case in _UNREAD_CODE["cases"]]


@pytest.mark.parametrize("case", _UNREAD_CODE_CASES, ids=[c.name for c in _UNREAD_CODE_CASES])
def test_cells_that_run_code_the_check_cant_read(case: "_UnreadCase", tmp_path, monkeypatch):
    """Each cell runs code the check can or can't read; the card follows the
    rule the fixture names. A missed run would go out with no card, so an
    unread source holds the cell, while a literal keeps its read."""
    if not case.empty_project:
        (tmp_path / "stock.csv").write_text("x")
    for rel, text in case.files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    monkeypatch.chdir(tmp_path)
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    writes = find_cell_writes(
        code=case.code, earlier=case.earlier_cells(), snapshots=(snap,),
        http_rules=parse_http_write_rules(_UNREAD_CODE_RULE))
    if case.card == "none":
        assert not writes.needs_approval()
    elif case.card == "unchecked":
        assert writes.unchecked
        assert "This step can't be checked before it runs." in _card_prompt(writes)
    else:
        assert writes.needs_approval() and not writes.unchecked
    for kind in case.kinds:
        assert kind in writes.kinds()
    for kind in case.not_kinds:
        assert kind not in writes.kinds()


def _unchecked_card(code: str, earlier: Sequence[str] = ()) -> str:
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(),
                              http_rules=parse_http_write_rules(_UNREAD_CODE_RULE))
    assert writes.unchecked
    return _card_prompt(writes)


def test_the_card_names_what_a_step_uses_and_the_earlier_line_that_made_it():
    card = _unchecked_card("print(orders.mean())", earlier=(
        "SCRIPT = open('job.txt').read()", "exec(SCRIPT)", "orders = load_orders()",
    ))
    assert ("It uses `orders`, which code an earlier step ran may have set. The check couldn't read that code:"
            "\n\n```\nexec(SCRIPT)\n```") in card


def test_the_card_quotes_the_line_that_runs_code_the_check_cant_read():
    card = _unchecked_card("import runpy\nrunpy.run_path('job.py')")
    assert "It runs code the check can't read:\n\n```\nrunpy.run_path('job.py')\n```" in card
    # Inside literal code a step hands exec(), the step's own line is quoted.
    card = _unchecked_card('exec("import runpy\\nrunpy.run_path(\'job.py\')")')
    assert "```\nexec(\"import runpy\\nrunpy.run_path('job.py')\")\n```" in card


def test_a_step_too_deep_to_parse_says_so():
    writes = find_cell_writes(code="-" * 200_000 + "1", earlier=(), snapshots=())
    assert writes.unchecked and [c.kind for c in writes.causes] == ["limit"]


def test_a_call_chain_past_the_depth_limit_quotes_the_function_it_stopped_at():
    defs = "\n".join(f"def f{i}():\n    return f{i + 1}()" for i in range(10))
    defs += "\ndef f10():\n    import os\n    os.remove('a.csv')"
    writes = find_cell_writes(code="f0()", earlier=_earlier([defs]), snapshots=())
    assert writes.unchecked and [c.kind for c in writes.causes] == ["limit"]
    assert writes.causes[0].code == "def f4():\n    return f5()"


def _chain(body: str) -> str:
    """An earlier cell whose f0() calls on to f4(), five calls deep from the
    cell that calls f0(), so the read stops at f4, which runs `body`."""
    return "\n".join(f"def f{i}():\n    return f{i + 1}()" for i in range(4)) + f"\ndef f4():\n{body}"


def test_a_call_chain_past_the_depth_limit_quotes_where_the_read_stopped():
    earlier = _chain("    import requests\n    requests.post('https://erp.example.com/orders', json={'qty': 5})")
    writes = find_cell_writes(code="f0()", earlier=_earlier([earlier]), snapshots=(),
                              http_rules=parse_http_write_rules("POST https://erp.example.com/**"))
    assert writes.stopped and not writes.unread_here
    card = _card_prompt(writes, "f0()")
    # The limit's own line is the gate's (`test_write_approval_gate.py`).
    assert (f"{scratchpad._WHOLE_STEP_CAUSES['limit']}\n\nThe check stopped at:\n\n"
            "```\ndef f4():\n    import requests\n    requests.post(") in card


def test_where_the_read_stopped_is_quoted_scrubbed_and_cut():
    body = ("    token = 'sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789'\n"
            + "".join(f"    step_{i} = {'x' * 80!r}\n" for i in range(8))
            + "    import requests\n    requests.delete('https://erp.example.com/orders/1')")
    writes = find_cell_writes(code="f0()", earlier=_earlier([_chain(body)]), snapshots=())
    quoted = writes.causes[0].code
    assert writes.causes[0].kind == "limit" and quoted.startswith("def f4():\n    token = ")
    assert "sk-ant-api03" not in quoted
    assert len(quoted.split("\n")) <= 6 and len(quoted) <= 400 and quoted.endswith("...")


def test_a_quoted_line_that_runs_literal_code_is_cut():
    """A reason inside literal code quotes the line of the cell that runs
    it, cut like any quote (`_cut`)."""
    host = f"exec({'exec(SRC)  # ' + 'x' * 600!r})"
    writes = find_cell_writes(code=f"SRC = open('job.txt').read()\n{host}", earlier=(), snapshots=())
    quoted = [c.code for c in writes.causes if c.kind == "runs"]
    assert quoted and all(len(q) <= 400 and q.endswith("...") for q in quoted)


def test_a_reason_of_a_kind_the_card_has_no_line_for_is_refused():
    with pytest.raises(ValidationError):
        UncheckedCause(kind="other")


def test_where_the_read_stopped_is_cut_to_400_characters_in_a_short_function():
    body = "    import requests\n    requests.delete('https://erp.example.com/orders/' + '" + "x" * 600 + "')"
    writes = find_cell_writes(code="f0()", earlier=_earlier([_chain(body)]), snapshots=())
    quoted = writes.causes[0].code
    assert writes.causes[0].kind == "limit" and quoted.startswith("def f4():")
    assert len(quoted.split("\n")) == 3 and len(quoted) == 400 and quoted.endswith("...")


def test_an_error_building_the_reasons_keeps_that_the_read_stopped(monkeypatch):
    from anton.core.utils import write_check

    def boom(*args, **kwargs):
        raise RuntimeError("reasons")

    monkeypatch.setattr(write_check, "_stopped_at", boom)
    delete = "    import requests\n    requests.delete('https://erp.example.com/orders/1')"
    writes = find_cell_writes(code="f0()", earlier=_earlier([_chain(delete)]), snapshots=())
    assert writes.unchecked and writes.stopped and writes.causes == ()


def test_a_cell_too_deep_to_parse_is_a_stopped_read():
    writes = find_cell_writes(code="-" * 200_000 + "1", earlier=(), snapshots=())
    assert writes.unchecked and writes.stopped
    assert [c.kind for c in writes.causes] == list(writes.cause_kinds) == ["limit"]


# A rename the check reads in literal code, eight literal code texts, a def
# that renames, and a command nested `n` shells deep (`sh -c` inside `sh -c`).
_RENAME = "import os\nos.rename('report.pdf', 'old.pdf')"
_EIGHT_EXECS = "\n".join(["exec('y = 2')"] * 8)
_DEFINES_PUSH = "import os\ndef push():\n    os.rename('report.pdf', 'old.pdf')"
# Literal code too deep for the check's parser: the parser runs out of stack
# on each, and a pad on another Python may still run it.
_TOO_DEEP = "1" + "+1" * 20_000
_TOO_NESTED = "-" * 20_000 + "1"
# Literal code longer than the check reads as one more cell.
_TOO_LONG = "\n".join(f"x{n} = {n}" for n in range(12_000))


def _chain_to_push(n: int) -> str:
    """Functions f0 to f`n`, each calling the next, the last running
    `_DEFINES_PUSH` in the cell namespace: a call of f0 reaches it `n`
    calls down."""
    calls = "\n".join(f"def f{i}():\n    return f{i + 1}()" for i in range(n))
    return f"{calls}\ndef f{n}():\n    exec({_DEFINES_PUSH!r}, globals())"


def _in_shells(command: str, n: int) -> str:
    for _ in range(n):
        command = "sh -c " + shlex.quote(command)
    return command


@pytest.mark.parametrize("earlier, code, quoted", [
    ([], f"{_EIGHT_EXECS}\nexec({_RENAME!r})", f"exec({_RENAME!r})"),
    ([], "\n".join(["exec('y = 2')"] * 7) + f"\nexec({f'exec({_RENAME!r})'!r})", f"exec({_RENAME!r})"),
    ([], "\n".join(["eval('1 + 1')"] * 8) + f"\nexec({_RENAME!r})", f"exec({_RENAME!r})"),
    ([], "import timeit\n" + "\n".join(["timeit.timeit('y = 2', number=1)"] * 8)
     + f"\ntimeit.timeit({_RENAME!r}, number=1)", f"timeit.timeit({_RENAME!r}, number=1)"),
    ([], f"import subprocess, sys\n{_EIGHT_EXECS}\nsubprocess.run([sys.executable, '-c', {_RENAME!r}])",
     "subprocess.run([sys.executable, '-c', "),
    ([], f"exec({'#' * 100_001 + chr(10) + _RENAME!r})", 'exec("###'),
    (["def job():\n    " + f"{_EIGHT_EXECS}\nexec({_RENAME!r})".replace("\n", "\n    ")], "job()",
     f"exec({_RENAME!r})"),
    ([], f"import os\nos.system({_in_shells('mv report.pdf old.pdf', 5)!r})", "os.system('sh -c "),
    ([], f"import subprocess\nsubprocess.run({['sh', '-c', _in_shells('mv report.pdf old.pdf', 4)]!r})",
     "subprocess.run(['sh', '-c', "),
    (["import os\ndef run(cmd):\n    os.system(cmd)"], f"run({_in_shells('mv report.pdf old.pdf', 5)!r})",
     "os.system(cmd)"),
    ([], "import os, threading\nthreading.Thread(target=os.system, args="
         f"({_in_shells('mv report.pdf old.pdf', 5)!r},)).start()", "threading.Thread(target=os.system"),
    ([], f"import os\nlist(map(os.system, [{_in_shells('mv report.pdf old.pdf', 5)!r}]))", "list(map(os.system"),
    ([], f"import os\nos.system({'echo a ' + chr(92) + chr(10) + _in_shells('mv report.pdf old.pdf', 5)!r})",
     "os.system("),
], ids=["ninth-exec", "eighth-exec-hands-on-a-ninth", "eight-evals-then-exec", "ninth-timeit", "python-c-after-eight",
        "literal-past-100000-characters", "ninth-exec-in-an-earlier-function", "five-shells-deep",
        "five-shells-deep-as-a-list", "five-shells-deep-through-a-parameter", "five-shells-deep-as-a-thread-target",
        "five-shells-deep-through-map", "five-shells-deep-after-a-backslash-newline"])
def test_code_left_unread_at_a_limit_of_the_check_is_a_stopped_read(earlier, code, quoted, snap):
    """Literal code past `_EXEC_LIMIT` texts or `_LITERAL_CODE_LIMIT`
    characters, and a script past `_SHELL_DEPTH` shells, are left unread at a
    limit of the check, not code it can't read: the read stops there
    (`stopped`), so no value of the setting runs the cell with no card, and
    the card quotes the call it stopped at. An earlier cell that holds such
    code stops this one too, which says so first."""
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    assert writes.unchecked and writes.stopped and not writes.unread_here
    own = [c for c in writes.causes if not c.in_earlier]
    assert [c.kind for c in own] == ["limit"] and own[0].code.startswith(quoted)
    assert writes.cause_kinds == ("limit",)


@pytest.mark.parametrize("earlier, code", [
    ([], "\n".join(["exec('y = 2')"] * 7) + f"\nexec({_RENAME!r})"),
    ([], f"exec({_RENAME + chr(10) + '#' * (100_000 - len(_RENAME) - 1)!r})"),
    ([], f"import os\nos.system({_in_shells('rm report.pdf', 4)!r})"),
    (["\n".join(["exec('y = 2')"] * 7) + f"\nexec({_DEFINES_PUSH!r})"], "push()"),
    ([_chain_to_push(3), "f0()"], "push()"),
], ids=["eighth-exec", "literal-of-100000-characters", "four-shells-deep", "call-into-the-eighth-exec",
        "call-into-what-a-call-chain-within-the-depth-limit-ran"])
def test_code_within_the_limits_of_the_check_is_read(earlier, code, snap):
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    assert [f.paths for f in writes.held_files] == [("report.pdf",)]
    assert not writes.unchecked


def test_a_call_with_a_literal_past_the_length_the_check_reads_quotes_its_first_line(snap):
    """The literal's own first lines say nothing of what it does: the card
    quotes the call's first line alone."""
    filler = "".join(f"# row {n} kept for the audit trail\n" for n in range(4_000))
    writes = find_cell_writes(code=f'exec("""\n{filler}{_RENAME}\n""")', earlier=(), snapshots=(snap,))
    assert writes.stopped and [(c.kind, c.code) for c in writes.causes] == [("limit", 'exec("""...')]


# Functions f0 to f5, each calling the one before it, f0 renaming
# report.pdf: a call of f5 reaches the rename five calls down.
_CHAIN_TO_RENAME = "import os\ndef f0():\n    os.rename('report.pdf', 'old.pdf')\n" + "".join(
    f"def f{i}():\n    f{i - 1}()\n" for i in range(1, 6))
# Eight literal execs and a ninth the check leaves unread, as a function body too.
_NINTH = f"{_EIGHT_EXECS}\nexec({_RENAME!r})"
_INDENTED_NINTH = textwrap.indent(_NINTH, "    ")
# Four literal execs, and a function body of four more and a fifth that
# renames: a cell that runs both runs nine.
_FOUR_EXECS = "\n".join(["exec('y = 2')"] * 4)
_FIVE_EXECS_BODY = textwrap.indent(f"{_FOUR_EXECS}\nexec({_RENAME!r})", "    ")
# Names a0 to a11, each the one before it, a0 a function that deletes stock.csv.
_ALIASES_OF_F = "a0 = f\n" + "".join(f"a{i} = a{i - 1}\n" for i in range(1, 12))
# Nine names, each another name for the one before, the first a URL: more
# than the command reading follows, so a command built on p9 is read past
# the depth limit (`_shell_commands`).
_NINE_TO_A_URL = "p0 = 'https://api.example.com/items/1'\n" + "".join(f"p{i} = p{i - 1}\n" for i in range(1, 10))
_DEEP_DELETE = "os.system('curl -X DELETE ' + p9)"
# Partials of partials of a delete of stock.csv, seven deep.
_PARTIALS_TO_DELETE = ["import subprocess\nfrom functools import partial\np0 = partial(subprocess.run, ['rm', 'stock.csv'])",
                       *(f"p{i} = partial(p{i - 1})" for i in range(1, 7))]
# The earlier cells that hold code a limit of the check leaves unread, and
# the code the card quotes where the read of them stopped, if any.
_LEFT_UNREAD = {
    "ninth-exec": ([f"{_EIGHT_EXECS}\nexec({_RENAME!r})"], f"exec({_RENAME!r})"),
    "literal-past-100000-characters": ([f"exec({'#' * 100_001 + chr(10) + _RENAME!r})"], 'exec("###'),
    "literal-too-deep-to-parse": ([f"exec({_TOO_DEEP!r})"], "exec('1+1+1"),
    "five-shells-deep": ([f"import os\nos.system({_in_shells('mv report.pdf old.pdf', 5)!r})"], "os.system('sh -c "),
    "call-chain-past-the-depth-limit": ([_CHAIN_TO_RENAME, "f5()"], "def f1():\n    f0()"),
    "partials-past-the-depth-limit": ([*_PARTIALS_TO_DELETE, "p6()"], "p6()"),
    "partials-past-the-depth-limit-in-a-function": ([*_PARTIALS_TO_DELETE, "def go():\n    p6()", "go()"], "p6()"),
    "cell-too-deep-to-parse": ([_TOO_DEEP], ""),
    # A shell command past the depth limit on its words that does more than
    # one plain read, delete or move (`_does_something`).
    "a-shell-command-past-the-depth-limit": ([f"{_NINE_TO_A_URL}import os\n{_DEEP_DELETE}"], _DEEP_DELETE),
    "a-shell-command-past-the-depth-limit-in-a-function-nothing-calls": (
        [f"{_NINE_TO_A_URL}import os\ndef later():\n    {_DEEP_DELETE}"], _DEEP_DELETE),
    "a-shell-command-past-the-depth-limit-in-a-property": (
        [f"{_NINE_TO_A_URL}import subprocess\nclass Job:\n    @property\n    def done(self):\n"
         "        subprocess.run(['kubectl', 'delete', 'pod', p9])"], "subprocess.run(['kubectl', 'delete', 'pod', p9])"),
    "a-shell-command-past-the-depth-limit-in-a-function-an-earlier-cell-defined": (
        [f"import os\ndef go():\n    {_DEEP_DELETE}", f"{_NINE_TO_A_URL}go()"], _DEEP_DELETE),
    "a-shell-command-past-the-depth-limit-through-a-parameter": (
        ["import os\ndef go(url):\n    os.system('curl -X DELETE ' + url)", f"{_NINE_TO_A_URL}go(p9)"],
        "os.system('curl -X DELETE ' + url)"),
    # Anywhere in the cell's code, whether that code runs or not: in a
    # function, method, lambda, decorator or default it defines, or one
    # nothing calls.
    "ninth-exec-in-a-function-nothing-calls": ([f"def later():\n{_INDENTED_NINTH}"], f"exec({_RENAME!r})"),
    "five-shells-deep-in-a-function-nothing-calls": (
        [f"import os\ndef later():\n    os.system({_in_shells('mv report.pdf old.pdf', 5)!r})"], "os.system('sh -c "),
    "ninth-exec-in-a-function-the-cell-calls": ([f"def go():\n{_INDENTED_NINTH}\ngo()"], f"exec({_RENAME!r})"),
    "ninth-exec-in-a-function-an-earlier-cell-defined": ([f"def go():\n{_INDENTED_NINTH}", "go()"], f"exec({_RENAME!r})"),
    "ninth-exec-in-a-method-the-cell-calls": (
        [f"class Job:\n    def run(self):\n{textwrap.indent(_NINTH, ' ' * 8)}\nJob().run()"], f"exec({_RENAME!r})"),
    "ninth-exec-in-a-thread-target": (
        [f"import threading\ndef go():\n{_INDENTED_NINTH}\nthreading.Thread(target=go).start()"], f"exec({_RENAME!r})"),
    "ninth-exec-in-a-function-handed-to-map": ([f"def go(_):\n{_INDENTED_NINTH}\nlist(map(go, [1]))"],
                                               f"exec({_RENAME!r})"),
    "ninth-exec-in-a-decorator": ([f"def deco(f):\n{_INDENTED_NINTH}\n    return f\n@deco\ndef g():\n    pass"],
                                  f"exec({_RENAME!r})"),
    "ninth-exec-in-a-decorator-an-earlier-cell-defined": (
        [f"def deco(f):\n{_INDENTED_NINTH}\n    return f", "@deco\ndef g():\n    pass"], f"exec({_RENAME!r})"),
    "ninth-exec-in-a-default": (["def f(x=exec('y = 2')):\n    pass\n" + _EIGHT_EXECS], "def f(x=exec('y = 2')):"),
    "ninth-exec-in-a-lambda-called-at-once": ([f"{_EIGHT_EXECS}\n(lambda: exec({_RENAME!r}))()"], "(lambda: exec("),
    "five-shells-deep-in-a-function-the-cell-calls": (
        [f"import os\ndef go():\n    os.system({_in_shells('mv report.pdf old.pdf', 5)!r})\ngo()"], "os.system('sh -c "),
    "five-shells-deep-in-a-function-an-earlier-cell-defined": (
        [f"import os\ndef g():\n    os.system({_in_shells('mv report.pdf old.pdf', 5)!r})", "g()"], "os.system('sh -c "),
    "call-chain-from-a-function-the-cell-calls": ([_CHAIN_TO_RENAME, "def run_all():\n    f5()\nrun_all()"],
                                                  "def f1():\n    f0()"),
    "call-chain-under-a-main-guard": (
        [_CHAIN_TO_RENAME, "def main():\n    f5()\nif __name__ == '__main__':\n    main()"], "def f1():\n    f0()"),
    "call-chain-in-a-function-nothing-calls": ([_CHAIN_TO_RENAME, "def go():\n    f5()"], "def f1():\n    f0()"),
    "call-chain-in-a-function-an-older-cells-function-calls-back": (
        [_CHAIN_TO_RENAME + "def run():\n    helper()", "def helper():\n    f5()\nrun()"], "def f1():\n    f0()"),
    "ninth-exec-in-a-function-an-older-cells-function-calls-back": (
        ["def run():\n    helper()", f"{_EIGHT_EXECS}\ndef helper():\n    exec({_RENAME!r})\nrun()"], f"exec({_RENAME!r})"),
    "call-chain-in-a-method-an-older-cells-function-calls-back": (
        [_CHAIN_TO_RENAME + "def run(job):\n    job.go()", "class Job:\n    def go(self):\n        f5()\nrun(Job())"],
        "def f1():\n    f0()"),
    # Functions reached other ways than by a plain call of a def.
    "call-chain-in-a-function-literal-code-defines": (
        [_CHAIN_TO_RENAME, "import types\nm = types.ModuleType('m')\nm.f4 = f4\nexec('def g():\\n    f4()', m.__dict__)"],
        "def f0():"),
    "recursion-past-the-limit-of-reads-again": (
        ["import os\ndef r(n):\n    if n == 3:\n        r(2)\n    if n == 2:\n        r(1)\n    if n == 1:\n"
         "        os.remove('stock.csv')", "r(3)"], "def r(n):"),
    "a-function-found-past-the-depth-limit-of-its-names": (
        ["import os\ndef f():\n    os.remove('stock.csv')", _ALIASES_OF_F, "a11()"], "def f():"),
    "a-function-found-past-the-depth-limit-of-its-names-in-a-function": (
        ["import os\ndef f():\n    os.remove('stock.csv')", _ALIASES_OF_F, "def h():\n    a11()", "h()"], "def f():"),
    # Stops only the calling cell's code reaches, in functions earlier cells
    # defined: a function defined before what it calls counts at the call.
    "a-function-found-past-the-depth-limit-of-its-names-after-the-function-that-calls-it": (
        ["def h():\n    a11()", "import os\ndef f():\n    os.remove('stock.csv')", _ALIASES_OF_F, "h()"], "def f():"),
    "call-chain-past-the-depth-limit-after-the-function-that-calls-it": (
        ["def run():\n    f5()", _CHAIN_TO_RENAME, "run()"], "def f2():\n    f1()"),
    "ninth-exec-across-a-function-an-earlier-cell-defined": (
        [f"def go():\n{_FIVE_EXECS_BODY}", f"{_FOUR_EXECS}\ngo()"], f"exec({_RENAME!r})"),
    "ninth-exec-across-two-functions-earlier-cells-defined": (
        [f"def inner():\n{_FIVE_EXECS_BODY}", "def go():\n    inner()", f"{_FOUR_EXECS}\ngo()"], f"exec({_RENAME!r})"),
    # Literal code a cell runs.
    "call-chain-past-the-depth-limit-in-literal-code": ([_CHAIN_TO_RENAME, "exec('f5()')"], "def f1():\n    f0()"),
    "partials-past-the-depth-limit-in-literal-code": ([*_PARTIALS_TO_DELETE, "exec('p6()')"], "p6()"),
    "ninth-exec-inside-literal-code": ([f"exec({_NINTH!r})"], "exec('y = 2')"),
}


@pytest.mark.parametrize("later", ["print(1)", "import json", "import os\nos.remove('summary.md')"],
                         ids=["read-only", "import", "write"])
@pytest.mark.parametrize("left", list(_LEFT_UNREAD), ids=list(_LEFT_UNREAD))
def test_a_later_cell_after_code_a_limit_left_unread_is_a_stopped_read(left, later, snap):
    """Once an earlier cell holds code a limit of the check left unread
    anywhere in its code, whether that code ran or not, a later cell may
    depend on what that code did, whatever it does itself: the read stops
    there too, under every value of the setting, with the card quoting
    where the read of that earlier cell stopped. What the cell itself does
    stays on the card."""
    earlier, quoted = _LEFT_UNREAD[left]
    writes = find_cell_writes(code=later, earlier=_earlier(earlier), snapshots=(snap,))
    assert writes.unchecked and writes.stopped and not writes.unread_here
    assert writes.cause_kinds == ("limit",)
    assert [(c.kind, c.in_earlier) for c in writes.causes] == [("limit", True)]
    assert writes.causes[0].code.startswith(quoted) and bool(writes.causes[0].code) is bool(quoted)
    assert [(f.action, f.paths) for f in writes.held_files] == ([("delete", ("summary.md",))] if "remove" in later else [])


def test_a_call_into_what_code_the_check_cant_read_defined_is_still_a_carried_hold():
    """Code the check can't read by nature, as exec() of a file's text, isn't
    a limit of the check: a call into what it defined stays a `uses` hold,
    which `own` may drop, and later cells don't stop."""
    writes = find_cell_writes(code="push()", snapshots=(),
                              earlier=_earlier(["SRC = open('job.txt').read()", "exec(SRC)"]))
    assert writes.unchecked and not writes.stopped and not writes.unread_here
    assert writes.cause_kinds == ("uses",)


@pytest.mark.parametrize("left", list(_LEFT_UNREAD), ids=list(_LEFT_UNREAD))
def test_a_reset_of_the_pad_clears_code_a_limit_left_unread(left, snap):
    """A reset clears the cells the check reads: a cell after it reads only
    the cells after the reset."""
    earlier, _ = _LEFT_UNREAD[left]
    assert find_cell_writes(code="print(1)", earlier=_earlier([*earlier, "x = 1"]), snapshots=(snap,)).stopped
    assert not find_cell_writes(code="print(1)", earlier=_earlier(["x = 1"]), snapshots=(snap,)).unchecked


@pytest.mark.parametrize("earlier", [
    [_CHAIN_TO_RENAME + "f5()"],
    [_CHAIN_TO_RENAME + "def run():\n    helper()", "def helper():\n    print('kept')", "run()",
     "def helper():\n    print('kept again')"],
    [_CHAIN_TO_RENAME + "def run():\n    helper()", "run()", "def helper():\n    f3()"],
    [_CHAIN_TO_RENAME + "def run():\n    helper()", "def helper():\n    f3()\nrun()"],
    [f"def go():\n{_FIVE_EXECS_BODY}"],
    ["def h():\n    a11()", "import os\ndef f():\n    os.remove('stock.csv')", _ALIASES_OF_F],
], ids=["call-chain-of-the-cells-own-functions", "a-harmless-helper-redefined-by-a-later-cell",
        "a-helper-only-a-later-cell-defines", "a-helper-of-its-own-an-older-function-calls-back",
        "five-execs-in-a-function-nothing-runs-beside-more", "a-function-past-the-depth-limit-of-its-names-nothing-calls"])
def test_an_earlier_cell_that_holds_no_code_a_limit_left_unread_stops_no_later_cell(earlier, snap):
    """A chain of a cell's own functions is no chain of earlier ones, so the
    cell's own read follows none of it, and nor does a helper that an
    older cell's function calls back into, as the cell's own check reads
    it: the helper's own code is read where the cell defines it, within
    the depth limit. An earlier cell's read follows no function a cell
    after it defines, so a function that names one before it exists holds
    no stop until a cell calls it. Five literal execs in a function are
    within the limit of the cell that defines them."""
    writes = find_cell_writes(code="print(1)", earlier=_earlier(earlier), snapshots=(snap,))
    assert not writes.unchecked


@pytest.mark.parametrize("parsed_kept", [True, False], ids=["parsed-cells-kept", "parsed-cells-not-kept"])
@pytest.mark.parametrize("ended", [False, True], ids=["ran", "its-process-ended-as-the-next-cell-came"])
def test_a_conversation_checked_cell_by_cell_reads_no_earlier_cell_again_for_a_stop(ended, parsed_kept, snap,
                                                                                     monkeypatch):
    """The gate checks each cell before it runs, and that check keeps
    whether the cell holds code a limit of the check left unread, by its
    place in the conversation (`_LEFT_UNREAD`), so no later check reads an
    earlier cell again for that (`_left_unread`), even where a busy process
    keeps no parsed cell (`_EarlierCells`). The gate may mark the cell
    before as one whose process ended as the next one comes, which no later
    check's cells say: that changes nothing the check keeps."""
    from anton.core.utils import write_check

    if not parsed_kept:
        monkeypatch.setattr(write_check, "_EARLIER_CELLS", write_check._EarlierCells(budget=0))
    reread: list[str] = []
    left_unread = write_check._left_unread
    monkeypatch.setattr(write_check, "_left_unread",
                        lambda *args: reread.append(args[-1].code) or left_unread(*args))
    cells = [f"x = 1  # kept {ended} {parsed_kept}", _CHAIN_TO_RENAME, "f5()", "def f4():\n    pass", "print(2)"]
    for i, code in enumerate(cells):
        before = _earlier(cells[:i])
        if ended and before:
            before = [*before[:-1], before[-1].model_copy(update={"ended_process": True})]
        find_cell_writes(code=code, earlier=before, snapshots=(snap,))
    writes = find_cell_writes(code="print(1)", earlier=_earlier(cells), snapshots=(snap,))
    assert reread == []
    assert [(c.kind, c.in_earlier) for c in writes.causes] == [("limit", True)]
    assert writes.causes[0].code.startswith("def f1():")


def test_the_cell_after_one_that_ran_a_chain_reads_it_without_its_own_redefinition(snap):
    """An earlier cell read here (`_left_unread`) is read with the earlier
    cells alone: this cell redefining a link of the chain it ran changes
    nothing in what it ran."""
    writes = find_cell_writes(code="def f4():\n    pass", earlier=_earlier([_CHAIN_TO_RENAME, "f5()  # read here"]),
                              snapshots=(snap,))
    assert [(c.kind, c.in_earlier) for c in writes.causes] == [("limit", True)]


# A command five shells deep that moves report.pdf.
_FIVE_SHELLS = _in_shells("mv report.pdf old.pdf", 5)
# Earlier cells a check reads where no check in this process kept what they
# hold, as after a restart of the process, and the code the card quotes. In
# most, a cell after the one that holds the stop rebinds a name its read
# reads, which that cell's own check didn't see.
_READ_AS_ITS_OWN_CHECK = {
    "a-link-of-the-chain-redefined-later": ([_CHAIN_TO_RENAME, "f5()", "def f4():\n    pass"], "def f1():\n    f0()"),
    "the-link-past-the-limit-redefined-later": ([_CHAIN_TO_RENAME, "f5()", "def f1():\n    pass"], "def f1():\n    f0()"),
    "a-link-under-a-wrapper-redefined-later": (
        ["import os\ndef f5():\n    os.remove('stock.csv')\n" + "".join(f"def f{i}():\n    f{i + 1}()\n" for i in range(4, 0, -1))
         + "def go():\n    f1()", "go()", "def f3():\n    pass"], "def f4():\n    f5()"),
    "a-command-a-later-cell-rebinds": (
        ["import os\ndef run():\n    os.system(CMD)", f"CMD = {_FIVE_SHELLS!r}", "run()", "CMD = 'ls'"], "os.system(CMD)"),
    "a-command-past-the-depth-limit-on-names-a-later-cell-binds": (
        [f"import os\ndef go():\n    {_DEEP_DELETE}", _NINE_TO_A_URL, "go()"], _DEEP_DELETE),
    "a-class-a-later-cell-defines-again": (
        ["class Job:\n    def go(self):\n        f5()", _CHAIN_TO_RENAME, "Job().go()", "class Job:\n    def go(self):\n        pass"],
        "def f2():\n    f1()"),
    "partials-built-after-the-function-that-calls-them": (["def go():\n    p6()", *_PARTIALS_TO_DELETE, "go()"], "p6()"),
    "call-chain-through-a-namespace-literal-code-filled": (
        [_CHAIN_TO_RENAME, "import types\nm = types.ModuleType('m')\nm.f2 = f2\nexec('def h():\\n    f2()', m.__dict__)",
         "def g():\n    m.h()", "g()"], "def f0():"),
    "an-import-alias-a-later-cell-imports-again": (
        [f"import os as o\no.system({_FIVE_SHELLS!r})", "import json as o"], "o.system("),
    "a-module-alias-a-later-cell-imports-under-its-name": (
        ["import os", "o = os", f"o.system({_FIVE_SHELLS!r})", "import json as o"], "o.system("),
    "a-command-literal-code-binds-that-a-later-cell-rebinds": (
        ["import os\ndef run():\n    os.system(CMD)", f"exec({'CMD = ' + repr(_FIVE_SHELLS)!r})", "run()", "CMD = 'ls'"],
        "os.system(CMD)"),
    "a-chain-literal-code-defines-then-a-link-redefined": (
        [f"exec({_CHAIN_TO_RENAME!r})", "f5()", "def f4():\n    pass"], "def f1():"),
    "a-link-later-literal-code-redefines": ([_CHAIN_TO_RENAME, "f5()", "exec('def f4():\\n    pass')"], "def f1():"),
    "partials-a-cell-builds-and-calls-then-a-link-redefined": (
        [_CHAIN_TO_RENAME, *_PARTIALS_TO_DELETE[:6], "p6 = partial(p5)\np6()", "def f4():\n    pass"], "p6()"),
    "partials-literal-code-builds-in-the-cell-that-calls-them-then-a-link-redefined": (
        [*_PARTIALS_TO_DELETE[:6], "exec('p6 = partial(p5)')\np6()", "def p5():\n    pass"], "p6()"),
}


@pytest.mark.parametrize("left", list(_READ_AS_ITS_OWN_CHECK), ids=list(_READ_AS_ITS_OWN_CHECK))
@pytest.mark.parametrize("kept", [True, False], ids=["checked-cell-by-cell", "read-after-a-restart"])
def test_an_earlier_cell_no_check_kept_a_stop_for_is_read_as_its_own_check_read_it(left, kept, snap, monkeypatch):
    """The gate checks each cell before it runs and keeps what it found
    (`_LEFT_UNREAD`). A cell no check in this process kept that for, as
    after a restart of the process, is read as its own check read it
    (`_earlier_stop`): with the cells up to it alone where a later cell
    rebinds a name its read reads (`_read_apart`), so a later
    redefinition, rebinding or new class hides no stop, and every later
    cell stops all the same."""
    from anton.core.utils import write_check

    monkeypatch.setattr(write_check, "_LEFT_UNREAD", write_check._LeftUnread(limit=100_000))
    cells, quoted = _READ_AS_ITS_OWN_CHECK[left]
    if kept:
        for i, code in enumerate(cells):
            find_cell_writes(code=code, earlier=_earlier(cells[:i]), snapshots=(snap,))
    writes = find_cell_writes(code="print(1)", earlier=_earlier(cells), snapshots=(snap,))
    assert [(c.kind, c.in_earlier) for c in writes.causes] == [("limit", True)]
    assert writes.causes[0].code.startswith(quoted)


def test_an_earlier_cell_read_after_a_restart_takes_no_stop_from_a_later_cells_call(snap, monkeypatch):
    """After a restart, an earlier cell that only defines a function is read
    as its own check read it: a later cell's call that hands the function a
    command's word past the depth limit stops that later cell, not the one
    that defines the function, so a conversation that starts with the same
    cell stops nothing on its account."""
    from anton.core.utils import write_check

    monkeypatch.setattr(write_check, "_LEFT_UNREAD", write_check._LeftUnread(limit=100_000))
    define = "import os\ndef go(url):\n    os.system('curl -X DELETE ' + url)"
    writes = find_cell_writes(code="print(1)", earlier=_earlier([define, f"{_NINE_TO_A_URL}go(p9)"]), snapshots=(snap,))
    assert [(c.kind, c.in_earlier) for c in writes.causes] == [("limit", True)]
    assert not find_cell_writes(code="print(1)", earlier=_earlier([define]), snapshots=(snap,)).stopped


@pytest.mark.parametrize("stopped_at", [2, None], ids=["failed-before-the-redefinition", "failed-where-the-record-cant-tell"])
def test_an_earlier_cell_read_after_a_restart_reads_a_redefinition_that_may_not_have_run_as_its_check_did(stopped_at, snap,
                                                                                                          monkeypatch):
    """A cell that failed before it redefined a link of the chain, or where
    its record can't tell, may have left the old link in place: the read
    of the cell that calls the chain, after a restart, follows it, as that
    cell's own check did (`_Reader.replaced`)."""
    from anton.core.utils import write_check

    monkeypatch.setattr(write_check, "_LEFT_UNREAD", write_check._LeftUnread(limit=100_000))
    earlier = _earlier([_CHAIN_TO_RENAME, "x = 1\nraise ValueError\ndef f4():\n    pass", "f5()"],
                       errored=(False, True, False), stopped_at=(None, stopped_at, None))
    writes = find_cell_writes(code="print(1)", earlier=earlier, snapshots=(snap,))
    assert [(c.kind, c.in_earlier) for c in writes.causes] == [("limit", True)]
    assert writes.causes[0].code.startswith("def f1():")


def test_conversations_that_share_earlier_cells_keep_each_its_own_verdict(snap, monkeypatch):
    """A server process checks many conversations, and what each check
    found is kept by its place (`_LEFT_UNREAD`): the cell's code after the
    code and record of every cell before it. A conversation whose earlier
    cell failed keeps its own verdict beside one where the same code ran;
    one cell's code beside another's at the same index keeps its own; and
    a read after a restart of one conversation, with a later redefinition,
    keeps what another conversation's first cells hold."""
    from anton.core.utils import write_check

    def checked(cells: list[EarlierCell]) -> list[EarlierCell]:
        for i in range(len(cells)):
            find_cell_writes(code=cells[i].code, earlier=cells[:i], snapshots=(snap,))
        return cells

    def stops(cells: list[EarlierCell]) -> bool:
        return find_cell_writes(code="print(1)", earlier=cells, snapshots=(snap,)).stopped

    # The same code: in A the pandas cell failed, so only the reading of
    # errored cells holds the chain's delete (`_added_holds`); in B it ran.
    codes = ["P = 'stock.csv'",
             "import pandas as pd\nfresh = pd.read_csv('missing.csv')\nP = 'scratch.tmp'\nfresh.to_csv(P, index=False)",
             "import os\ndef f0():\n    os.remove(P)\n" + "".join(f"def f{i}():\n    f{i - 1}()\n" for i in range(1, 5)),
             "f4()"]
    monkeypatch.setattr(write_check, "_LEFT_UNREAD", write_check._LeftUnread(limit=100_000))
    a = checked(_earlier(codes, errored=(False, True, False, False), stopped_at=(None, 2, None, None)))
    b = checked(_earlier(codes))
    assert (stops(a), stops(b)) == (True, False)
    # The same index, another code, in either order.
    for first, second in (([_CHAIN_TO_RENAME, "x = 1"], [_CHAIN_TO_RENAME, "f5()"]),
                          ([_CHAIN_TO_RENAME, "f5()"], [_CHAIN_TO_RENAME, "x = 1"])):
        monkeypatch.setattr(write_check, "_LEFT_UNREAD", write_check._LeftUnread(limit=100_000))
        done = [checked(_earlier(first)), checked(_earlier(second))]
        assert [stops(cells) for cells in done] == ["f5()" in first, "f5()" in second]
    # Two reads after a restart: the one with the redefinition comes first.
    monkeypatch.setattr(write_check, "_LEFT_UNREAD", write_check._LeftUnread(limit=100_000))
    redefined = _earlier([_CHAIN_TO_RENAME, "f5()", "def f4():\n    pass", "y = 1"])
    kept = _earlier([_CHAIN_TO_RENAME, "f5()", "w = 3", "y = 1"])
    assert (stops(redefined), stops(kept)) == (True, True)


# Five functions, each calling the one before it, the first deleting
# report.pdf: a call of f4() reaches the delete only past `_CALL_DEPTH`,
# where the check holds it only on a file the turn began with.
_CHAIN4_TO_DELETE = "import os\ndef f0():\n    os.remove('report.pdf')\n" + "".join(
    f"def f{i}():\n    f{i - 1}()\n" for i in range(1, 5))


def _project(root: Path, *files: str):
    """A snapshot of `root`, made to hold `files`."""
    for rel in files:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x")
    return snapshot_project_files(root=root, label="project", taken_at=time.time() + 60)


def test_a_stop_one_conversation_kept_stays_where_another_reads_the_same_cells_against_another_project(
        tmp_path, monkeypatch):
    """Two conversations with the same cells keep their verdicts by the same
    place (`_LEFT_UNREAD`). One whose project held report.pdf stops at the
    chain's delete; one whose project didn't keeps nothing for it, and a
    stop once kept stays (`_LeftUnread.keep`): the first conversation's
    later cells still stop, in either order."""
    from anton.core.utils import write_check

    with_file = _project(tmp_path / "a", "report.pdf")
    without = _project(tmp_path / "b", "stock.csv")
    cells = _earlier([_CHAIN4_TO_DELETE, "f4()"])
    for order in ((with_file, without), (without, with_file)):
        monkeypatch.setattr(write_check, "_LEFT_UNREAD", write_check._LeftUnread(limit=100_000))
        for project in order:
            monkeypatch.chdir(project.root)
            for i in range(len(cells)):
                find_cell_writes(code=cells[i].code, earlier=cells[:i], snapshots=(project,))
        monkeypatch.chdir(with_file.root)
        writes = find_cell_writes(code="print(1)", earlier=cells, snapshots=(with_file,))
        assert [(c.kind, c.in_earlier) for c in writes.causes] == [("limit", True)]


@pytest.mark.parametrize("never_there", [False, True], ids=["the-file-it-deleted", "a-file-the-project-never-held"])
def test_a_cell_read_after_a_restart_counts_every_path_as_one_the_project_held(never_there, tmp_path, monkeypatch):
    """After a restart the check can't tell what the project held when an
    earlier cell ran: the chain's delete of report.pdf ran, so a later
    turn's project lacks it. The read of that cell counts every path as one
    the turn began with (`ProjectSnapshot.failed`), so its stop past the
    depth limit holds every later cell still, as it did before the
    restart. So does a delete of a file no turn held, which the cell's own
    check would not have held: after a restart the check can't tell the
    two apart."""
    from anton.core.utils import write_check

    turn_one = _project(tmp_path, "report.pdf", "stock.csv")
    cells = _earlier([_CHAIN4_TO_DELETE.replace("report.pdf", "scratch.tmp") if never_there else _CHAIN4_TO_DELETE,
                      "f4()"])
    monkeypatch.chdir(tmp_path)
    for i in range(len(cells)):
        find_cell_writes(code=cells[i].code, earlier=cells[:i], snapshots=(turn_one,))
    (tmp_path / "report.pdf").unlink()
    turn_two = _project(tmp_path)
    monkeypatch.setattr(write_check, "_LEFT_UNREAD", write_check._LeftUnread(limit=100_000))
    writes = find_cell_writes(code="print(1)", earlier=cells, snapshots=(turn_two,))
    assert [(c.kind, c.in_earlier) for c in writes.causes] == [("limit", True)]
    assert writes.causes[0].code.startswith("def f0():")


_CHAIN4_OVERWRITES_STOCK = "def f0():\n    open('stock.csv', 'w').write('y')\n" + "".join(
    f"def f{i}():\n    f{i - 1}()\n" for i in range(1, 5))


@pytest.mark.parametrize("later", ["open('stock.csv', 'w').write('x')",
                                   "import pandas as pd\npd.DataFrame().to_csv('stock.csv')"],
                         ids=["open", "to-csv"])
def test_an_earlier_cell_read_after_a_restart_reads_the_files_the_agent_owned_before_it(later, snap, monkeypatch):
    """A cell read after a restart is read as its own check read it
    (`_left_unread`): a file a cell after it rewrote was not yet the agent's
    when it ran, so an overwrite of it past the call depth still stops it,
    and every later cell."""
    from anton.core.utils import write_check

    monkeypatch.setattr(write_check, "_LEFT_UNREAD", write_check._LeftUnread(limit=100_000))
    writes = find_cell_writes(code="print(1)", earlier=_earlier([_CHAIN4_OVERWRITES_STOCK, "f4()", later]),
                              snapshots=(snap,))
    assert [(c.kind, c.in_earlier) for c in writes.causes] == [("limit", True)]


def test_a_cell_read_after_a_restart_past_the_snapshots_cap_counts_a_file_the_turn_changed_as_one_held(
        tmp_path, monkeypatch):
    """Past its cap a snapshot doesn't name stock.csv, so the check reads
    the file on disk: unchanged since the turn began when the chain's cell
    was checked, so its overwrite past the depth limit stopped there. Once
    that overwrite ran, the file changed this turn, and what the pad's
    cells write would call it new (`_Reader.made`). The read of that cell
    after a restart reads no such record: every file on disk counts as one
    the project held when the cell ran, so every later cell still stops."""
    from anton.core.utils import write_check

    (tmp_path / "stock.csv").write_text("x")
    os.utime(tmp_path / "stock.csv", (time.time() - 120, time.time() - 120))
    monkeypatch.chdir(tmp_path)
    capped = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() - 60).model_copy(
        update={"files": frozenset(), "spellings": {}, "complete": False})
    earlier = _earlier([_CHAIN4_OVERWRITES_STOCK, "f4()"])
    assert find_cell_writes(code="f4()", earlier=earlier[:1], snapshots=(capped,)).stopped
    (tmp_path / "stock.csv").write_text("y")
    monkeypatch.setattr(write_check, "_LEFT_UNREAD", write_check._LeftUnread(limit=100_000))
    writes = find_cell_writes(code="print(1)", earlier=earlier, snapshots=(capped,))
    assert [(c.kind, c.in_earlier) for c in writes.causes] == [("limit", True)]


@pytest.mark.parametrize("cells", [
    [_CHAIN4_OVERWRITES_STOCK, "open('stock.csv', 'w').write('x')\nf4()"],
    [_CHAIN4_OVERWRITES_STOCK, "f4()\nopen('stock.csv', 'w').write('x')"],
    [_CHAIN4_OVERWRITES_STOCK, "open('stock.csv', 'w').write('x')\nraise ValueError", "f4()"],
], ids=["a-rewrite-in-the-cell-before-the-call", "a-rewrite-in-the-cell-after-the-call", "a-rewrite-in-a-cell-that-failed"])
def test_a_cell_read_after_a_restart_owns_no_file_it_or_a_failed_cell_wrote(cells, snap, monkeypatch):
    """Read after a restart, a cell owns only the files the cells before it
    rewrote and ran (`_OwnedFiles` of the cells before it, errored ones
    making none the agent's), as its own check did before it ran, so the
    overwrite of stock.csv past `_CALL_DEPTH` still stops it."""
    from anton.core.utils import write_check

    monkeypatch.setattr(write_check, "_LEFT_UNREAD", write_check._LeftUnread(limit=100_000))
    errored = tuple(code.endswith("raise ValueError") for code in cells)
    stopped_at = tuple(2 if failed else None for failed in errored)
    writes = find_cell_writes(code="print(1)", earlier=_earlier(cells, errored=errored, stopped_at=stopped_at),
                              snapshots=(snap,))
    assert [(c.kind, c.in_earlier) for c in writes.causes] == [("limit", True)]


# Six helpers, each calling the next, the last renaming report.pdf, as defs
# and as lambdas.
_PIPELINE = ("import os\ndef archive():\n    os.rename('report.pdf', 'old.pdf')\ndef publish():\n    archive()\n"
             "def render():\n    publish()\ndef build():\n    render()\ndef load():\n    build()\ndef main():\n    load()")
_PIPELINE_OF_LAMBDAS = ("import os\narchive = lambda: os.rename('report.pdf', 'old.pdf')\npublish = lambda: archive()\n"
                        "render = lambda: publish()\nbuild = lambda: render()\nload = lambda: build()\nmain = lambda: load()")
# Six functions, each calling the one before it, the first doing nothing.
_HARMLESS_CHAIN = "def f0():\n    pass\n" + "".join(f"def f{i}():\n    f{i - 1}()\n" for i in range(1, 6))


@pytest.mark.parametrize("earlier", [
    _earlier([_PIPELINE, f"import missing_module\n{_PIPELINE}"], errored=(False, True), stopped_at=(None, 1)),
    _earlier([_CHAIN_TO_RENAME, "import missing_module\ndef f5():\n    pass\ndef go():\n    f5()"],
             errored=(False, True), stopped_at=(None, 1)),
    _earlier([_PIPELINE_OF_LAMBDAS, _PIPELINE]),
    _earlier([_HARMLESS_CHAIN, "import os\ndef f0():\n    os.rename('report.pdf', 'old.pdf')\nf5()"]),
], ids=["helpers-defined-again-in-a-cell-that-failed-on-its-first-line",
        "the-top-of-a-chain-defined-again-in-a-cell-that-failed-on-its-first-line", "lambdas-defined-again-as-defs",
        "a-function-a-chain-past-the-depth-limit-reaches-defined-again-in-the-calling-cell"])
@pytest.mark.parametrize("kept", [True, False], ids=["checked-cell-by-cell", "read-after-a-restart"])
def test_an_earlier_cell_that_defines_again_what_an_older_one_defined_is_read_as_its_own_check_read_it(
        earlier, kept, snap, monkeypatch):
    """A cell's own check read it before it ran, with no record of its own,
    so a definition in it replaced the one before it (`_Reader.replaced`);
    it read each function an earlier cell defined at the first call it
    met, so a longer chain to it was read no further
    (`_CellRead.read_function`); and it followed no function the cell
    itself defined, past the depth limit too (`_CellRead.follows`). Read
    after a restart, the cell finds the same: none of these holds a stop."""
    from anton.core.utils import write_check

    monkeypatch.setattr(write_check, "_LEFT_UNREAD", write_check._LeftUnread(limit=100_000))
    if kept:
        for i in range(len(earlier)):
            find_cell_writes(code=earlier[i].code, earlier=earlier[:i], snapshots=(snap,))
    assert not find_cell_writes(code="print(1)", earlier=earlier, snapshots=(snap,)).stopped


def test_the_card_quotes_the_first_earlier_cell_that_holds_code_a_limit_left_unread(snap):
    earlier = _earlier([*_LEFT_UNREAD["ninth-exec"][0], *_LEFT_UNREAD["five-shells-deep"][0]])
    writes = find_cell_writes(code="print(1)", earlier=earlier, snapshots=(snap,))
    assert [(c.kind, c.code, c.in_earlier) for c in writes.causes] == [("limit", f"exec({_RENAME!r})", True)]
    # An earlier cell too deep to parse that came first has nothing to quote.
    writes = find_cell_writes(code="print(1)", earlier=_earlier([_TOO_DEEP, *_LEFT_UNREAD["ninth-exec"][0]]),
                              snapshots=(snap,))
    assert [(c.kind, c.code, c.in_earlier) for c in writes.causes] == [("limit", "", True)]


def test_a_cell_that_stops_itself_after_code_a_limit_left_unread_gives_both_stops(snap):
    writes = find_cell_writes(code=f"{_EIGHT_EXECS}\nexec({_RENAME!r})", earlier=_earlier(_LEFT_UNREAD["ninth-exec"][0]),
                              snapshots=(snap,))
    assert [(c.kind, c.code, c.in_earlier) for c in writes.causes] == [
        ("limit", f"exec({_RENAME!r})", True), ("limit", f"exec({_RENAME!r})", False)]
    assert writes.cause_kinds == ("limit",)


def test_a_cell_too_deep_to_parse_after_code_a_limit_left_unread_gives_both_stops(snap):
    """The check's parser gives up on the cell itself, and an earlier
    cell's stop still comes first on its card (`_earlier_stop`)."""
    writes = find_cell_writes(code="-" * 200_000 + "1", earlier=_earlier(_LEFT_UNREAD["ninth-exec"][0]),
                              snapshots=(snap,))
    assert [(c.kind, c.code, c.in_earlier) for c in writes.causes] == [
        ("limit", f"exec({_RENAME!r})", True), ("limit", "", False)]
    assert writes.stopped and writes.cause_kinds == ("limit",)


# Cells that reach the chain past the depth limit other than by a plain call
# of a def at their top level.
_REACHED_OTHERWISE = {
    "a-lambda-handed-to-apply": "import pandas as pd\npd.DataFrame({'a': [1]}).apply(lambda r: f5(), axis=1)",
    "a-lambda-handed-to-an-executor": "from concurrent.futures import ThreadPoolExecutor\n"
                                      "with ThreadPoolExecutor() as ex:\n    list(ex.map(lambda x: f5(), [1]))",
    "a-bound-method-as-a-thread-target": "import threading\nclass W:\n    def run(self):\n        f5()\n"
                                         "threading.Thread(target=W().run).start()",
    "a-loop-over-functions": "for step in (f5,):\n    step()",
    "a-list-of-steps": "cbs = [f5]\nfor cb in cbs:\n    cb()",
    "a-dict-of-steps": "steps = {'a': f5}\nfor name, step in steps.items():\n    step()",
    "a-comprehension": "[step() for step in [f5]]",
    "a-with-whose-enter-calls-it": "class K:\n    def __enter__(self):\n        f5()\n    def __exit__(self, *a):\n"
                                   "        pass\nwith K():\n    pass",
    "a-property": "class P:\n    @property\n    def x(self):\n        return f5()\nP().x",
    "dunder-call": "f5.__call__()",
    "a-sort-key": "sorted([1, 2], key=lambda x: f5())",
    "a-reduce": "import functools\nfunctools.reduce(lambda a, b: f5(), [1, 2, 3])",
    "a-default-argument": "def f(a=f5()):\n    pass",
    "a-callback-into-a-lambda": "def run(cb):\n    cb()\nrun(lambda: f5())",
    "an-override-a-base-method-calls": "class Base:\n    def go(self):\n        self.step()\nclass Mine(Base):\n"
                                       "    def step(self):\n        f5()\nMine().go()",
}


@pytest.mark.parametrize("cell", list(_REACHED_OTHERWISE.values()), ids=list(_REACHED_OTHERWISE))
def test_a_chain_past_the_depth_limit_anywhere_in_an_earlier_cell_stops_a_later_cell(cell, snap):
    """The check reads all of an earlier cell's code for a stop, as that
    cell's own check read it, so however the cell reaches the chain, every
    later cell stops."""
    writes = find_cell_writes(code="print(1)", earlier=_earlier([_CHAIN_TO_RENAME, cell]), snapshots=(snap,))
    assert [(c.kind, c.in_earlier) for c in writes.causes] == [("limit", True)]
    assert writes.causes[0].code.startswith("def f1():")


_NINTH_THEN_FAILS = f"{_EIGHT_EXECS}\nexec({_RENAME!r})\nx = 1\ny = 2\nraise ValueError"
_LONG_THEN_FAILS = f"exec({'#' * 100_001 + chr(10) + _RENAME!r})\nraise ValueError"


@pytest.mark.parametrize("before, cell, stopped_at, ended, stops", [
    ([], _NINTH_THEN_FAILS, 9, False, True), ([], _NINTH_THEN_FAILS, 12, False, True),
    ([], _NINTH_THEN_FAILS, None, False, True), ([], _NINTH_THEN_FAILS, 8, False, True),
    ([], _NINTH_THEN_FAILS, NOTHING_KEPT, True, False), ([], _NINTH_THEN_FAILS, NOTHING_KEPT, False, True),
    ([], _LONG_THEN_FAILS, NOTHING_KEPT, True, False), ([], _LONG_THEN_FAILS, NOTHING_KEPT, False, True),
    ([], _LONG_THEN_FAILS, 2, False, True),
    ([_CHAIN_TO_RENAME], "f5()\nraise ValueError", 2, False, True),
    ([_CHAIN_TO_RENAME], "f5(); raise ValueError('late')", NOTHING_KEPT, False, True),
    ([_CHAIN_TO_RENAME], "x = 1\nraise ValueError\nf5()", 2, False, True),
    ([_CHAIN_TO_RENAME], "raise ValueError\nf5()", NOTHING_KEPT, True, False),
    ([_CHAIN_TO_RENAME], "raise ValueError\nf5()", NOTHING_KEPT, False, True),
    ([], f"def go():\n{_INDENTED_NINTH}\nx = 1\nraise ValueError\ngo()", 12, False, True),
    ([*_PARTIALS_TO_DELETE, "def go():\n    p6()"], "x = 1\nraise ValueError\ngo()", 2, False, True),
    ([], _TOO_DEEP, 5, False, True), ([], _TOO_DEEP, None, False, True), ([], _TOO_DEEP, NOTHING_KEPT, False, True),
    ([], _TOO_DEEP, NOTHING_KEPT, True, False),
    ([], _NINTH_THEN_FAILS, 12, True, True), ([], f"{_NINTH}\nreturn", None, False, False),
], ids=["failed-at-the-call", "failed-after-the-call", "failed-where-the-record-cant-tell", "failed-before-the-call",
        "kept-nothing", "failed-on-its-first-line", "kept-nothing-by-a-long-literal",
        "failed-in-the-call-of-a-long-literal-on-its-first-line", "failed-after-a-call-of-a-long-literal",
        "call-chain-then-failed", "call-chain-then-failed-on-its-first-line", "call-chain-after-the-failure",
        "call-chain-after-a-failure-that-kept-nothing", "call-chain-after-a-failure-on-its-first-line",
        "ninth-exec-in-a-function-the-cell-calls-after-the-failure", "partials-after-the-failure",
        "too-deep-failed-past-its-first-line", "too-deep-failed-with-no-line", "too-deep-failed-on-its-first-line",
        "too-deep-kept-nothing", "failed-past-its-first-line-then-its-process-ended", "python-refused-to-compile-it"])
def test_an_earlier_cell_that_failed_stops_a_later_cell_unless_it_kept_nothing(before, cell, stopped_at, ended, stops,
                                                                                snap):
    """Whether an earlier cell's code ran doesn't matter: a cell that failed
    holds the code a limit left unread wherever it failed, before or after
    it. Only a record that says the cell kept nothing (`NOTHING_KEPT`, its
    process gone: `ended_process`) stops no later cell, though such a cell
    may have run some of its code; and so does a cell Python refused to
    compile, which ran none of it. A cell that failed on its first line in
    a process that lives on gives the same line, but ran that line, as a
    one-line call that does its work and then raises does; and a process
    that ended after a cell failed past its first line ran that far."""
    earlier = _earlier([*before, cell], errored=(*[False] * len(before), True),
                       stopped_at=(*[None] * len(before), stopped_at), ended=(*[False] * len(before), ended))
    assert find_cell_writes(code="print(1)", earlier=earlier, snapshots=(snap,)).stopped is stops


def test_an_earlier_cell_that_didnt_fail_ran_its_code_whatever_line_its_record_gives(snap):
    """`stopped_at` counts only for a cell that failed: a record that didn't
    fail ran its code, even one that gives `NOTHING_KEPT` with its process
    gone, so its literal code past the limit stops a later cell."""
    earlier = _earlier([_NINTH], stopped_at=[NOTHING_KEPT], ended=[True])
    assert find_cell_writes(code="print(1)", earlier=earlier, snapshots=(snap,)).stopped


@pytest.mark.parametrize("last, stops", [
    (("f4()", False, None, False), True),
    (("f4()\nraise ValueError", True, 2, False), True),
    (("x = 1\nraise ValueError\nf4()", True, 2, False), True),
    (("raise ValueError\nf4()", True, NOTHING_KEPT, True), False),
    (("raise ValueError\nf4()", True, NOTHING_KEPT, False), True),
], ids=["ran-to-its-end", "failed-after-the-call", "failed-before-the-call", "kept-nothing", "failed-on-its-first-line"])
def test_a_stop_only_the_reading_of_errored_cells_finds_stops_a_later_cell(last, stops, snap):
    """A function past `_CALL_DEPTH` that deletes a file only the reading of
    errored cells as maybe stopped anywhere places, as P before the cell
    that failed rebinds it, stops the cell that calls it, and so every later
    cell (`_earlier_stop`), unless that cell's record says it kept nothing."""
    code, errored, stopped_at, ended = last
    earlier = _earlier([
        "P = 'stock.csv'",
        "import pandas as pd\nfresh = pd.read_csv('missing.csv')\nP = 'scratch.tmp'\nfresh.to_csv(P, index=False)",
        "import os\ndef f0():\n    os.remove(P)\n" + "".join(f"def f{i}():\n    f{i - 1}()\n" for i in range(1, 5)),
        code,
    ], errored=(False, True, False, errored), stopped_at=(None, 2, None, stopped_at), ended=(False, False, False, ended))
    writes = find_cell_writes(code="print(1)", earlier=earlier, snapshots=(snap,))
    assert writes.stopped is stops
    if stops:
        assert [(c.kind, c.in_earlier) for c in writes.causes] == [("limit", True)]
        assert writes.causes[0].code.startswith("def f0():")


@pytest.mark.parametrize("write", ["os.remove(P)", "open(P, 'w').write('y')"], ids=["delete", "overwrite"])
def test_a_stop_only_the_reading_of_errored_cells_finds_stops_a_later_cell_after_a_restart(write, snap, monkeypatch):
    """Read after a restart, the cell that calls the chain finds its stop
    past `_CALL_DEPTH` only on the reading of errored cells as maybe stopped
    anywhere, which places P on stock.csv (`_left_unread`'s `wide`), for an
    overwrite as for a delete."""
    from anton.core.utils import write_check

    monkeypatch.setattr(write_check, "_LEFT_UNREAD", write_check._LeftUnread(limit=100_000))
    earlier = _earlier([
        "P = 'stock.csv'",
        "import pandas as pd\nfresh = pd.read_csv('missing.csv')\nP = 'scratch.tmp'\nfresh.to_csv(P, index=False)",
        f"import os\ndef f0():\n    {write}\n" + "".join(f"def f{i}():\n    f{i - 1}()\n" for i in range(1, 5)),
        "f4()",
    ], errored=(False, True, False, False), stopped_at=(None, 2, None, None))
    writes = find_cell_writes(code="print(1)", earlier=earlier, snapshots=(snap,))
    assert [(c.kind, c.in_earlier) for c in writes.causes] == [("limit", True)]


def test_a_call_through_partials_of_partials_past_the_depth_limit_is_a_stopped_read(snap):
    """`partials_of` reads a partial of a partial `_PARAM_DEPTH` deep: past
    that, the delete it hides stops the read."""
    partials = ["import subprocess\nfrom functools import partial\np0 = partial(subprocess.run, ['rm', 'stock.csv'])",
                *(f"p{i} = partial(p{i - 1})" for i in range(1, 7))]
    within = find_cell_writes(code="p5()", earlier=_earlier(partials[:6]), snapshots=(snap,))
    assert [(f.action, f.paths) for f in within.held_files] == [("delete", ("stock.csv",))]
    assert not within.unchecked
    past = find_cell_writes(code="p6()", earlier=_earlier(partials), snapshots=(snap,))
    assert past.unchecked and past.stopped and not past.unread_here
    assert [(c.kind, c.code) for c in past.causes] == [("limit", "p6()")]
    through = find_cell_writes(code="def go():\n    p6()\ngo()", earlier=_earlier(partials), snapshots=(snap,))
    assert [(c.kind, c.code) for c in through.causes] == [("limit", "p6()")]


@pytest.mark.parametrize("empty", [True, False], ids=["a-project-empty-as-the-turn-began", "no-snapshot"])
def test_a_call_through_partials_past_the_depth_limit_stops_whatever_the_snapshots_hold(empty, tmp_path):
    """The check stops at such a call wherever it reads code, not only
    where it reads the files a write may change (`_CellRead.read_stops`):
    a project that was empty as the turn began stops it too, and what its
    check keeps for its place stops each later cell, in this turn and in a
    later one that began with files."""
    (tmp_path / "empty").mkdir()
    snapshots = (snapshot_project_files(root=tmp_path / "empty", label="project", taken_at=time.time() + 60),) if empty else ()
    own = find_cell_writes(code="p6()", earlier=_earlier(_PARTIALS_TO_DELETE), snapshots=snapshots)
    assert own.stopped and [(c.kind, c.code) for c in own.causes] == [("limit", "p6()")]
    (tmp_path / "stock.csv").write_text("x")
    files = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    for later in (snapshots, (files,)):
        writes = find_cell_writes(code="print(1)", earlier=_earlier([*_PARTIALS_TO_DELETE, "p6()"]), snapshots=later)
        assert [(c.kind, c.code, c.in_earlier) for c in writes.causes] == [("limit", "p6()", True)]


def test_literal_code_past_the_length_the_check_reads_is_never_parsed(snap, monkeypatch):
    """The check parses no literal code past `_LITERAL_CODE_LIMIT`
    characters, whether a cell or a function runs it: it stops at the call
    that runs it instead (`_ExecRead.capped`, `_Cell.rebinds_globally`)."""
    from anton.core.utils import write_check

    lengths: list[int] = []
    parse_ran = write_check._parse_ran
    monkeypatch.setattr(write_check, "_parse_ran", lambda text: lengths.append(len(text)) or parse_ran(text))
    long = "\n".join(f"TARGET = 'r{n}.csv'" for n in range(8_000))
    earlier = _earlier(["TARGET = 'stock.csv'", f"def load():\n    exec({long!r})\n# {len(long)} never parsed"])
    writes = find_cell_writes(code="import os\nos.remove(TARGET)", earlier=earlier, snapshots=(snap,))
    assert len(long) > 100_000 and max(lengths, default=0) <= 100_000
    assert [(c.kind, c.in_earlier) for c in writes.causes] == [("limit", True)]


# A path four levels deep, built one level a cell, and its shallow twin.
_DAILY = ["from pathlib import Path\nimport pandas as pd\nPROJECT = Path('.')", "DATA = PROJECT / 'data'",
          "OUT = DATA / 'out'", "DAILY = OUT / 'daily'"]
_FLAT = "Path('data/out/daily') / "


@pytest.fixture
def daily(tmp_path: Path, monkeypatch):
    """A project with data/out/daily/stock.csv and data/out/daily/v1/stock.csv."""
    (tmp_path / "data" / "out" / "daily" / "v1").mkdir(parents=True)
    (tmp_path / "data" / "out" / "daily" / "stock.csv").write_text("x")
    (tmp_path / "data" / "out" / "daily" / "v1" / "stock.csv").write_text("x")
    (tmp_path / "stock.csv").write_text("x")
    monkeypatch.chdir(tmp_path)
    return snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)


@pytest.mark.parametrize("earlier, code", [
    (_DAILY, "pd.DataFrame().to_csv(DAILY / 'stock.csv')"),
    (["import os\nimport pandas as pd\nDATA = os.path.join('.', 'data')", "OUT = os.path.join(DATA, 'out')",
      "DAILY = os.path.join(OUT, 'daily')", "V1 = os.path.join(DAILY, 'v1')"],
     "pd.DataFrame().to_csv(os.path.join(V1, 'stock.csv'))"),
    (["p0 = 'data/out/daily/stock.csv'", *(f"p{i} = p{i - 1}" for i in range(1, 9))], "open(p8, 'w').write('x')"),
    (_DAILY, "for t in [DAILY / 'a.csv', DAILY / 'stock.csv']:\n    pd.DataFrame().to_csv(t)"),
    (_DAILY, "for t in (DAILY / 'a.csv', DAILY / 'stock.csv'):\n    open(t, 'w').write('x')"),
    (_DAILY, "for t in {DAILY / 'a.csv', DAILY / 'stock.csv'}:\n    pd.DataFrame().to_csv(t)"),
    (_DAILY, "A = DAILY / 'a.csv'\nB = DAILY / 'stock.csv'\nfor t in (A, B):\n    pd.DataFrame().to_csv(t)"),
    ([*_DAILY, "outs = [DAILY / 'a.csv', DAILY / 'stock.csv']"], "for t in outs:\n    pd.DataFrame().to_csv(t)"),
    (_DAILY, "import shutil\nshutil.copy('stock.csv', DAILY / 'v1')"),
], ids=["path-four-levels-deep", "join-four-levels-deep", "nine-names", "a-list", "a-tuple", "a-set",
        "names-in-a-loop", "a-list-an-earlier-cell-built", "copy-into-the-folder"])
def test_an_overwrite_through_a_path_past_the_depth_limit_is_a_stopped_read(earlier, code, daily):
    """A path `_Reader.paths` builds past its depth limit is one it can't
    place, so the overwrite through it would get no card in any mode: the
    file finder reads on past the limit, each node once, a folder two paths
    share among them, and when that holds a file the turn began with, the
    read stops."""
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(daily,))
    assert writes.unchecked and writes.stopped and not writes.unread_here
    # A path past the limit to a new file changes nothing the card holds.
    new = find_cell_writes(code=code.replace("stock.csv", "new.csv"),
                           earlier=_earlier(e.replace("stock.csv", "new.csv") for e in earlier), snapshots=(daily,))
    assert not new.needs_approval()


@pytest.mark.parametrize("earlier", [
    ["p0 = 'stock.csv'\n" + "".join(f"p{i} = p{i - 1}\n" for i in range(1, 71))],
    ["p0 = 'stock.csv'", *(f"p{i} = p{i - 1}" for i in range(1, 71))],
], ids=["seventy-names-in-a-cell", "seventy-names-a-cell"])
def test_an_overwrite_through_a_path_past_even_the_read_on_depth_is_a_stopped_read(earlier, daily):
    """Read on past the limits, a path is still read only `_PAST_DEPTH`
    levels deep: past that the read stops (`_Reader.paths`), and so does
    the check."""
    writes = find_cell_writes(code="open(p70, 'w').write('x')", earlier=_earlier(earlier), snapshots=(daily,))
    assert writes.unchecked and writes.stopped


@pytest.mark.parametrize("earlier, code", [
    ([], "pd.DataFrame().to_csv({p}'stock.csv')"),
    ([], "for t in [{p}'a.csv', {p}'stock.csv']:\n    pd.DataFrame().to_csv(t)"),
    ([], "pd.DataFrame().to_csv({p}'new.csv')"),
    ([], "open({p}'stock.csv', 'a').write('x')"),
    ([], "import shutil\nshutil.copy({p}'stock.csv', 'backup.csv')"),
    (["pd.DataFrame().to_csv({p}'stock.csv')"], "pd.DataFrame().to_csv({p}'stock.csv')"),
    ([], "import subprocess\nsubprocess.run(['cat', str({p}'stock.csv')])"),
    ([], "import subprocess\nprint(subprocess.check_output(['wc', '-l', str({p}'stock.csv')]))"),
    ([], "import subprocess\nsubprocess.run(['rm', str({p}'stock.csv')])"),
    ([], "print(pd.read_csv({p}'stock.csv'))"),
], ids=["overwrite", "overwrite-in-a-loop", "new-file", "append", "copy-source", "rewrite-of-a-file-the-agent-owns",
        "read-only-command", "read-only-command-output", "delete-command", "read"])
def test_a_path_past_the_depth_limit_gets_the_card_its_shallow_twin_gets(earlier, code, daily):
    """Past the depth limit a path gets the decision the same path built
    shallow gets: an append, a copy's source, a rewrite of a file the agent
    owns, a read and a command that changes nothing get no card, and a
    write the twin holds gets one."""
    deep = find_cell_writes(code=code.format(p="DAILY / "), snapshots=(daily,),
                            earlier=_earlier([*_DAILY, *(e.format(p="DAILY / ") for e in earlier)]))
    flat = find_cell_writes(code=code.format(p=_FLAT), snapshots=(daily,),
                            earlier=_earlier([*_DAILY, *(e.format(p=_FLAT) for e in earlier)]))
    assert deep.needs_approval() is flat.needs_approval()


# A path built on past `_PATH_LIMIT` paths: 512 variants, one per choice of
# nine branches.
_VARIANTS = "import random\nP = 'data'\n" + "".join(
    f"if random.random() < .5:\n    P += '/{s}'\n" for s in "abcdefghi") + "P += '/stock.csv'\n"


def _nine_names(value: str) -> str:
    """Nine names, each another name for the one before, the first `value`:
    more than the command reading follows (`_commands`)."""
    return f"p0 = {value!r}\n" + "".join(f"p{i} = p{i - 1}\n" for i in range(1, 10))


@pytest.mark.parametrize("code, value", [
    ("import subprocess\nsubprocess.run(['ls', {p}])", "stock.csv"),
    ("import os\nos.system('cat ' + {p})", "stock.csv"),
    ("import subprocess\nprint(subprocess.check_output(['wc', '-l', {p}]))", "stock.csv"),
    ("import subprocess\nsubprocess.run('head -n 3 ' + {p}, shell=True)", "stock.csv"),
    ("import subprocess\nsubprocess.run(['cat', {p}])", "job.py"),
    ("import os\nos.system('ls -l ' + {p})", "lib/job.py"),
    ("import os\nos.system({p})", "cat job.py"),
    ("import subprocess\nsubprocess.run(['python', '-m', {p}])", "job"),
    ("import subprocess\nsubprocess.run(['cp', 'job.txt', {p}])", "/usr/lib/python3/site-packages/hook.pth"),
    ("import subprocess\nsubprocess.run('python -m ' + {p}, shell=True)", "job"),
    ("import os\nos.system('cp job.txt ' + {p})", "/usr/lib/python3/site-packages/hook.pth"),
], ids=["ls", "cat", "wc-output", "head-through-a-shell", "cat-of-a-python-file", "ls-of-a-python-file-through-a-shell",
        "a-whole-command-that-reads-a-python-file", "a-module-python-runs", "a-copy-onto-a-startup-hook",
        "a-module-python-runs-through-a-shell", "a-copy-onto-a-startup-hook-through-a-shell"])
def test_a_command_word_past_the_depth_limit_gets_the_card_its_shallow_twin_gets(code, value, snap):
    """A command's word read through more names than the command reading
    follows is read on past the limit (`_shell_commands`): one plain
    command that only reads, even a Python file, gets no card, as its twin
    with the word written out gets none, and a later cell gets nothing from
    it. One that runs Python or writes a startup hook is code the check
    stopped reading, a card under every value, as its twin holds, and every
    later cell stops after it."""
    deep = find_cell_writes(code=code.format(p="p9"), earlier=_earlier([_nine_names(value)]), snapshots=(snap,))
    flat = find_cell_writes(code=code.format(p=repr(value)), earlier=_earlier(()), snapshots=(snap,))
    assert deep.needs_approval() is flat.needs_approval()
    assert deep.stopped is flat.needs_approval()
    later = find_cell_writes(code="print(1)", earlier=_earlier([_nine_names(value), code.format(p="p9")]),
                             snapshots=(snap,))
    assert later.needs_approval() is later.stopped is deep.stopped


@pytest.mark.parametrize("value", ["stock.csv", "new.csv"], ids=["a-file-there", "a-new-file"])
@pytest.mark.parametrize("code, twin", [
    ("import subprocess\nsubprocess.run(['rm', {p}])", "import os\nos.remove({p})"),
    ("import os\nos.system('rm -f ' + {p})", "import os\nos.remove({p})"),
    ("import subprocess\nsubprocess.run(['mv', {p}, '/tmp/x.csv'])", "import shutil\nshutil.move({p}, '/tmp/x.csv')"),
    ("import os\nos.system(f'mv {{{p}}} /tmp/x.csv')", "import shutil\nshutil.move({p}, '/tmp/x.csv')"),
], ids=["rm-words", "rm-script", "mv-words", "mv-script"])
def test_a_delete_or_move_whose_operand_is_past_the_depth_limit_holds_what_its_python_twin_holds(
    code, twin, value, snap,
):
    """An `rm` or `mv` whose operand is read through more names than the
    command reading follows holds whole, under every value, as the Python
    delete or move through the same name holds past the path depth limit,
    and neither stops (`_does_something`)."""
    earlier = [_nine_names(value)]
    deep = find_cell_writes(code=code.format(p="p9"), earlier=_earlier(earlier), snapshots=(snap,))
    python = find_cell_writes(code=twin.format(p="p9"), earlier=_earlier(earlier), snapshots=(snap,))
    assert [(c.action, c.paths) for c in deep.held_files] == [(c.action, c.paths) for c in python.held_files] == [
        ("move" if "mv" in code else "delete", ())]
    assert not deep.stopped and not python.stopped
    assert _cards(code.format(p="p9"), snap, earlier=earlier) == [True, True, True]


@pytest.mark.parametrize("command, held", [
    ("rm stock.csv", True), ("rm -f new.csv", False), ("rm -rf docs", True), ("mv stock.csv /tmp/x.csv", True),
    ("mv new.csv /tmp/x.csv", False),
], ids=["rm-a-file-there", "rm-a-new-file", "rm-a-folder-there", "mv-a-file-there", "mv-a-new-file"])
@pytest.mark.parametrize("run", ["import os\nos.system({p})", "import subprocess\nsubprocess.run({p}.split())"],
                         ids=["script", "words"])
def test_a_whole_delete_or_move_command_past_the_depth_limit_gets_its_shallow_twins_decision(command, held, run, snap):
    """An `rm` or `mv` read whole through more names than the command
    reading follows is read on past the limit as a path is
    (`_FileFinder.shell`, `_FileFinder.decided`): the check stops where its
    twin with the command written out holds a file, a card under every
    value, and one of a new file gets no card, as its twin gets none."""
    earlier = [_nine_names(command)]
    assert bool(_files(run.format(p=repr(command)), (snap,))) is held
    deep = find_cell_writes(code=run.format(p="p9"), earlier=_earlier(earlier), snapshots=(snap,))
    assert deep.stopped is held
    assert _cards(run.format(p="p9"), snap, earlier=earlier) == [held] * 3


@pytest.mark.parametrize("code, value, flat", [
    ("import subprocess\nsubprocess.run(['sed', '-i', 's/a/b/', {p}])", "stock.csv", False),
    ("import subprocess\nsubprocess.run(['cp', 'job.txt', {p}])", "stock.csv", False),
    ("import os\nos.system('echo x > ' + {p})", "stock.csv", False),
    ("import os\nos.system('echo x | tee ' + {p})", "stock.csv", False),
    ("import subprocess\nsubprocess.run(['curl', '-X', 'DELETE', {p}])", "https://example.com/items/1", False),
    ("import os\nos.system('cat a.txt > ' + {p})", "new.csv", False),
    ("import os\nos.system('ls ' + {p} + ' | head')", "docs", False),
    ("import os\nos.system({p})", "cat a.txt > b.txt", False),
    ("import os\nos.system({p})", "ls docs | head", False),
    ("import os\nos.system({p})", "cd docs && rm a.pdf", True),
], ids=["sed-i", "cp", "a-redirect", "tee", "a-remote-delete", "cat-into-a-file", "ls-into-head",
        "a-whole-cat-into-a-file", "a-whole-ls-into-head", "a-whole-list-that-deletes"])
def test_a_command_past_the_depth_limit_that_is_more_than_a_plain_read_delete_or_move_stops(code, value, flat, snap):
    """Past the depth limit on a command's words, the read within the limit
    may have missed what the command changes: unless it is one plain
    command that only reads, deletes or moves (`_does_something`), it is
    code the check stopped reading. The cell stops, under every value, and
    the card quotes the call it stopped at, though the twin with the word
    written out may get no card, as a shell write or a remote change gets
    none. Every later cell stops too, with the earlier step's line and the
    same quote."""
    earlier = [_nine_names(value)]
    deep = find_cell_writes(code=code.format(p="p9"), earlier=_earlier(earlier), snapshots=(snap,))
    assert deep.stopped and _cards(code.format(p="p9"), snap, earlier=earlier) == [True, True, True]
    [limit] = [c for c in deep.causes if c.kind == "limit"]
    assert limit.code == code.format(p="p9").splitlines()[-1] and not limit.in_earlier
    assert _held(code.format(p=repr(value)), snap) is flat
    before = [*earlier, code.format(p="p9")]
    later = find_cell_writes(code="print(1)", earlier=_earlier(before), snapshots=(snap,))
    assert [(c.kind, c.in_earlier, c.code) for c in later.causes] == [("limit", True, limit.code)]
    assert _cards("print(1)", snap, earlier=before) == [True, True, True]


@pytest.mark.parametrize("script", [
    "cat a.txt && curl -X DELETE https://example.com/items/1", "cat a.txt & curl -X DELETE https://example.com/items/1",
    "cat a.txt; curl -X DELETE https://example.com/items/1", "cat a.txt\ncurl -X DELETE https://example.com/items/1",
    "cat $(curl -X DELETE https://example.com/items/1)", "cat `curl -X DELETE https://example.com/items/1`",
    "cat a.txt && sed -i s/a/b/ stock.csv", "cat < a.txt", "$BIN/cat a.txt",
], ids=["and-list", "background", "semicolon", "newline", "substitution", "backticks", "and-sed-i", "input-redirect",
        "a-program-the-check-cant-name"])
def test_a_read_with_shell_syntax_around_it_past_the_depth_limit_stops(script, snap):
    """Only one plain command whose program the check names as one that
    only reads, with no redirect, pipe, list, substitution, backticks or
    here-document (`_SHELL_SYNTAX`), is free past the depth limit
    (`_does_something`): a read beside a remote delete or a shell write, or
    a program the check can't name, stops, under every value, though the
    twin with the script written out holds no file."""
    earlier = [_nine_names(script)]
    code = "import os\nos.system(p9)"
    assert not any(f.held for f in find_cell_writes(code=code.replace("p9", repr(script)), earlier=(),
                                                    snapshots=(snap,)).files)
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    assert writes.stopped and _cards(code, snap, earlier=earlier) == [True, True, True]


@pytest.mark.parametrize("command", [
    "os.system('curl -X DELETE https://api.example.com/items/' + p9)",
    "subprocess.run(['kubectl', 'delete', 'pod', p9])",
    "os.system('sed -i s/a/b/ ' + p9)",
    "subprocess.run(['python', p9])",
], ids=["a-remote-delete", "kubectl-delete", "sed-i", "python"])
@pytest.mark.parametrize("project", ["no-snapshot", "an-empty-folder"])
def test_a_command_past_the_depth_limit_stops_where_the_project_holds_no_file(command, project, tmp_path, monkeypatch):
    """With no file to hold, the file finder reads nothing, and the walk of
    what a cell launches stops on such a command too (`_shell_commands`):
    in the cell's own code, the card quoting the call it stopped at, and,
    as code the check stopped reading, for every later cell, as where
    files are there."""
    monkeypatch.chdir(tmp_path)
    snapshots = () if project == "no-snapshot" else (
        snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60),)
    code = "import os, subprocess\n" + _nine_names("x") + command
    writes = find_cell_writes(code=code, earlier=(), snapshots=snapshots)
    assert writes.stopped and [c.code for c in writes.causes if c.kind == "limit"] == [command]
    defined = "import os, subprocess\n" + _nine_names("x") + "def go():\n    " + command
    assert find_cell_writes(code="go()", earlier=_earlier([defined]), snapshots=snapshots).stopped
    later = find_cell_writes(code="print(1)", earlier=_earlier([code]), snapshots=snapshots)
    assert [(c.kind, c.in_earlier, c.code) for c in later.causes] == [("limit", True, command)]


# The ways a later cell may reach a function an earlier cell defined: what
# runs it and what the earlier cell defines beside it for that.
_REACHES_GO = {
    "a-call": ("", "go()"),
    "an-alias": ("", "g = go\ng()"),
    "a-thread-target": ("", "import threading\nt = threading.Thread(target=go)\nt.start()\nt.join()"),
    "a-key-function": ("", "sorted([1, 2], key=lambda _: go())"),
    "an-executor": ("", "import concurrent.futures as cf\ncf.ThreadPoolExecutor().submit(go).result()"),
    "a-partial": ("", "import functools\nfunctools.partial(go)()"),
    "a-decorator": ("def deco(f):\n    go()\n    return f\n", "@deco\ndef h():\n    pass"),
    "a-context-manager": ("class C:\n    def __enter__(self):\n        go()\n    def __exit__(self, *a):\n        pass\n",
                          "with C():\n    pass"),
    "a-property": ("class K:\n    @property\n    def x(self):\n        go()\n        return 1\n", "K().x"),
    "a-thread-subclass": ("import threading\nclass T(threading.Thread):\n    def run(self):\n        go()\n",
                          "t = T()\nt.start()\nt.join()"),
    "getattr-of-main": ("", "import __main__\ngetattr(__main__, 'go')()"),
    "globals-by-name": ("", "globals()['go']()"),
    "nothing-runs-it": ("", "print(1)"),
}


@pytest.mark.parametrize("reach", list(_REACHES_GO), ids=list(_REACHES_GO))
@pytest.mark.parametrize("command", [
    "os.system('curl -X DELETE ' + p9)", "os.system('sed -i s/a/b/ ' + p9)", "subprocess.run(['python', p9])",
], ids=["a-remote-delete", "sed-i", "python"])
def test_a_shell_command_past_the_depth_limit_in_an_earlier_function_stops_every_later_cell(command, reach, snap):
    """A shell command past the depth limit on its words that does more than
    one plain read, delete or move is code the check stopped reading
    wherever it sits in an earlier cell, in a function nothing has run
    among them, so every later cell stops under every value whatever
    reaches that function, or nothing does: the card gives the earlier
    step's line and quotes the command, and never the line for a stop in
    the cell's own code."""
    beside, code = _REACHES_GO[reach]
    earlier = [f"{_NINE_TO_A_URL}import os, subprocess\ndef go():\n    {command}\n{beside}"]
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    assert writes.stopped and [(c.kind, c.in_earlier, c.code) for c in writes.causes[:1]] == [("limit", True, command)]
    assert not any(c.kind == "limit" and not c.in_earlier for c in writes.causes)
    assert _cards(code, snap, earlier=earlier) == [True, True, True]


def test_a_shell_command_past_the_depth_limit_through_a_parameter_stops_the_cell_that_hands_it(snap):
    """A function an earlier cell defined may read a command's word past the
    depth limit only through what the checked cell hands it: then the
    checked cell stops, with the line for a stop in its own code and the
    quote of the command in the function."""
    earlier = ["import os\ndef go(url):\n    os.system('curl -X DELETE ' + url)"]
    writes = find_cell_writes(code=f"{_NINE_TO_A_URL}go(p9)", earlier=_earlier(earlier), snapshots=(snap,))
    assert [(c.kind, c.in_earlier, c.code) for c in writes.causes] == [
        ("limit", False, "os.system('curl -X DELETE ' + url)")]
    assert _cards(f"{_NINE_TO_A_URL}go(p9)", snap, earlier=earlier) == [True, True, True]


@pytest.mark.parametrize("source", [
    ("OUT = 'new.csv'\ndef configure(path):\n    global OUT\n    OUT = path\n", "OUT"),
    ("cfg = {'o': 'new.csv'}\ndef change():\n    cfg['o'] = 'stock.csv'\n", "cfg['o']"),
], ids=["a-rebound-name", "a-dict-entry-a-cell-may-change"])
@pytest.mark.parametrize("command, stops", [
    ("os.system(f'sed -i s/a/b/ {p9}')", True),
    ("os.system('curl -X DELETE https://api.example.com/items/' + p9)", True),
    ("subprocess.run(['python', p9])", True),
    ("os.system('cat ' + p9)", False),
], ids=["sed-i", "a-remote-delete", "python", "cat"])
def test_a_command_past_the_depth_limit_through_a_value_the_check_cant_place_stops(source, command, stops, snap):
    """Read on past the depth limit, a word a name code may rebind or a dict
    entry a cell may change gives is still a placeholder, as within the
    limit: the cut may hide its value, so the command stops the check
    (`_shell_commands`), unless it is one plain read, as `cat` is."""
    setup, value = source
    code = "import os, subprocess\n" + setup + f"p0 = {value}\n" + "".join(
        f"p{i} = p{i - 1}\n" for i in range(1, 10)) + command
    assert find_cell_writes(code=code, earlier=(), snapshots=(snap,)).stopped is stops


@pytest.mark.parametrize("command, value, later, stops", [
    ("['cp', 'job.txt', {p}]", "job.py", "import job", True),
    ("['git', 'clone', 'https://example.com/r.git', {p}]", "lib", "import sys\nsys.path.append('lib')\nimport thing",
     True),
    ("'cp job.txt ' + {p}, shell=True", "job.py", "import job", True),
    ("['mv', 'job.txt', {p}]", "job.py", "import job", False),
    ("'mv job.txt ' + {p}, shell=True", "job.py", "import job", False),
], ids=["a-copy-onto-a-module", "a-clone-into-a-folder", "a-copy-onto-a-module-through-a-shell", "a-move-onto-a-module",
        "a-move-onto-a-module-through-a-shell"])
def test_a_command_word_past_the_depth_limit_that_writes_a_module_keeps_a_later_imports_card(
    command, value, later, stops, snap,
):
    """A later cell's import of a module an earlier command wrote, or of one
    from a folder it cloned into, gets a card (`_written_modules`). Where
    the command's word is read past the depth limit, a copy or a clone is
    code the check stopped reading, so the import stops, under every value.
    A move is read on past the limit, by the module walk too
    (`_shell_commands`), even where the name is bound in another cell than
    the command's, so the later import gets the card its twin with the
    word written out gets, and nothing stops."""
    code = "import subprocess\nsubprocess.run(" + command + ")"
    flat = find_cell_writes(code=later, earlier=_earlier([code.format(p=repr(value))]), snapshots=(snap,))
    deep = find_cell_writes(code=later, earlier=_earlier([_nine_names(value), code.format(p="p9")]),
                            snapshots=(snap,))
    assert flat.needs_approval() and not flat.stopped
    assert deep.needs_approval() and deep.stopped is stops
    assert deep.cause_kinds == (("limit", *flat.cause_kinds) if stops else flat.cause_kinds)


@pytest.mark.parametrize("earlier", [
    [_nine_names("job.py"), "import subprocess\nsubprocess.run(['mv', 'job.txt', p9])"],
    ["P = 'job.py'", "import subprocess\nsubprocess.run(['cp', 'job.txt', P])"],
    ["P = 'job.py'", "import os\nos.system(f'cp job.txt {P}')"],
], ids=["a-move-past-the-depth-limit", "a-name-another-cell-binds", "a-name-in-a-script"])
def test_a_module_a_command_writes_through_a_name_another_cell_binds_holds_a_later_import(earlier, tmp_path,
                                                                                          monkeypatch):
    """A program's word may come from another cell than the command's, past
    the depth limit or not: a later import of the module the copy or move
    overwrote runs code the check didn't read, as after the twin with the
    word written out, and nothing stops."""
    (tmp_path / "job.py").write_text("x = 1\n")
    (tmp_path / "job.txt").write_text("x")
    monkeypatch.chdir(tmp_path)
    project = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    assert not find_cell_writes(code="import job", earlier=_earlier(()), snapshots=(project,)).needs_approval()
    flat = find_cell_writes(code="import job", snapshots=(project,),
                            earlier=_earlier(["import subprocess\nsubprocess.run(['cp', 'job.txt', 'job.py'])"]))
    later = find_cell_writes(code="import job", earlier=_earlier(earlier), snapshots=(project,))
    assert flat.needs_approval() and later.needs_approval() and later.cause_kinds == flat.cause_kinds
    assert not later.stopped


def test_a_command_past_the_depth_limit_that_only_reads_adds_no_stop_beside_one_python_runs(snap):
    """The check stops only where a command the depth limit cut short does
    something (`_shell_commands`, `_does_something`): the module Python runs
    within the limit gets its own card, and the listing past it, which its
    twin with the word written out holds nothing for, adds no stop."""
    code = ("import subprocess, random\ncmd = ['python', '-m', 'job']\nif random.random() > 0.5:\n"
            "    cmd = ['ls', p9]\nsubprocess.run(cmd)")
    writes = find_cell_writes(code=code, earlier=_earlier([_nine_names("stock.csv")]), snapshots=(snap,))
    assert writes.needs_approval()
    assert not writes.stopped


# An errored cell that failed on its first line, before it bound `name`
# again: only the reading of errored cells as maybe stopped anywhere gives
# `name` the value 'stock.csv'.
_FAILED_FIRST = "df0 = pd.read_csv('missing.csv')\nname = 'new.csv'"


@pytest.mark.parametrize("before", ["stock.csv", "other.csv"], ids=["a-file-there", "a-new-file"])
@pytest.mark.parametrize("stopped_at", [1, None], ids=["first-line", "line-not-given"])
@pytest.mark.parametrize("code", [
    "pd.DataFrame().to_csv({p}name)",
    "open({p}name, 'w').write('x')",
    "open({p}name, 'a').write('x')",
], ids=["to-csv", "open-for-writing", "append"])
def test_a_path_past_the_depth_limit_in_the_reading_of_an_errored_cell_gets_the_card_its_shallow_twin_gets(
    code, stopped_at, before, daily,
):
    """The reading of errored cells as maybe stopped anywhere reads on past
    the path depth limit too (`_FileFinder.decided`), so a deep path gets
    the decision its shallow twin gets there as well."""
    earlier = [*_DAILY, f"name = {before!r}", _FAILED_FIRST]
    errored, lines = [False] * 5 + [True], [None] * 5 + [stopped_at]
    deep = find_cell_writes(code=code.format(p="DAILY / "), snapshots=(daily,),
                            earlier=_earlier(earlier, errored=errored, stopped_at=lines))
    flat = find_cell_writes(code=code.format(p=_FLAT), snapshots=(daily,),
                            earlier=_earlier(earlier, errored=errored, stopped_at=lines))
    assert deep.needs_approval() is flat.needs_approval()
    assert flat.needs_approval() is (before == "stock.csv" and "'a'" not in code)


def test_a_delete_through_a_path_past_the_depth_limit_holds_whole_and_stops_nothing(daily):
    """A delete through a path the check can't place holds whole, and that
    holds the file the read past the limit finds too (`_covers`): no stop."""
    writes = find_cell_writes(code="(DAILY / 'stock.csv').unlink()", earlier=_earlier(_DAILY), snapshots=(daily,))
    assert [(f.action, f.paths) for f in writes.held_files] == [("delete", ())]
    assert not writes.stopped and not writes.unchecked


@pytest.mark.parametrize("where", ["project", "shared"])
def test_a_delete_past_the_depth_limit_onto_a_whole_root_is_a_stopped_read(where, tmp_path, monkeypatch):
    """Read within the depth limit, the path is a file in a folder of the
    project or of the shared folder, which the delete holds. Read on past
    the limit, it is that root itself, which its twin with the word written
    out holds whole. Those files don't cover the root (`_covers`), so the
    check stops, and the card names none of them, only the code that picks
    what it deletes (`_FileFinder.decided`), as they read as all it may
    delete."""
    root, shared = tmp_path / "proj", tmp_path / "shared"
    for path in (root / "docs" / "a.pdf", shared / "team" / "plan.docx"):
        path.parent.mkdir(parents=True)
        path.write_text("x")
    monkeypatch.chdir(root)
    taken_at = time.time() + 60
    snapshots = (snapshot_project_files(root=root, label="project", taken_at=taken_at),
                 snapshot_project_files(root=shared, label="shared", taken_at=taken_at))
    folder = "docs" if where == "project" else str(shared / "team")
    code = f"import shutil, os\nshutil.rmtree(os.path.join({folder!r}, {{p}}))"
    flat = find_cell_writes(code=code.format(p="'..'"), earlier=_earlier(()), snapshots=snapshots)
    deep = find_cell_writes(code=code.format(p="p9"), earlier=_earlier([_nine_names("..")]), snapshots=snapshots)
    assert [(f.whole_project, f.whole_shared) for f in flat.held_files] == [(where == "project", where == "shared")]
    assert deep.stopped
    assert [(f.action, f.paths, f.whole_project, f.whole_shared) for f in deep.held_files] == [
        ("delete", (), False, False)]


def test_an_overwrite_past_the_depth_limit_after_a_change_the_cell_made_first_is_a_stopped_read(snap):
    """`_FileFinder.decided` hands the deep twin of a node what the cell
    changed before that node, and reads only what the twin finds past it:
    an append first doesn't hide the overwrite read on past the limit."""
    code = "open('stock.csv', 'a').write('x')\nopen(p9, 'w').write('x')"
    writes = find_cell_writes(code=code, earlier=_earlier([_nine_names("stock.csv")]), snapshots=(snap,))
    assert writes.stopped


def test_an_overwrite_through_a_path_within_the_depth_limit_is_read(tmp_path, monkeypatch):
    (tmp_path / "data" / "out").mkdir(parents=True)
    (tmp_path / "data" / "out" / "stock.csv").write_text("x")
    monkeypatch.chdir(tmp_path)
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    earlier = ["from pathlib import Path\nimport pandas as pd\nPROJECT = Path('.')", "DATA = PROJECT / 'data'"]
    writes = find_cell_writes(code="pd.DataFrame().to_csv(DATA / 'out' / 'stock.csv')", earlier=_earlier(earlier),
                              snapshots=(snap,))
    assert [(f.action, f.paths) for f in writes.held_files] == [("overwrite", ("data/out/stock.csv",))]
    assert not writes.unchecked


def test_a_path_past_what_the_reading_of_an_errored_cell_keeps_is_a_stopped_read(snap):
    """An errored cell that may have stopped anywhere gives P each value it
    binds or the one before it: past `_PATH_LIMIT` of them that reading gives
    up, and the overwrite of stock.csv it would hold stops the read."""
    def rebinds(n: int) -> str:
        return "\n".join(f"P = 'a{i}.csv'" for i in range(n))

    held = find_cell_writes(code="open(P, 'w').write('x')", snapshots=(snap,),
                            earlier=_earlier(["P = 'stock.csv'", rebinds(3)], errored=(False, True)))
    assert [(f.action, f.paths) for f in held.held_files] == [("overwrite", ("stock.csv",))]
    past = find_cell_writes(code="open(P, 'w').write('x')", snapshots=(snap,),
                            earlier=_earlier(["P = 'stock.csv'", rebinds(300)], errored=(False, True)))
    assert past.unchecked and past.stopped and not past.unread_here
    # Given up on once, a name stays given up on where a later write reads it.
    again = find_cell_writes(code="Q = P\nopen(Q, 'a').write('x')\nopen(Q, 'w').write('x')", snapshots=(snap,),
                             earlier=_earlier(["P = 'stock.csv'", rebinds(300)], errored=(False, True)))
    assert again.stopped
    # A copy onto it lands on what the reading gave up on too.
    copied = find_cell_writes(code="import shutil\nshutil.copy('summary.md', P)", snapshots=(snap,),
                              earlier=_earlier(["P = 'stock.csv'", rebinds(300)], errored=(False, True)))
    assert copied.unchecked and copied.stopped
    # The files the reading still holds stay on the card beside the stop.
    both = find_cell_writes(code="open(P, 'w').write('x')\nopen(Q, 'w').write('x')", snapshots=(snap,),
                            earlier=_earlier(["P = 'stock.csv'\nQ = 'report.pdf'", rebinds(300) + "\nQ = 'new.txt'"],
                                             errored=(False, True)))
    assert both.stopped and [(f.action, f.paths) for f in both.held_files] == [("overwrite", ("report.pdf",))]


@pytest.mark.parametrize("fixture, earlier, errored", [
    ("daily", [*_DAILY, "pd.DataFrame().to_csv(DAILY / 'stock.csv')"], ()),
    ("snap", ["import os, random\nP = 'output'\n" + "".join(
        f"if random.random() < .5:\n    P += {part!r}\n" for part in ["/old.png", *(f"_v{i}" for i in range(8))])
              + "open(P, 'w').write('x')"], ()),
    ("snap", ["P = 'stock.csv'", "\n".join(f"P = 'a{i}.csv'" for i in range(300)), "open(P, 'w').write('x')"],
     (False, True, False)),
], ids=["path-past-the-depth-limit", "path-past-what-one-value-keeps", "reading-of-an-errored-cell-gave-up"])
def test_a_stop_that_left_only_a_value_unplaced_stops_no_later_cell(fixture, earlier, errored, request):
    """A limit that left a value unplaced, not code unread, stopped only the
    cell whose write lands on it: a later cell runs no code it left unread."""
    snapshot = request.getfixturevalue(fixture)
    own = find_cell_writes(code=earlier[-1], earlier=_earlier(earlier[:-1], errored=errored[:-1]),
                           snapshots=(snapshot,))
    assert own.stopped
    later = find_cell_writes(code="print(1)", earlier=_earlier(earlier, errored=errored), snapshots=(snapshot,))
    assert not later.unchecked


@pytest.mark.parametrize("code, value", [
    ("import subprocess\nsubprocess.run(['rm', {p}])", "new.csv"),
    ("import subprocess\nsubprocess.run(['mv', 'job.txt', {p}])", "new.csv"),
    ("import subprocess\nsubprocess.run(['cat', {p}])", "job.py"),
    ("open({p}, 'w').write('x')", "lib/mod.py"),
], ids=["rm", "mv-onto", "cat-a-module", "python-write-of-a-module"])
def test_a_later_cell_gets_nothing_from_an_earlier_cells_value_stop(code, value, snap):
    """A path past the depth limit, or the word of one plain command that
    only reads, deletes or moves, is a value the check left unplaced, not
    code it left unread: the module and hook walks read it on past the
    limit (`_shell_commands`), so a later cell that only reads gets no card
    from it, and never the line for a stop in its own code."""
    earlier = [_nine_names(value), code.format(p="p9")]
    later = find_cell_writes(code="print(1)", earlier=_earlier(earlier), snapshots=(snap,))
    assert not later.needs_approval() and not later.stopped


@pytest.mark.parametrize("earlier, code, quote", [
    ([_nine_names("stock.csv")], "open(p9, 'w').write('x')", "open(p9, 'w').write('x')"),
    ([_nine_names("stock.csv")], "import subprocess\nsubprocess.run(['sed', '-i', 's/a/b/', p9])",
     "subprocess.run(['sed', '-i', 's/a/b/', p9])"),
    ([_nine_names("rm stock.csv")], "import os\nos.system(p9)", "os.system(p9)"),
    ([], _VARIANTS + "open(P, 'w').write('x')", "open(P, 'w').write('x')"),
], ids=["python-write-past-the-depth-limit", "shell-write-past-the-depth-limit", "shell-delete-past-the-depth-limit",
        "a-path-past-the-path-cap"])
def test_a_value_stop_quotes_the_call_it_stopped_at(earlier, code, quote, snap):
    """Where a limit left the value a write lands on unplaced, the card
    quotes the call it stopped at under "The check stopped at:"
    (`_FileFinder.stopped_at`), as for code a limit left unread."""
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    assert writes.stopped
    [limit] = [c for c in writes.causes if c.kind == "limit"]
    assert limit.code == quote and not limit.in_earlier
    assert "The check stopped at:" in _card(code, snap, earlier=earlier)


def test_a_stop_quote_and_the_steps_code_keep_no_token_after_a_header_name(snap):
    """The card's quotes and the step's code go through `scrub_output`, so
    a token after a header name such as `X-Api-Key:` in a command the
    check stopped at shows as a marker, in the step's own card and in a
    later step's."""
    token = "ghp_" + "aB3dE5fG7hJ9kL1mN2pQ4rS6tU8vW0xY2zA4"
    command = f"os.system('curl -H \"X-Api-Key: {token}\" -X DELETE ' + p9)"
    earlier = [_nine_names("https://api.example.com/items/1")]
    code = f"import os\n{command}"
    for card in (_card(code, snap, earlier=earlier), _card("print(1)", snap, earlier=[*earlier, code])):
        assert "The check stopped at:" in card and "X-Api-Key: [REDACTED_TOKEN]" in card and token not in card


def test_a_dict_entry_read_that_gives_up_is_a_stopped_read(snap):
    """A read again with a dict display's entry may place the write on a
    file the turn began with; where it gives up past `_PATH_LIMIT` paths,
    the read stops, as its twin within the limit holds output/old.png."""
    def cell(branches: int) -> str:
        grows = "".join(f"if random.random() < .5:\n    P += {part!r}\n"
                        for part in ["old.png", *(f"_v{i}" for i in range(branches - 1))])
        return f"import random\ncfg = {{'d': 'output/'}}\nP = ''\n{grows}open(cfg['d'] + P, 'w').write('x')"

    assert find_cell_writes(code=cell(9), earlier=(), snapshots=(snap,)).stopped
    within = find_cell_writes(code=cell(8), earlier=(), snapshots=(snap,))
    assert not within.unchecked and any("output/old.png" in f.paths for f in within.held_files)


def test_a_read_outside_the_paths_a_write_lands_on_is_never_left_given_up_on():
    """Only a read of the paths a write lands on gives up on a value past
    `_PATH_LIMIT` paths (`_Reader.one_value`): any other read, as of the
    folder a chdir() reaches, reads the next value afresh."""
    from anton.core.utils import write_check

    code = _grown(9, "os.chdir(P)\nprint('x')")
    cell = write_check._Cell(write_check._parse(code), code, 0)
    reader = write_check._Reader([cell], current=cell)
    target = next(n for n in ast.walk(cell.tree) if isinstance(n, ast.Name) and n.id == "P"
                  and isinstance(cell.parents.get(n), ast.Call))
    text = next(n for n in ast.walk(cell.tree) if isinstance(n, ast.Constant) and n.value == "x")
    assert isinstance(reader.paths(target, cell=cell), write_check._Unknown)
    assert reader.paths(text, cell=cell) == ["x"]
    with reader.one_value():
        reader.paths(target, cell=cell)
        assert reader.past_limit
    assert not reader.past_limit
    # A read that gave up outside one doesn't carry into the next value read.
    reader.past_limit = True
    with reader.one_value():
        assert not reader.past_limit


def _grown(branches: int, use: str) -> str:
    """A cell that builds P on `output` under `branches` ifs, each of which
    may add a part, the first `/old.png`, then uses P."""
    parts = ["/old.png", *(f"_v{i}" for i in range(branches - 1))]
    grows = "".join(f"if random.random() < .5:\n    P += {part!r}\n" for part in parts)
    return f"import os, random\nP = 'output'\n{grows}{use}"


@pytest.mark.parametrize("use, stops", [
    ("open(P, 'w').write('x')", True),
    ("os.remove(P)", True),
    ("os.rename(P, 'kept.png')", True),
    ("open(P, 'a').write('x')", False),
    ("print(open(P).read())", False),
    ("print(os.path.exists(P))", False),
], ids=["overwrite", "delete", "move", "append", "read", "exists"])
def test_a_path_built_on_past_what_one_value_keeps_is_a_stopped_read(use, stops, snap):
    """Past `_PATH_LIMIT` paths built on one path, a reading gives up on the
    value: an overwrite, delete or move it lands on may be output/old.png,
    which its twin within the limit holds, so the read stops; an append, a
    read or a look stops nothing."""
    past = find_cell_writes(code=_grown(9, use), earlier=(), snapshots=(snap,))
    assert past.stopped is stops and past.unchecked is stops and not past.unread_here
    within = find_cell_writes(code=_grown(8, use), earlier=(), snapshots=(snap,))
    assert not within.unchecked
    assert any("output/old.png" in f.paths for f in within.held_files) is stops


@pytest.mark.parametrize("bindings, writes", [(5, 900), (20, 210), (60, 80)])
def test_the_reading_of_an_errored_cell_gives_up_one_value_at_a_time(bindings, writes, snap):
    """The reading of an errored cell as maybe stopped anywhere reads each
    value within its own budget (`_WIDE_BINDINGS`), so a cell of many
    writes through a name the errored cell binds a few ways is read whole,
    and gets no card for files that aren't there."""
    errored = "\n".join(f"if x == {i}:\n    out = 'r{i}.csv'" for i in range(bindings)) + "\n1 / 0"
    code = "import pandas as pd\n" + "\n".join("pd.DataFrame().to_csv(out)" for _ in range(writes))
    result = find_cell_writes(code=code, snapshots=(snap,), earlier=_earlier(["x = 3", errored], errored=(False, True)))
    assert not result.needs_approval()


def test_an_append_the_reading_of_an_errored_cell_gave_up_on_stops_nothing(snap):
    """An append is never held, so a value given up on that it lands on
    hides nothing the card would hold."""
    rebinds = "\n".join(f"P = 'a{i}.csv'" for i in range(300))
    writes = find_cell_writes(code="open(P, 'a').write('x')", snapshots=(snap,),
                              earlier=_earlier(["P = 'stock.csv'", rebinds], errored=(False, True)))
    assert not writes.needs_approval()


@pytest.mark.parametrize("earlier, code", [
    ([], "import pandas as pd\ncfg = {'out': 'stock.csv', 'n': 3}\npd.DataFrame().to_csv(cfg['out'])"),
    (["cfg = {'out': 'summary.md', 'out': 'stock.csv'}", "print(cfg, 'out' in cfg, cfg.get('out'))"],
     "open(cfg['out'], 'w').write('x')"),
    ([], "def save(df):\n    paths = {'out': 'stock.csv'}\n    df.to_csv(paths['out'])\n"
         "import pandas as pd\nsave(pd.DataFrame())"),
    (["cfg = {'outs': ('stock.csv',)}"], "import pandas as pd\nfor t in cfg['outs']:\n    pd.DataFrame().to_csv(t)"),
    (["cfg: dict = {'out': 'stock.csv'}"], "import pandas as pd\npd.DataFrame().to_csv(cfg['out'])"),
    (["cfg = {'out': 'stock.csv'}"], "from pathlib import Path\nPath('fresh.txt').rename(cfg['out'])"),
    (["cfg = {'out': 'stock.csv'}"], "from pathlib import Path\nPath('fresh.txt').replace(cfg['out'])"),
    (["cfg = {'out': 'stock.csv'}"], "import os\nos.rename('fresh.txt', cfg['out'])"),
    (["cfg = {'out': 'stock.csv'}"], "import shutil\nshutil.copy('fresh.txt', cfg['out'])"),
    (["cfg = {'out': 'stock.csv'}"], "import shutil\nshutil.move('fresh.txt', cfg['out'])"),
    (["cfg = {'out': 'stock.csv'}"], "from pathlib import Path\nhow = 'rename'\ngetattr(Path('fresh.txt'), how)(cfg['out'])"),
], ids=["to-csv", "last-entry-of-a-key-looked-at", "in-a-function", "a-tuple-entry", "an-annotated-display",
        "a-path-renamed-onto-it", "a-path-replaced-onto-it", "a-file-renamed-onto-it", "a-file-copied-onto-it",
        "a-file-moved-onto-it", "a-path-method-looked-up-by-name"])
def test_a_path_read_from_a_dict_display_by_a_literal_key_is_held(earlier, code, snap):
    """Where an overwrite lands, a copy's or a move's destination among
    them, however it is spelled, an entry places the path a write onto a
    file the turn began with lands on (`_FileFinder.paths`)."""
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    assert [(f.action, f.paths) for f in writes.held_files] == [("overwrite", ("stock.csv",))]


@pytest.mark.parametrize("earlier", [
    ["from pathlib import Path\ncfg = {'outs': Path('.').glob('*.csv')}"],
    ["name = 'stock'\ncfg = {'outs': (f'{name}.csv', 'summary.md')}"],
], ids=["an-entry-a-method-gives", "a-tuple-of-an-f-string-with-a-name-in-it"])
def test_a_dict_entry_that_isnt_a_literal_text_or_a_path_places_no_overwrite(earlier, snap):
    """Only an entry that builds a literal, text or a path, which nothing
    changes in place, places an overwrite (`_Reader._unchanging_entry`): a
    method's result, as a glob's, or a tuple that holds an f-string with a
    name in it, places none, so the overwrite counts as a new file, as it
    does with no entry read."""
    writes = find_cell_writes(code="for t in cfg['outs']:\n    open(t, 'w').write('x')", earlier=_earlier(earlier),
                              snapshots=(snap,))
    assert writes.held_files == ()


def test_a_dict_entry_read_through_more_names_than_the_check_follows_places_no_overwrite(snap):
    """An entry is read through at most `_PARAM_DEPTH` names
    (`_Reader._unchanging_entry`): past them it places none, so the
    overwrite counts as a new file, as with no entry read, and no read
    stops."""
    earlier = ["a0 = 'stock.csv'\n" + "".join(f"a{i} = a{i - 1}\n" for i in range(1, 12)) + "cfg = {'out': a11}"]
    writes = find_cell_writes(code="import pandas as pd\npd.DataFrame().to_csv(cfg['out'])", earlier=_earlier(earlier),
                              snapshots=(snap,))
    assert not writes.needs_approval()


@pytest.mark.parametrize("earlier, path", [
    (["OUT = 'stock.csv'\ncfg = {'out': OUT}"], "stock.csv"),
    (["OUT = 'stock.csv'\ncfg = {'out': OUT}\nOUT = 'new.csv'"], "stock.csv"),
    (["from pathlib import Path\ncfg = {'out': Path('output') / 'old.png'}"], "output/old.png"),
    (["import os\nD = 'output'\ncfg = {'out': os.path.join(D, 'old.png')}"], "output/old.png"),
    (["D = 'output'\ncfg = {'out': f'{D}/old.png'}"], "output/old.png"),
    (["D = 'output/'\ncfg = {'out': D + 'old.png'}"], "output/old.png"),
    (["from pathlib import Path\ncfg = {'out': Path('output').joinpath('old.png')}"], "output/old.png"),
    (["from pathlib import Path\ncfg = {'out': Path('output/old.txt').with_suffix('.png')}"], "output/old.png"),
    (["from pathlib import Path\ncfg = {'out': Path('output/new.png').with_name('old.png')}"], "output/old.png"),
    (["from pathlib import Path\ncfg = {'out': Path('output/new.png').with_stem('old')}"], "output/old.png"),
], ids=["a-name-of-text", "a-name-bound-again-after", "a-path-built-with-a-slash", "a-joined-path", "an-f-string",
        "text-added-to-text", "a-path-joinpath", "a-path-with-suffix", "a-path-with-name", "a-path-with-stem"])
def test_a_dict_entry_built_as_text_or_a_path_is_held(earlier, path, snap):
    """An entry that builds text or a path, which nothing changes in place,
    places the write as a literal entry does, read where the dict is built."""
    writes = find_cell_writes(code="import pandas as pd\npd.DataFrame().to_csv(cfg['out'])", earlier=_earlier(earlier),
                              snapshots=(snap,))
    assert [(f.action, f.paths) for f in writes.held_files] == [("overwrite", (path,))]


# Each change below points cfg['f'] from report.pdf to stock.csv where a
# read of the dict display alone would still name report.pdf.
@pytest.mark.parametrize("earlier, here", [
    (["cfg['f'] = 'stock.csv'"], ""),
    (["cfg.update(f='stock.csv')"], ""),
    (["cfg.setdefault('g', 'x')"], ""),
    (["cfg |= {'f': 'stock.csv'}"], ""),
    (["del cfg['f']"], ""),
    (["fill(cfg)"], ""),
    (["c = cfg\nc['f'] = 'stock.csv'"], ""),
    (["box = [cfg]\nbox[0]['f'] = 'stock.csv'"], ""),
    (["fix = cfg.update\nfix(f='stock.csv')"], ""),
    (["def f():\n    global cfg\n    cfg = {'f': 'stock.csv'}", "f()"], ""),
    ([], "def f():\n    global cfg\n    cfg = {'f': 'stock.csv'}\nf()\n"),
    (["def o():\n    cfg = 1\n    def i():\n        nonlocal cfg\n    i()\no()"], ""),
    (["for cfg in [{'f': 'stock.csv'}]:\n    pass"], ""),
    (["import contextlib\nwith contextlib.nullcontext({'f': 'stock.csv'}) as cfg:\n    pass"], ""),
    (["try:\n    1 / 0\nexcept ZeroDivisionError as cfg:\n    pass"], ""),
    (["from json import decoder as cfg"], ""),
    (["(cfg := {'f': 'stock.csv'})"], ""),
    (["def g(cfg):\n    cfg['f'] = 'stock.csv'"], ""),
    (["from os.path import *"], ""),
    (["globals()['cfg']['f'] = 'stock.csv'"], ""),
    (["ns = globals()\nns['cfg']['f'] = 'stock.csv'"], ""),
    (["globals().update(cfg={'f': 'stock.csv'})"], ""),
    (["vars()['cfg']['f'] = 'stock.csv'"], ""),
    (["import __main__\n__main__.cfg['f'] = 'stock.csv'"], ""),
    (["import sys\nsys.modules['__main__'].cfg['f'] = 'stock.csv'"], ""),
    (["import sys\nsetattr(sys.modules['__main__'], 'cfg', {'f': 'stock.csv'})"], ""),
    (["import sys\nsys._getframe().f_globals['cfg']['f'] = 'stock.csv'"], ""),
    (["exec(\"cfg['f'] = 'stock.csv'\")"], ""),
    (["exec(open('job.txt').read())"], ""),
    (["base = {'f': 'stock.csv'}", "cfg = {'f': 'report.pdf', **base}"], ""),
    (["match {'f': 'stock.csv'}:\n    case cfg:\n        pass"], ""),
    (["__import__('__main__').cfg['f'] = 'stock.csv'"], ""),
    (["import importlib\nimportlib.import_module('__main__').cfg['f'] = 'stock.csv'"], ""),
    (["from importlib import import_module\nimport_module('__main__').cfg['f'] = 'stock.csv'"], ""),
    (["from __main__ import cfg as c\nc['f'] = 'stock.csv'"], ""),
    (["f = lambda: 0\ng = f.__globals__\ng['cfg']['f'] = 'stock.csv'"], ""),
    (["import inspect\ninspect.currentframe().f_locals['cfg']['f'] = 'stock.csv'"], ""),
    (["import inspect\ninspect.getmodule(lambda: 0).cfg['f'] = 'stock.csv'"], ""),
    (["import inspect\nsetattr(inspect.getmodule(lambda: 0), 'cfg', {'f': 'stock.csv'})"], ""),
    (["import inspect\ninspect.getmodule(lambda: 0).__dict__['cfg']['f'] = 'stock.csv'"], ""),
    (["import operator, sys\noperator.attrgetter('f_globals')(sys._getframe())['cfg']['f'] = 'stock.csv'"], ""),
    (["import gc\nfor o in gc.get_objects():\n    if type(o) is dict and o.get('f') == 'report.pdf':\n"
      "        o['f'] = 'stock.csv'"], ""),
    (["match {'f': 'stock.csv'}:\n    case {**cfg}:\n        pass"], ""),
    (["try:\n    1 / 0\nexcept ZeroDivisionError as e:\n    e.__traceback__.tb_frame.f_globals['cfg']['f'] = 'stock.csv'"], ""),
    (["try:\n    1 / 0\nexcept ZeroDivisionError as e:\n    e.__traceback__.tb_frame.f_locals['cfg']['f'] = 'stock.csv'"], ""),
    (["import gc\nfor o in gc.get_referrers('report.pdf'):\n    if type(o) is dict:\n        o['f'] = 'stock.csv'"], ""),
    (["from sys import modules\nmodules[__name__].cfg['f'] = 'stock.csv'"], ""),
    (["def cfg():\n    return {'f': 'stock.csv'}"], ""),
    (["class cfg:\n    pass"], ""),
], ids=["entry-stored", "updated", "setdefault", "merged", "entry-deleted", "handed-on", "changed-through-a-name",
        "changed-through-a-list", "method-taken-as-a-value", "rebound-in-a-function", "rebound-in-a-function-here",
        "declared-nonlocal", "loop-target", "with-target", "except-target", "imported-as", "walrus", "parameter",
        "star-import", "through-globals", "through-an-alias-of-globals", "globals-updated", "through-vars",
        "through-main", "through-sys-modules", "setattr-on-main", "through-a-frame", "literal-exec", "unread-exec",
        "spread", "match-capture", "through-dunder-import-of-main", "through-import-module-of-main",
        "through-import-module-by-its-name", "imported-from-main", "through-a-functions-globals",
        "through-a-frames-locals", "through-getmodule", "setattr-on-getmodule", "through-getmodules-dict",
        "through-attrgetter", "through-every-live-object", "match-rest", "through-a-traceback-frames-globals",
        "through-a-traceback-frames-locals", "through-gc-referrers", "through-modules-imported-from-sys",
        "redefined-as-a-function", "redefined-as-a-class"])
def test_a_path_from_a_dict_the_cells_may_change_stays_one_the_check_cant_place(earlier, here, snap):
    """A dict a cell may change, or rebind, or reach through a namespace,
    or one code the check can't read may change, holds a path the check
    can't place: an overwrite through it names no file the display names."""
    writes = find_cell_writes(code=f"{here}open(cfg['f'], 'w').write('x')",
                              earlier=_earlier(["cfg = {'f': 'report.pdf'}", *earlier]), snapshots=(snap,))
    assert not any("report.pdf" in f.paths for f in writes.files)


@pytest.mark.parametrize("code", [
    "globals()", "vars()", "locals()", "setattr(o, 'a', 1)", "delattr(o, 'a')", "__builtins__.open",
    "__import__('m')", "sys._getframe()", "inspect.currentframe()", "inspect.getmodule(f)",
    "operator.attrgetter('a')", "operator.methodcaller('a')", "gc.get_objects()", "gc.get_referrers(cfg)",
    "gc.get_referents(f)", "o.__dict__", "f.__globals__", "f.__builtins__", "frame.f_globals", "frame.f_locals", "sys.modules",
    "importlib.import_module('m')", "__main__", "import __main__", "import builtins", "from __main__ import cfg",
    "from builtins import open", "from sys import modules", "_getframe()", "currentframe()", "getmodule(f)",
    "attrgetter('a')", "methodcaller('a')", "get_objects()", "get_referrers(cfg)", "get_referents(f)",
])
def test_each_way_to_reach_a_namespace_other_than_by_name_refuses_a_dict_entry_on_its_own(code):
    """A cell that may reach a namespace's names other than by name may
    change a dict there, so no entry places a path (`_reaches_a_namespace`).
    Most of these ways also make code the check can't read run, which
    refuses an entry too (`find_cell_writes`), so each is read here on its
    own."""
    from anton.core.utils import write_check

    assert write_check._reaches_a_namespace(ast.walk(ast.parse(code)))
    assert not write_check._reaches_a_namespace(ast.walk(ast.parse("print(cfg['f'], o.a, f(x))")))


@pytest.mark.parametrize("code", ["del cfg", "def f(cfg):\n    pass", "def f():\n    global cfg"],
                         ids=["a-delete", "a-parameter", "a-global-declaration"])
def test_a_delete_parameter_or_declaration_of_a_dicts_name_is_another_place_that_binds_it(code):
    """Each counts as a place that binds the name (`_Cell.binding_sites`),
    so a dict display under that name places no path: the name may hold
    another value there."""
    from anton.core.utils import write_check

    assert write_check._Cell(write_check._parse(code), code, 0).binding_sites["cfg"] == 1


@pytest.mark.parametrize("earlier", [
    [],
    ["import gc\ndef h():\n    pass\nfor o in gc.get_referents(h):\n    if type(o) is dict and 'cfg' in o:\n"
     "        o['cfg']['f'] = 'stock.csv'"],
    ["def h():\n    pass\nobject.__getattribute__(h, '__globals__')['cfg']['f'] = 'stock.csv'"],
    ["def h():\n    pass\ngetattr(h, '__glob' + 'als__')['cfg']['f'] = 'stock.csv'"],
    ["import pickle\npickle.loads(open('state.pkl', 'rb').read())"],
    ["import helpers"],
    ["cfg2 = {'f': 'new.txt'}"],
], ids=["unchanged", "through-gc-referents", "through-getattribute", "through-a-getattr-of-built-text",
        "through-a-pickle-load", "through-a-project-module", "an-entry-naming-a-file-not-there"])
@pytest.mark.parametrize("code, held", [
    ("import os\nos.remove(cfg['f'])", [("delete", ())]),
    ("import shutil\nshutil.move(cfg['f'], 'archive.pdf')", [("move", ())]),
    ("from pathlib import Path\nPath(cfg['f']).unlink()", [("delete", ())]),
], ids=["delete", "move", "unlink"])
def test_a_delete_or_move_through_a_dict_entry_holds_whole(earlier, code, held, snap):
    """A delete or move the check can't place already holds, whole, so a
    dict display's entry never narrows it to the file the display names:
    the entry may have changed since in ways no refusal names."""
    writes = find_cell_writes(code=code, earlier=_earlier(["cfg = {'f': 'report.pdf'}", *earlier]), snapshots=(snap,))
    assert [(f.action, f.paths) for f in writes.held_files] == held


@pytest.mark.parametrize("earlier", [
    ["cfg = alias = {'f': 'report.pdf'}", "alias['f'] = 'stock.csv'"],
    ["alias = (cfg := {'f': 'report.pdf'})", "alias['f'] = 'stock.csv'"],
], ids=["chained-assignment", "walrus-inside-an-assignment"])
def test_a_dict_display_another_name_holds_too_places_no_path(earlier, snap):
    """Another name bound to the same display may change it where the check
    reads only `cfg`: an overwrite through it names no file."""
    writes = find_cell_writes(code="open(cfg['f'], 'w').write('x')", earlier=_earlier(earlier), snapshots=(snap,))
    assert not any("report.pdf" in f.paths for f in writes.files)


@pytest.mark.parametrize("before", [["def f1():\n    f2()", "def f2():\n    f3()", "def f3():\n    f4()",
                                     "def f4():\n    f5()", "def f5():\n    f6()", "def f6():\n    pass"]],
                         ids=["call-chain-past-the-depth-limit"])
@pytest.mark.parametrize("here", ["f1()\n", "import functools\np0 = functools.partial(print, 1)\n"
                                  + "".join(f"p{i} = functools.partial(p{i - 1})\n" for i in range(1, 7)) + "p6()\n"],
                         ids=["call-chain", "partials"])
def test_a_dict_entry_in_a_cell_that_leaves_code_unread_at_a_limit_places_no_path(before, here, snap):
    """Code the checked cell itself leaves unread at a limit may change the
    dict before the overwrite runs: it names no file beside the stop."""
    writes = find_cell_writes(code=f"{here}open(cfg['f'], 'w').write('x')",
                              earlier=_earlier(["cfg = {'f': 'report.pdf'}", *before]), snapshots=(snap,))
    assert writes.stopped and not any("report.pdf" in f.paths for f in writes.files)


def test_a_tuple_entry_that_holds_a_list_stays_a_path_the_check_cant_place(snap):
    """A tuple of literals changes in no place, but a tuple that holds a list
    hands the list out to change."""
    writes = find_cell_writes(code="for t in cfg['f'][0]:\n    open(t, 'w').write('x')",
                              earlier=_earlier(["cfg = {'f': (['report.pdf'],)}", "x = cfg['f'][0]\nx.append('stock.csv')"]),
                              snapshots=(snap,))
    assert not any("report.pdf" in f.paths for f in writes.files)


# Each change below adds stock.csv to the list, set or name cfg['outs']
# holds, through a reference the dict hands out, where a read of the dict
# display alone would name report.pdf alone.
@pytest.mark.parametrize("earlier", [
    ["cfg = {'outs': ['report.pdf']}", "lst = cfg['outs']\nlst.append('stock.csv')"],
    ["cfg = {'outs': ['report.pdf']}", "for v in cfg.values():\n    v.append('stock.csv')"],
    ["cfg = {'outs': ['report.pdf']}", "for k, v in cfg.items():\n    v.append('stock.csv')"],
    ["cfg = {'outs': ['report.pdf']}", "cfg.get('outs').append('stock.csv')"],
    ["cfg = {'outs': ['report.pdf']}", "c2 = cfg.copy()\nc2['outs'].append('stock.csv')"],
    ["cfg = {'outs': ['report.pdf']}", "def add(xs):\n    xs.append('stock.csv')\nadd(cfg['outs'])"],
    ["cfg = {'outs': {'report.pdf'}}", "s = cfg['outs']\ns.add('stock.csv')"],
    ["P = ['report.pdf']", "cfg = {'outs': P}", "lst = cfg['outs']\nlst.append('stock.csv')"],
    ["P = list(['report.pdf'])", "cfg = {'outs': P}", "lst = cfg['outs']\nlst.append('stock.csv')"],
    ["for P in [['report.pdf']]:\n    cfg = {'outs': P}", "lst = cfg['outs']\nlst.append('stock.csv')"],
    ["def mk(P):\n    global cfg\n    cfg = {'outs': P}", "mk(['report.pdf'])",
     "lst = cfg['outs']\nlst.append('stock.csv')"],
    ["P = 'report.pdf'", "P = ['report.pdf']\ncfg = {'outs': P}", "lst = cfg['outs']\nlst.append('stock.csv')"],
], ids=["through-an-alias", "through-values", "through-items", "through-get", "through-a-copy",
        "through-a-function", "a-set-entry", "a-name-entry", "a-name-a-call-gives-a-list", "a-name-a-loop-binds",
        "a-parameter", "a-name-rebound-to-a-list"])
def test_a_dict_entry_a_reference_the_dict_hands_out_may_change_stays_a_path_the_check_cant_place(earlier, snap):
    """Only an entry nothing can change in place, a literal, a tuple of
    them, or text or a path, places a path: a list or set entry, or a name
    that may hold one, may hold what a cell added through a reference the
    dict hands out, so an overwrite through it names no file, not
    report.pdf alone."""
    writes = find_cell_writes(code="for t in cfg['outs']:\n    open(t, 'w').write('x')", earlier=_earlier(earlier),
                              snapshots=(snap,))
    assert not any("report.pdf" in f.paths for f in writes.files)


def test_an_overwrite_through_a_dict_entry_a_reference_may_change_names_no_file_it_may_not_write(snap):
    """`cfg['outs'][0]` names stock.csv once `lst[0]` changed it, so the
    card doesn't name report.pdf."""
    earlier = _earlier(["cfg = {'outs': ['report.pdf']}", "lst = cfg['outs']\nlst[0] = 'stock.csv'"])
    writes = find_cell_writes(code="import pandas as pd\npd.DataFrame().to_csv(cfg['outs'][0])", earlier=earlier,
                              snapshots=(snap,))
    assert not any("report.pdf" in f.paths for f in writes.files)


@pytest.mark.parametrize("earlier, code", [
    (["for cfg in {'f': 'report.pdf'}:\n    pass"], "open(cfg['f'], 'w').write('x')"),
    (["k = 'f'\ncfg = {'f': 'report.pdf', k: 'stock.csv'}"], "open(cfg['f'], 'w').write('x')"),
    ([], "def make():\n    cfg = {'f': 'report.pdf'}\nopen(cfg['f'], 'w').write('x')"),
    (["cfg = {'g': 'report.pdf'}"], "open(cfg['f'], 'w').write('x')"),
], ids=["loop-over-a-display", "a-key-that-isnt-literal", "a-display-another-scope-binds", "missing-key"])
def test_a_lookup_the_dict_display_doesnt_answer_stays_a_path_the_check_cant_place(earlier, code, snap):
    """A lookup whose name doesn't hold the display where it reads it, or
    whose key the display doesn't name, places an overwrite on no file."""
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    assert not any("report.pdf" in f.paths for f in writes.files)


def test_a_dict_display_is_read_once_per_name_however_many_lookups_a_cell_makes(snap, monkeypatch):
    """Whether no cell changes a dict is read once per check and name
    (`_Reader._dict_display`), not again at each of its lookups."""
    from anton.core.utils import write_check

    asked: list[str] = []
    changed = write_check._changed_in_place
    monkeypatch.setattr(write_check, "_changed_in_place",
                        lambda name, found, reader: asked.append(name) or changed(name, found, reader))
    cfg = "cfg = {" + ", ".join(f"'k{n}': 'new{n}.csv'" for n in range(50)) + "}"
    code = "\n".join(f"open(cfg['k{n}'], 'w').write('x')" for n in range(50))
    assert not find_cell_writes(code=code, earlier=_earlier([cfg]), snapshots=(snap,)).needs_approval()
    assert asked == ["cfg"]


def test_cause_kinds_names_every_kind_the_card_has_no_room_for(monkeypatch):
    """The card gives at most three reasons; `cause_kinds`, which the log
    line reads, names every kind, and an error building the reasons keeps it."""
    from anton.core.utils import write_check

    earlier = _earlier(["SCRIPT = open('job.txt').read()", "exec(SCRIPT)"])
    code = "exec(SCRIPT)\nexec(SCRIPT + '1')\nexec(SCRIPT + '2')\njob_main()"
    writes = find_cell_writes(code=code, earlier=earlier, snapshots=())
    assert [c.kind for c in writes.causes] == ["runs", "runs", "runs"]
    assert writes.cause_kinds == ("runs", "uses")

    def boom(*args, **kwargs):
        raise RuntimeError("reasons")

    monkeypatch.setattr(write_check, "_quoted", boom)
    failed = find_cell_writes(code=code, earlier=earlier, snapshots=())
    assert failed.causes == () and failed.cause_kinds == ("runs", "uses")


def test_a_client_bound_through_more_names_than_the_read_follows_is_a_stopped_read():
    code = ("import requests\ns0 = requests.Session()\n" + "".join(f"s{i + 1} = s{i}\n" for i in range(6))
            + "s6.delete('https://erp.example.com/api/orders/1')")
    writes = find_cell_writes(code=code, earlier=(), snapshots=())
    assert writes.unchecked and writes.stopped and not writes.unread_here


def test_a_delete_past_the_depth_limit_in_the_reread_for_an_errored_cell_is_a_stopped_read(snap):
    """Read again with the errored cell as maybe stopped anywhere, the
    delete five calls deep may name stock.csv."""
    defs = "import os\n" + "".join(f"def f{i}():\n    f{i + 1}()\n" for i in range(1, 5)) + "def f5():\n    os.remove(P)"
    writes = find_cell_writes(code="f1()", earlier=_earlier(["P = 'stock.csv'", "x = 1 / 0\nP = 'tmp.csv'", defs],
                                                            errored=(False, True, False)), snapshots=(snap,))
    assert writes.unchecked and writes.stopped and not writes.unread_here


def test_unread_here_is_code_the_cell_runs_not_a_hold_carried_from_an_earlier_cell():
    script = ["SCRIPT = open('job.txt').read()"]
    runs = find_cell_writes(code="exec(SCRIPT)", earlier=_earlier(script), snapshots=())
    assert runs.unchecked and runs.unread_here and not runs.stopped
    uses = find_cell_writes(code="print(orders.mean())", earlier=_earlier([*script, "exec(SCRIPT)"]), snapshots=())
    assert uses.unchecked and not uses.unread_here and not uses.stopped
    assert [c.kind for c in uses.causes] == ["uses"]


@pytest.mark.parametrize(("code", "hook"), [
    ("import site\nopen(site.getsitepackages()[0] + '/x.pth', 'w').write('import boot')", True),
    ("import site, subprocess\nsubprocess.run(['cp', 'h.txt', site.getsitepackages()[0] + '/x.pth'])", True),
    ("import subprocess\nsubprocess.run(['cp', '-vt', '/usr/lib/python3/site-packages', 'h.pth'])", True),
    ("import subprocess\nsubprocess.run(['rsync', '-avt', 'h.pth', '/usr/lib/python3/site-packages/'])", True),
    ("exec(input())", False),
], ids=["pth-open", "pth-cp", "pth-cp-into-a-folder-t-names-in-a-cluster", "pth-rsync-keeping-times", "exec"])
def test_writes_startup_hook_is_a_hook_write_among_the_code_the_cell_runs(code, hook):
    writes = find_cell_writes(code=code, earlier=(), snapshots=())
    assert writes.unchecked and writes.unread_here and writes.writes_startup_hook is hook


def test_a_hook_write_after_code_a_limit_left_unread_is_a_stop_and_a_hook_write_both():
    """Rule L's stop for code an earlier cell left unread and the cell's own
    write of a startup hook combine: each keeps the card under every value
    (`_unchecked_hold_dropped`), and the card gives both reasons."""
    hook = "import site\nopen(site.getsitepackages()[0] + '/x.pth', 'w').write('import boot')"
    writes = find_cell_writes(code=hook, earlier=_earlier([_NINTH]), snapshots=())
    assert writes.stopped and writes.writes_startup_hook and writes.unread_here
    assert writes.cause_kinds == ("limit", "runs")
    assert [(c.kind, c.in_earlier) for c in writes.causes] == [("limit", True), ("runs", False)]


def test_an_error_building_the_reasons_keeps_what_the_check_found(monkeypatch):
    from anton.core.utils import write_check

    def boom(*args, **kwargs):
        raise RuntimeError("reasons")

    monkeypatch.setattr(write_check, "_quoted", boom)
    code = "import requests\nrequests.put('https://erp.example.com/x')\nexec(SCRIPT)"
    writes = find_cell_writes(code=code, earlier=_earlier(["SCRIPT = open('job.txt').read()"]), snapshots=())
    assert writes.unchecked and writes.http and writes.causes == ()


def test_an_error_scrubbing_the_code_leaves_it_off_the_card(monkeypatch):
    from anton.core.utils import scratchpad as scratchpad_mod

    def boom(text):
        raise RuntimeError("scrub")

    monkeypatch.setattr(scratchpad_mod, "scrub_output", boom)
    writes = find_cell_writes(code="exec(SCRIPT)", earlier=_earlier(["SCRIPT = open('job.txt').read()"]),
                              snapshots=())
    card = _card_prompt(writes, "exec(SCRIPT)")
    assert writes.unchecked and card.count("exec(SCRIPT)") == 1, "only the reason quotes it; the code block is left off"


def test_the_card_gives_at_most_three_reasons_once_each():
    card = _unchecked_card("\n".join(["import runpy", *(f"runpy.run_path('job{i}.py')" for i in range(5)),
                                      "runpy.run_path('job0.py')"]))
    assert card.count("It runs code the check can't read:") == 3


def test_the_card_scrubs_a_credential_in_the_steps_code():
    code = 'exec(fetch("https://x", token="sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"))'
    writes = find_cell_writes(code=code, earlier=(), snapshots=())
    card = _card_prompt(writes, code)
    assert "sk-ant-api03" not in card and "[REDACTED_API_KEY]" in card


def test_the_card_scrubs_a_credential_in_a_line_it_quotes():
    card = _unchecked_card('exec(fetch("https://x", token="sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"))')
    assert "sk-ant-api03" not in card and "[REDACTED_API_KEY]" in card


class _EverydayContext(BaseModel):
    """What ran before an everyday cell (`write_check_everyday_cells.yaml`)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    earlier: list[str]


class _EverydayCell(BaseModel):
    """An everyday cell that changes nothing the card holds."""

    model_config = ConfigDict(extra="forbid")

    name: str
    code: str


_EVERYDAY = yaml.safe_load((Path(__file__).parent / "fixtures" / "write_check_everyday_cells.yaml").read_text())
_EVERYDAY_RULE = _EVERYDAY["rule"]
_EVERYDAY_CONTEXTS = [_EverydayContext.model_validate(c) for c in _EVERYDAY["contexts"]]
_EVERYDAY_CELLS = [_EverydayCell.model_validate(c) for c in _EVERYDAY["cells"]]


@pytest.mark.parametrize("cell", _EVERYDAY_CELLS, ids=[c.name for c in _EVERYDAY_CELLS])
@pytest.mark.parametrize("context", _EVERYDAY_CONTEXTS, ids=[c.name for c in _EVERYDAY_CONTEXTS])
def test_everyday_cells_raise_no_card(context: "_EverydayContext", cell: "_EverydayCell", tmp_path, monkeypatch):
    """A cell that changes nothing the card holds, as plots, reads, GETs,
    exports to new files and archives don't, raises no card, in a fresh
    conversation and after earlier cells that ran code the check can't
    read or defined helpers."""
    (tmp_path / "stock.csv").write_text("x")
    monkeypatch.chdir(tmp_path)
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    writes = find_cell_writes(
        code=cell.code, earlier=[EarlierCell(code=code) for code in context.earlier], snapshots=(snap,),
        http_rules=parse_http_write_rules(_EVERYDAY_RULE))
    assert not writes.needs_approval(), _card_prompt(writes)


@pytest.mark.parametrize("earlier, code, card", [
    ([f"exec({_TOO_NESTED!r})"], "print(1)", "stopped"),
    ([f"exec({_TOO_NESTED!r})"], "x = 1", "stopped"),
    ([], f"exec({_TOO_DEEP!r})", "stopped"),
    ([], f"import timeit\ntimeit.timeit({_TOO_DEEP!r})", "stopped"),
    ([], f"import subprocess\nsubprocess.run(['python3', '-c', {_TOO_DEEP!r}])", "stopped"),
    ([f"import timeit\ntimeit.timeit('pass', setup={_TOO_DEEP!r})"], "print(1)", "stopped"),
    ([f"import subprocess\nsubprocess.run(['python3', '-c', {_TOO_DEEP!r}])"], "print(1)", "stopped"),
    ([f"exec({_TOO_DEEP + chr(10) + 'import os' + chr(10) + 'def push():' + chr(10) + '    os.remove(1)'!r})"],
     "push()", "stopped"),
    ([], f"exec({_TOO_LONG!r})", "stopped"),
    ([], f"import subprocess\nsubprocess.run(['python3', '-c', {_TOO_LONG!r}])", "stopped"),
], ids=["earlier-exec-too-deep", "earlier-exec-too-deep-then-a-cell-that-loads-nothing", "exec-too-deep",
        "timeit-too-deep",
        "python-c-too-deep", "earlier-timeit-setup-too-deep", "earlier-python-c-too-deep",
        "call-into-what-earlier-too-deep-code-defined", "exec-too-long", "python-c-too-long"])
def test_literal_code_too_deep_or_too_long_to_read(earlier, code, card, snap):
    """Literal code too deep for the check's parser is a stop, as a cell too
    deep to parse is: a pad may still run it. So is literal code past the
    length the check reads. A later cell after such code an earlier cell
    ran, here or in another process, stops too, and the check doesn't
    crash on it."""
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    assert writes.stopped is (card == "stopped")
    assert writes.needs_approval() is (card != "none")
    if card == "stopped":
        assert writes.causes[0].kind == "limit" and not writes.unread_here


@pytest.mark.parametrize("deep", [_TOO_NESTED, _TOO_DEEP], ids=["nested", "a-long-sum"])
def test_an_earlier_cell_too_deep_to_parse_that_kept_nothing_stops_no_later_cell(deep, snap):
    """A cell too deep for the check's parser whose record says it kept
    nothing (`NOTHING_KEPT`, its process gone) stops no later cell, and
    later checks don't fail."""
    earlier = _earlier([deep, "x = 1"], errored=(True, False), stopped_at=(NOTHING_KEPT, None), ended=(True, False))
    assert not find_cell_writes(code="print(x)", earlier=earlier, snapshots=(snap,)).needs_approval()
    writes = find_cell_writes(code="import os\nos.remove('stock.csv')", earlier=earlier, snapshots=(snap,))
    assert [(f.action, f.paths) for f in writes.held_files] == [("delete", ("stock.csv",))]
    assert not writes.unchecked


def test_a_stop_in_the_steps_own_code_is_not_an_earlier_steps(snap):
    writes = find_cell_writes(code=f"{_EIGHT_EXECS}\nexec({_DEFINES_PUSH!r})", earlier=(), snapshots=(snap,))
    assert [(c.kind, c.in_earlier) for c in writes.causes] == [("limit", False)]


def test_a_long_line_of_code_the_check_cant_read_is_quoted_cut(snap):
    """A card's reason quotes at most `_QUOTED_CHARS` characters of a line,
    as a concatenation handed exec() may be one line of 100,000."""
    code = "exec(\"import os; os.remove('stock.csv')  # \" + 'zz' + '" + "q" * 100_050 + "')"
    writes = find_cell_writes(code=code, earlier=(), snapshots=(snap,))
    assert [c.kind for c in writes.causes] == ["runs"]
    assert writes.causes[0].code == code[:397] + "..."


def test_a_dict_entry_after_an_earlier_cell_too_deep_to_parse_that_may_have_run_places_no_path(snap):
    """Such a cell may have changed the dict, and every later cell stops
    for it (`_earlier_stop`), so its display's entry places no write: the
    card holds no file the display alone names."""
    earlier = _earlier(["cfg = {'out': 'stock.csv'}", _TOO_DEEP])
    writes = find_cell_writes(code="import pandas as pd\npd.DataFrame().to_csv(cfg['out'])", earlier=earlier,
                              snapshots=(snap,))
    assert writes.stopped and writes.held_files == ()
    alone = find_cell_writes(code="import pandas as pd\npd.DataFrame().to_csv(cfg['out'])",
                             earlier=_earlier(["cfg = {'out': 'stock.csv'}"]), snapshots=(snap,))
    assert [(f.action, f.paths) for f in alone.held_files] == [("overwrite", ("stock.csv",))]


def test_literal_code_past_the_limit_ends_what_the_check_reads_of_a_cell(snap):
    """Once a call takes a cell past `_EXEC_LIMIT` texts, the check reads no
    later literal code in it, as two texts one shell hands `python -c`
    take it past: the read stops at that call, and the rename after it
    isn't read, though one more text would fit."""
    seven = "\n".join(["exec('y = 2')"] * 7)
    code = f"import os\n{seven}\nos.system(\"python -c 'y = 1' && python -c 'y = 2'\")\nexec({_RENAME!r})"
    writes = find_cell_writes(code=code, earlier=(), snapshots=(snap,))
    assert writes.stopped and [c.code for c in writes.causes] == ["os.system(\"python -c 'y = 1' && python -c 'y = 2'\")"]
    assert writes.held_files == ()


# A cell with an invalid escape sequence, which Python's parser warns about,
# and one that sends the request after it.
_WARNS = "import re\npat = re.compile('\\d+')\nprint(pat)"
_WARNS_THEN_SENDS = (
    _WARNS + "\nimport requests\nrequests.post('https://erp.example.test/api/orders', json={'qty': 1})"
)


def test_a_check_leaves_no_warning_filter_behind_beside_another_block_of_filters(snap, monkeypatch):
    """Checks run in threads while other code saves and puts back the
    process's warning filters. A block that begins while a check parses and
    ends after it puts back the filters it saved: a check that changed them
    around its parse would leave its own behind, quieting every
    SyntaxWarning in the host for good."""
    other = warnings.catch_warnings()
    entered: list[bool] = []
    parse = ast.parse

    def parse_beside_another_block(*args, **kwargs):
        if not entered:
            entered.append(True)
            other.__enter__()
        return parse(*args, **kwargs)

    monkeypatch.setattr(ast, "parse", parse_beside_another_block)
    with warnings.catch_warnings():
        find_cell_writes(code=_WARNS, earlier=_earlier([]), snapshots=(snap,))
        assert entered
        other.__exit__(None, None, None)
        assert ("ignore", None, SyntaxWarning, None, 0) not in warnings.filters


def test_a_check_reads_a_cell_its_parse_warns_about_where_warnings_are_errors(snap):
    """The check's own parse of a cell Python warns about raises no warning
    in the host, even where the host turns warnings into errors, so it
    reads the cell the runner compiles."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        writes = find_cell_writes(code=_WARNS_THEN_SENDS, earlier=_earlier([]), snapshots=(snap,),
                                  http_rules=parse_http_write_rules("POST https://erp.example.test/**"))
    assert writes.needs_approval() and "http" in writes.kinds()


# Two thousand names that name no member of `types`, as a cell may list them.
_MANY_NAMES = ", ".join(repr(f"Name{n}") for n in range(2000))


@pytest.mark.parametrize("last, card", [("SimpleNamespace", "none"), ("FunctionType", "unchecked")],
                         ids=["no-name-runs-code", "the-last-name-runs-code"])
def test_a_lookup_over_many_literal_names_reads_each_name(last, card, snap):
    """A getattr on a module that holds a runner, over a long list of
    literal names, reads every one of them: none that runs code holds
    nothing, and one at the end that does holds the cell."""
    code = f"import types\nfor n in [{_MANY_NAMES}, {last!r}]:\n    print(getattr(types, n, None))"
    writes = find_cell_writes(code=code, earlier=_earlier([]), snapshots=(snap,))
    assert writes.unchecked is (card == "unchecked")
    assert writes.needs_approval() is (card != "none")


def test_an_import_of_a_module_written_since_the_turn_began_is_unchecked(tmp_path, monkeypatch):
    """A module on disk now but missing from the snapshot the turn began with
    is one a cell wrote where the check couldn't place it; one the snapshot
    holds is a project module. The snapshot comparison alone decides this,
    with no earlier cell: `job.py` and `pkg/` land after the snapshot, so an
    import of either is unchecked, while `helpers.py`, in the snapshot, is a
    project module an import runs no unread code to reach."""
    (tmp_path / "stock.csv").write_text("x")
    (tmp_path / "helpers.py").write_text("x = 1\n")
    monkeypatch.chdir(tmp_path)
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    (tmp_path / "job.py").write_text("import requests\nrequests.post('https://erp.example.test/api/orders')\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    for code in ("import job", "from pkg import run"):
        assert find_cell_writes(code=code, earlier=_earlier(()), snapshots=(snap,)).unchecked
    assert not find_cell_writes(code="import helpers", earlier=_earlier(()), snapshots=(snap,)).needs_approval()


@pytest.mark.parametrize("error, card", [(errno.EACCES, "unchecked"), (errno.EIO, "unchecked"),
                                         (errno.ENAMETOOLONG, "none")])
def test_a_module_the_system_cant_look_up_counts_as_one_a_cell_wrote(error, card, tmp_path, monkeypatch):
    """Where the system refuses to say whether a module the snapshot lacks
    is on disk now, a cell may have written it, so the import after a
    `sys.path` change is unchecked; a name too long to be a file is no
    module."""
    (tmp_path / "stock.csv").write_text("x")
    monkeypatch.chdir(tmp_path)
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    is_file = Path.is_file

    def refusing(path: Path) -> bool:
        if path.name == "helpers.py":
            raise OSError(error, os.strerror(error))
        return is_file(path)

    monkeypatch.setattr(Path, "is_file", refusing)
    writes = find_cell_writes(code="import sys\nsys.path.insert(0, '.')\nimport helpers", earlier=_earlier([]),
                              snapshots=(snap,))
    assert writes.unchecked is (card == "unchecked")


def test_an_import_of_a_module_written_since_the_turn_began_in_a_folder_on_sys_path_is_unchecked(
    tmp_path, monkeypatch,
):
    """A module a cell unpacks this turn into a folder the cells put on
    `sys.path`, or a generator writes there, is on disk but missing from the
    snapshot the turn began with, so an import of it after the change is
    unchecked; so is any other import, even of a module Python ships, which
    what the archive unpacked may shadow, but not of one the pad's boot
    loaded before any cell ran. Without the unpack, an installed package's
    import through the folder runs no unread code."""
    (tmp_path / "stock.csv").write_text("x")
    monkeypatch.chdir(tmp_path)
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    job = "import requests\ndef refresh():\n    requests.post('https://erp.example.test/api/orders')\n"
    for rel in ("vendor/erp_sync.py", "vendor/src/erp_pkg/core.py", ".anton/artifacts/sales/backend.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(job)
    (tmp_path / "vendor/src/erp_pkg/__init__.py").write_text("")
    unpack = "import zipfile\nzipfile.ZipFile('bundle.zip').extractall('vendor')"
    for earlier, code in [
        ([unpack], "import sys\nsys.path.insert(0, 'vendor')\nimport erp_sync\nerp_sync.refresh()"),
        ([unpack], "import sys\nsys.path.insert(0, 'vendor/src')\nfrom erp_pkg.core import refresh\nrefresh()"),
        ([], "import sys\nsys.path.insert(0, '.anton/artifacts/sales')\nfrom backend import refresh\nrefresh()"),
        ([], f"{unpack}\nimport sys\nsys.path.insert(0, 'vendor')\nimport erp_sync\nerp_sync.refresh()"),
    ]:
        assert find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,)).unchecked, code
    shadowing = "import sys\nsys.path.insert(0, 'vendor')\nimport pandas"
    assert find_cell_writes(code=shadowing, earlier=_earlier([unpack]), snapshots=(snap,)).unchecked
    shipped = "import sys\nsys.path.insert(0, 'vendor')\nimport csv"
    assert find_cell_writes(code=shipped, earlier=_earlier([unpack]), snapshots=(snap,)).unchecked
    for earlier, code in [([unpack], "import sys\nsys.path.insert(0, 'vendor')\nimport json"), ([], shadowing)]:
        assert not find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,)).needs_approval(), code


def test_literal_code_an_earlier_cell_runs_is_parsed_once_as_the_conversation_grows(snap, monkeypatch):
    """Literal code an earlier cell runs is cached by where it runs
    (`_Cell.order`), which a growing conversation doesn't move, so the next
    turn's check, with one more earlier cell, parses it no more."""
    from anton.core.utils import write_check

    monkeypatch.setattr(write_check, "_EXEC_CELLS", write_check._ExecCells(budget=1_000_000))
    literal = "\n".join(f"rows_{n} = [r for r in range({n})]" for n in range(40))
    earlier = [f"exec({literal!r})", "print(0)"]
    assert not find_cell_writes(code="print(1)", earlier=_earlier(earlier), snapshots=(snap,)).needs_approval()
    parsed: list[str] = []
    parse = write_check._parse_ran

    def counting(text: str):
        parsed.append(text)
        return parse(text)

    monkeypatch.setattr(write_check, "_parse_ran", counting)
    earlier.append("print(1)")
    assert not find_cell_writes(code="print(2)", earlier=_earlier(earlier), snapshots=(snap,)).needs_approval()
    assert literal not in parsed


def _nest_of_true_ifs(depth: int, inner: str) -> str:
    """A cell of `depth` nested `if True:` blocks around `inner`."""
    return "".join("    " * k + "if True:\n" for k in range(depth)) + "    " * depth + inner + "\n"


def _elif_chain(arms: int, *, last: str) -> str:
    """A cell that binds `push` in each of `arms` arms of an if and its
    elifs, runs `last` in its else, then calls `push`."""
    return "code = rows\n" + "\n".join(
        ("if" if k == 0 else "elif") + f" code == {k}:\n    push = print" for k in range(arms)
    ) + f"\nelse:\n    {last}\npush()"


def _flags_under_ifs(start: str, flag: str, *, count: int) -> str:
    """A cell that builds `cmd` from `start` with `flag`, its `{i}` the if's
    number, added under each of `count` ifs, then runs it."""
    adds = "".join(f"if os.environ.get('F{i}'):\n    cmd = cmd + {flag.format(i=i)}\n" for i in range(count))
    run = "subprocess.run(cmd)" if start.startswith("[") else "os.system(cmd)"
    return f"import os, subprocess\ncmd = {start}\n{adds}{run}"


@pytest.mark.parametrize("earlier, code, card", [
    (["def f(a, b):\n    print(a, b)\n" + "".join(f"    f({i}, b)\n" for i in range(1, 16))
      + "".join(f"    f(a, {i})\n" for i in range(1, 16))], "f(0, 0)", "none"),
    (["def step(state, retries=0):\n" + "".join(
        f"    if state == 's{i}':\n        step('s{i + 1}', retries)\n        step('s{i}', retries=1)\n"
        for i in range(40)
    ) + "    print(state)"], "step('s0')", "none"),
    (["def step(state, retries=0):\n" + "".join(
        f"    if state == 's{i}':\n        step('s{i + 1}', retries)\n        step('s{i}', retries=1)\n"
        for i in range(80)
    ) + "    print(state)"], "step('s0')", "none"),
    ([], "import subprocess\ncmd = ['ls']\n" + "".join(f"cmd = cmd + ['-{c}']\n" for c in "abcdefghijklmnop")
     + "subprocess.run(cmd)", "none"),
    ([], "import subprocess\ncmd = ['ls']\n" + "".join(f"cmd = cmd + ['-v{i}']\n" for i in range(300))
     + "subprocess.run(cmd)", "none"),
    ([], "import os\ncmd = 'ls'\n" + "".join(f"cmd = cmd + ' -v{i}'\n" for i in range(300)) + "os.system(cmd)", "none"),
    (["cmd = 'ls'", *(f"cmd = cmd + ' -v{i}'" for i in range(200))], "import os\nos.system(cmd)", "none"),
    (["cmd = ['ls']", *(f"cmd = cmd + ['-v{i}']" for i in range(200))], "import subprocess\nsubprocess.run(cmd)", "none"),
    (["m = 'js'", *("m = m + ''" for _ in range(200)), "m = m + 'on'"], "import importlib\nimportlib.import_module(m)",
     "none"),
    (["cmd = ['python3', 'job.py']", *(f"cmd = cmd + ['-v{i}']" for i in range(200))],
     "import subprocess\nsubprocess.run(cmd)", "unchecked"),
    ([], _flags_under_ifs("['ls']", "['-l']", count=30), "none"),
    ([], _flags_under_ifs("['ls']", "['-a{i}']", count=30), "none"),
    ([], _flags_under_ifs("['python3', 'job.py']", "['-v']", count=30), "unchecked"),
    ([], _flags_under_ifs("'ls'", "' -l'", count=30), "unchecked"),
    ([], "import os, sys\ncmd = sys.executable\n" + "".join(f"cmd = cmd + ' -v{i}'\n" for i in range(300))
     + "os.system(cmd)", "unchecked"),
    (["exec(open('helpers.py').read())", "d0 = load_orders()",
      *("\n".join(f"d{k + j} = d{k + j - 1}.assign(c={k + j})" for j in range(10)) for k in range(1, 1000, 10))],
     "print(1)", "none"),
    (["exec(open('helpers.py').read())", "d0 = load_orders()",
      "\n".join(f"d{k} = d{k - 1}.assign(c={k})" for k in range(1, 2001))], "print(1)", "none"),
    ([], f"import subprocess, sys\nfor p in {[f'pkg{i}' for i in range(3000)]!r}:\n"
     "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', p])", "none"),
    ([], f"import subprocess\nfor a in {[*(f'tool{i}' for i in range(999)), 'python3']!r}:\n"
     f"    for b in {[f'arg{i}' for i in range(1000)]!r}:\n"
     "        head = [a]\n        tail = [b]\n        subprocess.run(head + tail)", "unchecked"),
    (["import os\nimport pandas as pd\nout = 'reports'\n" + "".join(f"out = os.path.join(out, 'part{i}')\n" for i in range(300))
      + "pd.DataFrame({'a': [1]}).to_csv(os.path.join(out, 'summary.csv'))"], "import requests", "none"),
    (["import importlib.util, sys\nspec = importlib.util.spec_from_file_location('job', 'job.py')\n" + "spec = spec\n" * 200
      + "m = importlib.util.module_from_spec(spec)\nsys.modules[spec.name] = m\nspec.loader.exec_module(m)"],
     "import json\nprint(json.dumps(1))", "none"),
    ([], "import importlib.util, sys\nspec = importlib.util.spec_from_file_location('job', 'job.py')\n" + "spec = spec\n" * 200
     + "m = importlib.util.module_from_spec(spec)\nsys.modules[spec.name] = m\nspec.loader.exec_module(m)", "unchecked"),
    ([], "import os\ncmd = 'ls'; cmd = cmd + ' -a'; cmd = cmd + ' -l'\nos.system(cmd)", "none"),
    ([], "import os\ncmd = 'python3'; cmd = cmd + ' -u'; cmd = cmd + ' job.py'\nos.system(cmd)", "unchecked"),
    (["exec(open('helpers.py').read())"], _nest_of_true_ifs(30, "push = print") + "push()", "none"),
    (["exec(open('helpers.py').read())", "def setp():\n    global push\n    push = print"],
     _nest_of_true_ifs(30, "x = 1") + "setp()\npush()", "none"),
    (["import sys\nmake = eval(open('job.txt').read())\nsys.modules['m0'] = make()",
      *(f"import sys, m{i}\nsys.modules['m{i + 1}'] = m{i}.make()" for i in range(150))],
     "import m150\nm150.go()", "unchecked"),
    (["import sys\nmake = eval(open('job.txt').read())\nsys.modules['m0'] = make()",
      *(f"import sys, m{i}\nsys.modules['m{i + 1}'] = m{i}.make()" for i in range(300))],
     "import m300\nm300.go()", "unchecked"),
    ([f"import types, sys\nm{i} = types.ModuleType('m{i}')\nexec(open('job.txt').read(), m{i}.__dict__)\n"
      f"sys.modules['m{i}'] = m{i}" for i in range(1000)], "print(1)", "none"),
    (["def rows():\n" + "".join(f"    a{i} = {i}\n" for i in range(2000)) + "    return ['a', 'b']\n"],
     "import subprocess\n" + "for v in rows():\n    subprocess.run(['echo', v])\n" * 300, "none"),
    ([], "def h0(x):\n    help(x)\n" + "".join(
        f"def h{i}(x):\n" + f"    h{i - 1}(x)\n" * 12 for i in range(1, 7)) + "h6(open('stock.csv').read())\n", "none"),
    ([], "import timeit\ndef f():\n    return sum(range(10))\ndef h0(x):\n    return timeit.timeit(x, number=1)\n" + "".join(
        f"def h{i}(x):\n" + f"    h{i - 1}(x)\n" * 20 for i in range(1, 5)) + "print(h4(f))\n", "none"),
], ids=["recursion-with-two-literal-parameters", "recursive-state-machine", "recursive-state-machine-of-eighty-states",
        "command-list-rebuilt-sixteen-times",
        "command-list-rebuilt-three-hundred-times", "command-text-rebuilt-three-hundred-times",
        "command-text-rebuilt-over-two-hundred-cells", "command-list-rebuilt-over-two-hundred-cells",
        "module-name-rebuilt-over-two-hundred-cells", "python-command-rebuilt-over-two-hundred-cells",
        "command-list-flag-under-thirty-ifs", "command-list-thirty-flags-under-thirty-ifs",
        "python-command-flag-under-thirty-ifs", "command-text-flag-under-thirty-ifs",
        "python-command-text-rebuilt-three-hundred-times", "names-derived-from-unread-code-over-a-hundred-cells",
        "two-thousand-names-derived-from-unread-code-in-one-cell", "pip-install-loop-over-three-thousand-names",
        "command-joined-from-two-loops-over-a-thousand-names", "import-after-a-path-rebuilt-three-hundred-times",
        "import-after-a-module-stored-at-the-end-of-a-spec-alias-chain", "a-module-loaded-at-the-end-of-a-spec-alias-chain",
        "command-text-rebuilt-twice-on-its-first-line",
        "python-command-text-rebuilt-twice-on-its-first-line", "call-after-thirty-nested-true-ifs-after-unread-code",
        "call-after-thirty-nested-true-ifs-beside-a-global-setter", "import-at-the-end-of-150-stored-modules-each-from-the-last",
        "import-at-the-end-of-300-stored-modules-each-from-the-last",
        "a-thousand-cells-that-store-a-module-unread-code-filled", "loops-over-what-a-two-thousand-line-function-returns",
        "a-parameter-handed-down-six-helpers-each-called-twelve-times",
        "a-function-handed-down-four-helpers-to-timeit-each-called-twenty-times"])
def test_a_recursion_or_a_rebuilt_command_is_read_in_well_under_a_second(earlier, code, card, snap):
    """A function the chain already reads is read again for other literals
    only so often (`_REREAD_LIMIT`); a name rebound from itself reads each
    binding once (`_Reader.recalled`), down a long chain of them in a loop
    (`_Reader.reading_ahead`), with no deep recursion; and the commands a
    name rebuilt under many ifs gives are capped (`_capped`): so none of
    them stalls the turn or crashes the check. A function past a limit is
    scanned once per set of arguments, and once in all when the scan reads
    none of them (`_CellRead.past_the_limit`). Past `_COMMAND_LIMIT`
    choices, a command's text is one the check can't read, and its program
    with it. A path read from the interpreter's reads each name once
    (`_interpreter_text`). A chain of names, each built on the one before
    it from what code the check can't read made, reads each link again
    once (`_names_tainted`). A loop over a long literal list gives one
    command for each name in it, and two such lists joined with `+` join
    as one list each (`_word_lists`), not one for each pair. A path a write
    names, or the name a module was made under, read through a name
    rebuilt from itself reads each binding once (`_Reader.read_bindings`).
    A read sees only the bindings before it on its own line, so names
    rebuilt on one line read down to their first value. An if whose test is
    a constant reads the arm that runs, once (`_if_binds`); a chain of
    modules each stored in `sys.modules` from the one before reads where
    each store reaches by name (`_StoredReach`), each store read in the
    same worklist as the names it fills (`_names_tainted`); a name a cell
    binds by a def, class or import is found by its name
    (`_Reader.makers`); what a function returns is read once per def
    (`_Cell.returns_of`); and a name rebound from itself, handed to a call,
    reads each binding once per call and depth (`_Reader.callees`).

    The bound covers a later turn's check, with the earlier cells already
    parsed (`_EARLIER_CELLS`): each blowup these cases pin is in the read
    itself, which every check repeats, while a first check also parses each
    earlier cell once, a cost that grows only with their size."""
    find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    started = time.perf_counter()
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    assert time.perf_counter() - started < 1
    assert writes.unchecked is (card == "unchecked")
    assert writes.needs_approval() is (card != "none")


@pytest.mark.parametrize("earlier, code, card", [
    (["exec(open('helpers.py').read())", "rows = 3"], _elif_chain(1000, last="push = len"), "none"),
    (["exec(open('helpers.py').read())", "rows = 3"], _elif_chain(1000, last="pass"), "unchecked"),
    (["exec(open('helpers.py').read())"], "x = 1" + " + 1" * 3000 + "\nif (push := print):\n    pass\npush()", "none"),
    ([], "import os\nos.system('ls'" + " + ' -l'" * 1000 + ")", "none"),
    ([], "import os, sys\nos.system(sys.executable" + " + ' -u'" * 1000 + " + ' job.py')", "unchecked"),
], ids=["thousand-arm-elif-chain-that-binds-on-every-arm", "thousand-arm-elif-chain-with-an-arm-that-binds-nothing",
        "walrus-after-a-three-thousand-term-sum", "command-text-of-a-thousand-parts",
        "python-command-text-of-a-thousand-parts"])
def test_a_long_chain_in_a_cell_is_read_with_no_deep_recursion(earlier, code, card, snap):
    """An elif chain is read arm by arm (`_if_binds`), a walrus's reach with
    a stack of parts (`_walrus_binds`), and a path or command joined from
    many parts from its first part on (`_interpreter_text`), so a long one
    neither exhausts the stack, which would hold the cell for no reason,
    nor reads past what binds it."""
    if not _parses(code):
        card = "unchecked"
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    assert writes.unchecked is (card == "unchecked")
    assert writes.needs_approval() is (card != "none")


def test_a_helper_many_errored_cells_redefine_is_read_once_per_call(snap):
    """A def an errored cell runs before the line it stopped at replaced the
    one before it (`_Reader.ran_before_stop`), so each call reads the latest
    def alone, not every one before it that a cell that stopped early may
    have left in place."""
    earlier = [f"def helper(x):\n    print(x, {i})\nrows = int('a')" for i in range(200)]
    code = "\n".join(f"helper({k})" for k in range(50))
    started = time.perf_counter()
    writes = find_cell_writes(code=code, earlier=_earlier(earlier, errored=[True] * 200, stopped_at=[3] * 200), snapshots=(snap,))
    assert time.perf_counter() - started < 1
    assert not writes.needs_approval()


def test_a_long_conversation_of_helper_calls_is_read_in_well_under_a_second(snap):
    """Each earlier cell is read for a stop alone, once for its place in
    the conversation (`_LEFT_UNREAD`), so a check after many cells
    that each call helpers an earlier cell defined doesn't read each helper
    again at each call. The first call parses and reads the cells and isn't
    timed."""
    helpers = "import pandas as pd, sqlite3\ncon = sqlite3.connect(':memory:')\n" + "\n".join(
        f"def h{i}(df, name):\n" + "".join(f"    df = df.assign(c{j}=1)\n    con.execute('select {j}')\n"
                                           for j in range(25)) + "    df.to_csv(name)\n    return df"
        for i in range(20))
    earlier = [helpers, *("df = pd.DataFrame()\n" + "\n".join(f"df = h{(k + j) % 20}(df, 'out_{k}_{j}.csv')"
                                                              for j in range(20)) for k in range(200))]
    find_cell_writes(code="print(1)", earlier=_earlier(earlier), snapshots=(snap,))
    started = time.perf_counter()
    writes = find_cell_writes(code="print(1)", earlier=_earlier(earlier), snapshots=(snap,))
    assert time.perf_counter() - started < 1
    assert not writes.needs_approval()


def test_a_long_conversation_of_calls_that_go_five_deep_is_read_in_well_under_a_second(snap):
    """A call whose functions go past `_CALL_DEPTH` may stop the check, so
    the first check reads each earlier cell's calls for their stops alone
    (`_CellRead.stops_only`), not for every change they make, and keeps
    what it found by each cell's place (`_LEFT_UNREAD`): the timed check
    reads no earlier cell again."""
    chain = "def n4(df):\n    return df.dropna()\n" + "".join(f"def n{i}(df):\n    return n{i + 1}(df)\n"
                                                             for i in range(3, 0, -1))
    helpers = "import pandas as pd, sqlite3\ncon = sqlite3.connect(':memory:')\n" + chain + "\n".join(
        f"def h{i}(df, name):\n    df = n1(df)\n" + "".join(f"    df = df.assign(c{j}=1)\n    con.execute('select {j}')\n"
                                                         for j in range(10)) + "    df.to_csv(name)\n    return df"
        for i in range(5))
    earlier = [helpers, *("df = pd.DataFrame()\n" + "\n".join(f"df = h{(k + j) % 5}(df, 'out_{k}_{j}.csv')"
                                                              for j in range(3)) for k in range(300))]
    find_cell_writes(code="print(1)", earlier=_earlier(earlier), snapshots=(snap,))
    started = time.perf_counter()
    writes = find_cell_writes(code="print(1)", earlier=_earlier(earlier), snapshots=(snap,))
    assert time.perf_counter() - started < 1
    assert not writes.needs_approval()


@pytest.mark.parametrize("also", ["", "ns = {}\nexec('def f():\\n    return 1', ns)\n"],
                         ids=["plain-calls", "literal-code-that-defines-a-name-in-a-namespace-of-its-own"])
def test_earlier_cells_read_for_a_stop_read_each_helper_that_holds_none_once(also, snap, monkeypatch):
    """Earlier cells no check kept a stop for, as after a restart of the
    process, are read with one reader (`_earlier_stop`), which tells once,
    for every cell, that a helper's read can find no stop
    (`_stop_free_depth`, `_Reader.stop_free`): no cell's call reads the
    helper again. A later cell that rebinds what each cell binds, which no
    function reads, or that defines a name again in a namespace of its own,
    changes no earlier cell's read (`_read_apart`)."""
    import ast

    from anton.core.utils import write_check

    reads: list[str] = []
    unread_code = write_check._unread_code

    def counted(reader, root, *args, **kwargs):
        if isinstance(root, ast.FunctionDef):
            reads.append(root.name)
        return unread_code(reader, root, *args, **kwargs)

    monkeypatch.setattr(write_check, "_unread_code", counted)
    helpers = "\n".join(f"def h{i}(x):\n    y = x + {i}\n    return y" for i in range(3))
    earlier = [f"{helpers}  # read once {also!r}", *(f"{also}a = h0({k})\nb = h1(a)\nc = h2(b)" for k in range(20))]
    assert not find_cell_writes(code="print(1)", earlier=_earlier(earlier), snapshots=(snap,)).unchecked
    assert sorted(reads) == ["h0", "h1", "h2"]


@pytest.mark.parametrize("again, apart", [
    ("import pandas as pd\n", False),
    ("import polars as pd\n", True),
    ("pd = None\nimport pandas as pd\n", True),
], ids=["the-same-module", "another-module-under-the-name", "the-name-bound-another-way-too"])
def test_an_import_again_of_the_same_module_reads_no_earlier_cell_apart(again, apart, snap, monkeypatch):
    """A name each cell binds only by an import of one module holds that
    module from the first such import on, so a later import of it changes
    no earlier cell's read, at its top level or in a function: read after a
    restart, no cell is read with the cells up to it alone (`_read_apart`).
    A name a later cell binds to another module, or binds another way too,
    still sends a cell apart."""
    from anton.core.utils import write_check

    monkeypatch.setattr(write_check, "_LEFT_UNREAD", write_check._LeftUnread(limit=100_000))
    found: list[frozenset[int]] = []
    read_apart = write_check._read_apart
    monkeypatch.setattr(write_check, "_read_apart", lambda cells: found.append(read_apart(cells)) or found[-1])
    earlier = ["import pandas as pd\ndef load():\n    return pd.read_csv('stock.csv')\ndf0 = pd.read_csv('stock.csv')",
               *(f"{again}df{i} = load()" for i in range(1, 6))]
    find_cell_writes(code="print(1)", earlier=_earlier(earlier), snapshots=(snap,))
    assert bool(found[0]) is apart


def test_a_long_conversation_that_imports_a_module_again_in_each_cell_is_read_in_well_under_a_second(
    snap, monkeypatch,
):
    """Read after a restart, a conversation that imports pandas again in
    each cell costs about what one that imports it once does, not a read of
    each cell apart from the later ones (`_read_apart`). CPU time, with a
    parse cache of its own, the cells parsed before the timed read."""
    from anton.core.utils import write_check

    monkeypatch.setattr(write_check, "_EARLIER_CELLS", write_check._EarlierCells(budget=1_000_000))

    def first_check_after_a_restart(again: str) -> float:
        earlier = _earlier(("import pandas as pd\n" if i == 0 else again)
                           + f"df{i} = pd.read_csv('stock.csv')\ndf{i}['b'] = df{i}['a'] * 2" for i in range(400))
        find_cell_writes(code="print(1)", earlier=earlier, snapshots=(snap,))
        monkeypatch.setattr(write_check, "_LEFT_UNREAD", write_check._LeftUnread(limit=100_000))
        started = time.process_time()
        find_cell_writes(code="print(1)", earlier=earlier, snapshots=(snap,))
        return time.process_time() - started

    once = first_check_after_a_restart("")
    assert first_check_after_a_restart("import pandas as pd\n") < 2 * once + 0.05


def test_a_long_conversation_after_unread_code_reads_each_earlier_cell_for_a_stop_once(snap):
    """Code the check can't read in the first cell changes nothing in how
    often each later cell is read for a stop: once for its place in the
    conversation (`_LEFT_UNREAD`), not again at each check, for each
    helper call in it. CPU time, so a loaded machine doesn't fail it; the
    first call parses and reads the cells and isn't timed."""
    helpers = "\n".join(f"def h{i}(x):\n" + "".join(f"    x = x + {j}\n" for j in range(20)) + "    return x"
                        for i in range(5))
    earlier = ["exec(open('setup_helpers.py').read())", helpers,
               *("\n".join(f"y = h{(k + j) % 5}({k})" for j in range(3)) for k in range(200))]
    find_cell_writes(code="print(1)", earlier=_earlier(earlier), snapshots=(snap,))
    started = time.process_time()
    writes = find_cell_writes(code="print(1)", earlier=_earlier(earlier), snapshots=(snap,))
    assert time.process_time() - started < 0.25
    assert not writes.needs_approval()


def test_a_long_conversation_of_functions_that_rebind_names_is_read_in_well_under_a_second(snap, monkeypatch):
    """The names each earlier cell may rebind where the check can't order
    it, by a `global` in a function or in literal code a function runs, are
    read once per cell and kept with it (`_Cell.rebinds_globally`), not
    again at each check. CPU time, with a parse cache of its own so a busy
    worker's cache doesn't drop the cells; the first call parses and reads
    the cells and isn't timed."""
    from anton.core.utils import write_check

    monkeypatch.setattr(write_check, "_EARLIER_CELLS", write_check._EarlierCells(budget=1_000_000))
    body = "".join(f"    r{j} = df.shape[0] + {j}\n" for j in range(30))
    earlier = [f"import pandas as pd\ndf = pd.DataFrame()\ndef step{i}():\n    global df\n{body}    df = df.head()\n"
               f"    exec({''.join(f't{j} = {j} + {i}' + chr(10) for j in range(150))!r})" for i in range(200)]
    find_cell_writes(code="print(1)", earlier=_earlier(earlier), snapshots=(snap,))
    started = time.process_time()
    writes = find_cell_writes(code="print(1)", earlier=_earlier(earlier), snapshots=(snap,))
    assert time.process_time() - started < 0.1
    assert not writes.needs_approval()


def test_literal_code_a_function_runs_is_parsed_once_though_the_parse_cache_drops_its_cell(snap, monkeypatch):
    """The names literal code a function runs binds are kept by a hash of
    the text (`_LITERAL_BINDS`), apart from the parsed cells, which a busy
    process may not keep (`_EarlierCells`): a later check of a cell the
    parse cache dropped parses none of that literal code again
    (`_Cell.rebinds_globally`)."""
    from anton.core.utils import write_check

    monkeypatch.setattr(write_check, "_EARLIER_CELLS", write_check._EarlierCells(budget=0))
    monkeypatch.setattr(write_check, "_LITERAL_BINDS", write_check._Kept(limit=100))
    literal = "".join(f"t{j} = {j}\n" for j in range(50))
    parsed: list[str] = []
    parse_ran = write_check._parse_ran
    monkeypatch.setattr(write_check, "_parse_ran", lambda text: parsed.append(text) or parse_ran(text))
    earlier = _earlier([f"def step():\n    exec({literal!r})"])
    assert not find_cell_writes(code="print(1)", earlier=earlier, snapshots=(snap,)).needs_approval()
    first = parsed.count(literal)
    for _ in range(2):
        assert not find_cell_writes(code="print(1)", earlier=earlier, snapshots=(snap,)).needs_approval()
    assert parsed.count(literal) == first


def test_two_literal_codes_functions_run_each_keep_the_names_they_bind(snap, monkeypatch):
    """The names each literal code text a function runs binds are kept by a
    hash of that text (`_LITERAL_BINDS`), so a second text reads its own
    names, not the first one's: OUT, which only the second binds, is a name
    code may rebind where the check can't order it."""
    from anton.core.utils import write_check

    monkeypatch.setattr(write_check, "_LITERAL_BINDS", write_check._Kept(limit=100))
    earlier = _earlier(["OUT = 'new_out.csv'", "def a():\n    exec('Q = 1')",
                        "def b():\n    exec(\"OUT = 'stock.csv'\")", "b()"])
    writes = find_cell_writes(code="open(OUT, 'w').write('x')", earlier=earlier, snapshots=(snap,))
    assert [(f.action, f.paths) for f in writes.held_files] == [("overwrite", ())]


def test_a_long_conversation_of_literal_execs_is_read_in_under_two_seconds(snap):
    """What each cell adds to a name's list or dict is indexed once per
    check (`_Reader.growth`), so each literal exec's namespace, read for
    the calls through it, doesn't walk every earlier cell again. The bound
    covers a later turn's check, with the earlier cells already parsed: the
    first call parses each of them once and isn't timed."""
    earlier = [
        f"exec('def go{i}():\\n    return {i}')\ngo{i}()\nns{i} = {{}}\n"
        f"exec('def f():\\n    return 1', ns{i})\nns{i}['f']()\n"
        for i in range(1400)
    ]
    assert not find_cell_writes(code="go1399()", earlier=_earlier(earlier), snapshots=(snap,)).needs_approval()
    started = time.perf_counter()
    writes = find_cell_writes(code="go1399()", earlier=_earlier(earlier), snapshots=(snap,))
    # GitHub's runners take close to a second here, so a 1s bound failed on
    # the slower ones.
    assert time.perf_counter() - started < 2
    assert not writes.needs_approval()


def test_a_large_literal_cell_after_unread_code_is_walked_no_more_than_with_none(snap, monkeypatch):
    """What code the check can't read may have filled is read from the lists
    a cell keeps (`_Cell.stores`, `_Cell.calls`), not by walking the cell
    again, so a cell of literal rows costs about the same after unread code
    as with none."""
    import ast

    rows = "rows = [" + ", ".join(f"{{'id': {i}, 'tags': ['a', 'b']}}" for i in range(300)) + "]\n"
    code = rows + "import pandas as pd\nprint(pd.DataFrame(rows).shape)\n"
    walk = ast.walk
    walks: list[ast.AST] = []

    def counting(node: ast.AST):
        top = node.body[0] if isinstance(node, ast.Module) and node.body else node
        if isinstance(top, ast.Assign) and isinstance(top.value, ast.List) and len(top.value.elts) == 300:
            walks.append(node)
        return walk(node)

    monkeypatch.setattr(ast, "walk", counting)
    seeds = {"none": [], "a script ran": ["import subprocess, sys\nsubprocess.run([sys.executable, 'job.py'])"]}
    counts = {}
    for seed, earlier in seeds.items():
        walks.clear()
        assert not find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,)).needs_approval()
        counts[seed] = len(walks)
    assert counts["a script ran"] == counts["none"]


def test_a_long_conversation_after_unread_code_reads_few_stores_for_what_it_filled(snap, monkeypatch):
    """A store whose value reads only names read code bound before it,
    through methods that run no code, holds nothing code the check can't
    read made (`_plainly_read`), and a pass after the first reads again only
    stores that read a name the pass before filled more; so a long
    conversation after unread code reads few of its stores for that."""
    from anton.core.utils import write_check

    earlier = ["import pandas as pd\ndf = pd.read_csv('stock.csv')", "SCRIPT = open('helpers.py').read()",
               "exec(SCRIPT)"]
    earlier += [f"part_{i} = df[df.qty > {i}].copy()\nagg_{i} = part_{i}.groupby('sku').agg(total=('qty', 'sum'))\n"
                f"agg_{i}.to_csv('agg_{i}.csv')\nprint(agg_{i}.total.sum())" for i in range(300)]
    read: list[object] = []
    filled_names = write_check._filled_names

    def counting(node, at):
        read.append(node)
        return filled_names(node, at)

    monkeypatch.setattr(write_check, "_filled_names", counting)
    writes = find_cell_writes(code="print(agg_3.total.sum())", earlier=_earlier(earlier), snapshots=(snap,))
    assert not writes.needs_approval()
    assert len(read) < 30


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
    # A parser too shallow for it holds the cell, since the pad may run another Python.
    assert _held(code, snap) is not _parses(code)


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


def test_checks_that_enter_new_contexts_add_nothing_to_what_the_cached_cells_read(snap, monkeypatch):
    """A cached earlier cell keeps what it read of each block for the
    names it binds, keyed by its own withs and calls alone: a later check
    whose own cell enters a context no cell binds reads the same keys, so
    what the cache holds stays the same size from check to check."""
    from anton.core.utils import write_check

    monkeypatch.setattr(write_check, "_EARLIER_CELLS", write_check._EarlierCells(budget=1_000_000))
    earlier = ("exec(open('helpers.py').read())", *(
        f"if {i} % 2:\n    push{i} = print\nelse:\n    push{i} = len\n"
        f"try:\n    w{i} = 1\nexcept Exception:\n    w{i} = 2"
        for i in range(20)
    ))

    def kept() -> int:
        return sum(len(c.cell._first_sure) for c in write_check._EARLIER_CELLS._cells.values() if c.cell is not None)

    sizes = []
    for j in range(5):
        code = f"with ctx{j}:\n    pass\n" + "\n".join(f"push{i}(1)" for i in range(20))
        find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
        sizes.append(kept())
    assert sizes[0] > 0 and len(set(sizes)) == 1


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
    writes = find_cell_writes(code=code, earlier=_earlier(()), snapshots=(snap,))
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
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snapshot,))
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
    writes = find_cell_writes(code=code, earlier=_earlier(()), snapshots=(office,))
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
    writes = find_cell_writes(code=code, earlier=_earlier(()), snapshots=(office,))
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
    writes = find_cell_writes(code="open('stock.csv', 'w').write('y')", earlier=_earlier(earlier, errored=errored), snapshots=(office,))
    assert [(c.action, c.paths, c.held) for c in writes.files] == [("overwrite", ("stock.csv",), held)]


def test_a_name_an_errored_cell_binds_after_its_error_points_no_later_write_at_a_file(office):
    """The errored cell may have stopped before it bound `out` again, so the
    later cell's write may have gone to draft.csv: stock.csv isn't the agent's."""
    earlier = ("out = 'draft.csv'", "x = 1 / 0\nout = 'stock.csv'", "open(out, 'w').write('x')")
    writes = find_cell_writes(code="open('stock.csv', 'w').write('y')", earlier=_earlier(earlier, errored=(False, True, False)), snapshots=(office,))
    assert [(c.action, c.paths, c.held) for c in writes.files] == [("overwrite", ("stock.csv",), True)]


def test_a_write_through_a_name_an_errored_cell_binds_makes_no_file_the_agents(office):
    """The errored cell may have stopped with `p` still on stock.csv, so the
    later cell's write may have gone there rather than to log.txt."""
    earlier = ("p = 'log.txt'", "p = 'stock.csv'\nx = 1 / 0\np = 'log.txt'", "open(p, 'w').write('x')")
    writes = find_cell_writes(code="open('log.txt', 'w').write('y')", earlier=_earlier(earlier, errored=(False, True, False)), snapshots=(office,))
    assert [(c.action, c.paths, c.held) for c in writes.files] == [("overwrite", ("log.txt",), True)]
    built = ("out = 'stock.csv'", "x = 1 / 0\nout += '.bak'", "open(out, 'w').write('x')")
    writes = find_cell_writes(code="open('stock.csv', 'w').write('y')", earlier=_earlier(built, errored=(False, True, False)), snapshots=(office,))
    assert [(c.action, c.paths, c.held) for c in writes.files] == [("overwrite", ("stock.csv",), True)]
    rebound = (*earlier[:2], "p = 'log.txt'\nopen(p, 'w').write('x')")
    writes = find_cell_writes(code="open('log.txt', 'w').write('y')", earlier=_earlier(rebound, errored=(False, True, False)), snapshots=(office,))
    assert [(c.action, c.paths, c.held) for c in writes.files] == [("overwrite", ("log.txt",), False)]


@pytest.mark.parametrize("earlier", [
    ("out = 'stock.csv'", "out = 'new_out.csv'\nx = 1 / 0\nout = 'stock.csv'",
     "import os\nos.remove('stock.csv')\nopen(out, 'w').write('x')"),
    ("from pathlib import Path\np = Path('draft_new.csv')",
     PD + "p = 'draft_new.csv'\ndf['r'] = df['a'] / df['b']",
     "target = 'stock.csv'\ntry:\n    p.rename(target)\nexcept Exception:\n    pass"),
], ids=["delete-then-write", "rename-onto-it"])
def test_a_file_stays_the_users_when_an_errored_cell_may_have_pointed_the_write_elsewhere(earlier, office):
    """The errored cell may have stopped with `out` on new_out.csv, so the
    delete of stock.csv may never have been followed by a write to it; or
    with `p` still a Path, so the rename may never have replaced it. Either
    way stock.csv may still be the user's."""
    writes = find_cell_writes(code="open('stock.csv', 'w').write('y')", earlier=_earlier(earlier, errored=(False, True, False)), snapshots=(office,))
    assert [(c.action, c.paths, c.held) for c in writes.files] == [("overwrite", ("stock.csv",), True)]


def test_a_chdir_through_a_name_an_errored_cell_binds_makes_only_the_file_it_names_the_agents(
    tmp_path, monkeypatch,
):
    """Read for what an earlier write replaced, the folder the chdir reaches
    is the one the errored cell names: a folder the check can't place would
    make every draft.csv in the project the agent's."""
    for rel in ("draft.csv", "archive/2024/draft.csv"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    monkeypatch.chdir(tmp_path)
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    earlier = ("d = '.'\nx = 1 / 0", "import os\nos.chdir(d)\nopen('draft.csv', 'w').write('x')")
    for path, held in (("archive/2024/draft.csv", True), ("draft.csv", False)):
        writes = find_cell_writes(code=f"open({path!r}, 'w').write('y')", earlier=_earlier(earlier, errored=(True, False)), snapshots=(snap,))
        assert [(c.action, c.paths, c.held) for c in writes.files] == [("overwrite", (path,), held)], path


def test_a_name_an_errored_cell_binds_may_still_hold_the_value_before_its_error(office):
    """The errored cell may have stopped before it bound `out` again, so the
    write may go to the user's stock.csv. A later cell that ran to its end
    and bound `out` again leaves no doubt."""
    earlier = ("out = 'stock.csv'\nx = 1 / 0\nout = 'new_out.csv'",)
    code = "open(out, 'w').write('x')"
    writes = find_cell_writes(code=code, earlier=_earlier(earlier, errored=(True,)), snapshots=(office,))
    assert [(c.action, c.paths) for c in writes.held_files] == [("overwrite", ("stock.csv",))]
    writes = find_cell_writes(code=code, earlier=_earlier(earlier, errored=(False,)), snapshots=(office,))
    assert not writes.needs_approval()
    writes = find_cell_writes(code=code, earlier=_earlier((*earlier, "out = 'new_out.csv'"), errored=(True, False)), snapshots=(office,))
    assert not writes.needs_approval()


@pytest.mark.parametrize("earlier", [
    ("cfg = {}\nout = cfg['path']", "out = 'stock.csv'\nx = 1 / 0"),
    ("out = ''", "".join(f"out += {c!r}\n" for c in "stock.csv") + "x = 1 / 0"),
], ids=["older-value-unreadable", "built-on-past-the-limit"])
def test_a_name_an_errored_cell_binds_still_names_the_file_it_names_when_run_to_its_end(earlier, office):
    """What the errored cell may have left in `out` only adds paths: an older
    value the check can't read, or more candidates than it keeps, still
    leaves the stock.csv the cell names when read as having run to its end."""
    writes = find_cell_writes(code="open(out, 'w').write('x')", earlier=_earlier(earlier, errored=(False, True)), snapshots=(office,))
    assert [(c.action, c.paths) for c in writes.held_files] == [("overwrite", ("stock.csv",))]


@pytest.mark.parametrize("earlier, code, action, paths", [
    (("out = 'stock.csv'\nx = 1 / 0\nout = 'log.txt'",), "open(out, 'w').write('x')",
     "overwrite", ("stock.csv", "log.txt")),
    ((PD + "out = 'stock.csv'\nx = 1 / 0\nout = 'new_out.csv'",), "df.to_csv(out)", "overwrite", ("stock.csv",)),
    (("out = 'stock.csv'", "x = 1 / 0\nout = 'new_out.csv'"), "open(out, 'w').write('x')",
     "overwrite", ("stock.csv",)),
    (("from pathlib import Path\nout = 'stock.csv'\nx = 1 / 0\nout = 'new_out.csv'",), "Path(out).write_text('x')",
     "overwrite", ("stock.csv",)),
    (("name = 'stock'\nx = 1 / 0\nname = 'new_out'",), "open(f'{name}.csv', 'w').write('x')",
     "overwrite", ("stock.csv",)),
    (("out = 'stock.csv'", "x = 1 / 0\nout += '.bak'"), "open(out, 'w').write('x')", "overwrite", ("stock.csv",)),
    (("from pathlib import Path\np = Path('stock.csv')\nx = 1 / 0\np = 'notes'",), "p.rename('archive')",
     "rename", ("stock.csv",)),
    (("out = 'stock.csv'\ndef save():\n    open(out, 'w').write('x')", "x = 1 / 0\nout = 'new_out.csv'"), "save()",
     "overwrite", ("stock.csv",)),
], ids=["to-another-existing-file", "pandas", "bound-in-a-cell-before", "pathlib", "f-string", "built-on",
        "path-rebound-to-text", "read-in-an-earlier-function"])
def test_a_write_through_a_name_an_errored_cell_binds_holds_each_file_it_may_name(
    earlier, code, action, paths, office,
):
    errored = (*(False,) * (len(earlier) - 1), True)
    writes = find_cell_writes(code=code, earlier=_earlier(earlier, errored=errored), snapshots=(office,))
    assert sorted((c.action, p) for c in writes.held_files for p in c.paths) == sorted((action, p) for p in paths)
    ran = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(office,))
    assert "stock.csv" not in {p for c in ran.held_files for p in c.paths}


@pytest.fixture
def beside_shared(tmp_path: Path, monkeypatch):
    """A project in `proj`, beside a shared folder."""
    root = tmp_path / "proj"
    for rel in ("stock.csv", "draft.csv", "data/stock.csv", "data/q3.csv"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x")
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "forecast.xlsx").write_text("x")
    monkeypatch.chdir(root)
    taken_at = time.time() + 60
    return (snapshot_project_files(root=root, label="project", taken_at=taken_at),
            snapshot_project_files(root=tmp_path / "shared", label="shared", taken_at=taken_at))


TEMP_THEN_DATA = ("import tempfile\nwork = tempfile.mkdtemp()",
                  PD + "work = 'data'\ndf['r'] = df['a'] / df['b']")
CHDIR_PAST_THE_LIMIT = ("".join(f"d = 'f{i}'\n" for i in range(17)) + "x = 1 / 0\nd = '.'", "import os\nos.chdir(d)")
CHDIR_DOUBLING = ("d = 'raw'\nx = 1 / 0\nd = 'clean'", *["import os\nos.chdir(d)\nos.chdir('..')"] * 4)
OWNED_THEN_PATH = ("open('stock.csv', 'w').write('z')",
                   "from pathlib import Path\np = Path('draft_new.csv')\nx = 1 / 0\np = 'draft_new.csv'")


@pytest.mark.parametrize("earlier, errored, code, path", [
    (("out = input('file? ')", PD + "out = 'stock.csv'\ndf['r'] = df['a'] / df['b']"), (False, True),
     "df.to_csv(out, index=False)", "stock.csv"),
    (("import sys\nout = sys.argv[1]", "out = 'stock.csv'\nx = 1 / 0"), (False, True),
     "def save():\n    open(out, 'w').write('x')\nsave()", "stock.csv"),
    (("cfg = {}\nout = cfg['path']", "out = 'stock.csv'\nx = 1 / 0"), (False, True), "import os\nos.remove(out)",
     "stock.csv"),
    (("cfg = {}\nsrc = cfg['path']", "src = 'stock.csv'\nx = 1 / 0"), (False, True),
     "import shutil\nshutil.copy(src, 'data')", "data/stock.csv"),
    (("import os\nout = os.environ['OUT']", "from pathlib import Path\nout = Path('stock.csv')\nx = 1 / 0"),
     (False, True), "out.write_text('x')", "stock.csv"),
    (TEMP_THEN_DATA, (False, True), "df.to_csv(f'{work}/stock.csv', index=False)", "data/stock.csv"),
    (TEMP_THEN_DATA, (False, True), "import os\nopen(os.path.join(work, 'stock.csv'), 'w').write('x')",
     "data/stock.csv"),
    (TEMP_THEN_DATA, (False, True), "from pathlib import Path\n(Path(work) / 'stock.csv').write_text('x')",
     "data/stock.csv"),
    (TEMP_THEN_DATA, (False, True), "from pathlib import Path\nPath(work).joinpath('stock.csv').write_text('x')",
     "data/stock.csv"),
    (TEMP_THEN_DATA, (False, True), "open(work + '/stock.csv', 'w').write('x')", "data/stock.csv"),
    (TEMP_THEN_DATA, (False, True), "import os\nos.remove(os.path.join(work, 'stock.csv'))", "data/stock.csv"),
    (TEMP_THEN_DATA, (False, True), "import os, shutil\nshutil.copy('draft.csv', os.path.join(work, 'stock.csv'))",
     "data/stock.csv"),
    (("import tempfile\nwork = tempfile.mkdtemp()\nwork = 'data'\nx = 1 / 0",), (True,),
     "open(f'{work}/stock.csv', 'w').write('x')", "data/stock.csv"),
    (CHDIR_PAST_THE_LIMIT, (True, False), "open('../proj/stock.csv', 'w').write('x')", "stock.csv"),
    (CHDIR_PAST_THE_LIMIT, (True, False), "open('../shared/forecast.xlsx', 'w').write('x')",
     "shared folder: forecast.xlsx"),
    (CHDIR_PAST_THE_LIMIT, (True, False), "import os\nos.remove('../proj/data/q3.csv')", "data/q3.csv"),
    (CHDIR_PAST_THE_LIMIT, (True, False), "import os\nos.system('rm ../proj/data/q3.csv')", "data/q3.csv"),
    (CHDIR_DOUBLING, (True, False, False, False, False), "open('../proj/stock.csv', 'w').write('x')", "stock.csv"),
    (CHDIR_DOUBLING, (True, False, False, False, False), "open('../shared/forecast.xlsx', 'w').write('x')",
     "shared folder: forecast.xlsx"),
    (OWNED_THEN_PATH, (False, True), "import os\ntarget = 'stock.csv'\nos.remove('stock.csv')\np.rename(target)",
     "stock.csv"),
    (OWNED_THEN_PATH, (False, True), "import os\ntarget = 'stock.csv'\nos.remove('stock.csv')\np.replace(target)",
     "stock.csv"),
], ids=["unreadable-input", "unreadable-argv-in-function", "unreadable-delete", "unreadable-copy-into-folder",
        "unreadable-path-object", "temp-folder-f-string", "temp-folder-join", "temp-folder-path-div",
        "temp-folder-joinpath", "temp-folder-plus", "temp-folder-delete", "temp-folder-copy-dest",
        "temp-folder-same-cell", "chdir-past-the-limit-back-in", "chdir-past-the-limit-shared",
        "chdir-past-the-limit-delete", "chdir-past-the-limit-shell", "chdir-doubling-back-in",
        "chdir-doubling-shared", "delete-then-maybe-path-rename", "delete-then-maybe-path-replace"])
def test_what_an_errored_cell_may_have_bound_never_frees_a_file_its_run_to_end_reading_holds(
    earlier, errored, code, path, beside_shared,
):
    """The cell is read as before, every earlier cell run to its end, and
    read again only for the files that wider reading adds: an older value the
    check can't read, a temporary folder, more working folders than it keeps
    or a name that may be a path never drop a file the first reading holds."""
    writes = find_cell_writes(code=code, earlier=_earlier(earlier, errored=errored), snapshots=beside_shared)
    assert path in {p for c in writes.held_files for p in c.paths}


def test_a_name_read_inside_an_errored_cell_holds_what_that_cell_bound_before_the_read(beside_shared):
    """The line that builds `out` ran, so the line before it that bound
    `base` ran too: `out` may name tmp_out/stock.csv, or nothing, but never
    the data/stock.csv the older `base` would give."""
    earlier = ("base = 'data'", "base = 'tmp_out'\nout = base + '/stock.csv'\nx = 1 / 0")
    writes = find_cell_writes(code="open(out, 'w').write('x')", earlier=_earlier(earlier, errored=(False, True)), snapshots=beside_shared)
    assert not writes.needs_approval()


@pytest.mark.parametrize("stopped, code, expected", [
    ("out = 'stock.csv'\nx = 1 / 0\nout = 'new_out.csv'", "open(out, 'w').write('x')", [("overwrite", "stock.csv")]),
    ("out = 'stock.csv'\nx = 1 / 0\nout = 'new_out.csv'", "import os\nos.remove(out)", [("delete", "stock.csv")]),
    ("out = 'draft.csv'\nx = 1 / 0\nout = 'stock.csv'", "import os\nos.remove(out)",
     [("delete", "draft.csv"), ("delete", "stock.csv")]),
], ids=["overwrite", "delete", "delete-the-file-before-the-error"])
def test_an_older_value_the_check_cant_read_leaves_the_files_an_errored_cell_names(
    stopped, code, expected, beside_shared,
):
    """`out` may still hold the value from `cfg`, which the check can't
    read, or either value the errored cell gives it: each file those two
    name is held."""
    writes = find_cell_writes(code=code, earlier=_earlier(("cfg = {}\nout = cfg['path']", stopped), errored=(False, True)),
                              snapshots=beside_shared)
    assert sorted((c.action, p) for c in writes.held_files for p in c.paths) == expected


def test_a_folder_the_wide_reading_cant_place_is_held_beside_the_whole_project(beside_shared):
    """Read run to its end, the errored cell leaves the pad in the project,
    so rmtree('.') deletes all of it. It may have stopped with `d` naming any
    of 18 folders, more than the check keeps, so the folder deleted is also
    one the code picks when it runs."""
    writes = find_cell_writes(code="import shutil\nshutil.rmtree('.')", earlier=_earlier(CHDIR_PAST_THE_LIMIT, errored=(True, False)),
                              snapshots=beside_shared)
    assert sorted((c.whole_project, c.paths) for c in writes.held_files) == [(False, ()), (True, ())]


@pytest.mark.parametrize("retry, code, held, stopped", [
    ("folder = 'tmp_{i}'\nname = 'part_{i}'\next = 'tsv{i}'", "df.to_csv(p, index=False)",
     [("overwrite", ("data/stock.csv",))], True),
    ("folder = 'tmp_{i}'\nname = 'part_{i}'\next = 'tsv{i}'", "import os\nos.remove(p)",
     [("delete", ("data/stock.csv",))], True),
    ("folder = 'data'\nname = 'q3'\next = 'csv'", "df.to_csv(p, index=False)",
     [("overwrite", ("data/q3.csv",)), ("overwrite", ("data/stock.csv",))], False),
], ids=["new-values", "new-values-delete", "same-values"])
def test_a_run_of_failed_retries_is_read_in_well_under_a_second(retry, code, held, stopped, beside_shared, monkeypatch):
    """Each failed retry may have stopped before or after it gave `folder`,
    `name` and `ext` the values the plain reading reads, so the wide reading
    gives each name every value. A value counts once, so twelve retries of
    the same values still hold data/q3.csv too. Twelve new values for each
    name pass the path limit, and the wide reading gives up on the path the
    three build: it is never one path per pick of values. The plain
    reading's data/stock.csv stays held, alone, and the check stops, since
    a path it gave up on may be a file the turn began with."""
    from anton.core.utils import write_check

    largest = 0
    paths = write_check._Reader.paths

    def measuring(*args, **kwargs):
        nonlocal largest
        got = paths(*args, **kwargs)
        largest = max(largest, 0 if isinstance(got, write_check._Unknown) else len(got))
        return got

    monkeypatch.setattr(write_check._Reader, "paths", measuring)
    plain = "folder = 'data'\nname = 'stock'\next = 'csv'"
    retries = [PD + retry.format(i=i) + "\ndf['r'] = df['a'] / df['b']\n" + plain for i in range(12)]
    earlier = (PD + plain, *retries, "p = f'{folder}/{name}.{ext}'")
    started = time.perf_counter()
    writes = find_cell_writes(code=code, earlier=_earlier(earlier, errored=(False, *(True,) * len(retries), False)), snapshots=beside_shared)
    assert time.perf_counter() - started < 1
    assert sorted((c.action, c.paths) for c in writes.held_files) == held
    assert writes.stopped is stopped
    assert largest < 50


@pytest.mark.parametrize("c0, step, levels, retries, code", [
    ("'output'", "c{j} = c{p} + '/x'", 4, 80, "open(c4 + '/old.png', 'w').write('x')"),
    ("'stock.csv'", "c{j} = c{p}", 6, 12, "open(c6, 'w').write('x')"),
], ids=["grown", "aliased"])
def test_a_long_chain_of_failed_retries_is_read_in_well_under_a_second(c0, step, levels, retries, code, snap):
    """Each failed retry rebinds every name of a chain that builds the path,
    so the wide reading could read each binding again for every way down the
    chain. It reads each name once per read site and gives up past its
    budget, which keeps the plain reading's findings."""
    earlier = [f"c0 = {c0}"]
    for _ in range(retries):
        earlier += [step.format(j=j, p=j - 1) + "\nx = 1 / 0" for j in range(1, levels + 1)]
    started = time.perf_counter()
    writes = find_cell_writes(code=code, earlier=_earlier(earlier, errored=(False, *(True,) * (len(earlier) - 1))), snapshots=(snap,))
    assert time.perf_counter() - started < 1
    if c0 == "'stock.csv'":
        assert [c.paths for c in writes.held_files] == [("stock.csv",)]


def test_an_errored_history_is_read_again_only_for_a_file_an_earlier_cell_wrote(office, monkeypatch):
    """With an errored earlier cell, the earlier cells are read a second time,
    with it as never run, only to confirm a file the first read found written."""
    from anton.core.utils import write_check

    reads = 0
    writes_read_from = write_check._OwnedFiles._writes_read_from

    def counting(*args, **kwargs):
        nonlocal reads
        reads += 1
        return writes_read_from(*args, **kwargs)

    monkeypatch.setattr(write_check._OwnedFiles, "_writes_read_from", counting)
    earlier = ("open('log.txt', 'w').write('z')", "x = 1 / 0")
    writes = find_cell_writes(code="open('stock.csv', 'w').write('y')", earlier=_earlier(earlier, errored=(False, True)), snapshots=(office,))
    assert [(c.paths, c.held) for c in writes.files] == [(("stock.csv",), True)]
    assert reads == 1
    reads = 0
    writes = find_cell_writes(code="open('log.txt', 'w').write('y')", earlier=_earlier(earlier, errored=(False, True)), snapshots=(office,))
    assert [(c.paths, c.held) for c in writes.files] == [(("log.txt",), False)]
    assert reads == 2


def test_an_errored_cell_lines_up_with_its_code_past_one_that_does_not_parse(office):
    """A cell that doesn't parse is dropped, and the flags still name the right cells."""
    earlier = ("def (", "open('log.txt', 'w').write('z')", "x = 1 / 0\nopen('stock.csv', 'w').write('z')")
    code = "open('log.txt', 'w').write('y')\nopen('stock.csv', 'w').write('y')"
    writes = find_cell_writes(code=code, earlier=_earlier(earlier, errored=(False, False, True)), snapshots=(office,))
    assert [(c.paths, c.held) for c in writes.files] == [(("log.txt",), False), (("stock.csv",), True)]


async def test_the_gate_passes_each_earlier_cells_error_to_the_check(office, monkeypatch):
    """Compacting a failed cell cannot turn its unexecuted write into ownership."""
    asked: list[CellWrites] = []

    async def cancel(_session, writes, _code):
        asked.append(writes)
        return "cancelled"

    monkeypatch.setattr(scratchpad, "_ask_approval", cancel)
    session = SimpleNamespace(_write_gate=scratchpad.WriteGate(snapshots=(office,)))
    wrote = "x = 1 / 0\nopen('stock.csv', 'w').write('z')"
    failed = Cell(code=wrote, stdout="", stderr="", error="ZeroDivisionError: division by zero")
    rewrite = "open('stock.csv', 'w').write('y')"
    pad = SimpleNamespace(compacted_cells=(), cells=[failed])
    assert await scratchpad.hold_unapproved_write(session, pad=pad, code=rewrite) is not None
    assert [f.paths for f in asked[0].held_files] == [("stock.csv",)]
    compacted = SimpleNamespace(compacted_cells=(CompactedCell(code=wrote, errored=True),), cells=[])
    assert await scratchpad.hold_unapproved_write(session, pad=compacted, code=rewrite) is not None
    assert [f.paths for f in asked[1].held_files] == [("stock.csv",)]
    ran = SimpleNamespace(
        compacted_cells=(CompactedCell(code="open('stock.csv', 'w').write('z')", errored=False),),
        cells=[failed],
    )
    assert await scratchpad.hold_unapproved_write(session, pad=ran, code=rewrite) is None
    assert len(asked) == 2


async def test_the_gate_holds_the_first_cell_after_the_pad_starts_a_new_process_once_a_hook_was_written(
    office, monkeypatch,
):
    """A cell whose process ended has the pad start a new one before the
    next cell, and that start runs a startup hook an earlier cell wrote, so
    the gate passes `ended_process` on and the next cell is held, from the
    live cells and the compacted ones alike; a cell that only raised starts
    nothing new."""
    asked: list[CellWrites] = []

    async def cancel(_session, writes, _code):
        asked.append(writes)
        return "cancelled"

    monkeypatch.setattr(scratchpad, "_ask_approval", cancel)
    session = SimpleNamespace(_write_gate=scratchpad.WriteGate(snapshots=(office,)))
    hook = Cell(code="import site\nopen(site.getsitepackages()[0] + '/hook.pth', 'w').write('import job')",
                stdout="", stderr="", error=None)
    died = Cell(code="import os\nos._exit(1)", stdout="", stderr="", error="Process exited unexpectedly.",
                stopped_at=1, ended_process=True)
    raised = Cell(code="int('a')", stdout="", stderr="", error="ValueError", stopped_at=1)
    pad = SimpleNamespace(compacted_cells=(), cells=[hook, died])
    assert await scratchpad.hold_unapproved_write(session, pad=pad, code="print(1)") is not None
    assert asked[0].unchecked
    assert [c.kind for c in asked[0].causes] == ["restart"]
    compacted = SimpleNamespace(compacted_cells=(
        CompactedCell(code=hook.code, errored=False),
        CompactedCell(code=died.code, errored=True, stopped_at=1, ended_process=True),
    ), cells=[])
    assert await scratchpad.hold_unapproved_write(session, pad=compacted, code="print(1)") is not None
    assert asked[1].unchecked
    kept = SimpleNamespace(compacted_cells=(), cells=[hook, raised])
    assert await scratchpad.hold_unapproved_write(session, pad=kept, code="print(1)") is None
    assert len(asked) == 2


@pytest.mark.parametrize("unchecked", ["own", "off"])
async def test_own_and_off_let_the_first_cell_after_a_restart_run_and_show_a_held_one_whole(
    office, monkeypatch, unchecked,
):
    """The restart hold is no code the cell itself runs, so `own` drops it,
    as `off` does. A cell that holds for a write it makes shows the whole
    card, the restart's reason with it."""
    asked: list[str] = []

    async def cancel(_session, writes, code):
        asked.append(scratchpad._card_prompt(writes, code))
        return "cancelled"

    monkeypatch.setattr(scratchpad, "_ask_approval", cancel)
    session = SimpleNamespace(_write_gate=scratchpad.WriteGate(snapshots=(office,), unchecked=unchecked))
    hook = Cell(code="import site\nopen(site.getsitepackages()[0] + '/hook.pth', 'w').write('import job')",
                stdout="", stderr="", error=None)
    died = Cell(code="import os\nos._exit(1)", stdout="", stderr="", error="Process exited unexpectedly.",
                stopped_at=1, ended_process=True)
    pad = SimpleNamespace(compacted_cells=(), cells=[hook, died])
    assert await scratchpad.hold_unapproved_write(session, pad=pad, code="print(1)") is None
    assert asked == []
    assert await scratchpad.hold_unapproved_write(session, pad=pad, code="import os\nos.remove('log.txt')") is not None
    assert "This step would delete:" in asked[0]
    assert "The scratchpad starts again before this step" in asked[0]


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

@pytest.mark.parametrize("code, path, statements", [
    ("import sqlite3\nc = sqlite3.connect('warehouse.db')\nc.execute('DELETE FROM orders WHERE qty = 0')\nc.commit()",
     "warehouse.db", [("DELETE FROM orders WHERE qty = 0", "")]),
    ("import sqlite3\nwith sqlite3.connect('warehouse.db') as c:\n    c.cursor().execute('UPDATE t SET a = 1')",
     "warehouse.db", [("UPDATE t SET a = 1", "")]),
    ("import sqlite3\ncon = sqlite3.connect(database='warehouse.db')\ncur = con.cursor()\n"
     "cur.executemany('INSERT INTO t VALUES (?)', rows)", "warehouse.db", [("INSERT INTO t VALUES (?)", "{rows}")]),
    ("import sqlite3\nc = sqlite3.connect('warehouse.db')\nc.executescript('DROP TABLE t; CREATE TABLE t (a int);')",
     "warehouse.db", [("DROP TABLE t", ""), ("CREATE TABLE t (a int)", "")]),
    ("import duckdb\ncon = duckdb.connect('analytics.duckdb')\ncon.execute(\"CREATE TABLE s AS SELECT * FROM 'stock.csv'\")",
     "analytics.duckdb", [("CREATE TABLE s AS SELECT * FROM 'stock.csv'", "")]),
    ("import duckdb\ncon = duckdb.connect('analytics.duckdb')\ncon.sql('DELETE FROM s WHERE qty < 0')",
     "analytics.duckdb", [("DELETE FROM s WHERE qty < 0", "")]),
    ("import duckdb\nq = 'UPDATE s SET qty = 0'\nduckdb.connect('analytics.duckdb').execute(q)", "analytics.duckdb",
     [("UPDATE s SET qty = 0", "")]),
    ("import duckdb\ncon = duckdb.connect('analytics.duckdb')\ndef run(s):\n    con.execute(s)\nrun('DELETE FROM s')",
     "analytics.duckdb", [("DELETE FROM s", "")]),
    ("import pandas as pd, sqlite3\nc = sqlite3.connect('warehouse.db')\ndf.to_sql('stock', c, if_exists='replace')",
     "warehouse.db", [("to_sql into stock", "")]),
    ("from sqlalchemy import create_engine, text\ne = create_engine('sqlite:///warehouse.db')\n"
     "with e.begin() as c:\n    c.execute(text('DELETE FROM orders'))", "warehouse.db", [("DELETE FROM orders", "")]),
], ids=["sqlite-delete", "with-cursor", "keyword-executemany", "executescript", "duckdb-create", "duckdb-sql",
        "statement-in-a-name", "local-helper", "to_sql", "sqlalchemy-sqlite-url"])
def test_a_write_to_a_database_file_the_turn_began_with_is_held_as_its_overwrite(code, path, statements, office):
    """The overwrite carries each statement it runs, with its values, so the
    card can quote them beside the file."""
    writes = find_cell_writes(code=code, earlier=_earlier(()), snapshots=(office,))
    assert [(c.action, c.paths) for c in writes.held_files] == [("overwrite", (path,))]
    assert [(w.statement, w.values) for w in writes.held_files[0].statements] == statements
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
    assert not find_cell_writes(code=code, earlier=_earlier(()), snapshots=(office,)).needs_approval()


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


SHIP = "import sqlite3\ncon = sqlite3.connect('warehouse.db')\n"


@pytest.mark.parametrize("code, fenced", [
    (SHIP + "con.execute(\"UPDATE orders SET status = 'shipped' WHERE id = 42\")",
     ["warehouse.db", "UPDATE orders SET status = 'shipped' WHERE id = 42"]),
    (SHIP + "con.executemany('INSERT INTO orders VALUES (?, ?)', [(1, 'a'), (2, 'b')])",
     ["warehouse.db", "INSERT INTO orders VALUES (?, ?)\nwith values [(1, 'a'), (2, 'b')]"]),
    ("import duckdb\nduckdb.connect('analytics.duckdb').execute('DELETE FROM sales WHERE qty = 0')",
     ["analytics.duckdb", "DELETE FROM sales WHERE qty = 0"]),
    ("import pandas as pd\n" + SHIP + "pd.DataFrame({'a': [1, 2]}).to_sql('orders_copy', con)",
     ["warehouse.db", "to_sql into orders_copy"]),
], ids=["update", "executemany", "duckdb-delete", "to_sql"])
def test_the_card_names_the_database_file_and_quotes_each_statement_with_its_values(code, fenced, office):
    """The user can tell a one-row fix from an emptied table before pressing
    Approve: the card shows what runs on the file, not only its name."""
    card = _card(code, office)
    assert "This step would change this database file:" in card
    assert "It would run:" in card
    assert _fenced(card) == fenced


def test_every_write_to_one_database_file_is_on_its_one_change(office):
    """One Approve runs both statements, so the card quotes both, under the
    file named once."""
    code = SHIP + "con.execute('DELETE FROM orders WHERE id = 1')\ncon.execute(\"UPDATE orders SET status = 'x' WHERE id = 2\")"
    writes = find_cell_writes(code=code, earlier=_earlier(()), snapshots=(office,))
    assert [(c.paths, len(c.statements)) for c in writes.held_files] == [(("warehouse.db",), 2)]
    assert _fenced(_card_prompt(writes)) == [
        "warehouse.db", "DELETE FROM orders WHERE id = 1", "UPDATE orders SET status = 'x' WHERE id = 2",
    ]


@pytest.mark.parametrize("code", [
    SHIP + "con.execute('DELETE FROM orders')\nimport shutil\nshutil.copy('backup.db', 'warehouse.db')",
    "import shutil\nshutil.copy('backup.db', 'warehouse.db')\n" + SHIP + "con.execute('DELETE FROM orders')",
], ids=["copy-after", "copy-before"])
def test_a_copy_onto_a_database_file_stays_on_the_card_beside_its_statements(code, office):
    """The copy replaces the whole file, which a DELETE quoted under "would
    change this database file" doesn't say, so each keeps its own line."""
    card = _card(code, office)
    assert "This step would overwrite:\n\n```\nwarehouse.db\n```" in card
    assert "It would run:\n\n```\nDELETE FROM orders\n```" in card


def test_a_database_file_the_agent_made_shows_its_statement_when_another_change_raises_the_card(databases):
    """A write to a database an earlier turn made raises no card, but the
    Approve another change asks for runs it too, so the card quotes it."""
    earlier = "import sqlite3\ncon = sqlite3.connect('scratch.db'); con.execute('CREATE TABLE t (a int)')"
    card = _card(SQLITE_AGAIN + "\nimport os\nos.remove('stock.csv')", databases, earlier=(earlier,))
    assert "This step would delete:\n\n```\nstock.csv\n```" in card
    assert "This step would also change this database file:\n\n```\nscratch.db\n```" in card
    assert "It would run:\n\n```\nINSERT INTO t VALUES (1)\n```" in card


@pytest.mark.parametrize("code, fenced", [
    (SHIP + "with con:\n    con.execute('INSERT INTO archive SELECT * FROM orders')\ncon.execute('DELETE FROM orders')",
     ["warehouse.db", "INSERT INTO archive SELECT * FROM orders", "DELETE FROM orders"]),
    (SHIP + "for r in rows:\n    con.execute('INSERT INTO t VALUES (?)', r)\ncon.execute('UPDATE t SET a = 0')",
     ["warehouse.db", "INSERT INTO t VALUES (?)\nwith values {r}", "UPDATE t SET a = 0"]),
], ids=["with-block", "loop"])
def test_a_database_file_s_statements_are_quoted_in_the_order_they_run(code, fenced, office):
    """A write in a block runs before a later one outside it, so the card
    quotes it first: a DELETE shown before the INSERT that archives the rows
    would misstate the step."""
    assert _fenced(_card(code, office)) == fenced


def test_a_secret_in_a_database_file_s_statement_is_scrubbed_on_the_card(office):
    token = "ya29." + "a" * 30
    card = _card(SHIP + f"con.execute(\"UPDATE creds SET note = '{token}' WHERE id = 1\")", office)
    assert token not in card and "[REDACTED_TOKEN]" in card


# ── HTTP ──────────────────────────────────────────────────────────────────

def _requests(code: str, *, earlier=(), rules=()) -> list[tuple[str, str]]:
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(), http_rules=rules)
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
    writes = find_cell_writes(code=code, earlier=_earlier(()), snapshots=())
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
                              earlier=_earlier(()), snapshots=())
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
                              earlier=_earlier(_API_HELPER), snapshots=())
    card = _card_prompt(writes)
    assert "The code decides the request's method when it runs, from `verb`." in card
    assert "{method}" not in card and "`method`" not in card
    assert _fenced(card) == [URL, "api(verb, '/orders/42')"]


def test_an_address_the_code_picks_whole_is_said_in_words():
    """A URL that is nothing but run-time code used to show as a bare
    `{u}` fence, as if it were the address."""
    writes = find_cell_writes(code="import requests\nfor u in urls:\n    requests.delete(u)",
                              earlier=_earlier(()), snapshots=())
    card = _card_prompt(writes)
    assert "This step would send DELETE to an address the code decides when it runs, from `u`:" in card
    assert _fenced(card) == ["requests.delete(u)"]
    assert "{u}" not in card
    assert "A part in braces" not in card


def test_the_refusal_says_a_run_time_method_in_words_too():
    writes = find_cell_writes(code=f"import requests\nrequests.request(row['verb'], {URL!r})",
                              earlier=_earlier(()), snapshots=())
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
                              earlier=_earlier(helper), snapshots=())
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
    writes = find_cell_writes(code=code, earlier=_earlier(()), snapshots=(), http_rules=rules)
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
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=())
    assert [(w.line, w.call, w.url) for w in writes.http] == held
    assert writes.kinds() == (("http",) if held else ())


def test_a_client_base_url_is_joined_to_a_relative_path():
    code = "import httpx\nc = httpx.Client(base_url='https://erp.example.com/api/')\nc.delete('/orders/42')"
    assert _requests(code) == [("DELETE", URL)]


def test_a_request_in_an_earlier_function_is_held_at_the_call():
    earlier = ("import requests\ndef cancel(order_id):\n"
               "    requests.delete(f'https://erp.example.com/api/orders/{order_id}')",)
    writes = find_cell_writes(code="cancel(42)", earlier=_earlier(earlier), snapshots=())
    assert [(w.method, w.url, w.line, w.call) for w in writes.http] == [("DELETE", URL, 1, "cancel(42)")]


def test_a_token_in_the_url_is_scrubbed():
    token = "ya29." + "a" * 30
    writes = find_cell_writes(code=f"import requests\nrequests.delete('https://x.example.com/a?access_token={token}')",
                              earlier=_earlier(()), snapshots=())
    assert token not in writes.http[0].url and token not in _card_prompt(writes)


def test_a_url_with_a_character_that_does_not_show_as_itself_is_named_with_escapes():
    """As a path is: a text direction override in the URL can't reorder what the card shows."""
    writes = find_cell_writes(code="import requests\nrequests.put('https://x.example.com/a\\u202eb')",
                              earlier=_earlier(()), snapshots=())
    card = _card_prompt(writes)
    assert "'https://x.example.com/a\\u202eb'" in card and "A name in quotes is written with escapes" in card
    assert "`'https://x.example.com/a\\u202eb'`" in _refusal(writes, "cancelled").content


# ── HTTP endpoints the operator lists ─────────────────────────────────────

PLANNER = "POST https://planner.example.com/api/*/orders/*"


@pytest.mark.parametrize("url, held", [
    ("'https://planner.example.com/api/v2/orders/9'", True),
    ("'https://PLANNER.example.com/api/v2/orders/9?dry_run=0'", True),
    ("f'https://planner.example.com/api/{version}/orders/{order_id}'", True),
    ("f'{PLANNER_BASE}/orders/{order_id}'", True),
    ("url", True),
    ("'https://planner.example.com/api/v2/orders'", False),
    ("'https://planner.example.com/api/v2/orders/9/lines'", False),
    ("'https://planner.example.com/api/v2/forecasts/9'", False),
    ("f'https://other.example.com/api/{version}/orders/{order_id}'", False),
], ids=["literal", "case-and-query", "run-time-segments", "run-time-base", "run-time-whole-url",
        "too-short", "star-stays-in-one-segment", "other-path", "other-host"])
def test_a_post_to_a_listed_endpoint_is_held(url, held):
    rules = parse_http_write_rules(PLANNER)
    code = f"import requests\nrequests.post({url}, json=order)"
    assert bool(_requests(code, rules=rules)) is held
    assert _requests(code) == [], "with no rule a POST is free"


def test_a_rule_names_its_method_or_any():
    rules = parse_http_write_rules("GET https://planner.example.com/api/**\n* https://hooks.example.com/**")
    assert _requests("import requests\nrequests.get('https://planner.example.com/api/v2/orders/9/lines')",
                     rules=rules) == [("GET", "https://planner.example.com/api/v2/orders/9/lines")]
    assert _requests("import requests\nrequests.post('https://planner.example.com/api/v2/orders')", rules=rules) == []
    assert _requests("import requests\nrequests.head('https://hooks.example.com/x')", rules=rules) == [
        ("HEAD", "https://hooks.example.com/x")
    ]


def test_a_rule_matches_the_url_without_its_query_string():
    rules = parse_http_write_rules("POST https://planner.example.com/api/jobs")
    url = "https://planner.example.com/api/jobs?run=1&next=/home"
    assert _requests(f"import requests\nrequests.post({url!r})", rules=rules) == [("POST", url)]


@pytest.mark.parametrize("method, url, matches", [
    ("POST", "https://planner.example.com/api/v2/orders/9", True),
    ("post", "https://PLANNER.example.com/api/v2/orders/9?dry_run=0#top", True),
    ("POST", "https://planner.example.com/api/v2/orders/9/lines", False),
    ("POST", "https://planner.example.com/api/v2/forecasts/9", False),
    ("GET", "https://planner.example.com/api/v2/orders/9", False),
])
def test_a_forwarded_request_matches_a_rule_as_a_cells_does(method, url, matches):
    """The public matcher a relay uses: a concrete request, the gate's rule."""
    from anton.core.utils.write_check import http_write_rule_matches

    (rule,) = parse_http_write_rules(PLANNER)
    assert http_write_rule_matches(rule, method=method, url=url) is matches
    (any_method,) = parse_http_write_rules("* https://planner.example.com/**")
    assert http_write_rule_matches(any_method, method="DELETE", url=url) is True


def test_http_write_rules_parse_entries_and_skip_bad_ones_with_a_warning(caplog):
    text = f"{PLANNER}\n  , delete https://x.example.com/* ,\nhttps://no-method.example.com/*\nPUTT https://x/*\nGET a b"
    with caplog.at_level(logging.WARNING, logger="anton.core.utils.write_check"):
        rules = parse_http_write_rules(text)
    assert rules == (
        HttpWriteRule(method="POST", url_glob="https://planner.example.com/api/*/orders/*"),
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
    writes = find_cell_writes(code=code, earlier=_earlier(()), snapshots=(snap,))
    assert writes.needs_approval()
    assert "delete" in {c.action for c in writes.held_files}


def test_a_delete_counts_as_an_overwrite_only_when_a_write_surely_hits_the_file(snap):
    """A later write through a name an errored cell binds may go to stock.csv
    or to new_out.csv, so it doesn't make the delete of the agent's own
    stock.csv an overwrite of it."""
    earlier = ("open('stock.csv', 'w').write('z')", "out = 'stock.csv'\nx = 1 / 0\nout = 'new_out.csv'")
    for target, held in (("out", [("delete", ("stock.csv",))]), ("'stock.csv'", [])):
        writes = find_cell_writes(code=f"import os\nos.remove('stock.csv')\nopen({target}, 'w').write('x')",
                                  earlier=_earlier(earlier, errored=(False, True)), snapshots=(snap,))
        assert [(c.action, c.paths) for c in writes.held_files] == held, target


def test_cards_name_the_files(snap):
    writes = find_cell_writes(code="import glob, os\nfor f in glob.glob('*.pdf'):\n    os.remove(f)",
                              earlier=_earlier(()), snapshots=(snap,))
    assert writes.held_files[0].paths == ("report.pdf",), "a root glob doesn't reach docs/"
    picked = find_cell_writes(code="import os\nos.remove(os.environ['X'])",
                              earlier=_earlier(()), snapshots=(snap,))
    assert picked.held_files[0].paths == () and "os.environ['X']" in picked.held_files[0].call
    whole = find_cell_writes(code="import shutil\nshutil.rmtree('.')",
                             earlier=_earlier(()), snapshots=(snap,))
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
], ids=["path-join", "text-add", "folder-join"])
def test_an_addition_before_a_name_is_bound_again_does_not_change_it(earlier, code, expected, snap):
    assert _files(code, (snap,), earlier=(earlier,)) == [("delete", expected)]
    same_cell = "import os\np = 'stock'\np += '.csv'\nprint(open(p).read())\np = 'report.pdf'\nos.remove(p)"
    assert "report.pdf" in _files(same_cell, (snap,))[0][1]


# TARGET bound at module level, and a function that rebinds it as a global.
_RETARGETS = "TARGET = 'scratch.tmp'\ndef retarget():\n    global TARGET\n    TARGET = 'stock.csv'"


@pytest.mark.parametrize("earlier, code", [
    ([_RETARGETS, "retarget()"], "import os\nos.remove(TARGET)"),
    ([_RETARGETS + "\nretarget()"], "import os\nos.remove(TARGET)"),
    ([_RETARGETS], "import os\nos.remove(TARGET)"),
    ([_RETARGETS, "retarget()"], "import shutil\nshutil.move(TARGET, 'kept.csv')"),
    ([_RETARGETS, "retarget()"], "import os\ndef wipe():\n    os.remove(TARGET)\nwipe()"),
    (["from pathlib import Path\nout = Path('report.pdf')\ndef deeper():\n    global out\n    out /= 'charts'"],
     "out.unlink()"),
    (["def deeper():\n    global out\n    out /= 'charts'"],
     "from pathlib import Path\nout = Path('report.pdf')\nout.unlink()"),
    (["TARGET = 'scratch.tmp'\ndef retarget(x):\n    global TARGET\n    match x:\n        case TARGET:\n            pass",
      "retarget('stock.csv')"], "import os\nos.remove(TARGET)"),
    (["TARGET = 'scratch.tmp'\ndef retarget():\n    global TARGET\n    TARGET = 'other.tmp'", "retarget()"],
     "import os\nos.remove(os.path.join('docs', TARGET))"),
], ids=["delete-after-a-call", "delete-after-a-call-in-the-same-cell", "delete-where-no-call-is-seen", "move",
        "delete-in-a-function", "join-in-a-function", "join-in-a-function-before-the-name-is-bound",
        "rebound-by-a-match-capture", "delete-of-a-file-joined-to-a-folder-where-each-value-seen-is-new"])
def test_a_path_through_a_name_a_function_rebinds_as_a_global_is_one_the_check_cant_place(earlier, code, snap):
    """A call may rebind such a name at module level whenever it runs, which
    the check can't order against the read, so it never uses the name's
    module-level value: a delete or move of it gets the card a target the
    check can't place gets, as its twin through a name bound to input does."""
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    twin = find_cell_writes(code=code, earlier=_earlier([*earlier, "TARGET = input()\nout = input()"]),
                            snapshots=(snap,))
    assert [(f.action, f.paths) for f in writes.held_files] == [(twin.held_files[0].action, ())]
    assert [(f.action, f.paths) for f in twin.held_files] == [(writes.held_files[0].action, ())]


# A folder bound at module level, and a function that rebinds it as a global.
_REFOLDERS = "OUT = 'docs'\ndef refolder():\n    global OUT\n    OUT = 'output'"
# A file name bound at module level, one in the folder docs, and a function
# that rebinds it as a global.
_RENAMES = "NAME = 'a.pdf'\nDIR = 'docs'\ndef rename_to(n):\n    global NAME\n    NAME = n"
# A new file bound at module level, and a function that rebinds it as a
# global to docs' file a.pdf.
_RESETS = "NAME = 'new.csv'\ndef reset():\n    global NAME\n    NAME = 'a.pdf'"
# A new file bound at module level, and a function that rebinds it as a
# global to what each call passes.
_AIMS = "TARGET = 'new.csv'\ndef aim(x):\n    global TARGET\n    TARGET = x"
# A function that sets an output folder as a global from what each call passes.
_CONFIGURES = ("from pathlib import Path\nimport pandas as pd\ndef configure(run):\n    global OUT_DIR\n"
               "    OUT_DIR = Path('runs') / run")


@pytest.mark.parametrize("earlier, code", [
    ([_RETARGETS, "retarget()"], "open(TARGET, 'w').write('x')"),
    (["TARGET = 'stock.csv'", "def retarget():\n    global TARGET\n    TARGET = 'scratch.tmp'"],
     "import pandas as pd\npd.DataFrame().to_csv(TARGET)"),
    ([_RETARGETS], "TARGET = 'stock.csv'\nopen(TARGET, 'w').write('x')"),
    ([_RETARGETS], "import pandas as pd\nTARGET = 'stock.csv'\npd.DataFrame().to_csv(TARGET)"),
    ([_RETARGETS], "from pathlib import Path\nTARGET = 'stock.csv'\nPath(TARGET).write_text('x')"),
    ([_RETARGETS], "import shutil\nTARGET = 'stock.csv'\nshutil.copy('summary.md', TARGET)"),
    ([_REFOLDERS], "import os\nopen(os.path.join(OUT, 'a.pdf'), 'w').write('x')"),
    ([_REFOLDERS], "from pathlib import Path\n(Path(OUT) / 'a.pdf').write_text('x')"),
    ([_REFOLDERS], "open(f'{OUT}/a.pdf', 'w').write('x')"),
    # The rebound name as the file in a folder: a part the code picks when
    # it runs, which an overwrite would otherwise drop as a new file's.
    ([_RENAMES], "import os\nopen(os.path.join('docs', NAME), 'w').write('x')"),
    ([_RENAMES, "rename_to('b.pdf')"], "import os\nopen(os.path.join('docs', NAME), 'w').write('x')"),
    ([_RENAMES, "rename_to('b.pdf')"], "import os\nopen(os.path.join(DIR, NAME), 'w').write('x')"),
    ([_RENAMES], "from pathlib import Path\n(Path('docs') / NAME).write_text('x')"),
    ([_RENAMES], "open(f'docs/{NAME}', 'w').write('x')"),
    ([_RENAMES], "open('docs/' + NAME, 'w').write('x')"),
    ([_RENAMES], "import os\nimport pandas as pd\npd.DataFrame().to_csv(os.path.join('docs', NAME))"),
    ([_RENAMES], "import pandas as pd\nfrom pathlib import Path\npd.DataFrame().to_excel(Path('docs') / NAME)"),
    ([_RENAMES, "rename_to('b.pdf')"], "import os, shutil\nshutil.copy('summary.md', os.path.join('docs', NAME))"),
    ([_RENAMES], "import os\nfor p in [os.path.join('docs', NAME), 'docs/new.pdf']:\n    open(p, 'w').write('x')"),
    ([_RENAMES], "import os, random\np = 'docs/new.pdf'\nif random.random() > 0.5:\n    p = os.path.join('docs', NAME)\n"
     "open(p, 'w').write('x')"),
    ([_AIMS, "[aim(x) for x in ['stock.csv']]"], "open(TARGET, 'w').write('x')"),
    ([_AIMS, "def main():\n    aim('stock.csv')\nmain()"], "open(TARGET, 'w').write('x')"),
    ([_AIMS], "open(TARGET, 'w').write('x')"),
    (["BASE = 'output'\ndef aim(f):\n    global TARGET\n    TARGET = BASE + '/' + f", "aim('old.png')"],
     "open(TARGET, 'w').write('x')"),
    (["TARGET = 'stock'\ndef aim():\n    global TARGET\n    TARGET += '.csv'", "aim()"], "open(TARGET, 'w').write('x')"),
    (["TARGET = 'new.csv'\ndef aim():\n    global TARGET\n    TARGET, n = 'stock.csv', 1", "aim()"],
     "open(TARGET, 'w').write('x')"),
    (["TARGET = 'new.csv'\nclass Aim:\n    global TARGET\n    TARGET = 'stock.csv'"], "open(TARGET, 'w').write('x')"),
    (["TARGET = 'new.csv'\ndef aim(x):\n    global TARGET\n    match x:\n        case str() as TARGET:\n            pass",
      "aim('stock.csv')"], "open(TARGET, 'w').write('x')"),
    (["cfg = {'dir': 'docs'}", _RESETS, "reset()"], "import os\nopen(os.path.join(cfg['dir'], NAME), 'w').write('x')"),
    (["cfg = {'dir': 'docs'}", _RESETS, "reset()"], "open(f\"{cfg['dir']}/{NAME}\", 'w').write('x')"),
    (["cfg = {'dir': 'docs'}", _RESETS, "reset()"],
     "import os\nimport pandas as pd\npd.DataFrame().to_csv(os.path.join(cfg['dir'], NAME))"),
    (["cfg = {'dir': 'newdir'}", _RENAMES, "rename_to('b.pdf')"],
     "import os\nopen(os.path.join(cfg['dir'], NAME), 'w').write('x')"),
], ids=["after-a-call", "bound-in-an-earlier-cell", "bound-in-the-cell-just-before", "to-csv-bound-in-the-cell",
        "write-text-bound-in-the-cell", "copy-onto-it-bound-in-the-cell", "folder-joined", "folder-divided",
        "folder-in-an-f-string", "file-joined-to-a-folder-where-no-call-is-seen", "file-joined-to-a-folder-after-a-call",
        "file-joined-to-a-folder-a-name-holds", "file-divided-from-a-folder", "file-in-an-f-string",
        "file-added-to-a-folder", "to-csv-of-a-file-joined-to-a-folder", "to-excel-of-a-file-divided-from-a-folder",
        "copy-onto-a-file-joined-to-a-folder", "a-loop-over-a-file-joined-to-a-folder-and-a-new-file",
        "a-new-file-a-branch-rebinds-to-a-file-joined-to-a-folder", "a-value-a-setter-called-in-a-comprehension-gives",
        "a-value-a-setter-called-from-a-function-gives", "a-value-a-setter-no-call-passes-may-give",
        "a-value-a-setter-builds-on-another-name", "a-setter-that-adds-to-the-name",
        "a-setter-that-unpacks-into-the-name", "a-class-body-that-binds-the-name",
        "a-setter-that-captures-the-name-in-a-match", "a-file-joined-to-a-folder-there-a-dict-entry-names",
        "a-file-in-an-f-string-in-a-folder-there-a-dict-entry-names",
        "to-csv-of-a-file-joined-to-a-folder-there-a-dict-entry-names",
        "a-file-joined-to-a-new-folder-a-dict-entry-names"])
def test_an_overwrite_through_a_name_a_function_rebinds_as_a_global_is_held_whole(earlier, code, snap):
    """Such a name may hold a file the turn began with wherever the check
    reads it, even right after the cell binds it: the overwrite is held as
    one onto a file the code picks when it runs (`_FileFinder.unplaced`),
    under every value of the setting, where its twin through a name bound
    to input counts as a new file. So is one through a path the name is the
    file part of (`_FileFinder.paths`), a dict entry naming the folder or
    not: an entry only ever adds a hold, so it never takes this one off."""
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    assert [(f.action, f.paths) for f in writes.held_files] == [("overwrite", ())]
    assert not writes.unchecked
    twin = code.replace("TARGET = 'stock.csv'\n", "").replace("'stock.csv'", "input()")
    unplaced = find_cell_writes(code=twin, earlier=_earlier(["TARGET = input()\nOUT = input()\nNAME = input()\nDIR = 'docs'"]),
                                snapshots=(snap,))
    assert not unplaced.needs_approval()


# A function that sets OUT as a global to a new file from what each call passes.
_SETS_OUT = "def configure(r):\n    global OUT\n    OUT = f'out_{r}.csv'"


@pytest.mark.parametrize("earlier, code", [
    ([_CONFIGURES, "configure('exp1')"], "pd.DataFrame().to_csv(OUT_DIR / 'results.csv')"),
    ([_CONFIGURES, "configure('exp1')"], "import matplotlib.pyplot as plt\nplt.savefig(OUT_DIR / 'plot.png')"),
    ([_CONFIGURES, "configure('exp1')"], "open(OUT_DIR / 'notes.txt', 'w').write('x')"),
    ([_CONFIGURES, "configure('exp1')"], "import time\npd.DataFrame().to_csv(f'{OUT_DIR}/r_{time.time()}.csv')"),
    ([_CONFIGURES + "\nOUT_DIR = Path('runs/exp1')"], "pd.DataFrame().to_csv(OUT_DIR / 'results.csv')"),
    ([_AIMS, "aim('other_new.csv')"], "open(TARGET, 'w').write('x')"),
    ([_AIMS, "aim('other_new.csv')"], "TARGET = 'brand_new.csv'\nopen(TARGET, 'w').write('x')"),
    ([_RENAMES, "rename_to('b.pdf')"], "import shutil\nshutil.copy('summary.md', os.path.join('output', NAME))"),
    (["OUT = 'new.csv'", _SETS_OUT, "configure('a')", "exec(open('job.py').read())"], "pd.DataFrame().to_csv(OUT)"),
    (["OUT = 'new.csv'", _SETS_OUT, "configure('a')", "def load():\n    exec(open('job.py').read(), globals())\nload()"],
     "pd.DataFrame().to_csv(OUT)"),
    (["OUT = 'new.csv'", _SETS_OUT, "configure('a')", "k = 'OUT'\nglobals()[k] = 'report.pdf'"],
     "pd.DataFrame().to_csv(OUT)"),
    (["OUT = 'new.csv'", _SETS_OUT, "list(map(configure, ['a']))"], "pd.DataFrame().to_csv(OUT)"),
], ids=["to-csv-into-a-folder-a-setter-sets", "savefig-into-a-folder-a-setter-sets",
        "open-in-a-folder-a-setter-sets", "a-dated-file-in-a-folder-a-setter-sets",
        "into-a-folder-a-setter-no-call-runs-may-set", "a-new-file-a-setter-sets", "a-new-file-the-cell-binds",
        "copy-onto-a-new-file-joined-to-a-folder", "a-new-file-a-setter-sets-beside-code-the-check-cant-read",
        "a-new-file-a-setter-sets-beside-code-a-function-runs-in-the-globals",
        "a-new-file-a-setter-sets-beside-a-store-into-globals-by-a-name", "a-new-file-a-setter-map-calls-sets"])
def test_an_overwrite_through_a_name_a_function_rebinds_as_a_global_is_held_whole_where_each_value_seen_is_new(
    earlier, code, snap,
):
    """The check never places such a name on a value it sees for it: code
    it can't read, a store into the module's names it doesn't count, or a
    call it doesn't follow may have rebound the name to a file the turn
    began with. So an overwrite through it is held as one onto a file the
    code picks when it runs, under every value of the setting, even where
    each value the check sees for the name, the module-level ones and those
    a `global` in a function gives it, is a new file."""
    writes = find_cell_writes(code=f"import os\nimport pandas as pd\n{code}", earlier=_earlier(earlier),
                              snapshots=(snap,))
    assert [(f.action, f.paths) for f in writes.held_files] == [("overwrite", ())]


def test_each_overwrite_through_a_name_bound_to_a_rebound_global_is_held_whole(snap):
    """The second read of Q's binding is one the reader remembers
    (`_Reader.recalled`): it still counts the global it read
    (`_Reader.rebound_reads`), so the copy is held as the open is."""
    writes = find_cell_writes(code="open(Q, 'w').write('x')\nimport shutil\nshutil.copy('summary.md', Q)",
                              earlier=_earlier([_RETARGETS, "Q = TARGET"]), snapshots=(snap,))
    assert [(f.action, f.paths) for f in writes.held_files] == [("overwrite", ()), ("overwrite", ())]


def test_a_move_onto_a_name_a_function_rebinds_as_a_global_holds_the_overwrite_too(snap):
    """The move holds its source, and the file it may land on is held as one
    the code picks when it runs, so the card stays whole."""
    writes = find_cell_writes(code="import shutil\nTARGET = 'stock.csv'\nshutil.move('report.pdf', TARGET)",
                              earlier=_earlier([_RETARGETS]), snapshots=(snap,))
    assert [(f.action, f.paths) for f in writes.held_files] == [("move", ("report.pdf",)), ("overwrite", ())]


def test_an_append_through_a_name_a_function_rebinds_as_a_global_is_not_held(snap):
    writes = find_cell_writes(code="open(TARGET, 'a').write('x')", earlier=_earlier([_RETARGETS, "retarget()"]),
                              snapshots=(snap,))
    assert not writes.needs_approval()


@pytest.mark.parametrize("earlier, code", [
    (["TARGET = 'stock.csv'\ndef show():\n    global TARGET\n    print(TARGET)", "show()"],
     "import os\nos.remove(TARGET)"),
    (["import os\ndef retarget():\n    global TARGET\n    TARGET = 'stock.csv'\n    os.remove(TARGET)"],
     "retarget()"),
    (["TARGET = 'stock.csv'\ndef retarget():\n    TARGET = 'scratch.tmp'\n    return TARGET", "retarget()"],
     "import os\nos.remove(TARGET)"),
    (["global TARGET\nTARGET = 'stock.csv'"], "import os\nos.remove(TARGET)"),
    (["TARGET = 'stock.csv'\nclass O:\n    pass\no = O()\nvars(o)['TARGET'] = 'report.pdf'"],
     "import os\nos.remove(TARGET)"),
    (["TARGET = 'stock.csv'\nclass O:\n    pass\no = O()\nsetattr(o, 'TARGET', 'report.pdf')\no.TARGET = 'x'"],
     "import os\nos.remove(TARGET)"),
    (["TARGET = 'stock.csv'\nclass O:\n    pass\no = O()\no.__dict__['TARGET'] = 'report.pdf'"],
     "import os\nos.remove(TARGET)"),
    (["TARGET = 'stock.csv'\nglobals().__setitem__('LABEL', 'TARGET')"], "import os\nos.remove(TARGET)"),
    (["TARGET = 'stock.csv'\nd = {}\ndict.__setitem__(d, 'TARGET', 'report.pdf')"], "import os\nos.remove(TARGET)"),
    (["TARGET = 'stock.csv'\ndef show():\n    print(globals(), 'TARGET')", "show()"], "import os\nos.remove(TARGET)"),
    (["TARGET = 'stock.csv'\nclass C:\n    def globals(self):\n        return {}\nc = C()\nc.globals()['TARGET'] = 'report.pdf'"],
     "import os\nos.remove(TARGET)"),
    (["TARGET = 'stock.csv'\nd = {}\nd |= {'TARGET': 'report.pdf'}"], "import os\nos.remove(TARGET)"),
    (["TARGET = 'stock.csv'\nglobals().update(LABELS={'TARGET': 'report.pdf'})"], "import os\nos.remove(TARGET)"),
], ids=["global-a-function-only-reads", "read-in-the-function-after-it-binds-it", "local-of-the-same-name",
        "global-at-module-level", "a-store-into-vars-of-an-instance", "an-attribute-of-an-instance",
        "a-store-into-the-dict-of-an-instance", "a-setitem-on-globals-whose-value-names-the-name",
        "dict-setitem-handed-another-dict", "a-print-of-globals-beside-the-names-text",
        "a-store-into-what-another-objects-globals-method-gives", "a-merge-into-another-dict",
        "an-update-of-globals-whose-named-keyword-holds-a-dict-naming-the-name"])
def test_a_name_no_code_rebinds_out_of_order_keeps_its_place(earlier, code, snap):
    """A `global` at module level binds nothing a call may change later;
    `vars()` of an instance, its `__dict__` or its attribute holds none of
    the module's names; a store into the module's names sets only the key
    it names, not one its value names; and a store into another dict sets
    none (`_namespace_stores`)."""
    assert _files(code, (snap,), earlier=earlier) == [("delete", ("stock.csv",))]


def test_literal_code_too_deep_to_parse_in_a_function_rebinds_no_name_the_check_reads(snap):
    """The names literal code a function runs binds are read from what it
    parses to (`_Cell.rebinds_globally`): code too deep for the check's
    parser binds none it can name, and the check still reads the cell, a
    stop where the read of that code stopped (`_earlier_stop`)."""
    earlier = _earlier(["TARGET = 'stock.csv'", f"def f():\n    exec({_TOO_DEEP!r})"])
    writes = find_cell_writes(code="import os\nos.remove(TARGET)", earlier=earlier, snapshots=(snap,))
    assert [(f.action, f.paths) for f in writes.held_files] == [("delete", ("stock.csv",))]
    assert [(c.kind, c.in_earlier) for c in writes.causes] == [("limit", True)]


# T bound to a file this turn made, and a function that rebinds it, as a
# global, to one the turn began with.
_REBINDS_T = ["T = 'made_now.csv'", "open(T, 'w').write('x')", "def g0():\n    global T\n    T = 'report.pdf'", "g0()"]


@pytest.mark.parametrize("code, action", [
    ("import os\nos.system(f'rm {T}')", "delete"),
    ("import os\nos.system('rm ' + T)", "delete"),
    ("import subprocess\nsubprocess.run(['rm', T])", "delete"),
    ("import subprocess\nsubprocess.run(f'rm {T}', shell=True)", "delete"),
    ("import subprocess\nsubprocess.run(['mv', T, 'x.pdf'])", "move"),
], ids=["rm-in-an-f-string", "rm-added-to", "rm-in-a-list", "rm-through-a-shell", "mv-in-a-list"])
def test_a_command_through_a_name_a_function_rebinds_as_a_global_holds_whole(code, action, snap):
    """A command's word is read through the same lookup as a path
    (`_Reader.bindings_of`), so it never reads the name's module-level
    value: the rm or mv holds whole, not as a file the turn made."""
    writes = find_cell_writes(code=code, earlier=_earlier(_REBINDS_T), snapshots=(snap,))
    assert [(f.action, f.paths) for f in writes.held_files] == [(action, ())]


@pytest.mark.parametrize("earlier", [
    ["def arm():\n    globals()['OUT'] = 'stock.csv'", "arm()"],
    ["def arm():\n    globals().update(OUT='stock.csv')", "arm()"],
    ["def arm():\n    globals().update({'OUT': 'stock.csv'})", "arm()"],
    ["def arm():\n    exec(\"global OUT\\nOUT = 'stock.csv'\")", "arm()"],
    ["globals()['OUT'] = 'stock.csv'"],
    ["vars()['OUT'] = 'stock.csv'"],
    ["def arm():\n    exec(\"OUT = 'stock.csv'\", globals())", "arm()"],
    ["def arm():\n    g = globals()\n    g['OUT'] = 'stock.csv'", "arm()"],
    ["def arm():\n    globals().__setitem__('OUT', 'stock.csv')", "arm()"],
    ["def arm():\n    globals().update([('OUT', 'stock.csv')])", "arm()"],
    ["def arm():\n    globals().update(dict(OUT='stock.csv'))", "arm()"],
    ["def arm():\n    arm.__globals__['OUT'] = 'stock.csv'", "arm()"],
    ["import sys\ndef arm():\n    sys._getframe().f_globals['OUT'] = 'stock.csv'", "arm()"],
    ["import sys\ndef arm():\n    setattr(sys.modules[__name__], 'OUT', 'stock.csv')", "arm()"],
    ["import __main__\ndef arm():\n    __main__.OUT = 'stock.csv'", "arm()"],
    ["import sys\ndef arm():\n    sys.modules['__main__'].OUT = 'stock.csv'", "arm()"],
    ["import sys\ndef arm():\n    vars(sys.modules['__main__'])['OUT'] = 'stock.csv'", "arm()"],
    ["arm = lambda: globals().__setitem__('OUT', 'stock.csv')", "arm()"],
    ["import timeit\ndef arm():\n    timeit.timeit(stmt=\"OUT = 'stock.csv'\", number=1, globals=globals())", "arm()"],
    ["import sys\nsys._getframe().f_locals['OUT'] = 'stock.csv'"],
    ["locals()['OUT'] = 'stock.csv'"],
    ["import __main__\ndef arm():\n    __main__.__dict__['OUT'] = 'stock.csv'", "arm()"],
    ["import __main__ as m\ndef arm():\n    m.OUT = 'stock.csv'", "arm()"],
    ["import sys\ndef arm():\n    m = sys.modules['__main__']\n    m.OUT = 'stock.csv'", "arm()"],
    ["import sys\ndef arm():\n    sys.modules.get('__main__').OUT = 'stock.csv'", "arm()"],
    ["import inspect\ndef arm():\n    inspect.getmodule(arm).OUT = 'stock.csv'", "arm()"],
    ["def arm():\n    if (g := globals()) is not None:\n        g['OUT'] = 'stock.csv'", "arm()"],
    ["def arm():\n    (g := globals())['OUT'] = 'stock.csv'", "arm()"],
    ["(g := globals())['OUT'] = 'stock.csv'"],
    ["import operator\ndef arm():\n    operator.setitem(globals(), 'OUT', 'stock.csv')", "arm()"],
    ["def arm():\n    dict.__setitem__(globals(), 'OUT', 'stock.csv')", "arm()"],
    ["def arm():\n    dict.update(globals(), OUT='stock.csv')", "arm()"],
    ["def arm():\n    dict.setdefault(globals(), 'OUT', 'stock.csv')", "arm()"],
    ["def arm():\n    (globals() or {})['OUT'] = 'stock.csv'", "arm()"],
    ["G = globals\ndef arm():\n    G()['OUT'] = 'stock.csv'", "arm()"],
    ["V = vars\ndef arm():\n    V()['OUT'] = 'stock.csv'", "arm()"],
    ["L = locals\nL()['OUT'] = 'stock.csv'"],
    ["def arm():\n    g = globals()\n    g |= {'OUT': 'stock.csv'}", "arm()"],
    ["G = globals()\nG |= dict(OUT='stock.csv')"],
    ["(globals() if True else {})['OUT'] = 'stock.csv'"],
    ["globals().update(**{'OUT': 'stock.csv'})"],
    ["import builtins\ndef arm():\n    builtins.globals()['OUT'] = 'stock.csv'", "arm()"],
], ids=["a-store-into-globals-in-a-function", "an-update-of-globals", "an-update-of-globals-from-a-dict",
        "a-global-in-literal-code-a-function-runs", "a-store-into-globals-at-module-level",
        "a-store-into-vars-at-module-level", "literal-code-a-function-runs-in-the-globals",
        "a-store-through-a-name-for-globals", "a-setitem-on-globals", "an-update-of-globals-from-pairs",
        "an-update-of-globals-from-dict-keywords", "a-store-into-a-functions-globals",
        "a-store-into-a-frames-globals", "a-setattr-on-the-module-sys-modules-holds", "an-attribute-of-main",
        "an-attribute-of-the-module-sys-modules-holds", "a-store-into-vars-of-the-module",
        "a-lambda-that-sets-an-item-of-globals", "a-timeit-statement-by-keyword-in-the-globals",
        "a-store-into-the-frames-locals-at-module-level", "a-store-into-locals-at-module-level",
        "a-store-into-the-dict-of-main", "an-attribute-of-main-imported-under-another-name",
        "an-attribute-of-a-name-for-the-module-sys-modules-holds", "an-attribute-of-what-sys-modules-get-gives",
        "an-attribute-of-what-getmodule-gives", "a-store-through-a-walrus-name-for-globals",
        "a-store-into-a-walrus-for-globals-in-place", "a-store-into-a-walrus-for-globals-at-module-level",
        "operator-setitem-on-globals", "dict-setitem-handed-globals", "dict-update-handed-globals",
        "dict-setdefault-handed-globals", "a-store-into-globals-or-another-dict",
        "a-store-into-what-a-name-for-the-globals-builtin-gives", "a-store-into-what-a-name-for-the-vars-builtin-gives",
        "a-store-into-what-a-name-for-the-locals-builtin-gives-at-module-level", "a-merge-into-a-name-for-globals",
        "a-merge-of-dict-keywords-into-a-name-for-globals", "a-store-into-globals-a-conditional-expression-gives",
        "an-update-of-globals-from-a-dict-spread-as-keywords", "a-store-into-what-builtins-globals-gives"])
def test_an_overwrite_through_a_name_a_namespace_store_or_literal_code_rebinds_is_held_whole(earlier, snap):
    """A store into the module's names by a literal key, keyword or
    attribute, however the code reaches them (`_namespace_stores`), and
    literal code a function runs, rebind a name where the reader doesn't
    order it (`_Reader.rebound_globally`), as a global in a function does,
    to a value the reader doesn't read: an overwrite
    through it is held whole, and so is a delete, where the name's
    module-level value names no file the turn began with."""
    writes = find_cell_writes(code="import pandas as pd\npd.DataFrame().to_csv(OUT)",
                              earlier=_earlier(["OUT = 'new_out.csv'", *earlier]), snapshots=(snap,))
    assert [(f.action, f.paths) for f in writes.held_files] == [("overwrite", ())]
    writes = find_cell_writes(code="import os\nos.remove(OUT)", earlier=_earlier(["OUT = 'new_out.csv'", *earlier]),
                              snapshots=(snap,))
    assert [(f.action, f.paths) for f in writes.held_files] == [("delete", ())]


@pytest.mark.parametrize("code", [
    "import os\nos.system(CMD)",
    "import subprocess\nsubprocess.run(CMD, shell=True)",
    "import os\nos.popen(CMD)",
    "import subprocess\nsubprocess.run(ARGS)",
    "import subprocess\nsubprocess.check_output(ARGS)",
], ids=["os-system", "a-shell", "popen", "an-argument-list", "check-output"])
def test_a_whole_command_through_a_name_a_function_rebinds_as_a_global_is_one_the_check_cant_read(code, snap):
    """A command held whole in such a name may be any once a call rebound
    it (`_node_commands`), so the call reads as its twin through a name
    bound to input does: one that runs code the check can't read, never
    the harmless module-level command alone."""
    armed = ["CMD = 'echo hi'\nARGS = ['echo', 'hi']",
             "def arm():\n    global CMD, ARGS\n    CMD = 'rm report.pdf'\n    ARGS = ['rm', 'report.pdf']", "arm()"]
    writes = find_cell_writes(code=code, earlier=_earlier(armed), snapshots=(snap,))
    twin = find_cell_writes(code=code, earlier=_earlier(["CMD = input()\nARGS = input()"]), snapshots=(snap,))
    assert writes.unchecked and writes.unread_here
    assert writes.cause_kinds == twin.cause_kinds == ("runs",)


@pytest.mark.parametrize("bound", ["push = _mv", "push = lambda: _mv()", "def push():\n    _mv()"],
                         ids=["an-alias", "a-lambda", "a-def"])
def test_a_call_through_a_name_a_function_rebinds_as_a_global_still_reads_its_module_level_value(bound, snap):
    """A function may not have run, so a call through such a name still
    reads what its module-level value runs, beside one the check can't
    place: that adds to the card, never takes off it."""
    earlier = ["import os\ndef _mv():\n    os.rename('report.pdf', 'old.pdf')", bound,
               "def g():\n    global push\n    push = print", "g()"]
    writes = find_cell_writes(code="push()", earlier=_earlier(earlier), snapshots=(snap,))
    assert [(f.action, f.paths) for f in writes.held_files] == [("rename", ("report.pdf",))]


@pytest.mark.parametrize("earlier", [
    "(lambda: open('report.pdf', 'w').write('x'))()",
    "def reg(_):\n    return lambda f: f\n@reg(open('report.pdf', 'w').write('x'))\ndef g():\n    pass",
], ids=["a-lambda-called-at-once", "a-decorators-argument"])
def test_a_rewrite_after_a_write_in_a_lambda_or_a_decorator_keeps_its_card(earlier, snap):
    """An earlier cell's write inside a lambda or a decorator's argument
    isn't among the code `_nodes_that_run` counts, so it doesn't make the
    file the agent's own (`_OwnedFiles`): a later rewrite of it keeps its
    card, as its plain top-level twin's does not."""
    writes = find_cell_writes(code="open('report.pdf', 'w').write('y')", earlier=_earlier([earlier]), snapshots=(snap,))
    assert [(f.action, f.paths) for f in writes.held_files] == [("overwrite", ("report.pdf",))]
    twin = find_cell_writes(code="open('report.pdf', 'w').write('y')",
                            earlier=_earlier(["open('report.pdf', 'w').write('x')"]), snapshots=(snap,))
    assert not twin.held_files


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
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=snapshots)
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
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
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
    writes = find_cell_writes(code=code, earlier=_earlier((PG, earlier)), snapshots=(snap,))
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
    ("import os\nos.system('sudo -E rm report.pdf')", [("delete", ("report.pdf",))]),
    ("import os\nos.system('setsid rm report.pdf')", [("delete", ("report.pdf",))]),
    ("import os\nos.system('# tidy up\\nrm report.pdf')", [("delete", ("report.pdf",))]),
    ("import os\nos.system('rm report.pdf # tidy up\\nmv summary.md x.md')",
     [("delete", ("report.pdf",)), ("move", ("summary.md",))]),
    ("import os\nos.system('# rm report.pdf\\necho done')", []),
    ("import subprocess\nc0 = ['rm', 'report.pdf']\nc1 = c0\nsubprocess.run(c1)", [("delete", ("report.pdf",))]),
    ("import pty\npty.spawn(['rm', 'report.pdf'])", [("delete", ("report.pdf",))]),
    ("import subprocess\nsubprocess.run(['rsync', '--remove-source-files', 'report.pdf', 'docs/'])",
     [("move", ("report.pdf",))]),
], ids=["named-command", "named-words", "built-up-command", "f-string", "plus", "format", "percent",
        "f-string-root", "sh-c", "bash-lc", "getoutput", "getstatusoutput", "mv-t", "mv-target-directory",
        "sudo", "xargs", "find-delete", "find-exec-rm", "git-clean", "spawnlp", "shell-variable",
        "if-then", "for-do", "helper-command", "exec-code", "split-text", "shlex-split", "list-plus-list",
        "list-plus-glob", "mv-flags-then-t", "timeout", "stdbuf", "ionice", "busybox", "piped-into-sh",
        "sh-input", "command-substitution", "sudo-keep-env", "setsid", "comment-line-first",
        "comment-ends-at-its-line", "commented-out", "aliased-words", "pty-spawn", "rsync-removing-its-source"])
def test_shell_deletes_and_moves_in_every_form(code, expected, snap):
    assert _files(code, (snap,)) == expected


_REBINDS_OUT = "OUT = 'new.csv'\ndef configure(path):\n    global OUT\n    OUT = path"


@pytest.mark.parametrize("words", [
    ["curl", "-X", "DELETE", "https://example.com/items/1"],
    ["kubectl", "delete", "pod", "web-1"],
    ["aws", "s3", "rm", "s3://bucket/stock.csv"],
    ["git", "push", "origin", "main"],
], ids=["curl-delete", "kubectl-delete", "aws-s3-rm", "git-push"])
def test_a_command_whose_change_is_remote_gets_no_card(words, snap):
    """The write relay holds such a change to a host it relays, and anywhere
    else the account the command uses decides; the check reads only local
    files."""
    assert not _held(f"import subprocess\nsubprocess.run({words!r})", snap)
    assert not _held(f"import os\nos.system({' '.join(words)!r})", snap)


@pytest.mark.parametrize("script", [
    "f'rm {p}'", "'rm -f ' + p", "'rm {}'.format(p)", "'rm %s' % p", "f'mv {p} /tmp/x.csv'",
], ids=["f-string", "plus", "format", "percent", "mv"])
@pytest.mark.parametrize("shell", ["['sh', '-c'", "['bash', '-c'", "['bash', '-lc'"],
                         ids=["sh-c", "bash-c", "bash-lc"])
def test_a_shells_script_the_code_builds_in_a_list_of_words_is_read_as_os_systems_is(script, shell, snap):
    """A shell's script after its `-c` in a list of words is read as the
    text `os.system` is handed (`_listed_commands`), each part picked at
    run time a placeholder, so its deletes and moves are read and the card
    under every value matches `os.system`'s, not code the check can't
    read, which `off` drops."""
    from anton.core.utils.scratchpad import _unchecked_hold_dropped

    def cards(code: str) -> list[bool]:
        writes = find_cell_writes(code=code, earlier=_earlier(["p = 'stock.csv'"]), snapshots=(snap,))
        return [writes.needs_approval() and not _unchecked_hold_dropped(writes, mode) for mode in ("all", "own", "off")]

    listed = cards(f"import subprocess\nsubprocess.run({shell}, {script}])")
    assert listed == cards(f"import os\nos.system({script})") == [True, True, True]


@pytest.mark.parametrize("script", [
    "f'sed -i s/a/b/ {p}'", "f'echo x > {p}'", "'echo x | tee ' + p", "'cp job.txt {}'.format(p)",
    "'truncate -s 0 %s' % p",
], ids=["sed-i", "a-redirect", "tee", "cp", "truncate"])
@pytest.mark.parametrize("shell", ["['sh', '-c'", "['bash', '-c'", "['bash', '-lc'"],
                         ids=["sh-c", "bash-c", "bash-lc"])
def test_a_shells_script_the_code_builds_in_a_list_of_words_stays_code_the_check_cant_read(script, shell, snap):
    """Beside the deletes and moves read in it, a shell's script after its
    `-c` that the code builds stays a word the check can't read
    (`_listed_commands`): a write other than those runs code the check
    can't read, a card under all and own."""
    code = f"import subprocess\nsubprocess.run({shell}, {script}])"
    writes = find_cell_writes(code=code, earlier=_earlier(["p = 'stock.csv'"]), snapshots=(snap,))
    assert [c.kind for c in writes.causes] == ["runs"] and not writes.held_files
    assert _cards(code, snap, earlier=["p = 'stock.csv'"]) == [True, True, False]


@pytest.mark.parametrize("store", [
    "__import__('__main__').cfg['o'] = 'stock.csv'",
    "def h():\n    pass\nimport gc\nfor x in gc.get_referents(h):\n"
    "    if type(x) is dict and 'cfg' in x:\n        x['cfg']['o'] = 'stock.csv'",
    "vars()['cfg']['o'] = 'stock.csv'",
    "cfg.update(o='stock.csv')",
], ids=["through-main", "through-gc-referents", "through-vars", "updated"])
@pytest.mark.parametrize("code", [
    "import os\nos.system(f\"rm {cfg['o']}\")", "import subprocess\nsubprocess.run(['rm', cfg['o']])",
    "import os\nos.system(f\"mv {cfg['o']} /tmp/x\")",
], ids=["rm-script", "rm-words", "mv-script"])
def test_a_shell_delete_through_a_dict_entry_a_cell_may_change_holds_whole(store, code, snap):
    """The command reading refuses a dict entry where the reading of a path
    would (`_entry_refused`): a cell may change the dict out of sight, so
    the shell delete or move holds whole, as `os.remove(cfg['o'])` and
    `shutil.move(cfg['o'], ...)` do, and never reads the display's stale
    entry."""
    earlier = ["cfg = {'o': 'new2.csv'}", store]
    twin = "import shutil\nshutil.move(cfg['o'], '/tmp/x')" if "mv" in code else "import os\nos.remove(cfg['o'])"
    assert _files(code, (snap,), earlier=earlier) == _files(twin, (snap,), earlier=earlier) != []


@pytest.mark.parametrize("code", ["import os\nos.system(CMD)", "import subprocess\nsubprocess.run(CMD, shell=True)"],
                         ids=["os-system", "shell-true"])
def test_a_whole_command_held_in_a_rebound_name_names_no_stale_file(code, snap):
    """A command held in a name a function rebinds with `global` is one the
    check can't place: never its stale value's delete, but a delete it
    picks when it runs, beside the code the check can't read."""
    earlier = ["CMD = 'rm report.pdf'\ndef quiet():\n    global CMD\n    CMD = 'ls'", "quiet()"]
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    assert [(c.action, c.paths) for c in writes.held_files] == [("delete", ())]
    assert "report.pdf" not in _card(code, snap, earlier=earlier)


def test_a_write_through_a_rebound_name_past_the_depth_limit_stops(snap):
    """Its shallow twin holds whole, as onto a file the code picks when it
    runs (GLOBAL), which no hold the read within the limit finds covers
    (`_covers`): the check stops."""
    code = "open(p9, 'w').write('x')"
    earlier = [_REBINDS_OUT + "\np0 = OUT\n" + "".join(f"p{i} = p{i - 1}\n" for i in range(1, 10))]
    assert _files(code.replace("p9", "OUT"), (snap,), earlier=earlier) == [("overwrite", ())]
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    assert writes.stopped and writes.needs_approval()


def test_a_script_whose_quote_never_closes_reads_each_word_without_its_quotes(snap):
    """Where a quote never closes, the words split at spaces with each pair
    of quotes taken off (`_script_commands`), so a quoted operand still
    names its file."""
    assert _files("import os\nos.system('rm -f \\'stock.csv\\'\\necho \"x')", (snap,)) == [("delete", ("stock.csv",))]


def _cards(code: str, snap, *, earlier=()) -> list[bool]:
    """Whether the cell `code` gets a card under all, own and off."""
    from anton.core.utils.scratchpad import _unchecked_hold_dropped

    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    return [writes.needs_approval() and not _unchecked_hold_dropped(writes, mode) for mode in ("all", "own", "off")]


@pytest.mark.parametrize("binder", [
    ["def quiet():\n    global CMD\n    CMD = 'rm report.pdf'", "quiet()"],
    ["globals()['CMD'] = 'rm report.pdf'"],
    ["class Setup:\n    global CMD\n    CMD = 'rm report.pdf'"],
    ["def quiet():\n    exec(\"global CMD\\nCMD = 'rm report.pdf'\")", "quiet()"],
    ["def quiet():\n    global CMD\n    CMD = ['rm', 'report.pdf']", "quiet()"],
], ids=["a-functions-global", "a-counted-store", "a-class-bodys-global", "literal-code-in-a-function",
        "a-functions-global-of-a-list"])
@pytest.mark.parametrize("code", [
    "import os\nos.system(CMD)", "import subprocess\nsubprocess.run(CMD, shell=True)",
    "import os\nos.popen(CMD).read()", "import subprocess\nsubprocess.run(['sh', '-c', CMD])",
    "import subprocess\nsubprocess.run(('bash', '-lc', CMD))", "import subprocess\nsubprocess.run(['env', 'sh', '-c', CMD])",
    "import subprocess\nsubprocess.run(CMD.split())", "import subprocess\nsubprocess.Popen(CMD.split(' '))",
    "import shlex, subprocess\nsubprocess.run(shlex.split(CMD))", "import os\nos.execlp('sh', 'sh', '-c', CMD)",
    "import os\nos.system(str(CMD))", "import os\nos.system(CMD.strip())",
    "import os\nos.system(CMD.format(x=1))", "import os\nos.system('{} -v'.format(CMD))",
    "import os\nos.system(CMD.replace('-i', '-f'))", "import os\nos.system(' '.join(CMD))",
    "import os\nos.system(' && '.join([CMD, 'ls']))", "import subprocess\nsubprocess.run([*CMD])",
    "import subprocess\nsubprocess.run(['sudo', *CMD])", "import os\nos.system(f'sudo {CMD}')",
    "import os\nos.system('nohup ' + CMD + ' &')", "import os\nos.system(f'timeout 10 {CMD}')",
    "import os\nos.system('env ' + CMD)", "import subprocess\nsubprocess.run(f'sudo {CMD}', shell=True)",
    "import os\nos.system(\"bash -c '\" + CMD + \"'\")",
    "import subprocess\nsubprocess.run(['sh', '-c', 'sudo ' + CMD])",
    "import subprocess\nsubprocess.run(['sudo', *CMD.split()])", "import subprocess\nsubprocess.run(CMD + ['-f'])",
    "import subprocess\nsubprocess.run(['sudo'] + CMD)", "import os, shlex\nos.system(shlex.join(CMD))",
], ids=["os-system", "shell-true", "popen", "sh-c-in-a-list-of-words", "bash-lc-in-a-tuple", "env-sh-c", "split",
        "split-on-a-space", "shlex-split", "execlp-sh-c", "str", "strip", "format", "format-around-it", "replace",
        "a-join-of-it", "a-join-with-it", "spread-into-a-list-of-words", "spread-after-a-prefix", "after-sudo",
        "after-nohup", "after-timeout", "after-env", "after-sudo-shell-true", "a-shells-c-in-a-script",
        "after-sudo-in-sh-c", "split-spread-after-sudo", "a-list-added-after-it", "a-list-added-before-it",
        "shlex-join"])
def test_a_whole_command_only_a_rebinding_gives_a_value_is_a_delete_it_picks_when_it_runs(binder, code, snap):
    """With no top-level value, a command held in a name a function's or
    class body's `global`, a counted store or literal code binds is one
    the check can't place (`_rebound_name`): a delete it picks when it
    runs, under every value of the setting, as the path twin
    `os.system(f'rm {CMD}')` holds whole. It is a script, so a shell's
    `-c` in a list of words, `split()`, `shlex.split()`, `str()` and
    `strip()` read it as `os.system` does; and where the code around it
    gives no command the check reads, as `format()`, `replace()`, a join
    or a spread into a list of words does, the command is still the one
    the name holds when it runs (`_node_commands`, `_listed_commands`),
    after `sudo`, `env`, `nohup` or `timeout` too, as a shell's `-c` code
    and in a list added to another (`_program_filled`, `_word_lists`)."""
    writes = find_cell_writes(code=code, earlier=_earlier(binder), snapshots=(snap,))
    assert [(c.action, c.paths) for c in writes.held_files] == [("delete", ())]
    assert _cards(code, snap, earlier=binder) == [True, True, True]
    assert _cards("import os\nos.system(f'rm {CMD}')", snap, earlier=binder) == [True, True, True]


@pytest.mark.parametrize("binder, code", [
    ("def quiet():\n    global CMDS\n    CMDS = ['rm report.pdf']", "import os\nos.system(CMDS[0])"),
    ("def quiet():\n    global CMDS\n    CMDS = ['rm report.pdf']", "import os\nos.system(CMDS.pop())"),
    ("CMDS = ['ls']\ndef quiet():\n    global CMDS\n    CMDS = ['rm report.pdf']", "import os\nos.system(CMDS[0])"),
    ("def quiet():\n    global CFG\n    CFG = {'c': 'rm report.pdf'}", "import os\nos.system(CFG['c'])"),
    ("def quiet():\n    global CFG\n    CFG = {'c': 'rm report.pdf'}", "import os\nos.system(CFG.get('c'))"),
], ids=["an-index", "a-pop", "an-index-with-a-stale-list", "a-key", "a-get"])
def test_a_command_taken_out_of_a_rebound_list_or_dict_is_a_delete_it_picks_when_it_runs(binder, code, snap):
    """A command taken out of a list or dict a function rebinds with
    `global` gives no command the check reads, so it is the one the name
    holds when it runs (`_node_commands`): a delete it picks when it runs,
    under every value, never the stale list's `ls`."""
    earlier = [binder, "quiet()"]
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    assert [(c.action, c.paths) for c in writes.held_files] == [("delete", ())]
    assert _cards(code, snap, earlier=earlier) == [True, True, True]


@pytest.mark.parametrize("code", [
    "import os\nos.system(f'{PROG} stock.csv')", "import subprocess\nsubprocess.run([PROG, 'stock.csv'])",
    "import subprocess\nsubprocess.run(['sudo', PROG, 'stock.csv'])",
    "import subprocess\nsubprocess.run(['env', PROG, 'stock.csv'])",
    "import subprocess\nsubprocess.run(['timeout', '5', PROG, 'stock.csv'])",
    "import os\nos.system(f'sudo {PROG} stock.csv')",
], ids=["script", "words", "words-after-sudo", "words-after-env", "words-after-timeout", "script-after-sudo"])
@pytest.mark.parametrize("top", ["PROG = 'ls'\n", ""], ids=["a-top-level-value-too", "no-top-level-value"])
def test_a_program_held_in_a_rebound_name_is_a_delete_it_picks_when_it_runs(code, top, snap):
    """A program word held in such a name is the one it holds when it
    runs, as `$RM` is in a shell (`_rebound_name`), after a word such as
    `sudo` too: a delete it picks when it runs, under every value."""
    earlier = [top + "def f():\n    global PROG\n    PROG = 'rm'", "f()"]
    assert ("delete", ()) in _files(code, (snap,), earlier=earlier)
    assert _cards(code, snap, earlier=earlier) == [True, True, True]


@pytest.mark.parametrize("code, earlier", [
    ("import os\nos.system('sudo ' + CMD)", ["CMD = 'rm report.pdf'", "def g():\n    global CMD\n    CMD = 'ls'"]),
    ("import os\nos.system('nohup ' + CMD)", ["CMD = 'rm report.pdf'", "def g():\n    global CMD\n    CMD = 'ls'"]),
    ("import subprocess\nsubprocess.run(f'sudo {CMD}', shell=True)",
     ["CMD = 'rm report.pdf'", "def g():\n    global CMD\n    CMD = 'ls'"]),
    ("import subprocess\nsubprocess.run(ARGV + ['-f'])",
     ["ARGV = ['rm', 'report.pdf']", "def h():\n    global ARGV\n    ARGV = ['ls']"]),
    ("import subprocess\nsubprocess.run(['sudo'] + ARGV)",
     ["ARGV = ['rm', 'report.pdf']", "def h():\n    global ARGV\n    ARGV = ['ls']"]),
], ids=["after-sudo", "after-nohup", "after-sudo-shell-true", "a-list-added-after-it", "a-list-added-before-it"])
def test_a_whole_command_whose_top_level_value_deletes_holds_where_a_function_may_rebind_it(code, earlier, snap):
    """A function nothing called yet may rebind the name, so the command is
    the one it holds when it runs (`_rebound_name`), never the top-level
    value read alone: here that value deletes report.pdf, and the cell
    holds a delete it picks when it runs, under every value."""
    assert ("delete", ()) in _files(code, (snap,), earlier=earlier)
    assert _cards(code, snap, earlier=earlier) == [True, True, True]


@pytest.mark.parametrize("code", [
    "import os\nos.system(' '.join(CMD))", "import subprocess\nsubprocess.run(' '.join(CMD), shell=True)",
    "import os\nos.system('sudo ' + ' '.join(CMD))", "import os\nos.system(' '.join(CMD) + ' -v')",
    "import os\nos.system(f\"{' '.join(CMD)}\")", "import os\nos.system('{}'.format(' '.join(CMD)))",
    "import subprocess\nsubprocess.run(['sh', '-c', ' '.join(CMD)])",
    "import subprocess\nsubprocess.run(' '.join(CMD).split())", "import os\nos.system('; '.join(CMDS))",
    "import os\nc = ' '.join(CMD)\nos.system(c)", "import subprocess\nsubprocess.run(['sudo'] + sorted(CMD))",
    "import subprocess\nsubprocess.run(['sudo'] + [c for c in CMD])",
    "import subprocess\nsubprocess.run(['sudo'] + CMD[:])", "import subprocess\nsubprocess.run([CMD[0], 'report.pdf'])",
    "import os\nos.system(CMD[0] + ' report.pdf')",
], ids=["a-join", "a-join-shell-true", "a-join-after-sudo", "a-join-with-more", "a-join-in-an-f-string",
        "a-join-formatted", "a-join-as-a-shells-c", "a-join-split", "a-join-of-scripts", "a-name-bound-to-a-join",
        "a-sorted-copy-after-sudo", "a-comprehension-after-sudo", "a-slice-after-sudo", "its-program-in-words",
        "its-program-in-a-script"])
@pytest.mark.parametrize("earlier", [
    ["CMD = ['ls']\nCMDS = ['ls']\ndef quiet():\n    global CMD, CMDS\n    CMD = ['rm', 'report.pdf']\n"
     "    CMDS = ['rm report.pdf']", "quiet()"],
    ["CMD = ['rm', 'report.pdf']\nCMDS = ['rm report.pdf']\ndef quiet():\n    global CMD, CMDS\n    CMD = ['ls']\n"
     "    CMDS = ['ls']", "quiet()"],
    ["CMD = ['ls']\nCMDS = ['ls']", "globals()['CMD'] = ['rm', 'report.pdf']\nglobals()['CMDS'] = ['rm report.pdf']"],
], ids=["a-stale-listing", "a-stale-delete", "a-counted-store-over-a-stale-listing"])
def test_a_list_command_a_function_may_rebind_is_never_read_as_its_top_level_value(code, earlier, snap):
    """A list command a name code may have rebound holds is the one it holds
    when it runs, joined, copied or read by its program word too, never its
    top-level value (`_command_text`, `_unplaced_word`): a delete it picks
    when it runs, under every value, and no card naming report.pdf, which
    a stale `rm report.pdf` names though the step may only list. The same
    list no code rebinds is read: a listing holds nothing."""
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    assert [(c.action, c.paths) for c in writes.held_files] == [("delete", ())]
    assert _cards(code, snap, earlier=earlier) == [True, True, True]
    assert _cards(code, snap, earlier=["CMD = ['ls']\nCMDS = ['ls']"]) == [False, False, False]


@pytest.mark.parametrize("code", [
    "import os\nos.system(f\"rm {cfg['o']}\")", "import subprocess\nsubprocess.run(['rm', cfg['o']])",
    "import os\nos.system('mv ' + cfg['o'] + ' /tmp/x')",
], ids=["rm-script", "rm-words", "mv"])
@pytest.mark.parametrize("earlier", [
    ["cfg = {'o': 'new.csv'}"],
    ["cfg = {'o': 'new.csv'}", "import pickle\npickle.loads(blob)"],
    ["cfg = {'o': 'new.csv'}", "import helpers"],
    ["cfg = {'o': 'stock.csv'}"],
], ids=["a-clean-dict", "a-pickle-load-later", "a-project-module-later", "a-file-there"])
def test_a_shell_delete_or_move_through_a_dict_entry_holds_whole(code, earlier, snap):
    """A delete or a move never narrows to a dict entry: a cell may change
    the entry, however it reaches it, so the shell `rm` or `mv` holds
    whole, as `os.remove(cfg['o'])` and `shutil.move` do."""
    twin = "import shutil\nshutil.move(cfg['o'], '/tmp/x')" if "mv" in code else "import os\nos.remove(cfg['o'])"
    assert _files(code, (snap,), earlier=earlier) == _files(twin, (snap,), earlier=earlier) == [
        ("move" if "mv" in code else "delete", ())]


@pytest.mark.parametrize("change", [
    "cfg.update(tool='rm')", "__import__('__main__').cfg['tool'] = 'rm'", "vars()['cfg']['tool'] = 'rm'",
    "import helpers\nhelpers.fix(cfg)",
], ids=["updated", "through-main", "through-vars", "handed-to-a-call"])
@pytest.mark.parametrize("code", [
    "import os\nos.system(f\"{cfg['tool']} stock.csv\")", "import subprocess\nsubprocess.run([cfg['tool'], 'stock.csv'])",
    "import os\nos.system(cfg['tool'] + ' stock.csv')",
], ids=["f-string", "words", "plus"])
def test_a_program_a_dict_entry_a_cell_may_change_gives_is_one_the_check_cant_read(change, code, snap):
    """The command reading refuses a dict entry a cell may change out of
    sight (`_entry_refused`), so a program it gives is the one the entry
    holds when it runs, never the display's stale `ls`: a delete it picks
    when it runs, under every value, as a program a rebound name holds
    (`_unplaced_word`). The dict no cell changes gives `ls`, which holds
    nothing."""
    assert ("delete", ()) in _files(code, (snap,), earlier=["cfg = {'tool': 'ls'}", change])
    assert _cards(code, snap, earlier=["cfg = {'tool': 'ls'}", change]) == [True, True, True]
    assert _cards(code, snap, earlier=["cfg = {'tool': 'ls'}"]) == [False, False, False]


@pytest.mark.parametrize("change", [
    "cfg.update(t='rm report.pdf')", "cfg['t'] = 'rm report.pdf'", "d = cfg\nd['t'] = 'rm report.pdf'",
    "globals()['cfg']['t'] = 'rm report.pdf'", "__import__('__main__').cfg['t'] = 'rm report.pdf'",
    "import helpers\nhelpers.fix(cfg)", "cfg.pop('t')\ncfg.setdefault('t', 'rm report.pdf')",
    "import operator\noperator.setitem(cfg, 't', 'rm report.pdf')", "fix = cfg.update\nfix(t='rm report.pdf')",
    "holder = [cfg]\nholder[0]['t'] = 'rm report.pdf'", "d = cfg\nd.update(t='rm report.pdf')",
    "c = cfg\nimport helpers\nhelpers.fix(c)", "outer = {'c': cfg}\nouter['c']['t'] = 'rm report.pdf'",
], ids=["updated", "stored", "through-another-name", "through-globals", "through-main", "handed-to-a-call",
        "popped-then-set", "through-setitem", "a-method-taken", "in-a-list", "another-name-updated",
        "another-name-handed-on", "in-another-dict"])
@pytest.mark.parametrize("code", [
    "import os\nos.system(cfg['t'])", "import subprocess\nsubprocess.run(cfg['t'], shell=True)",
    "import subprocess\nsubprocess.run(cfg['t'].split())", "import os\nos.system(cfg.get('t'))",
    "import os\nc = cfg['t']\nos.system(c)",
], ids=["os-system", "shell-true", "split", "get", "a-name-bound-to-it"])
def test_a_whole_command_a_dict_entry_a_cell_may_change_gives_is_a_delete_it_picks_when_it_runs(change, code, snap):
    """A whole command read from a dict entry a cell may change out of
    sight (`_entry_refused`), by a store, a method of the dict or a call
    handed it, or after a cell hands the dict on under another name, in a
    list or another dict, is the one the entry holds when it runs, never
    the display's stale `ls stock.csv`: a delete it picks when it runs,
    under every value, as a whole command a rebound name holds
    (`_unplaced_in`). The dict no cell changes gives `ls`, which holds
    nothing."""
    earlier = ["cfg = {'t': 'ls stock.csv'}", change]
    assert ("delete", ()) in _files(code, (snap,), earlier=earlier)
    assert _cards(code, snap, earlier=earlier) == [True, True, True]
    assert _cards(code, snap, earlier=earlier[:1]) == [False, False, False]


@pytest.mark.parametrize("change, code", [
    ("cfg.update(t='rm report.pdf')", "import os\nos.system(cfg['t'])"),
    ("cfg.pop('t')\ncfg.setdefault('t', 'rm report.pdf')", "import os\nos.system(cfg['t'])"),
    ("import operator\noperator.setitem(cfg, 't', 'rm report.pdf')", "import os\nos.system(cfg.get('t'))"),
    ("cfg.update(t='rm')", "import subprocess\nsubprocess.run([cfg['t'], 'report.pdf'])"),
], ids=["updated", "popped-then-set", "through-setitem", "a-program-updated"])
def test_a_command_from_a_dict_a_dict_call_built_that_a_cell_changes_is_a_delete_it_picks_when_it_runs(
    change, code, snap,
):
    """A dict built by a `dict(...)` call holds its keywords as entries, as
    a display does (`_Reader.items_of`), so a cell that changes it refuses
    them too (`_entry_refused`): the command or program read from the
    entry is the one it holds when it runs, a delete it picks when it runs,
    under every value, never the stale `ls`. The dict no cell changes gives
    `ls`, which holds nothing."""
    built = "cfg = dict(t='ls')"
    assert ("delete", ()) in _files(code, (snap,), earlier=[built, change])
    assert _cards(code, snap, earlier=[built, change]) == [True, True, True]
    assert _cards(code, snap, earlier=[built]) == [False, False, False]


def test_whether_a_cell_may_change_a_dict_is_read_once_per_check(snap, monkeypatch):
    """Whether a cell may change a dict display out of sight
    (`_entry_refused`) is one read of every cell's loads of it per check
    (`_Reader.entries_refused`), not one per lookup in each command:
    counted, not timed."""
    from anton.core.utils import write_check

    asked: list[str] = []
    real = write_check._changed_in_place
    monkeypatch.setattr(write_check, "_changed_in_place",
                        lambda name, found, reader: asked.append(name) or real(name, found, reader))
    earlier = ["cfg = {" + ", ".join(f"'k{i}': 'ls f{i}.csv'" for i in range(200)) + "}"]
    code = "import os\n" + "".join(f"os.system(f\"cat {{cfg['k{i}']}}\")\n" for i in range(200))
    find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    assert asked.count("cfg") <= 3


@pytest.mark.parametrize("code", [
    "import os\nCMDS = {'clean': 'rm new.csv'}\nos.system(CMDS['clean'])",
    "import os\nCMDS = {'clean': 'rm new.csv'}\nos.system(CMDS.get('clean'))",
], ids=["by-key", "by-get"])
def test_a_shell_delete_a_dict_entry_holds_whole(code, snap):
    """A whole command read from a dict entry deletes what the entry holds
    when it runs: it holds whole, as a delete through an entry does."""
    assert _files(code, (snap,)) == [("delete", ())]
    assert _files(code.replace("CMDS['clean']", "'rm new.csv'").replace("CMDS.get('clean')", "'rm new.csv'"),
                  (snap,)) == []


@pytest.mark.parametrize("run", [
    "subprocess.run(['psql', '-h', 'db.example.com', '-c', sql])", "subprocess.run(['mysql', '-e', sql])",
    "subprocess.run(['sqlite3', 'app.db', sql])", "subprocess.run(['psql', '-h', 'db.example.com'], input=sql)",
], ids=["psql-c", "mysql-e", "sqlite3", "input"])
def test_a_sql_file_an_earlier_cell_reads_past_the_cap_is_held_where_a_command_line_client_runs_it(run, sql_files):
    """A database's command line client runs the SQL text a call hands it
    as a word or as its input (`_feeds_a_program`): the big file an
    earlier cell read gets the sql_file card, as in its own cell and as
    its small twin does."""
    code = "import subprocess\n" + run
    big = find_cell_writes(code=code, earlier=_earlier(["sql = open('big.sql').read()"]), snapshots=(sql_files,))
    assert [c.via for c in big.calls] == ["sql_file"]
    same = code.replace(", sql]", ", open('big.sql').read()]").replace("=sql)", "=open('big.sql').read())")
    assert [c.via for c in find_cell_writes(code=same, earlier=_earlier(()), snapshots=(sql_files,)).calls] == [
        "sql_file"]
    small = find_cell_writes(code=code, earlier=_earlier(["sql = open('fix.sql').read()"]), snapshots=(sql_files,))
    assert small.needs_approval()


@pytest.mark.parametrize("read", [
    ["script = ''", "script += open('db.sql').read()"],
    ["script = ''", "script = script + open('db.sql').read()"],
    ["script = ''", "script += 'DELETE FROM orders;'"],
], ids=["plus-equals-a-file", "plus-a-file", "plus-equals-a-literal"])
def test_a_small_sql_files_delete_added_to_a_name_in_another_cell_is_held(read, tmp_path, monkeypatch):
    """What another cell adds to a name, or a binding adds to the name's
    own value, is read on its own for the statements it holds
    (`_Reader.statements_held`): the DELETE runs on app.db, as it does
    where the same cell builds the text or the file is past the cap. A
    SELECT added the same way holds nothing."""
    (tmp_path / "app.db").write_text("")
    (tmp_path / "db.sql").write_text("DELETE FROM orders;\n")
    (tmp_path / "sel.sql").write_text("SELECT 1;\n")
    monkeypatch.chdir(tmp_path)
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    run = "import sqlite3\nsqlite3.connect('app.db').executescript(script)"
    assert find_cell_writes(code=run, earlier=_earlier(read), snapshots=(snap,)).needs_approval()
    reads = [r.replace("db.sql", "sel.sql").replace("DELETE FROM orders", "SELECT 1") for r in read]
    assert not find_cell_writes(code=run, earlier=_earlier(reads), snapshots=(snap,)).needs_approval()


@pytest.mark.parametrize("code", [
    "import subprocess\nsubprocess.run(['cp', 'boot.py', 'sitecustomize.py'])",
    "import os\nos.system('cp boot.py usercustomize.py')",
    "import os\nos.system('echo import boot > sitecustomize.py')",
    "import os\nos.system('echo import boot >sitecustomize.py')",
    "import os\nos.system('echo import boot >> sitecustomize.py')",
    "import os\nos.system('echo import boot | tee sitecustomize.py')",
    "import os\nos.system('install boot.py sitecustomize.py')",
    "import os\nos.system('ln -sf boot.py sitecustomize.py')",
    "import os\nos.system('rsync boot.py sitecustomize.py')",
    "import os\nos.system('mv boot.py sitecustomize.py')",
    "import os\nos.system('curl -o sitecustomize.py https://example.com/a.py')",
    "import os\nos.system('wget -O sitecustomize.py https://example.com/a.py')",
    "import os\nos.system('touch usercustomize.py')",
    "import os\nos.system('wget -P . https://example.com/sitecustomize.py')",
    "import os\nos.system('curl -O --output-dir . https://example.com/sitecustomize.py')",
], ids=["words", "script", "redirect", "redirect-glued-to-the-file", "append", "tee", "install", "ln", "rsync",
        "mv-onto", "curl-o", "wget-o", "touch", "wget-into-its-folder", "curl-into-its-folder"])
def test_a_shell_write_of_a_startup_hook_it_names_is_a_hook_write(code, snap):
    """A shell write of a startup hook it names, in a spelling the hook walk
    reads (`_command_hook_write`), as `echo x > sitecustomize.py` is, is a
    hook write, as a list of words or a script, held under every value."""
    assert find_cell_writes(code=code, earlier=_earlier(()), snapshots=(snap,)).writes_startup_hook
    assert _cards(code, snap) == [True, True, True]


@pytest.mark.parametrize("wrap", [
    "['sh', '-c', f'{cmd}']", "['bash', '-c', 'set -e; ' + cmd]", "['sh', '-c', '{}'.format(cmd)]",
    "['sh', '-c', '%s' % cmd]", "['sudo', 'sh', '-c', cmd]", "['env', 'bash', '-c', cmd]",
], ids=["f-string", "a-prefix", "format", "percent", "sudo", "env"])
@pytest.mark.parametrize("where", ["same-cell", "earlier-cell"])
def test_a_shells_script_built_around_a_command_a_file_holds_is_read_as_os_systems_is(
    wrap, where, tmp_path, monkeypatch,
):
    """A whole command a name holds, as a file's text read into it, is the
    script a shell's `-c` runs wherever the command starts in it, as
    `os.system(cmd)` reads it (`_program_filled`): its delete holds under
    every value."""
    (tmp_path / "stock.csv").write_text("x")
    (tmp_path / "run.sh").write_text("rm stock.csv\n")
    monkeypatch.chdir(tmp_path)
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    read = "cmd = open('run.sh').read()"
    code = f"import subprocess\nsubprocess.run({wrap})"
    earlier, code = ([read], code) if where == "earlier-cell" else ([], read + "\n" + code)
    assert _files(code, (snap,), earlier=earlier) == [("delete", ("stock.csv",))]
    assert _cards(code, snap, earlier=earlier) == [True, True, True]


@pytest.mark.parametrize("change", [
    "open('lsjob.txt', 'a').write('rm report.pdf\\n')\n",
    "from pathlib import Path\nPath('lsjob.txt').write_text('rm report.pdf\\n')\n",
    "",
], ids=["an-append", "an-overwrite", "no-change"])
@pytest.mark.parametrize("run", ["os.system(f'set -e; {cmd}')", "os.system('cd . && ' + cmd)"],
                         ids=["f-string", "plus"])
def test_a_command_a_file_holds_is_also_a_program_the_check_cant_name(change, run, tmp_path, monkeypatch):
    """A whole command a file's text gives, where a command starts in a
    wrapper, is also the placeholder (`_program_filled`): the cell may
    change the file before the command runs, so it runs a program the
    check can't name, a card under all and own, whatever the file held
    when the check read it."""
    (tmp_path / "lsjob.txt").write_text("ls\n")
    (tmp_path / "report.pdf").write_text("x")
    monkeypatch.chdir(tmp_path)
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    code = "import os\n" + change + "cmd = open('lsjob.txt').read()\n" + run
    writes = find_cell_writes(code=code, earlier=_earlier(()), snapshots=(snap,))
    assert "runs" in [c.kind for c in writes.causes]
    assert _cards(code, snap)[:2] == [True, True]


@pytest.mark.parametrize("write", [
    "open(p9, 'w').write('x = 1')", "import shutil\nshutil.copy('job.txt', p9)",
    "from pathlib import Path\nPath(p9).write_text('x = 1')",
], ids=["open", "copy", "write-text"])
def test_a_python_write_of_a_new_module_past_the_depth_limit_holds_a_later_import(write, snap):
    """The module walk reads a write's path on past the depth limit
    (`_Reader.paths_on`), so a later import of the module it lands gets
    the card its twin with the path written out gets."""
    flat = _cards("import job", snap, earlier=[write.replace("p9", "'job.py'")])
    assert _cards("import job", snap, earlier=[_nine_names("job.py"), write]) == flat == [True, True, False]


@pytest.mark.parametrize("change", [
    "import helpers\ncfg = {'o': 'stock.csv'}\nhelpers.fix(cfg)",
    "cfg = {'o': 'stock.csv'}\nbox = [cfg]\nbox[0]['o'] = 'new.csv'",
], ids=["handed-to-a-call", "held-in-a-list-and-changed-there"])
@pytest.mark.parametrize("code, twin", [
    ("os.system(f\"rm {cfg['o']}\")", "os.remove(cfg['o'])"),
    ("os.system(f\"mv {cfg['o']} /tmp/x\")", "shutil.move(cfg['o'], '/tmp/x')"),
], ids=["rm", "mv"])
def test_a_shell_delete_through_an_entry_of_a_dict_a_cell_hands_on_holds_whole(change, code, twin, snap):
    """The command reading refuses the entry where the path reading would
    (`_entry_refused`): a dict handed to a call may change in place, and
    one held in a list may change through it, so the delete or move never
    reads the display's stale text and holds whole, as its Python twin
    does."""
    cell = f"import os, shutil\n{change}\n"
    assert _files(cell + code, (snap,)) == _files(cell + twin, (snap,)) == [
        ("move" if "mv" in code else "delete", ())]


_HOOK_IN_A_FUNCTION = ("import os, site\nSP = site.getsitepackages()[0]\n"
                       "def boot():\n    open(os.path.join(SP, 'zz.pth'), 'w').write('import boot\\n')")


def test_a_hook_write_in_an_earlier_cells_function_is_a_hook_write_where_a_later_cell_calls_it(snap):
    """A function an earlier cell defined that writes a startup hook is
    read in its own cell (`_Reader.cell_of`), so the call keeps the hook's
    card under every value of the setting."""
    writes = find_cell_writes(code="boot()", earlier=_earlier([_HOOK_IN_A_FUNCTION]), snapshots=(snap,))
    assert writes.writes_startup_hook
    assert _cards("boot()", snap, earlier=[_HOOK_IN_A_FUNCTION]) == [True, True, True]


@pytest.mark.parametrize("rebind", [
    ["def g():\n    global CMD\n    CMD = 'ls'", "g()"],
    ["globals()['CMD'] = 'ls'"],
], ids=["a-global-in-a-function", "a-globals-store"])
def test_a_command_a_later_cell_rebinds_through_the_namespace_stops_later_cells_after_a_restart(
    rebind, snap, monkeypatch,
):
    """The command a later cell rebinds through `global` or a `globals()`
    store: after a restart the cell that ran it is read with the cells up
    to it alone, so every later cell still stops."""
    from anton.core.utils import write_check

    monkeypatch.setattr(write_check, "_LEFT_UNREAD", write_check._LeftUnread(limit=100_000))
    cells = ["import os\ndef run():\n    os.system(CMD)", f"CMD = {_FIVE_SHELLS!r}", "run()", *rebind]
    writes = find_cell_writes(code="print(1)", earlier=_earlier(cells), snapshots=(snap,))
    assert [(c.kind, c.in_earlier) for c in writes.causes] == [("limit", True)]
    assert writes.causes[0].code.startswith("os.system(CMD)")


def test_a_path_rebuilt_from_itself_many_times_is_read_in_well_under_a_second(snap):
    """A path a cell builds on itself step by step holds one value per
    step (`_rebuilds`); read on past the depth limit
    (`_Reader.reading_past`), the read of the value a write lands on gives
    up past `_PAST_DEPTH` levels, so the check stops in well under a
    second, and no later cell stops: it left a value unplaced, not code
    unread, and a later import reads it as one the check can't place
    (`_Reader.paths_on`). CPU time."""
    code = ("import os\nimport pandas as pd\nout = 'reports'\n"
            + "".join(f"out = os.path.join(out, 'part{i}')\n" for i in range(300))
            + "pd.DataFrame({'a': [1]}).to_csv(os.path.join(out, 'summary.csv'))")
    started = time.process_time()
    writes = find_cell_writes(code=code, earlier=_earlier(()), snapshots=(snap,))
    assert time.process_time() - started < 2
    assert writes.stopped and [c.kind for c in writes.causes] == ["limit"]
    assert not find_cell_writes(code="import job", earlier=_earlier([code]), snapshots=(snap,)).stopped


def test_a_value_stop_on_an_http_session_quotes_the_call_it_stopped_at(snap):
    """Where a session passed through more names than the HTTP walk
    follows stops the check, the card quotes the call it stopped at."""
    code = "import requests\ns0 = requests.Session()\n" + "".join(f"s{i} = s{i - 1}\n" for i in range(1, 9)) + (
        "s8.delete('https://api.example.com/holds/7')")
    writes = find_cell_writes(code=code, earlier=_earlier(()), snapshots=(snap,))
    [limit] = [c for c in writes.causes if c.kind == "limit"]
    assert writes.stopped and limit.code == "s8.delete('https://api.example.com/holds/7')"


@pytest.mark.parametrize("code, held", [
    ("import os\nCONFIG = {'bucket': 's3://b/x'}\n"
     "os.system(f\"aws s3 cp {CONFIG['bucket']} /tmp/x.csv && rm -f /tmp/old.csv\")", []),
    ("import os\ncfg = {'msg': 'hi'}\nos.system(f\"echo {cfg['msg']}; rm -f new_scratch.txt\")", []),
    ("import os\nfiles = ['new1.tmp', 'new2.tmp']\nfor i in range(2):\n    os.system(f'rm -f {files[i]}')", []),
    ("import os\ncmds = ['rm new.csv']\nos.system(cmds[0])", []),
    ("import os\ncfg = {'msg': 'hi'}\nos.system(f\"echo {cfg['msg']}; rm -f {cfg['msg']}.csv\")", [("delete", ())]),
    ("import os\nP = {'work': 'output'}\nos.system(f\"cd {P['work']} && rm -f scratch.tmp\")", [("delete", ())]),
    ("import os\nfiles = {'a': 'new1.tmp'}\nfor k in ['a']:\n    os.system(f'rm -f {files[k]}')", [("delete", ())]),
], ids=["an-entry-in-another-command", "an-entry-printed", "a-list-by-a-names-index", "a-list-by-its-index",
        "the-deletes-own-operand", "the-folder-a-cd-takes-it-to", "a-dict-by-a-names-key"])
def test_a_shell_delete_holds_whole_where_its_own_operand_takes_a_dict_entrys_text(code, held, snap):
    """A delete holds whole where its operand, or the folder a `cd` takes
    it to, takes text the reading read from a dict entry by its key
    (`_Reader.entry_texts`, `_keyed_entry`): a cell may change the entry.
    An entry read for another word of the command, or a list's item by its
    index, holds nothing more than the words written out do."""
    assert _files(code, (snap,)) == held


def test_an_earlier_cell_too_deep_to_reread_stops_later_cells(snap, monkeypatch):
    """Where reading an earlier cell again for its stops runs out of room
    (`_earlier_stop`), that cell stops every later one, as one too deep for
    the parser does, and the check doesn't fail."""
    from anton.core.utils import write_check

    def too_deep(*args, **kwargs):
        raise RecursionError

    monkeypatch.setattr(write_check, "_LEFT_UNREAD", write_check._LeftUnread(limit=100_000))
    monkeypatch.setattr(write_check, "_left_unread", too_deep)
    writes = find_cell_writes(code="print(1)", earlier=_earlier(["x = 1"]), snapshots=(snap,))
    assert [(c.kind, c.in_earlier) for c in writes.causes] == [("limit", True)]


@pytest.mark.parametrize("run", [
    "subprocess.run(args=['sqlite3', 'app.db', sql])",
    "cmd = ['sqlite3', 'app.db', sql]\nsubprocess.run(cmd)",
    "subprocess.run(['sqlite3', 'app.db'] + [sql])",
    "subprocess.run(f'sqlite3 app.db \"{sql}\"', shell=True)",
    "subprocess.run(('sqlite3', 'app.db', sql))",
    "p = subprocess.Popen(['sqlite3', 'app.db'], stdin=subprocess.PIPE, text=True)\np.communicate(sql)",
    "p = subprocess.Popen(['sqlite3', 'app.db'], stdin=subprocess.PIPE, text=True)\np.communicate(input=sql)",
    "p = subprocess.Popen(['sqlite3', 'app.db'], stdin=subprocess.PIPE, text=True)\np.stdin.write(sql)",
], ids=["args-keyword", "a-list-in-a-name", "a-list-plus-a-list", "an-f-string-shell", "a-tuple", "communicate",
        "communicate-input", "stdin-write"])
def test_a_big_sql_file_an_earlier_cell_reads_is_held_where_a_program_gets_it_in_more_forms(run, sql_files):
    """CR-3 across cells: a program a call runs that gets the big file's
    text an earlier cell read, through a keyword, a name, a `+`, an
    f-string, a tuple, `communicate()` or `stdin.write()`
    (`_feeds_a_program`), gets the sql_file card its same-cell twin gets.
    Shown or handed to a call's other keyword, it holds nothing."""
    code = "import subprocess\n" + run
    same = find_cell_writes(code="sql = open('big.sql').read()\n" + code, earlier=_earlier(()),
                            snapshots=(sql_files,))
    assert [c.via for c in same.calls] == ["sql_file"]
    later = find_cell_writes(code=code, earlier=_earlier(["sql = open('big.sql').read()"]), snapshots=(sql_files,))
    assert [c.via for c in later.calls] == ["sql_file"]
    for other in ("print(len(sql))", "import subprocess\nsubprocess.run(['ls'], encoding=sql)"):
        assert not find_cell_writes(code=other, earlier=_earlier(["sql = open('big.sql').read()"]),
                                    snapshots=(sql_files,)).calls


@pytest.mark.parametrize("steps, branch", [(3, False), (8, False), (12, False), (12, True)],
                         ids=["three-steps", "eight-steps", "twelve-steps", "twelve-branches"])
def test_a_path_rebuilt_from_itself_in_straight_line_code_holds_one_value_per_step(steps, branch, snap):
    """Each step of a name rebuilt from itself reads the step before it, as
    Python runs it (`_rebuilds`), so the name holds one value per
    step: a dozen joins onto a new file stop nothing. A step under an if
    may or may not run, so each may hold one more variant, and past
    `_PATH_LIMIT` of them the check stops. The same steps onto a file that
    is there hold it within the depth limit, and past it stop, as their
    twin holds."""
    lines = "".join((f"if x{i}:\n    " if branch else "") + f"s = os.path.join(s, 'p{i}')\n" for i in range(steps))
    code = "import os\ns = 'out'\n" + lines + "open(os.path.join(s, 'x.csv'), 'w').write('x')"
    writes = find_cell_writes(code=code, earlier=_earlier(()), snapshots=(snap,))
    assert writes.stopped is branch and not writes.held_files
    there = "s = 'stock'\n" + "".join("s = s + ''\n" for _ in range(steps)) + "open(s + '.csv', 'w').write('x')"
    onto = find_cell_writes(code=there, earlier=_earlier(()), snapshots=(snap,))
    assert onto.stopped if steps >= 8 else [(c.action, c.paths) for c in onto.held_files] == [
        ("overwrite", ("stock.csv",))]


def test_each_load_of_a_name_rebuilt_many_times_reads_its_bindings_once(snap, monkeypatch):
    """With no errored cell, a read that counts a cell as maybe run in part
    gets what its whole run gives, so it shares that read
    (`_Reader.bindings_of`): each load of a name rebuilt 300 times reads
    the bindings it reaches once. Counted, not timed."""
    from anton.core.utils import write_check

    reads: list[str] = []
    real = write_check._Reader._bindings_afresh
    monkeypatch.setattr(write_check._Reader, "_bindings_afresh",
                        lambda self, name, **kw: reads.append(name) or real(self, name, **kw))
    code = "s = 'out'\n" + "".join(f"s = s + '_{i}'\n" for i in range(300)) + "open(s + '.csv', 'w').write('x')"
    find_cell_writes(code=code, earlier=_earlier(()), snapshots=(snap,))
    assert reads.count("s") <= 301


@pytest.mark.parametrize("code, earlier, expected", [
    ("import os\np = 'stock.csv'\ndef clean():\n    os.remove(p)\nclean()\np = p + '.bak'", (),
     [("delete", ("stock.csv",))]),
    ("import os, shutil\nd = 'output'\ndef clean():\n    shutil.rmtree(d)\nclean()\nd = os.path.join(d, 'new')", (),
     [("delete", ("output",))]),
    ("p = 'stock.csv'\ndef save():\n    open(p, 'w').write('y')\nsave()\np = 'other_' + p", (),
     [("overwrite", ("stock.csv",))]),
    ("import os\np = 'stock.csv'\nf = lambda: os.remove(p)\nf()\np = p + '.bak'", (), [("delete", ("stock.csv",))]),
    ("import os\np = 'stock.csv'\nclass C:\n    def go(self):\n        os.remove(p)\nC().go()\np = p + '.bak'", (),
     [("delete", ("stock.csv",))]),
    ("import os\np = 'stock.csv'\n[os.remove(p) for _ in range(1)]\np = p + '.bak'", (), [("delete", ("stock.csv",))]),
    ("import os\np = 'stock.csv'\nlist(os.remove(p) for _ in range(1))\np = p + '.bak'", (),
     [("delete", ("stock.csv",))]),
    ("import os\np = 'stock.csv'\nlist(map(lambda _: os.remove(p), [1]))\np = p + '.bak'", (),
     [("delete", ("stock.csv",))]),
    ("import os, threading\np = 'stock.csv'\nt = threading.Thread(target=lambda: os.remove(p))\nt.start()\n"
     "p = p + '.bak'", (), [("delete", ("stock.csv",))]),
    ("import os, shutil\np = 'output'\ndef go():\n    shutil.rmtree(p, ignore_errors=True)\ngo()\n"
     "p = os.path.join(p, 'new')\ngo()", (), [("delete", ("output",))]),
    ("clean()", ("import os\np = 'stock.csv'\ndef clean():\n    os.remove(p)\np = p + '.bak'",),
     [("delete", ("stock.csv",))]),
    # A slice back off a rebuilt name, which the check reads as the name itself.
    ("import os\np = 'stock.csv'\np = p + '.bak'\np = p[:-4]\nos.remove(p)", (), [("delete", ("stock.csv",))]),
    ("import os\np = 'stock.csv'\np = p + '.bak'\nq = p[:-4]\nos.remove(q)", (), [("delete", ("stock.csv",))]),
    ("import os\nout = 'report.pdf'\nout = out + '.tmp'\nopen(out, 'wb').write(b'%PDF')\nout = out[:-4]\n"
     "os.replace(out + '.tmp', out)", (), [("overwrite", ("report.pdf",))]),
], ids=["function", "function-rmtree", "function-open", "lambda", "method", "list-comprehension", "generator",
        "map-lambda", "thread-target", "called-before-and-after", "earlier-cell", "slice-back",
        "slice-to-another-name", "temp-file-replace"])
def test_a_read_of_a_rebuilt_name_outside_its_own_step_gets_each_value_the_name_took(code, earlier, expected, snap):
    """Only a step of a rebuilt name reads the value just before it
    (`_rebuilds`). Code that runs before a later step, as a
    function, a lambda or a comprehension may, reads an earlier value, and
    so may a read through a slice the check doesn't read: each holds what
    the earlier value names."""
    assert _files(code, (snap,), earlier=earlier) == expected


@pytest.mark.parametrize("code, action", [
    ("import os\nout = os.path.join(out)\nopen(out, 'w').write('x')", "overwrite"),
    ("out = out + ''\nimport os\nos.remove(out)", "delete"),
    ("import shutil\nout = str(out)\nshutil.copy('summary.md', out)", "overwrite"),
], ids=["an-overwrite", "a-delete", "a-copy-onto"])
@pytest.mark.parametrize("stopped_at", [2, None], ids=["stopped-at-its-raise", "line-unknown"])
def test_a_rebuild_after_a_cell_that_errored_before_its_last_binding_reads_each_of_its_values(
    code, action, stopped_at, snap,
):
    """A step of a rebuilt name reads the value just before it (`_latest`),
    but an earlier cell that errored may have stopped before its own last
    binding, so each of that cell's values counts: here `out` is still
    stock.csv, a file the turn began with, and the write holds it."""
    earlier = _earlier(["out = 'stock.csv'\nraise ValueError('x')\nout = 'new.csv'"], errored=[True],
                       stopped_at=[stopped_at])
    writes = find_cell_writes(code=code, earlier=earlier, snapshots=(snap,))
    assert [(c.action, c.paths) for c in writes.held_files] == [(action, ("stock.csv",))]


def test_a_functions_own_local_of_a_rebound_name_holds_nothing(snap):
    """A function's own local of a rebound name is its own: the command it
    runs is the one its own value names."""
    code = ("import os\nCMD = 'ls'\ndef setup():\n    global CMD\n    CMD = 'rm stock.csv'\n"
            "def run():\n    CMD = 'ls'\n    os.system(CMD)\nrun()")
    assert _files(code, (snap,)) == []


def test_a_value_stop_in_the_wide_reading_quotes_the_call_it_stopped_at(tmp_path, monkeypatch):
    """A value stop the wide reading makes, past an errored cell, quotes
    the call it stopped at, as one in the cell's own reading does."""
    (tmp_path / "data" / "out" / "daily").mkdir(parents=True)
    (tmp_path / "data" / "out" / "daily" / "stock.csv").write_text("x")
    monkeypatch.chdir(tmp_path)
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    deep = ["from pathlib import Path\nimport pandas as pd\nPROJECT = Path('.')", "DATA_DIR = PROJECT / 'data'",
            "OUT_DIR = DATA_DIR / 'out'", "DAILY = OUT_DIR / 'daily'", "name = 'stock.csv'",
            "df0 = pd.read_csv('missing.csv'); name = 'new.csv'"]
    earlier = _earlier(deep, errored=[False] * 5 + [True], stopped_at=[None] * 5 + [1])
    writes = find_cell_writes(code="pd.DataFrame().to_csv(DAILY / name)", earlier=earlier, snapshots=(snap,))
    assert writes.stopped
    assert [c.code for c in writes.causes if c.kind == "limit"] == ["pd.DataFrame().to_csv(DAILY / name)"]


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
    writes = find_cell_writes(code=code.format(root=str(shared.root)), earlier=_earlier(()), snapshots=(snap, shared))
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
    whole = find_cell_writes(code=f"import shutil\nshutil.rmtree({str(linked)!r})", earlier=_earlier(()), snapshots=(snap,))
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
    write = find_cell_writes(code=code, earlier=_earlier(()), snapshots=(snap,)).sql[0]
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
    writes = find_cell_writes(code="import os, shutil\n" + call, earlier=_earlier(()), snapshots=(snap,))
    assert writes.held_files[0].call == call
    assert _fenced(_card_prompt(writes)) == [call]
    nested = "import os\nfor d in os.environ['DIRS'].split():\n    os.remove(\n        d)"
    assert find_cell_writes(code=nested, earlier=_earlier(()), snapshots=(snap,)).held_files[0].call == "os.remove(\n    d)"
    helper = find_cell_writes(code="wipe(\n    os.environ['X'],\n)", earlier=_earlier((WIPE,)), snapshots=(snap,))
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
        return find_cell_writes(code=PG + code, earlier=_earlier(()), snapshots=(snap,)).sql[0].statement

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
    assert find_cell_writes(code=code, earlier=_earlier(()), snapshots=(snap,)).sql[0].values == values


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
                              earlier=_earlier(()), snapshots=())
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
                              earlier=_earlier(()), snapshots=(snapshot,))
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


def _capped_with_touched(*, tmp_path: Path, monkeypatch) -> tuple[ProjectSnapshot, str, str]:
    """A project past the snapshot cap, with a user's file the walk didn't
    record and a new file beside it, both changed during the turn."""
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
    return capped, touched, new


def test_past_the_cap_a_write_through_a_name_an_errored_cell_binds_makes_only_its_last_file_new(
    tmp_path, monkeypatch,
):
    """The errored cell may have stopped with `out` still on the user's file,
    but a write that only may have hit a file doesn't make it the turn's:
    its delete stays held."""
    capped, touched, new = _capped_with_touched(tmp_path=tmp_path, monkeypatch=monkeypatch)
    earlier = (f"out = {touched!r}\nx = 1 / 0\nout = {new!r}", "open(out, 'w').write('x')")
    for path, held in ((touched, [("delete", (touched,))]), (new, [])):
        writes = find_cell_writes(code=f"import os\nos.remove({path!r})", earlier=_earlier(earlier, errored=(True, False)),
                                  snapshots=(capped,))
        assert [(c.action, c.paths) for c in writes.held_files] == held, path


@pytest.mark.parametrize("earlier, errored, code", [
    (("d = '.'", "d = {folder!r}\nx = 1 / 0\nd = '.'", "import os\nos.chdir(d)\nopen({name!r}, 'w').write('x')"),
     (False, True, False), "import os\nos.remove({touched!r})"),
    (("out = {touched!r}\nx = 1 / 0\nout = {new!r}", "open(out, 'w').write('x')"), (True, False),
     "import glob, os\nfor f in glob.glob('{folder}/*.csv'):\n    os.remove(f)"),
    (("out = {touched!r}\nx = 1 / 0\nout = {new!r}", "open(out, 'w').write('x')"), (True, False),
     "import shutil\nshutil.move({touched!r}, '/tmp/elsewhere.csv')"),
    (("from pathlib import Path\np = Path({touched!r})\nx = 1 / 0\np = Path({new!r})", "p.write_text('x')"),
     (True, False), "from pathlib import Path\nPath({touched!r}).unlink()"),
], ids=["chdir", "glob-delete", "move", "pathlib-unlink"])
def test_past_the_cap_a_user_file_an_errored_cell_may_have_pointed_a_write_at_is_not_new(
    earlier, errored, code, tmp_path, monkeypatch,
):
    """A file counts as one the turn made only for a write that surely hit
    it, read with every earlier cell run to its end, its working folders
    too: the user's file stays one the turn began with, and its delete or
    move is held."""
    capped, touched, new = _capped_with_touched(tmp_path=tmp_path, monkeypatch=monkeypatch)
    folder, name = touched.split("/")
    fill = {"folder": folder, "name": name, "touched": touched, "new": new}
    writes = find_cell_writes(code=code.format(**fill), earlier=_earlier(tuple(c.format(**fill) for c in earlier), errored=errored),
                              snapshots=(capped,))
    assert touched in {p for c in writes.held_files for p in c.paths}


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
                              earlier=_earlier(()), snapshots=(sql_files,))
    assert writes.needs_approval()
    assert [c.via for c in writes.calls] == ["sql_file"]
    assert writes.sql == ()
    card = _card_prompt(writes)
    assert "This step would run a SQL file too big to check:" in card
    assert "open('big.sql').read()" in card


@pytest.mark.parametrize("run", [
    "c.executescript(open('{f}').read())",
    "from pathlib import Path\nc.execute(Path('{f}').read_text())",
    "sql = open('{f}').read()\nc.executescript(sql)",
], ids=["executescript-of-open-read", "execute-of-read-text", "executescript-of-a-name-read-first"])
@pytest.mark.parametrize("big", [True, False], ids=["past-the-cap", "under-the-cap"])
def test_a_sql_file_run_on_a_local_handle_is_held_past_the_cap_too(run, big, tmp_path, monkeypatch):
    """A local database handle's statements are read as a change to its
    file, so under the cap the DELETE holds app.db. Past the cap the check
    reads none of them, so the read is held as a SQL file too big to
    check, as one sent to a server is, under every value of the setting."""
    (tmp_path / "app.db").write_text("")
    (tmp_path / "db.sql").write_text("-- padding\n" * (100_000 if big else 0) + "DELETE FROM orders;\n")
    monkeypatch.chdir(tmp_path)
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    code = "import sqlite3\nc = sqlite3.connect('app.db')\n" + run.format(f="db.sql")
    writes = find_cell_writes(code=code, earlier=_earlier(()), snapshots=(snap,))
    assert not writes.unchecked
    if big:
        assert [c.via for c in writes.calls] == ["sql_file"]
    else:
        assert [(f.action, f.paths) for f in writes.held_files] == [("overwrite", ("app.db",))]


@pytest.mark.parametrize("read, run, runs", [
    ("sql = open('db.sql').read()", "import sqlite3\nsqlite3.connect('app.db').executescript(sql)",
     "sqlite3.connect('app.db').executescript(sql)"),
    ("from pathlib import Path\nseed = Path('db.sql').read_text()", PG + "cur.execute(seed)", "cur.execute(seed)"),
    ("sql = open('db.sql').read()",
     "import sqlite3\ndef seed():\n    sqlite3.connect('app.db').executescript(sql)\nseed()",
     "sqlite3.connect('app.db').executescript(sql)"),
], ids=["executescript-on-sqlite", "execute-on-postgres", "executescript-in-a-function"])
@pytest.mark.parametrize("big", [True, False], ids=["past-the-cap", "under-the-cap"])
def test_a_sql_file_an_earlier_cell_reads_is_held_past_the_cap_where_a_later_cell_runs_it(
    read, run, runs, big, tmp_path, monkeypatch,
):
    """A name an earlier cell bound to the file's text, handed to a runner as
    the statement it runs, is held as a SQL file too big to check past the
    cap, as a read in the same cell is (`_call_findings`); under the cap its
    statements are read, the DELETE holding app.db or the server's data.
    The card quotes the statement that runs it and the binding."""
    (tmp_path / "app.db").write_text("")
    (tmp_path / "db.sql").write_text("-- padding\n" * (100_000 if big else 0) + "DELETE FROM orders;\n")
    monkeypatch.chdir(tmp_path)
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    writes = find_cell_writes(code=run, earlier=_earlier([read]), snapshots=(snap,))
    assert writes.needs_approval() and not writes.unchecked
    files_or_sql = bool(writes.held_files or writes.sql)
    if big:
        assert [(c.via, c.call, c.runs) for c in writes.calls] == [("sql_file", runs, read.splitlines()[-1])]
        assert not files_or_sql
    else:
        assert writes.calls == () and files_or_sql


_LITE = "import sqlite3\ncon = sqlite3.connect('app.db')\n"


@pytest.mark.parametrize("read, run", [
    ("sql = open('db.sql').read()", PG + "for stmt in sql.split(';'):\n    if stmt.strip():\n        cur.execute(stmt)"),
    ("sql = open('db.sql').read()", PG + "[cur.execute(s) for s in sql.split(';') if s.strip()]"),
    ("sql = open('db.sql').read()", PG + "parts = [p for p in sql.split(';') if p.strip()]\nfor p in parts:\n    cur.execute(p)"),
    ("sql = open('db.sql').read()", PG + "for stmt in sql.strip().split(';'):\n    cur.execute(stmt)"),
    ("sql = open('db.sql').read()", PG + "for line in sql.splitlines():\n    cur.execute(line)"),
    ("sql = open('db.sql').read()", PG + "for stmt in filter(None, sql.split(';')):\n    cur.execute(stmt)"),
    ("sql = open('db.sql').read()", PG + "cur.execute(sql.split(';')[0])"),
    ("sql = open('db.sql').read()", _LITE + "for stmt in sql.split(';'):\n    con.execute(stmt)"),
    ("sql = open('db.sql').read()", PG + "def go(q=sql):\n    cur.execute(q)\ngo()"),
    ("sql = open('db.sql').read().strip()", PG + "cur.execute(sql)"),
    ("sql = ''\nsql += open('db.sql').read()", PG + "cur.execute(sql)"),
    ("sql = open('db.sql').read()", PG + "for s in sql.rsplit(';'):\n    cur.execute(s)"),
    ("sql = open('db.sql').read()", PG + "def go(*, q=sql):\n    cur.execute(q)\ngo()"),
    ("sql = open('db.sql').read()", PG + "go = lambda q=sql: cur.execute(q)\ngo()"),
    ("sql = open('db.sql').read()", PG + "import asyncio\nasync def go(q=sql):\n    cur.execute(q)\nasyncio.run(go())"),
    ("sql = open('db.sql').read()", PG + "def go(c, q=sql):\n    c.execute(q)\ngo(cur)"),
], ids=["split-in-a-loop", "split-in-a-comprehension", "split-then-a-loop", "stripped-then-split",
        "split-into-lines", "split-through-filter-none", "one-part-of-a-split", "split-on-a-local-handle",
        "a-parameters-default", "read-then-stripped", "added-to-an-empty-text", "rsplit", "a-keyword-only-default",
        "a-lambdas-default", "an-async-functions-default", "a-default-after-a-plain-parameter"])
@pytest.mark.parametrize("big", [True, False], ids=["past-the-cap", "under-the-cap"])
def test_a_sql_file_an_earlier_cell_reads_past_the_cap_is_held_in_each_form_its_twin_under_the_cap_is(
    read, run, big, tmp_path, monkeypatch,
):
    """Split into statements or lines a loop runs one at a time
    (`_SPLITS`, under `any_handle`), through `filter(None, ...)` or a
    parameter's default, the text still reaches a runner; and a read
    wrapped in the binding's value, or added to the name with `+=`
    (`_Reader.growth`), still binds the file's text
    (`_Reader.reads_an_oversized_sql_file`). Past the cap each is held as
    a SQL file too big to check, as its twin under the cap is held for the
    DELETE it reads."""
    (tmp_path / "app.db").write_text("")
    (tmp_path / "db.sql").write_text("-- padding\n" * (100_000 if big else 0) + "DELETE FROM orders;\n")
    monkeypatch.chdir(tmp_path)
    snap = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)
    writes = find_cell_writes(code=run, earlier=_earlier([read]), snapshots=(snap,))
    assert writes.needs_approval() and not writes.unchecked
    assert ([c.via for c in writes.calls] == ["sql_file"]) is big


@pytest.mark.parametrize("read, run", [
    ("queries = {'seed': open('big.sql').read()}", "cur.execute(queries['seed'])"),
    ("sql, n = open('big.sql').read(), 1", "cur.execute(sql)"),
    ("sql = f\"{open('big.sql').read()}\"", "cur.execute(sql)"),
    ("sql = open('big.sql').read() if n else ''", "cur.execute(sql)"),
    ("sql = open('big.sql').read() or 'SELECT 1'", "cur.execute(sql)"),
], ids=["an-entry-of-a-dict-display", "a-tuple-unpacked", "an-f-string", "an-if-else", "an-or"])
def test_a_sql_file_read_inside_an_earlier_bindings_value_is_held_past_the_cap_as_its_same_cell_twin_is(
    read, run, sql_files,
):
    """A file read anywhere inside the value an earlier cell binds the name
    to binds its text (`_Reader.reads_an_oversized_sql_file`), as the same
    read in the cell that runs it is held (the same-cell rule)."""
    later = find_cell_writes(code=PG + run, earlier=_earlier(["n = 1", read]), snapshots=(sql_files,))
    same = find_cell_writes(code=f"n = 1\n{read}\n{PG}{run}", earlier=_earlier(()), snapshots=(sql_files,))
    assert [c.via for c in later.calls] == [c.via for c in same.calls] == ["sql_file"]


def test_a_sql_file_cards_quote_of_an_earlier_binding_is_cut_as_other_quotes_are(sql_files):
    """The card quotes the binding of the file's text from an earlier cell
    scrubbed and cut (`_cut`), so a long statement there doesn't fill the
    card."""
    pad = "x" * 5_000
    earlier = [f"queries = {{'seed': open('big.sql').read(), 'pad': '{pad}'}}"]
    writes = find_cell_writes(code=PG + "cur.execute(queries['seed'])", earlier=_earlier(earlier),
                              snapshots=(sql_files,))
    [call] = writes.calls
    assert call.via == "sql_file" and call.runs.startswith("queries = {'seed'") and call.runs.endswith("...")
    assert len(call.runs) == 400


def test_a_sql_file_card_quotes_a_few_earlier_bindings_and_counts_the_rest(sql_files):
    """Many earlier cells that each add the file's text to a name give the
    card three of those bindings and how many more there are, not each."""
    earlier = ["sql = ''", *(f"sql += open('big.sql').read()  # part {i}" for i in range(8))]
    writes = find_cell_writes(code=PG + "cur.execute(sql)", earlier=_earlier(earlier), snapshots=(sql_files,))
    [call] = writes.calls
    assert call.via == "sql_file"
    assert call.runs.split("\n") == [*(f"sql += open('big.sql').read()  # part {i}" for i in range(3)),
                                     "# ... and 5 more like these"]


def test_a_conversation_that_builds_a_name_up_many_times_is_read_in_well_under_a_second(snap):
    """With no cell reading a SQL file past the cap, the cross-cell rule
    reads no name's additions (`_Reader.reads_oversized_sql_files`), so
    many loads of a name built up many times cost about what loads of a
    plain name do. CPU time, the cells parsed before the timed read."""
    def check(earlier: list[str]) -> float:
        code = "\n".join(["print(acc)"] * 1500)
        find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
        started = time.process_time()
        find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
        return time.process_time() - started

    assert check(["acc = ''", "\n".join(["acc += 'x'"] * 1500)]) < 2 * check(["acc = ''"]) + 0.1


def test_a_name_many_cells_add_a_sql_files_text_to_is_read_in_well_under_a_second(sql_files):
    """Each earlier binding of the file's text is quoted once per check of a
    call's code (`_call_findings`), not once per load of the name, so many
    runs of a name many cells built up cost about what runs of one built
    once do. CPU time, the cells parsed before the timed read."""
    def check(additions: int) -> float:
        earlier = _earlier(["sql = ''", *(f"sql += open('big.sql').read()  # {i}" for i in range(additions))])
        code = PG + "\n".join(["cur.execute(sql)"] * 150)
        find_cell_writes(code=code, earlier=earlier, snapshots=(sql_files,))
        started = time.process_time()
        find_cell_writes(code=code, earlier=earlier, snapshots=(sql_files,))
        return time.process_time() - started

    assert check(150) < 3 * check(1) + 0.1


def test_a_sql_file_an_earlier_cell_reads_past_the_cap_is_held_at_the_call_of_a_function_that_runs_it(sql_files):
    """Read for a call, an earlier cell's function that runs the name holds
    it at the call, quoting the binding and the statement."""
    earlier = ["sql = open('big.sql').read()", "import sqlite3\ndef seed():\n    sqlite3.connect('app.db').executescript(sql)"]
    writes = find_cell_writes(code="seed()", earlier=_earlier(earlier), snapshots=(sql_files,))
    assert [(c.via, c.call, c.runs) for c in writes.calls] == [
        ("sql_file", "seed()", "sql = open('big.sql').read()\nsqlite3.connect('app.db').executescript(sql)")]


@pytest.mark.parametrize("earlier, code", [
    (["a = open('big.sql').read()\nb = open('big.sql').read()"], PG + "cur.execute(a + b)"),
    ([], PG + "exec(\"a = open('big.sql').read()\\nb = open('big.sql').read()\")\ncur.execute(a + b)"),
], ids=["bound-in-an-earlier-cell", "bound-in-literal-code-the-cell-runs"])
def test_two_names_bound_to_sql_files_past_the_cap_in_one_statement_are_held_once(earlier, code, sql_files):
    """Each name bound outside the code read, in an earlier cell or in
    literal code the cell runs, counts; the statement that runs them is
    held once, with each binding."""
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(sql_files,))
    assert [(c.via, c.call, c.runs) for c in writes.calls] == [
        ("sql_file", "cur.execute(a + b)", "a = open('big.sql').read()\nb = open('big.sql').read()")]


def test_an_earlier_function_that_reads_a_sql_file_past_the_cap_and_runs_it_is_held_once(sql_files):
    """The read inside the function is held where it reaches the runner, so
    the name it binds there adds no second finding."""
    earlier = ["import sqlite3\ndef seed():\n    sql = open('big.sql').read()\n"
               "    sqlite3.connect('app.db').executescript(sql)"]
    writes = find_cell_writes(code="seed()", earlier=_earlier(earlier), snapshots=(sql_files,))
    assert [c.via for c in writes.calls] == ["sql_file"]


@pytest.mark.parametrize("earlier, code", [
    ("sql = open('big.sql').read()", "print(len(sql))"),
    ("sql = open('big.sql').read()", "import re\nprint(len(re.findall('^UPDATE', sql, flags=re.M)))"),
    ("sql = open('big.sql').read()", "import io\nimport pandas as pd\nprint(pd.read_csv(io.StringIO(sql)).shape)"),
    ("sql = 'SELECT 1'\nbig = open('big.sql').read()", PG + "cur.execute(sql)"),
], ids=["its-length-printed", "a-regex-over-it", "read-as-a-table", "another-name-the-cell-reads-it-into"])
def test_a_big_file_an_earlier_cell_reads_raises_no_card_where_no_runner_runs_it(earlier, code, sql_files):
    """The name's use reaches no runner as the statement it runs
    (`_reaches_a_runner`): a print, or a use the check can't follow, as a
    regex or a table read over a big text file is, holds nothing. Nor does
    a statement a name holds that the file's text isn't in
    (`_Reader.reads_an_oversized_sql_file`)."""
    assert not _held(code, sql_files, earlier=[earlier])


def test_a_long_shell_script_of_deletes_is_read_in_well_under_a_second(tmp_path, monkeypatch):
    """Thousands of deletes in one script of files the turn began with
    cost about what as many Python writes do: the card's code for the call
    is scrubbed once (`_FileFinder.quote`), not once for each delete. CPU
    time, the cell parsed before the timed read."""
    for i in range(300):
        (tmp_path / f"f{i}.csv").write_text("x")
    monkeypatch.chdir(tmp_path)
    project = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60)

    def check(code: str, held: int) -> float:
        find_cell_writes(code=code, earlier=_earlier(()), snapshots=(project,))
        started = time.process_time()
        writes = find_cell_writes(code=code, earlier=_earlier(()), snapshots=(project,))
        assert len(writes.held_files) == held
        return time.process_time() - started

    script = "import os\nos.system(" + repr("; ".join(f"rm f{i % 300}.csv" for i in range(3000))) + ")"
    python = "\n".join(f"open('f{i % 300}.csv', 'w').write('x')" for i in range(3000))
    assert check(script, 3000) < 3 * check(python, 300) + 0.1


def test_names_bound_to_reads_of_a_big_sql_file_look_at_each_cells_reads_once(sql_files, monkeypatch):
    """Whether a name an earlier cell binds holds a read of a SQL file past
    the cap is one look at its cell's file reads
    (`_Reader.reads_an_oversized_sql_file`), not a walk of every file read
    of the cell for each name: counted, not timed."""
    from anton.core.utils import write_check

    looked: list[ast.Call] = []
    real = write_check._sql_file_path
    monkeypatch.setattr(write_check, "_sql_file_path", lambda call, reader: looked.append(call) or real(call, reader))
    earlier = ["\n".join(f"v{i} = open('big.sql').read()" for i in range(200))]
    find_cell_writes(code="print(" + ", ".join(f"v{i}" for i in range(200)) + ")", earlier=_earlier(earlier),
                     snapshots=(sql_files,))
    assert len(looked) <= 3 * 200


def test_a_sql_files_size_is_read_from_disk_once_per_check(sql_files, monkeypatch):
    """Many reads of one literal path cost one look at the disk
    (`_Reader.oversized_sql_file`)."""
    from anton.core.utils import write_check

    looked: list[str] = []
    past_the_cap = write_check._Reader._past_the_cap
    monkeypatch.setattr(write_check._Reader, "_past_the_cap",
                        lambda self, path: looked.append(path) or past_the_cap(self, path))
    earlier = ["\n".join(f"v{i} = open('big.sql').read()" for i in range(20))]
    writes = find_cell_writes(code=PG + "cur.execute(v0 + v1)", earlier=_earlier(earlier), snapshots=(sql_files,))
    assert [c.via for c in writes.calls] == ["sql_file"]
    assert looked == ["big.sql"]


def test_a_call_of_an_earlier_function_that_runs_no_sql_file_holds_none_its_cell_ran_past_the_cap(sql_files):
    """Read for a call, only the function's own code counts
    (`_Cell.loads_in`): the big file its cell ran at its top level was held
    when that cell ran, so a call of a function that only prints holds
    nothing, where a function that runs it is held at the call."""
    earlier = ["sql = open('big.sql').read()",
               "import sqlite3\ndef hello():\n    print('hi')\nsqlite3.connect('app.db').executescript(sql)"]
    assert not _held("hello()", sql_files, earlier=earlier)


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
    writes = find_cell_writes(code=code, earlier=_earlier(()), snapshots=(snap,))
    assert writes.needs_approval()
    assert [c.via for c in writes.calls] == [via]


ORA = "import oracledb\nconn = oracledb.connect(dsn='x')\ncur = conn.cursor()\n"
FUNC = "cur.callfunc('refresh_totals', int, [1, 2])"


@pytest.mark.parametrize("code, earlier, call, line, fences", [
    (FUNC, (), FUNC, 1, [FUNC]),
    (FUNC, (ORA,), FUNC, 1, [FUNC]),
    (ORA + FUNC, (), FUNC, 4, [FUNC]),
    ("refresh()", (ORA + "def refresh():\n    " + FUNC,), "refresh()", 1, ["refresh()", FUNC]),
    ("result = cur.callfunc(name='refresh_totals', return_type=int, parameters=[1, 2])", (ORA,),
     "result = cur.callfunc(name='refresh_totals', return_type=int, parameters=[1, 2])", 1,
     ["result = cur.callfunc(name='refresh_totals', return_type=int, parameters=[1, 2])"]),
], ids=["direct", "earlier-cursor", "oracle-cursor", "earlier-helper", "keywords"])
def test_stored_function_calls_are_held_and_quote_the_call(code, earlier, call, line, fences, snap):
    """The card quotes the call, and for a helper an earlier cell defined,
    the routine call inside it too."""
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    assert writes.needs_approval()
    assert [(c.via, c.call, c.line) for c in writes.calls] == [("procedure", call, line)]
    assert _fenced(_card_prompt(writes)) == fences
    assert "This step would run a stored procedure or function:" in _card_prompt(writes)


KEY = "sk-ant-api03-" + "A" * 40


@pytest.mark.parametrize("earlier, code", [
    ((ORA + f"def refresh():\n    cur.callfunc('refresh_totals', int, ['{KEY}'])",), "refresh()"),
    ((ORA + f"import functools\nrefresh = functools.partial(cur.callfunc, 'refresh_totals', int, ['{KEY}'])",),
     "refresh([1])"),
], ids=["earlier-helper", "earlier-partial"])
def test_a_secret_in_the_code_a_call_runs_is_scrubbed_on_the_card(earlier, code, snap):
    """The card's "That call runs:" block quotes an earlier cell's code, so
    it is scrubbed the same way the call is."""
    card = _card_prompt(find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,)))
    assert "That call runs:" in card
    assert KEY not in card and "[REDACTED_API_KEY]" in card


@pytest.mark.parametrize("code, earlier", [
    (ORA + "run = cur.callfunc\nrun('refresh_totals', int, [1, 2])", ()),
    ("run('refresh_totals', int, [1, 2])", (ORA + "run = cur.callproc",)),
    (ORA + "import functools\nrefresh = functools.partial(cur.callfunc, 'refresh_totals', int)\nrefresh([1, 2])", ()),
    (ORA + "pool.submit(cur.callproc, 'purge_suppliers', [1])", ()),
    (ORA + "routines = {'purge': cur.callproc}\nroutines['purge']('purge_suppliers', [1])", ()),
    (ORA + "getattr(cur, 'callfunc')('refresh_totals', int, [1, 2])", ()),
], ids=["alias", "earlier-alias", "partial", "submitted", "in-a-dict", "getattr"])
def test_a_routine_call_handed_on_as_a_value_is_held(code, earlier, snap):
    """A stored procedure or function bound, wrapped or passed on still runs
    the routine, so the card holds the cell."""
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    assert writes.needs_approval()
    assert {c.via for c in writes.calls} == {"procedure"}


SQLITE = "import sqlite3\nconn = sqlite3.connect(':memory:')\ncur = conn.cursor()\n"


@pytest.mark.parametrize("earlier, code, call, runs", [
    ((ORA + "import functools\npurge = functools.partial(cur.callproc, 'purge_suppliers')",),
     "for sid in range(100):\n    purge([sid])", "purge([sid])", "purge = functools.partial(cur.callproc, 'purge_suppliers')"),
    ((ORA + "from functools import partial\nrun = partial(cur.callfunc)",),
     "run('drop_all_suppliers', int, [])", "run('drop_all_suppliers', int, [])", "run = partial(cur.callfunc)"),
    ((ORA + "routines = {'purge': cur.callproc}",),
     "routines['purge']('purge_suppliers', [1])", "routines['purge']('purge_suppliers', [1])",
     "routines = {'purge': cur.callproc}"),
    ((ORA + "routines = {}\nroutines['purge'] = cur.callproc",),
     "routines['purge']('purge_suppliers', [1])", "routines['purge']('purge_suppliers', [1])",
     "routines['purge'] = cur.callproc"),
    ((ORA + "steps = [cur.callproc]",),
     "steps[0]('purge_suppliers', [1])", "steps[0]('purge_suppliers', [1])", "steps = [cur.callproc]"),
    ((ORA + "run = cur.callfunc", "go = run"),
     "go('purge_suppliers', int, [1])", "go('purge_suppliers', int, [1])", "run = cur.callfunc\ngo = run"),
    ((ORA + "run, other = cur.callfunc, 1",),
     "run('purge_suppliers', int, [1])", "run('purge_suppliers', int, [1])", "run, other = cur.callfunc, 1"),
    ((ORA + "class Holder:\n    pass\nh = Holder()\nh.run = cur.callfunc",),
     "h.run('purge_suppliers', int, [1])", "h.run('purge_suppliers', int, [1])", "h.run = cur.callfunc"),
    ((ORA + "f = getattr(cur, 'callfunc', None)\nprint(f is None)",),
     "f('purge_suppliers', int, [1])", "f('purge_suppliers', int, [1])", "f = getattr(cur, 'callfunc', None)"),
    ((ORA + "import functools\npurge = functools.partial(cur.callproc, 'purge_suppliers')",),
     "pool.submit(purge, [1])", "pool.submit(purge, [1])", "purge = functools.partial(cur.callproc, 'purge_suppliers')"),
    ((ORA + "run = cur.callfunc",), "def make():\n    return run\npool.submit(make())", "return run",
     "run = cur.callfunc"),
    ((ORA + "run = getattr(cur, 'call' 'func')",),
     "run('purge_suppliers', int, [1])", "run('purge_suppliers', int, [1])", "run = getattr(cur, 'call' 'func')"),
    ((ORA + "run = getattr(cur, 'call\\x66unc')",),
     "run('purge_suppliers', int, [1])", "run('purge_suppliers', int, [1])", "run = getattr(cur, 'call\\x66unc')"),
    ((ORA + "run = cur.\uff43\uff41\uff4c\uff4c\uff46\uff55\uff4e\uff43",),
     "run('purge_suppliers', int, [1])", "run('purge_suppliers', int, [1])",
     "run = cur.\uff43\uff41\uff4c\uff4c\uff46\uff55\uff4e\uff43"),
    ((ORA + "run = cur.callfunc if ok else None",),
     "run('purge_suppliers', int, [1])", "run('purge_suppliers', int, [1])", "run = cur.callfunc if ok else None"),
    ((ORA + "run = cur.callfunc or None",),
     "run('purge_suppliers', int, [1])", "run('purge_suppliers', int, [1])", "run = cur.callfunc or None"),
    ((ORA + "routines = {'purge': cur.callproc}",),
     "routines.get('purge')('purge_suppliers', [1])", "routines.get('purge')('purge_suppliers', [1])",
     "routines = {'purge': cur.callproc}"),
    ((ORA + "routines = {'purge': cur.callproc}",),
     "for f in routines.values():\n    f('purge_suppliers', [1])", "f('purge_suppliers', [1])",
     "routines = {'purge': cur.callproc}\nfor f in routines.values():"),
    ((ORA + "routines = {'purge': cur.callproc}",),
     "for name, f in routines.items():\n    f('purge_suppliers', [1])", "f('purge_suppliers', [1])",
     "routines = {'purge': cur.callproc}\nfor name, f in routines.items():"),
    ((ORA + "steps = [cur.callproc]",),
     "steps.pop()('purge_suppliers', [1])", "steps.pop()('purge_suppliers', [1])", "steps = [cur.callproc]"),
    # The first element reaches `r` only at the depth limit, which counts it.
    ((ORA + "r = cur.callfunc if ok else None\nb1 = r\nb2 = b1\nb3 = b2\nb4 = b3\ngo = [b4, r]",),
     "go[1]('purge_suppliers', int, [1])", "go[1]('purge_suppliers', int, [1])",
     "r = cur.callfunc if ok else None\nb1 = r\nb2 = b1\nb3 = b2\nb4 = b3\ngo = [b4, r]"),
], ids=["partial-in-a-loop", "from-functools-partial", "dict", "dict-entry-stored", "list", "alias-of-alias",
        "tuple-unpack", "member-store", "getattr-default", "partial-submitted", "returned-then-submitted",
        "getattr-concatenated", "getattr-escaped", "attribute-fullwidth", "conditional", "or-default", "dict-get",
        "dict-values-loop", "dict-items-loop", "list-pop", "reached-again-from-nearer"])
def test_a_routine_bound_in_an_earlier_cell_is_held_where_it_runs(earlier, code, call, runs, snap):
    """A stored procedure or function an earlier cell binds runs in the cell
    that calls it or hands it on: the card holds that cell, quotes the call
    and shows the code that bound it. On an in-process handle nothing is held."""
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    assert [(c.via, c.call, c.runs) for c in writes.calls] == [("procedure", call, runs)]
    local = [e.replace(ORA, SQLITE) for e in earlier]
    assert not _held(code, snap, earlier=local)


NESTED = ORA + "r = cur.callfunc\na1 = [r]\na2 = [a1]\na3 = [a2]\na4 = [a3]"


@pytest.mark.parametrize("earlier, code, call", [
    ((ORA + "a0 = cur.callfunc", *(f"a{i} = a{i - 1}" for i in range(1, 11))),
     "a10('purge_suppliers', int, [1])", "a10('purge_suppliers', int, [1])"),
    ((ORA + "a0 = cur.callfunc",),
     "".join(f"a{i} = a{i - 1}\n" for i in range(1, 11)) + "pool.submit(a10, 'purge_suppliers', int, [1])",
     "pool.submit(a10, 'purge_suppliers', int, [1])"),
    ((ORA + "a0 = cur.callfunc", *(f"a{i} = a{i - 1}" for i in range(1, 9))),
     "a8('purge_suppliers', int, [1])", "a8('purge_suppliers', int, [1])"),
    ((ORA + "a0 = cur.callfunc", *(f"a{i} = a{i - 1}" for i in range(1, 9))),
     "pool.submit(a8, 'purge_suppliers', int, [1])", "pool.submit(a8, 'purge_suppliers', int, [1])"),
    ((ORA + "import functools\np0 = cur.callfunc", *(f"p{i} = functools.partial(p{i - 1})" for i in range(1, 9))),
     "p8('purge_suppliers', int, [1])", "p8('purge_suppliers', int, [1])"),
    ((NESTED,), "a4[0][0][0][0]('purge_suppliers', int, [1])", "a4[0][0][0][0]('purge_suppliers', int, [1])"),
    ((NESTED,), "pool.submit(a4)", "pool.submit(a4)"),
    ((ORA + "a0 = getattr(cur, 'callfunc', fallback)", *(f"a{i} = a{i - 1}" for i in range(1, 9))),
     "a8('purge_suppliers', int, [1])", "a8('purge_suppliers', int, [1])"),
], ids=["called", "handed-on", "eight-aliases", "eight-aliases-handed-on", "eight-partials", "nested-four-deep",
        "nested-four-deep-handed-on", "eight-aliases-of-a-getattr"])
def test_a_name_that_may_hold_a_routine_past_the_follow_limit_is_held(earlier, code, call, snap):
    """The check follows a chain of aliases only so far. A name past that
    limit that may still hold a routine is held where it is called or
    handed on, even when the chain's last links are in the same cell."""
    writes = find_cell_writes(code=code, earlier=_earlier(earlier), snapshots=(snap,))
    assert [(c.via, c.call) for c in writes.calls] == [("procedure", call)]


def test_a_routine_reached_first_past_the_limit_is_held_where_it_is_read_directly(snap):
    """`g(d4)` reaches `v`'s routine only past the depth limit; the next
    line reads `v` directly. Each read is held on its own path."""
    earlier = ORA + "x1 = cur.callfunc\nx2 = x1\nx3 = x2\nv = [x3, 0]\nd1 = v\nd2 = d1\nd3 = d2\nd4 = d3"
    writes = find_cell_writes(code="g(d4)\nv[0]('purge_suppliers', int, [1])", earlier=_earlier((earlier,)),
                              snapshots=(snap,))
    assert ("procedure", "v[0]('purge_suppliers', int, [1])", "x1 = cur.callfunc\nx2 = x1\nx3 = x2\nv = [x3, 0]") in [
        (c.via, c.call, c.runs) for c in writes.calls
    ]


@pytest.mark.parametrize("code, calls", [
    (ORA + "run = cur.callfunc\nrun('refresh_totals', int, [1, 2])",
     [("run('refresh_totals', int, [1, 2])", "run = cur.callfunc")]),
    (ORA + "import functools\nf = functools.partial(cur.callfunc, 'refresh_totals', int)\nf([1])\nf([2])",
     [("f([1])", "f = functools.partial(cur.callfunc, 'refresh_totals', int)"),
      ("f([2])", "f = functools.partial(cur.callfunc, 'refresh_totals', int)")]),
    (ORA + "routines = {'purge': cur.callproc}\nroutines['purge']('purge_suppliers', [1])",
     [("routines['purge']('purge_suppliers', [1])", "routines = {'purge': cur.callproc}")]),
    (ORA + "run = cur.callproc\nrun('purge_suppliers', [1])\npool.submit(run, 'purge_suppliers', [2])",
     [("run = cur.callproc", ""), ("run('purge_suppliers', [1])", "run = cur.callproc")]),
    (ORA + "run = cur.callproc\nrun('purge_suppliers', [1])\ndef later():\n    return run",
     [("run = cur.callproc", ""), ("run('purge_suppliers', [1])", "run = cur.callproc")]),
], ids=["alias", "partial-called-twice", "dict", "also-submitted", "also-returned"])
def test_a_routine_bound_and_called_in_one_cell_is_quoted_at_each_call(code, calls, snap):
    """The card shows each call once, with the binding that says which
    routine and fixed arguments it runs. The binding keeps its own block
    when the method is also handed on where the check can't follow it."""
    writes = find_cell_writes(code=code, earlier=_earlier(()), snapshots=(snap,))
    assert [(c.call, c.runs) for c in writes.calls] == calls


def test_a_routine_name_read_through_nested_lists_is_followed_once_per_binding(snap, monkeypatch):
    """Every element of each list reads the same binding, so the check
    follows each binding once: the work grows with the lists' width, not as
    its power."""
    from anton.core.utils import write_check

    reads = 0
    bindings = write_check._Reader.bindings_of

    def counting(*args, **kwargs):
        nonlocal reads
        reads += 1
        return bindings(*args, **kwargs)

    monkeypatch.setattr(write_check._Reader, "bindings_of", counting)

    def binding_reads(width: int) -> int:
        nonlocal reads
        reads = 0
        earlier = SQLITE + "r = cur.callfunc if cur else None\n" + "".join(
            f"a{level} = [{', '.join([f'a{level - 1}' if level > 1 else 'r'] * width)}]\n" for level in range(1, 5)
        )
        assert not _held("print(len(a4))", snap, earlier=(earlier,))
        return reads

    assert binding_reads(24) <= 8 * binding_reads(3)
    started = time.perf_counter()
    binding_reads(45)
    assert time.perf_counter() - started < 5


def test_a_long_alias_chain_of_a_routine_is_read_in_well_under_a_second(snap):
    """The names that may hold a routine grow one link at a time down a
    chain of aliases, and each link is followed once."""
    earlier = ORA + "a0 = cur.callfunc\n" + "".join(f"a{i} = a{i - 1}\n" for i in range(1, 6000))
    started = time.perf_counter()
    assert not _held("print(len([1, 2]))", snap, earlier=(earlier,))
    assert time.perf_counter() - started < 1
    assert _held("a5999('purge_suppliers', int, [1])", snap, earlier=(earlier,))


def test_a_stored_function_name_is_not_a_call(snap):
    assert not _held("print('cur.callfunc(\\'refresh_totals\\', int)')", snap)
    assert not _held("import sqlite3\nc = sqlite3.connect(':memory:')\nc.callfunc('p', int)", snap)


@pytest.mark.parametrize("code", [
    ORA + "run = cur.callfunc",
    ORA + "run = cur.callfunc if ok else None",
], ids=["bound-and-unused", "bound-in-a-conditional"])
def test_a_routine_bound_and_not_run_here_is_held_where_it_is_bound(code, snap):
    """A later cell may run the routine where the check can't follow it,
    so the cell that binds it holds."""
    writes = find_cell_writes(code=code, earlier=_earlier(()), snapshots=(snap,))
    assert [(c.via, c.call) for c in writes.calls] == [("procedure", code.removeprefix(ORA))]


@pytest.mark.parametrize("code", [
    ORA + "print(cur.callfunc)",
    ORA + "ok = callable(cur.callfunc)",
    ORA + "if cur.callfunc is None:\n    raise SystemExit('no callfunc')",
], ids=["print", "callable", "is-none"])
def test_a_driver_routine_looked_at_where_it_stands_is_held(code, snap):
    """A driver cursor's routine method read anywhere but as the called
    function is held where it is read."""
    writes = find_cell_writes(code=code, earlier=_earlier(()), snapshots=(snap,))
    assert writes.needs_approval() and [c.via for c in writes.calls] == ["procedure"]


@pytest.mark.parametrize("code, earlier", [
    ("ok = callable(f)", ()),
    ("help(f)", ()),
    ("print(f)", ()),
    ("print(sep=f)", ()),
    ("print([f])", ()),
    ("callable(f)", ("callable = lambda g: g('purge_suppliers', int, [1])",)),
    ("def help(g):\n    return g\nhelp(f)('purge_suppliers', int, [1])", ()),
], ids=["callable", "help", "print", "keyword", "in-a-container", "rebound-builtin", "returned-and-called"])
def test_a_routine_looked_at_where_it_may_still_run_is_held(code, earlier, snap):
    """A routine an earlier cell binds, handed to a builtin or any other
    call, may run there: the cell that hands it on holds."""
    writes = find_cell_writes(code=code, earlier=_earlier((ORA + "f = cur.callfunc", *earlier)), snapshots=(snap,))
    assert [(c.via, c.runs) for c in writes.calls] == [("procedure", "f = cur.callfunc")]


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
    writes = find_cell_writes(code="conn.execute(suppliers.delete())", earlier=_earlier(()), snapshots=(snap,))
    card = _card_prompt(writes)
    assert "This step would change data through SQLAlchemy:" in card
    assert _fenced(card) == ["conn.execute(suppliers.delete())"]


def test_a_store_write_quotes_the_code_line_on_the_card(snap):
    writes = find_cell_writes(code="db.users.insert_one({'id': 1})", earlier=_earlier(()), snapshots=(snap,))
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
    writes = find_cell_writes(code=code, earlier=_earlier(()), snapshots=(snap,))
    assert writes.needs_approval()
    assert [c.via for c in writes.calls] == [via]


def test_a_bigquery_table_delete_names_bigquery_on_the_card(snap):
    card = _card(BQ + "client.delete_table('p.d.orders')", snap)
    assert "This step would change data in BigQuery:" in card
    assert "DynamoDB" not in card
    assert "This step would change data in a data store:" in _card("client.delete_table('p.d.orders')", snap)


def test_a_call_write_inside_an_earlier_function_is_held_at_the_call(snap):
    earlier = ("def wipe(coll):\n    coll.delete_many({})",)
    writes = find_cell_writes(code="wipe(users)", earlier=_earlier(earlier), snapshots=(snap,))
    assert [(c.via, c.line, c.call) for c in writes.calls] == [("mongodb", 1, "wipe(users)")]


def test_the_write_rule_matcher_is_exported() -> None:
    """The relay imports it by name; the module's export list says so too."""
    from anton.core.utils import write_check

    assert "http_write_rule_matches" in write_check.__all__
