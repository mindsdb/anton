"""What `find_cell_writes` holds, and what it lets through.

A missed write runs with no approval. A false hold costs an unattended answer
the full card timeout and then the answer itself. So each rule here has cells
it must hold and cells, often the same call shape, it must not.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from anton.core.utils.write_check import (
    find_cell_writes,
    glob_match,
    snapshot_project_files,
    write_statement,
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


def _held(code: str, snap, *, earlier=(), connected=False) -> bool:
    return find_cell_writes(
        code=code, earlier_code=earlier, snapshots=(snap,), connected=connected,
    ).needs_approval()


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
])
def test_write_statements(sql):
    assert write_statement(sql) is not None


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
])
def test_reads_and_prose(text):
    assert write_statement(text) is None


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
], ids=["literal", "multi-line", "f-string", "concat", "named", "psycopg2-sql", "sqlalchemy-text",
        "pad-helper", "to_sql", "list-loop", "duckdb-attach", "motherduck"])
def test_cells_that_write_connected_data(code, snap):
    assert _held(code, snap, connected=True)


def test_ds_text_alone_makes_a_cell_connected(snap):
    assert _held(PG + "cur.execute('UPDATE t SET a = 1')", snap, connected=False)
    creds = "import os\npg = {k: v for k, v in os.environ.items() if k.startswith('DS_')}\n"
    assert _held(creds + "conn.cursor().execute('UPDATE t SET a = 1')", snap)


def test_a_write_named_in_an_earlier_cell_and_run_here(snap):
    earlier = (PG + "q = 'DELETE FROM suppliers WHERE id = 5'",)
    assert _held("cur.execute(q)", snap, earlier=earlier)


def test_a_connection_from_an_earlier_cell(snap):
    assert _held("cur.execute('UPDATE t SET a = 1')", snap, earlier=(PG,))


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
], ids=["select", "printed", "logged", "written-to-file", "docstring", "unused-name", "duckdb-csv",
        "duckdb-module", "duckdb-alias", "sqlite-memory", "sqlite-file", "sqlalchemy-sqlite",
        "local-helper", "local-loop", "local-to_sql", "prose"])
def test_cells_that_change_no_connected_data(code, snap):
    assert not _held(code, snap, connected=True)


def test_sql_is_unchecked_without_a_connection(snap):
    assert not _held("c = make()\nc.execute('UPDATE t SET a = 1')", snap, connected=False)


def test_the_card_text_holds_the_whole_statement_and_scrubs_it(snap):
    code = PG + 'cur.execute("""UPDATE suppliers\n   SET lead_time_days = 21\n   WHERE supplier_id = \'S005\'""")'
    writes = find_cell_writes(code=code, earlier_code=(), snapshots=(snap,), connected=True)
    assert writes.sql[0].statement == "UPDATE suppliers SET lead_time_days = 21 WHERE supplier_id = 'S005'"


def test_a_long_concatenation_does_not_exhaust_recursion(snap):
    code = "html = " + " + ".join(f"'<td>{i}</td>'" for i in range(3000)) + "\nprint(html)"
    assert not _held(code, snap, connected=True)


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
], ids=["os.remove", "os.unlink", "os.rename", "shutil.move", "Path.unlink", "Path.rename",
        "next-glob", "sorted-glob", "comprehension", "target-kw", "glob-loop", "map", "listdir",
        "absolute", "rmtree-folder", "rmtree-root", "rmtree-cwd", "rmtree-git", "artifact", "rmdir",
        "subprocess-rm", "os.system-mv", "aliased-os", "from-import", "env-path", "after-chdir",
        "helper-param"])
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
    "import os\nos.remove('.anton/memory/notes.md')",
    "import os\ndef write_atomic(path, text):\n    tmp = path + '.tmp'\n    open(tmp, 'w').write(text)\n    os.replace(tmp, path)\nwrite_atomic('lead_times.md', 's')",
    "import os\nfor f in os.listdir('pages_tmp'):\n    os.remove(os.path.join('pages_tmp', f))",
    "open('new_summary.md', 'w').write('x')",
    "from pathlib import Path\np = Path.cwd() / ('beat-' + TOKEN + '.json')\ntmp = p.with_suffix('.tmp')\ntmp.write_text('x')\ntmp.replace(p)",
    "import os\nown = 'move-in-' + str(os.getpid())\nopen(own, 'w').close()\nfor path in (own, own + '.moved'):\n    os.remove(path)",
], ids=["df-rename-columns", "series-rename", "series-replace", "df-rename-callable", "str-replace",
        "temp-renamed-into-place", "tempfile-name", "mkstemp", "tempdir", "outside-project",
        "new-file-in-existing-folder", "new-path-in-existing-folder", "move-into-folder",
        "new-file", "anton-own-files", "helper-temp-file", "cleanup-new-folder", "new-summary",
        "with-suffix-temp", "tuple-loop-cleanup"])
def test_cells_that_change_no_existing_file(code, snap):
    assert not _held(code, snap)


@pytest.mark.parametrize("code", [
    "open('summary.md', 'w').write('new')",
    "import os\nos.remove('summary.md')\nopen('summary.md', 'w').write('new')",
    "from pathlib import Path\nout = Path('summary.md')\nout.unlink(missing_ok=True)\nout.write_text('new')",
    "import os\nopen('summary.md.tmp', 'w').write('x')\nos.replace('summary.md.tmp', 'summary.md')",
])
def test_an_overwrite_is_reported_not_held(code, snap):
    writes = find_cell_writes(code=code, earlier_code=(), snapshots=(snap,), connected=False)
    assert not writes.needs_approval()
    assert [c.action for c in writes.files] and {c.action for c in writes.files} == {"overwrite"}


def test_cards_name_the_files(snap):
    writes = find_cell_writes(code="import glob, os\nfor f in glob.glob('*.pdf'):\n    os.remove(f)",
                              earlier_code=(), snapshots=(snap,), connected=False)
    assert writes.held_files[0].paths == ("report.pdf",), "a root glob doesn't reach docs/"
    picked = find_cell_writes(code="import os\nos.remove(os.environ['X'])",
                              earlier_code=(), snapshots=(snap,), connected=False)
    assert picked.held_files[0].paths == () and "os.environ['X']" in picked.held_files[0].call
    whole = find_cell_writes(code="import shutil\nshutil.rmtree('.')",
                             earlier_code=(), snapshots=(snap,), connected=False)
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
    assert _held("cur.execute('UPDATE t SET a = 1')", snap, earlier=("def broken(:", PG), connected=True)


# ── Snapshots ─────────────────────────────────────────────────────────────

def test_the_snapshot_keeps_artifacts_and_skips_the_hosts_own_folders(snap):
    assert ".anton/artifacts/dash/index.html" in snap.files
    assert ".anton/memory/notes.md" not in snap.files
    assert ".anton" in snap.dirs
    assert ".git" in snap.unwalked and ".git/index" not in snap.files
    assert snap.holds(rel=".git/index"), "a path under an unwalked folder counts as existing"


def test_past_the_cap_an_older_file_still_counts(tmp_path):
    for i in range(30):
        (tmp_path / f"f{i}.csv").write_text("x")
    capped = snapshot_project_files(root=tmp_path, label="project", taken_at=time.time() + 60, cap=10)
    assert not capped.complete
    assert all(capped.holds(rel=f"f{i}.csv") for i in range(30))
    assert not capped.holds(rel="missing.csv")


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
