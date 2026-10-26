"""Synthetic analyst cells checked as source text, never executed.

The no-card corpus is read and report work, writes the gate allows by design,
accepted limitations, and read work that still gets a card, each in its own
dict so neither an allowed write nor a missed write is described as a read.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from anton.core.utils.write_check import ProjectSnapshot, find_cell_writes, snapshot_project_files

PG = ("import os, psycopg2\nconn = psycopg2.connect(host=os.environ['DS_POSTGRES_QA__HOST'], "
      "dbname=os.environ['DS_POSTGRES_QA__DATABASE'])\ncur = conn.cursor()\n")
SA = ("import os\nimport pandas as pd\nfrom sqlalchemy import create_engine, text\n"
      "engine = create_engine(os.environ['DS_POSTGRES_QA__URL'])\n")

READ_AND_REPORT_CASES: dict[str, str] = {
    "reportlab-recommendations": """from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
styles = getSampleStyleSheet()
story = [Paragraph('Q3 Supplier Review', styles['Title'])]
recs = [
    'Merge the two supplier lists using the supplier ID.',
    'Delete orders where the quantity is zero before the next import.',
    'Update the lead time for S005 to 21 days and set a review date.',
    'Drop index funds below 2% weight from the watchlist.',
    'Insert the table of lead times below the summary.',
]
for r in recs:
    story.append(Paragraph(r, styles['BodyText']))
    story.append(Spacer(1, 6))
SimpleDocTemplate('q3_review.pdf', pagesize=letter).build(story)""",
    "reportlab-next-steps-paragraph": """from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
s = getSampleStyleSheet()
text = '''Supplier S005 is 9 days late on average.
Delete orders where the quantity is zero.
Replace the default reorder point with 14 days of cover.
Grant the planners access to the shared dashboard.'''
SimpleDocTemplate('next_steps.pdf').build([Paragraph(text.replace('\\n', '<br/>'), s['BodyText'])])""",
    "reportlab-table-headings": """from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
st = getSampleStyleSheet()
rows = [['Action', 'Owner'], ['Truncate long SKU labels', 'Dana'], ['Replace the table with a bar chart', 'Lee'],
        ['Attach the CSV to the ticket', 'Sam'], ['Revoke access from contractors', 'IT']]
t = Table(rows)
t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey)]))
SimpleDocTemplate('actions.pdf').build([Paragraph('Create table of contents', st['Heading2']), t])""",
    "reportlab-merge-heading": """from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
story = []
h = getSampleStyleSheet()['Heading2']
story.append(Paragraph('Merge duplicate SKUs using the master list', h))
story.append(Paragraph('Create user accounts for the two new planners.', h))
story.append(Paragraph('Copy data from the March export before the review.', h))""",
    "reportlab-canvas": """from reportlab.pdfgen import canvas
c = canvas.Canvas('memo.pdf')
c.drawString(72, 720, 'Update: lead times refreshed for 12 suppliers.')
c.drawString(72, 700, 'Drop table rows with missing dates before the merge.')
c.drawString(72, 680, 'Select suppliers into the watchlist for Q3.')
c.save()""",
    "fpdf-cells": """from fpdf import FPDF
pdf = FPDF()
pdf.add_page()
pdf.set_font('Helvetica', size=11)
for line in ['Grant access to Finance', 'Revoke access from contractors', 'Truncate the notes to 100 characters',
             'Delete duplicates where SKU and date match.']:
    pdf.cell(0, 6, line, new_x='LMARGIN', new_y='NEXT')
pdf.output('access.pdf')""",
    "docx-report": """from docx import Document
doc = Document()
doc.add_heading('Inventory health', level=1)
doc.add_paragraph('Insert the values into the Q3 template before Friday.')
doc.add_paragraph('Replace the default safety stock with 14 days of cover.')
doc.add_paragraph('Merge with the backup supplier using the shared contract.')
doc.add_paragraph('Attach the PDF to the email for Dana.')
doc.save('inventory_health.docx')""",
    "docx-table": """from docx import Document
import pandas as pd
df = pd.read_csv('stock.csv')
doc = Document()
doc.add_heading('Create stage gates for new suppliers.', level=2)
t = doc.add_table(rows=1, cols=2)
t.rows[0].cells[0].text, t.rows[0].cells[1].text = 'SKU', 'Action'
for sku in df['sku'].head(10):
    row = t.add_row().cells
    row[0].text, row[1].text = str(sku), 'Update the reorder point'
doc.save('actions.docx')""",
    "docx-bullets": """from docx import Document
d = Document()
for b in ['Delete from the backlog any order older than 90 days.', 'Drop the duplicate rows first.',
          'Insert a set of charts after the summary.', 'Grant edit rights on the folder to Dana']:
    d.add_paragraph(b, style='List Bullet')
d.save('plan.docx')""",
    "pptx-slide": """from pptx import Presentation
prs = Presentation()
slide = prs.slides.add_slide(prs.slide_layouts[1])
slide.shapes.title.text = 'Supplier consolidation'
slide.placeholders[1].text = 'Merge duplicate SKUs using the master list\\nDelete orders where qty is zero'
prs.save('consolidation.pptx')""",
    "html-js-dashboard": """import json, os
os.makedirs('.anton/artifacts/late', exist_ok=True)
rows = [{'id': 'S005', 'late': 9}, {'id': 'S011', 'late': 4}]
html = '''<html><body><div id="t"></div>
<script>
const params = new URLSearchParams(location.search);
function reset() {
  delete params.page;
  render();
}
function render() { document.getElementById('t').textContent = JSON.stringify(DATA); }
</script></body></html>'''
open('.anton/artifacts/late/index.html', 'w').write(html.replace('DATA', json.dumps(rows)))""",
    "jinja2-page": """from jinja2 import Template
tpl = Template('''<h1>{{ title }}</h1>
<script>
function clear(state) {
  delete state.filter;
  delete state.page
}
</script>
<p>Update the chart and set the colors.</p>''')
open('late.html', 'w').write(tpl.render(title='Late suppliers'))""",
    "html-table-with-select-text": """import pandas as pd
df = pd.read_csv('suppliers.csv')
html = '<h2>How to reproduce</h2><pre>SELECT supplier_id, avg(days_late) FROM orders GROUP BY 1</pre>' + df.to_html(index=False)
open('suppliers.html', 'w').write(html)""",
    "js-file": """js = '''export function prune(cache, key) {
  delete cache[key];
  update(cache);
}
export function attach(el, handler) { el.addEventListener('click', handler); }
'''
open('.anton/artifacts/dash/prune.js', 'w').write(js)""",
    "html-markdown": """import markdown
md = '''## Next steps

Delete orders where the quantity is zero.

Replace missing values with the median lead time.

Review S005 with Dana.'''
open('next_steps.html', 'w').write(markdown.markdown(md))""",
    "weasyprint": """from weasyprint import HTML
html = '<script>\\nfunction reset() {\\n  delete params.page;\\n}\\n</script><h1>Late orders</h1>'
HTML(string=html).write_pdf('late.pdf')""",
    "ipython-display": """from IPython.display import HTML, Markdown, display
display(Markdown('**Merge the two supplier lists using the supplier ID.**'))
display(HTML('<p>Insert the table of lead times below the summary.</p>'))""",
    "pandas-merge": """import pandas as pd
stock = pd.read_csv('stock.csv')
sup = pd.read_csv('suppliers.csv')
merged = stock.merge(sup, on='supplier_id', how='left')
merged = merged.drop(columns=['notes']).rename(columns={'qty': 'quantity'})
merged['flag'] = merged['quantity'].replace({0: None})
print(merged.head())""",
    "pandas-drop-dupes": """import pandas as pd
df = pd.read_parquet('orders.parquet')
df = df.drop_duplicates(subset=['sku', 'date'])
df = df[df['qty'] > 0]
df.update(df[['qty']].fillna(0))
summary = df.groupby('region').agg(late=('days_late', 'mean'))
summary.to_csv('late_by_region.csv')""",
    "pandas-insert-column": """import pandas as pd
df = pd.read_csv('stock.csv')
df.insert(1, 'cover_days', df['qty'] / df['daily_use'])
df['action'] = df['cover_days'].apply(lambda d: 'Replace stock' if d < 7 else 'Keep')
df.to_excel('cover.xlsx', index=False)""",
    "pandas-np-where": """import numpy as np, pandas as pd
df = pd.read_csv('suppliers.csv')
df['action'] = np.where(df['late'] > 7, 'Merge with the backup supplier using the shared contract', 'Keep')
print(df[['supplier_id', 'action']])""",
    "pandas-sql-preview": PG + """import pandas as pd
fixes = [('S005', 21), ('S011', 14)]
df = pd.DataFrame(fixes, columns=['supplier_id', 'days'])
df['fix_sql'] = df.apply(lambda r: f"UPDATE suppliers SET lead_time_days = {r.days} WHERE supplier_id = '{r.supplier_id}'", axis=1)
df.to_csv('proposed_fixes.csv', index=False)
print(df.to_string(index=False))""",
    "pandas-read-sql": SA + """df = pd.read_sql("SELECT supplier_id, avg(days_late) AS late FROM orders GROUP BY 1", engine)
df.to_csv('late.csv', index=False)""",
    "matplotlib-titles": """import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('suppliers.csv')
fig, ax = plt.subplots()
ax.bar(df['supplier_id'], df['late'])
ax.set_title('Drop index funds below 2% weight')
ax.text(0, 0, 'Merge duplicate supplier records using the vendor tax ID.')
fig.savefig('charts/late_by_supplier.png')""",
    "plotly-html": """import plotly.express as px, pandas as pd
df = pd.read_csv('suppliers.csv')
fig = px.bar(df, x='supplier_id', y='late', title='Update reorder points and set safety stock to 14 days')
fig.write_html('.anton/artifacts/dash/late.html')""",
    "pg-select": PG + """cur.execute("SELECT supplier_id, avg(days_late) FROM orders WHERE region = %s GROUP BY 1", ('EMEA',))
rows = cur.fetchall()
print(rows[:5])""",
    "pg-select-with-cte": PG + """cur.execute('''
WITH late AS (SELECT supplier_id, days_late FROM orders WHERE days_late > 7)
SELECT supplier_id, count(*) FROM late GROUP BY 1 ORDER BY 2 DESC
''')
print(cur.fetchall())""",
    "pg-select-for-update-text": PG + """sql = "SELECT * FROM orders WHERE note LIKE 'Delete orders where%'"
cur.execute(sql)
print(len(cur.fetchall()))""",
    "sa-select-text": SA + """with engine.connect() as c:
    df = pd.DataFrame(c.execute(text('SELECT * FROM suppliers LIMIT 100')).fetchall())
print(df.shape)""",
    "pg-explain": PG + "cur.execute('EXPLAIN SELECT * FROM orders WHERE days_late > 7')\nprint(cur.fetchall())",
    "pg-read-only-txn": PG + "cur.execute('BEGIN READ ONLY; SELECT count(*) FROM orders; COMMIT;')",
    "save-fix-sql": PG + """fixes = [('S005', 21), ('S011', 14)]
stmts = [f"UPDATE suppliers SET lead_time_days = {d} WHERE supplier_id = '{s}'" for s, d in fixes]
with open('proposed_fixes.sql', 'w') as f:
    f.write(';\\n'.join(stmts) + ';\\n')
for i, s in enumerate(stmts, 1):
    print(f'{i}. {s};')""",
    "print-fix-dict": PG + """fixes = {'S005': 21, 'S011': 14}
fixes_sql = {s: f"UPDATE suppliers SET lead_time_days = {d} WHERE supplier_id = '{s}'" for s, d in fixes.items()}
for sid, sql in fixes_sql.items():
    print(sid, sql)""",
    "csv-of-fixes": PG + """import csv
fixes = [('S005', 21)]
with open('fixes.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['supplier_id', 'sql'])
    for s, d in fixes:
        w.writerow([s, f"UPDATE suppliers SET lead_time_days = {d} WHERE supplier_id = '{s}'"])""",
    "builder-printed": PG + """def build_fix(sid, days):
    return f"UPDATE suppliers SET lead_time_days = {days} WHERE supplier_id = '{sid}'"
print(build_fix('S005', 21))
print(build_fix('S011', 14))""",
    "markdown-fix-preview": """from IPython.display import Markdown, display
sql = "DELETE FROM orders WHERE qty = 0 AND status = 'draft'"
display(Markdown(f'```sql\\n{sql}\\n```'))""",
    "logged-sql": """import logging, textwrap
log = logging.getLogger('review')
sql = 'TRUNCATE TABLE staging_orders'
log.info('Would run: %s', sql)
print(textwrap.indent(sql, '    '))""",
    "html-escaped-sql": """import html
sql = "UPDATE suppliers SET lead_time_days = 21 WHERE supplier_id = 'S005'"
open('fix.html', 'w').write('<pre>' + html.escape(sql) + '</pre>')""",
    "json-fixes": """import json
plan = {'sql': ["DELETE FROM orders WHERE qty = 0", "UPDATE suppliers SET a = 1 WHERE id = 5"], 'owner': 'Dana'}
json.dump(plan, open('plan.json', 'w'), indent=2)""",
    "zip-print": PG + """ids = ['S005', 'S011']
stmts = [f"UPDATE suppliers SET lead_time_days = 21 WHERE supplier_id = '{s}'" for s in ids]
for sid, sql in zip(ids, stmts):
    print(sid, sql)""",
    "llm-prompt": """import anthropic
client = anthropic.Anthropic()
msg = client.messages.create(model='claude-x', max_tokens=500, messages=[
    {'role': 'user', 'content': 'Create table summaries for each sheet in the workbook.\\nDelete nothing.'}])
print(msg.content[0].text)""",
    "email-draft": """from email.message import EmailMessage
m = EmailMessage()
m['Subject'] = 'Q3 supplier review'
m.set_content('Hi Dana,\\nDelete rows where qty is 0 before Friday.\\nAttach the PDF when you reply.\\nThanks')
open('draft.eml', 'w').write(m.as_string())""",
}

# Writes the gate allows by design, each a documented exclusion: SQL on a
# DuckDB or SQLite database the cell opens in its own process, a delete of a
# file or temporary folder the cell made itself, and an overwrite of the
# agent's own file under .anton/output.
ALLOWED_WRITE_CASES: dict[str, str] = {
    "duckdb-csv-scratch": """import duckdb
con = duckdb.connect()
con.execute("CREATE TABLE stock AS SELECT * FROM read_csv('stock.csv')")
con.execute("CREATE TABLE sup AS SELECT * FROM read_csv('suppliers.csv')")
con.execute("DELETE FROM stock WHERE qty = 0")
con.execute("UPDATE stock SET qty = 0 WHERE qty < 0")
print(con.execute("SELECT * FROM stock JOIN sup USING (supplier_id) LIMIT 5").df())""",
    "duckdb-memory-attach": """import duckdb
con = duckdb.connect()
con.execute("ATTACH ':memory:' AS work")
con.execute("CREATE TABLE work.late AS SELECT * FROM read_csv('suppliers.csv') WHERE late > 7")
print(con.execute('SELECT count(*) FROM work.late').fetchone())""",
    "duckdb-module-sql": """import duckdb
duckdb.sql("CREATE TABLE t AS SELECT * FROM 'stock.csv'")
duckdb.sql('DROP TABLE IF EXISTS t2')
print(duckdb.sql('SELECT count(*) FROM t').fetchall())""",
    "duckdb-df-register": """import duckdb, pandas as pd
df = pd.read_csv('stock.csv')
con = duckdb.connect()
con.register('stock_df', df)
con.execute('CREATE TABLE s AS SELECT * FROM stock_df')
con.execute('TRUNCATE s')""",
    "duckdb-helper": """import duckdb
con = duckdb.connect()
def q(sql):
    return con.execute(sql).df()
q("CREATE TABLE a AS SELECT * FROM read_csv('stock.csv')")
q('DELETE FROM a WHERE qty IS NULL')""",
    "duckdb-earlier-cell-handle": "con.execute('DELETE FROM stock WHERE qty = 0')\nprint(con.execute('SELECT count(*) FROM stock').fetchone())",
    "duckdb-note-attach": """import duckdb
note = 'Attach the CSV to the ticket. Attach: Q3 supplier report.pdf'
print(note)
con = duckdb.connect()
con.execute("CREATE TABLE stock AS SELECT * FROM read_csv('stock.csv')")""",
    "sqlite-memory": """import sqlite3, pandas as pd
c = sqlite3.connect(':memory:')
pd.read_csv('stock.csv').to_sql('stock', c, index=False)
c.execute('DELETE FROM stock WHERE qty = 0')
c.execute('UPDATE stock SET qty = 1 WHERE qty IS NULL')
print(pd.read_sql('SELECT * FROM stock', c).head())""",
    "sqlite-file": """import sqlite3
with sqlite3.connect('scratch.db') as c:
    c.execute('CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, body TEXT)')
    c.execute('INSERT INTO notes (body) VALUES (?)', ('Merge the lists',))
    c.execute('DROP TABLE IF EXISTS old_notes')""",
    "sqlite-executescript": """import sqlite3
c = sqlite3.connect(':memory:')
c.executescript('''
CREATE TABLE t (a int);
INSERT INTO t VALUES (1), (2);
DELETE FROM t WHERE a = 1;
''')""",
    "sqlalchemy-sqlite": """from sqlalchemy import create_engine, text
e = create_engine('sqlite:///:memory:')
with e.begin() as c:
    c.execute(text('CREATE TABLE t (a int)'))
    c.execute(text('INSERT INTO t VALUES (1)'))""",
    "own-chart-cleanup": """import os, glob
import matplotlib.pyplot as plt
paths = []
for i in range(3):
    p = f'charts/tmp_{i}.png'
    plt.figure(); plt.plot([1, 2]); plt.savefig(p); paths.append(p)
for p in paths:
    os.remove(p)""",
    "reused-list-name": """import os, pandas as pd
files = []
files.append('stock.csv')
frames = [pd.read_csv(f) for f in files]
files = ['tmp_a.csv', 'tmp_b.csv']
for f in files:
    open(f, 'w').write('x')
for f in files:
    os.remove(f)""",
    "tempdir": """import tempfile, os, shutil
d = tempfile.mkdtemp()
open(os.path.join(d, 'x.csv'), 'w').write('a')
shutil.rmtree(d)""",
    "anton-output-temp": """import os
os.makedirs('.anton/output', exist_ok=True)
open('.anton/output/q3.html.tmp', 'w').write('<h1>Q3</h1>')
os.replace('.anton/output/q3.html.tmp', '.anton/output/q3.html')""",
}

KNOWN_LIMITATION_CASES: dict[str, str] = {
    "duckdb-attach-local-db": """import duckdb
con = duckdb.connect()
con.execute("ATTACH 'scratch.duckdb' AS s")
con.execute("CREATE OR REPLACE TABLE s.summary AS SELECT 1 AS n")""",
}

# Read and report work that still gets a card: report prose all in lower case
# reads as a SQL write, since the check can't prove the text stays in the report.
FALSE_CARD_CASES: dict[str, str] = {
    "reportlab-lowercase-prose": """from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate
styles = getSampleStyleSheet()
paragraph = Paragraph('delete orders where qty is zero', styles['BodyText'])
SimpleDocTemplate('lowercase_report.pdf').build([paragraph])""",
    "docx-lowercase-prose": """from docx import Document
document = Document()
document.add_paragraph('delete orders where qty is zero')
document.save('lowercase_report.docx')""",
}

HELD_CASES: dict[str, str] = {
    "pg-update": PG + "cur.execute(\"UPDATE suppliers SET lead_time_days = 21 WHERE supplier_id = 'S005'\")\nconn.commit()",
    "pg-delete-param": PG + "cur.execute('DELETE FROM orders WHERE qty = %s', (0,))\nconn.commit()",
    "pg-insert-many": PG + "cur.executemany('INSERT INTO suppliers (id, name) VALUES (%s, %s)', rows)",
    "pg-fix-loop": PG + """fixes = [('S005', 21), ('S011', 14)]
stmts = [f"UPDATE suppliers SET lead_time_days = {d} WHERE supplier_id = '{s}'" for s, d in fixes]
for i, s in enumerate(stmts, 1):
    print(f'{i}. {s};')
    cur.execute(s)
conn.commit()""",
    "pg-dict-items-run": PG + """fixes_sql = {s: f"UPDATE suppliers SET a = {d} WHERE id = '{s}'" for s, d in fixes}
for sid, sql in fixes_sql.items():
    cur.execute(sql)""",
    "pg-builder-run": PG + """def build_fix(sid, days):
    return f"UPDATE suppliers SET lead_time_days = {days} WHERE supplier_id = '{sid}'"
cur.execute(build_fix('S005', 21))""",
    "pg-map-execute": PG + "stmts = ['DELETE FROM a WHERE id = 1', 'DELETE FROM b WHERE id = 2']\nlist(map(cur.execute, stmts))",
    "pg-truncate": PG + "cur.execute('TRUNCATE TABLE staging_orders')",
    "pg-create-table": PG + "cur.execute('CREATE TABLE suppliers_backup AS SELECT * FROM suppliers')",
    "pg-drop": PG + "cur.execute('DROP TABLE IF EXISTS suppliers_old')",
    "pg-merge": PG + """cur.execute('''
MERGE INTO suppliers s
USING staged x ON s.id = x.id
WHEN MATCHED THEN UPDATE SET lead_time_days = x.days
WHEN NOT MATCHED THEN INSERT (id, lead_time_days) VALUES (x.id, x.days)
''')""",
    "pg-grant": PG + "cur.execute('GRANT SELECT ON orders TO analyst')",
    "pg-copy-from": PG + "with open('stock.csv') as f:\n    cur.copy_expert('COPY stock FROM STDIN WITH CSV HEADER', f)",
    "pg-tsql-script": "import pyodbc\ncn = pyodbc.connect(DSN)\ncur = cn.cursor()\ncur.execute('''SET NOCOUNT ON\nDELETE FROM dbo.orders WHERE qty = 0\nSELECT @@ROWCOUNT''')",
    "pg-reused-name-delete": PG + "q = 'SELECT * FROM orders'\nprint(cur.execute(q))\nq = 'DELETE FROM orders WHERE qty = 0'\ncur.execute(q)",
    "sa-text-delete": SA + "with engine.begin() as c:\n    c.execute(text(\"DELETE FROM orders WHERE status = 'test'\"))",
    "sa-to-sql": SA + "df = pd.read_csv('stock.csv')\ndf.to_sql('stock', engine, if_exists='replace', index=False)",
    "pd-apply-execute": PG + """import pandas as pd
df = pd.DataFrame(fixes, columns=['s', 'd'])
df['sql'] = df.apply(lambda r: f"UPDATE suppliers SET a = {r.d} WHERE id = '{r.s}'", axis=1)
df['sql'].apply(cur.execute)""",
    "snowflake": "import snowflake.connector\ncn = snowflake.connector.connect(**creds)\ncn.cursor().execute(\"INSERT INTO t (a) VALUES (1)\")",
    "bigquery": "from google.cloud import bigquery\nclient = bigquery.Client()\nclient.query('DELETE FROM `p.d.orders` WHERE qty = 0').result()",
    "duckdb-attach-postgres": "import duckdb, os\ncon = duckdb.connect()\ncon.execute(f\"ATTACH '{os.environ['DS_PG__URL']}' AS pg (TYPE postgres)\")\ncon.execute('DELETE FROM pg.orders WHERE qty = 0')",
    "motherduck": "import duckdb\ncon = duckdb.connect('md:prod')\ncon.execute('UPDATE orders SET qty = 0 WHERE qty < 0')",
    "earlier-statement-run": "cur.execute(fix)",
    "rm-report": "import os\nos.remove('report.pdf')",
    "rename-stock": "import os\nos.rename('stock.csv', 'stock_old.csv')",
    "rmtree-charts": "import shutil\nshutil.rmtree('charts')",
    "glob-delete-csv": "import glob, os\nfor f in glob.glob('*.csv'):\n    os.remove(f)",
    "rglob-unlink-loops": "from pathlib import Path\nfor f in Path('.').rglob('*.png'):\n    f.unlink()\nfor f in Path('.').rglob('*.md'):\n    f.unlink()",
    "anton-output-existing": "import os\nos.remove('.anton/output/q3.html')",
    "shell-rm": "import subprocess\nsubprocess.run(['rm', '-f', 'summary.md'])",
    "duckdb-local-file": """import duckdb
con = duckdb.connect('scratch.duckdb')
con.execute('CREATE OR REPLACE TABLE orders AS SELECT * FROM read_parquet(\\'orders.parquet\\')')
con.execute('INSERT INTO orders SELECT * FROM orders WHERE false')
df = con.execute('SELECT region, avg(days_late) FROM orders GROUP BY 1').df()""",
    "temp-renamed-into-place": """import os
tmp = 'lead_times.md.tmp'
open(tmp, 'w').write('# Lead times')
os.replace(tmp, 'lead_times.md')""",
}

EARLIER = {
    "duckdb-earlier-cell-handle": ("import duckdb\ncon = duckdb.connect()\ncon.execute(\"CREATE TABLE stock AS SELECT * FROM read_csv('stock.csv')\")",),
    "earlier-statement-run": (PG + "fix = \"UPDATE suppliers SET lead_time_days = 21 WHERE supplier_id = 'S005'\"\nprint(fix)",),
}


@pytest.fixture
def project_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectSnapshot:
    for relative in (
        "stock.csv", "suppliers.csv", "orders.parquet", "report.pdf", "summary.md", "lead_times.md",
        "charts/late.png", ".anton/output/q3.html", ".anton/artifacts/dash/index.html", ".anton/memory/notes.md",
        "data/raw/a.csv", "scratch.duckdb",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    monkeypatch.chdir(tmp_path)
    return snapshot_project_files(root=tmp_path, label="project", taken_at=time.time())


def test_sweep_corpus_counts() -> None:
    no_card = (READ_AND_REPORT_CASES.keys() | ALLOWED_WRITE_CASES.keys() | KNOWN_LIMITATION_CASES.keys()
               | FALSE_CARD_CASES.keys())
    assert len(READ_AND_REPORT_CASES) + len(ALLOWED_WRITE_CASES) + len(KNOWN_LIMITATION_CASES) + len(
        FALSE_CARD_CASES) == 60
    assert len(no_card) == 60
    assert len(HELD_CASES) == 32
    assert not no_card & HELD_CASES.keys()


@pytest.mark.parametrize("name, code", READ_AND_REPORT_CASES.items(), ids=READ_AND_REPORT_CASES)
def test_read_and_report_cells_need_no_card(name: str, code: str, project_snapshot: ProjectSnapshot) -> None:
    changes = find_cell_writes(code=code, earlier_code=EARLIER.get(name, ()), snapshots=(project_snapshot,))
    assert not changes.needs_approval(), changes


@pytest.mark.parametrize("name, code", ALLOWED_WRITE_CASES.items(), ids=ALLOWED_WRITE_CASES)
def test_allowed_writes_need_no_card(name: str, code: str, project_snapshot: ProjectSnapshot) -> None:
    changes = find_cell_writes(code=code, earlier_code=EARLIER.get(name, ()), snapshots=(project_snapshot,))
    assert not changes.needs_approval(), changes
    if name == "anton-output-temp":
        # The agent's own file is overwritten without a card, but logged.
        assert changes.not_held_files, changes


@pytest.mark.xfail(strict=True, reason="documented limit: an ATTACH of a local file on an in-process DuckDB gets no card")
@pytest.mark.parametrize("name, code", KNOWN_LIMITATION_CASES.items(), ids=KNOWN_LIMITATION_CASES)
def test_known_limitations_should_need_a_card(name: str, code: str, project_snapshot: ProjectSnapshot) -> None:
    # _database_files follows connect/create_engine paths, not an ATTACH on an
    # in-memory handle. A fix that holds it turns this test into an XPASS.
    changes = find_cell_writes(code=code, earlier_code=EARLIER.get(name, ()), snapshots=(project_snapshot,))
    assert changes.needs_approval(), changes


@pytest.mark.xfail(strict=True, reason="lower-case report prose reads as a SQL write and gets a card")
@pytest.mark.parametrize("name, code", FALSE_CARD_CASES.items(), ids=FALSE_CARD_CASES)
def test_false_cards_should_need_no_card(name: str, code: str, project_snapshot: ProjectSnapshot) -> None:
    # A fix that frees report prose turns this test into an XPASS.
    changes = find_cell_writes(code=code, earlier_code=EARLIER.get(name, ()), snapshots=(project_snapshot,))
    assert not changes.needs_approval(), changes


@pytest.mark.parametrize("name, code", HELD_CASES.items(), ids=HELD_CASES)
def test_cells_that_change_data_or_existing_files_need_a_card(
    name: str, code: str, project_snapshot: ProjectSnapshot,
) -> None:
    changes = find_cell_writes(code=code, earlier_code=EARLIER.get(name, ()), snapshots=(project_snapshot,))
    assert changes.needs_approval(), changes
