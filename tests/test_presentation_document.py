import json
import shutil
import subprocess
from pathlib import Path

import pytest

from anton.core.tools.generate_artifact import component_page

HERE = Path(component_page.__file__).parent


def test_presentation_bootstrap_preserves_program_for_followup_edits():
    program="await CoworkPage({title:'Appointments',sections:[]});"
    page=component_page.document(program)
    assert component_page.existing_program(page)==program
    assert '<meta name="viewport"' in page


def test_presentation_data_headers_timestamps_and_series_labels():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    script = f"""
const fs = require('fs');
const code = fs.readFileSync({json.dumps(str(HERE / 'presentation_data.js'))}, 'utf8');
class El {{
  constructor(tag) {{ this.tag = tag; this.children = []; this.attrs = {{}}; this.dataset = {{}}; this.textContent = ''; this.hidden = false; }}
  append(...nodes) {{ this.children.push(...nodes); }}
  replaceChildren(...nodes) {{ this.children = [...nodes]; }}
  setAttribute(k, v) {{ this.attrs[k] = v; }}
}}
global.document = {{
  documentElement: {{ lang: 'en' }},
  createElement: tag => new El(tag),
}};
global.window = {{}};
eval(code);

function runCase(seriesDefs) {{
  const mockChart = {{
    on() {{}}, off() {{}},
    getModel() {{ return {{ eachSeries(fn) {{ seriesDefs.forEach(fn); }} }}; }},
  }};
  const header = new El('div'), section = new El('section'), open = new Set(['s1']), cleanup = [];
  window.CoworkChartData(mockChart, header, section, {{}}, 's1', open, cleanup);
  const panel = section.children[0];
  return panel.children.map(group => {{
    const h3s = group.children.filter(c => c.tag === 'h3').map(c => c.textContent);
    const table = group.children.find(c => c.tag === 'table');
    const headers = table.children[0].children[0].children.map(th => th.textContent);
    const rows = table.children[1].children.map(tr => tr.children.map(td => td.children[0].textContent));
    return {{ h3s, headers, rows }};
  }});
}}

// 1. Single unnamed time-series with all 4 timestamp precisions
const ts = [
  Date.UTC(2026, 10, 2, 0, 0, 0, 0),
  Date.UTC(2026, 10, 2, 14, 30, 0, 0),
  Date.UTC(2026, 10, 2, 14, 30, 45, 0),
  Date.UTC(2026, 10, 2, 14, 30, 45, 250),
];
const timeCase = runCase([{{
  name: 'series\\u00000', seriesIndex: 0, subType: 'line',
  coordinateSystem: {{ getAxis: () => ({{ type: 'value', model: {{ get: () => '' }} }}) }},
  getData: () => ({{
    dimensions: ['x', 'y'], count: () => 4, getName: () => '',
    getDimensionInfo: d => d === 'x' ? {{ name: 'x', coordDim: 'x', type: 'time' }} : {{ name: 'y', coordDim: 'y', type: 'float' }},
    get: (d, i) => d === 'x' ? ts[i] : [10, 20, 30, 40][i],
  }}),
}}]);

// 2. Category axis + named single series
const catNamed = runCase([{{
  name: 'Revenue', seriesIndex: 0, subType: 'bar',
  coordinateSystem: {{ getAxis: d => ({{ type: d === 'x' ? 'category' : 'value', model: {{ get: () => '' }} }}) }},
  getData: () => ({{
    dimensions: ['x', 'y'], count: () => 2, getName: () => '',
    getDimensionInfo: d => d === 'x'
      ? {{ name: 'x', coordDim: 'x', ordinalMeta: {{ categories: ['North', 'South'] }} }}
      : {{ name: 'y', coordDim: 'y', type: 'float' }},
    get: (d, i) => d === 'x' ? i : [100, 200][i],
  }}),
}}]);

// 3. Scatter with two unnamed numeric axes -> ['X', 'Y']
const scatter = runCase([{{
  name: 'series\\u00000', seriesIndex: 0, subType: 'scatter',
  coordinateSystem: {{ getAxis: () => ({{ type: 'value', model: {{ get: () => '' }} }}) }},
  getData: () => ({{
    dimensions: ['x', 'y'], count: () => 1, getName: () => '',
    getDimensionInfo: d => ({{ name: d, coordDim: d, type: 'float' }}),
    get: (d) => d === 'x' ? 5 : 9,
  }}),
}}]);

// 4. Multi-series unnamed keeps Series 1 / Series 2 headings
const multi = runCase([0, 1].map(idx => ({{
  name: 'series\\u0000' + idx, seriesIndex: idx, subType: 'bar',
  coordinateSystem: {{ getAxis: d => ({{ type: d === 'x' ? 'category' : 'value', model: {{ get: () => '' }} }}) }},
  getData: () => ({{
    dimensions: ['x', 'y'], count: () => 1, getName: () => '',
    getDimensionInfo: d => d === 'x'
      ? {{ name: 'x', coordDim: 'x', ordinalMeta: {{ categories: ['A'] }} }}
      : {{ name: 'y', coordDim: 'y', type: 'float' }},
    get: (d) => d === 'x' ? 0 : idx + 1,
  }}),
}})));

console.log(JSON.stringify({{ timeCase, catNamed, scatter, multi }}));
"""
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    out = json.loads(proc.stdout)
    assert out["timeCase"] == [{
        "h3s": [],
        "headers": ["Date", "Value"],
        "rows": [
            ["2026-11-02", "10"],
            ["2026-11-02 14:30 UTC", "20"],
            ["2026-11-02 14:30:45 UTC", "30"],
            ["2026-11-02 14:30:45.250 UTC", "40"],
        ],
    }], out["timeCase"]
    assert out["catNamed"] == [{
        "h3s": ["Revenue"],
        "headers": ["Item", "Revenue"],
        "rows": [["North", "100"], ["South", "200"]],
    }], out["catNamed"]
    assert out["scatter"] == [{
        "h3s": [],
        "headers": ["X", "Y"],
        "rows": [["5", "9"]],
    }], out["scatter"]
    assert [g["h3s"] for g in out["multi"]] == [["Series 1"], ["Series 2"]], out["multi"]


def test_presentation_reconcile_accepts_row_column_and_mixed_zero_reasons():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    script = f"""
const fs = require('fs');
const code = fs.readFileSync({json.dumps(str(HERE / 'presentation_reconcile.js'))}, 'utf8');
const errors = [];
global.console = {{ ...console, error: (...a) => errors.push(a.join(' ')) }};
global.document = {{ getElementById: () => null }};
let captured;
global.window = {{
  ARTIFACT_DATA: {{ k: {{ rows: [{{ a: 1 }}] }} }},
  CoworkNumber: v => String(v),
  CoworkPage: async cfg => {{ captured = cfg; }},
}};
eval(code);
const baseCols = [
  {{ key: 's', label: 'Scenario' }},
  {{ key: 'a', label: 'A', unit: 'USD' }},
  {{ key: 'b', label: 'B', unit: 'USD' }},
  {{ key: 'c', label: 'C', unit: 'USD' }},
];
(async () => {{
  // 1. row.zeroReason resolves an all-zero row and appends to section caption
  await window.CoworkPage({{
    title: 'Costs',
    sections: [{{
      type: 'table', title: 'Scenarios', columns: baseCols,
      rows: [
        {{ zeroReason: 'No extra cost in baseline', s: 'Base', a: 0, b: 0, c: 0 }},
        {{ s: 'Surge', a: 10, b: 20, c: 30 }},
      ],
    }}],
  }});
  const r1 = {{ errs: errors.slice(), caption: captured.sections[0].caption }};

  // 2. all columns in the block carrying column.zeroReason resolves all-zero row
  errors.length = 0;
  await window.CoworkPage({{
    title: 'Costs',
    sections: [{{
      type: 'table', title: 'Scenarios',
      columns: [
        {{ key: 's', label: 'Scenario' }},
        {{ key: 'a', label: 'A', unit: 'USD', zeroReason: 'No A spend' }},
        {{ key: 'b', label: 'B', unit: 'USD', zeroReason: 'No B spend' }},
        {{ key: 'c', label: 'C', unit: 'USD', zeroReason: 'No C spend' }},
      ],
      rows: [
        {{ s: 'Base', a: 0, b: 0, c: 0 }},
        {{ s: 'Surge', a: 10, b: 20, c: 30 }},
      ],
    }}],
  }});
  const r2 = {{ errs: errors.slice(), caption: captured.sections[0].caption }};

  // 3. mixed explained + unexplained zero rows: keeps explained reason, flags ONLY unexplained row (and ignores zeroReason key as label)
  errors.length = 0;
  await window.CoworkPage({{
    title: 'Costs',
    sections: [{{
      type: 'table', title: 'Scenarios', columns: baseCols,
      rows: [
        {{ zeroReason: 'Base has no surcharge', s: 'Base', a: 0, b: 0, c: 0 }},
        {{ s: 'Quiet', a: 0, b: 0, c: 0 }},
        {{ s: 'Surge', a: 10, b: 20, c: 30 }},
      ],
    }}],
  }});
  const r3 = {{ errs: errors.slice(), caption: captured.sections[0].caption }};

  console.log(JSON.stringify({{ r1, r2, r3 }}));
}})();
"""
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    out = json.loads(proc.stdout)
    assert out["r1"]["errs"] == [] and "No extra cost in baseline" in out["r1"]["caption"], out["r1"]
    assert out["r2"]["errs"] == [] and "No A spend" in out["r2"]["caption"], out["r2"]
    assert len(out["r3"]["errs"]) == 1, out["r3"]
    assert 'row "Quiet"' in out["r3"]["errs"][0] and "Base" not in out["r3"]["errs"][0], out["r3"]
    assert "Base has no surcharge" in out["r3"]["caption"], out["r3"]
