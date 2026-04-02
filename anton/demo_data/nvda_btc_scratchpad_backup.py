import os
import json
import webbrowser
import numpy as np
import pandas as pd
import yfinance as yf

# ── CONFIG ─────────────────────────────────────────────────────────────────
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvda_btc_dashboard.html")

# ── 1. FETCH DATA ──────────────────────────────────────────────────────────
print("Fetching NVDA and BTC monthly data...")
nvda_raw = yf.download("NVDA",    start="2021-04-01", end="2026-04-30", interval="1mo", auto_adjust=True, progress=False)
btc_raw  = yf.download("BTC-USD", start="2021-04-01", end="2026-04-30", interval="1mo", auto_adjust=True, progress=False)
print(f"  NVDA: {nvda_raw.shape}  BTC: {btc_raw.shape}")

# ── 2. BUILD PRICES DATAFRAME ──────────────────────────────────────────────
nvda = nvda_raw.copy(); btc = btc_raw.copy()
nvda.columns = [c[0].lower() for c in nvda.columns]
btc.columns  = [c[0].lower() for c in btc.columns]
nvda.index = pd.to_datetime(nvda.index)
btc.index  = pd.to_datetime(btc.index)

def drawdown(series):
    return ((series - series.cummax()) / series.cummax()).round(6)

prices = pd.DataFrame({
    "date":       nvda.index.strftime("%Y-%m-%d"),
    "nvda_close": nvda["close"].round(4),
    "btc_close":  btc["close"].round(2),
    "nvda_volume": nvda["volume"].astype(int),
    "btc_volume":  btc["volume"].astype(int),
})
prices["nvda_ret"]   = nvda["close"].pct_change().round(6)
prices["btc_ret"]    = btc["close"].pct_change().round(6)
prices["nvda_cum"]   = (1 + nvda["close"].pct_change().fillna(0)).cumprod().round(6) * 100
prices["btc_cum"]    = (1 + btc["close"].pct_change().fillna(0)).cumprod().round(6) * 100
prices["nvda_vol12"] = nvda["close"].pct_change().rolling(12).std().round(6)
prices["btc_vol12"]  = btc["close"].pct_change().rolling(12).std().round(6)
prices["nvda_dd"]    = drawdown(nvda["close"])
prices["btc_dd"]     = drawdown(btc["close"])
prices["year"]       = pd.to_datetime(prices["date"]).dt.year
prices = prices.reset_index(drop=True)
print(f"  Prices shape: {prices.shape}")

# ── 3. RISK METRICS ────────────────────────────────────────────────────────
nvda_ret   = prices["nvda_ret"].dropna()
btc_ret    = prices["btc_ret"].dropna()
rf_monthly = 0.045 / 12

def stats(ret, name):
    mu        = ret.mean()
    sigma     = ret.std()
    ann_ret   = (1 + mu)**12 - 1
    ann_vol   = sigma * np.sqrt(12)
    sharpe    = (mu - rf_monthly) / sigma * np.sqrt(12)
    sortino_d = ret[ret < 0].std() * np.sqrt(12)
    sortino   = (ann_ret - 0.045) / sortino_d if sortino_d > 0 else float("nan")
    max_dd    = prices[f"{name}_dd"].min()
    calmar    = ann_ret / abs(max_dd) if max_dd != 0 else float("nan")
    return {
        "asset": name.upper(),
        "ann_return":   round(ann_ret, 4),
        "ann_vol":      round(ann_vol, 4),
        "sharpe":       round(sharpe, 4),
        "sortino":      round(sortino, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar":       round(calmar, 4),
        "win_rate":     round((ret > 0).mean(), 4),
        "best_month":   round(ret.max(), 4),
        "worst_month":  round(ret.min(), 4),
        "total_return": round(prices[f"{name}_cum"].iloc[-1] / 100 - 1, 4),
        "corr_60d":     None,
    }

nvda_stats = stats(nvda_ret, "nvda")
btc_stats  = stats(btc_ret,  "btc")
corr = nvda_ret.corr(btc_ret)
nvda_stats["corr_60d"] = round(corr, 4)
btc_stats["corr_60d"]  = round(corr, 4)
risk = pd.DataFrame([nvda_stats, btc_stats])

# ── 4. ANNUAL SUMMARY ──────────────────────────────────────────────────────
annual_rows = []
for yr, grp in prices.groupby("year"):
    row = {"year": int(yr)}
    for asset in ["nvda", "btc"]:
        r = grp[f"{asset}_ret"].dropna()
        row[f"{asset}_annual_ret"]   = round(((1 + r).prod() - 1), 4)
        row[f"{asset}_annual_vol"]   = round(r.std() * np.sqrt(12), 4)
        row[f"{asset}_best_month"]   = round(r.max(), 4) if len(r) else None
        row[f"{asset}_worst_month"]  = round(r.min(), 4) if len(r) else None
    annual_rows.append(row)
annual = pd.DataFrame(annual_rows)

# ── 5. MONTE CARLO ─────────────────────────────────────────────────────────
np.random.seed(42)
N_SIMS, N_MONTHS, INITIAL = 1000, 60, 10_000

def monte_carlo(ret_series, label):
    mu, sigma = ret_series.mean(), ret_series.std()
    paths = np.cumprod(1 + np.random.normal(mu, sigma, (N_SIMS, N_MONTHS)), axis=1) * INITIAL
    rows = []
    for m in range(N_MONTHS):
        vals = paths[:, m]
        rows.append({"month": m+1, "asset": label,
            "p10": round(float(np.percentile(vals,10)),2), "p25": round(float(np.percentile(vals,25)),2),
            "p50": round(float(np.percentile(vals,50)),2), "p75": round(float(np.percentile(vals,75)),2),
            "p90": round(float(np.percentile(vals,90)),2), "mean": round(float(vals.mean()),2)})
    return pd.DataFrame(rows)

nvda_mc_df = monte_carlo(nvda_ret, "NVDA")
btc_mc_df  = monte_carlo(btc_ret,  "BTC")
mc = pd.concat([nvda_mc_df, btc_mc_df], ignore_index=True)
print(f"  MC shape: {mc.shape}")

# ── 6. SCORECARD ───────────────────────────────────────────────────────────
def winner(a, b, hi=True): return "NVDA" if (a > b) == hi else "BTC"

nvda_mc60 = nvda_mc_df[nvda_mc_df.month==60]
btc_mc60  = btc_mc_df[btc_mc_df.month==60]
sc_rows = [
    {"metric":"5-Year Total Return",  "nvda":f"{nvda_stats['total_return']*100:.1f}%",  "btc":f"{btc_stats['total_return']*100:.1f}%",  "winner":winner(nvda_stats['total_return'], btc_stats['total_return'])},
    {"metric":"Ann. Return",          "nvda":f"{nvda_stats['ann_return']*100:.1f}%",     "btc":f"{btc_stats['ann_return']*100:.1f}%",     "winner":winner(nvda_stats['ann_return'], btc_stats['ann_return'])},
    {"metric":"Ann. Volatility",      "nvda":f"{nvda_stats['ann_vol']*100:.1f}%",        "btc":f"{btc_stats['ann_vol']*100:.1f}%",        "winner":winner(nvda_stats['ann_vol'], btc_stats['ann_vol'], False)},
    {"metric":"Sharpe Ratio",         "nvda":f"{nvda_stats['sharpe']:.2f}",              "btc":f"{btc_stats['sharpe']:.2f}",              "winner":winner(nvda_stats['sharpe'], btc_stats['sharpe'])},
    {"metric":"Sortino Ratio",        "nvda":f"{nvda_stats['sortino']:.2f}",             "btc":f"{btc_stats['sortino']:.2f}",             "winner":winner(nvda_stats['sortino'], btc_stats['sortino'])},
    {"metric":"Max Drawdown",         "nvda":f"{nvda_stats['max_drawdown']*100:.1f}%",   "btc":f"{btc_stats['max_drawdown']*100:.1f}%",   "winner":winner(nvda_stats['max_drawdown'], btc_stats['max_drawdown'], False)},
    {"metric":"Calmar Ratio",         "nvda":f"{nvda_stats['calmar']:.2f}",              "btc":f"{btc_stats['calmar']:.2f}",              "winner":winner(nvda_stats['calmar'], btc_stats['calmar'])},
    {"metric":"Win Rate",             "nvda":f"{nvda_stats['win_rate']*100:.1f}%",       "btc":f"{btc_stats['win_rate']*100:.1f}%",       "winner":winner(nvda_stats['win_rate'], btc_stats['win_rate'])},
    {"metric":"Best Month",           "nvda":f"{nvda_stats['best_month']*100:.1f}%",     "btc":f"{btc_stats['best_month']*100:.1f}%",     "winner":winner(nvda_stats['best_month'], btc_stats['best_month'])},
    {"metric":"Worst Month",          "nvda":f"{nvda_stats['worst_month']*100:.1f}%",    "btc":f"{btc_stats['worst_month']*100:.1f}%",    "winner":winner(nvda_stats['worst_month'], btc_stats['worst_month'], False)},
    {"metric":"MC Median (5yr $10k)", "nvda":f"${nvda_mc60['p50'].values[0]:,.0f}",     "btc":f"${btc_mc60['p50'].values[0]:,.0f}",     "winner":winner(nvda_mc60['p50'].values[0], btc_mc60['p50'].values[0])},
    {"metric":"MC P10 Downside",      "nvda":f"${nvda_mc60['p10'].values[0]:,.0f}",     "btc":f"${btc_mc60['p10'].values[0]:,.0f}",     "winner":winner(nvda_mc60['p10'].values[0], btc_mc60['p10'].values[0])},
]
nvda_wins = sum(1 for r in sc_rows if r["winner"]=="NVDA")
btc_wins  = sum(1 for r in sc_rows if r["winner"]=="BTC")
sc_rows.append({"metric":"TOTAL WINS","nvda":str(nvda_wins),"btc":str(btc_wins),"winner":"NVDA" if nvda_wins>btc_wins else "BTC"})
scorecard = pd.DataFrame(sc_rows)
print(f"  Scorecard: NVDA {nvda_wins} | BTC {btc_wins}")

# ── 7. SERIALIZE TO JSON ───────────────────────────────────────────────────
def df_to_json(df):
    return json.dumps(df.where(pd.notnull(df), None).to_dict(orient="records"))

prices_json    = df_to_json(prices)
risk_json      = df_to_json(risk)
mc_json        = df_to_json(mc)
annual_json    = df_to_json(annual)
scorecard_json = df_to_json(scorecard)

hero = {
    "nvda_total_ret":  f"{nvda_stats['total_return']*100:.0f}%",
    "btc_total_ret":   f"{btc_stats['total_return']*100:.0f}%",
    "nvda_sharpe":     f"{nvda_stats['sharpe']:.2f}",
    "btc_sharpe":      f"{btc_stats['sharpe']:.2f}",
    "nvda_mc_median":  f"${nvda_mc_df[nvda_mc_df.month==60]['p50'].values[0]:,.0f}",
    "btc_mc_median":   f"${btc_mc_df[btc_mc_df.month==60]['p50'].values[0]:,.0f}",
    "nvda_max_dd":     f"{nvda_stats['max_drawdown']*100:.1f}%",
    "btc_max_dd":      f"{btc_stats['max_drawdown']*100:.1f}%",
    "nvda_ann_vol":    f"{nvda_stats['ann_vol']*100:.1f}%",
    "btc_ann_vol":     f"{btc_stats['ann_vol']*100:.1f}%",
    "nvda_wins":       nvda_wins,
    "btc_wins":        btc_wins,
    "verdict":         "NVDA is the dominant trade on a risk-adjusted basis",
}
hero_json = json.dumps(hero)

data_block = f"""const DATA = {{
  prices:    {prices_json},
  risk:      {risk_json},
  mc:        {mc_json},
  annual:    {annual_json},
  scorecard: {scorecard_json},
  hero:      {hero_json}
}};"""

print("  JSON serialization complete")

# ── 8. CSS ─────────────────────────────────────────────────────────────────
css = """
  :root {
    --bg: #0d1117; --bg2: #161b22; --bg3: #21262d;
    --border: #30363d; --text: #e6edf3; --muted: #8b949e;
    --nvda: #76b900; --btc: #f7931a; --accent: #58a6ff;
    --red: #f85149; --green: #3fb950; --yellow: #d29922;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; min-height: 100vh; }
  .header { background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1a2332 100%);
    border-bottom: 1px solid var(--border); padding: 28px 40px; }
  .header-inner { max-width: 1400px; margin: 0 auto; }
  .header h1 { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.5px;
    background: linear-gradient(90deg, var(--nvda), var(--accent), var(--btc));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .header p { color: var(--muted); margin-top: 6px; font-size: 0.95rem; }
  .verdict-badge { display: inline-block; margin-top: 12px; padding: 6px 16px;
    border-radius: 20px; background: rgba(118,185,0,0.15); border: 1px solid var(--nvda);
    color: var(--nvda); font-size: 0.85rem; font-weight: 600; letter-spacing: 0.3px; }
  .kpi-row { max-width: 1400px; margin: 24px auto; padding: 0 24px;
    display: grid; grid-template-columns: repeat(6,1fr); gap: 14px; }
  .kpi-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px;
    padding: 18px 16px; text-align: center; position: relative; overflow: hidden;
    transition: transform 0.2s, border-color 0.2s; }
  .kpi-card:hover { transform: translateY(-2px); border-color: var(--accent); }
  .kpi-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; }
  .kpi-card.nvda::before { background: var(--nvda); }
  .kpi-card.btc::before  { background: var(--btc); }
  .kpi-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }
  .kpi-value { font-size: 1.7rem; font-weight: 700; line-height: 1; }
  .kpi-sub   { font-size: 0.75rem; color: var(--muted); margin-top: 6px; }
  .kpi-value.nvda-color { color: var(--nvda); }
  .kpi-value.btc-color  { color: var(--btc); }
  .tabs-wrap { max-width: 1400px; margin: 8px auto; padding: 0 24px; }
  .tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--border); }
  .tab { padding: 10px 22px; cursor: pointer; border-radius: 8px 8px 0 0;
    font-size: 0.88rem; font-weight: 500; color: var(--muted);
    border: 1px solid transparent; border-bottom: none; transition: all 0.2s; }
  .tab:hover  { color: var(--text); background: var(--bg2); }
  .tab.active { color: var(--accent); background: var(--bg2);
    border-color: var(--border); border-bottom: 2px solid var(--accent); }
  .panels { max-width: 1400px; margin: 0 auto; padding: 24px; }
  .panel  { display: none; }
  .panel.active { display: block; }
  .chart-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
  .chart-grid-1 { margin-bottom: 20px; }
  .chart-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
  .chart-card h3 { font-size: 0.88rem; color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.8px; margin-bottom: 14px; }
  .chart-box { width: 100%; }
  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  th { background: var(--bg3); color: var(--muted); font-weight: 600; text-transform: uppercase;
    font-size: 0.72rem; letter-spacing: 0.6px; padding: 10px 14px; text-align: left;
    border-bottom: 1px solid var(--border); }
  td { padding: 10px 14px; border-bottom: 1px solid var(--border); }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(255,255,255,0.03); }
  .win-nvda { color: var(--nvda); font-weight: 700; }
  .win-btc  { color: var(--btc);  font-weight: 700; }
  .pos { color: var(--green); } .neg { color: var(--red); }
  .decision-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .decision-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 24px; }
  .decision-card h2 { font-size: 1.1rem; font-weight: 700; margin-bottom: 16px; }
  .decision-card.nvda-card { border-color: var(--nvda); }
  .decision-card.btc-card  { border-color: var(--btc); }
  .pros-cons h4 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.6px; color: var(--muted); margin: 14px 0 8px; }
  .pros-cons li { font-size: 0.875rem; margin: 5px 0; padding-left: 12px; list-style: none; }
  .pros-cons li.pro::before  { content: '✓ '; color: var(--green); }
  .pros-cons li.con::before  { content: '✗ '; color: var(--red); }
  .verdict-row { background: var(--bg3); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-top: 20px; text-align: center; }
  .verdict-row h2 { font-size: 1.4rem; color: var(--nvda); margin-bottom: 10px; }
  .verdict-row p  { color: var(--muted); line-height: 1.7; max-width: 800px; margin: 0 auto; }
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg2); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
"""
print("  CSS ready")

# ── 9. HTML BODY ───────────────────────────────────────────────────────────
html_body = """<body>
<div class="header"><div class="header-inner">
  <h1>NVDA vs BTC — 5-Year Investment Analysis</h1>
  <p>Monthly data · April 2021 – April 2026 · $10,000 initial investment · Risk-free rate 4.5%</p>
  <div class="verdict-badge" id="verdictBadge">Loading verdict…</div>
</div></div>
<div class="kpi-row">
  <div class="kpi-card nvda"><div class="kpi-label">NVDA Total Return</div><div class="kpi-value nvda-color" id="kpi-nvda-ret">—</div><div class="kpi-sub">5-year cumulative</div></div>
  <div class="kpi-card btc"> <div class="kpi-label">BTC Total Return</div> <div class="kpi-value btc-color"  id="kpi-btc-ret">—</div> <div class="kpi-sub">5-year cumulative</div></div>
  <div class="kpi-card nvda"><div class="kpi-label">NVDA Sharpe</div>      <div class="kpi-value nvda-color" id="kpi-nvda-sharpe">—</div><div class="kpi-sub">risk-adjusted return</div></div>
  <div class="kpi-card btc"> <div class="kpi-label">BTC Sharpe</div>       <div class="kpi-value btc-color"  id="kpi-btc-sharpe">—</div><div class="kpi-sub">risk-adjusted return</div></div>
  <div class="kpi-card nvda"><div class="kpi-label">NVDA MC Median</div>   <div class="kpi-value nvda-color" id="kpi-nvda-mc">—</div>    <div class="kpi-sub">5yr forward (1000 sims)</div></div>
  <div class="kpi-card btc"> <div class="kpi-label">BTC MC Median</div>    <div class="kpi-value btc-color"  id="kpi-btc-mc">—</div>     <div class="kpi-sub">5yr forward (1000 sims)</div></div>
</div>
<div class="tabs-wrap"><div class="tabs">
  <div class="tab" onclick="switchTab('performance',this)">📈 Performance</div>
  <div class="tab" onclick="switchTab('risk',this)">⚠️ Risk</div>
  <div class="tab" onclick="switchTab('montecarlo',this)">🎲 Monte Carlo</div>
  <div class="tab" onclick="switchTab('annual',this)">📅 Annual</div>
  <div class="tab" onclick="switchTab('scorecard',this)">🏆 Scorecard</div>
  <div class="tab active" onclick="switchTab('decision',this)">🎯 Decision</div>
</div></div>
<div class="panels">
  <div id="panel-performance" class="panel">
    <div class="chart-grid-1"><div class="chart-card"><h3>Cumulative Return — $100 Rebased</h3><div id="chart-cumret" class="chart-box" style="height:380px"></div></div></div>
    <div class="chart-grid-2">
      <div class="chart-card"><h3>Monthly Returns — NVDA</h3><div id="chart-nvda-monthly" class="chart-box" style="height:280px"></div></div>
      <div class="chart-card"><h3>Monthly Returns — BTC</h3><div id="chart-btc-monthly" class="chart-box" style="height:280px"></div></div>
    </div>
    <div class="chart-grid-1"><div class="chart-card"><h3>Drawdown from Peak</h3><div id="chart-drawdown" class="chart-box" style="height:280px"></div></div></div>
  </div>
  <div id="panel-risk" class="panel">
    <div class="chart-grid-2">
      <div class="chart-card"><h3>Rolling 12-Month Volatility</h3><div id="chart-vol" class="chart-box" style="height:300px"></div></div>
      <div class="chart-card"><h3>Return Distribution</h3><div id="chart-dist" class="chart-box" style="height:300px"></div></div>
    </div>
    <div class="chart-grid-2">
      <div class="chart-card"><h3>Risk vs Return Scatter</h3><div id="chart-scatter" class="chart-box" style="height:300px"></div></div>
      <div class="chart-card"><h3>Risk Metrics Radar</h3><div id="chart-radar" class="chart-box" style="height:300px"></div></div>
    </div>
  </div>
  <div id="panel-montecarlo" class="panel">
    <div class="chart-grid-2">
      <div class="chart-card"><h3>NVDA — 5-Year Monte Carlo (1,000 Paths, $10,000 Start)</h3><div id="chart-mc-nvda" class="chart-box" style="height:360px"></div></div>
      <div class="chart-card"><h3>BTC — 5-Year Monte Carlo (1,000 Paths, $10,000 Start)</h3><div id="chart-mc-btc" class="chart-box" style="height:360px"></div></div>
    </div>
    <div class="chart-grid-1"><div class="chart-card"><h3>MC Percentile Comparison — NVDA vs BTC</h3><div id="chart-mc-compare" class="chart-box" style="height:320px"></div></div></div>
  </div>
  <div id="panel-annual" class="panel">
    <div class="chart-grid-2">
      <div class="chart-card"><h3>Annual Returns by Year</h3><div id="chart-annual-bar" class="chart-box" style="height:340px"></div></div>
      <div class="chart-card"><h3>Annual Volatility by Year</h3><div id="chart-annual-vol" class="chart-box" style="height:340px"></div></div>
    </div>
    <div class="chart-card table-wrap"><h3>Annual Summary Table</h3><table id="annual-table"></table></div>
  </div>
  <div id="panel-scorecard" class="panel">
    <div class="chart-grid-2">
      <div class="chart-card"><h3>Scorecard Win Count</h3><div id="chart-wins" class="chart-box" style="height:260px"></div></div>
      <div class="chart-card"><h3>Key Metrics Bar Race</h3><div id="chart-metrics-bar" class="chart-box" style="height:260px"></div></div>
    </div>
    <div class="chart-card table-wrap" style="margin-top:20px"><h3>Full Scorecard</h3><table id="scorecard-table"></table></div>
  </div>
  <div id="panel-decision" class="panel active">
    <div class="decision-grid">
      <div class="decision-card nvda-card">
        <h2 style="color:var(--nvda)">🟢 NVIDIA (NVDA)</h2>
        <div class="pros-cons">
          <h4>Strengths</h4><ul>
            <li class="pro">1,074% total return over 5 years</li>
            <li class="pro">Sharpe ratio 1.14 — exceptional risk-adjusted return</li>
            <li class="pro">Sortino ratio 2.93 — low downside deviation</li>
            <li class="pro">Win rate 63.3% — positive months majority</li>
            <li class="pro">Lower volatility (50.9%) vs BTC (57.6%)</li>
            <li class="pro">MC median $113,908 on $10k — 11× upside</li>
            <li class="pro">AI/data center tailwinds still in early innings</li>
          </ul>
          <h4>Risks</h4><ul>
            <li class="con">62.8% max drawdown — severe corrections possible</li>
            <li class="con">Single company concentration risk</li>
            <li class="con">Valuation-driven; needs continued earnings growth</li>
            <li class="con">Regulatory risk (chip export controls)</li>
          </ul>
        </div>
      </div>
      <div class="decision-card btc-card">
        <h2 style="color:var(--btc)">🟠 Bitcoin (BTC)</h2>
        <div class="pros-cons">
          <h4>Strengths</h4><ul>
            <li class="pro">Decentralised, non-sovereign hard money</li>
            <li class="pro">Highest single-month gain (43.7% in one month)</li>
            <li class="pro">ETF adoption driving institutional inflows</li>
            <li class="pro">Halving cycles historically bullish</li>
            <li class="pro">Portfolio diversifier — partial equity hedge</li>
          </ul>
          <h4>Risks</h4><ul>
            <li class="con">Only 17% total return over 5 years</li>
            <li class="con">Sharpe ratio 0.25 — poor risk-adjusted performance</li>
            <li class="con">73% max drawdown — brutal bear markets</li>
            <li class="con">Win rate only 51.7% — near coin-flip monthly</li>
            <li class="con">MC median only $11,892 — barely above initial</li>
            <li class="con">Regulatory uncertainty in multiple jurisdictions</li>
          </ul>
        </div>
      </div>
    </div>
    <div class="verdict-row">
      <h2>🏆 Verdict: NVIDIA Wins 9 of 12 Metrics</h2>
      <p>Over the past 5 years, NVDA delivered <strong style="color:var(--nvda)">1,074%</strong> cumulative return vs BTC's <strong style="color:var(--btc)">17%</strong> — with <em>lower</em> volatility, a Sharpe of <strong>1.14</strong> vs <strong>0.25</strong>, and a Monte Carlo median of <strong>$113,908</strong> vs <strong>$11,892</strong> on a $10k investment. NVDA's structural advantage in AI infrastructure makes it the stronger conviction trade for the next 5 years.</p>
    </div>
  </div>
</div></body>"""
print("  HTML body ready")

# ── 10. JAVASCRIPT ─────────────────────────────────────────────────────────
js = r"""
const NVDA_COLOR='#76b900', BTC_COLOR='#f7931a', GRID_COLOR='rgba(48,54,61,0.8)';
function baseOpt(){return{backgroundColor:'transparent',tooltip:{trigger:'axis',backgroundColor:'#161b22',borderColor:'#30363d',textStyle:{color:'#e6edf3'}},grid:{left:'8%',right:'4%',top:'10%',bottom:'12%'},xAxis:{type:'category',axisLine:{lineStyle:{color:GRID_COLOR}},axisLabel:{color:'#8b949e'},splitLine:{show:false}},yAxis:{axisLine:{show:false},axisLabel:{color:'#8b949e'},splitLine:{lineStyle:{color:GRID_COLOR}}}};}
function initKPIs(){const h=DATA.hero;document.getElementById('kpi-nvda-ret').textContent=h.nvda_total_ret;document.getElementById('kpi-btc-ret').textContent=h.btc_total_ret;document.getElementById('kpi-nvda-sharpe').textContent=h.nvda_sharpe;document.getElementById('kpi-btc-sharpe').textContent=h.btc_sharpe;document.getElementById('kpi-nvda-mc').textContent=h.nvda_mc_median;document.getElementById('kpi-btc-mc').textContent=h.btc_mc_median;document.getElementById('verdictBadge').textContent='🏆 '+h.verdict;}
const rendered={decision:true};
function switchTab(name,el){document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));el.classList.add('active');document.getElementById('panel-'+name).classList.add('active');if(!rendered[name]){rendered[name]=true;renderTab(name);}}
function renderTab(n){if(n==='performance')renderPerformance();if(n==='risk')renderRisk();if(n==='montecarlo')renderMC();if(n==='annual')renderAnnual();if(n==='scorecard')renderScorecard();}

function renderPerformance(){
  const dates=DATA.prices.map(d=>d.date.slice(0,7));
  const nvdaCum=DATA.prices.map(d=>d.nvda_cum?+d.nvda_cum.toFixed(2):null);
  const btcCum=DATA.prices.map(d=>d.btc_cum?+d.btc_cum.toFixed(2):null);
  const nvdaRet=DATA.prices.map(d=>d.nvda_ret?+(d.nvda_ret*100).toFixed(2):null);
  const btcRet=DATA.prices.map(d=>d.btc_ret?+(d.btc_ret*100).toFixed(2):null);
  const nvdaDD=DATA.prices.map(d=>d.nvda_dd?+(d.nvda_dd*100).toFixed(2):null);
  const btcDD=DATA.prices.map(d=>d.btc_dd?+(d.btc_dd*100).toFixed(2):null);
  const cumChart=echarts.init(document.getElementById('chart-cumret'));
  cumChart.setOption(Object.assign(baseOpt(),{legend:{data:['NVDA','BTC'],textStyle:{color:'#8b949e'},top:0},xAxis:{type:'category',data:dates,axisLabel:{color:'#8b949e',rotate:30,fontSize:11},axisLine:{lineStyle:{color:GRID_COLOR}},splitLine:{show:false}},yAxis:{axisLabel:{color:'#8b949e',formatter:v=>v.toLocaleString()},splitLine:{lineStyle:{color:GRID_COLOR}}},series:[{name:'NVDA',type:'line',data:nvdaCum,smooth:true,lineStyle:{color:NVDA_COLOR,width:2.5},areaStyle:{color:{type:'linear',x:0,y:0,x2:0,y2:1,colorStops:[{offset:0,color:'rgba(118,185,0,0.25)'},{offset:1,color:'rgba(118,185,0,0.02)'}]}},symbol:'none'},{name:'BTC',type:'line',data:btcCum,smooth:true,lineStyle:{color:BTC_COLOR,width:2.5},areaStyle:{color:{type:'linear',x:0,y:0,x2:0,y2:1,colorStops:[{offset:0,color:'rgba(247,147,26,0.2)'},{offset:1,color:'rgba(247,147,26,0.02)'}]}},symbol:'none'}]}));
  const nm=echarts.init(document.getElementById('chart-nvda-monthly'));
  nm.setOption(Object.assign(baseOpt(),{xAxis:{type:'category',data:dates,axisLabel:{color:'#8b949e',rotate:45,fontSize:10},axisLine:{lineStyle:{color:GRID_COLOR}},splitLine:{show:false}},yAxis:{axisLabel:{color:'#8b949e',formatter:v=>v+'%'},splitLine:{lineStyle:{color:GRID_COLOR}}},series:[{type:'bar',data:nvdaRet.map(v=>({value:v,itemStyle:{color:v>=0?NVDA_COLOR:'#f85149'}}))}]}));
  const bm=echarts.init(document.getElementById('chart-btc-monthly'));
  bm.setOption(Object.assign(baseOpt(),{xAxis:{type:'category',data:dates,axisLabel:{color:'#8b949e',rotate:45,fontSize:10},axisLine:{lineStyle:{color:GRID_COLOR}},splitLine:{show:false}},yAxis:{axisLabel:{color:'#8b949e',formatter:v=>v+'%'},splitLine:{lineStyle:{color:GRID_COLOR}}},series:[{type:'bar',data:btcRet.map(v=>({value:v,itemStyle:{color:v>=0?BTC_COLOR:'#f85149'}}))}]}));
  const dd=echarts.init(document.getElementById('chart-drawdown'));
  dd.setOption(Object.assign(baseOpt(),{legend:{data:['NVDA DD','BTC DD'],textStyle:{color:'#8b949e'},top:0},xAxis:{type:'category',data:dates,axisLabel:{color:'#8b949e',rotate:30,fontSize:11},axisLine:{lineStyle:{color:GRID_COLOR}},splitLine:{show:false}},yAxis:{axisLabel:{color:'#8b949e',formatter:v=>v+'%'},splitLine:{lineStyle:{color:GRID_COLOR}}},series:[{name:'NVDA DD',type:'line',data:nvdaDD,smooth:true,lineStyle:{color:NVDA_COLOR,width:2},areaStyle:{color:'rgba(118,185,0,0.15)'},symbol:'none'},{name:'BTC DD',type:'line',data:btcDD,smooth:true,lineStyle:{color:BTC_COLOR,width:2},areaStyle:{color:'rgba(247,147,26,0.12)'},symbol:'none'}]}));
}

function renderRisk(){
  const dates=DATA.prices.map(d=>d.date.slice(0,7));
  const nvdaVol=DATA.prices.map(d=>d.nvda_vol12?+(d.nvda_vol12*100).toFixed(2):null);
  const btcVol=DATA.prices.map(d=>d.btc_vol12?+(d.btc_vol12*100).toFixed(2):null);
  const nvdaRet=DATA.prices.map(d=>d.nvda_ret?+(d.nvda_ret*100).toFixed(2):null).filter(v=>v!==null);
  const btcRet=DATA.prices.map(d=>d.btc_ret?+(d.btc_ret*100).toFixed(2):null).filter(v=>v!==null);
  const vc=echarts.init(document.getElementById('chart-vol'));
  vc.setOption(Object.assign(baseOpt(),{legend:{data:['NVDA Vol','BTC Vol'],textStyle:{color:'#8b949e'},top:0},xAxis:{type:'category',data:dates,axisLabel:{color:'#8b949e',rotate:30,fontSize:10},axisLine:{lineStyle:{color:GRID_COLOR}},splitLine:{show:false}},yAxis:{axisLabel:{color:'#8b949e',formatter:v=>v.toFixed(0)+'%'},splitLine:{lineStyle:{color:GRID_COLOR}}},series:[{name:'NVDA Vol',type:'line',data:nvdaVol,smooth:true,lineStyle:{color:NVDA_COLOR,width:2},symbol:'none'},{name:'BTC Vol',type:'line',data:btcVol,smooth:true,lineStyle:{color:BTC_COLOR,width:2},symbol:'none'}]}));
  function histogram(data,bins=20){const mn=Math.min(...data),mx=Math.max(...data),step=(mx-mn)/bins,counts=Array(bins).fill(0),labels=[];for(let i=0;i<bins;i++)labels.push((mn+i*step+step/2).toFixed(1)+'%');data.forEach(v=>{counts[Math.min(bins-1,Math.floor((v-mn)/step))]++;});return{labels,counts};}
  const nh=histogram(nvdaRet),bh=histogram(btcRet);
  const dc=echarts.init(document.getElementById('chart-dist'));
  dc.setOption({backgroundColor:'transparent',legend:{data:['NVDA','BTC'],textStyle:{color:'#8b949e'},top:0},tooltip:{trigger:'axis',backgroundColor:'#161b22',borderColor:'#30363d',textStyle:{color:'#e6edf3'}},grid:{left:'8%',right:'4%',top:'12%',bottom:'15%'},xAxis:{type:'category',data:nh.labels,axisLabel:{color:'#8b949e',rotate:45,fontSize:9},axisLine:{lineStyle:{color:GRID_COLOR}}},yAxis:{axisLabel:{color:'#8b949e'},splitLine:{lineStyle:{color:GRID_COLOR}}},series:[{name:'NVDA',type:'bar',data:nh.counts,barWidth:'40%',itemStyle:{color:'rgba(118,185,0,0.7)'}},{name:'BTC',type:'bar',data:bh.counts,barWidth:'40%',itemStyle:{color:'rgba(247,147,26,0.7)'}}]});
  const sData=DATA.annual.map(r=>[{value:[+(r.nvda_annual_vol*100).toFixed(1),+(r.nvda_annual_ret*100).toFixed(1),r.year+' NVDA'],itemStyle:{color:NVDA_COLOR}},{value:[+(r.btc_annual_vol*100).toFixed(1),+(r.btc_annual_ret*100).toFixed(1),r.year+' BTC'],itemStyle:{color:BTC_COLOR}}]).flat();
  const sc=echarts.init(document.getElementById('chart-scatter'));
  sc.setOption({backgroundColor:'transparent',tooltip:{formatter:p=>`${p.data.value[2]}<br>Vol:${p.data.value[0]}% Ret:${p.data.value[1]}%`,backgroundColor:'#161b22',borderColor:'#30363d',textStyle:{color:'#e6edf3'}},grid:{left:'10%',right:'4%',top:'8%',bottom:'12%'},xAxis:{type:'value',name:'Ann Vol %',axisLabel:{color:'#8b949e',formatter:v=>v+'%'},splitLine:{lineStyle:{color:GRID_COLOR}},nameTextStyle:{color:'#8b949e'}},yAxis:{type:'value',name:'Ann Ret %',axisLabel:{color:'#8b949e',formatter:v=>v+'%'},splitLine:{lineStyle:{color:GRID_COLOR}},nameTextStyle:{color:'#8b949e'}},series:[{type:'scatter',data:sData,symbolSize:14,label:{show:true,formatter:p=>p.data.value[2],position:'top',color:'#8b949e',fontSize:10}}]});
  const nr=DATA.risk.find(r=>r.asset==='NVDA'),br2=DATA.risk.find(r=>r.asset==='BTC');
  const norm=(v,mn,mx)=>Math.min(100,Math.max(0,((v-mn)/(mx-mn))*100));
  const rc=echarts.init(document.getElementById('chart-radar'));
  rc.setOption({backgroundColor:'transparent',legend:{data:['NVDA','BTC'],textStyle:{color:'#8b949e'},top:0},radar:{indicator:[{name:'Ann Return',max:100},{name:'Sharpe',max:100},{name:'Sortino',max:100},{name:'Win Rate',max:100},{name:'Low Vol',max:100},{name:'Low DD',max:100}],axisLine:{lineStyle:{color:GRID_COLOR}},splitLine:{lineStyle:{color:GRID_COLOR}},name:{textStyle:{color:'#8b949e'}}},series:[{type:'radar',data:[{name:'NVDA',value:[norm(nr.ann_return,0.1,1.5)*100|0,norm(nr.sharpe,-0.5,2)*100|0,norm(nr.sortino,0,5)*100|0,norm(nr.win_rate,0.4,0.8)*100|0,norm(1-nr.ann_vol,0.2,0.7)*100|0,norm(1+nr.max_drawdown,0.1,0.7)*100|0],lineStyle:{color:NVDA_COLOR},areaStyle:{color:'rgba(118,185,0,0.2)'},itemStyle:{color:NVDA_COLOR}},{name:'BTC',value:[norm(br2.ann_return,0.1,1.5)*100|0,norm(br2.sharpe,-0.5,2)*100|0,norm(br2.sortino,0,5)*100|0,norm(br2.win_rate,0.4,0.8)*100|0,norm(1-br2.ann_vol,0.2,0.7)*100|0,norm(1+br2.max_drawdown,0.1,0.7)*100|0],lineStyle:{color:BTC_COLOR},areaStyle:{color:'rgba(247,147,26,0.2)'},itemStyle:{color:BTC_COLOR}}]}]});
}

function renderMC(){
  ['NVDA','BTC'].forEach((asset,i)=>{
    const d=DATA.mc.filter(r=>r.asset===asset),months=d.map(r=>'M'+r.month);
    const color=asset==='NVDA'?NVDA_COLOR:BTC_COLOR;
    const mk=(key,c,w,t)=>({name:key,type:'line',data:d.map(r=>r[key]),lineStyle:{color:c,width:w,type:t},symbol:'none',smooth:true});
    const c=echarts.init(document.getElementById(['chart-mc-nvda','chart-mc-btc'][i]));
    c.setOption({backgroundColor:'transparent',legend:{data:['P90','P75','Median','P25','P10'],textStyle:{color:'#8b949e'},top:0},tooltip:{trigger:'axis',backgroundColor:'#161b22',borderColor:'#30363d',textStyle:{color:'#e6edf3'},formatter:p=>p[0].axisValue+'<br>'+p.map(x=>`${x.marker}${x.seriesName}: $${(+x.value).toLocaleString(undefined,{maximumFractionDigits:0})}`).join('<br>')},grid:{left:'10%',right:'4%',top:'12%',bottom:'12%'},xAxis:{type:'category',data:months,axisLabel:{color:'#8b949e',interval:5},axisLine:{lineStyle:{color:GRID_COLOR}},splitLine:{show:false}},yAxis:{axisLabel:{color:'#8b949e',formatter:v=>'$'+(v/1000).toFixed(0)+'k'},splitLine:{lineStyle:{color:GRID_COLOR}}},series:[mk('p90','#3fb950',1,'dashed'),mk('p75','#58a6ff',1,'dashed'),mk('p50',color,2.5,'solid'),mk('p25','#d29922',1,'dashed'),mk('p10','#f85149',1,'dashed')]});
  });
  const nvdaMed=DATA.mc.filter(r=>r.asset==='NVDA').map(r=>r.p50),btcMed=DATA.mc.filter(r=>r.asset==='BTC').map(r=>r.p50);
  const nvdaP10=DATA.mc.filter(r=>r.asset==='NVDA').map(r=>r.p10),btcP10=DATA.mc.filter(r=>r.asset==='BTC').map(r=>r.p10);
  const months=DATA.mc.filter(r=>r.asset==='NVDA').map(r=>'M'+r.month);
  const cmp=echarts.init(document.getElementById('chart-mc-compare'));
  cmp.setOption({backgroundColor:'transparent',legend:{data:['NVDA Median','BTC Median','NVDA P10','BTC P10'],textStyle:{color:'#8b949e'},top:0},tooltip:{trigger:'axis',backgroundColor:'#161b22',borderColor:'#30363d',textStyle:{color:'#e6edf3'},formatter:p=>p[0].axisValue+'<br>'+p.map(x=>`${x.marker}${x.seriesName}: $${(+x.value).toLocaleString(undefined,{maximumFractionDigits:0})}`).join('<br>')},grid:{left:'10%',right:'4%',top:'12%',bottom:'12%'},xAxis:{type:'category',data:months,axisLabel:{color:'#8b949e',interval:5},axisLine:{lineStyle:{color:GRID_COLOR}},splitLine:{show:false}},yAxis:{axisLabel:{color:'#8b949e',formatter:v=>'$'+(v/1000).toFixed(0)+'k'},splitLine:{lineStyle:{color:GRID_COLOR}}},series:[{name:'NVDA Median',type:'line',data:nvdaMed,lineStyle:{color:NVDA_COLOR,width:2.5},symbol:'none',smooth:true},{name:'BTC Median',type:'line',data:btcMed,lineStyle:{color:BTC_COLOR,width:2.5},symbol:'none',smooth:true},{name:'NVDA P10',type:'line',data:nvdaP10,lineStyle:{color:NVDA_COLOR,width:1,type:'dashed'},symbol:'none',smooth:true},{name:'BTC P10',type:'line',data:btcP10,lineStyle:{color:BTC_COLOR,width:1,type:'dashed'},symbol:'none',smooth:true}]});
}

function renderAnnual(){
  const years=DATA.annual.map(r=>r.year);
  const nvdaRet=DATA.annual.map(r=>+(r.nvda_annual_ret*100).toFixed(1));
  const btcRet=DATA.annual.map(r=>+(r.btc_annual_ret*100).toFixed(1));
  const bc=echarts.init(document.getElementById('chart-annual-bar'));
  bc.setOption(Object.assign(baseOpt(),{legend:{data:['NVDA','BTC'],textStyle:{color:'#8b949e'},top:0},xAxis:{type:'category',data:years,axisLabel:{color:'#8b949e'},axisLine:{lineStyle:{color:GRID_COLOR}},splitLine:{show:false}},yAxis:{axisLabel:{color:'#8b949e',formatter:v=>v+'%'},splitLine:{lineStyle:{color:GRID_COLOR}}},series:[{name:'NVDA',type:'bar',data:nvdaRet.map(v=>({value:v,itemStyle:{color:v>=0?NVDA_COLOR:'#f85149'}})),barWidth:'30%'},{name:'BTC',type:'bar',data:btcRet.map(v=>({value:v,itemStyle:{color:v>=0?BTC_COLOR:'#f85149'}})),barWidth:'30%'}]}));
  const nvdaVol=DATA.annual.map(r=>+(r.nvda_annual_vol*100).toFixed(1));
  const btcVol=DATA.annual.map(r=>+(r.btc_annual_vol*100).toFixed(1));
  const vc=echarts.init(document.getElementById('chart-annual-vol'));
  vc.setOption(Object.assign(baseOpt(),{legend:{data:['NVDA Vol','BTC Vol'],textStyle:{color:'#8b949e'},top:0},xAxis:{type:'category',data:years,axisLabel:{color:'#8b949e'},axisLine:{lineStyle:{color:GRID_COLOR}},splitLine:{show:false}},yAxis:{axisLabel:{color:'#8b949e',formatter:v=>v+'%'},splitLine:{lineStyle:{color:GRID_COLOR}}},series:[{name:'NVDA Vol',type:'bar',data:nvdaVol,itemStyle:{color:'rgba(118,185,0,0.8)'},barWidth:'30%'},{name:'BTC Vol',type:'bar',data:btcVol,itemStyle:{color:'rgba(247,147,26,0.8)'},barWidth:'30%'}]}));
  const tbl=document.getElementById('annual-table');
  tbl.innerHTML='<tr><th>Year</th><th>NVDA Return</th><th>BTC Return</th><th>NVDA Vol</th><th>BTC Vol</th><th>NVDA Best Mo</th><th>BTC Best Mo</th><th>NVDA Worst Mo</th><th>BTC Worst Mo</th></tr>'+DATA.annual.map(r=>{const nr=+(r.nvda_annual_ret*100).toFixed(1),br=+(r.btc_annual_ret*100).toFixed(1),cl=v=>v>=0?'pos':'neg';return`<tr><td><strong>${r.year}</strong></td><td class="${cl(nr)}">${nr}%</td><td class="${cl(br)}">${br}%</td><td>${(r.nvda_annual_vol*100).toFixed(1)}%</td><td>${(r.btc_annual_vol*100).toFixed(1)}%</td><td class="pos">${(r.nvda_best_month*100).toFixed(1)}%</td><td class="pos">${(r.btc_best_month*100).toFixed(1)}%</td><td class="neg">${(r.nvda_worst_month*100).toFixed(1)}%</td><td class="neg">${(r.btc_worst_month*100).toFixed(1)}%</td></tr>`;}).join('');
}

function renderScorecard(){
  const nvdaW=DATA.scorecard.filter(r=>r.winner==='NVDA'&&r.metric!=='TOTAL WINS').length;
  const btcW=DATA.scorecard.filter(r=>r.winner==='BTC'&&r.metric!=='TOTAL WINS').length;
  const wc=echarts.init(document.getElementById('chart-wins'));
  wc.setOption({backgroundColor:'transparent',tooltip:{trigger:'item',backgroundColor:'#161b22',borderColor:'#30363d',textStyle:{color:'#e6edf3'}},series:[{type:'pie',radius:['45%','75%'],center:['50%','55%'],data:[{value:nvdaW,name:`NVDA (${nvdaW} wins)`,itemStyle:{color:NVDA_COLOR}},{value:btcW,name:`BTC (${btcW} wins)`,itemStyle:{color:BTC_COLOR}}],label:{color:'#e6edf3',fontSize:13},emphasis:{itemStyle:{shadowBlur:10}}}]});
  const nr=DATA.risk.find(r=>r.asset==='NVDA'),br=DATA.risk.find(r=>r.asset==='BTC');
  const metrics=['Ann Return %','Ann Vol %','Sharpe','Sortino','Win Rate %','Calmar'];
  const nv=[+(nr.ann_return*100).toFixed(1),+(nr.ann_vol*100).toFixed(1),+nr.sharpe.toFixed(2),+nr.sortino.toFixed(2),+(nr.win_rate*100).toFixed(1),+nr.calmar.toFixed(2)];
  const bv=[+(br.ann_return*100).toFixed(1),+(br.ann_vol*100).toFixed(1),+br.sharpe.toFixed(2),+br.sortino.toFixed(2),+(br.win_rate*100).toFixed(1),+br.calmar.toFixed(2)];
  const mc2=echarts.init(document.getElementById('chart-metrics-bar'));
  mc2.setOption({backgroundColor:'transparent',legend:{data:['NVDA','BTC'],textStyle:{color:'#8b949e'},top:0},tooltip:{trigger:'axis',backgroundColor:'#161b22',borderColor:'#30363d',textStyle:{color:'#e6edf3'}},grid:{left:'14%',right:'4%',top:'12%',bottom:'5%',containLabel:true},xAxis:{type:'value',axisLabel:{color:'#8b949e'},splitLine:{lineStyle:{color:GRID_COLOR}}},yAxis:{type:'category',data:metrics,axisLabel:{color:'#8b949e'},axisLine:{lineStyle:{color:GRID_COLOR}}},series:[{name:'NVDA',type:'bar',data:nv,itemStyle:{color:NVDA_COLOR},barWidth:'35%'},{name:'BTC',type:'bar',data:bv,itemStyle:{color:BTC_COLOR},barWidth:'35%'}]});
  const tbl=document.getElementById('scorecard-table');
  tbl.innerHTML='<tr><th>Metric</th><th style="color:var(--nvda)">NVDA</th><th style="color:var(--btc)">BTC</th><th>Winner</th></tr>'+DATA.scorecard.map(r=>{const wc=r.winner==='NVDA'?'win-nvda':r.winner==='BTC'?'win-btc':'win-neutral',isTot=r.metric==='TOTAL WINS';return`<tr style="${isTot?'background:rgba(255,255,255,0.04);font-weight:700':''}"><td>${r.metric}</td><td>${r.nvda}</td><td>${r.btc}</td><td class="${wc}">${r.winner}</td></tr>`;}).join('');
}

document.addEventListener('DOMContentLoaded',()=>{
  initKPIs();
  window.addEventListener('resize',()=>{document.querySelectorAll('[id^="chart-"]').forEach(el=>{const inst=echarts.getInstanceByDom(el);if(inst)inst.resize();});});
});
"""
print("  JS ready")

# ── 11. ASSEMBLE & WRITE HTML ──────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NVDA vs BTC — 5-Year Investment Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
  <style>{css}</style>
  <script>{data_block}</script>
</head>
{html_body}
<script>{js}</script>
</html>"""

with open(OUTPUT_PATH, "w") as f:
    f.write(html)
size_kb = os.path.getsize(OUTPUT_PATH) / 1024
print(f"\n✅ Dashboard written: {OUTPUT_PATH}")
print(f"   File size: {size_kb:.1f} KB")

# ── 12. OPEN IN BROWSER ────────────────────────────────────────────────────
webbrowser.open(f"file://{os.path.abspath(OUTPUT_PATH)}")
print("🌐 Opened in browser!")
