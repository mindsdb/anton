import csv, io
# Per-account monthly revenue. Story: total dips 18% Apr->May, but it's ENTIRELY
# the Enterprise account "Acme" churning. Mid-Market GROWS, SMB flat -> a naive
# "across-the-board decline" read is FALSE.
rows = []
def add(month, seg, acct, rev): rows.append((month, seg, acct, rev))

# Jan-Apr: Acme stable ~23-24k; others gently growing. May: Acme collapses to 6k.
plan = {
 # account: (seg, [Jan,Feb,Mar,Apr,May])
 "Acme":   ("Enterprise", [23000,23500,24000,24000, 6000]),
 "Globex": ("Enterprise", [12000,12500,13000,14000,13000]),
 "Initech":("Enterprise", [10000,10500,11000,12000,11000]),
 "MidCorp":("Mid-Market", [14000,15000,15500,16000,17000]),
 "Meridian":("Mid-Market",[12000,12500,13500,14000,15000]),
 "Bluefin": ("SMB",       [ 7000, 7200, 7500, 8000, 8000]),
 "Cedar":   ("SMB",       [ 6000, 6300, 6700, 7000, 7000]),
 "Dune":    ("SMB",       [ 4500, 4700, 4800, 5000, 5000]),
}
months = ["2026-01","2026-02","2026-03","2026-04","2026-05"]
for acct,(seg,vals) in plan.items():
    for m,v in zip(months,vals): add(m,seg,acct,v)

# write
buf = io.StringIO(); w = csv.writer(buf)
w.writerow(["month","segment","account","revenue_usd"])
for r in sorted(rows): w.writerow(r)
open("evals/cases/fixtures/sales.csv","w").write(buf.getvalue())

# verify the story
from collections import defaultdict
tot = defaultdict(int); seg = defaultdict(lambda: defaultdict(int))
for m,s,a,v in rows:
    tot[m]+=v; seg[m][s]+=v
apr,may = tot["2026-04"], tot["2026-05"]
print(f"Apr total={apr}  May total={may}  drop={100*(apr-may)/apr:.1f}%")
for s in ["Enterprise","Mid-Market","SMB"]:
    a,b = seg["2026-04"][s], seg["2026-05"][s]
    print(f"  {s:12} Apr={a:6} May={b:6}  {100*(b-a)/a:+.1f}%")
print("Acme Apr->May:", plan['Acme'][1][3], "->", plan['Acme'][1][4])
