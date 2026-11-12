# Artifact quick-edit bar benchmark, November 2026 (ENG-4227)

Question: a message typed in the artifact viewer's bar can either go straight to
the artifact editor (no chat turn) or to the agent. Does Jev route bar messages
correctly, how fast is the bar end to end, and what happens when the router is wrong?

Answer: with the artifact's name and data shape as context, Jev (`jev-1.13.0`)
routed 142 of 144 labelled messages correctly, with no agent request ever routed
to the editor, in a median 0.65 s. End to end (route, edit, browser check) the
bar took a median 14.2 s (p90 25.2 s) on the #790 benchmark's 28 edit tasks, against
18.0 s for an agent turn calling `edit_artifact` and 34.3 s for today's agent.
When the editor gets a message the agent should have had, it hands it back
instead of faking data in 31 of 32 cases.

## Setup

* **Labelled set**: `tests/fixtures/artifact_edits/routing.json`, 72 messages as typed in the
  bar, on the 7 benchmark artifacts of #790 (`tests/fixtures/artifact_edits/`, synthetic data):
  * 40 `edit`: the #790 benchmark's 28 tasks with the artifact's name dropped (the
    bar is under the artifact), plus 12 harder ones. Some sound like data work but only use
    data already in the page ("sort the table by on-hand quantity", "add a gross margin chart
    by month" when gross margin is in the data, "show each region's share of total revenue").
    Two are follow-ups ("bigger", "undo that, put it back to teal") with earlier messages.
  * 32 `agent`: data the artifact does not have ("add last quarter's numbers", "add 2025",
    "add the Mexican peso", "add a filter by customer segment" when there is no segment field),
    queries and refreshes, analysis and forecasts, questions ("what's our overall NPS?"), a new
    artifact, sharing, exporting, scheduling, and "hi".
* **Router**: `route_request` against MindsHub (`https://api.mindshub.ai/v1/decisions`),
  `jev-1.13.0`, 3 s timeout, accepted at p ≥ 0.8. There are two variants. `contract` sends the
  message, file list and kind. `context` also sends the artifact's name, the shape of its
  embedded data (`describe_artifact`) and the earlier bar messages. Each message ran 2 times per
  variant (288 calls), one at a time.
* **End to end**: for each of the 28 #790 tasks, twice (56 runs), on a fresh copy of the
  artifact: `route_request` (context variant) then, when routed to edit, `run_quick_edit`
  with the checker the #790 benchmark's arm C used (static checks, then a local render-check
  service with Cowork's preview policy). Model: `mindshub_air` (planning role). Each run is its
  own process with a throwaway HOME and workspace and its own render-check service, 4 at a
  time. Same task assertions and independent "page renders after" check as #790.
* **Escalation probe**: `run_quick_edit` called directly (as if the router had said edit) on
  all 32 agent-labelled messages, once each.
* **Script**: `scripts/bench_quick_edit.py --routing | --e2e | --escalation`. Raw rows are in
  `artifact-quick-edit-2026-11/`.

## Routing

| Variant | N | Accuracy | edit→edit | edit→agent | agent→agent | agent→edit | Not "jev" (uncertain/error) | Median ms | p90 ms |
|---|---|---|---|---|---|---|---|---|---|
| contract (message, files, kind) | 144 | 140/144 (97%) | 78 | 2 | 62 | 2 | 10 | 653 | 849 |
| **context** (+ name, data shape, earlier messages) | 144 | **142/144 (99%)** | 78 | 2 | **64** | **0** | 18 | 648 | 846 |

The two directions of error do not cost the same. Edit→agent is only slower, because the
agent can edit too. Agent→edit sends a data request to the editor, which then has to hand it
back.

* `contract` misrouted "add a filter by customer segment" to edit (p 0.85–0.88): without the
  data shape it cannot know the survey has no segment field. It was uncertain on "add a gross
  margin chart by month" (p 0.56–0.59) and sent it to the agent.
* `context` routed both correctly, and sent no agent message to the editor. Its only miss was
  "highlight milestones whose date has passed but are not done in red", which it sent to the
  agent (p 0.23) in both runs. That costs time, not correctness.
* No timeouts or errors in 288 calls. Every misroute was the same on both runs.

The setup is documented below so the numbers can be reproduced. The criteria were written
before this run and not tuned on it.

## End to end: the bar vs #790's numbers (`mindshub_air`, the same 28 tasks)

| Path | N | Median (s) | p90 (s) | Assertions pass |
|---|---|---|---|---|
| #790 A: agent turn, editor off (today) | 84 | 34.3 | 50.5 | 82/84 |
| #790 B: agent turn calling `edit_artifact` | 84 | 18.0 | 27.9 | 82/84 |
| #790 C: editor called directly (no routing) | 84 | 9.8 | 19.1 | 83/84 |
| **Quick-edit bar: route + edit + check** | 56 | **14.2** | **25.2** | 53/56 |

* All 56 runs routed to edit (median 0.92 s, p90 1.20 s for the router in this run) and all 56
  applied. None escalated, failed or came back as noop. Median 1 LLM call. A fix round from the
  check's findings ran in 2 of 56 runs; both then passed.
* The html-app pages all rendered afterwards (48/48), and no run edited files outside its task
  (0/56). The fullstack app's 8 runs were checked statically only (unverified), as in #790.
* By category, median total: cosmetic 14.2 s, content 11.5 s, behaviour 20.4 s, bug fix 13.1 s.
* The editor part matches #790 C component by component. The median time outside the check
  was 4.5 s here and 4.5 s in #790 C's raw runs, and the median browser check 6.2 s vs 6.0 s.
  The bar adds the router (about 0.9 s). In this sample the median of the editor's total was
  13.0 s against 10.2 s over #790 C's raw runs. That is run-to-run variance in the model and
  the render check (a laptop, 4 checks at once), not a different code path.
* **Assertion failures (3)** are the same kinds #790 recorded. In one, the renamed card's old
  label survived in a second place. In another, the dropdown was written in a shape the
  assertion's regex does not match. In the third, the currency fix was correct by its own
  account ("100 USD now converts to 92 EUR") but written differently from what the assertion
  expects. We count all three as failures.

Against #790 B, the bar saves the two agent model calls around the edit, which is about 4 s
at the median. The p90 gap is smaller because the browser check dominates the tail.

## Escalation: the safety net behind the router

The 32 agent-labelled messages were sent straight to `run_quick_edit` (1 run each):

* **escalate 30**: the editor called `hand_to_agent` with a reason the bar shows, for example
  "The embedded dataset contains only 2026 records; adding a 2025 comparison requires 2025
  source data that is not present in the artifact." The folder was left unchanged every time.
* **noop 1**: "log my sessions to a Google Sheet every day". The editor replied that it has no
  account or sheet to connect to and changed nothing.
* **applied 1**: "add last quarter's numbers" on the sales dashboard, which holds all of 2026.
  The editor added Q4 2026 KPIs computed from that data. This is defensible when the data
  covers the quarter. The router sends this message to the agent anyway (p ≥ 0.8 agent in
  every run).
* Median 4.1 s (p90 6.7 s) to hand back, one model call. Every folder not applied was left
  byte for byte as it was (31/31).

## Caveats

* The labelled set is ours, written for this benchmark on 7 synthetic artifacts. Real bar
  traffic will differ. The routing rows (`routing.jsonl`) should be re-run against real bar
  messages once the bar ships.
* Router latency was measured one call at a time from a laptop. In the end-to-end runs, with
  4 jobs at once, the router's median was 0.92 s.
* Without a decisions model (`ANTON_DECISIONS_MODEL`; Jev by default only on MindsHub) the bar
  never edits directly and every message goes to the agent. Those setups get #790 B's timing.
