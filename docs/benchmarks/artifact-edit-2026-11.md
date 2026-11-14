# Artifact edit benchmark, November 2026 (ENG-4207)

Question: when someone asks Anton to change an artifact they already have, is the
new `edit_artifact` path faster than today's, and is it as correct?

Answer: yes. On MindsHub's production model (`mindshub_air`) the median edit turn
went from 34.3 s to 18.0 s, with 3 model calls instead of 5 and 61% fewer prompt
tokens, and the same share of edits passed their checks. On `sonnet-5-5` it went
from 31.6 s to 20.1 s. The editor on its own (no agent turn around it) takes a
median 9.5-9.8 s and one model call.

## Setup

* **Tasks**: `tests/fixtures/artifact_edits/tasks.json`, 28 tasks on 7 artifacts:
  a cosmetic change, a content or label change, a behaviour change (a filter, a
  table, a mode, a swap button) and a seeded bug fix for each. The artifacts were
  generated once with anton's own `generate_artifact` (3 component pages with
  bound data, 3 single-file pages, 1 fullstack-stateless app) from synthetic data.
  Each task has deterministic assertions on the resulting files.
* **Arms**, same task, same model, a fresh copy of the artifact per run, each run
  its own process with a throwaway HOME and workspace:
  * **A, baseline**: a real `ChatSession` turn with `ANTON_ARTIFACT_EDITOR=off`,
    i.e. staging's behaviour.
  * **B, new**: the same turn with `ANTON_ARTIFACT_EDITOR=on`.
  * **C, editor direct**: `ArtifactEditor.edit(task)` with the same checker, no
    agent turn around it. The lower bound.
* **Models**: MindsHub (`https://api.mindshub.ai`), planning and coding both set
  to the model under test: `mindshub_air` (Cowork's production default), N = 3
  runs per task per arm (84 runs per arm); `sonnet-5-5`, N = 2 (56 per arm).
* **Browser check**: a local render-check service (the cowork-self-hosted
  `render-check` image with Cowork's real preview policy and shim), one service
  per running job, used by every arm. Every run's resulting page was also loaded
  by the same service afterwards ("page renders after"). The fullstack app has no
  live-app slot here, so it was checked statically and is unverified in every arm.
* **Script**: `scripts/bench_artifact_edit.py --arms A,B,C --models mindshub_air
  --runs 3 --render-check-url ...`. Per-run numbers are in
  `artifact-edit-2026-11/*/runs.csv`.

## Results (final code)

| Model | Arm | N | Median time (s) | p90 time (s) | Median LLM calls | Median prompt tokens (incl. cached) | Median output tokens | Assertions pass | Page renders after | Checked in a browser | Edited other files |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mindshub_air | A baseline (flag off) | 84 | 34.3 | 50.5 | 5 | 99,292 | 874 | 82/84 | 72/72 | 72/84 | 2/84 |
| mindshub_air | B edit_artifact (flag on) | 84 | **18.0** | **27.9** | **3** | **38,472** | 518 | 82/84 | 72/72 | 72/84 | 0/84 |
| mindshub_air | C editor direct | 84 | 9.8 | 19.1 | 1 | 4,998 | 176 | 83/84 | 72/72 | 72/84 | 0/84 |
| sonnet-5-5 | A baseline (flag off) | 56 | 31.6 | 59.7 | 5 | 130,585 | 1,388 | 56/56 | 48/48 | 48/56 | 0/56 |
| sonnet-5-5 | B edit_artifact (flag on) | 56 | **20.1** | **28.3** | **3** | **55,736** | 901 | 55/56 | 48/48 | 48/56 | 0/56 |
| sonnet-5-5 | C editor direct | 56 | 9.5 | 16.5 | 1 | 7,732 | 310 | 55/56 | 48/48 | 48/56 | 0/56 |

"Checked in a browser" is 72/84 and 48/56 in every arm because the 12 (8)
fullstack runs had no live app to load. "Page renders after" counts the html-app
runs only.

By category (median seconds, `mindshub_air`): cosmetic 29.3 → 17.6, content
31.2 → 16.3, behaviour 36.9 → 24.1, bug fix 34.8 → 18.5. Fullstack runs: 37.1 →
11.9. Full table in `artifact-edit-2026-11/final/summary.md`.

The editor itself (arm C) landed the edit in one model call in 80/84
(`mindshub_air`) and 50/56 (`sonnet-5-5`) runs, and in at most two calls in 84/84
and 55/56. No run needed the whole-file rewrite, and none gave up.

## What the baseline actually did

With the flag off the agent edited the page by hand in 138 of 140 runs: it found
the artifact (`list_artifacts`), read it (`read_text_file`) and rewrote it in a
scratchpad cell, after which the existing cell check loaded the page in the
render check. In 2 runs (`mindshub_air`, the fullstack app's "add a swap button")
it called `generate_artifact` to rebuild the app instead; both rebuilds failed in
this environment (`No module named 'anton_state'` in the generated backend's
check), took 96-122 s and 22-31 model calls, and changed `backend.py` and
`requirements.txt` without completing the task. Those are the two baseline
assertion failures and the two "edited other files".

With the flag on, the agent called `edit_artifact` in 140 of 140 runs. It went
straight to it, by the artifact's name, in 83 of 84 `mindshub_air` runs and 37
of 56 `sonnet-5-5` runs; the rest called `list_artifacts` first. In 2
`sonnet-5-5` runs the agent also ran scratchpad cells after `edit_artifact` (to
look at the page again), which the tool result asks it not to do.

## Where the time goes

A turn on the new path is three model calls: the agent deciding to call
`edit_artifact` (about 3-4 s), the editor's one call (3-8 s), and the agent's
reply (3-5 s), plus the browser check (3-20 s; a component page is checked twice,
once as it was before the edit for the comparison). The baseline spends two more
agent calls (finding and reading the file) and writes the edit as a Python
program in a cell, with the full chat history in every call: that is the 2.6x
difference in prompt tokens.

## Failures and caveats

* **Assertion failures on the new path** (B and C, final code: 2 + 1 on
  `mindshub_air`, 1 + 1 on `sonnet-5-5`) were partial or differently shaped
  edits: the revenue line made purple in one place while the old teal stayed in
  another (2), the renamed card's old label left in a second spot (1), score
  colors written as RGB arrays rather than the hex the assertion looks for (1),
  and the Start button's rule reworded so the assertion's regex no longer matched
  (1). The render check passed every one of them. The baseline had no such
  failure on html pages; its two failures were the fullstack rebuilds above. We
  count all of these as failures.
* **Two code versions.** B and C in the table above ran on this PR's final code;
  A ran the day's earlier batch (its code path, with the flag off, is unchanged
  by the later commits). The earlier batch also ran B and C interleaved with A in
  the same window, on the code before the last three fixes (two patch-header
  slips, the data shape beside the masked data, and `list_artifacts` routing).
  That same-window comparison, in `artifact-edit-2026-11/same-window/`, is the
  strict apples-to-apples one: `mindshub_air` A 34.3 s / B 20.7 s / C 10.8 s,
  `sonnet-5-5` A 31.6 s / B 28.2 s / C 13.9 s. The `sonnet-5-5` gap there was
  small because Sonnet opened most patches with a bare `@@` (one wasted editor
  call each) and called `list_artifacts` first in 33 of 56 runs. The final code
  reads the bare `@@` as meant and routes named artifacts straight to
  `edit_artifact` (19 of 56 still listed first), which took its B median to
  20.1 s.
* **Fullstack apps** were not loaded in a browser in any arm (no live-app slot
  here). The render check's URL mode is wired for them (`RenderUrlChecker`) but
  was not exercised by this benchmark.
* **Prompt-token counts** are what the provider reported (fresh + cache read +
  cache write). Model calls made by code inside scratchpad cells are not
  counted; the baseline made none in these runs.
* Times include the browser check, which ran on a laptop alongside other work.
  The arms were interleaved (run, task, arm) so load fell on every arm alike.
