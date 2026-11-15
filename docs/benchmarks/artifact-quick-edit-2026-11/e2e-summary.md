Runs: 56 (0 harness errors excluded). Routed to edit: 56/56; outcomes: applied 56, noop 0, escalate 0, failed 0, agent 0.

| Path | N | Median (s) | p90 (s) | Assertions pass |
|---|---|---|---|---|
| #790 A baseline turn (flag off) | 84 | 34.3 | 50.5 | 82/84 |
| #790 B agent turn + edit_artifact | 84 | 18.0 | 27.9 | 82/84 |
| #790 C editor direct | 84 | 9.8 | 19.1 | 83/84 |
| **Quick-edit bar: route + edit + check** | 56 | **14.2** | **25.2** | 53/56 |

Router: median 920 ms, p90 1203 ms. Editor + check: median 13.0 s, p90 23.9 s. Median LLM calls 1; fix rounds used: 2/56 runs; page renders after (html-app, applied): 48/48; checked in a browser: 48/56; edited other files: 0/56.

| Category | N | Median total (s) | Assertions pass |
|---|---|---|---|
| cosmetic | 14 | 14.2 | 13/14 |
| content | 14 | 11.5 | 13/14 |
| behaviour | 14 | 20.4 | 14/14 |
| bugfix | 14 | 13.1 | 13/14 |

Not applied, or applied with a failed assertion:

- customer-survey-results-content.2: applied (route edit/jev, assert False)
- product-launch-timeline-cosmetic.2: applied (route edit/jev, assert False)
- currency-converter-bugfix.2: applied (route edit/jev, assert False)
