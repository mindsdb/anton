| Model | Arm | N | Median time (s) | p90 time (s) | Median LLM calls | Median input tok (incl. cached) | Median output tok | Assertions pass | Page renders after | Checked in browser | Edited other files |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mindshub_air | A baseline (flag off) | 84 | 34.3 | 50.5 | 5 | 99,292 | 874 | 82/84 | 72/72 | 72/84 | 2/84 |
| mindshub_air | B edit_artifact (flag on) | 84 | 18.0 | 27.9 | 3 | 38,472 | 518 | 82/84 | 72/72 | 72/84 | 0/84 |
| mindshub_air | C editor direct | 84 | 9.8 | 19.1 | 1 | 4,998 | 176 | 83/84 | 72/72 | 72/84 | 0/84 |
| sonnet-5-5 | A baseline (flag off) | 56 | 31.6 | 59.7 | 5 | 130,585 | 1,388 | 56/56 | 48/48 | 48/56 | 0/56 |
| sonnet-5-5 | B edit_artifact (flag on) | 56 | 20.1 | 28.3 | 3 | 55,736 | 901 | 55/56 | 48/48 | 48/56 | 0/56 |
| sonnet-5-5 | C editor direct | 56 | 9.5 | 16.5 | 1 | 7,732 | 310 | 55/56 | 48/48 | 48/56 | 0/56 |

| Model | Arm | Category | N | Median time (s) | Median calls | Assertions pass |
|---|---|---|---|---|---|---|
| mindshub_air | A | cosmetic | 21 | 29.3 | 5 | 21/21 |
| mindshub_air | A | content | 21 | 31.2 | 5 | 21/21 |
| mindshub_air | A | behaviour | 21 | 36.9 | 5 | 19/21 |
| mindshub_air | A | bugfix | 21 | 34.8 | 5 | 21/21 |
| mindshub_air | B | cosmetic | 21 | 17.6 | 3 | 20/21 |
| mindshub_air | B | content | 21 | 16.3 | 3 | 20/21 |
| mindshub_air | B | behaviour | 21 | 24.1 | 3 | 21/21 |
| mindshub_air | B | bugfix | 21 | 18.5 | 3 | 21/21 |
| mindshub_air | C | cosmetic | 21 | 9.8 | 1 | 20/21 |
| mindshub_air | C | content | 21 | 9.2 | 1 | 21/21 |
| mindshub_air | C | behaviour | 21 | 16.6 | 1 | 21/21 |
| mindshub_air | C | bugfix | 21 | 9.5 | 1 | 21/21 |
| sonnet-5-5 | A | cosmetic | 14 | 30.4 | 5 | 14/14 |
| sonnet-5-5 | A | content | 14 | 28.9 | 5 | 14/14 |
| sonnet-5-5 | A | behaviour | 14 | 33.0 | 5 | 14/14 |
| sonnet-5-5 | A | bugfix | 14 | 34.3 | 5 | 14/14 |
| sonnet-5-5 | B | cosmetic | 14 | 18.5 | 3 | 13/14 |
| sonnet-5-5 | B | content | 14 | 14.2 | 3 | 14/14 |
| sonnet-5-5 | B | behaviour | 14 | 24.5 | 4 | 14/14 |
| sonnet-5-5 | B | bugfix | 14 | 22.2 | 4 | 14/14 |
| sonnet-5-5 | C | cosmetic | 14 | 10.0 | 1 | 13/14 |
| sonnet-5-5 | C | content | 14 | 8.8 | 1 | 14/14 |
| sonnet-5-5 | C | behaviour | 14 | 15.4 | 1 | 14/14 |
| sonnet-5-5 | C | bugfix | 14 | 9.1 | 1 | 14/14 |

What the baseline did (tools the agent called, arm A):

- mindshub_air: {'scratchpad': 82, 'generate_artifact': 2}
- mindshub_air, arm B: edit_artifact used in 84/84 runs
- sonnet-5-5: {'scratchpad': 56}
- sonnet-5-5, arm B: edit_artifact used in 56/56 runs
