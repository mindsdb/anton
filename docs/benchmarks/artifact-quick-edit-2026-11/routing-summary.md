| Variant | N | Accuracy | edit→edit | edit→agent | agent→agent | agent→edit | Uncertain/error | Median ms | p90 ms |
|---|---|---|---|---|---|---|---|---|---|
| contract | 144 | 140/144 (97%) | 78 | 2 | 62 | 2 | 10 | 653 | 849 |
| context | 144 | 142/144 (99%) | 78 | 2 | 64 | 0 | 18 | 648 | 846 |

Misrouted:

- contract run 1: "add a gross margin chart by month" labelled edit, routed agent (uncertain, p=0.56)
- contract run 1: "add a filter by customer segment" labelled agent, routed edit (jev, p=0.88)
- contract run 2: "add a gross margin chart by month" labelled edit, routed agent (uncertain, p=0.59)
- contract run 2: "add a filter by customer segment" labelled agent, routed edit (jev, p=0.85)
- context run 1: "highlight milestones whose date has passed but are not done in red" labelled edit, routed agent (uncertain, p=0.23)
- context run 2: "highlight milestones whose date has passed but are not done in red" labelled edit, routed agent (uncertain, p=0.23)
