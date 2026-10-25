# Static artifact candidate

This opt-in path gathers observed source data once, computes request-scoped measures,
binds the exact JSON to the page, and generates either a reusable presentation program
or a normal HTML page when the runtime cannot express the requested interface.
The normal static and desktop/phone browser verifiers remain required.
Live-service and durable-state requests retain the existing fullstack pipeline.

## Provenance and staging port

The evaluated Anton package was frozen at base commit
`e0b568bedb3a5fc99ac67ff1010452ec0507739e`; its 198-file package tree SHA-256 was
`e7dabc21692e34cdc8d342991401ac0ad4aa9d3cf0d4340fc52eee857c508eee`.
This stack ports those changes onto staging `83aa4e7e9f60056d0c1b743fd39c3f89b6a4d069`.
It preserves staging's per-call effort arguments, known-data source verification,
project rules, streamed discovery, and default failure messages. These adaptations
mean this staging tree has not itself received the historical performance scores.
Later source-review and idle-globe experiments are excluded.

The companion renderer change adds optional initial-main materialization, then checks
the exact returned HTML at both viewport sizes with the ordinary rules.
Anton validates input/output hashes before saving the returned HTML.
Use it only with the matching renderer feature branch; an older renderer will not
accept the optional materialization request.

## Enabling the evaluated configuration

All behavior is opt-in. Existing deployments keep their default workflow.
The following settings reproduce the feature selection; configure models and provider
credentials through the deployment's existing supported configuration. No credentials,
customer fixtures, expected answers, task IDs, or provider bridges are included here.

```dotenv
ANTON_STATIC_ATOMIC_GATHER=on
ANTON_STATIC_BATCH_FINISH=on
ANTON_STATIC_BATCH_GATHER=on
ANTON_STATIC_CANONICAL_ONLY=on
ANTON_STATIC_CHART_TABLES=on
ANTON_STATIC_COHERENT_ENTRY=on
ANTON_STATIC_COHERENT_GATHER=on
ANTON_STATIC_COMPACT_CONTRACT=on
ANTON_STATIC_COMPONENT_WRITER=on
ANTON_STATIC_CONNECTED_PROCEDURES=on
ANTON_STATIC_DATA_BINDING=on
ANTON_STATIC_DECISION_LAYOUT=on
ANTON_STATIC_FAST_PATH=direct
ANTON_STATIC_GATHER_EFFORT=low
ANTON_STATIC_GATHER_GEOCODING=on
ANTON_STATIC_GEO_CAPABILITY=on
ANTON_STATIC_GEO_CONTRACT=on
ANTON_STATIC_GEO_CONTROLS=on
ANTON_STATIC_LAYOUT_GUARD=on
ANTON_STATIC_LAYOUT_OPTIONS=on
ANTON_STATIC_METRIC_SCOPE=on
ANTON_STATIC_PAGE_CHARS=9000
ANTON_STATIC_PERSISTENT_SUMMARY=on
ANTON_STATIC_PROFILE_FIRST=off
ANTON_STATIC_QUERY_CLIENT=off
ANTON_STATIC_REFERENCE_METRICS=on
ANTON_STATIC_RENDER_CONTRACT=on
ANTON_STATIC_REQUEST_FIDELITY=on
ANTON_STATIC_REQUEST_METRICS=on
ANTON_STATIC_REUSE_SNAPSHOT=on
ANTON_STATIC_SEMANTIC_HTML=on
ANTON_STATIC_SINGLE_CALL=on
ANTON_STATIC_SOURCE_SEMANTICS=on
ANTON_STATIC_STANDARD_DESIGN=on
ANTON_STATIC_STATE_SETS=on
ANTON_STATIC_TIMELINE_AXIS=on
ANTON_STATIC_WRITER_TOKENS=4000
CODING_REASONING_EFFORT=low
PLANNING_REASONING_EFFORT=low
```

The page size and token values are writer budgets, not correctness limits. Source
limits and model truncation can still require the fallback or return a concrete failure.
Do not enable only the entry prompt without the generator and source-data changes.

## Validation

The staging port passed 505 focused and compatibility tests, including configured
effort, provider overrides, project rules, discovery truncation, gathering, source
contracts, source-data binding, presentation generation, and renderer hash checks.
The historical three-run MindsHub validation of the frozen package scored 57.4074%
passing over 132 primary turns, with 68.2727 s mean and 97.6 s p90. It did not meet
the 90 s p90 target. The seen energy regression scored 60% over 39 turns, with
59.8795 s mean and 79.2 s p90. These are separate from the Codex/Astra judging series.
No generalization beyond the evaluated tasks is claimed.
