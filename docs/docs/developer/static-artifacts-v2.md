# Static artifacts v2

The replacement separates evidence from presentation. The agent chooses the story,
structure, visual form, and conclusions. The host resolves data queries and scalar
calculations, compiles standard reports and slides, and checks the actual page.
A custom HTML route preserves the ability to build an unusual visualization or
interaction without waiting for another component to be added.

## Evaluation rollout

`ANTON_STATIC_ARTIFACTS=v2` selects the new engine and its parent-agent authoring
contract for `html-app` generation in act-first mode. Fully specified HTML reports
and subsequent edits use the same engine. Reading a connected source at build time
does not require a backend; client-side filters and drill-downs use embedded data.
Apps that actually require protected-source refresh, authenticated actions, or
external writes at viewing time retain the fullstack route.
The flag is deliberately separate from the earlier experimental
static flags. Unset, the existing pipeline runs. Fullstack and confirmation-first
workflows retain their current behavior. Do not enable the replacement by default
until its quality and edit behavior have been measured on the unmodified artifact
benchmark. No benchmark names, fixtures, answers, or family classifiers belong in
this implementation.

The implementation is a five-PR stack: evidence/document contract; report
compiler; slide reader/presentation runtime; render/data integrity checks; generation and publication.

## Evidence

Collection reuses the session's existing scratchpad, text-file, trusted source-procedure, web, and
question handlers. It cannot grant access a handler does not already grant.
Source procedures come from the same artifact-trusted skill store. The delegate receives the full body even when the parent already recalled it; conversation-memory recall is not a source documentation tool.

Source selection remains an agent responsibility: acquisition success and a
source label are **not proof** that the agent selected the authorized source.
Explicit source restrictions take precedence over plans, prior pages, and
same-named records elsewhere.

The collector saves a fresh, bounded JSON snapshot under the reserved `.static/`
working directory. The artifact store excludes that whole directory from content
listing and publication. No acquisition code, credentials, private working
columns, candidate pages, or recipes should ship as separate artifact files.

Each dataset has a source label, optional URL, scope/caveats, and scalar records.
The host reads the actual file, rather than a model's transcription of console
output. Decimal JSON numbers retain their precision when parsed. Queries project
explicit columns, filter, group, aggregate, and sort; missing fields and wrong
numeric types are errors. Aggregates propagate nulls unless `missing="skip"`
explicitly requests observed values only. Count without a field counts rows;
count with a field counts non-null values. An empty sum is unavailable, not zero.

Facts select one exact query result or calculate over named facts. Division by
zero and missing inputs produce unavailable values. Text uses `{{fact_name}}`
and metrics bind to facts, so changing layout does not retype a total. The
compiler cannot prove that a written conclusion follows from the data, or that an
arbitrary literal in agent-authored HTML is correct; those require generation
and semantic evaluation.

For small datasets, the composer receives complete evidence. Above 80,000
characters it receives an explicitly labelled schema/row-count/sample catalog and
can call `inspect_static_data` for exact read-only queries. Oversize query results
are rejected without returning a truncated table. Render bindings still resolve
all records on the host.

## Composition

`compose_static_artifact(document=...)` accepts a versioned document. Its sections
contain prose, metrics, tables, charts, callouts, and custom HTML fragments. Flow,
split, and metric layouts recompose at narrow widths. The theme controls light or
dark mode, accent, typography, and density. Reports have an outline, inspectable
chart data, and print styles. Standard bar/line/scatter charts use container-sized
SVG, readable labels, units, focusable marks, and full data tables. Tables with
more than four columns recompose into labelled records on phones; every field
stays visible, while desktop retains the comparison table. Missing line
values remain gaps. Large value labels are formatted on the host, before browser
number conversion can lose integer precision.

Slides use the same evidence and blocks. Each section is exactly one slide; title and source sections are authored explicitly, and reading metadata adds no slides. Reading mode exposes the whole deck;
presentation mode shows one slide with controls and keyboard navigation. Long
content scrolls instead of being clipped or scaled into unreadable text. Print
styles include every slide even while presentation mode hides inactive slides.

`compose_custom_html(html=..., bindings=...)` accepts a complete portable HTML
page. Named queries are resolved and inserted as `window.ARTIFACT_DATA` before
any author script executes. Custom document fragments can read the page's
`window.ARTIFACT_DATA` bindings and use queries declared in `document.bindings`. This
route supports arbitrary HTML, SVG, canvas, and client-side interaction, under
the existing artifact sandbox and content security policy. It requires embedded data/media, inline styles, and no build, server, or remote fonts. Explicitly requested script libraries may use a pinned version from the preview-approved `cdn.jsdelivr.net` host and must pass the actual browser check; the standard compiler has no network dependency.

Only resolved public bindings and referenced fact values enter a compiled
page. Unselected fields and unused facts stay out. Query definitions can refer
to private fields to filter/sort without publishing those fields in the result.
The exported bindings include a canonical evidence digest and source metadata.
Custom bindings require declared render checks that tie selectors to complete row counts (or one aggregate view). The browser reports an error when those elements do not render, even if the rest of the page loads cleanly. These are minimum checks, not a proof of semantic correctness.

A binding is an explicit publication decision: every selected row/field is
public, including content in collapsed tables or inactive slides.

## Publication and editing

The engine stages attached assets and candidate pages under `.static/`, runs the
existing static/browser verifiers, and feeds concrete failures back to the
composer. It replaces the primary HTML atomically only after an accepted check,
under the existing cancellation lock. A failed build or abandoned generation
leaves the previous page intact. Unavailable browser checks are reported as
unverified, never as a pass.

The saved editable recipe is paired with the page's SHA-256. An externally edited
page or interrupted recipe/page update invalidates the pair, so a later edit
uses the actual page instead of trusting a stale recipe. Edits must preserve the
artifact's purpose, facts, and useful interactions except where the user's
current request changes them.

## Validation

The focused Python tests cover numeric precision, aggregation, nulls, projections,
script-safe JSON, data leakage, custom bindings, source failures, candidate
validation, cancellation, edit preservation, and unavailable browser checks.
The opt-in Chromium suite (`ANTON_TEST_BROWSER=1`) exercises seven exact viewport
sizes, chart/data inspection, long labels, negative/zero/missing values, slide
navigation, and print visibility. Set the standard `PLAYWRIGHT_BROWSERS_PATH`
when the surrounding test harness isolates HOME.

Fixture screenshots establish layout behavior only. Actual generation quality
must also be tested through Inference, including reports, decks, custom HTML,
updates, unavailable sources, and the original benchmark. Retain failed attempts,
source/configuration hashes, generated pages, timings, and grader results; do not
substitute hand-authored specimens for generated quality measurements.
