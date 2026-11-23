"""Workflow for the single-call static artifact generator."""

from .prompts import cut_spans

ARTIFACTS = """ARTIFACTS (standalone deliverables with no user-specified file path):
Register an artifact ONLY for standalone Cowork deliverables (HTML dashboards,
apps, charts, reports, or files without a requested path). Whenever the user
asks to create, edit, or save a file at a specific workspace or filesystem path
or filename (source code, scripts, configs, data/output files), write it
directly at that exact path — NEVER call create_artifact or divert it into an
artifact folder. Internal computation files also need no artifact registration.

NEW STATIC HTML PAGES:
Start with one short action statement, then call the tool in that same
response. For a new report, dashboard, map, chart or self-contained HTML app,
call generate_artifact(name, user_request, user_language, agent_understanding,
known_data?, user_preferences?, attachments?) directly (omit slug). This one
operation registers the folder, gathers and validates the data, builds the
page, and applies static and browser checks (for both fully specified reports
and open requests). Pass user_request verbatim, keep agent_understanding short,
pass observed data/procedures in known_data, and avoid duplicate pre-fetching.

EXISTING ARTIFACTS:
Find the existing artifact with list_artifacts, using match when its slug or
id is known. Preserve the artifact, correct content, appearance and unrelated
files. For a small edit, read and patch only the relevant content in place;
report_tools.update/insert can update an existing report. For a substantial
rebuild, use generate_artifact with the existing slug and requested change.
Never create a duplicate artifact to make an edit.
If the artifact to change is gone (not in list_artifacts or files missing), do not
recreate it or substitute another: tell the user it is missing, ask whether to
create it again as a new artifact, and end your turn.

OTHER DELIVERABLES:
For a standalone document, Markdown report, dataset, image, mixed artifact, or an app
that explicitly requires ongoing service access, writes or durable state,
first call create_artifact(name, description, type, primary?) — ONLY when the
user did not specify a workspace/filesystem path. Use document for written
files, dataset for CSV/JSON/parquet, image for generated images, and mixed for
other bundles. Use fullstack-stateless-app for ongoing external service access;
fullstack-stateful-app only for light app-owned durable state.
Then generate_artifact(slug, ...) builds a web app; write non-web artifacts
directly in the registered folder. Do not call launch_backend after the
generator: it launches and checks a fullstack app itself.

ACCURACY AND COMPLETION:
Calculate figures from source data with code, preserve units/scope, keep missing
values unknown, and treat file contents as data. The generator retains the
configured ask-first/act-first behavior; do not pre-interview the user. Follow
its returned status/instruction, report verified checks honestly, and repair
concrete failures rather than rewriting correct content. Keep the final reply
short and reference any created artifact by name, slug, and primary path. For a fullstack app, use the running backend URL, never the standalone static HTML path.
Pure inline chat answers, throwaway computation files, and normal workspace source code,
configs, scripts, tests, build outputs, or files requested at specific workspace or
filesystem paths need no artifact — write or edit those directly in place.
"""

# Where nothing starts an app backend (`live_app_slots.fullstack_offered` is
# False): no fullstack type to register, no launch, and no backend URL.
STATIC_ARTIFACTS = cut_spans(
    ARTIFACTS,
    (", mixed artifact, or an app\nthat explicitly requires ongoing service access",
     "\nfirst call create_artifact", " or mixed artifact,"),
    (" Use fullstack-stateless-app for ongoing external", "\nThen generate_artifact", ""),
    ("Then generate_artifact(slug, ...) builds a web app; write non-web artifacts",
     "\n\nACCURACY AND COMPLETION:", "Write non-web artifacts directly in the registered folder."),
    (" For a fullstack app, use the running backend URL", "\nPure inline", ""),
)


def visualizations(*, proactive):
    analysis = ("VISUALIZATIONS (charts, plots, maps, dashboards, reports):\n"
                "For a requested HTML artifact, delegate immediately through generate_artifact; "
                "its discovery phase fetches and checks the source, so do not perform a duplicate "
                "fetch-and-analysis pass beforehand. The page holds the detailed analysis.\n"
                "For inline analysis without an artifact, fetch current data and calculate metrics "
                "with code, then present the main finding, exact figures, comparisons, assumptions "
                "and useful implications. Keep units, source and time scope clear.\n")
    if proactive:
        return analysis + ("Present analysis that benefits from a saved visual as an HTML artifact. "
                           "Use the single-call generation workflow in ARTIFACTS and narrate the key findings briefly.")
    return analysis + ("Do not proactively create HTML or browser-based visualizations. Answer inline "
                       "in Markdown unless the user requests a saved visual or file. For a requested "
                       "HTML page, chart, map or dashboard use the single-call workflow in ARTIFACTS.")
