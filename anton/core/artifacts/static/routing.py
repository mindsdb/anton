"""One authoring route for static web artifacts, including subsequent edits."""
from dataclasses import replace
import os


def enabled(act_first: bool) -> bool:
    return act_first and os.environ.get('ANTON_STATIC_ARTIFACTS') == 'v2'


WEB_WORKFLOW = '''STATIC WEB ARTIFACTS:
Reports from data, charts, write-ups, slides, dashboards, demos, and custom HTML
use type="html-app", primary="index.html". Filters, drill-downs, animations,
sorting, and client-side calculations work with embedded data and need no backend.
Reading a connected source while building a report also needs no backend.
Register with create_artifact, then call generate_artifact with the original
user_request, user_language, and agent_understanding. Pass exact known_data,
source procedures, requested style, and actual attached file paths when available.
When using an existing computation, pass its complete records or export the complete
records to a private file and pass that path. The builder uses a separate Python
scratchpad and cannot access variables left in the parent notebook.
The engine gathers evidence, composes the artifact, and checks the actual page.
It can acquire source records during the build even when an earlier parent read
failed. Its result's data_publication lists the evidence origin and dataset row
counts. Use that current result; an earlier absence of parent records does not
establish that the builder also lacked them. Cached origin still means cached.
Use it for fully specified work as well as open-ended requests. Do not write
the page in scratchpad or use the older report helpers or dashboard skill.
For edits, find the existing slug with list_artifacts and call generate_artifact
on that same slug with the current change request. The engine reads the saved
page and composition and preserves useful content and interactions. Do not
register a duplicate or patch the HTML through a competing authoring path.
Use the original user constraints over an earlier plan. Follow the result's
status and instruction; report a failure accurately and never claim an
unavailable check passed. A failed candidate preserves the existing page.
Point to the named artifact and its primary file in the final reply.
'''


def artifact_prompt(fullstack: bool) -> str:
    text = '''ARTIFACTS:
Register every user-facing output with create_artifact before writing it.
The returned folder belongs to that artifact; keep its files there.
Use document for PDF/Word/Markdown, dataset for downloadable data, image for
images, and mixed for other file sets. Write those non-web files into the
registered folder. Inline chat answers and private computation files need no
registration. Keep credentials and private working files out of deliverables.
'''+WEB_WORKFLOW
    if fullstack:
        text += '''
Use fullstack-stateless-app only for an application that actually requires
server-side execution at viewing time, such as live refresh from a protected
source, authenticated actions, or writes to an external service. A static
comparison of connected data is an html-app. Use fullstack-stateful-app only
when that server also owns durable application state. Fullstack apps use
primary="static/index.html" and the existing generate_artifact backend flow;
their result URL is the entry point. Do not change an existing backend app's
type merely to restyle it.
'''
    return text


def visualization_prompt(proactive: bool) -> str:
    return ('Create a visual artifact when it helps explain the analysis. ' if proactive else
            'Keep ordinary answers inline; create a visual artifact when the user requests one. ') + (
            'Use the static web artifact workflow for HTML reports, charts, slides and dashboards. '
            'State only findings supported by the actual data, retain uncertainty, and keep the chat reply concise.')


def authoring_tools(create, generate, *, fullstack: bool):
    """Keep the host's schemas/handlers and remove conflicting legacy prose."""
    create_description = (
        'Register a user-facing artifact and claim its dedicated output folder. '
        'Use html-app with primary index.html for static reports, slides, charts, '
        'dashboards and custom HTML, including filters and drill-downs over embedded data. '
        'Querying connected data at build time does not require a backend. '
        'Then call generate_artifact. For document/dataset/image/mixed, write the '
        'non-web files yourself in the returned folder. To edit an existing artifact, '
        'use list_artifacts and its existing slug instead of creating a duplicate.'
    )
    if fullstack:
        create_description += (' Use fullstack-stateless-app for required server-side work at viewing time '
            '(live protected-source refresh, authenticated actions, external writes); '
            'use fullstack-stateful-app when that server also owns durable application state. '
            'Fullstack primary is static/index.html.')
    generate_description = (
        'Create or edit an already-registered web artifact. For html-app, the static '
        'engine gathers exact evidence and composes a responsive report, slides, or '
        'custom HTML; it checks the candidate before publication. Use this for every '
        'HTML artifact and edit, including fully specified reports. Pass slug, the '
        'current original user_request, user_language, agent_understanding, and '
        'optional exact known_data, user_preferences and user-provided attachments. '
        'Do not write the page yourself. Follow the returned status and instruction. '
        'Report checks_skipped and warnings; do not claim a failed build succeeded.'
    )
    if fullstack:
        generate_description += ' Fullstack types retain their existing backend generation and launch flow.'
    return replace(create, description=create_description), replace(generate, description=generate_description, prompt=WEB_WORKFLOW)
