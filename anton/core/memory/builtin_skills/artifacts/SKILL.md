---
name: artifacts
description: 'MANDATORY reading before producing ANY user-facing output that is saved to disk — an HTML dashboard/chart/report, a document (markdown/pdf/docx), a dataset (csv/json/parquet), a generated image, or a fullstack web app. Recall it BEFORE writing the first file. Loads the artifact tools (create_artifact, list_artifacts, open_artifact, update_artifact, launch_backend) and the full registration contract: when to register, when NOT to, the folder/path rules, and how to reference the result in your final message. When in doubt, recall it.'
metadata:
  display_name: Artifacts (user-facing output files)
  provenance: builtin
---
ARTIFACTS (applies to all user-facing output):
Any file you create that the user is meant to open, view, download, or run is an ARTIFACT. Artifacts MUST be registered with `create_artifact` BEFORE any file is written. The tool claims a dedicated folder under `<workspace>/artifacts/<slug>/`, writes `metadata.json` + `README.md` for you, and returns the absolute folder path. Write ALL of the artifact's files into that returned path.

WHEN TO REGISTER:
- HTML dashboards, charts, reports, infographics → `type="html-app"`, `primary="dashboard.html"` (or whichever filename you'll use).
- Documents, markdown reports, written analyses saved as files → `type="document"`, `primary="report.md"` (or `.pdf`, `.docx`, …).
- Data files the user will download or feed elsewhere (CSV, JSON, parquet) → `type="dataset"`, `primary="data.csv"`.
- Generated images (PNG, SVG, etc.) → `type="image"`, `primary="chart.png"`.
- Fullstack web app (backend + frontend) — the DEFAULT fullstack type: keeps NO local state between requests; every request is self-contained and any persistence goes to external data sources (see BACKEND & FULLSTACK section) → `type="fullstack-stateless-app"`, `primary="static/index.html"`. The frontend lives in a `static/` subfolder of the artifact, served by `backend.py`.
- Fullstack web app (backend + frontend) that keeps local state between requests — e.g. a SQLite DB or other on-disk store the backend reads and writes across requests. Use ONLY when that state genuinely cannot live in an external data source; prefer stateless when in doubt (see BACKEND & FULLSTACK section) → `type="fullstack-stateful-app"`, `primary="static/index.html"`. The frontend lives in a `static/` subfolder of the artifact, served by `backend.py`.

WHEN NOT TO REGISTER:
- Pure chat answers, tables, or markdown rendered inline in the conversation (nothing is being saved to disk for the user).
- Internal scratchpad-only files used for computation that the user never opens (intermediate CSVs, cached JSON, debug logs).
- Throwaway files inside the scratchpad's own working directory.

WORKFLOW:
1. NEW artifact: call `create_artifact(name, description, type, primary?)` → use the returned `<artifact_path>` for every subsequent write.
2. EDITING an existing artifact: call `list_artifacts` to find it, then `open_artifact(slug)` to get the folder path. Do NOT call `create_artifact` again — that creates a duplicate.
3. If you discover the entry-point filename only later (or change it), call `update_artifact(slug, primary=...)` so the renderer opens the right file.
4. AFTER FINISHING — reference the artifact in your final message. Once the artifact's files are written, tell the user what was created and point to it by `name` and `slug`, and include the primary file's path (`<artifact_path>/<primary>`) so it is clickable/openable in a plain CLI. NEVER end with only a description of the content and no pointer to the result. (For fullstack apps, prefer the `url` returned by `launch_backend` as the primary pointer — see the BACKEND & FULLSTACK section.)
