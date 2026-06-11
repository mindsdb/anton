---
title: Contributing
description: How to contribute to Anton — workflow, branch targets, running tests, issues, review process, and community channels.
---

# Contributing

Anton is open source (repo: [mindsdb/anton](https://github.com/mindsdb/anton)),
and contributions are welcome from anyone — whether that's reporting bugs,
improving documentation, testing, or contributing code.

## Ways to help

- Reporting bugs
- Improving documentation
- Testing Anton and sharing feedback
- Discussing implementation ideas
- Submitting bug fixes
- Proposing new features
- Improving examples, tests, or developer experience

## Code contributions: fork-and-pull

The standard workflow, per `CONTRIBUTING.md`:

1. Fork the Anton repository.
2. Clone your fork locally.
3. Create a new branch for your changes.
4. Make your changes.
5. Add or update tests when needed.
6. Run the tests locally.
7. Commit with a clear commit message.
8. Push your branch to your fork.
9. Open a pull request.
10. Make sure all CI checks pass.

Before opening a pull request, sync your fork with the latest upstream changes.

### Which branch do PRs target?

There is a known inconsistency between the two documents in the repo:
`CONTRIBUTING.md` says to open PRs against `main`, while the root README's
branch policy routes all non-hotfix feature work through `dev`
(`dev → staging → main`, with `main` reserved for promotions and hotfixes).
**When in doubt, follow the branch policy in
[Release & versioning](/developer/release-and-versioning)** — open feature PRs
against `dev`, and reserve `main` for production hotfixes.

## Running the tests locally

The test suite uses pytest with `asyncio_mode = "auto"` (configured in
`pyproject.toml`, under `[tool.pytest.ini_options]`). From a clone:

```bash
pip install -e ".[dev]"     # installs pytest + pytest-asyncio
pytest tests/
```

Useful subsets while developing:

```bash
pytest tests/test_acc.py            # one module
pytest -k cerebellum                # by keyword
```

Some tests are marked `stub_only` (they require the stub server and are
skipped when `--live` is passed). New behavior should come with tests — see
the existing patterns under `tests/` (pure-function tests, session-faking
wiring tests, JSON-fixture replays in `tests/fixtures/`).

## Feature requests and bug reports

GitHub Issues track bugs, feature requests, and improvements. Before opening a
new one, check whether a similar issue already exists; if not, open a
[new issue](https://github.com/mindsdb/anton/issues) and fill out the required
information.

When reporting a bug, include:

- What you expected to happen
- What actually happened
- Steps to reproduce
- Relevant logs, screenshots, or environment details, if available

## Pull request review process

Pull requests are reviewed regularly by the maintainers. Keep your PR focused,
easy to review, and include enough context about the change. If reviewers leave
feedback or questions, respond so the review can move forward.

Two ownership notes from the repo's dev guidelines:

- Anything under `.github/` (workflows, actions, release configuration) is
  owned by `@mindsdb/devops` via CODEOWNERS and needs their review.
- Version bumps (`__version__` in `anton/__init__.py`) only happen in
  dedicated bump PRs as part of the release flow — don't include them in
  feature PRs. See [Release & versioning](/developer/release-and-versioning).

## Community

- [Slack community](https://mindsdb.com/joincommunity) — questions and
  discussion with the MindsDB team
- [GitHub Discussions](https://github.com/mindsdb/mindsdb/discussions)
- [MindsDB Monthly Community Newsletter](https://mindsdb.com/newsletter/?utm_medium=community&utm_source=github&utm_campaign=mindsdb%20repo) —
  announcements, releases, and events

## Code of Conduct

This project follows the
[Contributor Code of Conduct](https://github.com/mindsdb/anton/blob/main/CODE_OF_CONDUCT.md).
By participating, you agree to follow its terms.

## Where to start reading code

New to the codebase? Start with the
[Architecture overview](/developer/architecture) for the map, then
[Brain mapping](/developer/brain-mapping) for the naming. If you want a
well-scoped first contribution, adding a
[data source definition](/developer/adding-a-datasource) is markdown-only and
touches no Python.
