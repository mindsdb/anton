---
title: Release & versioning
description: Branch policy (dev → staging → main), promotion cadence, hotfix rules, the CalVer scheme, and the automated release workflow.
---

# Release & versioning

## Branches

Anton uses three long-lived branches: `dev` → `staging` → `main`.

```
feature/*  ──▶  dev  ──▶  staging  ──(soak ~1 day)──▶  main
                                                        ▲
                            hotfix/*  ──────────────────┘  (and back-merged to dev)
```

### Branch policy

- Anything you're working on that you feel is ready for production gets merged
  into `dev`. That's the integration line.
- **All non-hotfix PRs target `dev`.** Don't open feature PRs against
  `staging` or `main`.
- `staging` is for soak — never merge feature branches into it directly. It
  only receives the scheduled `dev → staging` promotion.
- `main` is the release line. The only things that land on `main` are the
  scheduled `staging → main` promotion and hotfixes.

### Hotfixes

- Production-only fixes target `main` directly.
- Every hotfix that lands on `main` **must** also be merged back into `dev` so
  the branches don't drift. If `staging` is mid-soak when the hotfix ships,
  bring it into `staging` too — otherwise the next promotion will overwrite it.
- Hotfix back-merges to `dev`/`staging` carry the fix only — there's no
  version file for them to touch.

### Promotion cadence

Twice a week, on a fixed schedule:

1. Merge `dev → staging`. Leave it ~1 day for soak tests. Each staging push
   publishes a release candidate (see below).
2. The day after the soak, merge `staging → main`. The release workflow tags
   and publishes from `main` automatically.

Net rhythm: two `dev → staging` promotions and two `staging → main` promotions
per week, each offset by a soak day.

## Versioning: calendar-derived

The version is **derived from the release git tag**, not written by hand:
`pyproject.toml` sets `[tool.hatch.version] source = "vcs"`, so hatch-vcs reads
it off the tag at build time and `anton/__init__.py` re-exports it at runtime
via `importlib.metadata`. The scheme:

```
<MAJOR>.<YY>.<MONTH>.<DAY>.<PATCH>
```

| Field | Meaning | When it bumps |
|---|---|---|
| `MAJOR` | Milestone or breaking-change signal | Only on an announced milestone (a launch, a major rewrite, a public "X.0" event) **or** a breaking change. Intentional and announced — never automatic |
| `YY` | Last two digits of the calendar year | Auto-bumps on the first release of each January |
| `MONTH` | Month of the release (1–12) | Each release. No zero-padding |
| `DAY` | Day of the release (1–31) | Each release. No zero-padding |
| `PATCH` | Hotfix counter for the specific dated release | `0` for scheduled releases; `1`, `2`, ... for hotfixes patching that release |

Rules:

- Nothing to write by hand — the release workflows mint the tag and hatch-vcs
  builds the wheel from it. PyPI may canonicalize a trailing `.0` away — that's
  fine.
- The version is set when the tag is cut on the `staging → main` promotion. The
  version *is* the actual ship date.

**Worked example:**

```
2026-04-30   2.26.4.30.0     ← cutover release
2026-07-15   3.26.7.15.0     ← announced milestone or breaking change → MAJOR bumps
2026-12-20   3.26.12.20.0
2027-01-05   3.27.1.5.0      ← YY auto-bumps; MAJOR stays
hotfix       3.26.7.15.1     ← patches the 3.26.7.15.0 release
```

**Cutover note.** Anton was on `2.0.4` under the old SemVer scheme. The first
CalVer release is `2.26.4.30.0` — keeping `MAJOR=2` (no milestone or break
warranted a bump) and letting `YY=26` carry the year. PEP 440 sees
`2.0.4 < 2.26.4.30.0`, so nothing rolls backward.

## The automated release flow

Anton publishes two streams: **stable** from `main` and **release candidates**
from `staging`.

How to ship a stable version:

1. Merge the scheduled `staging → main` promotion (reviewed as usual). No
   version file to touch — the tag carries the version.
2. That's it. On merge, `.github/workflows/release.yml` automatically:
   - computes today's CalVer version and creates the matching git tag,
   - builds the wheel (hatch-vcs derives the version from that tag) and
     publishes it to PyPI,
   - publishes a GitHub release with auto-generated notes,
   - triggers `tests_e2e_release.yml` to run live e2e tests against the
     released version.

### Staging release candidates

Every push to `staging` publishes a release candidate through
`.github/workflows/publish-staging.yml`: it cuts a PEP 440 pre-release tag
(`v2.YY.M.DD.SEQrcN`), publishes a GitHub pre-release, and uploads the wheel to
PyPI, so cowork-server staging installs an immutable, versioned Anton instead of
a mutable branch or hand-pinned commit (ENG-1159). These never reach production:
resolvers ignore pre-releases unless a specifier names one, and PyPI's
`info.version` (read by the prod desktop updater) excludes them.

### What you should NOT do

- **Don't create GitHub releases manually.** The `v*` tag namespace is locked
  via a repo ruleset — only the release workflow can create them. Manual
  attempts are rejected by GitHub.
- **Don't push `v*` tags directly.** Same protection.
- **Don't hand-edit a version.** There's no version file to bump — the tag is
  the source of truth, and both publishers derive the wheel version from it.

### Out-of-band releases

If you genuinely need to release outside the normal flow (an admin hotfix),
coordinate with `@mindsdb/devops` to bypass the tag ruleset. The e2e workflow's
version-match guard still verifies the release tag matches `anton.__version__`
and fails loudly on mismatch.

## CODEOWNERS

Everything under `.github/` is owned by `@mindsdb/devops` via
`.github/CODEOWNERS`. PRs touching workflows, actions, or release configuration
require their review before merge.

For the contribution workflow itself (forks, PRs, review), see
[Contributing](/developer/contributing). For how users receive updates, see
[Updating](/start/updating).
