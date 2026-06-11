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
- Hotfix back-merges to `dev`/`staging` carry the fix only — never the
  `__version__` bump.

### Promotion cadence

Twice a week, on a fixed schedule:

1. Bump the version in `dev`, then merge `dev → staging`. Leave it ~1 day for
   soak tests.
2. The day after the soak, merge `staging → main`. The release workflow tags
   and publishes from `main` automatically.

Net rhythm: two `dev → staging` promotions and two `staging → main` promotions
per week, each offset by a soak day.

## Versioning: calendar-derived

The single source of truth is `__version__` in `anton/__init__.py`
(hatch reads it via regex; see `pyproject.toml`). The scheme:

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

- Always write all 5 components (`__version__ = "2.26.4.30.0"`). PyPI may
  canonicalize a trailing `.0` away — that's fine.
- The bump happens on the `staging → main` promotion. The version *is* the
  actual ship date.

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

How to ship a new version:

1. On the scheduled `staging → main` promotion, bump `__version__` in
   `anton/__init__.py` to today's date.
2. Get it reviewed and merge to `main`.
3. That's it. On merge, `.github/workflows/release.yml` automatically:
   - creates the matching git tag,
   - publishes a GitHub release with auto-generated notes,
   - triggers `tests_e2e_release.yml` to run live e2e tests against the
     released version.

### What you should NOT do

- **Don't create GitHub releases manually.** The `v*` tag namespace is locked
  via a repo ruleset — only the release workflow can create them. Manual
  attempts are rejected by GitHub.
- **Don't push `v*` tags directly.** Same protection.
- **Don't edit `__version__` outside a dedicated bump PR.** Keep version bumps
  small and reviewable so the auto-release diff is easy to audit.

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
