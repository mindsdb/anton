---
title: Updating Anton
description: How Anton's versioning works, how auto-update behaves, and how to disable or pin versions.
---

# Updating Anton

Anton keeps itself up to date automatically. This page explains the version
scheme, what the auto-updater does at startup, and how to opt out or pin a
specific release.

## Version scheme

Anton versions are calendar-derived:

```
MAJOR.YY.MONTH.DAY.PATCH
```

| Field | Meaning |
| --- | --- |
| `MAJOR` | Milestone or breaking-change signal — bumped intentionally, never automatically |
| `YY` | Last two digits of the release year |
| `MONTH` | Month of the release (1–12, no zero-padding) |
| `DAY` | Day of the release (1–31, no zero-padding) |
| `PATCH` | `0` for scheduled releases; `1`, `2`, … for hotfixes of that release |

In short: the version *is* the ship date. A worked example:

```
2026-04-30   2.26.4.30.0     ← released April 30, 2026
2026-07-15   3.26.7.15.0     ← MAJOR bumped for an announced milestone
2027-01-05   3.27.1.5.0      ← YY rolls over in January; MAJOR stays
hotfix       3.26.7.15.1     ← patches the 3.26.7.15.0 release
```

## Checking your version

```bash
anton version
```

## How auto-update works

Every time you start `anton`, it:

1. Checks the latest GitHub release of `mindsdb/anton`.
2. If a newer version exists, reinstalls Anton from that release tag using `uv`.
3. Verifies the installed version matches the release tag, then restarts itself so you're immediately on the new version.

The whole check runs in a background thread with a hard 10-second ceiling —
it never blocks startup longer than that, even on a flaky network. If the
check times out or fails, Anton simply continues with the current version.

## Disabling auto-update

Add this to your config (`~/.anton/.env` for all workspaces, or a project's
`.anton/.env`):

```
ANTON_DISABLE_AUTOUPDATES=true
```

## Pinning a specific version

To stay on an exact release:

1. Disable auto-updates as above.
2. Install the release you want by tag:

```bash
uv tool install "git+https://github.com/mindsdb/anton.git@v2.26.4.30.0" --force
```

Replace the tag with the release you want from the
[releases page](https://github.com/mindsdb/anton/releases). To go back to
the latest, remove `ANTON_DISABLE_AUTOUPDATES` and restart `anton`, or run
`uv tool upgrade anton`.
