```
        ▐
   ▄█▀██▀█▄   ♡♡♡♡
 ██  (°ᴗ°) ██
   ▀█▄██▄█▀      ▄▀█ █▄ █ ▀█▀ █▀█ █▄ █
    ▐   ▐        █▀█ █ ▀█  █  █▄█ █ ▀█
    ▐   ▐
```

# Meet Anton 

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mindsdb/anton)

Anton is a self-improving AI agent you can hand off any task to; Create and send reports, clear your inbox, send emails, manage your calendar, CRM,  book flights, etc. An open, powerful alternative to Claude-Cowork that you can run anywhere and use with any model you want — OpenAI, Anthropic, OpenRouter (200+ models), NVIDIA Nemotron, z.ai/GLM, Kimi/Moonshot, MiniMax, or your own endpoint.


## Quick Install
Anton can be installed as a desktop application or as a command-line tool.

### Desktop App:

- **macOS**: Click [here to download](https://downloads.mindsdb.com/anton/mac/anton-latest.pkg) the Anton Desktop App for MacOS.

- **Windows**: Click [here to download](https://downloads.mindsdb.com/anton/windows/anton-latest.exe) the Anton Desktop App for Windows.
 
### or - Command-Line App:

Open your terminal and use the following command to install

- **macOS/Linux**: 
```bash
curl -sSf https://raw.githubusercontent.com/mindsdb/anton/main/install.sh | sh && export PATH="$HOME/.local/bin:$PATH" 
```

- **Windows** (PowerShell):
```powershell
irm https://raw.githubusercontent.com/mindsdb/anton/main/install.ps1 | iex
```

That's it, you can now run it by simply typing the command.

```
anton
```

## What can Anton do?

### 🔧 Ask for anything that requires action

- **Send emails** - connect accounts, draft messages or even send them on your behalf.
- **Manage Calendarss** - Summarize your day, create meetings, block time, etc. All just by asking.
- **Automated reporting** - pull from multiple databases, crunch numbers, deliver a report on a schedule.
- **Workflow automation** - monitor a source, react to changes, take action.
- **Research & synthesis** - scrape the web, summarize findings, build a reference document.
- **Data pipeline prototyping** - connect sources, transform data, load into a destination.
- **System administration** - audit configurations, generate reports, fix issues.

The pattern is always the same: you describe the outcome, Anton figures out the steps. From one-off tasks to scheduled workflows — Anton handles it. Here are a few examples:

### 📊 Data analysis & Reports
```
I hold 50 AAPL, 200 NVDA, and 10 AMZN. Get today's prices, calculate my
total portfolio value, show me the 30-day performance of each stock, and
any other information that might be useful. Give me a complete dashboard.
```

What happens next is the interesting part. At first, Anton doesn't have any particular skill related to this question. However, it figures it out live: scrapes live prices, writes code on the fly, crunches the numbers, and builds you a full dashboard - all in one conversation, with no setup.


![ezgif-24b9e7c74652f0dc](https://github.com/user-attachments/assets/c92f87c1-ff30-4272-92ba-49a8585d5954)


### 📬 Email cleanup
```
Dear Anton, please help me clear unwanted emails...
```

Anton scans your inbox, classifies emails by signal vs. noise, identifies unsubscribable marketing, cold outreach, and internal tool notifications - then surfaces a breakdown and handles the cleanup. One user ran it on ~1,000 emails and found ~35% were un-subscribable. Anton surfaced everything AND handled the cleanup.

### 💬 Build its own integrations
```
Set up a WhatsApp integration so I can message you from my phone.
```

Anton doesn't wait for someone to build a connector. It writes the integration code itself, sets it up, and gets it running - so you can chat with it from WhatsApp, Telegram, or whatever channel you need.




---

## Key features
- **Credential vault** - prevents secrets from being exposed to LLMs.
- **Isolated code execution** - protected, reproducible "show your work" environment.
- **Multi-layer memory & continuous learning** - session, semantic and long-term knowledge. Anton remembers what it learned and gets better at your specific workflows over time.
- **Web search & fetch** - the agent can query the live web and retrieve URL contents. Routed natively through your LLM provider when possible (no extra setup), with a transparent fallback for third-party endpoints. See below.

---

## Web search & fetch

Anton exposes two web tools to the agent — `web_search` and `web_fetch` — both on by default. How they execute depends on your LLM provider:

| Provider | `web_search` | `web_fetch` | Setup |
| --- | --- | --- | --- |
| Anthropic BYOK | Anthropic native server tool | Anthropic native server tool | None — billed on your Anthropic key |
| OpenAI BYOK | OpenAI Responses API native | covered by `web_search` | None — billed on your OpenAI key |
| Minds-Enterprise-Cloud (mdb.ai) | mdb.ai passthrough | mdb.ai passthrough | None — billed on your Minds key |
| Generic OpenAI-compatible (Together, Groq, Ollama, vLLM, …) | Exa.ai or Brave (you choose at setup) | stdlib HTTP GET (no key) | Run `anton setup-search` once |

For the first three rows there's nothing to configure — the LLM provider executes the tools server-side and the results are folded directly into its response. For the fourth row, after `anton setup` finishes configuring a custom OpenAI-compatible endpoint Anton will offer to set up Exa or Brave; you can also (re)run that step at any time with `anton setup-search`. The chosen search-provider key is persisted to `~/.anton/.env` so it carries across sessions and workspaces, exactly like your LLM key.

To opt out, set `ANTON_WEB_SEARCH_ENABLED=false` and/or `ANTON_WEB_FETCH_ENABLED=false`.

Caveats: provider rate limits apply; `web_fetch` has a 30-second timeout and strips HTML to plain text (works best on article-style pages); paywalled and JS-heavy SPAs may return little useful content; treat fetched page bodies as untrusted input.

---

#### Connect your data and apps
Anton can connect an interact with files, databases, applications, APIs,... etc..

```powershell
/connect

(anton) What type of datasource (postgres, posthog, gmail, ..):

```

Tell Anton to connect and ask questions about your data. It will find credentials in the vault, fetch the schema, and retrieve what it needs.

```terminal
YOU> Connect to my Gmail and find emails from potential customers that haven’t been handled.

ANTON>
⎿ Connecting and fetching emails...
   ~3s
```

---

## What's inside

A big part of what makes Anton work is that it doesn’t need a huge collection of separate tools for web, DB, files etc. Most of the work is done through one core harness: The execution scratchpad, which can dynamically become whatever Anton needs for the task.

For the full architecture of Anton, and developer guide, see **[anton/README.md](anton/README.md)**.

---

## Workspace layout
When you run `anton` in a directory:

- `.anton/` - workspace folder containing scratchpad state, episodic memory, and local secrets.  
- `.anton/anton.md` - optional project context (Anton reads this at conversation start).  
- `.anton/.env` - workspace configuration variables file (local file). 
- `.anton/episodes/*` - episodic memories, one file per session.
- `.anton/memory/rules.md` - behavioral rules: Always/never/when rules (e.g., never hardcode credentials, how to build HTML)     
- `.anton/memory/lessons.md` - factual knowledge: Things I've learned (stock API quirks, dashboard patterns, data fetching notes)   
- `.anton/memory/topics/*` - topic-specific lessons:  Deeper notes organized by subject (dashboard-visualization, stock-data-api, etc.) 

Override the working folder:
```bash
anton --folder /path/to/workspace
```

---

### Windows scratchpad firewall
The Windows installer can add a firewall rule so the scratchpad can reach the internet. If you skipped it, run in an elevated PowerShell:

```powershell
netsh advfirewall firewall add rule name="Anton Scratchpad" dir=out action=allow program="$env:USERPROFILE\.anton\scratchpad-venv\Scripts\python.exe"
```

---

## How Anton differs from coding agents
Anton is a *doing* agent: code is a means, not the end. Where coding agents focus on producing code for a codebase, Anton focuses on delivering the outcome - a cleaned inbox, a live dashboard, a working integration, an automated workflow - and will write whatever code is necessary to achieve that goal.

---

## Is "Anton" a Mind?
Yes, at MindsDB we build AI systems that collaborate with people to accomplish tasks, inspired by the culture series books, so yes, Anton is a Mind :)

## Why the name "Anton"?
We really enjoyed the show *Silicon Valley*. Gilfoyle's AI - Son of Anton - was an autonomous system that wrote code, made its own decisions, and occasionally went rogue. We thought it was was great name for an AI that can learn on its own, so we kept Anton, dropped the "Son of".

---

## Analytics
Anton collects anonymous usage events (e.g. session started, first query) to help us understand how the product is used. No personal data or query content is sent.

To disable analytics, set the environment variable:

```bash
export ANTON_ANALYTICS_ENABLED=false
```

Or add it to your workspace config (`.anton/.env`):

```
ANTON_ANALYTICS_ENABLED=false
```

---

## Trace headers
When the planning provider is openai-compatible Anton can attach `Langfuse-Session-Id`, `Langfuse-Tags`, and `Langfuse-Metadata` headers so the router can attribute traces. To enable the same headers against any other openai-compatible endpoint (e.g. a self-hosted Langfuse proxy in front of ollama or vLLM), set:

```bash
export ANTON_LANGFUSE_HEADERS=1
```

Or add it to your workspace config (`.anton/.env`):

```
ANTON_LANGFUSE_HEADERS=1
```

---

## Dev guidelines

We use three long-lived branches: `dev` → `staging` → `main`.

```
feature/*  ──▶  dev  ──▶  staging  ──(soak ~1 day)──▶  main
                                                        ▲
                            hotfix/*  ──────────────────┘  (and back-merged to dev)
```

### Branch policy

- Anything you're working on that you feel is ready for production gets merged into `dev`. That's the integration line.
- **All non-hotfix PRs target `dev`.** Don't open feature PRs against `staging` or `main`.
- `staging` is for soak — never merge feature branches into it directly. It only receives the scheduled `dev → staging` promotion.
- `main` is the release line. The only things that land on `main` are the scheduled `staging → main` promotion and hotfixes.

### Hotfixes

- Production-only fixes target `main` directly.
- Every hotfix that lands on `main` **must** also be merged back into `dev` so the branches don't drift. If `staging` is mid-soak when the hotfix ships, bring it into `staging` too — otherwise the next promotion will overwrite it.

### Promotion cadence

Twice a week, on a fixed schedule:

1. Bump the version in `dev`, then merge `dev → staging`. Leave it for ~1 day for soak tests.
2. The day after the soak, merge `staging → main`. The release workflow tags and publishes from `main` automatically (see [Releasing](#releasing)).

Net rhythm: two `dev → staging` promotions and two `staging → main` promotions per week, each promotion offset by a soak day.

---

## Versioning

Anton versions follow a calendar-derived scheme:

```
<MAJOR>.<YY>.<MONTH>.<DAY>.<PATCH>
```

| Field | Meaning | When it bumps |
| --- | --- | --- |
| `MAJOR` | Milestone or breaking-change signal | Only when we hit an announced milestone (e.g. a launch, a major rewrite, a public "X.0" event) **or** ship a breaking change. Intentional and announced — never bumped automatically. |
| `YY` | Last two digits of the calendar year | Auto-bumps on the first release of each January. |
| `MONTH` | Month of the release (1–12) | Each release. No zero-padding. |
| `DAY` | Day of the release (1–31) | Each release. No zero-padding. |
| `PATCH` | Hotfix counter for the specific dated release | `0` for scheduled releases. `1`, `2`, … for hotfixes patching that release. |

**Rules**

- Always write all 5 components in [`anton/__init__.py`](anton/__init__.py) (`__version__ = "2.26.4.30.0"`). PyPI may canonicalize a trailing `.0` away — that's fine.
- The version bump happens on the `staging → main` promotion (see [Dev guidelines](#dev-guidelines)). The version *is* the actual ship date.
- Hotfix back-merges to `dev`/`staging` carry the fix only — never the `__version__` bump.

**Worked example**

```
2026-04-30   2.26.4.30.0     ← cutover release
2026-07-15   3.26.7.15.0     ← announced milestone or breaking change → MAJOR bumps
2026-12-20   3.26.12.20.0
2027-01-05   3.27.1.5.0      ← YY auto-bumps; MAJOR stays
hotfix       3.26.7.15.1     ← patches the 3.26.7.15.0 release
```

**Cutover note.** Anton was on `2.0.4` under the old SemVer scheme. The first CalVer release is `2.26.4.30.0` — keeping `MAJOR=2` (no announced milestone or break warrants a bump) and letting `YY=26` carry the year. PEP 440 sees `2.0.4 < 2.26.4.30.0` so nothing rolls backward.

---

## Releasing

Anton uses an automated release flow. The single source of truth for the package version is [`anton/__init__.py`](anton/__init__.py) (`__version__`); the format is documented in [Versioning](#versioning).

### How to ship a new version

1. On the scheduled `staging → main` promotion, bump `__version__` in [`anton/__init__.py`](anton/__init__.py) to today's date (see [Versioning](#versioning) for the format).
2. Get it reviewed and merge to `main`.
3. That's it. On merge, [`.github/workflows/release.yml`](.github/workflows/release.yml) automatically:
   - Creates the matching git tag (`v2.0.5`).
   - Publishes a GitHub release with auto-generated notes.
   - Triggers [`tests_e2e_release.yml`](.github/workflows/tests_e2e_release.yml) to run live e2e tests against the released version.

### What you should NOT do

- **Don't create GitHub releases manually.** The `v*` tag namespace is locked via a repo ruleset — only the release workflow can create them. Manual attempts will be rejected by GitHub.
- **Don't push `v*` tags directly.** Same protection applies.
- **Don't edit `__version__` outside a dedicated bump PR.** Keep version bumps small and reviewable so the auto-release diff is easy to audit.

### Editing CI / workflows

Anything under [`.github/`](.github/) is owned by `@mindsdb/devops` via [CODEOWNERS](.github/CODEOWNERS). PRs touching workflows, actions, or release configuration require their review before merge.

### Hotfixes / out-of-band releases

If you genuinely need to release outside the normal flow (e.g. an admin hotfix), coordinate with `@mindsdb/devops` to bypass the tag ruleset. The e2e workflow's version-match guard will still verify the release tag matches `anton.__version__` and fail loudly on mismatch.

---

## License
AGPL-3.0 license
