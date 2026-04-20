```
        ▐
   ▄█▀██▀█▄   ♡♡♡♡
 ██  (°ᴗ°) ██
   ▀█▄██▄█▀      ▄▀█ █▄ █ ▀█▀ █▀█ █▄ █
    ▐   ▐        █▀█ █ ▀█  █  █▄█ █ ▀█
    ▐   ▐
```

# Meet Anton - an agent that gets real work done

Anton is a personal AI agent that helps you get actual-work done. Tell it what you need in plain language and it takes it from there - sending emails, calling APIs, connecting to data sources, building dashboards, and delivering results. No crazy setups, no plugins, no fuss.


![ezgif-24b9e7c74652f0dc](https://github.com/user-attachments/assets/c92f87c1-ff30-4272-92ba-49a8585d5954)

## Quick start
**macOS - Desktop App:**

<a href="https://mindsdb-anton.s3.us-east-2.amazonaws.com/mac/anton-latest.pkg">
<img width="64" alt="DesktopApp" src="https://github.com/user-attachments/assets/ed7c1e3a-3700-45cc-a9a8-efb57b43dcfd" />
</a>

 Click [here to download](https://mindsdb-anton.s3.us-east-2.amazonaws.com/mac/anton-latest.pkg) the Anton Desktop App for MacOS.


**macOS / Linux - CLI:**
```bash
curl -sSf https://raw.githubusercontent.com/mindsdb/anton/main/install.sh | sh && export PATH="$HOME/.local/bin:$PATH" 
```

**Windows CLI** (PowerShell):
```powershell
irm https://raw.githubusercontent.com/mindsdb/anton/main/install.ps1 | iex
```

That's it, you can now run it by simply typing the command:
```
anton
```

## What can Anton do?

Help you with work, Anton starts as a blank canvas that molds to your needs, you simply ask and anton figures things out live. It doesn't rely on pre-built plugins or predefined workflows - it writes code on the fly, calls APIs, and chains together whatever steps are needed to get the job done.

Here are a few examples of what people are using it for:

### 📊 Data analysis & dashboards
```
I hold 50 AAPL, 200 NVDA, and 10 AMZN. Get today's prices, calculate my
total portfolio value, show me the 30-day performance of each stock, and
any other information that might be useful. Give me a complete dashboard.
```

What happens next is the interesting part. At first, Anton doesn't have any particular skill related to this question. However, it figures it out live: scrapes live prices, writes code on the fly, crunches the numbers, and builds you a full dashboard - all in one conversation, with no setup.


```text
anton> Dashboard is open in your browser.
Summary: Concentration risk is your #1 issue. If you're comfortable being a high-conviction NVDA...
```

<p align="center"> 
        <img width="800" alt="Anton's response" src="https://github.com/user-attachments/assets/6dc6ee81-2a2c-4358-be05-bfe884c32685" />
</p>

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


### 🔧 Ask for anything that requires action

- **Send emails** - connect accounts, draft messages or even send them on your behalf.
- **Manage Calendarss** - Summarize your day, create meetings, block time, etc. All just by asking.
- **Automated reporting** - pull from multiple databases, crunch numbers, deliver a report on a schedule.
- **Workflow automation** - monitor a source, react to changes, take action.
- **Research & synthesis** - scrape the web, summarize findings, build a reference document.
- **Data pipeline prototyping** - connect sources, transform data, load into a destination.
- **System administration** - audit configurations, generate reports, fix issues.

The pattern is always the same: you describe the outcome, Anton figures out the steps.

---

## Key features
- **Credential vault** - prevents secrets from being exposed to LLMs.
- **Isolated code execution** - protected, reproducible "show your work" environment.
- **Multi-layer memory & continuous learning** - session, semantic and long-term knowledge. Anton remembers what it learned and gets better at your specific workflows over time.

---

#### Connect your data
Although you can use Anton with just public data, the real power happens when you combine that with your own data. This can be anything: files, databases, application APIs,... etc. Open the Local Vault with `/connect` command, then follow the prompts to add your secrets. Anton only has access to secret names - secret values remain hidden.

```powershell
/connect

(anton) What type of datasource (postgres, posthog, gmail, ..):

```

Tell Anton to connect and ask questions about your data. It will look for credentials in the vault (by their name), fetch the schema, and retrieve the necessary data. 
```test
YOU> Connect to STAPLECACHE company data. Check if there is a correlation between the discount given 
and the review rating in the last 6 months?

ANTON>
⎿ Scratchpad (connecting and fetching schema…) 
   ~3s
```

---

You can always ask Anton to explain what it did. Ask it to dump its scratchpad and you get a full notebook-style breakdown: every cell of code it ran, the outputs, and errors - so you can follow its reasoning step by step.

---

## What's inside

A big part of what makes Anton work is that it doesn’t need a huge collection of separate tools for web, DB, files etc. Most of the work is done through one core harness: The scratchpad, which can dynamically become whatever Anton needs for the task.

For the full architecture of Anton, file formats, and developer guide, see **[anton/README.md](anton/README.md)**.

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

## Memory systems
Anton provides two human-readable memory systems:

- **Semantic memory** - rules, lessons, identity and domain expertise stored as markdown at global and project scope.  
- **Episodic memory** - a timestamped archive of every conversation (JSONL in `.anton/episodes/`). Anton can recall prior sessions with the `recall` tool.

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

## License
AGPL-3.0 license
