```
        ▐
   ▄█▀██▀█▄   ♡♡♡♡
 ██  (°ᴗ°) ██
   ▀█▄██▄█▀      ▄▀█ █▄ █ ▀█▀ █▀█ █▄ █
    ▐   ▐        █▀█ █ ▀█  █  █▄█ █ ▀█
    ▐   ▐
```

# MindsDB Anton — What Business Intelligence is supposed to be

Business intelligence was supposed to give you the right data, at the right time, to get real work done.

That is Anton. You ask questions in plain language, and Anton takes ownership of the entire analytical process:
it pulls and unifies data from multiple sources, runs the analysis, surfaces insights, builds rich dashboards, suggests next steps, and can even take action - A business intelligence agent that works like an expert analyst — 24/7, at machine speed.

![ezgif-24b9e7c74652f0dc](https://github.com/user-attachments/assets/c92f87c1-ff30-4272-92ba-49a8585d5954)


## Quick start 
**macOS - Desktop App:**

<a href="https://mindsdb-anton.s3.us-east-2.amazonaws.com/anton-latest-universal-signed.pkg">
<img width="64" alt="DesktopApp" src="https://github.com/user-attachments/assets/ed7c1e3a-3700-45cc-a9a8-efb57b43dcfd" />
</a>

 Click [here to download](https://mindsdb-anton.s3.us-east-2.amazonaws.com/anton-latest-universal-signed.pkg) the Anton Desktop App for MacOS.


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

## Using Anton

Talk to Anton like a person, for example, ask Anton this:

```
I hold 50 AAPL, 200 NVDA, and 10 AMZN. Get today's prices, calculate my
total portfolio value, show me the 30-day performance of each stock, and
any other information that might be useful. Give me a complete dashboard.
```

What happens next is the interesting part. At first, Anton doesn't have any particular skill related to this question. However, it figures it out live: scrapes live prices, writes code on the fly, crunches the numbers, and builds you a full dashboard — all in one conversation, with no setup.


```text
anton> Dashboard is open in your browser.
Summary: Concentration risk is your #1 issue. If you're comfortable being a high-conviction NVDA...
```

<p align="center"> 
        <img width="800" alt="Anton's response" src="https://github.com/user-attachments/assets/6dc6ee81-2a2c-4358-be05-bfe884c32685" />
</p>

**Key features**
- **Credential vault** - prevents secrets from being exposed to LLMs.
- **Isolated code execution** - protected, reproducible “show your work” environment.
- **Multi-layer memory & continuous learning** - session, semantic and long-term business knowledge.

#### Connect your data
Although you can use Anton with just public data, the real power happens when you combine that with your own data. This can be anything: files,  databases, application APIs,... etc. Open the Local Vault with `/connect` command, then follow the prompts to add your secrets. Anton only has access to secret names - secret values remain hidden.

```powershell
/connect

(anton) Choose a data source:

         Primary
           0. Custom datasource (connect anything via API, SQL, or MCP)

         Most popular
           1. Amazon Redshift
           2. Databricks
           3. Google BigQuery
           4. HubSpot
           5. MariaDB
           6. Microsoft SQL Server
           7. MySQL
           8. Oracle Database
           9. PostgreSQL
          10. Salesforce
          11. Shopify
          12. Snowflake

(anton) Enter a number or type a name:

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

### Explainable by default

You can always ask Anton to explain what it did. Ask it to dump its scratchpad and you get a full notebook-style breakdown: every cell of code it ran, the outputs, and errors — so you can follow its reasoning step by step.

---

## What's inside

<p align="center"><img width="800"  alt="image" src="/assets/anton-diagram.png" /></p>

For the full architecture of Anton, file formats, and developer guide, see **[anton/README.md](anton/README.md)**.

---

## Workspace layout

When you run `anton` in a directory:

- `.anton/` — workspace folder containing scratchpad state, episodic memory, and local secrets.  
- `.anton/anton.md` — optional project context (Anton reads this at conversation start).  
- `.anton/.env` — workspace configuration variables file (local file). 
- `.anton/episodes/*` — episodic memories, one file per session.
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

- **Semantic memory** — rules, lessons, identity and domain expertise stored as markdown at global and project scope.  
- **Episodic memory** — a timestamped archive of every conversation (JSONL in `.anton/episodes/`). Anton can recall prior sessions with the `recall` tool.

Configure memory via `/setup` > Memory or via environment variables.

---

### Prerequisites

- `git` — required  
- Python **3.11+** (Anton will bootstrap an environment if missing)  
- `curl` — macOS / Linux installs  
- Internet connection (scratchpad may access web sources)

### Windows scratchpad firewall

The Windows installer can add a firewall rule so the scratchpad can reach the internet. If you skipped it, run in an elevated PowerShell:

```powershell
netsh advfirewall firewall add rule name="Anton Scratchpad" dir=out action=allow program="$env:USERPROFILE\.anton\scratchpad-venv\Scripts\python.exe"
```

---

## How Anton differs from coding agents

Anton is a *doing* agent: code is a tool to get results. Where coding agents focus on producing code for a codebase, Anton focuses on delivering the outcome — a dataset, report, dashboard, or automated workflow — and will write whatever code is necessary to achieve that goal.

---

## Is "Anton" a Mind?

Yes, at MindsDB we build AI systems that collaborate with people to accomplish tasks, inspired by the culture series books, so yes, Anton is a Mind :)

## Why the name "Anton"?

We really enjoyed the show *Silicon Valley*. Gilfoyle's AI — Son of Anton — was an autonomous system that wrote code, made its own decisions, and occasionally went rogue. We thought it was was great name for an AI that can learn on its own, so we kept Anton, dropped the "Son of".

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
