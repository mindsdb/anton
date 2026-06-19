---
id: quickstart
title: Quickstart
description: Install Anton, run your first task, and watch it learn — in about five minutes.
---

# Quickstart

Anton is an open-source AI coworker that can execute tasks, connect to tools
and data, remember lessons, and improve its workflows over time. This page
takes you from nothing to a working Anton — and shows you the learning loop
that makes it different from any other agent — in about five minutes.

## 1. Install Anton

**macOS / Linux:**

```bash
curl -sSf https://raw.githubusercontent.com/mindsdb/anton/main/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
```

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/mindsdb/anton/main/install.ps1 | iex
```

Prefer a desktop app? See [Desktop app](/use/desktop). For offline installs,
Linux distro notes, and troubleshooting, see [Installation](/start/install).

## 2. Run it

```bash
anton
```

On first run Anton shows its terms, then asks you to pick an LLM provider.

## 3. Pick a provider

The recommended default is **Minds** ([mdb.ai](https://mdb.ai)) — choose
option `1`. Minds gives you smart model routing, cost optimization, and secure
data connectors with a single API key. If you don't have a key yet, Anton
opens the signup page for you; it takes a few seconds.

<details>
<summary>Already have an Anthropic, OpenAI, or Gemini key?</summary>

Choose option `3` (**Bring your own key**), then pick your provider and paste
your API key. Anton validates the key with a quick probe call before saving
it. You can also point Anton at any OpenAI-compatible endpoint — Ollama, vLLM,
Together, Groq, and friends. The full comparison lives in
[Pick a provider](/start/pick-a-provider), and you can switch any time with
the `/llm` command.

</details>

Your key is stored in `~/.anton/.env`, so it carries across sessions and
workspaces.

## 4. Ask Anton to do something

Try a small local task first — no connections or extra keys needed:

```
you> Write a Python script that prints the first 10 Fibonacci numbers, and run it.
```

Watch what happens: Anton opens a **scratchpad** — an isolated code execution
environment — writes the script, runs it, and shows you the output. That
scratchpad is the core of how Anton works: most tasks, from web scraping to
database analysis to building dashboards, run through it. You describe the
outcome; Anton writes and executes whatever code gets there.

## 5. Save what it learned as a skill

This is where Anton stops being a stateless chat agent. Tell it to remember
the procedure it just performed:

```
/skill save fibonacci
```

Anton reads the recent work, drafts a step-by-step procedure, and saves it to
its skill library at `~/.anton/skills/`. Next time a request matches, Anton
recalls the procedure instead of reasoning from scratch. More in
[Skills](/teach/skills).

## 6. Peek at Anton's memory

Anton has been taking notes the whole time. Look inside the workspace folder
it created:

```bash
ls .anton/
```

You'll find, among other things:

- `memory/lessons.md` — facts Anton learned while working
- `memory/rules.md` — behavioral rules (always / never / when)
- `episodes/` — a timestamped log of this session

These are plain markdown files — open them, read them, edit them. You can
also inspect everything from inside the chat with `/memory`. See
[What Anton remembers](/teach/memory-overview).

## 7. Resume the session

Quit (`exit` or `Ctrl+D`), then start Anton again and pick up where you left
off:

```
/resume
```

Anton lists your previous sessions and restores the conversation. Between the
skill you saved, the lessons on disk, and the resumable session, you've now
seen the loop that defines Anton: **work → learn → improve**.

## Next steps

- [Pick a provider](/start/pick-a-provider) — all five provider options compared
- [Connect things](/connect/overview) — databases, Gmail, APIs, web search
- [Teach Anton](/teach/memory-overview) — memory, lessons, rules, and skills
- [Slash commands](/reference/slash-commands) — the full in-chat command surface
