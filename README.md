<p align="center">
  <pre align="center">
 ▄▀█ █▄ █ ▀█▀ █▀█ █▄ █
 █▀█ █ ▀█  █  █▄█ █ ▀█
  autonomous coworker</pre>
</p>

<p align="center">
  Collaborate with Anton to get stuff done.
</p>

<br>

## How it works

Anton is your coworker that's excellent at many things — research, data work, more data work ;), being your thinking partner, and yes, coding solutions to problems so you dont have to.

```
anton run "cross-reference our postgres customers table with the revenue spreadsheet and flag mismatches"
anton run "draft responses to the 12 unread support emails"
anton run "pull last quarter's sales data, combine it with the marketing spend sheet, and show me what's working"
anton run "analyze churn patterns in the database and give me the top 5 insights"
```

You don't write code. You don't manage prompts. You just tell Anton what you need done — the same way you'd ask a coworker — and he figures out the rest. If it needs code, he writes it. If it needs research, he does it. If it needs both, he handles that too.

He plans the approach, executes, and learns from every task so he gets sharper over time.

You focus on the *what*. Anton handles the *how*.

## Install

```bash
curl -sSf https://raw.githubusercontent.com/mindsdb/anton/main/install.sh | sh
```

## Quick start

```bash
anton run "summarize yesterday's sales calls and email me the highlights"
```

That's it. Anton takes it from there.

## Commands

| Command | What it does |
|---|---|
| `anton` | Dashboard |
| `anton run "task"` | Give Anton a task |
| `anton skills` | See what Anton can do |
| `anton sessions` | Browse past work |
| `anton learnings` | What Anton has learned |

## Configuration

```bash
~/.anton/.env            # API keys
~/.anton/skills/         # Custom skills
~/.anton/memory/         # Sessions & learnings
```

## Why "Anton"?

If you've seen *Silicon Valley*, you know.

Gilfoyle's AI — **Son of Anton** — was an autonomous system that wrote code, made its own decisions, and occasionally went rogue in spectacular fashion.

We kept the name. Dropped the "Son of." Same energy — an autonomous agent that thinks for itself and gets work done.

## License

MIT
