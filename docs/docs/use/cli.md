---
title: The terminal interface
description: Running Anton in your terminal — the banner, prompt, slash commands, keybinds, and themes.
---

# The terminal interface

Anton's primary interface is your terminal. Start it with:

```bash
anton
```

By default Anton uses the current directory as its workspace. To use a
different one:

```bash
anton --folder /path/to/workspace
```

There's also `anton --resume` (or `-r`) to pick up a previous session — see
[Sessions](/use/sessions).

## The banner

On startup Anton draws its robot, plays a short animation on interactive
terminals, and prints the version and a tagline:

```
        ▐
   ▄█▀██▀█▄   ♡♡♡♡
 ██  (°ᴗ°) ██
   ▀█▄██▄█▀      ▄▀█ █▄ █ ▀█▀ █▀█ █▄ █
    ▐   ▐        █▀█ █ ▀█  █  █▄█ █ ▀█
    ▐   ▐
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 v2.26.4.30.0 — "autonomous by design"
 type '/help' for commands or 'exit' to quit.

you>
```

Type what you want done at the `you>` prompt — plain language, no special
syntax. See [How a turn works](/use/chat-basics) for what happens next.

## Slash commands

Type `/` and the autocomplete menu opens, listing every command with a short
description. Keep typing to filter; commands with subcommands (like
`/share` or `/skill`) complete their next token too. Type `/help` to print
the full list at any time. The complete inventory is in
[Slash commands](/reference/slash-commands).

## Keybinds and exiting

| Key / input | Where | What it does |
| --- | --- | --- |
| `Esc` | Setup prompts (provider, API key, …) | Go back to the previous step |
| `Ctrl+D` | Chat prompt | Exit Anton |
| `exit`, `quit`, or `bye` | Chat prompt | Exit Anton |

## Themes

Anton ships a dark theme (the default) and a light theme:

```
/theme          # toggle between light and dark
/theme light
/theme dark
```

To set the theme before startup, use the `ANTON_THEME` environment variable
(`dark` or `light`).

## Images and files

- `/paste` attaches an image from your clipboard to your next message. You can also press Enter on an empty prompt — if your clipboard holds an image, Anton attaches it.
- Drag and drop a file from your file manager into the terminal and Anton picks up the path and includes the file in the conversation.

Images are sent to the model as multimodal content; very large images
(roughly over 3.7 MB raw) are rejected with a hint to resize.
