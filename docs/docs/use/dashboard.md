---
title: Status dashboard
description: A quick health check of your Anton install with the anton dashboard command.
---

# Status dashboard

For a quick, read-only look at your Anton install without starting a chat:

```bash
anton dashboard
```

![Anton dashboard](/img/anton-dashboard-example.png)

The dashboard shows the robot banner with your installed version, then two
panels:

- **Commands** — pointers to handy terminal commands: `sessions` (browse sessions), `learnings` (review learnings), and `version` (show version). See the [CLI command reference](/reference/cli-commands).
- **Status** — the current state of this workspace:

| Field | Meaning |
| --- | --- |
| Memory | Whether memory is enabled or disabled |
| Sessions | How many sessions are stored in this workspace |
| Channel | The active channel (`cli`) |
| Theme | Current theme (`dark` or `light`) |
| Model | The configured coding model |

The dashboard reads state and exits — it doesn't start a session. To start
working, just run `anton`.
