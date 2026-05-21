# Minds Agents

This document explains how to implement a custom agent that integrates cleanly with the Minds runtime.

---

## Overview

An agent must:

1. Implement the `BaseAgent` interface
2. Stream user-facing output via `MessageStreamer`
3. Return a final result that conforms to `AgentResponse`

> **Agent layout:** Each agent implementation resides in its own subdirectory under `agents/`. The directory name must end in `_agent` and include an `agent.py` module — `agent_controller.py` discovers agents by scanning subdirectories whose names end in `_agent` and importing `<name>.agent`.

---

## Implementing an Agent

To create a new agent, add a new subdirectory (Python package) under `agents/`. Each agent package **must** include an `agent.py` module, which acts as the entry point for that agent's implementations.

Example structure:

```text
agents/
  └── my_agent/
      ├── __init__.py
      ├── agent.py
      └── ...
```

> **Note:** `passthrough_agent/` follows the same `agent.py` convention but is excluded from auto-discovery (it doesn't subclass `BaseAgent`); it's invoked through `OpenAIRequestHandler`'s passthrough short-circuit instead.

Then, subclass `BaseAgent` and implement the `run()` method.

---

## The `run()` Contract

The `run()` method receives:

- **`messages`**  
  A list of `Message` objects representing the conversation so far. The most recent user message will typically be the last element.

- **`streamer`**  
  A `MessageStreamer` used to push incremental, user-visible output.

- **`stream`**  
  A boolean flag indicating whether streaming is enabled. Agents should respect this flag.

The method **must** return an `AgentResponse`.

The content pushed into the streamer should represent user-facing chunks of the agent’s final textual answer.

When streaming is enabled, these chunks are incremental portions of the answer as it is being produced. When streaming is disabled, the streamer should receive a single message containing the complete answer text.

This streamed content is not the final `AgentResponse`. Instead, it corresponds specifically to the value that will ultimately be returned in `AgentResponse.answer`.

---

