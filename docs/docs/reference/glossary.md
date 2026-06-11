---
title: Glossary
description: Mapping the plain names used in these docs to the brain-inspired names used in the code and developer guide.
---

# Glossary

Anton's internals are organized around a brain-inspired architecture, and the source code and [developer guide](/developer/architecture) use neuroscience names for its parts. The user docs deliberately don't. This page is the bridge: if you're crossing from the user docs into the code or developer guide, here's what each plain name is called on the other side.

| In the user docs | In the code & developer guide | One-line meaning |
|---|---|---|
| Memory (as a whole) | Engrams, stored via the hippocampus and coordinated by the cortex (`hippocampus.py`, `cortex.py`) | Each saved entry is an "engram" — a single memory trace; the hippocampus reads/writes them and the cortex decides what to load. See [Memory systems](/developer/memory-systems) |
| Identity / profile | `profile.md` — the Default Mode Network analog | Always-on facts about you that contextualize everything Anton does. See [Brain mapping](/developer/brain-mapping) |
| Rules | `rules.md` — basal-ganglia-style go/no-go gates | "Always" enables, "never" suppresses, "when" handles conditions. See [Brain mapping](/developer/brain-mapping) |
| Lessons | `lessons.md` — semantic memory (anterior temporal lobe analog) | Facts distilled from experience into general knowledge. See [Memory systems](/developer/memory-systems) |
| Topics | `topics/*.md` — cortical association areas analog | Per-subject knowledge loaded on demand, not in every prompt. See [Memory systems](/developer/memory-systems) |
| Skills | Procedural memory — the striatum analog (`skills.py`) | Reusable procedures retrieved on demand via the `recall_skill` tool. See [Skills internals](/developer/skills-internals) |
| Episodes | Episodic memory — medial temporal lobe analog (`episodes.py`) | The raw, timestamped session archive searched by the `recall` tool. See [Memory systems](/developer/memory-systems) |
| Background error learning | Cerebellum (`cerebellum.py`) and anterior cingulate cortex / ACC (`acc.py`) | The cerebellum learns from individual failed code cells; the ACC spots error *patterns* across a whole turn. See [Cerebellum and ACC](/developer/cerebellum-and-acc) |
| Session review (learning from finished work) | Consolidation — sleep-replay analog (`consolidator.py`) | After a work session, a background pass replays it and extracts durable rules and lessons. See [Memory systems](/developer/memory-systems) |
| Memory cleanup / `/memory vacuum` | Compaction — synaptic homeostasis analog | Deduplicates and merges memory entries when files grow too large. See [Memory systems](/developer/memory-systems) |
| Memory modes (autopilot / copilot / off) | The encoding gate — locus coeruleus analog | Controls how aggressively new memories get written, and which need your confirmation. See [Memory systems](/developer/memory-systems) |
| Scratchpad | Working-memory execution environment | The isolated environment where Anton writes and runs code, one venv per scratchpad. See [Scratchpad runtime](/developer/scratchpad-runtime) |
| Legacy migration (old learnings → memory) | Reconsolidation (`reconsolidator.py`) | One-time, automatic re-encoding of old memory formats into the current schema. See [Memory systems](/developer/memory-systems) |

For the full table of brain regions and their modules, see [Brain mapping](/developer/brain-mapping).
