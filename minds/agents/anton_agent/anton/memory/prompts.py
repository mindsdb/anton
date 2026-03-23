# --- Cortex prompts ---
_IDENTITY_EXTRACT_PROMPT = """\
Extract identity facts from this user message. Return a JSON array of strings,
each a concise fact about the user (name, timezone, expertise, preferences, tools).

If no identity-relevant information is found, return [].

Examples of identity facts:
- "Name: Jorge"
- "Timezone: PST"
- "Prefers dark mode"
- "Uses uv over pip"

Only extract facts that are clearly about the user's identity, preferences,
or working style. Ignore transient conversation details.
"""


_COMPACTION_PROMPT = """\
You are a memory compaction system. Review these memory entries and:
1. Remove exact duplicates
2. Merge entries that say the same thing differently — keep the clearest version
3. Remove entries that are superseded by newer, more specific entries
4. Keep all unique, useful entries

Return a JSON object with:
- "kept": array of entry strings to keep (cleaned up, no metadata comments)
- "merged": array of strings describing what was merged
- "pruned": array of strings describing what was removed and why

Be conservative — when in doubt, keep the entry.
"""


# --- Consolidator prompts ---
_CONSOLIDATION_PROMPT = """\
You are a memory consolidation system for an AI coding assistant.

Review this scratchpad session (sequence of code cells with their results) and
extract durable, reusable lessons. Focus on:

1. **Rules** — patterns to always/never follow:
   - "Always call progress() before long API calls in scratchpad"
   - "Never use time.sleep() in scratchpad cells"
   - Conditional rules: "If fetching paginated data → use async + progress()"

2. **Lessons** — factual knowledge discovered:
   - API behaviors: "CoinGecko free tier rate-limits at ~50 req/min"
   - Library quirks: "pandas read_csv needs encoding='utf-8-sig' for BOM files"
   - Data facts: "Bitcoin price data via /coins/bitcoin/market_chart/range"

Return a JSON array of objects:
[
  {
    "text": "the memory to encode",
    "kind": "always" | "never" | "when" | "lesson",
    "scope": "global" | "project",
    "topic": "optional-topic-slug",
    "confidence": "high" | "medium"
  }
]

Rules for scope:
- "global": universal knowledge useful across any project
- "project": specific to this workspace (file paths, project-specific APIs)

Rules for confidence:
- "high": clearly correct, verified by the session results
- "medium": probably correct but worth confirming

If no meaningful lessons exist, return [].
Do NOT extract trivial observations. Only encode genuinely reusable knowledge.
"""
