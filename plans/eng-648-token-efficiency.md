# ENG-648: Reduce anton + cowork-server token usage

Linear: [ENG-648](https://linear.app/mindsdb/issue/ENG-648/reduce-anton-cowork-server-token-usage-promptskills-migration-and)

## Problem

Anton (and cowork-server around it) consume excessive tokens for everything. Even a
trivial "answer from context" turn pays for: a ~11.4K-token system prompt, ~4.2K
tokens of tool schemas, the full conversation history, and a frontier reasoning
model at high effort — multiplied by up to 25 tool rounds per turn, with **no prompt
caching whatsoever**.

## Findings (measured)

### Anton (`mindsdb/anton`)

**A1. Prompt caching designed but never implemented — the single largest waste.**
`prompt_builder.py` carefully orders content into a "cache-stable prefix" and
"volatile tail", but no `cache_control` breakpoints are ever set.
`AnthropicProvider.complete/stream` (`anton/core/llm/anthropic.py:83-300`) sends
`system` as a plain string, tools without markers. Result: ~15K tokens of static
prefix billed at full input price on every call, every tool round (up to 25/turn,
`anton/core/settings.py:9`).

**A2. System prompt is ~11,377 tokens (45,511 chars, default config), and ~57% of it
is two always-included tutorials:**

| Section | ~tokens | Needed when |
|---|---|---|
| `BACKEND_GENERATION_PROMPT` (`prompts.py:523-805`) | ~3,832 | only when building a backend app |
| `VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT` (`prompts.py:338-497`) | ~2,675 | only when building a dashboard |
| `CHAT_SYSTEM_PROMPT` incl. public-data-source URL catalog (`prompts.py:55-96`) | ~3,201 | catalog only when sourcing public data |
| Artifacts, viz preamble, conversation discipline, etc. | ~1,600 | mostly always |

**A3. Tool schemas: ~4,200 tokens, all tools always sent** (`registry.py:44-56` —
no lazy/deferred loading). Scratchpad guidance is duplicated between
`CHAT_SYSTEM_PROMPT` and `SCRATCHPAD_TOOL.description` (~2K chars each). The
module-level `SCRATCHPAD_TOOL` singleton is mutated at session build
(`session.py:720-731`) and can accumulate appended text.

**A4. Model routing exists but there is no cheap front-tier.** `LLMClient` has a
planning role (Sonnet) and a coding role (Haiku) used for summarization/memory
(`anton/config/settings.py:34-52`, `client.py`). Every user request goes straight
to the planning model with the full prompt — no classification step, no
"answer-from-context" path.

**A5. A lazy-load skills mechanism already exists and is the right vehicle**:
procedural memory lists one-line skill summaries in the prompt
(`prompt_builder.py:65-108`) and the `recall_skill` tool
(`anton/core/tools/recall_skill.py`) pulls the full `SKILL.md` on demand via
`SkillStore` (`anton/core/memory/skills.py`). Nothing built-in uses it yet.

**A6. Other:** full history re-sent every round (compaction only triggers at 0.7
context pressure or on overflow, `session.py:799-918, 1745-1751`); 10K-char
scratchpad results persist whole in history; `anton.md` inlined uncapped
(`workspace.py:142-152`); auto-retry re-runs whole turns up to 3×; per-turn memory
LLM calls (cortex rules retrieval, identity extraction every 5 turns, background
consolidation flushes).

### cowork-server (`mindsdb/cowork-server`)

**C1. Full history replayed to anton every turn, no cowork-side windowing**
(`anton_harness/harness.py:414-417`, hermes `:147-151`). Scheduler
(`scheduler.py:112`) and channels (`runtime.py:384`) fire full agent turns on
growing conversations indefinitely. Full scratchpad cell state is also
reconstructed from all messages each turn (`harness.py:400`).

**C2. ~558-token static suffix appended to the system prompt every turn**
(`harness.py:425-437`) plus an unbounded attachment path list rebuilt per turn.
The hermes harness dumps entire memory slots + per-datasource env-var rosters
every turn, unbounded (`hermes_harness/harness.py:352-366`, `adapter.py:35-48`).

**C3. Expensive defaults:** planning = Sonnet at reasoning effort "high"
(`app_settings.py:46-50, 73-78`); memory autopilot + episodic memory on by default
(`user_settings.py:267-281`) → extra per-turn LLM calls; 5 cowork tool schemas
always registered (`harness.py:452-460`).

**C4. Credential probe is a full autonomous agent turn** with a ~2.5KB "TRY HARD"
prompt and a 90s budget on every connection test (`probe.py:189-261, 435`).

**C5. Zero token accounting** anywhere in cowork — no counters, budgets, or
per-conversation cost visibility. Only Langfuse trace passthrough.

## Plan

### Phase 0 — Measure (prereq, small)
- Surface anton's per-turn usage (input/output/cache-read tokens) through the
  harness event stream and log it per conversation/turn in cowork.
- Gives before/after numbers for every phase below.

### Phase 1 — Prompt caching (biggest win ÷ smallest diff)
- Add `cache_control: ephemeral` breakpoints in `AnthropicProvider`: one after the
  tools block, one at the end of the static system prefix (the ordering work in
  `prompt_builder.py` is already done). Convert `system` string → content-block list.
- Expected effect: ~90% input-price reduction on the static ~15K-token prefix for
  every call after the first, compounding across ≤25 rounds/turn.

> **Status (2026-07-07):** Phase 3 (the **thalamus** gate) and Phase 2
> (prompts→skills) are implemented in this PR — measured base prompt: 45,511 →
> 18,847 chars (~11,377 → ~4,711 tokens, −59%) in dashboards mode. The thalamus
> prompt is ~1.7K chars (~420 tokens), no tool schemas. Phase 1 (caching) is
> pending a design decision with the `mindshub_inference` gateway (client-driven
> `cache_control` passthrough vs the gateway's existing `cache_align`
> heuristic); Phase 0 metrics will ride with that work. Phase 4 tracked
> separately in cowork-server.

### Phase 2 — Prompts → skills (built-in skills, loaded on demand)
- Move `BACKEND_GENERATION_PROMPT`, `VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT`, and
  the public-data-source catalog into **built-in skills** shipped with anton,
  surfaced as one-line entries in the procedural-memory section and recalled via
  the existing `recall_skill` tool.
- De-duplicate scratchpad guidance (keep it in the tool description, drop from
  `CHAT_SYSTEM_PROMPT`); cap `anton.md` inlining; stop mutating the shared
  `SCRATCHPAD_TOOL` singleton.
- Expected effect: base system prompt ~11.4K → ~4.5K tokens.

### Phase 3 — The thalamus (cheap front-model gate)

Named after the brain's central relay: nearly every signal passes through the
thalamus on the way to the cortex, and it *gates* what gets relayed up versus
handled by a fast subcortical path — exactly the respond-vs-delegate decision.
anton already names its subsystems after brain regions (Cortex, Cerebellum,
Anterior Cingulate); the thalamus joins them. Lives in
`anton/core/llm/thalamus.py`.

- New `thalamus` role on `LLMClient` (`.gate(...)`), a very cost-effective model
  (Haiku-class, defaults to the coding role). On each text turn it runs first
  with a minimal prompt (no tutorials, no tool schemas) over a condensed
  text-only history view and decides:
  1. **Respond directly** (*tonic relay*) — the answer is derivable from
     conversation context, no scratchpad/tools needed (greetings, follow-ups
     about prior results, rephrase/summarize asks).
  2. **Delegate** (*burst — alert the cortex*) — task needs tools/reasoning →
     forced `delegate` tool call hands off to the planning model, naming which
     built-in skills to preload (so the big model doesn't spend a round calling
     `recall_skill`).
- **Default-inhibit (the TRN gate):** every ambiguity resolves to delegate —
  empty response, an answer over the output budget, any error, image turns.
  Fails *open* so a mis-gate never drops a signal; user-visible behavior with
  the gate on is identical to off, minus latency/cost on trivial turns.
- **Corticothalamic feedback (future):** today the only top-down signal is the
  skill list. The principled next step is letting recent cortical outcomes
  (per-project delegation rate, last-turn tool usage) bias the gate. Seam is
  documented at `thalamus.gate_turn`, not yet built.
- Off by default (`ANTON_THALAMUS_ENABLED`) pending eval on a transcript corpus.

### Phase 4 — cowork-server diet (separate PR in cowork-server)
- Cap/trim the per-turn suffix and attachment listing; bound hermes memory/datasource
  dumps.
- Revisit defaults: reasoning effort "high" → "medium" for planning; memory
  autopilot opt-in per workspace; lazy-register the 5 cowork tools.
- Budget the credential probe (rounds/timeout/max tokens).
- Optional: cowork-side history windowing so anton isn't re-compacting the same
  transcript every turn.

## Sequencing & risk

1 → 2 → 3 in this repo (each independently shippable, measurable via Phase 0);
Phase 4 tracked in cowork-server. Phase 3 carries the main product risk
(misclassification → worse answers); mitigate with conservative routing (when in
doubt, delegate), an env flag to disable, and eval on a transcript corpus before
default-on.
