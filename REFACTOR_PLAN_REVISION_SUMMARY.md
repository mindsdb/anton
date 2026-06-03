# Phase 1 Refactor Plan — Revision Summary

## What Changed

This revision reflects your decision to **remove PassthroughAgent from the production request flow** entirely, replacing it with a cleaner `InferenceService`-driven architecture.

---

## Old Design (Initial Plan)

```
Handler
  → PassthroughAgent (wrapper) ← Runtime dependency
    → Adapter
```

**Problem:** PassthroughAgent remains in the critical request path, just delegating to adapters. Extra indirection.

---

## New Design (Revised Plan)

```
Handler
  → InferenceService (orchestrator) ← Runtime dependency
    → ModelResolver
    → ProviderAdapter
```

**Benefit:** Cleaner separation of concerns. Handler talks to a proper service, not a mis-named "agent."

---

## Key Changes to the Plan

### 1. **Handler Flow (Major Change)**

**OLD:**
```python
# Handler creates PassthroughAgent
handler.agent = PassthroughAgent(config=config, instrument=instrument)

# Handler calls agent.proxy()
response = await handler.agent.proxy(...)

# Handler calls agent.get_last_run_*()
usage = await handler.agent.get_last_run_usage()
```

**NEW:**
```python
# Handler creates InferenceService
handler.inference_service = InferenceService(model_resolver=ModelResolver(settings))

# Handler calls inference_service.inference()
response, result = await handler.inference_service.inference(...)

# Handler reads result tuple
usage = result.usage
```

**Impact:** Handler code changes, but behavior is identical.

---

### 2. **PassthroughAgent Status**

**OLD:** Compatibility wrapper (stays in runtime flow)

**NEW:** Optional deprecated shim (for tests/imports only, not used by handler)
- If tests use it directly → keep as shim that delegates to InferenceService
- If no tests use it → delete entirely

---

### 3. **Concurrent Request Safety**

**NEW CONSIDERATION:** InferenceService creates a **fresh adapter per request**, ensuring:
- No shared state across concurrent requests
- UsageBox is per-request (no cross-request pollution)
- Service is stateless

This is actually **better than** the old PassthroughAgent approach.

---

### 4. **PR Breakdown Changes**

**OLD:** 8 PRs
1. Adapter interface
2. Move types
3. Move providers
4. Create adapters
5. Create ModelResolver
6. PassthroughAgent wrapper
7. Update handler imports
8. Tests

**NEW:** 9 PRs (more explicit)
1. Adapter interface + Move types
2. Move provider modules
3. Create concrete adapters
4. Create ModelResolver
5. **Create InferenceService** (new main orchestrator)
6. **Update handler to call InferenceService** (replaces step 7)
7. PassthroughAgent shim (optional, only if tests use it)
8. Add comprehensive tests
9. Documentation (optional)

**Key difference:** InferenceService gets its own PR (not a thin factory anymore).

---

### 5. **Architecture Diagram**

**After Phase 1:**
```
HTTP POST /api/v1/chat/completions
  ↓
openai_request_handler.proxy_chat_completions()
  ↓
is_passthrough_model(model) → True
  ↓
InferenceService.inference(
    model_name="latest:sonnet",
    messages=[...],
    stream=True,
    ...
)
  ├─ ModelResolver.resolve("latest:sonnet") 
  │  → PassthroughModelConfig(api_kind=ANTHROPIC_MESSAGES, model_name="claude-3-sonnet-4", ...)
  │
  ├─ _create_adapter(ANTHROPIC_MESSAGES)
  │  → AnthropicAdapter() instance (fresh per request)
  │
  └─ adapter.complete(config, messages, stream, ...) 
     → calls anthropic_module.proxy_anthropic()
     → returns OpenAI-format response
  ↓
(response: StreamingResponse, result: InferenceResult)
  ├─ response.body_iterator (yields SSE chunks)
  └─ result.config, result.usage, result.output, result.artifacts
  ↓
Handler's _wrapped_body() consumes result after stream completes
```

---

## What Stays the Same

- ✅ Request input format (OpenAI chat/completions)
- ✅ Response output format (OpenAI chat.completion)
- ✅ Streaming behavior (SSE chunks, usage captured after)
- ✅ Tool calling (function-calling translation)
- ✅ Error handling (JSONResponse with error codes)
- ✅ Usage tracking (tokens recorded in database)
- ✅ Backward compat for imports (ModelResolver is optional)

---

## Why This Revision is Better

| Aspect | Old Design | New Design |
|--------|-----------|-----------|
| Handler dependency | PassthroughAgent (misnamed) | InferenceService (clear purpose) |
| State management | PassthroughAgent instance stores last usage | Fresh adapter per request (stateless service) |
| Request isolation | Potential sharing if agent reused | Guaranteed isolation (new adapter each call) |
| Code clarity | Wrapper pattern | Direct orchestration |
| Phase 2 prep | Adapter interface ready | Adapter interface + clean service boundary |

---

## Timeline Impact

- **Old estimate:** ~7 days
- **New estimate:** ~9-10 days (one extra day for InferenceService PR, two days for handler refactor)

Longer refactor, but cleaner architecture + better concurrency safety.

---

## No Behavior Changes for Handler Tests

All existing handler tests should pass **with zero changes** because:
1. Handler's request parsing: unchanged
2. Handler's response format: unchanged
3. Handler's streaming logic: unchanged
4. Handler's error responses: unchanged

The handler's implementation changes, but not its contract or output.

---

## PassthroughAgent Fate

**Option A (if tests use it):**
```python
# Keep as deprecated shim
class PassthroughAgent:
    def __init__(self, config):
        self._service = InferenceService(...)
    
    async def proxy(...) -> Response:
        response, result = await self._service.inference(...)
        self._last_result = result
        return response
    
    async def get_last_run_usage(...):
        return self._last_result.usage
```

**Option B (if no tests use it):**
- Delete entire `minds/agents/passthrough_agent/` directory
- Keep re-exports only for backward compatibility

---

## Next Action Items

1. **Review this revision** — Confirm the handler → InferenceService approach is acceptable
2. **Approve the architecture** — Check concurrency/streaming patterns make sense
3. **Check test dependencies** — Determine if PassthroughAgent is used in tests (for shim decision)
4. **Start PR 1** — Adapter interface + type movement

---

## Questions for You

1. Do any tests outside of `handlers/` directly instantiate PassthroughAgent?
   - If yes → Keep shim
   - If no → Delete it

2. Is the "InferenceService creates fresh adapter per request" pattern acceptable for performance?
   - (Each request gets new SDK client instances, which is already the current behavior)

3. Should InferenceService be created once (singleton) or per request?
   - **Recommended:** Per request (or inject into handler factory) — stateless service

4. Any concerns about the timing of result capture (after stream completes)?
   - Pattern is identical to current PassthroughAgent behavior — should be safe

---

## Files to Update

**Production code:**
- `minds/inference/adapter.py` (new)
- `minds/inference/service.py` (new)
- `minds/inference/model_resolver.py` (new)
- `minds/inference/providers/openai_adapter.py` (new)
- `minds/inference/providers/anthropic_adapter.py` (new)
- `minds/inference/providers/gemini_adapter.py` (new)
- `minds/inference/types.py` (new, moved from common.py)
- `minds/handlers/openai_request_handler.py` (MODIFIED — calls InferenceService)
- `minds/agents/passthrough_agent/agent.py` (SHIM or DELETE)
- `minds/agents/passthrough_agent/common.py` (re-export)
- `minds/common/passthrough_config.py` (backward compat shim)

**Test code:**
- `tests/unit/inference/` (new, comprehensive)

**Documentation:**
- `CLAUDE.md` (update architecture diagram)

---

## Ready to Proceed?

The revised plan is **final and actionable**. Each PR is clear, with acceptance criteria.

Start with **PR 1: Adapter Interface + Type Movement**
