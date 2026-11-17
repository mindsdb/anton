# cowork-product: where the facts come from

For maintainers. Only `SKILL.md` is loaded into the model's context; this file
is not, so provenance notes live here instead of costing tokens on every recall.

Each fact in `SKILL.md` was read out of the code that implements it. Nothing
fails automatically when that code changes, so re-check these when touching:

- installed app name — `package.json` `productName` (cowork)
- platforms and installer types — `.github/workflows/build-installers.yml`
- download URL — `src/renderer/lib/mindsUrls.ts` and the `mindshub.ai/download`
  call sites in `ComingSoonModal`, `ConnectorPicker`, `useAppUpdates`
- account-wide harness toggle — the `host.isWeb` gate on the `Agent Harness`
  group in `src/renderer/cowork/views/settings/SettingsView.jsx`
- whether Hermes still exists — `cowork/harnesses/` in cowork-server (the
  directory is the answer) against the UI references above; when they agree
  again, replace the HERMES section with the settled fact
- Coding Mode being desktop-only — `codeModeAvailable` in
  `src/renderer/platform/host.ts`, and `renderCodingModeSection`
- local models being desktop-only — the `{!orgMode && …}` gate on the
  `LLM Providers` group in `SettingsView.jsx` `renderAgentSection`, and
  `setOrgMode(decision.orgMode ?? host.isWeb)` in `src/renderer/App.tsx`
  (the hosted browser deployment is org mode)
- local-model setup steps and labels — `PROVIDER_TYPE_DESC` /
  `PROVIDER_LABELS_LOCAL` and the `openai-compatible` Base URL field in
  `SettingsView.jsx`, the `Model Router` group there, and the "Custom" Base URL
  field in `src/renderer/pages/arcade/OnboardingScreen.tsx`
- the three Model Router roles, and a role left alone keeping its provider —
  the `RoleRow` calls in the `Model Router` group, and
  `defaultModeProviderType` in `SettingsView.jsx`
- the server-rejection limitation — `_translate_tool_choice` in
  `anton/core/llm/openai.py` sends a named `tool_choice`. `send_structured` in
  `anton/core/llm/structured.py` resends a refused one unforced only when the
  400 names `tool_choice` (`refuses_forced_tool_choice`); a server whose refusal
  says something else still fails the verifier, so the bullet stays until
  every refusal gets the fallback
- what Cowork cannot make — anton's tool set (`anton/core/tools/tool_defs.py`
  has no video or audio generator). Images: `generate_image` (ENG-4024) is
  offered per turn by `ChatSession._settle_image_offer` and
  `_provider_offers_images` in `anton/core/session.py` — on MindsHub when an
  image model is configured or the key's catalogue lists one. The section is
  worded in terms of that tool; revisit it when a video or audio tool lands
- free Air allowance, refill and balance fallback — cowork's own copy:
  `usageBar.freeLow*` / `freeUsed*` and `connectProviderBody` in
  `src/shared/i18n/locales/en/chat.json`; other models needing credits —
  `needsCreditsTitle` / `needsCreditsBody` there. Sizes, prices and refill
  times are deliberately not stated: they are server-side settings that change
- billing page — `MINDS_BILLING_URL` in `src/renderer/lib/mindsUrls.ts`
- chat subscriptions not usable, API keys on desktop only — the `Provider` enum
  in cowork-server `cowork/common/settings/user_settings.py` (API keys and
  OpenAI-compatible servers only, no account sign-in) and the `!orgMode` gate on
  `LLM Providers` above
- the existing-folder route — `canChooseFolder = host.isElectron && !orgMode`
  and the `newProjectModal.useExistingFolder` / `locationLabel` /
  `locationHelpChosen` strings in cowork's `NewProjectModal.jsx` and
  `src/shared/i18n/locales/en/projects.json`
