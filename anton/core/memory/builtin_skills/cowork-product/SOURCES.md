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
  `anton/core/llm/openai.py` sends a named `tool_choice` with no unforced
  fallback; remove that bullet when the verifier gains one
