# Runtime diagnostics

`LOG_LEVEL=WARNING` and `LOG_FORMAT=json` are the process defaults. Set
`LOG_LEVEL=DEBUG` (or `INFO`) temporarily for CLI/cloud-turn diagnostics; unset it
to restore WARNING. `LOG_FORMAT=text` is a local readability override with the
same exception privacy policy. Restore `json` for collected deployments.

Cloud turns bind `conversation_id` for their entire lifetime and reset it when
the turn ends. Dependency loggers remain at WARNING during troubleshooting.
`anton.usage` remains an independent plain-text INFO stream for its existing
usage/evaluation consumers. Agent stdout/protocol events, generated output,
saved conversations and tracing are separate contracts and retain their current
content. Arbitrary user-code output is not claimed to be a safe diagnostic.

Publish an Anton release before downstream packages or images claim these
diagnostics. Rebuild those consumers with that release; changing a source
checkout does not update installed agents.

Diagnostics use `logged_at`, `logger`, `log_level`, and `message`. Explicit events,
reasons, source locations and correlation fields identify failures. Exception
values, source lines, locals, arbitrary extras, HTTP queries and provider bodies
are excluded. Use the structured fields when querying logs; the previous text
prefixes are no longer a parsing contract.

Enable extra detail only for an owned troubleshooting session, repeat the failing
operation, and restore the default afterward. Logging controls do not enable
framework debug responses. Roll back a release by restoring its previous image
or package version and its previous logging settings together. Before promotion,
verify collector parsing, saved searches, alerts and retained operational streams
against the emitted records. A passing unit suite does not establish deployment
or collector compatibility.
