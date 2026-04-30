"""Optional HTTP server for Anton.

Exposes Anton's chat session over an OpenAI-compatible Responses API,
mirroring the surface of anton_servicesrepo/scratchpad_service so the
antontron app (and any other client) can talk to a local or remote
Anton instance over HTTP.

Install with: pip install anton[server]
Run with:    anton serve
"""
