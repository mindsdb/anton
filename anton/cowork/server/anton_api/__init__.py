"""Anton API protocol layer.

The Pydantic models, SSE event formatter, and lifecycle managers that back
the /v1/responses, /v1/conversations, and /v1/scratchpad endpoints. Kept
separate from cowork-specific routes so this layer can be lifted into a
shared package (or moved into the cloud container) without dragging
cowork-side state with it.
"""
