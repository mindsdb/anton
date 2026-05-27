"""
Constants for the application.
"""

# =============================================================================
# HTTP Header Constants
# =============================================================================

HEADER_USER_ID = "X-User-Id"
HEADER_USER_EMAIL = "X-User-Email"
HEADER_ORGANIZATION_ID = "X-Organization-Id"
HEADER_USER_ROLES = "X-User-Roles"
HEADER_BILLING_PERIOD_START = "X-Billing-Period-Start"
HEADER_BILLING_PERIOD_END = "X-Billing-Period-End"

# Langfuse-proxy convention headers (see ``docs`` / anton-core integration).
# Emitted by harnesses like cowork to identify a conversation, turn, and host
# so traces can be grouped into sessions and named with the originating app.
# All three are optional; absent or malformed values are logged and ignored.
HEADER_LANGFUSE_SESSION_ID = "Langfuse-Session-Id"
HEADER_LANGFUSE_TAGS = "Langfuse-Tags"
HEADER_LANGFUSE_METADATA = "Langfuse-Metadata"

# Distributed-trace propagation headers. When an upstream caller (e.g. the
# Anton harness) has already opened a Langfuse trace in the *same* Langfuse
# project, it can pass its trace id here so this service's observations nest
# onto that trace instead of starting a fresh one — letting task-success evals
# run against the whole multi-service trace. Both are optional:
# - ``Langfuse-Trace-Id``: 32 lowercase hex chars (W3C/OTel trace id). Alone,
#   it co-locates this request's spans on the caller's trace.
# - ``Langfuse-Parent-Observation-Id``: 16 lowercase hex chars. Optional; when
#   present the request's root span nests under that specific upstream span.
HEADER_LANGFUSE_TRACE_ID = "Langfuse-Trace-Id"
HEADER_LANGFUSE_PARENT_OBSERVATION_ID = "Langfuse-Parent-Observation-Id"

# =============================================================================
# Dynamic Configs
# =============================================================================

DYNAMIC_CONFIG_MIND_USAGE_LIMITS = "mind-usage-limits"

# =============================================================================
# CONTEXT FIELDS
# =============================================================================

CONTEXT_FIELD_REQUEST_ID = "request_id"
CONTEXT_FIELD_ORGANIZATION_ID = "organization_id"
CONTEXT_FIELD_USER_ID = "user_id"
CONTEXT_FIELD_USER_ROLES = "user_roles"
CONTEXT_FIELD_ENV = "env"
