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
