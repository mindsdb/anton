"""STATE API errors — identical across both drivers."""


class StateError(Exception):
    """Base class for all STATE errors."""


class StateValidationError(StateError):
    """Item/key failed validation (empty key, type, size, TTL format)."""


class ConditionalCheckFailed(StateError):
    """Conditional write not applied (if_not_exists / if_version)."""


class StateThrottled(StateError):
    """Operation rate limit exceeded (DynamoDB throttle / OnDemandThroughput cap)."""
