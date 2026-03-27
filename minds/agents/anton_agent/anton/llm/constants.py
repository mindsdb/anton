_CONTEXT_WINDOWS: list[tuple[str, int]] = [
    # Anton defaults (exact model IDs first)
    ("claude-sonnet-4-6", 200_000),
    ("claude-haiku-4-5-20251001", 200_000),
    # Claude families
    ("claude-opus-4", 200_000),
    ("claude-sonnet-4", 200_000),
    ("claude-haiku-4", 200_000),
    ("claude-3", 200_000),
    ("claude-", 200_000),
    # OpenAI families
    ("gpt-5", 400_000),
    ("gpt-4.1", 1_000_000),
    ("gpt-4o", 128_000),
    ("gpt-4", 128_000),
    ("o3", 200_000),
    ("o1", 200_000),
]
_DEFAULT_CONTEXT_WINDOW = 128_000
_DEFAULT_MAX_TOKENS = 16384
