"""Message streaming implementations for different API modes."""

from minds.requests.stream import MessageStreamer
from minds.schemas.chat import Role


class SimpleStreamer(MessageStreamer):
    """No-op streamer for responses API — kept for interface compatibility."""

    async def push(self, role: Role, content: str):
        """No-op implementation."""
        pass
