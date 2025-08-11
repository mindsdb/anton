from typing import List

from langfuse.decorators import observe

from minds.client.openai_client import open_ai_client
from minds.requests.schemas import Message, Role
from minds.requests.stream import MessageStreamer


class ChatCompletionsHandler:
	
	def __init__(self,
	             messages: List[Message],
	             model: str,
	             stream: bool):
		"""
		Initialize the ChatCompletionsHandler with a list of messages.
		
		Args:
			messages (List[Message]): List of messages to handle.
			model (str): The model to use for chat completions.
			stream (bool): Whether to stream the response.
		"""
		self.messages = messages
		self.model = model
		self.stream = stream
	
	@observe
	async def chat_completions(self, streamer: MessageStreamer) -> str | None:
		"""
		Dummy chat completions method.
		
		Args:
			streamer (MessageStreamer): The streamer to push messages to.
		Returns:
			str | None: A string response or None if no response is generated.
		"""
		
		await streamer.push(role=Role.system, content=f"Using model: {self.model}")
		await streamer.push(role=Role.system, content="Messages received:")
		for message in self.messages:
			await streamer.push(role=message.role, content=f"\t{message.content}")

		await streamer.push(role=Role.system, content="This is a dummy chat completion response.")
		
		# Simulate a request - EXAMPLE ONLY
		dummy_messages = [
			Message(role=Role.user, content="Hello!"),
		]
		
		response_generator = open_ai_client.chat_completions(
			messages=dummy_messages,
			stream=self.stream,
		)
		
		# Collect all response chunks
		result_parts = []
		async for content in response_generator:
			await streamer.push(role=Role.assistant, content=content)
			result_parts.append(content)
		
		result = "".join(result_parts)
		
		return result
