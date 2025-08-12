from typing import Union

from langfuse.decorators import observe
from sqlmodel import Session
from starlette.responses import StreamingResponse, JSONResponse

from minds.common.logger import setup_logging
from minds.handlers.chat_completions_handler import ChatCompletionsHandler
from minds.requests.chat_completions_request import ChatCompletionsRequest
from minds.requests.stream import process_streaming_producer, process_non_streaming_producer

# Set up logging
logger = setup_logging()

@observe
async def chat_completions_request_handler(
		request_id: str,
		session: Session,
		chat_completions_request: ChatCompletionsRequest
) -> Union[StreamingResponse, JSONResponse]:
	"""
	Handle chat completions requests.
	
	Args:
		request_id (str): The unique identifier for the request.
		session (Session): The SQLAlchemy session for database operations.
		chat_completions_request (ChatCompletionsRequest): The request object containing chat completion parameters.
	Returns:
		Union[StreamingResponse, JSONResponse]: A streaming response if the request is for streaming, otherwise a JSON response.
	"""
	
	logger.debug(f"🔄[{request_id}] Chat Completion Request: {chat_completions_request.model_dump()}")
	
	stream = chat_completions_request.stream if chat_completions_request.stream is not None else False
	logger.debug(f"🔄[{request_id}] Stream: {stream}")
	
	messages = chat_completions_request.messages
	logger.debug(f"🔄[{request_id}] Messages: {messages}")
	
	model = chat_completions_request.model
	logger.debug(f"🔄[{request_id}] Model: {model}")
	
	chat_completions_handler = ChatCompletionsHandler(
		session=session,
		messages=messages,
		model=model,
		stream=stream
	)
	
	if stream:
		logger.debug(f"🔄[{request_id}] Chat completions request is streaming.")
		response = await process_streaming_producer(
			producer=lambda streamer: chat_completions_handler.chat_completions(streamer=streamer),
			request_id=request_id,
			model=model
		)
	else:
		logger.debug(f"🔄[{request_id}] Chat completions request is non-streaming.")
		response = await process_non_streaming_producer(
			producer=lambda streamer: chat_completions_handler.chat_completions(streamer=streamer),
			request_id=request_id,
			model=model
		)
	
	return response
