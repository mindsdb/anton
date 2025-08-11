import os

from dotenv import load_dotenv

from minds.common.logger import setup_logging

logger = setup_logging()

# Load environment variables from .env file
load_dotenv()

LANGFUSE_ENABLED = bool(os.getenv('LANGFUSE_ENABLED', False))
LANGFUSE_HOST = os.getenv('LANGFUSE_HOST', 'http://localhost:3001')
LANGFUSE_PUBLIC_KEY = os.getenv('LANGFUSE_PUBLIC_KEY', 'not set')
LANGFUSE_SECRET_KEY = os.getenv('LANGFUSE_SECRET_KEY', 'not set')

# ====================================
# LLM AND MODEL CONFIGURATIONS
# ====================================

# API keys and basic LLM settings
OPEN_AI_API_URL = os.getenv('OPEN_AI_API_URL', 'https://api.openai.com/v1')
OPEN_AI_API_KEY = os.getenv('OPEN_AI_API_KEY', 'not set')
OPEN_AI_MAX_TOKENS = int(os.getenv('OPEN_AI_MAX_TOKENS', 400000))
OPEN_AI_MODEL_NAME = os.getenv('OPEN_AI_MODEL_NAME', 'gpt-4o')
