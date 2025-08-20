import os

from dotenv import load_dotenv

from minds.common.logger import setup_logging

logger = setup_logging()

# Load environment variables from .env file
load_dotenv()

# ====================================
# DB
# ====================================

DATABASE_URI = os.getenv(
    "DATABASE_URI", "postgresql://minds:minds@localhost:35432/minds"
)

DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", 20))  # Default max overflow
DB_POOL_PRE_PING = bool(os.getenv("DB_POOL_PRE_PING", True))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", 300))  # 60 seconds
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", 20))  # Default pool size
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", 300))  # 60 seconds

# Add query timeout configurations
DB_QUERY_TIMEOUT = int(os.getenv("DB_QUERY_TIMEOUT", 300))  # 5 minutes in seconds
DB_STATEMENT_TIMEOUT = int(
    os.getenv("DB_STATEMENT_TIMEOUT", 300000)
)  # 5 minutes in milliseconds

# ====================================
# Langfuse
# ====================================

LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_ENABLED", False))
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3001")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "not set")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "not set")

# ====================================
# LLM AND MODEL CONFIGURATIONS
# ====================================

# API keys and basic LLM settings
OPEN_AI_API_URL = os.getenv("OPEN_AI_API_URL", "https://api.openai.com/v1")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY", "not set")
OPEN_AI_MAX_TOKENS = int(os.getenv("OPEN_AI_MAX_TOKENS", 400000))
OPEN_AI_MODEL_NAME = os.getenv("OPEN_AI_MODEL_NAME", "gpt-4o")

# ====================================
# MindsDB
# ====================================

MINDSDB_API_SERVER = os.getenv("MINDSDB_API_SERVER", "http://localhost:47334")
