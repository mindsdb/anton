# minds/tests/integration/config.py
import logging
import os

from dotenv import load_dotenv

# ===================================================================
# 1. CONFIGURATION
# ===================================================================

# Set up basic logging to output INFO level messages.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables from a .env file if it exists.
load_dotenv()

# --- API and Schema Configuration ---
MINDS_API_BASE_URL = os.getenv("MINDS_API_BASE_URL")
logging.info(f"--- USING API BASE URL: {MINDS_API_BASE_URL} ---")
MINDS_API_KEY = os.getenv("MINDS_API_KEY")

# --- DATASOURCE CONFIGURATIONS ---
DATASOURCE_CONFIGS = []

# --- PostgreSQL Configuration (Reads from your existing PG_ environment variables) ---
POSTGRES_CONFIG = {
    "host": os.getenv("PG_HOST", "samples.mindsdb.com"),
    "port": int(os.getenv("PG_PORT", 5432)),
    "user": os.getenv("PG_USER", "demo_user"),
    "password": os.getenv("PG_PASSWORD", "demo_password"),
    "database": os.getenv("PG_DB_NAME", "demo"),
    "schema": os.getenv("PG_SCHEMA", "demo"),
}
if all(POSTGRES_CONFIG.values()):
    DATASOURCE_CONFIGS.append(
        {
            "engine": "postgres",
            "name_prefix": "test-pg-ds",
            "connection_data": POSTGRES_CONFIG,
            "sample_table": "house_sales",  # A known table in the demo PG database
        }
    )

# --- Snowflake Configuration (Only enabled if all credentials are provided) ---
SNOWFLAKE_CONFIG = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
}
if all(SNOWFLAKE_CONFIG.values()):
    DATASOURCE_CONFIGS.append(
        {
            "engine": "snowflake",
            "name_prefix": "test-sf-ds",
            "connection_data": SNOWFLAKE_CONFIG,
            "sample_table": "CUSTOMER",
        }
    )
