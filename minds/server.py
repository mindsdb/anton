"""
Minds FastAPI Application Server.

This module sets up the FastAPI application with middleware, routing,
and all necessary configurations for the Minds service.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from minds.api.v1.router import api_router as v1_router
from minds.common.logger import setup_logging

# Set up logging
logger = setup_logging()


# Create FastAPI app
app = FastAPI(
    title="Minds API",
    description="FastAPI-based service providing OpenAI-compatible chat completions with MindsDB integration",
    version="1.0.0",
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any origin. This will be controlled by the ingress controller.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Include v1 API routes
app.include_router(v1_router)

logger.info("Minds application created successfully")
