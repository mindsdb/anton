"""Database connection and session management for PostgreSQL using SQLAlchemy"""

from enum import Enum

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session as SQLModelSession

from minds.common.logger import setup_logging
from minds.common.vars import (
    DATABASE_URI,
    DB_MAX_OVERFLOW,
    DB_POOL_PRE_PING,
    DB_POOL_RECYCLE,
    DB_POOL_SIZE,
    DB_POOL_TIMEOUT,
)

logger = setup_logging()

# Global cache for engines and session factories
_engines = {}
_session_factories = {}


class DatabaseURI(str, Enum):
    """Enum for database URIs."""

    DEFAULT = DATABASE_URI


def _create_engine(db_uri: DatabaseURI):
    """Create a proper SQLAlchemy engine that connects to a PostgreSQL database.

    Args:
        db_uri: Database URI.

    Returns:
        SQLAlchemy engine instance
    """
    # Log connection string with a hidden password
    name = db_uri.name

    hidden_uri = db_uri.replace(":", ":*****@", 1) if "@" in db_uri else db_uri
    logger.debug(f"Creating PostgreSQL engine '{name}' with: {hidden_uri}")

    try:
        # Create a regular SQLAlchemy engine
        engine = create_engine(
            db_uri,
            pool_size=DB_POOL_SIZE,
            max_overflow=DB_MAX_OVERFLOW,
            pool_timeout=DB_POOL_TIMEOUT,
            pool_recycle=DB_POOL_RECYCLE,
            pool_pre_ping=DB_POOL_PRE_PING,
        )
        # logger.info(f"PostgreSQL engine '{name}' created successfully")
        return engine
    except Exception as e:
        error_msg = str(e).lower()
        logger.error(f"Failed to create PostgreSQL engine '{name}': {error_msg}")
        raise RuntimeError(f"PostgreSQL engine creation failed: {error_msg}") from e


def get_engine(db_uri: DatabaseURI):
    """Get or create a database engine for the specified connection URI.

    Args:
        db_uri: Database connection URI

    Returns:
        SQLAlchemy engine instance
    """
    name = db_uri.name
    if name not in _engines:
        _engines[name] = _create_engine(db_uri=db_uri)
    return _engines[name]


def get_session_factory(engine):
    """Get or create a session factory for the specified engine.

    Args:
        engine: SQLAlchemy engine instance

    Returns:
        SQLAlchemy session maker instance
    """
    engine_id = id(engine)
    if engine_id not in _session_factories:
        _session_factories[engine_id] = sessionmaker(
            class_=SQLModelSession,  # <- SQLModel-compatible session
            autocommit=False,
            autoflush=False,
            bind=engine,
            expire_on_commit=False,
        )
    return _session_factories[engine_id]


def get_session(db_uri: DatabaseURI = DatabaseURI.DEFAULT):
    """
    FastAPI dependency that provides a database session with automatic cleanup.

    This is a generator-based dependency that ensures sessions are always closed
    after the request completes, preventing database connection leaks.

    Usage:
        @router.get("/example")
        async def endpoint(session: Session = Depends(get_session)):
            # Session will be automatically closed after this function completes
            pass

    Args:
        db_uri: Database connection URI

    Yields:
        SQLModelSession: Database session that will be automatically closed
    """
    engine = get_engine(db_uri=db_uri)
    session_factory = get_session_factory(engine)

    db = session_factory()
    try:
        logger.debug(f"🔗 Created database session for '{db_uri.name}'")
        yield db
    except Exception as e:
        logger.error(f"❌ Session error for '{db_uri.name}': {str(e)}")
        db.rollback()
        raise
    finally:
        logger.debug(f"🔒 Closing database session for '{db_uri.name}'")
        db.close()
