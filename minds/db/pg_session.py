"""Database connection and session management for PostgreSQL using SQLAlchemy"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session as SQLModelSession

from minds.common.logger import setup_logging
from minds.common.settings.app_settings import get_app_settings

logger = setup_logging()
settings = get_app_settings()

# Global cache for engines and session factories
_engines = {}
_session_factories = {}


def _create_engine(db_uri: str):
    """Create a proper SQLAlchemy engine that connects to a PostgreSQL database.

    Args:
        db_uri: Database URI.

    Returns:
        SQLAlchemy engine instance
    """

    # Log connection string with a hidden password
    hidden_uri = db_uri.replace(":", ":*****@", 1) if "@" in db_uri else db_uri
    logger.debug(f"Creating PostgreSQL engine with: {hidden_uri}")

    try:
        # Create a regular SQLAlchemy engine
        engine = create_engine(
            db_uri,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_timeout=settings.database.pool_timeout,
            pool_recycle=settings.database.pool_recycle,
            pool_pre_ping=settings.database.pool_pre_ping,
        )
        # logger.info(f"PostgreSQL engine created successfully")
        return engine
    except Exception as e:
        error_msg = str(e).lower()
        logger.error(f"Failed to create PostgreSQL engine: {error_msg}")
        raise RuntimeError(f"PostgreSQL engine creation failed: {error_msg}") from e


def get_engine(db_uri: str):
    """Get or create a database engine for the specified connection URI.

    Args:
        db_uri: Database connection URI

    Returns:
        SQLAlchemy engine instance
    """
    if db_uri not in _engines:
        _engines[db_uri] = _create_engine(db_uri=db_uri)
    return _engines[db_uri]


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


def get_session(db_uri: str = settings.database.uri):
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
         db_uri: Database connection URI (defaults to settings value)

    Yields:
        SQLModelSession: Database session that will be automatically closed
    """

    engine = get_engine(db_uri=db_uri)
    session_factory = get_session_factory(engine)

    db = session_factory()
    try:
        logger.debug("🔗 Created database session")
        yield db
    except Exception as e:
        logger.error(f"❌ Session error: {str(e)}")
        db.rollback()
        raise
    finally:
        logger.debug("🔒 Closing database session")
        db.close()


def get_open_session(db_uri: str = settings.database.uri):
    """
    Get an open SQLModel session.

    Args:
        db_uri: Database connection URI (defaults to settings value)

    Returns:
        SQLModelSession: Open SQLModel session
    """
    engine = get_engine(db_uri=db_uri)
    session_factory = get_session_factory(engine)
    return session_factory()
