import logging
import sys
import os
from pathlib import Path
import time
import functools
from datetime import datetime

try:
    import colorlog

    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False

try:
    from rich.logging import RichHandler
    from rich.console import Console

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class EventLoopClosedFilter(logging.Filter):
    """Filter out 'Event loop is closed' errors that occur during cleanup"""

    def filter(self, record):
        # Filter out the specific RuntimeError about event loop being closed
        if record.name == "asyncio" and record.levelno == logging.ERROR:
            if "Event loop is closed" in record.getMessage():
                return False
        return True


class CustomFormatter(logging.Formatter):
    """Custom formatter with enhanced formatting"""

    def format(self, record):
        # Add extra context to the record
        if hasattr(record, "user_id"):
            record.user_context = f"[User:{record.user_id}]"
        else:
            record.user_context = ""

        if hasattr(record, "request_id"):
            record.request_context = f"[Req:{record.request_id}]"
        else:
            record.request_context = ""

        return super().format(record)


def get_colored_formatter():
    """Get a colored formatter if colorlog is available"""
    if not HAS_COLORLOG:
        return logging.Formatter(
            "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    return colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)8s] %(name)s: %(message)s%(reset)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={
            "message": {
                "DEBUG": "white",
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red",
            }
        },
    )


def setup_file_logging(
    log_dir: str = "logs", max_bytes: int = 10485760, backup_count: int = 5
):
    """Setup file logging with rotation"""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Create handlers for different log levels
    handlers = []

    # All logs file
    all_logs_handler = logging.handlers.RotatingFileHandler(
        log_path / "cw_summary.log", maxBytes=max_bytes, backupCount=backup_count
    )
    all_logs_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)8s] %(name)s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handlers.append(all_logs_handler)

    # Error logs file
    error_handler = logging.handlers.RotatingFileHandler(
        log_path / "errors.log", maxBytes=max_bytes, backupCount=backup_count
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)8s] %(name)s [%(filename)s:%(lineno)d] %(message)s\n%(stack_info)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handlers.append(error_handler)

    return handlers


def setup_console_handler():
    """Setup console handler with colors and rich formatting if available"""
    if HAS_RICH and os.getenv("RICH_LOGGING", "false").lower() == "true":
        console = Console(stderr=True)
        handler = RichHandler(
            console=console,
            show_path=False,
            show_time=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        handler.setFormatter(
            logging.Formatter("[%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(get_colored_formatter())

    return handler


def setup_logging():
    """Setup comprehensive logging configuration"""
    log_level_str = os.getenv("LOG_LEVEL", "INFO")
    enable_file_logging = os.getenv("ENABLE_FILE_LOGGING", "false").lower() == "true"
    log_dir = os.getenv("LOG_DIR", "logs")

    # Map string log level to logging constants
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    # Default to INFO if an invalid level is provided
    log_level = log_level_map.get(log_level_str.upper(), logging.INFO)

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Setup handlers
    handlers = []

    # Console handler
    console_handler = setup_console_handler()
    console_handler.setLevel(log_level)
    handlers.append(console_handler)

    # File handlers (if enabled)
    if enable_file_logging:
        file_handlers = setup_file_logging(log_dir)
        for handler in file_handlers:
            handler.setLevel(log_level)
            handlers.append(handler)

    # Configure root logger
    logging.basicConfig(level=log_level, handlers=handlers, force=True)

    # Suppress verbose logging from third-party libraries
    third_party_loggers = [
        "httpcore.http11",
        "openai._base_client",
        "httpcore.connection",
        "httpx",
        "urllib3",
        "faiss",
        "asyncio",
        "requests",
        "boto3",
        "botocore",
        "s3transfer",
        "transformers",
        "torch",
        "tensorflow",
        "urllib3.connectionpool",
    ]

    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # Add filter to suppress "Event loop is closed" errors during cleanup
    event_loop_filter = EventLoopClosedFilter()
    logging.getLogger("asyncio").addFilter(event_loop_filter)

    # Create application logger
    logger = logging.getLogger(__name__)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name"""
    return logging.getLogger(name)


def log_performance(func):
    """Decorator to log function performance"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"⏱️  {func.__name__} completed in {execution_time:.5f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"❌ {func.__name__} failed after {execution_time:.5f}s: {str(e)}"
            )
            raise

    return wrapper


# Add logging import for rotating file handler
import logging.handlers
