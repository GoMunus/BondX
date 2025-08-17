"""
Logging configuration module for BondX Backend.

This module sets up structured logging with proper formatting,
file rotation, and integration with monitoring systems.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from structlog.stdlib import LoggerFactory

from .config import settings


def setup_logging() -> None:
    """
    Set up structured logging for the application.
    
    Configures logging with proper formatting, file rotation,
    and integration with monitoring systems.
    """
    # Create logs directory if it doesn't exist
    if settings.logging.file_path:
        log_path = Path(settings.logging.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.logging.level.upper())
    )
    
    # Set up file logging if configured
    if settings.logging.file_path:
        setup_file_logging()
    
    # Set up external logging if configured
    if settings.monitoring.sentry_dsn:
        setup_sentry_logging()
    
    # Configure third-party loggers
    configure_third_party_loggers()
    
    # Log application startup
    logger = structlog.get_logger(__name__)
    logger.info(
        "Logging configured successfully",
        level=settings.logging.level,
        format=settings.logging.format,
        file_path=settings.logging.file_path,
        environment=settings.environment
    )


def setup_file_logging() -> None:
    """Set up file logging with rotation."""
    if not settings.logging.file_path:
        return
    
    # Parse max size (e.g., "100MB" -> 100 * 1024 * 1024)
    max_size_str = settings.logging.max_size
    if max_size_str.endswith("MB"):
        max_size = int(max_size_str[:-2]) * 1024 * 1024
    elif max_size_str.endswith("KB"):
        max_size = int(max_size_str[:-2]) * 1024
    else:
        max_size = int(max_size_str)
    
    # Create rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        filename=settings.logging.file_path,
        maxBytes=max_size,
        backupCount=settings.logging.backup_count,
        encoding="utf-8"
    )
    
    # Set formatter
    if settings.logging.format == "json":
        formatter = logging.Formatter("%(message)s")
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    file_handler.setFormatter(formatter)
    
    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)


def setup_sentry_logging() -> None:
    """Set up Sentry logging for error tracking."""
    try:
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration
        
        # Configure Sentry
        sentry_logging = LoggingIntegration(
            level=logging.INFO,
            event_level=logging.ERROR
        )
        
        sentry_sdk.init(
            dsn=settings.monitoring.sentry_dsn,
            integrations=[sentry_logging],
            environment=settings.environment,
            traces_sample_rate=0.1 if settings.is_production() else 1.0,
            profiles_sample_rate=0.1 if settings.is_production() else 1.0,
        )
        
        logger = structlog.get_logger(__name__)
        logger.info("Sentry logging configured successfully")
        
    except ImportError:
        logger = structlog.get_logger(__name__)
        logger.warning("Sentry SDK not installed, skipping Sentry configuration")


def configure_third_party_loggers() -> None:
    """Configure logging levels for third-party libraries."""
    # Reduce noise from verbose libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.WARNING)
    logging.getLogger("celery").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Set specific levels based on environment
    if settings.is_development():
        logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
        logging.getLogger("uvicorn.access").setLevel(logging.INFO)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structured logger
    """
    return structlog.get_logger(name)


def log_function_call(
    logger: structlog.BoundLogger,
    func_name: str,
    args: tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    result: Any = None,
    error: Optional[Exception] = None,
    execution_time: Optional[float] = None
) -> None:
    """
    Log function call details for debugging and monitoring.
    
    Args:
        logger: Logger instance
        func_name: Name of the function being called
        args: Function arguments
        kwargs: Function keyword arguments
        result: Function result
        error: Exception if function failed
        execution_time: Function execution time in seconds
    """
    log_data = {
        "function": func_name,
        "args": args,
        "kwargs": kwargs or {},
        "execution_time": execution_time
    }
    
    if error:
        log_data["error"] = str(error)
        log_data["error_type"] = type(error).__name__
        logger.error("Function call failed", **log_data)
    else:
        log_data["result"] = result
        logger.debug("Function call completed", **log_data)


def log_database_operation(
    logger: structlog.BoundLogger,
    operation: str,
    table: str,
    query: str,
    execution_time: float,
    rows_affected: Optional[int] = None,
    error: Optional[Exception] = None
) -> None:
    """
    Log database operation details for monitoring and debugging.
    
    Args:
        logger: Logger instance
        operation: Type of operation (SELECT, INSERT, UPDATE, DELETE)
        table: Target table name
        query: SQL query or query description
        execution_time: Query execution time in seconds
        rows_affected: Number of rows affected
        error: Exception if operation failed
    """
    log_data = {
        "operation": operation,
        "table": table,
        "query": query,
        "execution_time": execution_time,
        "rows_affected": rows_affected
    }
    
    if error:
        log_data["error"] = str(error)
        log_data["error_type"] = type(error).__name__
        logger.error("Database operation failed", **log_data)
    else:
        logger.info("Database operation completed", **log_data)


def log_api_request(
    logger: structlog.BoundLogger,
    method: str,
    path: str,
    status_code: int,
    execution_time: float,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    error: Optional[Exception] = None
) -> None:
    """
    Log API request details for monitoring and audit trails.
    
    Args:
        logger: Logger instance
        method: HTTP method
        path: Request path
        status_code: HTTP status code
        execution_time: Request execution time in seconds
        user_id: Authenticated user ID
        ip_address: Client IP address
        user_agent: Client user agent
        error: Exception if request failed
    """
    log_data = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "execution_time": execution_time,
        "user_id": user_id,
        "ip_address": ip_address,
        "user_agent": user_agent
    }
    
    if error:
        log_data["error"] = str(error)
        log_data["error_type"] = type(error).__name__
        logger.error("API request failed", **log_data)
    else:
        logger.info("API request completed", **log_data)


def log_market_data_update(
    logger: structlog.BoundLogger,
    source: str,
    instrument_count: int,
    update_time: str,
    execution_time: float,
    error: Optional[Exception] = None
) -> None:
    """
    Log market data update details for monitoring.
    
    Args:
        logger: Logger instance
        source: Data source name (NSE, BSE, RBI, etc.)
        instrument_count: Number of instruments updated
        update_time: Timestamp of the update
        execution_time: Update execution time in seconds
        error: Exception if update failed
    """
    log_data = {
        "source": source,
        "instrument_count": instrument_count,
        "update_time": update_time,
        "execution_time": execution_time
    }
    
    if error:
        log_data["error"] = str(error)
        log_data["error_type"] = type(error).__name__
        logger.error("Market data update failed", **log_data)
    else:
        logger.info("Market data update completed", **log_data)


def log_bond_calculation(
    logger: structlog.BoundLogger,
    isin: str,
    calculation_type: str,
    parameters: Dict[str, Any],
    result: Any,
    execution_time: float,
    error: Optional[Exception] = None
) -> None:
    """
    Log bond calculation details for debugging and performance monitoring.
    
    Args:
        logger: Logger instance
        isin: Bond ISIN code
        calculation_type: Type of calculation (price, yield, duration, etc.)
        parameters: Calculation parameters
        result: Calculation result
        execution_time: Calculation execution time in seconds
        error: Exception if calculation failed
    """
    log_data = {
        "isin": isin,
        "calculation_type": calculation_type,
        "parameters": parameters,
        "result": result,
        "execution_time": execution_time
    }
    
    if error:
        log_data["error"] = str(error)
        log_data["error_type"] = type(error).__name__
        logger.error("Bond calculation failed", **log_data)
    else:
        logger.debug("Bond calculation completed", **log_data)


# Export commonly used functions
__all__ = [
    "setup_logging",
    "get_logger",
    "log_function_call",
    "log_database_operation",
    "log_api_request",
    "log_market_data_update",
    "log_bond_calculation",
]
