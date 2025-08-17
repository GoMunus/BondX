"""
Database base configuration for BondX Backend.

This module sets up SQLAlchemy with async support, connection pooling,
and session management for the bond pricing engine.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)

# Database metadata with naming conventions
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)

# Create declarative base
Base = declarative_base(metadata=metadata)

# Database engine configuration
engine = create_async_engine(
    settings.get_database_url(),
    echo=settings.is_development(),
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    pool_timeout=settings.database.pool_timeout,
    pool_recycle=settings.database.pool_recycle,
    pool_pre_ping=True,
    poolclass=NullPool if settings.is_testing() else None,
    future=True,
)

# Session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session.
    
    Yields:
        AsyncSession: Database session instance
        
    Example:
        async with get_session() as session:
            result = await session.execute(query)
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Database session error", error=str(e))
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session context manager.
    
    Yields:
        AsyncSession: Database session instance
        
    Example:
        async with get_session_context() as session:
            result = await session.execute(query)
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Database session error", error=str(e))
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables."""
    try:
        async with engine.begin() as conn:
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise


async def close_db() -> None:
    """Close database connections."""
    try:
        await engine.dispose()
        logger.info("Database connections closed successfully")
    except Exception as e:
        logger.error("Failed to close database connections", error=str(e))
        raise


async def check_db_connection() -> bool:
    """
    Check database connection health.
    
    Returns:
        bool: True if connection is healthy, False otherwise
    """
    try:
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error("Database connection check failed", error=str(e))
        return False


async def get_db_stats() -> dict:
    """
    Get database statistics and health metrics.
    
    Returns:
        dict: Database statistics
    """
    try:
        async with engine.begin() as conn:
            # Get connection pool info
            pool_info = {
                "pool_size": engine.pool.size(),
                "checked_in": engine.pool.checkedin(),
                "checked_out": engine.pool.checkedout(),
                "overflow": engine.pool.overflow(),
            }
            
            # Get database version
            result = await conn.execute("SELECT version()")
            version = result.scalar()
            
            return {
                "status": "healthy",
                "version": version,
                "pool_info": pool_info,
                "connection_count": engine.pool.size() + engine.pool.overflow(),
            }
    except Exception as e:
        logger.error("Failed to get database stats", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "pool_info": {},
            "connection_count": 0,
        }


# Database health check function for FastAPI
async def health_check() -> dict:
    """
    Health check function for FastAPI health endpoint.
    
    Returns:
        dict: Health status information
    """
    is_healthy = await check_db_connection()
    stats = await get_db_stats()
    
    return {
        "database": {
            "status": "healthy" if is_healthy else "unhealthy",
            "details": stats,
        },
        "overall_status": "healthy" if is_healthy else "unhealthy",
        "timestamp": asyncio.get_event_loop().time(),
    }


# Export commonly used functions and classes
__all__ = [
    "Base",
    "engine",
    "async_session_factory",
    "get_session",
    "get_session_context",
    "init_db",
    "close_db",
    "check_db_connection",
    "get_db_stats",
    "health_check",
    "metadata",
]
