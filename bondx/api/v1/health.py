"""
Health check endpoints for BondX Backend.

This module provides health check and monitoring endpoints for the system.
"""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ...core.config import settings
from ...core.logging import get_logger
from ...database.base import health_check as db_health_check

logger = get_logger(__name__)

# Create health router
router = APIRouter()


@router.get("/", tags=["health"])
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment
    }


@router.get("/detailed", tags=["health"])
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check including all system components.
    
    Returns:
        Detailed health status for all components
    """
    try:
        # Check database health
        db_health = await db_health_check()
        
        # Overall health status
        overall_status = "healthy"
        if db_health["overall_status"] != "healthy":
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "components": {
                "database": db_health["database"],
                "redis": {
                    "status": "not_implemented",
                    "message": "Redis health check not yet implemented"
                },
                "celery": {
                    "status": "not_implemented",
                    "message": "Celery health check not yet implemented"
                }
            },
            "uptime": "not_implemented"  # Would need to track application start time
        }
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "error": str(e),
            "components": {
                "database": {
                    "status": "unhealthy",
                    "error": str(e)
                }
            }
        }


@router.get("/ready", tags=["health"])
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check for the application.
    
    This endpoint checks if the application is ready to serve requests.
    
    Returns:
        Readiness status
    """
    try:
        # Check if database is accessible
        db_health = await db_health_check()
        
        if db_health["overall_status"] == "healthy":
            return {
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Application is ready to serve requests"
            }
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": "Application is not ready to serve requests",
                    "reason": "Database is not healthy"
                }
            )
            
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Application is not ready to serve requests",
                "reason": f"Health check failed: {str(e)}"
            }
        )


@router.get("/live", tags=["health"])
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check for the application.
    
    This endpoint checks if the application is alive and running.
    
    Returns:
        Liveness status
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "message": "Application is alive and running"
    }


@router.get("/info", tags=["health"])
async def system_info() -> Dict[str, Any]:
    """
    System information endpoint.
    
    Returns:
        System configuration and information
    """
    return {
        "service": {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "debug": settings.debug
        },
        "server": {
            "host": settings.host,
            "port": settings.port,
            "workers": settings.workers
        },
        "database": {
            "pool_size": settings.database.pool_size,
            "max_overflow": settings.database.max_overflow,
            "pool_timeout": settings.database.pool_timeout
        },
        "cache": {
            "ttl": settings.cache.ttl,
            "max_size": settings.cache.max_size
        },
        "rate_limiting": {
            "requests_per_window": settings.rate_limit.requests,
            "window_seconds": settings.rate_limit.window
        },
        "monitoring": {
            "prometheus_enabled": settings.monitoring.prometheus_enabled,
            "health_check_interval": settings.monitoring.health_check_interval
        }
    }


# Export the router
__all__ = ["router"]
