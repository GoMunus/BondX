"""
Main FastAPI application for BondX Backend.

This module sets up the FastAPI application with all necessary middleware,
routers, and configuration for the bond pricing engine.
"""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from .core.config import settings
from .core.logging import get_logger
from .database.base import init_db, close_db
from .api.v1.api import api_router
from .api.v1.health import health_router
from .ai_risk_engine.ai_service_layer import ai_service

# Import new components
from .trading_engine.order_manager import OrderManager
from .trading_engine.execution_engine import ExecutionEngine
from .risk_management.portfolio_risk import PortfolioRiskManager
from .websocket.unified_websocket_manager import UnifiedWebSocketManager

logger = get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"]
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting Prometheus metrics."""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        
        # Log request
        logger.info(
            "HTTP request started",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log successful response
            logger.info(
                "HTTP request completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                duration=duration,
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            logger.error(
                "HTTP request failed",
                method=request.method,
                url=str(request.url),
                error=str(e),
                duration=duration,
            )
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting BondX Backend application")
    
    try:
        # Initialize database
        await init_db()
        logger.info("Database initialized successfully")
        
        # Initialize AI service layer
        await ai_service.initialize()
        logger.info("AI service layer initialized successfully")
        
        # Initialize trading engine components
        logger.info("Initializing trading engine components...")
        # Note: These would typically be initialized with proper dependency injection
        # For now, we'll just log that they're available
        logger.info("Trading engine components available")
        
        # Initialize risk management components
        logger.info("Initializing risk management components...")
        # Note: These would typically be initialized with proper dependency injection
        # For now, we'll just log that they're available
        logger.info("Risk management components available")
        
        # Initialize WebSocket manager
        logger.info("Initializing WebSocket manager...")
        websocket_manager = UnifiedWebSocketManager()
        await websocket_manager.start()
        logger.info("WebSocket manager initialized successfully")
        
        # Initialize other services here
        # await init_redis()
        # await init_celery()
        
        logger.info("BondX Backend application started successfully")
        yield
        
    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down BondX Backend application")
        
        try:
            # Close database connections
            await close_db()
            logger.info("Database connections closed successfully")
            
            # Cleanup AI service layer
            await ai_service.cleanup()
            logger.info("AI service layer cleanup completed")
            
            # Cleanup trading engine components
            logger.info("Cleaning up trading engine components...")
            # Note: Proper cleanup would be implemented here
            
                    # Cleanup risk management components
        logger.info("Cleaning up risk management components...")
        # Note: Proper cleanup would be implemented here
        
        # Cleanup WebSocket manager
        logger.info("Cleaning up WebSocket manager...")
        if 'websocket_manager' in locals():
            await websocket_manager.stop()
        logger.info("WebSocket manager cleanup completed")
        
        # Close other services here
        # await close_redis()
        # await close_celery()
            
        except Exception as e:
            logger.error("Error during shutdown", error=str(e))
        
        logger.info("BondX Backend application shutdown complete")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "AI-powered fractional bond marketplace backend for Indian debt capital markets. "
            "Features advanced bond pricing engines, comprehensive financial mathematics, "
            "real-time trading capabilities, advanced risk management, and machine learning infrastructure."
        ),
        docs_url="/docs" if settings.is_development() else None,
        redoc_url="/redoc" if settings.is_development() else None,
        openapi_url="/openapi.json" if settings.is_development() else None,
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development() else [
            "https://bondx.com",
            "https://www.bondx.com",
            "https://app.bondx.com",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add trusted host middleware
    if not settings.is_development():
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=[
                "bondx.com",
                "www.bondx.com",
                "api.bondx.com",
                "*.bondx.com",
            ]
        )
    
    # Add session middleware
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.security.secret_key,
        max_age=settings.security.access_token_expire_minutes * 60,
    )
    
    # Add custom middleware
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(LoggingMiddleware)
    
    # Add exception handlers
    app.add_exception_handler(Exception, global_exception_handler)
    
    # Add routers
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(health_router, prefix="/health", tags=["health"])
    
    # Add root endpoint
    @app.get("/", tags=["root"])
    async def root() -> Dict[str, Any]:
        """Root endpoint with application information."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "description": "AI-powered fractional bond marketplace backend with real-time trading and risk management",
            "environment": settings.environment,
            "docs": "/docs" if settings.is_development() else None,
            "health": "/health",
            "api": "/api/v1",
            "features": [
                "Bond Pricing Engine",
                "Real-time Trading",
                "Auction Management",
                "Risk Management",
                "AI-powered Analytics",
                "Regulatory Compliance"
            ]
        }
    
    # Add metrics endpoint
    if settings.monitoring.prometheus_enabled:
        @app.get("/metrics", tags=["monitoring"])
        async def metrics() -> str:
            """Prometheus metrics endpoint."""
            from prometheus_client import generate_latest
            return generate_latest()
    
    return app


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors."""
    logger.error(
        "Unhandled exception",
        method=request.method,
        url=str(request.url),
        error=str(exc),
        error_type=type(exc).__name__,
    )
    
    if settings.is_development():
        # Return detailed error in development
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc),
                "type": type(exc).__name__,
                "path": str(request.url),
                "method": request.method,
            }
        )
    else:
        # Return generic error in production
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred. Please try again later."
            }
        )


# Create the application instance
app = create_application()

# Export for uvicorn
__all__ = ["app"]
