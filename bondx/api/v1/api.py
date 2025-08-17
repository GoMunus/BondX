"""
Main API router for BondX Backend v1.

This module includes all API endpoint routers for the bond marketplace.
"""

from fastapi import APIRouter

from .bonds import router as bonds_router
from .market import router as market_router
from .analytics import router as analytics_router
from .issuers import router as issuers_router
from .ratings import router as ratings_router
from .yield_curves import router as yield_curves_router
from .ai import router as ai_router
from .monitoring import router as monitoring_router
from .auctions import router as auctions_router
from .trading import router as trading_router
from .risk_management import router as risk_management_router

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    bonds_router,
    prefix="/bonds",
    tags=["bonds"]
)
api_router.include_router(
    market_router,
    prefix="/market",
    tags=["market"]
)
api_router.include_router(
    analytics_router,
    prefix="/analytics",
    tags=["analytics"]
)
api_router.include_router(
    issuers_router,
    prefix="/issuers",
    tags=["issuers"]
)
api_router.include_router(
    ratings_router,
    prefix="/ratings",
    tags=["ratings"]
)
api_router.include_router(
    yield_curves_router,
    prefix="/yield-curves",
    tags=["yield-curves"]
)

# Include AI router
api_router.include_router(
    ai_router,
    prefix="/ai",
    tags=["ai"]
)

# Include monitoring router
api_router.include_router(
    monitoring_router,
    prefix="/monitoring",
    tags=["monitoring"]
)

# Include auctions router
api_router.include_router(
    auctions_router,
    prefix="/auctions",
    tags=["auctions"]
)

# Include trading router
api_router.include_router(
    trading_router,
    prefix="/trading",
    tags=["trading"]
)

# Include risk management router
api_router.include_router(
    risk_management_router,
    prefix="/risk",
    tags=["risk_management"]
)

# Export the main router
__all__ = ["api_router"]
