"""
Main API router for BondX Backend v1.

This module includes all API endpoint routers for the bond marketplace.
"""

from fastapi import APIRouter

# Import only the routers that exist
from .dashboard import router as dashboard_router
from .fractional_bonds import router as fractional_bonds_router
from .simple_auctions import router as simple_auctions_router

# Create main API router
api_router = APIRouter()

# Include only working routers
api_router.include_router(
    dashboard_router,
    prefix="/dashboard",
    tags=["dashboard"]
)

api_router.include_router(
    fractional_bonds_router,
    prefix="/bonds",
    tags=["fractional_bonds"]
)

api_router.include_router(
    simple_auctions_router,
    prefix="/auctions",
    tags=["auctions"]
)
