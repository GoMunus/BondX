"""
Main FastAPI application for BondX Backend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import routers
from bondx.api.v1.api import api_router

# Create FastAPI app
app = FastAPI(
    title="BondX Backend",
    version="1.0.0",
    description="AI-powered fractional bond marketplace backend"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "BondX Backend",
        "version": "1.0.0",
        "description": "AI-powered fractional bond marketplace backend",
        "endpoints": {
            "api": "/api/v1",
            "docs": "/docs",
            "health": "/health"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "BondX Backend is running"
    }
