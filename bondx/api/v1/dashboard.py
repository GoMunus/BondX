"""
Dashboard API endpoints for BondX Frontend Integration.

This module provides REST API endpoints specifically designed for the frontend dashboard,
including portfolio summaries, market status, risk metrics, and trading activity.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Query

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# Mock data for demonstration
MOCK_PORTFOLIO_DATA = {
    "total_aum": 1250000000,
    "daily_pnl": 1250000,
    "mtd_return": 2.8,
    "qtd_return": 8.5,
    "ytd_return": 12.3,
    "allocation": {
        "corporate_bonds": 45.2,
        "government_securities": 32.1,
        "money_market": 15.7,
        "other": 7.0
    }
}

MOCK_MARKET_DATA = {
    "status": "open",
    "last_update": datetime.now().isoformat(),
    "indices": {
        "nifty_bond": 1250.5,
        "bse_bond": 1180.2
    },
    "volatility": "medium"
}

MOCK_RISK_DATA = {
    "var_95": 2.8,
    "var_99": 4.2,
    "duration": 5.2,
    "convexity": 28.5,
    "liquidity_score": 0.75
}

MOCK_TRADING_DATA = [
    {
        "id": "T001",
        "bond_name": "Reliance Industries 7.5% 2030",
        "type": "buy",
        "amount": 1000000,
        "price": 98.5,
        "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat()
    },
    {
        "id": "T002", 
        "bond_name": "HDFC Bank 6.8% 2028",
        "type": "sell",
        "amount": 750000,
        "price": 99.2,
        "timestamp": (datetime.now() - timedelta(minutes=12)).isoformat()
    }
]

@router.get("/summary")
async def get_dashboard_summary(
    user_id: Optional[str] = Query(None, description="User ID for personalized summary"),
    portfolio_id: Optional[str] = Query(None, description="Specific portfolio ID")
):
    """Get comprehensive dashboard summary including all key metrics."""
    return {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "portfolio": MOCK_PORTFOLIO_DATA,
            "market": MOCK_MARKET_DATA,
            "risk": MOCK_RISK_DATA,
            "trading": MOCK_TRADING_DATA
        }
    }

@router.get("/portfolio-summary")
async def get_portfolio_summary(
    user_id: Optional[str] = Query(None, description="User ID for personalized summary")
):
    """Get portfolio summary data."""
    return MOCK_PORTFOLIO_DATA

@router.get("/market-status")
async def get_market_status():
    """Get current market status."""
    return MOCK_MARKET_DATA

@router.get("/risk-metrics")
async def get_risk_metrics(
    user_id: Optional[str] = Query(None, description="User ID for personalized metrics")
):
    """Get portfolio risk metrics."""
    return MOCK_RISK_DATA

@router.get("/trading-activity")
async def get_trading_activity(
    limit: int = Query(10, description="Number of recent trades to return")
):
    """Get recent trading activity."""
    return MOCK_TRADING_DATA[:limit]

@router.get("/system-health")
async def get_system_health():
    """Get system health status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "operational",
            "database": "operational", 
            "websocket": "operational",
            "risk_engine": "operational"
        }
    }
