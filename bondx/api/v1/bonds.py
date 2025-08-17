"""
Bonds API endpoints for BondX Backend.

This module provides comprehensive bond operations including:
- Bond information retrieval
- Pricing calculations
- Yield calculations
- Duration and convexity analytics
- Cash flow projections
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...core.logging import get_logger
from ...database.base import get_session
from ...database.models import Instrument, MarketQuote, DayCountConvention
from ...mathematics.bond_pricing import BondPricingEngine
from ...mathematics.yield_calculations import YieldCalculator
from ...mathematics.duration_convexity import DurationCalculator

logger = get_logger(__name__)

# Create bonds router
router = APIRouter()

# Initialize calculators
bond_pricing_engine = BondPricingEngine()
yield_calculator = YieldCalculator()
duration_calculator = DurationCalculator()


# Pydantic models for request/response
class BondInfo(BaseModel):
    """Bond information model."""
    
    isin: str = Field(..., description="Bond ISIN code")
    name: str = Field(..., description="Bond name")
    short_name: Optional[str] = Field(None, description="Short name")
    bond_type: str = Field(..., description="Type of bond")
    coupon_type: str = Field(..., description="Coupon structure type")
    face_value: Decimal = Field(..., description="Face value")
    coupon_rate: Optional[Decimal] = Field(None, description="Annual coupon rate")
    maturity_date: date = Field(..., description="Maturity date")
    day_count_convention: str = Field(..., description="Day count convention")
    issuer_name: str = Field(..., description="Issuer name")
    market_status: str = Field(..., description="Market status")


class BondPricingRequest(BaseModel):
    """Request model for bond pricing calculations."""
    
    settlement_date: date = Field(..., description="Settlement date")
    yield_rate: Optional[Decimal] = Field(None, description="Yield rate for pricing")
    day_count_convention: Optional[str] = Field(None, description="Day count convention to use")


class BondPricingResponse(BaseModel):
    """Response model for bond pricing calculations."""
    
    isin: str = Field(..., description="Bond ISIN code")
    clean_price: Decimal = Field(..., description="Clean price")
    dirty_price: Decimal = Field(..., description="Dirty price")
    accrued_interest: Decimal = Field(..., description="Accrued interest")
    yield_rate: Optional[Decimal] = Field(None, description="Yield rate used")
    calculation_date: datetime = Field(..., description="Calculation timestamp")
    day_count_convention: str = Field(..., description="Day count convention used")


class YieldCalculationRequest(BaseModel):
    """Request model for yield calculations."""
    
    clean_price: Decimal = Field(..., description="Clean price of the bond")
    settlement_date: date = Field(..., description="Settlement date")
    day_count_convention: Optional[str] = Field(None, description="Day count convention to use")


class YieldCalculationResponse(BaseModel):
    """Response model for yield calculations."""
    
    isin: str = Field(..., description="Bond ISIN code")
    yield_to_maturity: Decimal = Field(..., description="Yield to maturity")
    current_yield: Decimal = Field(..., description="Current yield")
    yield_spread: Decimal = Field(..., description="Yield spread over coupon")
    calculation_date: datetime = Field(..., description="Calculation timestamp")
    calculation_method: str = Field(..., description="Method used for calculation")
    convergence_achieved: bool = Field(..., description="Whether calculation converged")


class DurationResponse(BaseModel):
    """Response model for duration calculations."""
    
    isin: str = Field(..., description="Bond ISIN code")
    modified_duration: Decimal = Field(..., description="Modified duration")
    macaulay_duration: Decimal = Field(..., description="Macaulay duration")
    convexity: Decimal = Field(..., description="Convexity")
    calculation_date: datetime = Field(..., description="Calculation timestamp")


@router.get("/", tags=["bonds"])
async def list_bonds(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    bond_type: Optional[str] = Query(None, description="Filter by bond type"),
    issuer_sector: Optional[str] = Query(None, description="Filter by issuer sector"),
    min_maturity: Optional[date] = Query(None, description="Minimum maturity date"),
    max_maturity: Optional[date] = Query(None, description="Maximum maturity date"),
    market_status: Optional[str] = Query(None, description="Filter by market status"),
    search: Optional[str] = Query(None, description="Search in bond name or ISIN")
) -> Dict[str, Any]:
    """
    List bonds with filtering and pagination.
    
    Args:
        skip: Number of records to skip for pagination
        limit: Maximum number of records to return
        bond_type: Filter by bond type (e.g., GOVERNMENT_SECURITY, CORPORATE_BOND)
        issuer_sector: Filter by issuer sector (e.g., BANKING, INFRASTRUCTURE)
        min_maturity: Minimum maturity date
        max_maturity: Maximum maturity date
        market_status: Filter by market status (e.g., ACTIVE, INACTIVE)
        search: Search term for bond name or ISIN
        
    Returns:
        Paginated list of bonds with metadata
    """
    try:
        # This would typically query the database
        # For now, return a placeholder response
        bonds = [
            {
                "isin": "INE001A01001",
                "name": "Government of India 7.17% 2023",
                "bond_type": "GOVERNMENT_SECURITY",
                "coupon_rate": 7.17,
                "maturity_date": "2023-12-31",
                "issuer_sector": "GOVERNMENT",
                "market_status": "ACTIVE"
            },
            {
                "isin": "INE002A01002",
                "name": "HDFC Bank 8.50% 2025",
                "bond_type": "CORPORATE_BOND",
                "coupon_rate": 8.50,
                "maturity_date": "2025-06-15",
                "issuer_sector": "BANKING",
                "market_status": "ACTIVE"
            }
        ]
        
        return {
            "bonds": bonds,
            "pagination": {
                "skip": skip,
                "limit": limit,
                "total": len(bonds),
                "has_more": False
            },
            "filters_applied": {
                "bond_type": bond_type,
                "issuer_sector": issuer_sector,
                "min_maturity": min_maturity.isoformat() if min_maturity else None,
                "max_maturity": max_maturity.isoformat() if max_maturity else None,
                "market_status": market_status,
                "search": search
            }
        }
        
    except Exception as e:
        logger.error("Failed to list bonds", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve bonds")


@router.get("/{isin}", tags=["bonds"])
async def get_bond_info(
    isin: str = Path(..., description="Bond ISIN code")
) -> BondInfo:
    """
    Get detailed information about a specific bond.
    
    Args:
        isin: Bond ISIN code
        
    Returns:
        Detailed bond information
        
    Raises:
        HTTPException: If bond not found
    """
    try:
        # This would typically query the database
        # For now, return a placeholder response
        if isin == "INE001A01001":
            return BondInfo(
                isin=isin,
                name="Government of India 7.17% 2023",
                short_name="GOI 7.17% 2023",
                bond_type="GOVERNMENT_SECURITY",
                coupon_type="FIXED",
                face_value=Decimal("100.00"),
                coupon_rate=Decimal("7.17"),
                maturity_date=date(2023, 12, 31),
                day_count_convention="ACT/ACT",
                issuer_name="Government of India",
                market_status="ACTIVE"
            )
        else:
            raise HTTPException(status_code=404, detail="Bond not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get bond info", isin=isin, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve bond information")


@router.post("/{isin}/price", tags=["bonds"])
async def calculate_bond_price(
    isin: str = Path(..., description="Bond ISIN code"),
    request: BondPricingRequest = ...,
) -> BondPricingResponse:
    """
    Calculate bond price given yield rate.
    
    Args:
        isin: Bond ISIN code
        request: Pricing calculation parameters
        
    Returns:
        Bond pricing information
        
    Raises:
        HTTPException: If calculation fails or bond not found
    """
    try:
        # This would typically use the bond pricing engine
        # For now, return a placeholder response
        if isin == "INE001A01001":
            return BondPricingResponse(
                isin=isin,
                clean_price=Decimal("98.50"),
                dirty_price=Decimal("99.25"),
                accrued_interest=Decimal("0.75"),
                yield_rate=request.yield_rate or Decimal("7.50"),
                calculation_date=datetime.utcnow(),
                day_count_convention=request.day_count_convention or "ACT/ACT"
            )
        else:
            raise HTTPException(status_code=404, detail="Bond not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to calculate bond price", isin=isin, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to calculate bond price")


@router.post("/{isin}/yield", tags=["bonds"])
async def calculate_bond_yield(
    isin: str = Path(..., description="Bond ISIN code"),
    request: YieldCalculationRequest = ...,
) -> YieldCalculationResponse:
    """
    Calculate yield-to-maturity for a bond.
    
    Args:
        isin: Bond ISIN code
        request: Yield calculation parameters
        
    Returns:
        Yield calculation results
        
    Raises:
        HTTPException: If calculation fails or bond not found
    """
    try:
        # This would typically use the yield calculator
        # For now, return a placeholder response
        if isin == "INE001A01001":
            return YieldCalculationResponse(
                isin=isin,
                yield_to_maturity=Decimal("7.50"),
                current_yield=Decimal("7.28"),
                yield_spread=Decimal("0.33"),
                calculation_date=datetime.utcnow(),
                calculation_method="newton_raphson",
                convergence_achieved=True
            )
        else:
            raise HTTPException(status_code=404, detail="Bond not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to calculate bond yield", isin=isin, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to calculate bond yield")


@router.get("/{isin}/duration", tags=["bonds"])
async def calculate_bond_duration(
    isin: str = Path(..., description="Bond ISIN code"),
    settlement_date: date = Query(..., description="Settlement date"),
    yield_rate: Optional[Decimal] = Query(None, description="Yield rate for calculation")
) -> DurationResponse:
    """
    Calculate duration and convexity for a bond.
    
    Args:
        isin: Bond ISIN code
        settlement_date: Settlement date for calculation
        yield_rate: Yield rate to use (optional, will use market yield if not provided)
        
    Returns:
        Duration and convexity metrics
        
    Raises:
        HTTPException: If calculation fails or bond not found
    """
    try:
        # This would typically use the duration calculator
        # For now, return a placeholder response
        if isin == "INE001A01001":
            return DurationResponse(
                isin=isin,
                modified_duration=Decimal("2.45"),
                macaulay_duration=Decimal("2.65"),
                convexity=Decimal("8.92"),
                calculation_date=datetime.utcnow()
            )
        else:
            raise HTTPException(status_code=404, detail="Bond not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to calculate bond duration", isin=isin, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to calculate bond duration")


@router.get("/{isin}/cash-flows", tags=["bonds"])
async def get_bond_cash_flows(
    isin: str = Path(..., description="Bond ISIN code"),
    settlement_date: date = Query(..., description="Settlement date")
) -> Dict[str, Any]:
    """
    Get projected cash flows for a bond.
    
    Args:
        isin: Bond ISIN code
        settlement_date: Settlement date
        
    Returns:
        Projected cash flows with timing and amounts
        
    Raises:
        HTTPException: If bond not found or calculation fails
    """
    try:
        # This would typically calculate cash flows
        # For now, return a placeholder response
        if isin == "INE001A01001":
            return {
                "isin": isin,
                "settlement_date": settlement_date.isoformat(),
                "cash_flows": [
                    {
                        "payment_date": "2023-06-30",
                        "payment_type": "COUPON",
                        "amount": 3.585,
                        "days_from_settlement": 180
                    },
                    {
                        "payment_date": "2023-12-31",
                        "payment_type": "COUPON_AND_PRINCIPAL",
                        "amount": 103.585,
                        "days_from_settlement": 365
                    }
                ],
                "total_coupon_payments": 7.17,
                "principal_payment": 100.00,
                "total_present_value": 99.25
            }
        else:
            raise HTTPException(status_code=404, detail="Bond not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get bond cash flows", isin=isin, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve bond cash flows")


@router.get("/{isin}/market-data", tags=["bonds"])
async def get_bond_market_data(
    isin: str = Path(..., description="Bond ISIN code"),
    days_back: int = Query(30, ge=1, le=365, description="Number of days of historical data")
) -> Dict[str, Any]:
    """
    Get market data for a bond including price history and trading information.
    
    Args:
        isin: Bond ISIN code
        days_back: Number of days of historical data to retrieve
        
    Returns:
        Market data including price history, volume, and trading statistics
        
    Raises:
        HTTPException: If bond not found or data unavailable
    """
    try:
        # This would typically query market data
        # For now, return a placeholder response
        if isin == "INE001A01001":
            return {
                "isin": isin,
                "data_period_days": days_back,
                "price_history": [
                    {
                        "date": "2023-01-01",
                        "clean_price": 98.50,
                        "dirty_price": 99.25,
                        "yield": 7.50,
                        "volume": 1000000
                    },
                    {
                        "date": "2023-01-02",
                        "clean_price": 98.75,
                        "dirty_price": 99.50,
                        "yield": 7.45,
                        "volume": 1200000
                    }
                ],
                "trading_statistics": {
                    "avg_daily_volume": 1100000,
                    "price_volatility": 0.15,
                    "bid_ask_spread": 0.05,
                    "last_trade_price": 98.75,
                    "last_trade_time": "2023-01-02T15:30:00Z"
                }
            }
        else:
            raise HTTPException(status_code=404, detail="Bond not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get bond market data", isin=isin, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve bond market data")


@router.get("/{isin}/analytics", tags=["bonds"])
async def get_bond_analytics(
    isin: str = Path(..., description="Bond ISIN code"),
    settlement_date: date = Query(..., description="Settlement date")
) -> Dict[str, Any]:
    """
    Get comprehensive analytics for a bond.
    
    Args:
        isin: Bond ISIN code
        settlement_date: Settlement date for calculations
        
    Returns:
        Comprehensive bond analytics including risk metrics
        
    Raises:
        HTTPException: If bond not found or calculation fails
    """
    try:
        # This would typically calculate comprehensive analytics
        # For now, return a placeholder response
        if isin == "INE001A01001":
            return {
                "isin": isin,
                "settlement_date": settlement_date.isoformat(),
                "pricing_analytics": {
                    "clean_price": 98.50,
                    "dirty_price": 99.25,
                    "accrued_interest": 0.75,
                    "yield_to_maturity": 7.50
                },
                "risk_metrics": {
                    "modified_duration": 2.45,
                    "macaulay_duration": 2.65,
                    "convexity": 8.92,
                    "price_value_of_basis_point": 0.0245
                },
                "income_metrics": {
                    "current_yield": 7.28,
                    "yield_to_maturity": 7.50,
                    "yield_to_call": None,
                    "yield_to_put": None
                },
                "market_metrics": {
                    "bid_ask_spread": 0.05,
                    "liquidity_score": 0.85,
                    "credit_spread": 0.33,
                    "option_adjusted_spread": None
                }
            }
        else:
            raise HTTPException(status_code=404, detail="Bond not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get bond analytics", isin=isin, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve bond analytics")


# Export the router
__all__ = ["router"]
