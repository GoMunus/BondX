"""
Portfolio analytics API endpoints for BondX Backend.

This module provides REST API endpoints for portfolio analytics including
risk metrics, performance attribution, and turnover analysis.
"""

import uuid
from datetime import datetime, date
from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from ...core.logging import get_logger
from ...core.model_contracts import ModelResultStore, ModelValidator, ModelType, ModelStatus
from ...risk_management.portfolio_analytics import (
    PortfolioAnalytics, PortfolioMetrics, AttributionResult, TurnoverMetrics,
    AttributionFactor, TenorBucket
)
from ...risk_management.stress_testing import Position, RatingBucket, SectorBucket
from ...mathematics.yield_curves import YieldCurve
from .schemas import (
    PositionSchema, PortfolioMetricsRequest, PortfolioMetricsResponse,
    AttributionRequest, AttributionResponse, TurnoverRequest, TurnoverResponse,
    ErrorResponse
)

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/portfolio-analytics", tags=["Portfolio Analytics"])

# Global instances
portfolio_analytics = PortfolioAnalytics(
    enable_pca=True,
    curve_factors=3,
    attribution_method="FACTOR_MODEL"
)
model_store = ModelResultStore(enable_caching=True, max_cache_size=1000)
model_validator = ModelValidator(strict_mode=False)


@router.post("/metrics", response_model=PortfolioMetricsResponse)
async def calculate_portfolio_metrics(
    request: PortfolioMetricsRequest,
    background_tasks: BackgroundTasks
) -> PortfolioMetricsResponse:
    """
    Calculate comprehensive portfolio risk metrics.
    
    Args:
        request: Portfolio metrics request
        background_tasks: FastAPI background tasks
        
    Returns:
        Portfolio metrics response
    """
    try:
        calculation_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Starting portfolio metrics calculation {calculation_id}")
        
        # Convert API request to engine inputs
        positions = await _convert_positions(request.positions)
        
        # Get yield curves if requested
        yield_curves = None
        if request.include_risk_metrics and request.yield_curves:
            yield_curves = await _get_yield_curves(request.yield_curves)
        
        # Get spread surfaces if provided
        spread_surfaces = None
        if request.spread_surfaces:
            spread_surfaces = await _convert_spread_surfaces(request.spread_surfaces)
        
        # Calculate portfolio metrics
        metrics = portfolio_analytics.calculate_portfolio_metrics(
            positions=positions,
            yield_curves=yield_curves,
            spread_surfaces=spread_surfaces
        )
        
        # Calculate execution time
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Create response
        response = PortfolioMetricsResponse(
            success=True,
            message="Portfolio metrics calculated successfully",
            timestamp=datetime.now(),
            data=metrics,
            calculation_id=calculation_id,
            execution_time_ms=execution_time_ms
        )
        
        logger.info(f"Portfolio metrics calculation {calculation_id} completed in {execution_time_ms:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in portfolio metrics calculation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Portfolio metrics calculation failed: {str(e)}"
        )


@router.post("/attribution", response_model=AttributionResponse)
async def calculate_performance_attribution(
    request: AttributionRequest,
    background_tasks: BackgroundTasks
) -> AttributionResponse:
    """
    Calculate performance attribution for a period.
    
    Args:
        request: Attribution request
        background_tasks: FastAPI background tasks
        
    Returns:
        Attribution response
    """
    try:
        calculation_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Starting performance attribution calculation {calculation_id}")
        
        # Convert API request to engine inputs
        positions_start = await _convert_positions(request.positions_start)
        positions_end = await _convert_positions(request.positions_end)
        
        # Get yield curves
        yield_curves_start = await _get_yield_curves(request.yield_curves_start)
        yield_curves_end = await _get_yield_curves(request.yield_curves_end)
        
        # Calculate attribution
        attribution = portfolio_analytics.calculate_performance_attribution(
            positions_start=positions_start,
            positions_end=positions_end,
            yield_curves_start=yield_curves_start,
            yield_curves_end=yield_curves_end,
            period_start=request.period_start,
            period_end=request.period_end,
            benchmark_returns=request.benchmark_returns
        )
        
        # Calculate execution time
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Create response
        response = AttributionResponse(
            success=True,
            message="Performance attribution calculated successfully",
            timestamp=datetime.now(),
            data=attribution,
            calculation_id=calculation_id,
            execution_time_ms=execution_time_ms
        )
        
        logger.info(f"Performance attribution calculation {calculation_id} completed in {execution_time_ms:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in performance attribution calculation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Performance attribution calculation failed: {str(e)}"
        )


@router.post("/turnover", response_model=TurnoverResponse)
async def calculate_turnover_metrics(
    request: TurnoverRequest,
    background_tasks: BackgroundTasks
) -> TurnoverResponse:
    """
    Calculate portfolio turnover metrics.
    
    Args:
        request: Turnover request
        background_tasks: FastAPI background tasks
        
    Returns:
        Turnover response
    """
    try:
        calculation_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Starting turnover metrics calculation {calculation_id}")
        
        # Convert API request to engine inputs
        positions_start = await _convert_positions(request.positions_start)
        positions_end = await _convert_positions(request.positions_end)
        
        # Calculate turnover metrics
        turnover = portfolio_analytics.calculate_turnover_metrics(
            positions_start=positions_start,
            positions_end=positions_end,
            period_start=request.period_start,
            period_end=request.period_end
        )
        
        # Calculate execution time
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Create response
        response = TurnoverResponse(
            success=True,
            message="Turnover metrics calculated successfully",
            timestamp=datetime.now(),
            data=turnover,
            calculation_id=calculation_id,
            execution_time_ms=execution_time_ms
        )
        
        logger.info(f"Turnover metrics calculation {calculation_id} completed in {execution_time_ms:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in turnover metrics calculation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Turnover metrics calculation failed: {str(e)}"
        )


@router.get("/factors")
async def get_attribution_factors():
    """
    Get available performance attribution factors.
    
    Returns:
        Available attribution factors
    """
    try:
        factors = [
            {
                "factor": AttributionFactor.CARRY_ROLLDOWN.value,
                "description": "Carry and roll-down effects",
                "explanation": "Interest income and price appreciation due to time passage"
            },
            {
                "factor": AttributionFactor.CURVE_LEVEL.value,
                "description": "Yield curve level changes",
                "explanation": "Parallel shifts in the yield curve"
            },
            {
                "factor": AttributionFactor.CURVE_SLOPE.value,
                "description": "Yield curve slope changes",
                "explanation": "Steepening or flattening of the yield curve"
            },
            {
                "factor": AttributionFactor.CURVE_CURVATURE.value,
                "description": "Yield curve curvature changes",
                "explanation": "Changes in the shape of the yield curve"
            },
            {
                "factor": AttributionFactor.CREDIT_SPREAD.value,
                "description": "Credit spread changes",
                "explanation": "Changes in credit risk premiums"
            },
            {
                "factor": AttributionFactor.SELECTION.value,
                "description": "Security selection effects",
                "explanation": "Performance relative to benchmark due to specific security choices"
            },
            {
                "factor": AttributionFactor.TRADING.value,
                "description": "Trading effects",
                "explanation": "Performance due to timing of trades"
            },
            {
                "factor": AttributionFactor.IDIOSYNCRATIC.value,
                "description": "Idiosyncratic effects",
                "explanation": "Security-specific factors not captured by other factors"
            }
        ]
        
        return {
            "success": True,
            "message": "Attribution factors retrieved successfully",
            "timestamp": datetime.now(),
            "data": factors
        }
        
    except Exception as e:
        logger.error(f"Error getting attribution factors: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving attribution factors: {str(e)}"
        )


@router.get("/tenor-buckets")
async def get_tenor_buckets():
    """
    Get available tenor buckets for portfolio analysis.
    
    Returns:
        Available tenor buckets
    """
    try:
        tenor_buckets = [
            {
                "bucket": TenorBucket.ZERO_TO_ONE_YEAR.value,
                "description": "0 to 1 year",
                "typical_instruments": "T-bills, short-term notes, money market instruments"
            },
            {
                "bucket": TenorBucket.ONE_TO_THREE_YEARS.value,
                "description": "1 to 3 years",
                "typical_instruments": "Short-term bonds, floating rate notes"
            },
            {
                "bucket": TenorBucket.THREE_TO_FIVE_YEARS.value,
                "description": "3 to 5 years",
                "typical_instruments": "Medium-term bonds, corporate notes"
            },
            {
                "bucket": TenorBucket.FIVE_TO_TEN_YEARS.value,
                "description": "5 to 10 years",
                "typical_instruments": "Medium-long term bonds, benchmark maturities"
            },
            {
                "bucket": TenorBucket.TEN_PLUS_YEARS.value,
                "description": "10+ years",
                "typical_instruments": "Long-term bonds, perpetual securities"
            }
        ]
        
        return {
            "success": True,
            "message": "Tenor buckets retrieved successfully",
            "timestamp": datetime.now(),
            "data": tenor_buckets
        }
        
    except Exception as e:
        logger.error(f"Error getting tenor buckets: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving tenor buckets: {str(e)}"
        )


@router.get("/concentration-limits")
async def get_concentration_limits():
    """
    Get recommended concentration limits for portfolio management.
    
    Returns:
        Recommended concentration limits
    """
    try:
        concentration_limits = {
            "issuer_concentration": {
                "warning_threshold": 5.0,  # 5%
                "limit_threshold": 10.0,   # 10%
                "description": "Maximum exposure to any single issuer"
            },
            "sector_concentration": {
                "warning_threshold": 25.0,  # 25%
                "limit_threshold": 40.0,   # 40%
                "description": "Maximum exposure to any single sector"
            },
            "rating_concentration": {
                "warning_threshold": 30.0,  # 30%
                "limit_threshold": 50.0,   # 50%
                "description": "Maximum exposure to below-investment-grade securities"
            },
            "tenor_concentration": {
                "warning_threshold": 40.0,  # 40%
                "limit_threshold": 60.0,   # 60%
                "description": "Maximum exposure to any single tenor bucket"
            },
            "liquidity_concentration": {
                "warning_threshold": 20.0,  # 20%
                "limit_threshold": 30.0,   # 30%
                "description": "Maximum exposure to illiquid securities"
            }
        }
        
        return {
            "success": True,
            "message": "Concentration limits retrieved successfully",
            "timestamp": datetime.now(),
            "data": concentration_limits
        }
        
    except Exception as e:
        logger.error(f"Error getting concentration limits: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving concentration limits: {str(e)}"
        )


@router.get("/performance-targets")
async def get_performance_targets():
    """
    Get performance targets for portfolio analytics.
    
    Returns:
        Performance targets
    """
    try:
        performance_targets = {
            "portfolio_metrics": {
                "target_time": "≤50ms for 10,000 positions",
                "description": "Portfolio risk metrics calculation time"
            },
            "performance_attribution": {
                "target_time": "≤100ms for 10,000 positions",
                "description": "Performance attribution calculation time"
            },
            "turnover_metrics": {
                "target_time": "≤30ms for 10,000 positions",
                "description": "Turnover metrics calculation time"
            },
            "curve_factor_decomposition": {
                "target_time": "≤20ms for 10 curves",
                "description": "PCA-based curve factor decomposition time"
            },
            "real_time_updates": {
                "target_time": "≤10ms for incremental updates",
                "description": "Real-time portfolio analytics updates"
            }
        }
        
        return {
            "success": True,
            "message": "Performance targets retrieved successfully",
            "timestamp": datetime.now(),
            "data": performance_targets
        }
        
    except Exception as e:
        logger.error(f"Error getting performance targets: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving performance targets: {str(e)}"
        )


async def _convert_positions(positions_data: List[PositionSchema]) -> List[Position]:
    """Convert API position schemas to Position objects."""
    positions = []
    
    for pos_data in positions_data:
        position = Position(
            instrument_id=pos_data.instrument_id,
            face_value=pos_data.face_value,
            book_value=pos_data.book_value,
            market_value=pos_data.market_value,
            coupon_rate=pos_data.coupon_rate,
            maturity_date=pos_data.maturity_date,
            duration=pos_data.duration,
            convexity=pos_data.convexity,
            spread_dv01=pos_data.spread_dv01,
            liquidity_score=pos_data.liquidity_score,
            issuer_id=pos_data.issuer_id,
            sector=pos_data.sector,
            rating=pos_data.rating,
            tenor_bucket=pos_data.tenor_bucket,
            oas_sensitive=pos_data.oas_sensitive
        )
        positions.append(position)
    
    return positions


async def _get_yield_curves(curve_ids: dict) -> dict:
    """Get yield curves by currency (placeholder implementation)."""
    # This would typically fetch from database
    # For now, create mock curves
    from ...mathematics.yield_curves import YieldCurve, CurveType, CurveConstructionConfig
    import numpy as np
    
    curves = {}
    
    for currency, curve_id in curve_ids.items():
        # Create mock curve for each currency
        tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])
        
        # Different base rates for different currencies
        if currency == "INR":
            base_rate = 0.06
        elif currency == "USD":
            base_rate = 0.04
        elif currency == "EUR":
            base_rate = 0.02
        else:
            base_rate = 0.05
        
        rates = np.array([
            base_rate + i * 0.005 for i in range(len(tenors))
        ])
        
        config = CurveConstructionConfig()
        
        curves[currency] = YieldCurve(
            curve_type=CurveType.ZERO_CURVE,
            tenors=tenors,
            rates=rates,
            construction_date=datetime.now().date(),
            config=config
        )
    
    return curves


async def _convert_spread_surfaces(spread_data: dict) -> dict:
    """Convert spread surface data to proper format."""
    # Convert string ratings to RatingBucket enums
    spread_surfaces = {}
    
    for rating_str, tenor_spreads in spread_data.items():
        try:
            rating = RatingBucket(rating_str)
            spread_surfaces[rating] = tenor_spreads
        except ValueError:
            logger.warning(f"Invalid rating: {rating_str}")
    
    return spread_surfaces


# Export router
__all__ = ["router"]
