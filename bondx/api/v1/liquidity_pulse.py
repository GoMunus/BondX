"""
Liquidity Pulse API Router for BondX

This module provides REST API endpoints for the Liquidity Pulse service,
including pulse calculation, heatmap generation, and service management.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse

from ...core.logging import get_logger
from ...core.model_contracts import ModelResultStore, ModelValidator, ModelType, ModelStatus
from ...liquidity.pulse import LiquidityPulseEngine
from .schemas_liquidity import (
    LiquidityPulseRequest, LiquidityPulseResponse, HeatmapRequest, HeatmapResponse,
    LiquidityPulse, HeatmapCell, PulseValidationRequest, PulseValidationResponse,
    ViewType
)

logger = get_logger(__name__)

# Initialize router
router = APIRouter(prefix="/liquidity-pulse", tags=["liquidity-pulse"])

# Initialize components
model_store = ModelResultStore(enable_caching=True, max_cache_size=1000)
model_validator = ModelValidator()

# Initialize pulse engine with configuration
pulse_config = {
    "signal_adapters": {
        "alt_data": {"enabled": True},
        "microstructure": {"enabled": True},
        "auction_mm": {"enabled": True},
        "sentiment": {"enabled": True}
    },
    "feature_engine": {
        "rolling_windows": [7, 30, 90],
        "seasonality_periods": [7, 30],
        "stability_threshold": 0.1,
        "anomaly_threshold": 2.0
    },
    "forecast_engine": {
        "forecast_horizons": [1, 2, 3, 4, 5],
        "max_history_days": 90,
        "min_training_samples": 30,
        "model_type": "gradient_boosting"
    },
    "models": {
        "liquidity_model": {
            "base_liquidity": 50.0,
            "calibration": {
                "min_spread_bps": 1.0,
                "max_spread_bps": 100.0,
                "spread_sensitivity": 0.5,
                "depth_threshold": 1000000
            }
        },
        "repayment_model": {
            "base_repayment": 60.0,
            "sector_adjustments": {
                "GOVERNMENT": 20,
                "UTILITIES": 15,
                "FINANCIAL": -5,
                "INDUSTRIAL": 0
            }
        },
        "bondx_weights": {
            "liquidity_index_weight": 0.6,
            "repayment_support_weight": 0.4
        }
    },
    "max_calculation_time_ms": 50,
    "enable_forecasting": True,
    "enable_driver_analysis": True,
    "max_history_length": 100
}

pulse_engine = LiquidityPulseEngine(pulse_config)

# Mock data for demonstration (replace with actual database queries)
MOCK_ISINS = [
    "INE001A07BM4", "INE002A07BM5", "INE003A07BM6", "INE004A07BM7",
    "INE005A07BM8", "INE006A07BM9", "INE007A07BM0", "INE008A07BM1"
]

MOCK_SECTORS = ["FINANCIAL", "UTILITIES", "INDUSTRIAL", "GOVERNMENT"]
MOCK_RATINGS = ["AAA", "AA", "A", "BBB", "BB"]
MOCK_TENORS = ["SHORT", "MEDIUM", "LONG", "VERY_LONG"]

@router.get("/{isin}")
async def get_liquidity_pulse(
    isin: str,
    forecast: bool = Query(True, description="Include T+1 to T+5 forecasts"),
    detail: str = Query("drivers", description="Detail level: drivers, minimal, full"),
    view_type: ViewType = Query(ViewType.PROFESSIONAL, description="View type for role-based access")
):
    """
    Get liquidity pulse for a specific ISIN.
    
    Returns the current liquidity pulse including:
    - Liquidity Index (0-100)
    - Repayment Support (0-100) 
    - BondX Score (0-100)
    - Forecasts (if requested)
    - Driver analysis (if requested)
    """
    try:
        logger.info(f"Getting liquidity pulse for {isin}")
        
        # Calculate pulse
        result = await pulse_engine.calculate_pulse(
            isin=isin,
            mode="fast",
            include_forecast=forecast,
            include_drivers=(detail in ["drivers", "full"])
        )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error_message)
        
        # Convert to API schema
        pulse = pulse_engine.convert_to_liquidity_pulse(result)
        
        # Apply role-based filtering
        if view_type == ViewType.RETAIL:
            # Remove sensitive information for retail users
            pulse.drivers = [d for d in pulse.drivers if d.source not in ["microstructure", "auction_mm"]]
            pulse.missing_signals = []
            pulse.uncertainty = min(pulse.uncertainty, 0.5)  # Cap uncertainty for retail
        
        # Store result in model store
        model_result = ModelResult(
            model_type=ModelType.LIQUIDITY_PULSE,
            inputs={"isin": isin, "forecast": forecast, "detail": detail, "view_type": view_type.value},
            outputs={"pulse": pulse.dict()},
            model_id="liquidity_pulse_v1",
            execution_id=f"pulse_{isin}_{datetime.now().timestamp()}",
            created_date=datetime.now(),
            updated_date=datetime.now(),
            metadata={
                "calculation_time_ms": result.calculation_time_ms,
                "view_type": view_type.value,
                "detail_level": detail
            }
        )
        
        model_store.store_result(model_result)
        
        return LiquidityPulseResponse(
            success=True,
            message="Liquidity pulse retrieved successfully",
            data=[pulse]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting liquidity pulse for {isin}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/recompute")
async def recompute_liquidity_pulse(
    request: LiquidityPulseRequest,
    background_tasks: BackgroundTasks
):
    """
    Recompute liquidity pulse for multiple ISINs.
    
    Supports both fast and accurate modes:
    - Fast mode: Uses cached features and simplified calculations
    - Accurate mode: Full feature computation and model evaluation
    """
    try:
        logger.info(f"Recomputing liquidity pulse for {len(request.isins)} ISINs in {request.mode} mode")
        
        # Calculate pulses
        results = await pulse_engine.calculate_batch_pulse(
            isins=request.isins,
            mode=request.mode,
            include_forecast=request.include_forecast,
            include_drivers=request.include_drivers
        )
        
        # Convert to API schemas
        pulses = []
        for result in results:
            if result.success:
                pulse = pulse_engine.convert_to_liquidity_pulse(result)
                
                # Apply role-based filtering
                if request.view_type == ViewType.RETAIL:
                    pulse.drivers = [d for d in pulse.drivers if d.source not in ["microstructure", "auction_mm"]]
                    pulse.missing_signals = []
                    pulse.uncertainty = min(pulse.uncertainty, 0.5)
                
                pulses.append(pulse)
        
        # Store results in model store
        for i, result in enumerate(results):
            if result.success:
                model_result = ModelResult(
                    model_type=ModelType.LIQUIDITY_PULSE,
                    inputs={"batch_request": request.dict(), "index": i},
                    outputs={"pulse": pulses[i].dict()},
                    model_id="liquidity_pulse_v1",
                    execution_id=f"batch_pulse_{datetime.now().timestamp()}_{i}",
                    created_date=datetime.now(),
                    updated_date=datetime.now(),
                    metadata={
                        "calculation_time_ms": result.calculation_time_ms,
                        "mode": request.mode,
                        "view_type": request.view_type.value
                    }
                )
                model_store.store_result(model_result)
        
        # Schedule cleanup task
        background_tasks.add_task(pulse_engine.cleanup_old_data)
        
        return LiquidityPulseResponse(
            success=True,
            message=f"Liquidity pulse recomputed for {len(pulses)} ISINs",
            data=pulses
        )
        
    except Exception as e:
        logger.error(f"Error recomputing liquidity pulse: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/heatmap")
async def get_heatmap(
    sector: Optional[str] = Query(None, description="Sector filter"),
    rating: Optional[str] = Query(None, description="Rating filter"),
    tenor: Optional[str] = Query(None, description="Tenor filter"),
    view: str = Query("liquidity", description="View type: liquidity or bondx"),
    view_type: ViewType = Query(ViewType.PROFESSIONAL, description="View type for role-based access")
):
    """
    Get heatmap data for liquidity pulse across sectors, ratings, and tenors.
    
    Returns aggregated data suitable for visualization in heatmap format.
    """
    try:
        logger.info(f"Getting heatmap data: sector={sector}, rating={rating}, tenor={tenor}, view={view}")
        
        # Generate mock heatmap data (replace with actual aggregation logic)
        heatmap_data = _generate_mock_heatmap(sector, rating, tenor, view, view_type)
        
        return HeatmapResponse(
            success=True,
            message="Heatmap data retrieved successfully",
            data=heatmap_data
        )
        
    except Exception as e:
        logger.error(f"Error getting heatmap data: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/validate")
async def validate_pulse(
    request: PulseValidationRequest
):
    """
    Validate liquidity pulse calculations against reference data.
    
    Performs backtesting and validation to ensure model accuracy.
    """
    try:
        logger.info(f"Validating pulse for {request.isin}")
        
        # Mock validation (replace with actual validation logic)
        validation_results = {
            "forecast_accuracy": {
                "t_plus_1": 0.85,
                "t_plus_2": 0.78,
                "t_plus_3": 0.72,
                "t_plus_4": 0.68,
                "t_plus_5": 0.65
            },
            "driver_stability": 0.82,
            "freshness_accuracy": 0.91,
            "overall_score": 0.81
        }
        
        recommendations = [
            "Consider retraining forecast models with recent data",
            "Review signal quality thresholds for alt-data sources",
            "Optimize feature engineering for better stability"
        ]
        
        return PulseValidationResponse(
            success=True,
            message="Pulse validation completed successfully",
            validation_results=validation_results,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error validating pulse: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health")
async def get_pulse_health():
    """
    Get health status of the liquidity pulse engine.
    
    Returns component health, performance metrics, and system status.
    """
    try:
        health = pulse_engine.get_engine_health()
        return health
        
    except Exception as e:
        logger.error(f"Error getting pulse health: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/performance")
async def get_pulse_performance():
    """
    Get performance metrics for the liquidity pulse engine.
    
    Returns calculation times, throughput, and efficiency metrics.
    """
    try:
        metrics = pulse_engine.get_performance_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting pulse performance: {e}")
        return {"error": str(e)}

@router.post("/train")
async def train_forecast_models(
    background_tasks: BackgroundTasks
):
    """
    Train forecast models using historical data.
    
    This is a background task that may take significant time.
    """
    try:
        logger.info("Starting forecast model training")
        
        # Mock training data (replace with actual historical data)
        training_data = {}
        for isin in MOCK_ISINS[:5]:  # Use subset for training
            training_data[isin] = [
                (None, 75.0),  # Mock feature set and liquidity value
                (None, 78.0),
                (None, 72.0)
            ]
        
        # Schedule training in background
        background_tasks.add_task(pulse_engine.train_forecast_models, training_data)
        
        return {
            "success": True,
            "message": "Forecast model training started in background",
            "training_isins": len(training_data)
        }
        
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/models/status")
async def get_model_status():
    """
    Get status of all models in the pulse engine.
    
    Returns training status, performance metrics, and version information.
    """
    try:
        forecast_performance = pulse_engine.forecast_engine.get_model_performance()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "forecast_models": forecast_performance,
            "model_versions": pulse_engine.model_versions,
            "overall_status": "active"
        }
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return {"error": str(e)}

def _generate_mock_heatmap(sector: Optional[str], rating: Optional[str], 
                          tenor: Optional[str], view: str, view_type: ViewType) -> List[HeatmapCell]:
    """Generate mock heatmap data for demonstration."""
    heatmap_data = []
    
    # Filter sectors
    sectors = [sector] if sector else MOCK_SECTORS
    ratings = [rating] if rating else MOCK_RATINGS
    tenors = [tenor] if tenor else MOCK_TENORS
    
    for s in sectors:
        for r in ratings:
            for t in tenors:
                # Generate mock values based on view type
                if view == "liquidity":
                    base_value = 70.0
                    if s == "GOVERNMENT":
                        base_value += 15
                    elif s == "FINANCIAL":
                        base_value -= 10
                    
                    if r == "AAA":
                        base_value += 10
                    elif r in ["BB", "B"]:
                        base_value -= 15
                    
                    if t == "SHORT":
                        base_value += 5
                    elif t == "VERY_LONG":
                        base_value -= 10
                        
                else:  # bondx view
                    base_value = 75.0
                    if s == "GOVERNMENT":
                        base_value += 20
                    elif s == "FINANCIAL":
                        base_value -= 5
                    
                    if r == "AAA":
                        base_value += 15
                    elif r in ["BB", "B"]:
                        base_value -= 10
                    
                    if t == "SHORT":
                        base_value += 3
                    elif t == "VERY_LONG":
                        base_value -= 8
                
                # Add some randomness
                value = max(0, min(100, base_value + np.random.normal(0, 5)))
                
                # Determine trend
                trend = "→"  # neutral
                if value > 80:
                    trend = "↑"
                elif value < 30:
                    trend = "↓"
                
                cell = HeatmapCell(
                    sector=s,
                    rating=r,
                    tenor=t,
                    value=round(value, 1),
                    count=np.random.randint(5, 50),
                    trend=trend,
                    confidence=np.random.uniform(0.7, 0.95)
                )
                
                heatmap_data.append(cell)
    
    return heatmap_data

# Add router to FastAPI app
def include_liquidity_pulse_router(app):
    """Include the liquidity pulse router in the FastAPI app."""
    app.include_router(router)
    logger.info("Liquidity Pulse router included in FastAPI app")
