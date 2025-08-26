"""
Stress testing API endpoints for BondX Backend.

This module provides REST API endpoints for portfolio stress testing including
single scenarios, batch processing, and predefined scenario management.
"""

import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from ...core.logging import get_logger
from ...core.model_contracts import ModelResultStore, ModelValidator, ModelType, ModelStatus
from ...risk_management.stress_testing import (
    StressTestingEngine, Position, StressScenario, StressTestResult,
    ScenarioType, CalculationMode, RatingBucket, SectorBucket
)
from ...mathematics.yield_curves import YieldCurve
from .schemas import (
    PositionSchema, StressScenarioSchema, StressTestRequest, StressTestResponse,
    BatchStressTestRequest, BatchStressTestResponse, StressTestProgressUpdate,
    ErrorResponse
)

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/stress-testing", tags=["Stress Testing"])

# Global instances
stress_engine = StressTestingEngine(
    calculation_mode=CalculationMode.FAST_APPROXIMATION,
    parallel_processing=True,
    cache_results=True
)
model_store = ModelResultStore(enable_caching=True, max_cache_size=1000)
model_validator = ModelValidator(strict_mode=False)

# In-memory storage for progress tracking
progress_tracker = {}


@router.post("/run", response_model=StressTestResponse)
async def run_stress_test(
    request: StressTestRequest,
    background_tasks: BackgroundTasks
) -> StressTestResponse:
    """
    Run stress test on portfolio with specified scenarios.
    
    Args:
        request: Stress test request
        background_tasks: FastAPI background tasks
        
    Returns:
        Stress test response
    """
    try:
        test_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Starting stress test {test_id} with {len(request.scenarios)} scenarios")
        
        # Convert API request to engine inputs
        portfolio = await _convert_positions(request.portfolio)
        scenarios = await _convert_scenarios(request.scenarios)
        
        # Get base curves and spread surfaces (placeholder)
        base_curves = await _get_base_curves()
        spread_surfaces = await _get_spread_surfaces()
        
        # Run stress tests
        results = stress_engine.run_multiple_scenarios(
            portfolio=portfolio,
            base_curves=base_curves,
            spread_surfaces=spread_surfaces,
            scenarios=scenarios,
            calculation_mode=request.calculation_mode
        )
        
        # Calculate total execution time
        total_execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Count results
        successful_scenarios = len([r for r in results if r.status == "COMPLETED"])
        failed_scenarios = len(results) - successful_scenarios
        
        # Create response
        response = StressTestResponse(
            success=True,
            message=f"Stress test completed: {successful_scenarios} successful, {failed_scenarios} failed",
            timestamp=datetime.now(),
            data=results,
            total_scenarios=len(request.scenarios),
            successful_scenarios=successful_scenarios,
            failed_scenarios=failed_scenarios,
            total_execution_time_ms=total_execution_time_ms
        )
        
        logger.info(f"Stress test {test_id} completed in {total_execution_time_ms:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in stress test: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Stress test failed: {str(e)}"
        )


@router.post("/run/batch", response_model=BatchStressTestResponse)
async def run_batch_stress_test(
    request: BatchStressTestRequest,
    background_tasks: BackgroundTasks
) -> BatchStressTestResponse:
    """
    Run stress tests on multiple portfolios in batch.
    
    Args:
        request: Batch stress test request
        background_tasks: FastAPI background tasks
        
    Returns:
        Batch stress test response
    """
    try:
        batch_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Starting batch stress test {batch_id} with {len(request.portfolios)} portfolios")
        
        # Get base curves and spread surfaces
        base_curves = await _get_base_curves()
        spread_surfaces = await _get_spread_surfaces()
        
        # Process portfolios
        all_results = []
        successful_runs = 0
        failed_runs = 0
        
        if request.enable_parallel and request.max_workers > 1:
            # Parallel processing
            all_results = await _process_batch_stress_parallel(
                request.portfolios, request.scenarios, request.calculation_mode,
                base_curves, spread_surfaces, request.max_workers
            )
        else:
            # Sequential processing
            all_results = await _process_batch_stress_sequential(
                request.portfolios, request.scenarios, request.calculation_mode,
                base_curves, spread_surfaces
            )
        
        # Count results
        for portfolio_results in all_results:
            successful_runs += len([r for r in portfolio_results if r.status == "COMPLETED"])
            failed_runs += len([r for r in portfolio_results if r.status != "COMPLETED"])
        
        # Calculate total execution time
        total_execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Create response
        response = BatchStressTestResponse(
            success=True,
            message=f"Batch stress test completed: {successful_runs} successful, {failed_runs} failed",
            timestamp=datetime.now(),
            data=all_results,
            total_portfolios=len(request.portfolios),
            total_scenarios=len(request.scenarios),
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            total_execution_time_ms=total_execution_time_ms
        )
        
        logger.info(f"Batch stress test {batch_id} completed in {total_execution_time_ms:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in batch stress test: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch stress test failed: {str(e)}"
        )


@router.get("/scenarios/predefined")
async def get_predefined_scenarios():
    """
    Get list of predefined stress test scenarios.
    
    Returns:
        List of predefined scenarios
    """
    try:
        scenarios = stress_engine.get_predefined_scenarios()
        
        # Convert to API format
        scenario_data = []
        for scenario in scenarios:
            scenario_data.append({
                "scenario_id": scenario.scenario_id,
                "scenario_type": scenario.scenario_type.value,
                "name": scenario.name,
                "description": scenario.description,
                "severity": scenario.severity,
                "probability": scenario.probability,
                "tags": scenario.tags,
                "parameters": {
                    "parallel_shift_bps": scenario.parallel_shift_bps,
                    "curve_steepening_bps": scenario.curve_steepening_bps,
                    "curve_flattening_bps": scenario.curve_flattening_bps,
                    "credit_spread_shocks": {
                        rating.value: bps for rating, bps in scenario.credit_spread_shocks.items()
                    } if scenario.credit_spread_shocks else None,
                    "liquidity_spread_bps": scenario.liquidity_spread_bps,
                    "bid_ask_widening_bps": scenario.bid_ask_widening_bps,
                    "volatility_multiplier": scenario.volatility_multiplier
                }
            })
        
        return {
            "success": True,
            "message": "Predefined scenarios retrieved successfully",
            "timestamp": datetime.now(),
            "data": scenario_data
        }
        
    except Exception as e:
        logger.error(f"Error getting predefined scenarios: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving scenarios: {str(e)}"
        )


@router.get("/scenarios/custom")
async def get_custom_scenarios():
    """
    Get list of custom stress test scenarios.
    
    Returns:
        List of custom scenarios
    """
    try:
        # This would typically fetch from database
        # For now, return empty list
        custom_scenarios = []
        
        return {
            "success": True,
            "message": "Custom scenarios retrieved successfully",
            "timestamp": datetime.now(),
            "data": custom_scenarios
        }
        
    except Exception as e:
        logger.error(f"Error getting custom scenarios: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving custom scenarios: {str(e)}"
        )


@router.post("/scenarios/custom")
async def create_custom_scenario(scenario: StressScenarioSchema):
    """
    Create a custom stress test scenario.
    
    Args:
        scenario: Custom scenario definition
        
    Returns:
        Created scenario
    """
    try:
        # This would typically save to database
        # For now, just return the scenario
        scenario_id = str(uuid.uuid4())
        
        return {
            "success": True,
            "message": "Custom scenario created successfully",
            "timestamp": datetime.now(),
            "data": {
                "scenario_id": scenario_id,
                "scenario": scenario.dict()
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating custom scenario: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating scenario: {str(e)}"
        )


@router.get("/progress/{test_id}")
async def get_stress_test_progress(test_id: str) -> StressTestProgressUpdate:
    """
    Get progress update for stress test.
    
    Args:
        test_id: Test ID to track
        
    Returns:
        Progress update
    """
    try:
        if test_id not in progress_tracker:
            raise HTTPException(
                status_code=404,
                detail=f"Test ID {test_id} not found"
            )
        
        progress = progress_tracker[test_id]
        return StressTestProgressUpdate(**progress)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stress test progress: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting progress: {str(e)}"
        )


@router.get("/methods")
async def get_calculation_methods():
    """
    Get available stress test calculation methods.
    
    Returns:
        Available calculation methods
    """
    try:
        methods = [
            {
                "method": CalculationMode.FAST_APPROXIMATION.value,
                "description": "Fast approximation using duration/convexity and spread DV01",
                "performance": "≤100ms for 10,000 positions",
                "suitable_for": "Quick analysis, large portfolios, real-time monitoring"
            },
            {
                "method": CalculationMode.FULL_REPRICE.value,
                "description": "Full revaluation using yield curves and cash flows",
                "performance": "Configurable, typically 1-10 seconds for 10,000 positions",
                "suitable_for": "Detailed analysis, regulatory reporting, accurate P&L"
            }
        ]
        
        return {
            "success": True,
            "message": "Calculation methods retrieved successfully",
            "timestamp": datetime.now(),
            "data": methods
        }
        
    except Exception as e:
        logger.error(f"Error getting calculation methods: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving methods: {str(e)}"
        )


@router.get("/cache/stats")
async def get_stress_test_cache_stats():
    """
    Get stress testing cache statistics.
    
    Returns:
        Cache statistics
    """
    try:
        cache_stats = model_store.get_cache_stats()
        
        return {
            "success": True,
            "message": "Cache statistics retrieved successfully",
            "timestamp": datetime.now(),
            "data": cache_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving cache stats: {str(e)}"
        )


@router.delete("/cache/clear")
async def clear_stress_test_cache():
    """
    Clear stress testing cache.
    
    Returns:
        Success message
    """
    try:
        model_store.clear_cache()
        
        return {
            "success": True,
            "message": "Stress testing cache cleared successfully",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing cache: {str(e)}"
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


async def _convert_scenarios(scenarios_data: List[StressScenarioSchema]) -> List[StressScenario]:
    """Convert API scenario schemas to StressScenario objects."""
    scenarios = []
    
    for scenario_data in scenarios_data:
        # Convert credit spread shocks
        credit_spread_shocks = None
        if scenario_data.credit_spread_shocks:
            credit_spread_shocks = {}
            for rating_str, bps in scenario_data.credit_spread_shocks.items():
                try:
                    rating = RatingBucket(rating_str)
                    credit_spread_shocks[rating] = bps
                except ValueError:
                    logger.warning(f"Invalid rating: {rating_str}")
        
        scenario = StressScenario(
            scenario_id=scenario_data.scenario_id,
            scenario_type=scenario_data.scenario_type,
            name=scenario_data.name,
            description=scenario_data.description,
            parallel_shift_bps=scenario_data.parallel_shift_bps,
            curve_steepening_bps=scenario_data.curve_steepening_bps,
            curve_flattening_bps=scenario_data.curve_flattening_bps,
            credit_spread_shocks=credit_spread_shocks,
            liquidity_spread_bps=scenario_data.liquidity_spread_bps,
            bid_ask_widening_bps=scenario_data.bid_ask_widening_bps,
            volatility_multiplier=scenario_data.volatility_multiplier,
            custom_shocks=scenario_data.custom_shocks,
            severity=scenario_data.severity,
            probability=scenario_data.probability,
            tags=scenario_data.tags
        )
        scenarios.append(scenario)
    
    return scenarios


async def _get_base_curves() -> dict:
    """Get base yield curves (placeholder implementation)."""
    # This would typically fetch from database
    # For now, create mock curves
    from ...mathematics.yield_curves import YieldCurve, CurveType, CurveConstructionConfig
    import numpy as np
    
    # Create mock INR curve
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])
    rates = np.array([0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.10])
    
    config = CurveConstructionConfig()
    
    inr_curve = YieldCurve(
        curve_type=CurveType.ZERO_CURVE,
        tenors=tenors,
        rates=rates,
        construction_date=datetime.now().date(),
        config=config
    )
    
    return {"INR": inr_curve}


async def _get_spread_surfaces() -> dict:
    """Get credit spread surfaces (placeholder implementation)."""
    # This would typically fetch from database
    # For now, create mock surfaces
    
    spread_surfaces = {}
    
    # Create mock spreads for each rating
    for rating in RatingBucket:
        spread_surfaces[rating] = {
            "0-1Y": 0.001,  # 10 bps
            "1-3Y": 0.002,  # 20 bps
            "3-5Y": 0.003,  # 30 bps
            "5-10Y": 0.004,  # 40 bps
            "10Y+": 0.005    # 50 bps
        }
    
    return spread_surfaces


async def _process_batch_stress_parallel(
    portfolios: List[List[PositionSchema]],
    scenarios: List[StressScenarioSchema],
    calculation_mode: CalculationMode,
    base_curves: dict,
    spread_surfaces: dict,
    max_workers: int
) -> List[List[StressTestResult]]:
    """Process batch stress tests in parallel."""
    # This would implement actual parallel processing
    # For now, return sequential results
    return await _process_batch_stress_sequential(
        portfolios, scenarios, calculation_mode, base_curves, spread_surfaces
    )


async def _process_batch_stress_sequential(
    portfolios: List[List[PositionSchema]],
    scenarios: List[StressScenarioSchema],
    calculation_mode: CalculationMode,
    base_curves: dict,
    spread_surfaces: dict
) -> List[List[StressTestResult]]:
    """Process batch stress tests sequentially."""
    all_results = []
    
    for i, portfolio_data in enumerate(portfolios):
        try:
            # Update progress
            test_id = f"batch_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            progress_tracker[test_id] = {
                "test_id": test_id,
                "status": "RUNNING",
                "completed_scenarios": 0,
                "total_scenarios": len(scenarios),
                "current_scenario": f"Processing portfolio {i+1} of {len(portfolios)}",
                "estimated_time_remaining_ms": None
            }
            
            # Convert portfolio
            portfolio = await _convert_positions(portfolio_data)
            scenarios_objects = await _convert_scenarios(scenarios)
            
            # Run stress tests
            results = stress_engine.run_multiple_scenarios(
                portfolio=portfolio,
                base_curves=base_curves,
                spread_surfaces=spread_surfaces,
                scenarios=scenarios_objects,
                calculation_mode=calculation_mode
            )
            
            all_results.append(results)
            
            # Update progress
            progress_tracker[test_id]["status"] = "COMPLETED"
            progress_tracker[test_id]["completed_scenarios"] = len(scenarios)
            
        except Exception as e:
            logger.error(f"Error processing portfolio {i}: {str(e)}")
            # Add empty results for failed portfolio
            all_results.append([])
    
    return all_results


# Export router
__all__ = ["router"]
