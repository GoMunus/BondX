"""
OAS (Option-Adjusted Spread) API endpoints for BondX Backend.

This module provides REST API endpoints for OAS calculations including
single calculations, batch processing, and progress monitoring.
"""

import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
import asyncio

from ...core.logging import get_logger
from ...core.model_contracts import ModelResultStore, ModelValidator, ModelType, ModelStatus
from ...mathematics.option_adjusted_spread import (
    OASCalculator, OASInputs, OASOutputs, OptionType, PricingMethod, LatticeModel,
    CallSchedule, PutSchedule, PrepaymentFunction, VolatilitySurface
)
from ...mathematics.yield_curves import YieldCurve
from ...mathematics.cash_flows import CashFlow
from ...database.models import DayCountConvention
from .schemas import (
    OASCalculationRequest, OASCalculationResponse, BatchOASRequest, BatchOASResponse,
    OASProgressUpdate, ErrorResponse
)

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/oas", tags=["Option-Adjusted Spread"])

# Global instances
oas_calculator = OASCalculator()
model_store = ModelResultStore(enable_caching=True, max_cache_size=1000)
model_validator = ModelValidator(strict_mode=False)

# In-memory storage for progress tracking
progress_tracker = {}


@router.post("/calculate", response_model=OASCalculationResponse)
async def calculate_oas(
    request: OASCalculationRequest,
    background_tasks: BackgroundTasks
) -> OASCalculationResponse:
    """
    Calculate OAS for a bond with embedded options.
    
    Args:
        request: OAS calculation request
        background_tasks: FastAPI background tasks
        
    Returns:
        OAS calculation response
    """
    try:
        calculation_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Starting OAS calculation {calculation_id}")
        
        # Convert API request to OAS inputs
        oas_inputs = await _convert_request_to_inputs(request)
        
        # Run OAS calculation
        oas_outputs = oas_calculator.calculate_oas(oas_inputs)
        
        # Calculate execution time
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Store result in model store
        cache_key = oas_calculator.get_cache_key(oas_inputs)
        
        # Create response
        response = OASCalculationResponse(
            success=True,
            message="OAS calculation completed successfully",
            timestamp=datetime.now(),
            data=oas_outputs,
            calculation_id=calculation_id,
            cache_key=cache_key,
            execution_time_ms=execution_time_ms
        )
        
        logger.info(f"OAS calculation {calculation_id} completed in {execution_time_ms:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in OAS calculation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"OAS calculation failed: {str(e)}"
        )


@router.post("/calculate/batch", response_model=BatchOASResponse)
async def calculate_oas_batch(
    request: BatchOASRequest,
    background_tasks: BackgroundTasks
) -> BatchOASResponse:
    """
    Calculate OAS for multiple bonds in batch.
    
    Args:
        request: Batch OAS calculation request
        background_tasks: FastAPI background tasks
        
    Returns:
        Batch OAS calculation response
    """
    try:
        batch_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Starting batch OAS calculation {batch_id} with {len(request.calculations)} calculations")
        
        # Process calculations
        results = []
        successful_calculations = 0
        failed_calculations = 0
        
        if request.enable_parallel and request.max_workers > 1:
            # Parallel processing
            results = await _process_batch_parallel(request.calculations, request.max_workers)
        else:
            # Sequential processing
            results = await _process_batch_sequential(request.calculations)
        
        # Count results
        successful_calculations = len([r for r in results if r.success])
        failed_calculations = len([r for r in results if not r.success])
        
        # Calculate total execution time
        total_execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Create response
        response = BatchOASResponse(
            success=True,
            message=f"Batch OAS calculation completed: {successful_calculations} successful, {failed_calculations} failed",
            timestamp=datetime.now(),
            data=results,
            total_calculations=len(request.calculations),
            successful_calculations=successful_calculations,
            failed_calculations=failed_calculations,
            total_execution_time_ms=total_execution_time_ms
        )
        
        logger.info(f"Batch OAS calculation {batch_id} completed in {total_execution_time_ms:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in batch OAS calculation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch OAS calculation failed: {str(e)}"
        )


@router.get("/progress/{calculation_id}")
async def get_oas_progress(calculation_id: str) -> OASProgressUpdate:
    """
    Get progress update for OAS calculation.
    
    Args:
        calculation_id: Calculation ID to track
        
    Returns:
        Progress update
    """
    try:
        if calculation_id not in progress_tracker:
            raise HTTPException(
                status_code=404,
                detail=f"Calculation ID {calculation_id} not found"
            )
        
        progress = progress_tracker[calculation_id]
        return OASProgressUpdate(**progress)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting OAS progress: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting progress: {str(e)}"
        )


@router.get("/scenarios")
async def get_oas_scenarios():
    """
    Get available OAS calculation scenarios and configurations.
    
    Returns:
        Available scenarios and configurations
    """
    try:
        scenarios = {
            "pricing_methods": [
                {
                    "method": method.value,
                    "description": f"Use {method.value.lower()} method for OAS calculation",
                    "suitable_for": _get_method_suitability(method)
                }
                for method in PricingMethod
            ],
            "lattice_models": [
                {
                    "model": model.value,
                    "description": f"Use {model.value} short-rate model",
                    "characteristics": _get_model_characteristics(model)
                }
                for model in LatticeModel
            ],
            "option_types": [
                {
                    "type": option.value,
                    "description": _get_option_description(option),
                    "requires_schedule": _requires_schedule(option)
                }
                for option in OptionType
            ],
            "performance_targets": {
                "lattice_mode": "≤30ms for 500-step tree typical case",
                "monte_carlo_mode": "Configurable paths with performance profiling",
                "fast_mode": "≤10ms for option-free bonds",
                "accurate_mode": "≤100ms for complex option structures"
            }
        }
        
        return {
            "success": True,
            "message": "OAS scenarios retrieved successfully",
            "timestamp": datetime.now(),
            "data": scenarios
        }
        
    except Exception as e:
        logger.error(f"Error getting OAS scenarios: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving scenarios: {str(e)}"
        )


@router.get("/cache/stats")
async def get_oas_cache_stats():
    """
    Get OAS calculation cache statistics.
    
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
async def clear_oas_cache():
    """
    Clear OAS calculation cache.
    
    Returns:
        Success message
    """
    try:
        model_store.clear_cache()
        
        return {
            "success": True,
            "message": "OAS calculation cache cleared successfully",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing cache: {str(e)}"
        )


async def _convert_request_to_inputs(request: OASCalculationRequest) -> OASInputs:
    """Convert API request to OAS inputs."""
    try:
        # Get base curve (this would typically fetch from database)
        base_curve = await _get_yield_curve(request.curve_id)
        
        # Convert volatility surface
        vol_surface = VolatilitySurface(
            tenors=np.array(request.volatility_surface.tenors),
            volatilities=np.array(request.volatility_surface.volatilities),
            mean_reversion=request.volatility_surface.mean_reversion,
            correlation_matrix=np.array(request.volatility_surface.correlation_matrix) if request.volatility_surface.correlation_matrix else None
        )
        
        # Convert cash flows (simplified - would need proper conversion)
        cash_flows = await _convert_cash_flows(request.cash_flows)
        
        # Convert option schedules
        call_schedule = None
        if request.call_schedule:
            call_schedule = [
                CallSchedule(
                    call_date=call.call_date,
                    call_price=call.call_price,
                    notice_period_days=call.notice_period_days,
                    make_whole_spread=call.make_whole_spread
                )
                for call in request.call_schedule
            ]
        
        put_schedule = None
        if request.put_schedule:
            put_schedule = [
                PutSchedule(
                    put_date=put.put_date,
                    put_price=put.put_price,
                    notice_period_days=put.notice_period_days
                )
                for put in request.put_schedule
            ]
        
        # Convert prepayment function
        prepayment_function = None
        if request.prepayment_function:
            prepayment_function = PrepaymentFunction(
                cpr_base=request.prepayment_function.cpr_base,
                psa_multiplier=request.prepayment_function.psa_multiplier,
                burnout_factor=request.prepayment_function.burnout_factor,
                age_factor=request.prepayment_function.age_factor
            )
        
        # Convert day count convention
        day_count_convention = DayCountConvention(request.day_count_convention)
        
        # Create OAS inputs
        oas_inputs = OASInputs(
            base_curve=base_curve,
            volatility_surface=vol_surface,
            cash_flows=cash_flows,
            option_type=request.option_type,
            call_schedule=call_schedule,
            put_schedule=put_schedule,
            prepayment_function=prepayment_function,
            market_price=request.market_price,
            day_count_convention=day_count_convention,
            compounding_frequency=request.compounding_frequency,
            settlement_date=request.settlement_date
        )
        
        return oas_inputs
        
    except Exception as e:
        logger.error(f"Error converting request to inputs: {str(e)}")
        raise ValueError(f"Failed to convert request: {str(e)}")


async def _get_yield_curve(curve_id: str) -> YieldCurve:
    """Get yield curve by ID (placeholder implementation)."""
    # This would typically fetch from database
    # For now, create a mock curve
    import numpy as np
    
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])
    rates = np.array([0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.10])
    
    from ...mathematics.yield_curves import YieldCurve, CurveType, CurveConstructionConfig
    
    config = CurveConstructionConfig()
    
    return YieldCurve(
        curve_type=CurveType.ZERO_CURVE,
        tenors=tenors,
        rates=rates,
        construction_date=datetime.now().date(),
        config=config
    )


async def _convert_cash_flows(cash_flows_data: List[dict]) -> List[CashFlow]:
    """Convert cash flow data to CashFlow objects (placeholder implementation)."""
    # This would need proper conversion based on actual cash flow structure
    # For now, return empty list
    return []


async def _process_batch_parallel(
    calculations: List[OASCalculationRequest],
    max_workers: int
) -> List[OASCalculationResponse]:
    """Process batch calculations in parallel."""
    # This would implement actual parallel processing
    # For now, return sequential results
    return await _process_batch_sequential(calculations)


async def _process_batch_sequential(
    calculations: List[OASCalculationRequest]
) -> List[OASCalculationResponse]:
    """Process batch calculations sequentially."""
    results = []
    
    for i, calc_request in enumerate(calculations):
        try:
            # Update progress
            calculation_id = f"batch_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            progress_tracker[calculation_id] = {
                "calculation_id": calculation_id,
                "status": "RUNNING",
                "progress_percent": (i / len(calculations)) * 100,
                "current_step": f"Processing calculation {i+1} of {len(calculations)}",
                "estimated_time_remaining_ms": None
            }
            
            # Process calculation
            oas_inputs = await _convert_request_to_inputs(calc_request)
            oas_outputs = oas_calculator.calculate_oas(oas_inputs)
            
            # Create response
            response = OASCalculationResponse(
                success=True,
                message="OAS calculation completed successfully",
                timestamp=datetime.now(),
                data=oas_outputs,
                calculation_id=calculation_id,
                cache_key=oas_calculator.get_cache_key(oas_inputs),
                execution_time_ms=0.0  # Would calculate actual time
            )
            
            results.append(response)
            
            # Update progress
            progress_tracker[calculation_id]["status"] = "COMPLETED"
            progress_tracker[calculation_id]["progress_percent"] = 100.0
            
        except Exception as e:
            logger.error(f"Error processing calculation {i}: {str(e)}")
            
            # Create error response
            error_response = OASCalculationResponse(
                success=False,
                message=f"Calculation failed: {str(e)}",
                timestamp=datetime.now(),
                data=None,
                calculation_id=f"batch_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                cache_key="",
                execution_time_ms=0.0
            )
            
            results.append(error_response)
    
    return results


def _get_method_suitability(method: PricingMethod) -> str:
    """Get suitability description for pricing method."""
    suitability = {
        PricingMethod.LATTICE: "Callable/putable bonds, fast calculations",
        PricingMethod.MONTE_CARLO: "Complex option structures, path-dependent features"
    }
    return suitability.get(method, "General purpose")


def _get_model_characteristics(model: LatticeModel) -> str:
    """Get characteristics description for lattice model."""
    characteristics = {
        LatticeModel.HO_LEE: "Simple, fast, suitable for basic option pricing",
        LatticeModel.BLACK_DERMAN_TOY: "More sophisticated, handles mean reversion"
    }
    return characteristics.get(model, "Standard model")


def _get_option_description(option: OptionType) -> str:
    """Get description for option type."""
    descriptions = {
        OptionType.NONE: "No embedded options",
        OptionType.CALLABLE: "Issuer can call the bond",
        OptionType.PUTABLE: "Holder can put the bond back",
        OptionType.CALLABLE_MAKE_WHOLE: "Callable with make-whole provision",
        OptionType.PREPAYMENT: "Prepayment capability (MBS)"
    }
    return descriptions.get(option, "Unknown option type")


def _requires_schedule(option: OptionType) -> bool:
    """Check if option type requires schedule."""
    return option in [OptionType.CALLABLE, OptionType.PUTABLE, OptionType.CALLABLE_MAKE_WHOLE]


# Export router
__all__ = ["router"]
