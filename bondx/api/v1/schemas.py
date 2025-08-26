"""
API schemas for BondX Backend.

This module contains Pydantic schemas for API request/response validation
including schemas for Phase B components.
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator, root_validator

from ...mathematics.option_adjusted_spread import (
    OptionType, PricingMethod, LatticeModel, CallSchedule, PutSchedule,
    PrepaymentFunction, VolatilitySurface, OASInputs, OASOutputs
)
from ...risk_management.stress_testing import (
    ScenarioType, CalculationMode, RatingBucket, SectorBucket,
    Position, StressScenario, StressTestResult
)
from ...risk_management.portfolio_analytics import (
    AttributionFactor, TenorBucket, PortfolioMetrics,
    AttributionResult, TurnoverMetrics
)
from ...core.model_contracts import (
    ModelType, ValidationSeverity, ModelStatus, ValidationWarning,
    ModelInputs, ModelOutputs, ModelResult
)


# Base schemas
class BaseResponse(BaseModel):
    """Base response schema."""
    success: bool = True
    message: str = "Success"
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseResponse):
    """Error response schema."""
    success: bool = False
    error_code: str
    error_details: Optional[Dict[str, Any]] = None


# OAS Calculation Schemas
class CallScheduleSchema(BaseModel):
    """Call schedule schema for API."""
    call_date: date
    call_price: Decimal
    notice_period_days: int = 30
    make_whole_spread: Optional[Decimal] = None


class PutScheduleSchema(BaseModel):
    """Put schedule schema for API."""
    put_date: date
    put_price: Decimal
    notice_period_days: int = 30


class PrepaymentFunctionSchema(BaseModel):
    """Prepayment function schema for API."""
    cpr_base: float = Field(ge=0.0, le=1.0)
    psa_multiplier: float = Field(default=1.0, ge=0.0)
    burnout_factor: float = Field(default=1.0, ge=0.0)
    age_factor: float = Field(default=1.0, ge=0.0)


class VolatilitySurfaceSchema(BaseModel):
    """Volatility surface schema for API."""
    tenors: List[float] = Field(min_items=1)
    volatilities: List[float] = Field(min_items=1)
    mean_reversion: Optional[float] = None
    correlation_matrix: Optional[List[List[float]]] = None
    
    @validator('volatilities')
    def validate_volatilities_length(cls, v, values):
        if 'tenors' in values and len(v) != len(values['tenors']):
            raise ValueError('Volatilities must have same length as tenors')
        return v
    
    @validator('volatilities')
    def validate_volatilities_positive(cls, v):
        if any(vol < 0 for vol in v):
            raise ValueError('Volatilities must be non-negative')
        return v


class OASCalculationRequest(BaseModel):
    """OAS calculation request schema."""
    # Base curve reference
    curve_id: str = Field(..., description="ID of base yield curve")
    
    # Volatility surface
    volatility_surface: VolatilitySurfaceSchema
    
    # Cash flows (simplified for API)
    cash_flows: List[Dict[str, Any]] = Field(..., min_items=1)
    
    # Option type and schedules
    option_type: OptionType
    call_schedule: Optional[List[CallScheduleSchema]] = None
    put_schedule: Optional[List[PutScheduleSchema]] = None
    prepayment_function: Optional[PrepaymentFunctionSchema] = None
    
    # Market data
    market_price: Decimal = Field(..., gt=0)
    
    # Calculation parameters
    day_count_convention: str = "THIRTY_360"
    compounding_frequency: int = Field(default=2, ge=1, le=12)
    settlement_date: Optional[date] = None
    
    # Method selection
    pricing_method: PricingMethod = PricingMethod.LATTICE
    lattice_model: Optional[LatticeModel] = None
    lattice_steps: Optional[int] = Field(default=500, ge=100, le=1000)
    monte_carlo_paths: Optional[int] = Field(default=10000, ge=1000, le=100000)
    
    # Validation
    @validator('call_schedule')
    def validate_call_schedule(cls, v, values):
        if values.get('option_type') == OptionType.CALLABLE and not v:
            raise ValueError('Call schedule required for callable bonds')
        return v
    
    @validator('put_schedule')
    def validate_put_schedule(cls, v, values):
        if values.get('option_type') == OptionType.PUTABLE and not v:
            raise ValueError('Put schedule required for putable bonds')
        return v
    
    @validator('lattice_model')
    def validate_lattice_model(cls, v, values):
        if values.get('pricing_method') == PricingMethod.LATTICE and not v:
            raise ValueError('Lattice model required for lattice pricing method')
        return v


class OASCalculationResponse(BaseResponse):
    """OAS calculation response schema."""
    data: OASOutputs
    calculation_id: str
    cache_key: str
    execution_time_ms: float


# Stress Testing Schemas
class PositionSchema(BaseModel):
    """Position schema for API."""
    instrument_id: str
    face_value: Decimal = Field(..., gt=0)
    book_value: Decimal = Field(..., gt=0)
    market_value: Decimal = Field(..., gt=0)
    coupon_rate: Decimal
    maturity_date: date
    duration: float
    convexity: float
    spread_dv01: float
    liquidity_score: float = Field(ge=0.0, le=1.0)
    issuer_id: str
    sector: SectorBucket
    rating: RatingBucket
    tenor_bucket: str
    oas_sensitive: bool = False


class StressScenarioSchema(BaseModel):
    """Stress scenario schema for API."""
    scenario_id: str
    scenario_type: ScenarioType
    name: str
    description: str
    
    # Rate curve shocks
    parallel_shift_bps: Optional[int] = Field(None, ge=-1000, le=1000)
    curve_steepening_bps: Optional[int] = Field(None, ge=-500, le=500)
    curve_flattening_bps: Optional[int] = Field(None, ge=-500, le=500)
    
    # Credit spread shocks
    credit_spread_shocks: Optional[Dict[str, int]] = None  # rating -> bps
    
    # Liquidity shocks
    liquidity_spread_bps: Optional[int] = Field(None, ge=0, le=500)
    bid_ask_widening_bps: Optional[int] = Field(None, ge=0, le=500)
    
    # Volatility shocks
    volatility_multiplier: Optional[float] = Field(None, ge=0.1, le=10.0)
    
    # Custom shocks
    custom_shocks: Optional[Dict[str, float]] = None
    
    # Metadata
    severity: str = "MODERATE"
    probability: float = Field(default=0.01, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)


class StressTestRequest(BaseModel):
    """Stress test request schema."""
    portfolio: List[PositionSchema] = Field(..., min_items=1)
    scenarios: List[StressScenarioSchema] = Field(..., min_items=1)
    calculation_mode: CalculationMode = CalculationMode.FAST_APPROXIMATION
    enable_caching: bool = True
    
    # Validation
    @validator('portfolio')
    def validate_portfolio_size(cls, v):
        if len(v) > 10000:
            raise ValueError('Portfolio size cannot exceed 10,000 positions')
        return v


class StressTestResponse(BaseResponse):
    """Stress test response schema."""
    data: List[StressTestResult]
    total_scenarios: int
    successful_scenarios: int
    failed_scenarios: int
    total_execution_time_ms: float


# Portfolio Analytics Schemas
class PortfolioMetricsRequest(BaseModel):
    """Portfolio metrics request schema."""
    positions: List[PositionSchema] = Field(..., min_items=1)
    include_risk_metrics: bool = False
    yield_curves: Optional[Dict[str, str]] = None  # currency -> curve_id
    spread_surfaces: Optional[Dict[str, Dict[str, float]]] = None  # rating -> tenor -> spread


class PortfolioMetricsResponse(BaseResponse):
    """Portfolio metrics response schema."""
    data: PortfolioMetrics
    calculation_id: str
    execution_time_ms: float


class AttributionRequest(BaseModel):
    """Performance attribution request schema."""
    positions_start: List[PositionSchema] = Field(..., min_items=1)
    positions_end: List[PositionSchema] = Field(..., min_items=1)
    period_start: date
    period_end: date
    yield_curves_start: Dict[str, str] = Field(..., description="currency -> curve_id")
    yield_curves_end: Dict[str, str] = Field(..., description="currency -> curve_id")
    benchmark_returns: Optional[Dict[str, float]] = None
    
    @validator('period_end')
    def validate_period_end(cls, v, values):
        if 'period_start' in values and v <= values['period_start']:
            raise ValueError('Period end must be after period start')
        return v


class AttributionResponse(BaseResponse):
    """Performance attribution response schema."""
    data: AttributionResult
    calculation_id: str
    execution_time_ms: float


class TurnoverRequest(BaseModel):
    """Turnover metrics request schema."""
    positions_start: List[PositionSchema] = Field(..., min_items=1)
    positions_end: List[PositionSchema] = Field(..., min_items=1)
    period_start: date
    period_end: date


class TurnoverResponse(BaseResponse):
    """Turnover metrics response schema."""
    data: TurnoverMetrics
    calculation_id: str
    execution_time_ms: float


# Model Management Schemas
class ModelResultRequest(BaseModel):
    """Model result request schema."""
    model_type: ModelType
    cache_key: Optional[str] = None
    model_id: Optional[str] = None
    execution_id: Optional[str] = None


class ModelResultResponse(BaseResponse):
    """Model result response schema."""
    data: ModelResult


class ModelSearchRequest(BaseModel):
    """Model search request schema."""
    model_type: Optional[ModelType] = None
    curve_id: Optional[str] = None
    vol_id: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=1000)


class ModelSearchResponse(BaseResponse):
    """Model search response schema."""
    data: List[ModelResult]
    total_results: int
    page: int
    page_size: int


class CacheStatsResponse(BaseResponse):
    """Cache statistics response schema."""
    cache_size: int
    max_cache_size: int
    cache_hits: int
    cache_misses: int
    cache_evictions: int
    hit_rate: float


# Batch Processing Schemas
class BatchOASRequest(BaseModel):
    """Batch OAS calculation request schema."""
    calculations: List[OASCalculationRequest] = Field(..., min_items=1, max_items=100)
    enable_parallel: bool = True
    max_workers: int = Field(default=4, ge=1, le=16)


class BatchOASResponse(BaseResponse):
    """Batch OAS calculation response schema."""
    data: List[OASCalculationResponse]
    total_calculations: int
    successful_calculations: int
    failed_calculations: int
    total_execution_time_ms: float


class BatchStressTestRequest(BaseModel):
    """Batch stress test request schema."""
    portfolios: List[List[PositionSchema]] = Field(..., min_items=1, max_items=50)
    scenarios: List[StressScenarioSchema] = Field(..., min_items=1)
    calculation_mode: CalculationMode = CalculationMode.FAST_APPROXIMATION
    enable_parallel: bool = True
    max_workers: int = Field(default=4, ge=1, le=16)


class BatchStressTestResponse(BaseResponse):
    """Batch stress test response schema."""
    data: List[List[StressTestResult]]
    total_portfolios: int
    total_scenarios: int
    successful_runs: int
    failed_runs: int
    total_execution_time_ms: float


# Health and Monitoring Schemas
class HealthCheckResponse(BaseResponse):
    """Health check response schema."""
    status: str
    version: str
    uptime_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    active_connections: int
    cache_stats: Dict[str, Any]


class PerformanceMetricsResponse(BaseResponse):
    """Performance metrics response schema."""
    oas_calculation_avg_ms: float
    stress_test_avg_ms: float
    portfolio_metrics_avg_ms: float
    attribution_avg_ms: float
    cache_hit_rate: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float


# WebSocket Schemas
class WebSocketMessage(BaseModel):
    """WebSocket message schema."""
    message_type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    sequence_number: Optional[int] = None


class OASProgressUpdate(BaseModel):
    """OAS calculation progress update."""
    calculation_id: str
    status: str
    progress_percent: float
    current_step: str
    estimated_time_remaining_ms: Optional[float] = None


class StressTestProgressUpdate(BaseModel):
    """Stress test progress update."""
    test_id: str
    status: str
    completed_scenarios: int
    total_scenarios: int
    current_scenario: str
    estimated_time_remaining_ms: Optional[float] = None


# Export all schemas
__all__ = [
    # Base schemas
    "BaseResponse", "ErrorResponse",
    
    # OAS schemas
    "CallScheduleSchema", "PutScheduleSchema", "PrepaymentFunctionSchema",
    "VolatilitySurfaceSchema", "OASCalculationRequest", "OASCalculationResponse",
    
    # Stress testing schemas
    "PositionSchema", "StressScenarioSchema", "StressTestRequest", "StressTestResponse",
    
    # Portfolio analytics schemas
    "PortfolioMetricsRequest", "PortfolioMetricsResponse",
    "AttributionRequest", "AttributionResponse",
    "TurnoverRequest", "TurnoverResponse",
    
    # Model management schemas
    "ModelResultRequest", "ModelResultResponse",
    "ModelSearchRequest", "ModelSearchResponse",
    "CacheStatsResponse",
    
    # Batch processing schemas
    "BatchOASRequest", "BatchOASResponse",
    "BatchStressTestRequest", "BatchStressTestResponse",
    
    # Health and monitoring schemas
    "HealthCheckResponse", "PerformanceMetricsResponse",
    
    # WebSocket schemas
    "WebSocketMessage", "OASProgressUpdate", "StressTestProgressUpdate"
]
