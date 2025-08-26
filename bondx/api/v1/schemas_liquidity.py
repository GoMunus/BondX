"""
Liquidity Pulse API Schemas for BondX

This module contains Pydantic schemas for the Liquidity Pulse service,
ensuring proper validation and documentation of API requests and responses.
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
from pydantic import BaseModel, Field, validator, root_validator

# Enums
class SignalQuality(str, Enum):
    """Signal quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    UNRELIABLE = "unreliable"

class DataFreshness(str, Enum):
    """Data freshness levels."""
    REAL_TIME = "real_time"      # < 1 second
    FRESH = "fresh"              # < 1 minute
    RECENT = "recent"            # < 1 hour
    STALE = "stale"              # < 1 day
    OUTDATED = "outdated"        # > 1 day

class ForecastHorizon(str, Enum):
    """Forecast horizons."""
    T_PLUS_1 = "T+1"
    T_PLUS_2 = "T+2"
    T_PLUS_3 = "T+3"
    T_PLUS_4 = "T+4"
    T_PLUS_5 = "T+5"

class ViewType(str, Enum):
    """View types for different user roles."""
    RETAIL = "retail"
    PROFESSIONAL = "professional"
    REGULATOR = "regulator"
    RISK = "risk"

# Input Data Schemas
class AltDataSignal(BaseModel):
    """Alternative data signal from external sources."""
    timestamp: datetime = Field(..., description="Signal timestamp")
    issuer_id: Optional[str] = Field(None, description="Issuer identifier")
    asset_id: Optional[str] = Field(None, description="Asset identifier")
    source_id: str = Field(..., description="Data source identifier")
    value: float = Field(..., description="Signal value")
    unit: str = Field(..., description="Value unit")
    quality: SignalQuality = Field(..., description="Signal quality")
    freshness_s: float = Field(..., description="Age of signal in seconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class MicrostructureData(BaseModel):
    """Market microstructure data."""
    timestamp: datetime = Field(..., description="Data timestamp")
    isin: str = Field(..., description="ISIN identifier")
    bid: Optional[float] = Field(None, description="Best bid price")
    ask: Optional[float] = Field(None, description="Best ask price")
    l2_depth_qty: Optional[float] = Field(None, description="L2 depth quantity")
    l2_levels: Optional[int] = Field(None, description="Number of L2 levels")
    trades_count: Optional[int] = Field(None, description="Number of trades")
    vwap: Optional[float] = Field(None, description="Volume weighted average price")
    volume_face: Optional[float] = Field(None, description="Face value volume")
    time_since_last_trade_s: Optional[float] = Field(None, description="Seconds since last trade")

class AuctionMMData(BaseModel):
    """Auction and market maker telemetry data."""
    timestamp: datetime = Field(..., description="Data timestamp")
    isin: str = Field(..., description="ISIN identifier")
    auction_demand_index: Optional[float] = Field(None, description="Auction demand index")
    mm_online: Optional[bool] = Field(None, description="Market maker online status")
    mm_spread_bps: Optional[float] = Field(None, description="Market maker spread in bps")
    quotes_last_24h: Optional[int] = Field(None, description="Quotes in last 24 hours")
    mm_inventory_band: Optional[str] = Field(None, description="Market maker inventory band")

class SentimentData(BaseModel):
    """Sentiment and news data."""
    timestamp: datetime = Field(..., description="Data timestamp")
    issuer_id: str = Field(..., description="Issuer identifier")
    sentiment_score: float = Field(..., ge=-1, le=1, description="Sentiment score from -1 to 1")
    buzz_volume: Optional[float] = Field(None, description="Buzz volume")
    topics: Optional[Dict[str, float]] = Field(None, description="Topic breakdown")
    quality: SignalQuality = Field(..., description="Data quality")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

# Feature Schemas
class RollingStats(BaseModel):
    """Rolling statistics for features."""
    window_7d: Optional[float] = Field(None, description="7-day rolling average")
    window_30d: Optional[float] = Field(None, description="30-day rolling average")
    window_90d: Optional[float] = Field(None, description="90-day rolling average")
    seasonality_adjustment: Optional[float] = Field(None, description="Seasonality adjustment")
    stability_metric: Optional[float] = Field(None, description="Stability metric")
    anomaly_zscore: Optional[float] = Field(None, description="Anomaly z-score")

class MicrostructureFeatures(BaseModel):
    """Microstructure-derived features."""
    spread_bps: Optional[float] = Field(None, description="Bid-ask spread in bps")
    depth_density: Optional[float] = Field(None, description="Market depth density")
    turnover_velocity: Optional[float] = Field(None, description="Turnover velocity")
    order_imbalance: Optional[float] = Field(None, description="Order imbalance proxy")
    time_decay: Optional[float] = Field(None, description="Time decay factor")

class SentimentFeatures(BaseModel):
    """Sentiment-derived features."""
    intensity: Optional[float] = Field(None, description="Sentiment intensity")
    volatility: Optional[float] = Field(None, description="Sentiment volatility")
    topic_splits: Optional[Dict[str, float]] = Field(None, description="Topic distribution")
    momentum: Optional[float] = Field(None, description="Sentiment momentum")

# Output Schemas
class ForecastPoint(BaseModel):
    """Single forecast point."""
    horizon_d: int = Field(..., description="Forecast horizon in days")
    liquidity_index: float = Field(..., ge=0, le=100, description="Predicted liquidity index")
    confidence: float = Field(..., ge=0, le=1, description="Forecast confidence")
    upper_bound: Optional[float] = Field(None, description="Upper confidence bound")
    lower_bound: Optional[float] = Field(None, description="Lower confidence bound")

class Driver(BaseModel):
    """Driver contribution to the score."""
    name: str = Field(..., description="Driver name")
    contribution: float = Field(..., description="Contribution magnitude")
    direction: str = Field(..., description="Direction: ↑ or ↓")
    source: str = Field(..., description="Data source")
    confidence: float = Field(..., ge=0, le=1, description="Driver confidence")

class LiquidityPulse(BaseModel):
    """Complete liquidity pulse output."""
    isin: str = Field(..., description="ISIN identifier")
    as_of: datetime = Field(..., description="Calculation timestamp")
    
    # Core indices
    liquidity_index: float = Field(..., ge=0, le=100, description="Liquidity index 0-100")
    repayment_support: float = Field(..., ge=0, le=100, description="Repayment support index 0-100")
    bondx_score: float = Field(..., ge=0, le=100, description="Combined BondX score 0-100")
    
    # Forecasts
    forecast: List[ForecastPoint] = Field(..., description="T+1 to T+5 forecasts")
    
    # Drivers and metadata
    drivers: List[Driver] = Field(..., description="Top drivers with contributions")
    missing_signals: List[str] = Field(..., description="Missing signal sources")
    freshness: DataFreshness = Field(..., description="Overall data freshness")
    uncertainty: float = Field(..., ge=0, le=1, description="Overall uncertainty")
    inputs_hash: str = Field(..., description="Hash of input data")
    model_versions: Dict[str, str] = Field(..., description="Model version information")

# API Request/Response Schemas
class LiquidityPulseRequest(BaseModel):
    """Request for liquidity pulse calculation."""
    isins: List[str] = Field(..., description="List of ISINs to process")
    mode: str = Field("fast", description="Processing mode: fast or accurate")
    include_forecast: bool = Field(True, description="Include T+1 to T+5 forecasts")
    include_drivers: bool = Field(True, description="Include driver analysis")
    view_type: ViewType = Field(ViewType.PROFESSIONAL, description="View type for role-based access")

class LiquidityPulseResponse(BaseModel):
    """Response containing liquidity pulse data."""
    success: bool = Field(True, description="Request success status")
    message: str = Field("Success", description="Response message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    data: List[LiquidityPulse] = Field(..., description="Liquidity pulse data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class HeatmapRequest(BaseModel):
    """Request for heatmap data."""
    sector: Optional[str] = Field(None, description="Sector filter")
    rating: Optional[str] = Field(None, description="Rating filter")
    tenor: Optional[str] = Field(None, description="Tenor filter")
    view: str = Field("liquidity", description="View type: liquidity or bondx")
    view_type: ViewType = Field(ViewType.PROFESSIONAL, description="View type for role-based access")

class HeatmapCell(BaseModel):
    """Single heatmap cell."""
    sector: str = Field(..., description="Sector name")
    rating: str = Field(..., description="Rating grade")
    tenor: str = Field(..., description="Tenor bucket")
    value: float = Field(..., description="Aggregated value")
    count: int = Field(..., description="Number of instruments")
    trend: Optional[str] = Field(None, description="Trend direction")
    confidence: float = Field(..., ge=0, le=1, description="Aggregation confidence")

class HeatmapResponse(BaseModel):
    """Response containing heatmap data."""
    success: bool = Field(True, description="Request success status")
    message: str = Field("Success", description="Response message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    data: List[HeatmapCell] = Field(..., description="Heatmap data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

# WebSocket Schemas
class PulseWebSocketMessage(BaseModel):
    """WebSocket message for liquidity pulse updates."""
    type: str = Field(..., description="Message type: snapshot, delta, forecast_update, alert")
    isin: str = Field(..., description="ISIN identifier")
    sequence_number: int = Field(..., description="Sequence number for ordering")
    timestamp: datetime = Field(..., description="Message timestamp")
    payload: Union[LiquidityPulse, Dict[str, Any]] = Field(..., description="Message payload")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracking")

# Validation schemas
class PulseValidationRequest(BaseModel):
    """Request for pulse validation."""
    isin: str = Field(..., description="ISIN to validate")
    reference_data: Optional[Dict[str, Any]] = Field(None, description="Reference data for validation")
    validation_type: str = Field("backtest", description="Validation type")

class PulseValidationResponse(BaseModel):
    """Response containing validation results."""
    success: bool = Field(True, description="Validation success status")
    message: str = Field("Success", description="Response message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    validation_results: Dict[str, Any] = Field(..., description="Validation results")
    recommendations: List[str] = Field(..., description="Improvement recommendations")
