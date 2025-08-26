"""
Liquidity-Risk Translator API Schemas for BondX

This module contains Pydantic schemas for the Liquidity-Risk Translator service,
ensuring proper validation and documentation of API requests and responses.
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
from pydantic import BaseModel, Field, validator, root_validator

# Enums
class LiquidityLevel(str, Enum):
    """Liquidity level classifications."""
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    ILLIQUID = "illiquid"

class ExitPath(str, Enum):
    """Available exit paths for bonds."""
    MARKET_MAKER = "market_maker"
    AUCTION = "auction"
    RFQ_BATCH = "rfq_batch"
    TOKENIZED_P2P = "tokenized_p2p"

class ExitPriority(str, Enum):
    """Exit priority levels for recommendation ranking."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    FALLBACK = "fallback"
    UNAVAILABLE = "unavailable"

class ExitConstraint(str, Enum):
    """Types of constraints that may limit exit paths."""
    INVENTORY_LIMIT = "inventory_limit"
    WINDOW_CLOSED = "window_closed"
    LOT_SIZE = "lot_size"
    RATING_RESTRICTION = "rating_restriction"
    TIME_CONSTRAINT = "time_constraint"
    REGULATORY = "regulatory"

class NarrativeMode(str, Enum):
    """Narrative generation modes."""
    RETAIL = "retail"
    PROFESSIONAL = "professional"
    COMPLIANCE = "compliance"

class DataFreshness(str, Enum):
    """Data freshness levels."""
    REAL_TIME = "real_time"
    FRESH = "fresh"
    RECENT = "recent"
    STALE = "stale"
    OUTDATED = "outdated"

# Base schemas
class BaseLiquidityRiskResponse(BaseModel):
    """Base response schema for liquidity-risk API."""
    success: bool = True
    message: str = "Success"
    timestamp: datetime = Field(default_factory=datetime.now)
    isin: str
    as_of: datetime

class ErrorLiquidityRiskResponse(BaseLiquidityRiskResponse):
    """Error response schema for liquidity-risk API."""
    success: bool = False
    error_code: str
    error_details: Optional[Dict[str, Any]] = None

# Risk assessment schemas
class RiskCategoryScoreSchema(BaseModel):
    """Risk score for a specific category."""
    name: str = Field(..., description="Risk category name")
    score_0_100: float = Field(..., ge=0, le=100, description="Risk score from 0-100")
    level: str = Field(..., description="Risk level: low, medium, high, critical")
    probability_note: str = Field(..., description="Explanation of risk probability")
    citations: List[str] = Field(..., description="Sources for risk assessment")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in assessment")

class RiskSummarySchema(BaseModel):
    """Risk summary for a bond."""
    overall_score: float = Field(..., ge=0, le=100, description="Overall risk score")
    categories: List[RiskCategoryScoreSchema] = Field(..., description="Risk category breakdown")
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence")
    citations: List[str] = Field(..., description="Sources for risk assessment")
    last_updated: datetime = Field(..., description="Last update timestamp")
    methodology_version: str = Field(..., description="Risk assessment methodology version")

# Liquidity profile schemas
class LiquidityProfileSchema(BaseModel):
    """Comprehensive liquidity profile for a bond."""
    liquidity_index: float = Field(..., ge=0, le=100, description="Liquidity index 0-100")
    spread_bps: float = Field(..., ge=0, description="Bid-ask spread in basis points")
    depth_score: float = Field(..., ge=0, le=100, description="Market depth score 0-100")
    turnover_rank: float = Field(..., ge=0, le=100, description="Turnover rank percentile")
    time_since_last_trade_s: int = Field(..., ge=0, description="Seconds since last trade")
    expected_time_to_exit_minutes: float = Field(..., ge=0, description="Expected time to exit")
    liquidity_level: LiquidityLevel = Field(..., description="Liquidity level classification")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in liquidity assessment")
    data_freshness: DataFreshness = Field(..., description="Data freshness indicator")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional liquidity data")

# Exit recommendation schemas
class ExitRecommendationSchema(BaseModel):
    """Complete exit recommendation for a specific path."""
    path: ExitPath = Field(..., description="Exit pathway")
    priority: ExitPriority = Field(..., description="Recommendation priority")
    expected_price: float = Field(..., description="Expected execution price")
    expected_spread_bps: float = Field(..., ge=0, description="Expected spread in basis points")
    fill_probability: float = Field(..., ge=0, le=1, description="Probability of successful execution")
    expected_time_to_exit_minutes: float = Field(..., ge=0, description="Expected time to exit")
    rationale: str = Field(..., description="Explanation for recommendation")
    constraints: List[ExitConstraint] = Field(default_factory=list, description="Path constraints")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in recommendation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional path data")

# Market data schemas
class MarketMicrostructureSchema(BaseModel):
    """Real-time market microstructure data."""
    timestamp: datetime = Field(..., description="Data timestamp")
    isin: str = Field(..., description="Bond ISIN")
    bid: float = Field(..., description="Best bid price")
    ask: float = Field(..., description="Best ask price")
    bid_size: float = Field(..., ge=0, description="Bid size")
    ask_size: float = Field(..., ge=0, description="Ask size")
    l2_depth_qty: float = Field(..., ge=0, description="Level 2 depth quantity")
    l2_levels: int = Field(..., ge=0, description="Number of L2 levels")
    trades_count: int = Field(..., ge=0, description="Trade count")
    vwap: float = Field(..., description="Volume weighted average price")
    volume_face: float = Field(..., ge=0, description="Face value volume")
    time_since_last_trade_s: int = Field(..., ge=0, description="Seconds since last trade")
    
    @validator('ask')
    def ask_must_be_greater_than_bid(cls, v, values):
        if 'bid' in values and v <= values['bid']:
            raise ValueError('Ask must be greater than bid')
        return v

class AuctionSignalsSchema(BaseModel):
    """Auction telemetry data."""
    timestamp: datetime = Field(..., description="Data timestamp")
    isin: str = Field(..., description="Bond ISIN")
    auction_id: str = Field(..., description="Auction identifier")
    lots_offered: int = Field(..., ge=0, description="Number of lots offered")
    bids_count: int = Field(..., ge=0, description="Number of bids received")
    demand_curve_points: List[Tuple[float, float]] = Field(..., description="Price-yield vs quantity points")
    clearing_price_estimate: float = Field(..., description="Estimated clearing price")
    next_window: datetime = Field(..., description="Next auction window")

class MarketMakerStateSchema(BaseModel):
    """Market maker telemetry data."""
    timestamp: datetime = Field(..., description="Data timestamp")
    isin: str = Field(..., description="Bond ISIN")
    mm_online: bool = Field(..., description="Market maker online status")
    mm_inventory_band: Tuple[float, float, float] = Field(..., description="Inventory band (low, target, high)")
    mm_min_spread_bps: float = Field(..., ge=0, description="Minimum spread in basis points")
    last_quote_spread_bps: float = Field(..., ge=0, description="Last quote spread in basis points")
    quotes_last_24h: int = Field(..., ge=0, description="Quotes in last 24 hours")
    
    @validator('mm_inventory_band')
    def validate_inventory_band(cls, v):
        if len(v) != 3:
            raise ValueError('Inventory band must have exactly 3 values')
        low, target, high = v
        if not (low <= target <= high):
            raise ValueError('Inventory band must be: low <= target <= high')
        return v

# Bond metadata schemas
class BondMetadataSchema(BaseModel):
    """Bond characteristics and metadata."""
    rating: str = Field(..., description="Credit rating")
    tenor: str = Field(..., description="Maturity tenor")
    issuer_class: str = Field(..., description="Issuer classification")
    coupon_rate: float = Field(..., description="Coupon rate percentage")
    maturity_date: date = Field(..., description="Maturity date")
    issue_size: float = Field(..., ge=0, description="Issue size in currency units")

# Request schemas
class LiquidityRiskRequestSchema(BaseModel):
    """Request schema for liquidity-risk translation."""
    isin: str = Field(..., description="Bond ISIN identifier")
    mode: NarrativeMode = Field(NarrativeMode.RETAIL, description="Narrative generation mode")
    detail: str = Field("summary", description="Detail level: summary or full")
    trade_size: float = Field(100000, ge=0, description="Trade size in currency units")

class RecomputeRequestSchema(BaseModel):
    """Request schema for triggering recomputation."""
    isins: List[str] = Field(..., min_items=1, description="List of ISINs to recompute")
    mode: str = Field("accurate", description="Recomputation mode")

class AuditRequestSchema(BaseModel):
    """Request schema for audit trail."""
    isin: str = Field(..., description="Bond ISIN identifier")
    as_of: Optional[datetime] = Field(None, description="Audit timestamp")

# Response schemas
class LiquidityRiskSummaryResponse(BaseLiquidityRiskResponse):
    """Summary response for liquidity-risk translation."""
    risk_summary: Dict[str, Union[float, str]] = Field(..., description="Risk summary")
    liquidity_profile: Dict[str, Union[float, str]] = Field(..., description="Liquidity profile")
    exit_recommendations: List[Dict[str, Union[str, float]]] = Field(..., description="Top exit recommendations")
    retail_narrative: str = Field(..., description="Plain-English narrative")
    confidence_overall: float = Field(..., ge=0, le=1, description="Overall confidence")
    data_freshness: DataFreshness = Field(..., description="Data freshness indicator")

class LiquidityRiskFullResponse(BaseLiquidityRiskResponse):
    """Full response for liquidity-risk translation."""
    risk_summary: RiskSummarySchema = Field(..., description="Complete risk summary")
    liquidity_profile: LiquidityProfileSchema = Field(..., description="Complete liquidity profile")
    exit_recommendations: List[ExitRecommendationSchema] = Field(..., description="All exit recommendations")
    retail_narrative: str = Field(..., description="Plain-English narrative")
    professional_summary: str = Field(..., description="Technical summary")
    risk_warnings: List[str] = Field(..., description="Risk warnings")
    confidence_overall: float = Field(..., ge=0, le=1, description="Overall confidence")
    data_freshness: DataFreshness = Field(..., description="Data freshness indicator")
    inputs_hash: str = Field(..., description="Inputs hash for auditability")
    model_versions: Dict[str, str] = Field(..., description="Model versions used")
    caveats: List[str] = Field(..., description="Disclaimers and caveats")

class RecomputeResponseSchema(BaseModel):
    """Response schema for recomputation request."""
    success: bool = Field(..., description="Request success status")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(..., description="Response timestamp")
    requested_isins: List[str] = Field(..., description="Requested ISINs")
    mode: str = Field(..., description="Requested mode")

class AuditResponseSchema(BaseModel):
    """Response schema for audit trail."""
    isin: str = Field(..., description="Bond ISIN")
    audit_timestamp: datetime = Field(..., description="Audit timestamp")
    requested_as_of: Optional[datetime] = Field(None, description="Requested timestamp")
    data_lineage: Dict[str, str] = Field(..., description="Data source lineage")
    model_versions: Dict[str, str] = Field(..., description="Model versions")
    data_quality_metrics: Dict[str, str] = Field(..., description="Data quality indicators")
    computation_metadata: Dict[str, Union[int, bool]] = Field(..., description="Computation details")

class HealthResponseSchema(BaseModel):
    """Response schema for health check."""
    service: str = Field(..., description="Service name")
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    components: Dict[str, str] = Field(..., description="Component statuses")
    version: str = Field(..., description="Service version")

# WebSocket schemas
class WebSocketMessageSchema(BaseModel):
    """WebSocket message schema."""
    type: str = Field(..., description="Message type")
    topic: str = Field(..., description="Message topic")
    seq: int = Field(..., description="Sequence number")
    ts: str = Field(..., description="Timestamp ISO string")
    payload: Dict[str, Any] = Field(..., description="Message payload")
    meta: Optional[Dict[str, Any]] = Field(None, description="Message metadata")
    correlation_id: Optional[str] = Field(None, description="Correlation identifier")

class WebSocketSubscriptionSchema(BaseModel):
    """WebSocket subscription request schema."""
    action: str = Field(..., description="Action: subscribe or unsubscribe")
    topic: str = Field(..., description="Topic to subscribe/unsubscribe")
    user_id: Optional[str] = Field(None, description="User identifier")

# Validation schemas
class LiquidityRiskValidationSchema(BaseModel):
    """Schema for validating liquidity-risk data."""
    isin: str = Field(..., description="Bond ISIN")
    risk_scores: Dict[str, float] = Field(..., description="Risk category scores")
    liquidity_metrics: Dict[str, Union[float, int]] = Field(..., description="Liquidity metrics")
    exit_paths: List[str] = Field(..., description="Available exit paths")
    
    @validator('risk_scores')
    def validate_risk_scores(cls, v):
        for category, score in v.items():
            if not (0 <= score <= 100):
                raise ValueError(f'Risk score for {category} must be between 0 and 100')
        return v
    
    @validator('liquidity_metrics')
    def validate_liquidity_metrics(cls, v):
        required_keys = ['spread_bps', 'depth_score', 'turnover_rank']
        for key in required_keys:
            if key not in v:
                raise ValueError(f'Missing required liquidity metric: {key}')
        return v

# Utility schemas
class PaginationSchema(BaseModel):
    """Pagination parameters."""
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(20, ge=1, le=100, description="Page size")
    total: Optional[int] = Field(None, description="Total number of items")

class FilterSchema(BaseModel):
    """Filter parameters for queries."""
    rating_min: Optional[str] = Field(None, description="Minimum rating filter")
    rating_max: Optional[str] = Field(None, description="Maximum rating filter")
    liquidity_level: Optional[LiquidityLevel] = Field(None, description="Liquidity level filter")
    risk_level: Optional[str] = Field(None, description="Risk level filter")
    issuer_class: Optional[str] = Field(None, description="Issuer class filter")
    tenor_range: Optional[str] = Field(None, description="Tenor range filter")

class SortSchema(BaseModel):
    """Sorting parameters."""
    field: str = Field(..., description="Field to sort by")
    direction: str = Field("asc", regex="^(asc|desc)$", description="Sort direction")

# Batch operation schemas
class BatchLiquidityRiskRequestSchema(BaseModel):
    """Request schema for batch liquidity-risk analysis."""
    isins: List[str] = Field(..., min_items=1, max_items=100, description="List of ISINs")
    mode: NarrativeMode = Field(NarrativeMode.RETAIL, description="Narrative mode")
    detail: str = Field("summary", description="Detail level")
    trade_size: float = Field(100000, ge=0, description="Trade size")
    filters: Optional[FilterSchema] = Field(None, description="Filter criteria")
    sort: Optional[SortSchema] = Field(None, description="Sort criteria")
    pagination: Optional[PaginationSchema] = Field(None, description="Pagination")

class BatchLiquidityRiskResponseSchema(BaseModel):
    """Response schema for batch liquidity-risk analysis."""
    results: List[Union[LiquidityRiskSummaryResponse, LiquidityRiskFullResponse]] = Field(..., description="Analysis results")
    pagination: PaginationSchema = Field(..., description="Pagination information")
    summary: Dict[str, Any] = Field(..., description="Batch summary statistics")
    processing_time_ms: int = Field(..., description="Total processing time")
    errors: List[Dict[str, str]] = Field(default_factory=list, description="Processing errors")
