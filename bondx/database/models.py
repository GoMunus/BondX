"""
Database models for BondX Backend.

This module contains all database models for the bond marketplace including
instruments, market quotes, yield curves, corporate actions, and more.
"""

import enum
from datetime import date, datetime, time
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    String,
    Text,
    Time,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import text

from .base import Base


class DayCountConvention(enum.Enum):
    """Day count conventions used in bond calculations."""
    
    ACT_ACT = "ACT/ACT"  # Actual/Actual (used for government securities)
    ACT_365 = "ACT/365"  # Actual/365
    ACT_360 = "ACT/360"  # Actual/360
    THIRTY_360 = "30/360"  # 30/360 (used for most corporate bonds)
    THIRTY_365 = "30/365"  # 30/365


class CouponType(enum.Enum):
    """Types of coupon structures."""
    
    FIXED = "FIXED"  # Fixed rate coupon
    FLOATING = "FLOATING"  # Floating rate coupon
    ZERO_COUPON = "ZERO_COUPON"  # Zero coupon bond
    INFLATION_INDEXED = "INFLATION_INDEXED"  # Inflation-indexed bond
    STEP_UP = "STEP_UP"  # Step-up coupon
    STEP_DOWN = "STEP_DOWN"  # Step-down coupon


class BondType(enum.Enum):
    """Types of bonds."""
    
    GOVERNMENT_SECURITY = "GOVERNMENT_SECURITY"  # G-sec
    STATE_DEVELOPMENT_LOAN = "STATE_DEVELOPMENT_LOAN"  # SDL
    CORPORATE_BOND = "CORPORATE_BOND"  # Corporate bond
    MUNICIPAL_BOND = "MUNICIPAL_BOND"  # Municipal bond
    BANK_BOND = "BANK_BOND"  # Bank bond
    PSU_BOND = "PSU_BOND"  # Public sector undertaking bond
    SUPRANATIONAL = "SUPRANATIONAL"  # Supranational bond


class RatingAgency(enum.Enum):
    """Credit rating agencies."""
    
    CRISIL = "CRISIL"
    ICRA = "ICRA"
    CARE = "CARE"
    MOODYS = "MOODYS"
    FITCH = "FITCH"
    INDIA_RATINGS = "INDIA_RATINGS"
    BRICKWORK = "BRICKWORK"


class RatingGrade(enum.Enum):
    """Credit rating grades."""
    
    AAA = "AAA"
    AA_PLUS = "AA+"
    AA = "AA"
    AA_MINUS = "AA-"
    A_PLUS = "A+"
    A = "A"
    A_MINUS = "A-"
    BBB_PLUS = "BBB+"
    BBB = "BBB"
    BBB_MINUS = "BBB-"
    BB_PLUS = "BB+"
    BB = "BB"
    BB_MINUS = "BB-"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"
    CCC_PLUS = "CCC+"
    CCC = "CCC"
    CCC_MINUS = "CCC-"
    CC = "CC"
    C = "C"
    D = "D"
    NR = "NR"  # Not Rated


class SettlementCycle(enum.Enum):
    """Settlement cycles for Indian markets."""
    
    T_PLUS_0 = "T+0"  # Same day settlement
    T_PLUS_1 = "T+1"  # Next day settlement
    T_PLUS_2 = "T+2"  # Two days settlement (most common)
    T_PLUS_3 = "T+3"  # Three days settlement


class MarketStatus(enum.Enum):
    """Market status indicators."""
    
    ACTIVE = "ACTIVE"  # Actively traded
    INACTIVE = "INACTIVE"  # Not actively traded
    SUSPENDED = "SUSPENDED"  # Trading suspended
    DELISTED = "DELISTED"  # Delisted from exchange
    MATURED = "MATURED"  # Bond has matured


class BaseModel(Base):
    """Base model with common fields."""
    
    __abstract__ = True
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        return cls.__name__.lower()


class AuditModel(BaseModel):
    """Base model with audit fields."""
    
    __abstract__ = True
    
    created_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    updated_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


class Issuer(BaseModel):
    """Bond issuer information."""
    
    name: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    legal_name: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    issuer_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    sector: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    subsector: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    country: Mapped[str] = mapped_column(String(100), default="India", nullable=False)
    state: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    city: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    registration_number: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    tax_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    website: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    contact_email: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    contact_phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Relationships
    instruments = relationship("Instrument", back_populates="issuer")
    ratings = relationship("CreditRating", back_populates="issuer")
    
    # Indexes
    __table_args__ = (
        Index("ix_issuer_name_type", "name", "issuer_type"),
        Index("ix_issuer_sector_subsector", "sector", "subsector"),
    )


class Instrument(BaseModel):
    """Bond instrument information."""
    
    isin: Mapped[str] = mapped_column(String(12), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    short_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Basic bond information
    bond_type: Mapped[BondType] = mapped_column(Enum(BondType), nullable=False, index=True)
    coupon_type: Mapped[CouponType] = mapped_column(Enum(CouponType), nullable=False, index=True)
    face_value: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    issue_size: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 2), nullable=True)
    minimum_lot_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Coupon information
    coupon_rate: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)
    coupon_frequency: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # months
    first_coupon_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    last_coupon_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    
    # Maturity information
    issue_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    maturity_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    
    # Day count and settlement
    day_count_convention: Mapped[DayCountConvention] = mapped_column(
        Enum(DayCountConvention), nullable=False, default=DayCountConvention.THIRTY_360
    )
    settlement_cycle: Mapped[SettlementCycle] = mapped_column(
        Enum(SettlementCycle), nullable=False, default=SettlementCycle.T_PLUS_2
    )
    
    # Market information
    market_status: Mapped[MarketStatus] = mapped_column(
        Enum(MarketStatus), nullable=False, default=MarketStatus.ACTIVE
    )
    listing_exchanges: Mapped[list] = mapped_column(JSONB, nullable=True)  # ["NSE", "BSE"]
    
    # Embedded options
    is_callable: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_putable: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    call_schedule: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    put_schedule: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Sinking fund
    has_sinking_fund: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    sinking_fund_schedule: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Reference rates for floating bonds
    reference_rate: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    reference_rate_spread: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)
    
    # Inflation indexing
    inflation_index: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    inflation_lag: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # months
    
    # Additional metadata
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prospectus_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    regulatory_approval: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    
    # Foreign keys
    issuer_id: Mapped[int] = mapped_column(Integer, ForeignKey("issuer.id"), nullable=False, index=True)
    
    # Relationships
    issuer = relationship("Issuer", back_populates="instruments")
    market_quotes = relationship("MarketQuote", back_populates="instrument")
    ratings = relationship("CreditRating", back_populates="instrument")
    corporate_actions = relationship("CorporateAction", back_populates="instrument")
    cash_flows = relationship("CashFlow", back_populates="instrument")
    
    # Indexes
    __table_args__ = (
        Index("ix_instrument_maturity_date", "maturity_date"),
        Index("ix_instrument_issue_date", "issue_date"),
        Index("ix_instrument_bond_type_coupon_type", "bond_type", "coupon_type"),
        Index("ix_instrument_market_status", "market_status"),
        Index("ix_instrument_issuer_maturity", "issuer_id", "maturity_date"),
    )


class MarketQuote(BaseModel):
    """Real-time market quotes for bonds."""
    
    # Quote information
    bid_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    ask_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    last_trade_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    volume: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    trades_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Yield calculations
    yield_to_maturity: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)
    yield_to_call: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)
    yield_to_put: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)
    current_yield: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)
    
    # Price calculations
    clean_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    dirty_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    accrued_interest: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    
    # Risk metrics
    modified_duration: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)
    macaulay_duration: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)
    convexity: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6), nullable=True)
    
    # Spread metrics
    option_adjusted_spread: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)
    z_spread: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)
    credit_spread: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)
    
    # Market data
    quote_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    data_source: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    exchange: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    
    # Quality indicators
    is_stale: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    data_quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Foreign keys
    instrument_id: Mapped[int] = mapped_column(Integer, ForeignKey("instrument.id"), nullable=False, index=True)
    
    # Relationships
    instrument = relationship("Instrument", back_populates="market_quotes")
    
    # Indexes
    __table_args__ = (
        Index("ix_market_quote_timestamp", "quote_timestamp"),
        Index("ix_market_quote_data_source", "data_source"),
        Index("ix_market_quote_instrument_timestamp", "instrument_id", "quote_timestamp"),
        Index("ix_market_quote_yield", "yield_to_maturity"),
        Index("ix_market_quote_price", "clean_price"),
    )


class YieldCurve(BaseModel):
    """Yield curve data for different market segments."""
    
    curve_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    currency: Mapped[str] = mapped_column(String(3), default="INR", nullable=False)
    curve_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    
    # Term structure data
    tenors: Mapped[list] = mapped_column(JSONB, nullable=False)  # [1, 3, 6, 12, 24, 36, 60, 120, 240, 360]
    zero_rates: Mapped[list] = mapped_column(JSONB, nullable=False)  # Corresponding zero rates
    par_rates: Mapped[list] = mapped_column(JSONB, nullable=True)  # Par rates
    forward_rates: Mapped[list] = mapped_column(JSONB, nullable=True)  # Forward rates
    discount_factors: Mapped[list] = mapped_column(JSONB, nullable=True)  # Discount factors
    
    # Curve characteristics
    interpolation_method: Mapped[str] = mapped_column(String(50), default="cubic", nullable=False)
    extrapolation_method: Mapped[str] = mapped_column(String(50), default="flat", nullable=False)
    
    # Data quality
    data_source: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence_interval: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index("ix_yield_curve_type_date", "curve_type", "curve_date"),
        Index("ix_yield_curve_currency", "currency"),
        Index("ix_yield_curve_date", "curve_date"),
    )


class CreditRating(BaseModel):
    """Credit ratings from various agencies."""
    
    rating: Mapped[RatingGrade] = mapped_column(Enum(RatingGrade), nullable=False, index=True)
    rating_outlook: Mapped[str] = mapped_column(String(20), nullable=False, index=True)  # Positive, Stable, Negative
    rating_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    effective_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    
    # Rating details
    rating_agency: Mapped[RatingAgency] = mapped_column(Enum(RatingAgency), nullable=False, index=True)
    rating_rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rating_report_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Watch status
    is_on_watch: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    watch_direction: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # Positive, Negative, Developing
    
    # Foreign keys
    issuer_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("issuer.id"), nullable=True, index=True)
    instrument_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("instrument.id"), nullable=True, index=True)
    
    # Relationships
    issuer = relationship("Issuer", back_populates="ratings")
    instrument = relationship("Instrument", back_populates="ratings")
    
    # Indexes
    __table_args__ = (
        Index("ix_credit_rating_agency_date", "rating_agency", "rating_date"),
        Index("ix_credit_rating_grade", "rating"),
        Index("ix_credit_rating_outlook", "rating_outlook"),
        Index("ix_credit_rating_watch", "is_on_watch", "watch_direction"),
    )


class CorporateAction(BaseModel):
    """Corporate actions affecting bonds."""
    
    action_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    action_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    ex_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    record_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    
    # Action details
    description: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Impact
    impact_on_price: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # Positive, Negative, Neutral
    impact_on_yield: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(50), default="Announced", nullable=False, index=True)
    
    # Foreign keys
    instrument_id: Mapped[int] = mapped_column(Integer, ForeignKey("instrument.id"), nullable=False, index=True)
    
    # Relationships
    instrument = relationship("Instrument", back_populates="corporate_actions")
    
    # Indexes
    __table_args__ = (
        Index("ix_corporate_action_type_date", "action_type", "action_date"),
        Index("ix_corporate_action_status", "status"),
        Index("ix_corporate_action_instrument", "instrument_id"),
    )


class CashFlow(BaseModel):
    """Projected cash flows for bonds."""
    
    payment_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    payment_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # Coupon, Principal, Call, Put
    
    # Cash flow amounts
    coupon_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)
    principal_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)
    total_amount: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    
    # Calculation details
    days_since_last_payment: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    days_to_next_payment: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Embedded option details
    is_call_exercise: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_put_exercise: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Foreign keys
    instrument_id: Mapped[int] = mapped_column(Integer, ForeignKey("instrument.id"), nullable=False, index=True)
    
    # Relationships
    instrument = relationship("Instrument", back_populates="cash_flows")
    
    # Indexes
    __table_args__ = (
        Index("ix_cash_flow_date", "payment_date"),
        Index("ix_cash_flow_type", "payment_type"),
        Index("ix_cash_flow_instrument_date", "instrument_id", "payment_date"),
    )


class MacroIndicator(BaseModel):
    """Macroeconomic indicators affecting bond markets."""
    
    indicator_name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    indicator_value: Mapped[Decimal] = mapped_column(Numeric(15, 6), nullable=False)
    indicator_unit: Mapped[str] = mapped_column(String(50), nullable=False)
    indicator_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    
    # Data source and quality
    data_source: Mapped[str] = mapped_column(String(100), nullable=False)
    frequency: Mapped[str] = mapped_column(String(20), nullable=False)  # Daily, Weekly, Monthly, Quarterly
    revision_number: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # Additional metadata
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Indexes
    __table_args__ = (
        Index("ix_macro_indicator_name_date", "indicator_name", "indicator_date"),
        Index("ix_macro_indicator_category", "category"),
        Index("ix_macro_indicator_source", "data_source"),
    )


class MarketHoliday(BaseModel):
    """Market holidays and trading calendar."""
    
    holiday_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    holiday_name: Mapped[str] = mapped_column(String(200), nullable=False)
    holiday_type: Mapped[str] = mapped_column(String(100), nullable=False)  # National, Religious, Exchange-specific
    
    # Trading status
    is_trading_holiday: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_settlement_holiday: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Market segments affected
    affected_markets: Mapped[list] = mapped_column(JSONB, nullable=True)  # ["NSE", "BSE", "RBI"]
    
    # Indexes
    __table_args__ = (
        Index("ix_market_holiday_date", "holiday_date"),
        Index("ix_market_holiday_type", "holiday_type"),
        Index("ix_market_holiday_trading", "is_trading_holiday"),
    )


class SettlementCycle(BaseModel):
    """Settlement cycle definitions."""
    
    cycle_name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    cycle_days: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Applicable instruments
    applicable_instruments: Mapped[list] = mapped_column(JSONB, nullable=True)  # ["GOVT", "CORP", "MUNI"]
    
    # Market hours
    market_open_time: Mapped[Optional[time]] = mapped_column(Time, nullable=True)
    market_close_time: Mapped[Optional[time]] = mapped_column(Time, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index("ix_settlement_cycle_days", "cycle_days"),
    )


class DataQualityLog(BaseModel):
    """Data quality monitoring and validation logs."""
    
    data_source: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    data_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    
    # Quality metrics
    total_records: Mapped[int] = mapped_column(Integer, nullable=False)
    valid_records: Mapped[int] = mapped_column(Integer, nullable=False)
    invalid_records: Mapped[int] = mapped_column(Integer, nullable=False)
    missing_records: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Quality score
    quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Validation details
    validation_errors: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    outlier_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Timestamps
    data_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    validation_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    
    # Indexes
    __table_args__ = (
        Index("ix_data_quality_source_type", "data_source", "data_type"),
        Index("ix_data_quality_timestamp", "data_timestamp"),
        Index("ix_data_quality_score", "quality_score"),
    )


class AuditLog(AuditModel):
    """Audit trail for regulatory compliance."""
    
    table_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    record_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    
    # Action details
    action: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # INSERT, UPDATE, DELETE
    old_values: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    new_values: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Context
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Additional metadata
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index("ix_audit_log_table_record", "table_name", "record_id"),
        Index("ix_audit_log_action", "action"),
        Index("ix_audit_log_created_at", "created_at"),
        Index("ix_audit_log_created_by", "created_by"),
    )


# Export all models
__all__ = [
    "BaseModel",
    "AuditModel",
    "Issuer",
    "Instrument",
    "MarketQuote",
    "YieldCurve",
    "CreditRating",
    "CorporateAction",
    "CashFlow",
    "MacroIndicator",
    "MarketHoliday",
    "SettlementCycle",
    "DataQualityLog",
    "AuditLog",
    "DayCountConvention",
    "CouponType",
    "BondType",
    "RatingAgency",
    "RatingGrade",
    "SettlementCycle",
    "MarketStatus",
]
