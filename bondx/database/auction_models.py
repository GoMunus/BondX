"""
Database models for BondX Auction Engine and Settlement Systems.

This module contains all database models for the auction engine, fractional ownership,
settlement systems, and related infrastructure required for Phase 3.
"""

import enum
from datetime import date, datetime, time
from decimal import Decimal
from typing import Optional, List

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
    CheckConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import text

from .base import Base
from .models import BaseModel, AuditModel


class AuctionType(enum.Enum):
    """Types of auction mechanisms."""
    
    DUTCH = "DUTCH"  # Dutch auction (descending price)
    ENGLISH = "ENGLISH"  # English auction (ascending price)
    SEALED_BID = "SEALED_BID"  # Sealed bid auction
    MULTI_ROUND = "MULTI_ROUND"  # Multi-round auction
    HYBRID = "HYBRID"  # Hybrid auction mechanism


class AuctionStatus(enum.Enum):
    """Auction status indicators."""
    
    DRAFT = "DRAFT"  # Auction in preparation
    ANNOUNCED = "ANNOUNCED"  # Auction announced to market
    BIDDING_OPEN = "BIDDING_OPEN"  # Bidding period active
    BIDDING_CLOSED = "BIDDING_CLOSED"  # Bidding period closed
    PROCESSING = "PROCESSING"  # Processing bids and determining results
    SETTLED = "SETTLED"  # Auction completed and settled
    CANCELLED = "CANCELLED"  # Auction cancelled
    FAILED = "FAILED"  # Auction failed to meet criteria


class BidStatus(enum.Enum):
    """Bid status indicators."""
    
    PENDING = "PENDING"  # Bid submitted and pending
    ACCEPTED = "ACCEPTED"  # Bid accepted
    REJECTED = "REJECTED"  # Bid rejected
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # Bid partially filled
    FILLED = "FILLED"  # Bid completely filled
    CANCELLED = "CANCELLED"  # Bid cancelled
    EXPIRED = "EXPIRED"  # Bid expired


class SettlementStatus(enum.Enum):
    """Settlement status indicators."""
    
    PENDING = "PENDING"  # Settlement pending
    IN_PROGRESS = "IN_PROGRESS"  # Settlement in progress
    COMPLETED = "COMPLETED"  # Settlement completed
    FAILED = "FAILED"  # Settlement failed
    PARTIAL = "PARTIAL"  # Partial settlement
    CANCELLED = "CANCELLED"  # Settlement cancelled


class ParticipantType(enum.Enum):
    """Types of auction participants."""
    
    PRIMARY_DEALER = "PRIMARY_DEALER"  # Primary dealer
    INSTITUTIONAL = "INSTITUTIONAL"  # Institutional investor
    RETAIL = "RETAIL"  # Retail investor
    FOREIGN = "FOREIGN"  # Foreign investor
    MARKET_MAKER = "MARKET_MAKER"  # Market maker
    PROPRIETARY = "PROPRIETARY"  # Proprietary trading


class OrderType(enum.Enum):
    """Types of trading orders."""
    
    MARKET = "MARKET"  # Market order
    LIMIT = "LIMIT"  # Limit order
    STOP = "STOP"  # Stop order
    STOP_LIMIT = "STOP_LIMIT"  # Stop limit order
    TIME_IN_FORCE = "TIME_IN_FORCE"  # Time in force order


class TimeInForce(enum.Enum):
    """Time in force specifications."""
    
    DAY = "DAY"  # Valid for the day
    GOOD_TILL_CANCELLED = "GOOD_TILL_CANCELLED"  # Good till cancelled
    IMMEDIATE_OR_CANCEL = "IMMEDIATE_OR_CANCEL"  # Immediate or cancel
    FILL_OR_KILL = "FILL_OR_KILL"  # Fill or kill
    GOOD_TILL_DATE = "GOOD_TILL_DATE"  # Good till specific date


class Auction(BaseModel):
    """Auction configuration and management."""
    
    # Basic auction information
    auction_code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    auction_name: Mapped[str] = mapped_column(String(500), nullable=False)
    auction_type: Mapped[AuctionType] = mapped_column(Enum(AuctionType), nullable=False, index=True)
    status: Mapped[AuctionStatus] = mapped_column(Enum(AuctionStatus), nullable=False, default=AuctionStatus.DRAFT, index=True)
    
    # Auction parameters
    total_lot_size: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False)
    minimum_lot_size: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    lot_size_increment: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    
    # Price parameters
    reserve_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    minimum_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    maximum_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    price_increment: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)
    
    # Timing parameters
    announcement_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    bidding_start_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    bidding_end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    settlement_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    
    # Eligibility and restrictions
    eligible_participants: Mapped[List[str]] = mapped_column(JSONB, nullable=True)  # ["PD", "INST", "RETAIL"]
    maximum_allocation_per_participant: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)
    minimum_participation_requirement: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)
    
    # Auction rules
    allocation_method: Mapped[str] = mapped_column(String(100), default="PRO_RATA", nullable=False)
    priority_rules: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    anti_manipulation_rules: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Results
    clearing_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    total_bids_received: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_lots_allocated: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 2), nullable=True)
    bid_to_cover_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Foreign keys
    instrument_id: Mapped[int] = mapped_column(Integer, ForeignKey("instrument.id"), nullable=False, index=True)
    
    # Relationships
    instrument = relationship("Instrument")
    bids = relationship("Bid", back_populates="auction")
    allocations = relationship("Allocation", back_populates="auction")
    settlements = relationship("Settlement", back_populates="auction")
    
    # Indexes
    __table_args__ = (
        Index("ix_auction_status_type", "status", "auction_type"),
        Index("ix_auction_instrument_status", "instrument_id", "status"),
        Index("ix_auction_bidding_time", "bidding_start_time", "bidding_end_time"),
        Index("ix_auction_settlement_date", "settlement_date"),
    )


class Participant(BaseModel):
    """Auction participants and their eligibility."""
    
    # Basic information
    participant_code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    participant_name: Mapped[str] = mapped_column(String(500), nullable=False)
    participant_type: Mapped[ParticipantType] = mapped_column(Enum(ParticipantType), nullable=False, index=True)
    
    # Contact information
    contact_email: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    contact_phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    contact_address: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Regulatory information
    regulatory_license: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    tax_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    compliance_status: Mapped[str] = mapped_column(String(50), default="ACTIVE", nullable=False, index=True)
    
    # Financial parameters
    credit_limit: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 2), nullable=True)
    margin_requirement: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)
    maximum_position_limit: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 2), nullable=True)
    
    # Eligibility
    eligible_auction_types: Mapped[List[str]] = mapped_column(JSONB, nullable=True)
    eligible_instruments: Mapped[List[str]] = mapped_column(JSONB, nullable=True)
    trading_restrictions: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    bids = relationship("Bid", back_populates="participant")
    allocations = relationship("Allocation", back_populates="participant")
    positions = relationship("Position", back_populates="participant")
    settlements = relationship("Settlement", back_populates="participant")
    
    # Indexes
    __table_args__ = (
        Index("ix_participant_type_status", "participant_type", "compliance_status"),
        Index("ix_participant_compliance", "compliance_status"),
    )


class Bid(BaseModel):
    """Bid submissions for auctions."""
    
    # Bid information
    bid_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    bid_price: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False, index=True)
    bid_quantity: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False)
    bid_yield: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)
    
    # Bid status
    status: Mapped[BidStatus] = mapped_column(Enum(BidStatus), nullable=False, default=BidStatus.PENDING, index=True)
    allocation_quantity: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 2), nullable=True)
    allocation_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    
    # Timing
    submission_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    last_modified: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Order details
    order_type: Mapped[OrderType] = mapped_column(Enum(OrderType), nullable=False, default=OrderType.LIMIT)
    time_in_force: Mapped[TimeInForce] = mapped_column(Enum(TimeInForce), nullable=False, default=TimeInForce.DAY)
    
    # Risk checks
    risk_check_passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    risk_check_details: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Foreign keys
    auction_id: Mapped[int] = mapped_column(Integer, ForeignKey("auction.id"), nullable=False, index=True)
    participant_id: Mapped[int] = mapped_column(Integer, ForeignKey("participant.id"), nullable=False, index=True)
    
    # Relationships
    auction = relationship("Auction", back_populates="bids")
    participant = relationship("Participant", back_populates="bids")
    
    # Indexes
    __table_args__ = (
        Index("ix_bid_auction_status", "auction_id", "status"),
        Index("ix_bid_participant_status", "participant_id", "status"),
        Index("ix_bid_price_quantity", "bid_price", "bid_quantity"),
        Index("ix_bid_submission_time", "submission_time"),
    )


class Allocation(BaseModel):
    """Auction allocations and results."""
    
    # Allocation details
    allocation_quantity: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False)
    allocation_price: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    allocation_yield: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)
    
    # Allocation method
    allocation_method: Mapped[str] = mapped_column(String(100), nullable=False)
    allocation_priority: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    allocation_round: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Status
    is_confirmed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    confirmation_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Foreign keys
    auction_id: Mapped[int] = mapped_column(Integer, ForeignKey("auction.id"), nullable=False, index=True)
    participant_id: Mapped[int] = mapped_column(Integer, ForeignKey("participant.id"), nullable=False, index=True)
    bid_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("bid.id"), nullable=True)
    
    # Relationships
    auction = relationship("Auction", back_populates="allocations")
    participant = relationship("Participant", back_populates="allocations")
    bid = relationship("Bid")
    
    # Indexes
    __table_args__ = (
        Index("ix_allocation_auction_participant", "auction_id", "participant_id"),
        Index("ix_allocation_price_quantity", "allocation_price", "allocation_quantity"),
        Index("ix_allocation_confirmation", "is_confirmed"),
    )


class Position(BaseModel):
    """Fractional ownership positions and holdings."""
    
    # Position details
    position_quantity: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)  # High precision for fractional
    average_cost: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    current_market_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)
    
    # Position status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    position_type: Mapped[str] = mapped_column(String(50), default="LONG", nullable=False)  # LONG, SHORT
    
    # Cost basis
    total_cost: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    unrealized_pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)
    realized_pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), default=Decimal('0'), nullable=False)
    
    # Risk metrics
    duration_exposure: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)
    credit_exposure: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)
    liquidity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Foreign keys
    instrument_id: Mapped[int] = mapped_column(Integer, ForeignKey("instrument.id"), nullable=False, index=True)
    participant_id: Mapped[int] = mapped_column(Integer, ForeignKey("participant.id"), nullable=False, index=True)
    
    # Relationships
    instrument = relationship("Instrument")
    participant = relationship("Participant", back_populates="positions")
    trades = relationship("Trade", back_populates="position")
    
    # Indexes
    __table_args__ = (
        Index("ix_position_instrument_participant", "instrument_id", "participant_id"),
        Index("ix_position_quantity_value", "position_quantity", "current_market_value"),
        Index("ix_position_active", "is_active"),
    )


class Trade(BaseModel):
    """Secondary market trades and position adjustments."""
    
    # Trade details
    trade_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    trade_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # BUY, SELL, TRANSFER
    trade_quantity: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    trade_price: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    trade_value: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    
    # Trade status
    status: Mapped[str] = mapped_column(String(50), default="PENDING", nullable=False, index=True)
    execution_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    settlement_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Counterparty information
    counterparty_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("participant.id"), nullable=True)
    counterparty_name: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Order details
    order_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    order_type: Mapped[OrderType] = mapped_column(Enum(OrderType), nullable=False, default=OrderType.MARKET)
    time_in_force: Mapped[TimeInForce] = mapped_column(Enum(TimeInForce), nullable=False, default=TimeInForce.DAY)
    
    # Risk and compliance
    risk_check_passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    compliance_check_passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Foreign keys
    instrument_id: Mapped[int] = mapped_column(Integer, ForeignKey("instrument.id"), nullable=False, index=True)
    position_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("position.id"), nullable=True)
    buyer_id: Mapped[int] = mapped_column(Integer, ForeignKey("participant.id"), nullable=False, index=True)
    seller_id: Mapped[int] = mapped_column(Integer, ForeignKey("participant.id"), nullable=False, index=True)
    
    # Relationships
    instrument = relationship("Instrument")
    position = relationship("Position", back_populates="trades")
    buyer = relationship("Participant", foreign_keys=[buyer_id])
    seller = relationship("Participant", foreign_keys=[seller_id])
    counterparty = relationship("Participant", foreign_keys=[counterparty_id])
    
    # Indexes
    __table_args__ = (
        Index("ix_trade_instrument_status", "instrument_id", "status"),
        Index("ix_trade_buyer_seller", "buyer_id", "seller_id"),
        Index("ix_trade_execution_time", "execution_time"),
        Index("ix_trade_settlement_time", "settlement_time"),
    )


class Settlement(BaseModel):
    """Settlement and clearing records."""
    
    # Settlement details
    settlement_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    settlement_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # AUCTION, TRADE, COUPON
    settlement_amount: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    settlement_currency: Mapped[str] = mapped_column(String(3), default="INR", nullable=False)
    
    # Settlement status
    status: Mapped[SettlementStatus] = mapped_column(Enum(SettlementStatus), nullable=False, default=SettlementStatus.PENDING, index=True)
    settlement_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    actual_settlement_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    
    # Cash and securities
    cash_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)
    securities_quantity: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 6), nullable=True)
    
    # Settlement instructions
    settlement_instructions: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    bank_account_details: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    depository_details: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Risk and compliance
    risk_check_passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    compliance_check_passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Foreign keys
    auction_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("auction.id"), nullable=True, index=True)
    trade_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("trade.id"), nullable=True, index=True)
    participant_id: Mapped[int] = mapped_column(Integer, ForeignKey("participant.id"), nullable=False, index=True)
    
    # Relationships
    auction = relationship("Auction", back_populates="settlements")
    trade = relationship("Trade")
    participant = relationship("Participant", back_populates="settlements")
    
    # Indexes
    __table_args__ = (
        Index("ix_settlement_type_status", "settlement_type", "status"),
        Index("ix_settlement_date_status", "settlement_date", "status"),
        Index("ix_settlement_participant", "participant_id"),
    )


class CashFlow(BaseModel):
    """Cash flow tracking for fractional ownership."""
    
    # Cash flow details
    flow_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # COUPON, PRINCIPAL, DIVIDEND
    flow_amount: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    flow_currency: Mapped[str] = mapped_column(String(3), default="INR", nullable=False)
    
    # Timing
    due_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    payment_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    
    # Allocation
    total_eligible_quantity: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    participant_quantity: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    participant_amount: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    
    # Status
    is_paid: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    payment_status: Mapped[str] = mapped_column(String(50), default="PENDING", nullable=False, index=True)
    
    # Tax and compliance
    tax_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)
    tax_rate: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)
    compliance_status: Mapped[str] = mapped_column(String(50), default="COMPLIANT", nullable=False)
    
    # Foreign keys
    instrument_id: Mapped[int] = mapped_column(Integer, ForeignKey("instrument.id"), nullable=False, index=True)
    participant_id: Mapped[int] = mapped_column(Integer, ForeignKey("participant.id"), nullable=False, index=True)
    
    # Relationships
    instrument = relationship("Instrument")
    participant = relationship("Participant")
    
    # Indexes
    __table_args__ = (
        Index("ix_cash_flow_type_date", "flow_type", "due_date"),
        Index("ix_cash_flow_participant_status", "participant_id", "payment_status"),
        Index("ix_cash_flow_instrument_date", "instrument_id", "due_date"),
    )


class OrderBook(BaseModel):
    """Real-time order book for secondary market trading."""
    
    # Order details
    order_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    order_type: Mapped[OrderType] = mapped_column(Enum(OrderType), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False, index=True)  # BUY, SELL
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    
    # Order status
    status: Mapped[str] = mapped_column(String(50), default="ACTIVE", nullable=False, index=True)
    filled_quantity: Mapped[Decimal] = mapped_column(Numeric(20, 6), default=Decimal('0'), nullable=False)
    remaining_quantity: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    
    # Timing
    time_in_force: Mapped[TimeInForce] = mapped_column(Enum(TimeInForce), nullable=False, default=TimeInForce.DAY)
    expiry_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_modified: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    
    # Risk and compliance
    risk_check_passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    compliance_check_passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Foreign keys
    instrument_id: Mapped[int] = mapped_column(Integer, ForeignKey("instrument.id"), nullable=False, index=True)
    participant_id: Mapped[int] = mapped_column(Integer, ForeignKey("participant.id"), nullable=False, index=True)
    
    # Relationships
    instrument = relationship("Instrument")
    participant = relationship("Participant")
    
    # Indexes
    __table_args__ = (
        Index("ix_order_book_instrument_side", "instrument_id", "side"),
        Index("ix_order_book_price_quantity", "price", "quantity"),
        Index("ix_order_book_status_time", "status", "last_modified"),
        Index("ix_order_book_participant_status", "participant_id", "status"),
    )


class MarketData(BaseModel):
    """Real-time market data and analytics."""
    
    # Market data
    last_trade_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    last_trade_quantity: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 6), nullable=True)
    last_trade_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Bid-ask spread
    best_bid_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    best_bid_quantity: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 6), nullable=True)
    best_ask_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    best_ask_quantity: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 6), nullable=True)
    
    # Volume and liquidity
    total_volume: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 6), nullable=True)
    total_trades: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    open_interest: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 6), nullable=True)
    
    # Market indicators
    volatility: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    liquidity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    market_depth: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Data quality
    data_source: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    data_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    is_stale: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Foreign keys
    instrument_id: Mapped[int] = mapped_column(Integer, ForeignKey("instrument.id"), nullable=False, index=True)
    
    # Relationships
    instrument = relationship("Instrument")
    
    # Indexes
    __table_args__ = (
        Index("ix_market_data_instrument_timestamp", "instrument_id", "data_timestamp"),
        Index("ix_market_data_source_timestamp", "data_source", "data_timestamp"),
        Index("ix_market_data_price_volume", "last_trade_price", "total_volume"),
    )


# Export all models
__all__ = [
    "AuctionType",
    "AuctionStatus",
    "BidStatus",
    "SettlementStatus",
    "ParticipantType",
    "OrderType",
    "TimeInForce",
    "Auction",
    "Participant",
    "Bid",
    "Allocation",
    "Position",
    "Trade",
    "Settlement",
    "CashFlow",
    "OrderBook",
    "MarketData",
]
