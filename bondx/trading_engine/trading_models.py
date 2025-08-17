"""
Trading Models for BondX Trading Engine.

This module defines all data models and enums used in the trading system.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from uuid import uuid4


class OrderType(Enum):
    """Types of trading orders."""
    
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LIMIT = "STOP_LIMIT"
    FILL_OR_KILL = "FILL_OR_KILL"
    IMMEDIATE_OR_CANCEL = "IMMEDIATE_OR_CANCEL"
    GOOD_TILL_CANCELLED = "GOOD_TILL_CANCELLED"
    DAY_ORDER = "DAY_ORDER"


class OrderSide(Enum):
    """Order side (buy/sell)."""
    
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status throughout its lifecycle."""
    
    PENDING = "PENDING"
    VALIDATED = "VALIDATED"
    ACTIVE = "ACTIVE"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TimeInForce(Enum):
    """Time in force for orders."""
    
    IMMEDIATE = "IMMEDIATE"
    DAY = "DAY"
    GOOD_TILL_CANCELLED = "GOOD_TILL_CANCELLED"
    FILL_OR_KILL = "FILL_OR_KILL"
    IMMEDIATE_OR_CANCEL = "IMMEDIATE_OR_CANCEL"


class OrderPriority(Enum):
    """Order priority for matching."""
    
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"


class MarketMakerStatus(Enum):
    """Market maker operational status."""
    
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    SUSPENDED = "SUSPENDED"
    MAINTENANCE = "MAINTENANCE"


@dataclass
class Order:
    """Trading order representation."""
    
    order_id: str = field(default_factory=lambda: str(uuid4()))
    participant_id: int = None
    bond_id: str = None
    order_type: OrderType = OrderType.LIMIT
    side: OrderSide = OrderSide.BUY
    quantity: Decimal = Decimal('0')
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    priority: OrderPriority = OrderPriority.NORMAL
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal('0')
    average_fill_price: Optional[Decimal] = None
    remaining_quantity: Decimal = Decimal('0')
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    risk_check_passed: bool = False
    risk_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.price is None:
            raise ValueError("Limit orders must have a price")
        
        if self.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError("Stop orders must have a stop price")
        
        self.remaining_quantity = self.quantity - self.filled_quantity


@dataclass
class Trade:
    """Trade execution record."""
    
    trade_id: str = field(default_factory=lambda: str(uuid4()))
    order_id: str = None
    participant_id: int = None
    bond_id: str = None
    side: OrderSide = None
    quantity: Decimal = Decimal('0')
    price: Decimal = Decimal('0')
    trade_value: Decimal = Decimal('0')
    execution_time: datetime = field(default_factory=datetime.utcnow)
    trade_type: str = "REGULAR"  # REGULAR, AUCTION, BLOCK
    counterparty_id: Optional[int] = None
    venue: str = "SECONDARY_MARKET"
    fees: Decimal = Decimal('0')
    taxes: Decimal = Decimal('0')
    net_value: Decimal = Decimal('0')
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived fields."""
        self.trade_value = self.quantity * self.price
        self.net_value = self.trade_value - self.fees - self.taxes


@dataclass
class OrderBookEntry:
    """Order book entry for price level."""
    
    price: Decimal
    total_quantity: Decimal
    order_count: int
    side: OrderSide
    orders: List[Order] = field(default_factory=list)
    
    def add_order(self, order: Order):
        """Add order to this price level."""
        self.orders.append(order)
        self.total_quantity += order.remaining_quantity
        self.order_count += 1
    
    def remove_order(self, order: Order):
        """Remove order from this price level."""
        if order in self.orders:
            self.orders.remove(order)
            self.total_quantity -= order.remaining_quantity
            self.order_count -= 1
    
    def update_quantity(self, order: Order, new_quantity: Decimal):
        """Update quantity for an order."""
        old_quantity = order.remaining_quantity
        order.remaining_quantity = new_quantity
        self.total_quantity += (new_quantity - old_quantity)


@dataclass
class OrderBook:
    """Complete order book for a bond."""
    
    bond_id: str
    bid_entries: List[OrderBookEntry] = field(default_factory=list)
    ask_entries: List[OrderBookEntry] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def get_best_bid(self) -> Optional[OrderBookEntry]:
        """Get best bid (highest price)."""
        return max(self.bid_entries, key=lambda x: x.price) if self.bid_entries else None
    
    def get_best_ask(self) -> Optional[OrderBookEntry]:
        """Get best ask (lowest price)."""
        return min(self.ask_entries, key=lambda x: x.price) if self.ask_entries else None
    
    def get_spread(self) -> Optional[Decimal]:
        """Get current bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None
    
    def get_mid_price(self) -> Optional[Decimal]:
        """Get mid-price between best bid and ask."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / 2
        return None


@dataclass
class MarketData:
    """Real-time market data for a bond."""
    
    bond_id: str
    last_price: Optional[Decimal] = None
    last_quantity: Optional[Decimal] = None
    last_trade_time: Optional[datetime] = None
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    bid_quantity: Optional[Decimal] = None
    ask_quantity: Optional[Decimal] = None
    spread: Optional[Decimal] = None
    mid_price: Optional[Decimal] = None
    volume_24h: Decimal = Decimal('0')
    high_24h: Optional[Decimal] = None
    low_24h: Optional[Decimal] = None
    change_24h: Optional[Decimal] = None
    change_percent_24h: Optional[float] = None
    yield_to_maturity: Optional[float] = None
    modified_duration: Optional[float] = None
    convexity: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def update_from_trade(self, trade: Trade):
        """Update market data from a trade."""
        self.last_price = trade.price
        self.last_quantity = trade.quantity
        self.last_trade_time = trade.execution_time
        self.volume_24h += trade.quantity
        
        if self.high_24h is None or trade.price > self.high_24h:
            self.high_24h = trade.price
        
        if self.low_24h is None or trade.price < self.low_24h:
            self.low_24h = trade.price
        
        if self.last_price and self.last_price != trade.price:
            self.change_24h = trade.price - self.last_price
            if self.last_price > 0:
                self.change_percent_24h = float(self.change_24h / self.last_price * 100)
        
        self.timestamp = datetime.utcnow()


@dataclass
class MarketMakerQuote:
    """Market maker quote for a bond."""
    
    bond_id: str
    market_maker_id: int
    bid_price: Decimal
    ask_price: Decimal
    bid_quantity: Decimal
    ask_quantity: Decimal
    spread: Decimal
    mid_price: Decimal
    quote_time: datetime = field(default_factory=datetime.utcnow)
    expiry_time: datetime = None
    is_active: bool = True
    risk_score: Optional[float] = None
    inventory_position: Optional[Decimal] = None
    
    def __post_init__(self):
        """Calculate derived fields."""
        self.spread = self.ask_price - self.bid_price
        self.mid_price = (self.bid_price + self.ask_price) / 2


@dataclass
class TradingSession:
    """Trading session information."""
    
    session_id: str = field(default_factory=lambda: str(uuid4()))
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: str = "ACTIVE"  # ACTIVE, CLOSED, SUSPENDED
    total_orders: int = 0
    total_trades: int = 0
    total_volume: Decimal = Decimal('0')
    total_value: Decimal = Decimal('0')
    active_participants: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskMetrics:
    """Risk metrics for trading operations."""
    
    participant_id: int
    bond_id: Optional[str] = None
    position_value: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    var_95: Optional[float] = None
    var_99: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    volatility: Optional[float] = None
    beta: Optional[float] = None
    correlation: Optional[float] = None
    concentration_risk: Optional[float] = None
    liquidity_risk: Optional[float] = None
    credit_risk: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
