"""
Market Making Algorithms for BondX Trading Engine.

This module implements sophisticated market making strategies with:
- Fair value computation from base curve + spread model + liquidity premium + risk inventory skew
- Two-sided price quoting with target inventory bands
- Dynamic quote adjustment based on volatility and inventory
- Risk-based quote withdrawal
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import math
import statistics
from uuid import uuid4

from ..core.logging import get_logger
from .trading_models import Order, OrderSide, OrderType, TimeInForce
from .order_book import OrderBook, OrderBookManager
from ..mathematics.yield_calculations import YieldCalculator
from ..mathematics.bond_pricing import BondPricer
from ..risk_management.risk_models import RiskCheckResult

logger = get_logger(__name__)


class MarketMakerStatus(Enum):
    """Market maker operational status."""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    SUSPENDED = "SUSPENDED"
    MAINTENANCE = "MAINTENANCE"
    RISK_LIMIT_BREACH = "RISK_LIMIT_BREACH"


class QuoteSide(Enum):
    """Quote side enumeration."""
    BID = "BID"
    ASK = "ASK"
    BOTH = "BOTH"


@dataclass
class MarketMakerQuote:
    """Market maker quote."""
    quote_id: str
    instrument_id: str
    bid_price: Optional[Decimal]
    ask_price: Optional[Decimal]
    bid_quantity: Optional[Decimal]
    ask_quantity: Optional[Decimal]
    timestamp: datetime
    valid_until: datetime
    status: str  # "ACTIVE", "EXPIRED", "CANCELLED"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FairValueModel:
    """Fair value model components."""
    base_curve_yield: Decimal
    credit_spread: Decimal
    liquidity_premium: Decimal
    inventory_skew: Decimal
    volatility_adjustment: Decimal
    fair_value: Decimal
    confidence_interval: Tuple[Decimal, Decimal]
    last_update: datetime


@dataclass
class InventoryPosition:
    """Market maker inventory position."""
    instrument_id: str
    long_position: Decimal
    short_position: Decimal
    net_position: Decimal
    target_position: Decimal
    max_position: Decimal
    position_limit: Decimal
    last_update: datetime


class MarketMaker:
    """
    Sophisticated market maker for bond instruments.
    
    Features:
    - Dynamic fair value computation
    - Two-sided price quoting
    - Inventory management
    - Risk-based quote adjustment
    - Volatility response
    """
    
    def __init__(self, 
                 instrument_id: str,
                 order_book_manager: OrderBookManager,
                 yield_calculator: Optional[YieldCalculator] = None,
                 bond_pricer: Optional[BondPricer] = None,
                 risk_checker: Optional[Callable] = None):
        """Initialize the market maker."""
        self.instrument_id = instrument_id
        self.order_book_manager = order_book_manager
        self.yield_calculator = yield_calculator
        self.bond_pricer = bond_pricer
        self.risk_checker = risk_checker
        
        # Configuration
        self.min_spread_bps = Decimal('5')  # 5 basis points minimum spread
        self.max_spread_bps = Decimal('100')  # 100 basis points maximum spread
        self.target_inventory_band = Decimal('0.1')  # 10% of max position
        self.max_inventory_skew = Decimal('0.3')  # 30% of max position
        self.quote_refresh_interval = 5  # seconds
        self.quote_validity_duration = 30  # seconds
        
        # State
        self.status = MarketMakerStatus.INACTIVE
        self.active_quotes: Dict[str, MarketMakerQuote] = {}
        self.inventory_position = InventoryPosition(
            instrument_id=instrument_id,
            long_position=Decimal('0'),
            short_position=Decimal('0'),
            net_position=Decimal('0'),
            target_position=Decimal('0'),
            max_position=Decimal('1000000'),  # 1M notional
            position_limit=Decimal('2000000'),  # 2M notional
            last_update=datetime.utcnow()
        )
        
        # Fair value tracking
        self.fair_value_history: List[FairValueModel] = []
        self.volatility_window = 100  # number of observations for volatility calculation
        
        # Performance tracking
        self.quotes_posted = 0
        self.quotes_filled = 0
        self.total_pnl = Decimal('0')
        self.start_time = datetime.utcnow()
        
        # Task management
        self.quote_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info(f"Market Maker initialized for instrument {instrument_id}")
    
    async def start(self) -> None:
        """Start the market maker."""
        if self.is_running:
            logger.warning("Market maker is already running")
            return
        
        self.is_running = True
        self.status = MarketMakerStatus.ACTIVE
        
        logger.info(f"Starting market maker for {self.instrument_id}")
        
        # Start quote management task
        self.quote_task = asyncio.create_task(self._quote_management_loop())
        
        logger.info(f"Market maker started for {self.instrument_id}")
    
    async def stop(self) -> None:
        """Stop the market maker."""
        if not self.is_running:
            logger.warning("Market maker is not running")
            return
        
        self.is_running = False
        self.status = MarketMakerStatus.INACTIVE
        
        logger.info(f"Stopping market maker for {self.instrument_id}")
        
        # Cancel quote task
        if self.quote_task:
            self.quote_task.cancel()
            try:
                await self.quote_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all active quotes
        await self._cancel_all_quotes()
        
        logger.info(f"Market maker stopped for {self.instrument_id}")
    
    def suspend(self, reason: str = "Manual suspension") -> None:
        """Suspend the market maker."""
        self.status = MarketMakerStatus.SUSPENDED
        logger.info(f"Market maker suspended for {self.instrument_id}: {reason}")
    
    def resume(self) -> None:
        """Resume the market maker."""
        if self.status == MarketMakerStatus.SUSPENDED:
            self.status = MarketMakerStatus.ACTIVE
            logger.info(f"Market maker resumed for {self.instrument_id}")
    
    async def _quote_management_loop(self) -> None:
        """Main loop for quote management."""
        while self.is_running:
            try:
                if self.status != MarketMakerStatus.ACTIVE:
                    await asyncio.sleep(1)
                    continue
                
                # Update fair value
                await self._update_fair_value()
                
                # Check risk limits
                if not await self._check_risk_limits():
                    self.status = MarketMakerStatus.RISK_LIMIT_BREACH
                    logger.warning(f"Risk limit breach for {self.instrument_id}")
                    continue
                
                # Update quotes
                await self._update_quotes()
                
                # Wait for next cycle
                await asyncio.sleep(self.quote_refresh_interval)
                
            except Exception as e:
                logger.error(f"Error in market maker quote loop for {self.instrument_id}: {e}")
                await asyncio.sleep(5)  # Longer delay on error
    
    async def _update_fair_value(self) -> None:
        """Update the fair value model."""
        try:
            # Get current market data
            order_book = self.order_book_manager.get_order_book(self.instrument_id)
            market_data = order_book.get_level_1_snapshot()
            
            # Base curve yield (simplified - in practice would come from yield curve service)
            base_yield = self._get_base_curve_yield()
            
            # Credit spread (simplified - in practice would come from credit models)
            credit_spread = self._get_credit_spread()
            
            # Liquidity premium based on bid-ask spread
            liquidity_premium = self._calculate_liquidity_premium(market_data)
            
            # Inventory skew adjustment
            inventory_skew = self._calculate_inventory_skew()
            
            # Volatility adjustment
            volatility_adjustment = self._calculate_volatility_adjustment()
            
            # Calculate fair value
            fair_value = base_yield + credit_spread + liquidity_premium + inventory_skew + volatility_adjustment
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(fair_value)
            
            # Create fair value model
            fair_value_model = FairValueModel(
                base_curve_yield=base_yield,
                credit_spread=credit_spread,
                liquidity_premium=liquidity_premium,
                inventory_skew=inventory_skew,
                volatility_adjustment=volatility_adjustment,
                fair_value=fair_value,
                confidence_interval=confidence_interval,
                last_update=datetime.utcnow()
            )
            
            # Store in history
            self.fair_value_history.append(fair_value_model)
            if len(self.fair_value_history) > self.volatility_window:
                self.fair_value_history.pop(0)
            
            logger.debug(f"Fair value updated for {self.instrument_id}: {fair_value}")
            
        except Exception as e:
            logger.error(f"Error updating fair value for {self.instrument_id}: {e}")
    
    def _get_base_curve_yield(self) -> Decimal:
        """Get base curve yield (simplified implementation)."""
        # In practice, this would come from a yield curve service
        # For now, return a fixed rate
        return Decimal('6.5')  # 6.5% base rate
    
    def _get_credit_spread(self) -> Decimal:
        """Get credit spread (simplified implementation)."""
        # In practice, this would come from credit models
        # For now, return a fixed spread
        return Decimal('1.2')  # 120 basis points
    
    def _calculate_liquidity_premium(self, market_data: Dict[str, Any]) -> Decimal:
        """Calculate liquidity premium based on market data."""
        try:
            spread = market_data.get('spread')
            if spread is None:
                return Decimal('0.5')  # Default premium
            
            # Convert spread to basis points and calculate premium
            spread_bps = spread * 10000  # Convert to basis points
            if spread_bps < 10:
                return Decimal('0.2')  # Low spread = low premium
            elif spread_bps < 50:
                return Decimal('0.5')  # Medium spread = medium premium
            else:
                return Decimal('1.0')  # High spread = high premium
                
        except Exception as e:
            logger.error(f"Error calculating liquidity premium: {e}")
            return Decimal('0.5')
    
    def _calculate_inventory_skew(self) -> Decimal:
        """Calculate inventory skew adjustment."""
        try:
            net_position = self.inventory_position.net_position
            max_position = self.inventory_position.max_position
            
            if max_position == 0:
                return Decimal('0')
            
            # Calculate position ratio
            position_ratio = net_position / max_position
            
            # Apply skew adjustment
            if abs(position_ratio) > self.max_inventory_skew:
                # Beyond max skew - apply penalty
                skew_adjustment = position_ratio * Decimal('0.5')
            else:
                # Within normal range - small adjustment
                skew_adjustment = position_ratio * Decimal('0.1')
            
            return skew_adjustment
            
        except Exception as e:
            logger.error(f"Error calculating inventory skew: {e}")
            return Decimal('0')
    
    def _calculate_volatility_adjustment(self) -> Decimal:
        """Calculate volatility adjustment."""
        try:
            if len(self.fair_value_history) < 2:
                return Decimal('0')
            
            # Calculate yield volatility
            yields = [model.fair_value for model in self.fair_value_history]
            if len(yields) < 2:
                return Decimal('0')
            
            # Calculate standard deviation
            mean_yield = sum(yields) / len(yields)
            variance = sum((y - mean_yield) ** 2 for y in yields) / len(yields)
            std_dev = Decimal(str(math.sqrt(float(variance))))
            
            # Apply volatility adjustment
            volatility_adjustment = std_dev * Decimal('0.1')  # 10% of volatility
            
            return volatility_adjustment
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjustment: {e}")
            return Decimal('0')
    
    def _calculate_confidence_interval(self, fair_value: Decimal) -> Tuple[Decimal, Decimal]:
        """Calculate confidence interval for fair value."""
        try:
            # Simple confidence interval based on volatility
            if len(self.fair_value_history) < 2:
                return (fair_value - Decimal('0.1'), fair_value + Decimal('0.1'))
            
            # Calculate volatility
            yields = [model.fair_value for model in self.fair_value_history]
            mean_yield = sum(yields) / len(yields)
            variance = sum((y - mean_yield) ** 2 for y in yields) / len(yields)
            std_dev = Decimal(str(math.sqrt(float(variance))))
            
            # 95% confidence interval (2 standard deviations)
            confidence_width = std_dev * Decimal('2')
            
            return (fair_value - confidence_width, fair_value + confidence_width)
            
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}")
            return (fair_value - Decimal('0.1'), fair_value + Decimal('0.1'))
    
    async def _check_risk_limits(self) -> bool:
        """Check risk limits."""
        try:
            # Check position limits
            if abs(self.inventory_position.net_position) > self.inventory_position.position_limit:
                logger.warning(f"Position limit breached for {self.instrument_id}")
                return False
            
            # Check PnL limits (simplified)
            if self.total_pnl < Decimal('-100000'):  # -100k loss limit
                logger.warning(f"PnL limit breached for {self.instrument_id}")
                return False
            
            # Additional risk checks can be added here
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False
    
    async def _update_quotes(self) -> None:
        """Update market maker quotes."""
        try:
            if not self.fair_value_history:
                logger.warning(f"No fair value available for {self.instrument_id}")
                return
            
            # Get latest fair value
            latest_fair_value = self.fair_value_history[-1]
            
            # Calculate bid and ask prices
            bid_price, ask_price = self._calculate_quote_prices(latest_fair_value)
            
            # Calculate quote quantities
            bid_quantity, ask_quantity = self._calculate_quote_quantities()
            
            # Post or update quotes
            await self._post_quotes(bid_price, ask_price, bid_quantity, ask_quantity)
            
        except Exception as e:
            logger.error(f"Error updating quotes for {self.instrument_id}: {e}")
    
    def _calculate_quote_prices(self, fair_value_model: FairValueModel) -> Tuple[Decimal, Decimal]:
        """Calculate bid and ask prices."""
        try:
            fair_value = fair_value_model.fair_value
            
            # Calculate spread based on volatility and inventory
            base_spread = self.min_spread_bps / Decimal('10000')  # Convert to decimal
            
            # Adjust spread based on volatility
            volatility_factor = min(Decimal('2.0'), max(Decimal('0.5'), 
                                   fair_value_model.volatility_adjustment / Decimal('0.1')))
            adjusted_spread = base_spread * volatility_factor
            
            # Ensure spread is within limits
            adjusted_spread = max(self.min_spread_bps / Decimal('10000'), 
                                min(self.max_spread_bps / Decimal('10000'), adjusted_spread))
            
            # Calculate bid and ask
            half_spread = adjusted_spread / Decimal('2')
            bid_price = fair_value - half_spread
            ask_price = fair_value + half_spread
            
            # Round to appropriate decimal places
            bid_price = round(bid_price, 4)
            ask_price = round(ask_price, 4)
            
            return bid_price, ask_price
            
        except Exception as e:
            logger.error(f"Error calculating quote prices: {e}")
            return Decimal('0'), Decimal('0')
    
    def _calculate_quote_quantities(self) -> Tuple[Decimal, Decimal]:
        """Calculate quote quantities."""
        try:
            # Base quantity
            base_quantity = Decimal('100000')  # 100k notional
            
            # Adjust based on inventory
            net_position = self.inventory_position.net_position
            max_position = self.inventory_position.max_position
            
            if max_position == 0:
                return base_quantity, base_quantity
            
            position_ratio = net_position / max_position
            
            # If long position, reduce bid quantity, increase ask quantity
            if position_ratio > 0:
                bid_quantity = base_quantity * (Decimal('1') - position_ratio * Decimal('0.5'))
                ask_quantity = base_quantity * (Decimal('1') + position_ratio * Decimal('0.5'))
            else:
                # If short position, increase bid quantity, reduce ask quantity
                bid_quantity = base_quantity * (Decimal('1') - position_ratio * Decimal('0.5'))
                ask_quantity = base_quantity * (Decimal('1') + position_ratio * Decimal('0.5'))
            
            # Ensure minimum quantities
            min_quantity = Decimal('10000')  # 10k minimum
            bid_quantity = max(min_quantity, bid_quantity)
            ask_quantity = max(min_quantity, ask_quantity)
            
            return bid_quantity, ask_quantity
            
        except Exception as e:
            logger.error(f"Error calculating quote quantities: {e}")
            return Decimal('100000'), Decimal('100000')
    
    async def _post_quotes(self, bid_price: Decimal, ask_price: Decimal, 
                          bid_quantity: Decimal, ask_quantity: Decimal) -> None:
        """Post or update market maker quotes."""
        try:
            # Cancel existing quotes
            await self._cancel_all_quotes()
            
            # Create new quotes
            if bid_price > 0 and bid_quantity > 0:
                bid_quote = await self._create_quote(
                    OrderSide.BUY, bid_price, bid_quantity
                )
                if bid_quote:
                    self.active_quotes[bid_quote.quote_id] = bid_quote
            
            if ask_price > 0 and ask_quantity > 0:
                ask_quote = await self._create_quote(
                    OrderSide.SELL, ask_price, ask_quantity
                )
                if ask_quote:
                    self.active_quotes[ask_quote.quote_id] = ask_quote
            
            logger.debug(f"Posted quotes for {self.instrument_id}: bid={bid_price}, ask={ask_price}")
            
        except Exception as e:
            logger.error(f"Error posting quotes for {self.instrument_id}: {e}")
    
    async def _create_quote(self, side: OrderSide, price: Decimal, 
                           quantity: Decimal) -> Optional[MarketMakerQuote]:
        """Create a market maker quote."""
        try:
            quote_id = str(uuid4())
            valid_until = datetime.utcnow() + timedelta(seconds=self.quote_validity_duration)
            
            # Create order
            order = Order(
                order_id=quote_id,
                participant_id=0,  # Market maker participant ID
                bond_id=self.instrument_id,
                order_type=OrderType.LIMIT,
                side=side,
                quantity=quantity,
                price=price,
                time_in_force=TimeInForce.GOOD_TILL_CANCELLED,
                status=OrderStatus.PENDING
            )
            
            # Add to order book
            order_book = self.order_book_manager.get_order_book(self.instrument_id)
            success = order_book.add_order(order)
            
            if not success:
                logger.warning(f"Failed to add market maker quote to order book: {quote_id}")
                return None
            
            # Create quote record
            quote = MarketMakerQuote(
                quote_id=quote_id,
                instrument_id=self.instrument_id,
                bid_price=price if side == OrderSide.BUY else None,
                ask_price=price if side == OrderSide.SELL else None,
                bid_quantity=quantity if side == OrderSide.BUY else None,
                ask_quantity=quantity if side == OrderSide.SELL else None,
                timestamp=datetime.utcnow(),
                valid_until=valid_until,
                status="ACTIVE",
                metadata={"side": side.value, "order": order}
            )
            
            self.quotes_posted += 1
            return quote
            
        except Exception as e:
            logger.error(f"Error creating quote: {e}")
            return None
    
    async def _cancel_all_quotes(self) -> None:
        """Cancel all active market maker quotes."""
        try:
            for quote_id in list(self.active_quotes.keys()):
                await self._cancel_quote(quote_id)
        except Exception as e:
            logger.error(f"Error cancelling all quotes: {e}")
    
    async def _cancel_quote(self, quote_id: str) -> None:
        """Cancel a specific quote."""
        try:
            if quote_id in self.active_quotes:
                quote = self.active_quotes[quote_id]
                
                # Remove from order book
                order_book = self.order_book_manager.get_order_book(self.instrument_id)
                order_book.remove_order(quote_id)
                
                # Update quote status
                quote.status = "CANCELLED"
                
                # Remove from active quotes
                del self.active_quotes[quote_id]
                
                logger.debug(f"Cancelled quote {quote_id}")
                
        except Exception as e:
            logger.error(f"Error cancelling quote {quote_id}: {e}")
    
    def update_inventory(self, side: OrderSide, quantity: Decimal, price: Decimal) -> None:
        """Update inventory position after trade."""
        try:
            if side == OrderSide.BUY:
                # Market maker bought (sold to client)
                self.inventory_position.long_position += quantity
                self.inventory_position.net_position += quantity
            else:
                # Market maker sold (bought from client)
                self.inventory_position.short_position += quantity
                self.inventory_position.net_position -= quantity
            
            # Update PnL (simplified calculation)
            trade_value = quantity * price
            if side == OrderSide.BUY:
                self.total_pnl -= trade_value  # Cost
            else:
                self.total_pnl += trade_value  # Proceeds
            
            self.inventory_position.last_update = datetime.utcnow()
            
            logger.debug(f"Inventory updated for {self.instrument_id}: {side.value} {quantity} @ {price}")
            
        except Exception as e:
            logger.error(f"Error updating inventory: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get market maker statistics."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "instrument_id": self.instrument_id,
            "status": self.status.value,
            "is_running": self.is_running,
            "quotes_posted": self.quotes_posted,
            "quotes_filled": self.quotes_filled,
            "active_quotes": len(self.active_quotes),
            "total_pnl": self.total_pnl,
            "inventory_position": {
                "net_position": self.inventory_position.net_position,
                "long_position": self.inventory_position.long_position,
                "short_position": self.inventory_position.short_position,
                "position_limit": self.inventory_position.position_limit
            },
            "fair_value_history_size": len(self.fair_value_history),
            "uptime_seconds": uptime,
            "start_time": self.start_time
        }
    
    def get_fair_value(self) -> Optional[FairValueModel]:
        """Get the latest fair value model."""
        if self.fair_value_history:
            return self.fair_value_history[-1]
        return None


class MarketMakerManager:
    """
    Manages multiple market makers across different instruments.
    
    Provides:
    - Instrument-specific market makers
    - Global market making statistics
    - Centralized risk management
    """
    
    def __init__(self, order_book_manager: OrderBookManager):
        """Initialize the market maker manager."""
        self.order_book_manager = order_book_manager
        self.market_makers: Dict[str, MarketMaker] = {}
        self.logger = logger
    
    def create_market_maker(self, instrument_id: str, **kwargs) -> MarketMaker:
        """Create a new market maker for an instrument."""
        if instrument_id in self.market_makers:
            logger.warning(f"Market maker already exists for {instrument_id}")
            return self.market_makers[instrument_id]
        
        market_maker = MarketMaker(
            instrument_id=instrument_id,
            order_book_manager=self.order_book_manager,
            **kwargs
        )
        
        self.market_makers[instrument_id] = market_maker
        logger.info(f"Created market maker for {instrument_id}")
        
        return market_maker
    
    def get_market_maker(self, instrument_id: str) -> Optional[MarketMaker]:
        """Get market maker for an instrument."""
        return self.market_makers.get(instrument_id)
    
    def remove_market_maker(self, instrument_id: str) -> bool:
        """Remove market maker for an instrument."""
        if instrument_id in self.market_makers:
            market_maker = self.market_makers[instrument_id]
            asyncio.create_task(market_maker.stop())
            del self.market_makers[instrument_id]
            logger.info(f"Removed market maker for {instrument_id}")
            return True
        return False
    
    async def start_all(self) -> None:
        """Start all market makers."""
        for market_maker in self.market_makers.values():
            await market_maker.start()
    
    async def stop_all(self) -> None:
        """Stop all market makers."""
        for market_maker in self.market_makers.values():
            await market_maker.stop()
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global market making statistics."""
        total_quotes_posted = sum(mm.quotes_posted for mm in self.market_makers.values())
        total_quotes_filled = sum(mm.quotes_filled for mm in self.market_makers.values())
        total_pnl = sum(mm.total_pnl for mm in self.market_makers.values())
        active_market_makers = sum(1 for mm in self.market_makers.values() if mm.is_running)
        
        return {
            "total_instruments": len(self.market_makers),
            "active_market_makers": active_market_makers,
            "total_quotes_posted": total_quotes_posted,
            "total_quotes_filled": total_quotes_filled,
            "total_pnl": total_pnl,
            "timestamp": datetime.utcnow()
        }


# Export classes
__all__ = ["MarketMaker", "MarketMakerManager", "MarketMakerStatus", "MarketMakerQuote", 
           "FairValueModel", "InventoryPosition", "QuoteSide"]
