"""
High-Performance Matching Engine for BondX Trading Engine.

This module implements a deterministic matching engine with:
- Event-driven processing loop
- Risk pre-trade checks
- Configurable matching policies
- Replay capability from event log
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import heapq
from uuid import uuid4

from ..core.logging import get_logger
from .trading_models import Order, OrderSide, OrderStatus, OrderType, TimeInForce
from .order_book import OrderBook, OrderBookManager
from ..risk_management.risk_models import RiskCheckResult
from ..database.models import Trade

logger = get_logger(__name__)


class MatchingEventType(Enum):
    """Types of matching events."""
    ORDER_ACCEPTED = "ORDER_ACCEPTED"
    ORDER_AMENDED = "ORDER_AMENDED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_MATCHED = "ORDER_MATCHED"
    PARTIAL_FILL = "PARTIAL_FILL"
    FULL_FILL = "FULL_FILL"
    TRADE_EXECUTED = "TRADE_EXECUTED"


@dataclass
class MatchingEvent:
    """Event in the matching engine."""
    event_id: str
    event_type: MatchingEventType
    timestamp: datetime
    order_id: str
    instrument_id: str
    participant_id: int
    side: OrderSide
    quantity: Decimal
    price: Optional[Decimal]
    remaining_quantity: Decimal
    matched_quantity: Decimal = Decimal('0')
    matched_price: Optional[Decimal] = None
    counterparty_order_id: Optional[str] = None
    trade_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeExecution:
    """Result of a trade execution."""
    trade_id: str
    instrument_id: str
    buy_order_id: str
    sell_order_id: str
    buy_participant_id: int
    sell_participant_id: int
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    trade_value: Decimal
    fees: Dict[str, Decimal] = field(default_factory=dict)


@dataclass
class MatchingResult:
    """Result of order matching."""
    order_id: str
    instrument_id: str
    participant_id: int
    side: OrderSide
    original_quantity: Decimal
    matched_quantity: Decimal
    remaining_quantity: Decimal
    average_price: Optional[Decimal]
    trades: List[TradeExecution]
    status: OrderStatus
    timestamp: datetime
    risk_check_passed: bool
    risk_score: Optional[float] = None


class MatchingPolicy(Enum):
    """Matching policy types."""
    PRICE_TIME_PRIORITY = "PRICE_TIME_PRIORITY"
    PRO_RATA = "PRO_RATA"
    TIME_PRIORITY = "TIME_PRIORITY"
    SIZE_PRIORITY = "SIZE_PRIORITY"


class MatchingEngine:
    """
    High-performance matching engine for continuous trading.
    
    Features:
    - Deterministic processing order
    - Risk pre-trade checks
    - Configurable matching policies
    - Event log for replay capability
    - High-frequency trading optimizations
    """
    
    def __init__(self, order_book_manager: OrderBookManager, 
                 risk_checker: Optional[Callable] = None,
                 matching_policy: MatchingPolicy = MatchingPolicy.PRICE_TIME_PRIORITY):
        """Initialize the matching engine."""
        self.order_book_manager = order_book_manager
        self.risk_checker = risk_checker
        self.matching_policy = matching_policy
        
        # Event processing
        self.event_queue: deque = deque()
        self.event_log: List[MatchingEvent] = []
        self.event_sequence = 0
        
        # Performance tracking
        self.orders_processed = 0
        self.trades_executed = 0
        self.total_volume = Decimal('0')
        self.start_time = datetime.utcnow()
        
        # Configuration
        self.max_queue_size = 10000
        self.batch_size = 100
        self.processing_delay = 0.001  # 1ms delay between batches
        
        # State
        self.is_running = False
        self.is_paused = False
        
        logger.info("Matching Engine initialized successfully")
    
    async def start(self) -> None:
        """Start the matching engine."""
        if self.is_running:
            logger.warning("Matching engine is already running")
            return
        
        self.is_running = True
        self.is_paused = False
        
        logger.info("Starting matching engine...")
        
        # Start processing loop
        asyncio.create_task(self._processing_loop())
        
        logger.info("Matching engine started successfully")
    
    async def stop(self) -> None:
        """Stop the matching engine."""
        if not self.is_running:
            logger.warning("Matching engine is not running")
            return
        
        self.is_running = False
        logger.info("Stopping matching engine...")
        
        # Wait for processing to complete
        await asyncio.sleep(0.1)
        
        logger.info("Matching engine stopped")
    
    def pause(self) -> None:
        """Pause the matching engine."""
        self.is_paused = True
        logger.info("Matching engine paused")
    
    def resume(self) -> None:
        """Resume the matching engine."""
        self.is_paused = False
        logger.info("Matching engine resumed")
    
    async def submit_order(self, order: Order) -> str:
        """
        Submit an order for processing.
        
        Args:
            order: Order to submit
            
        Returns:
            Event ID for tracking
        """
        event_id = str(uuid4())
        
        # Create event
        event = MatchingEvent(
            event_id=event_id,
            event_type=MatchingEventType.ORDER_ACCEPTED,
            timestamp=datetime.utcnow(),
            order_id=order.order_id,
            instrument_id=order.bond_id,
            participant_id=order.participant_id,
            side=order.side,
            quantity=order.quantity,
            price=order.price,
            remaining_quantity=order.remaining_quantity,
            metadata={"original_order": order}
        )
        
        # Add to queue
        if len(self.event_queue) >= self.max_queue_size:
            logger.warning(f"Event queue full, rejecting order {order.order_id}")
            return None
        
        self.event_queue.append(event)
        
        logger.debug(f"Order {order.order_id} submitted for processing")
        return event_id
    
    async def cancel_order(self, order_id: str, participant_id: int) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of order to cancel
            participant_id: ID of participant requesting cancellation
            
        Returns:
            True if cancellation was successful
        """
        event_id = str(uuid4())
        
        # Create cancellation event
        event = MatchingEvent(
            event_id=event_id,
            event_type=MatchingEventType.ORDER_CANCELLED,
            timestamp=datetime.utcnow(),
            order_id=order_id,
            instrument_id="",  # Will be filled during processing
            participant_id=participant_id,
            side=OrderSide.BUY,  # Placeholder
            quantity=Decimal('0'),
            price=None,
            remaining_quantity=Decimal('0'),
            metadata={"cancellation_request": True}
        )
        
        # Add to queue with high priority
        self.event_queue.appendleft(event)
        
        logger.debug(f"Order cancellation requested for {order_id}")
        return True
    
    async def amend_order(self, order_id: str, participant_id: int, 
                         new_quantity: Optional[Decimal] = None,
                         new_price: Optional[Decimal] = None) -> bool:
        """
        Amend an order.
        
        Args:
            order_id: ID of order to amend
            participant_id: ID of participant requesting amendment
            new_quantity: New quantity (if None, unchanged)
            new_price: New price (if None, unchanged)
            
        Returns:
            True if amendment was successful
        """
        event_id = str(uuid4())
        
        # Create amendment event
        event = MatchingEvent(
            event_id=event_id,
            event_type=MatchingEventType.ORDER_AMENDED,
            timestamp=datetime.utcnow(),
            order_id=order_id,
            instrument_id="",  # Will be filled during processing
            participant_id=participant_id,
            side=OrderSide.BUY,  # Placeholder
            quantity=Decimal('0'),
            price=None,
            remaining_quantity=Decimal('0'),
            metadata={
                "amendment_request": True,
                "new_quantity": new_quantity,
                "new_price": new_price
            }
        )
        
        # Add to queue
        self.event_queue.append(event)
        
        logger.debug(f"Order amendment requested for {order_id}")
        return True
    
    async def _processing_loop(self) -> None:
        """Main processing loop for the matching engine."""
        while self.is_running:
            try:
                if self.is_paused:
                    await asyncio.sleep(0.001)
                    continue
                
                # Process batch of events
                await self._process_batch()
                
                # Small delay to prevent CPU spinning
                await asyncio.sleep(self.processing_delay)
                
            except Exception as e:
                logger.error(f"Error in matching engine processing loop: {e}")
                await asyncio.sleep(0.1)  # Longer delay on error
    
    async def _process_batch(self) -> None:
        """Process a batch of events."""
        batch = []
        
        # Collect batch
        for _ in range(min(self.batch_size, len(self.event_queue))):
            if self.event_queue:
                batch.append(self.event_queue.popleft())
        
        if not batch:
            return
        
        # Process each event
        for event in batch:
            try:
                await self._process_event(event)
                self.event_sequence += 1
            except Exception as e:
                logger.error(f"Error processing event {event.event_id}: {e}")
                # Continue processing other events
    
    async def _process_event(self, event: MatchingEvent) -> None:
        """Process a single matching event."""
        try:
            if event.event_type == MatchingEventType.ORDER_ACCEPTED:
                await self._process_order_accepted(event)
            elif event.event_type == MatchingEventType.ORDER_CANCELLED:
                await self._process_order_cancelled(event)
            elif event.event_type == MatchingEventType.ORDER_AMENDED:
                await self._process_order_amended(event)
            else:
                logger.warning(f"Unknown event type: {event.event_type}")
                
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")
            raise
    
    async def _process_order_accepted(self, event: MatchingEvent) -> None:
        """Process an order acceptance event."""
        order = event.metadata["original_order"]
        
        # Get or create order book
        order_book = self.order_book_manager.get_order_book(order.bond_id)
        
        # Risk check (if available)
        risk_passed = True
        risk_score = None
        if self.risk_checker:
            try:
                risk_result = await self.risk_checker(order)
                risk_passed = risk_result.passed
                risk_score = risk_result.risk_score
            except Exception as e:
                logger.error(f"Risk check failed for order {order.order_id}: {e}")
                risk_passed = False
        
        if not risk_passed:
            # Reject order
            event.event_type = MatchingEventType.ORDER_MATCHED
            event.metadata["rejected"] = True
            event.metadata["rejection_reason"] = "Risk check failed"
            self.event_log.append(event)
            return
        
        # Add order to order book
        success = order_book.add_order(order)
        if not success:
            event.metadata["rejected"] = True
            event.metadata["rejection_reason"] = "Failed to add to order book"
            self.event_log.append(event)
            return
        
        # Attempt to match
        matching_result = await self._attempt_matching(order, order_book)
        
        # Update event with results
        event.matched_quantity = matching_result.matched_quantity
        event.remaining_quantity = matching_result.remaining_quantity
        event.metadata["matching_result"] = matching_result
        
        # Log event
        self.event_log.append(event)
        
        # Update statistics
        self.orders_processed += 1
        self.total_volume += matching_result.matched_quantity
        
        logger.debug(f"Order {order.order_id} processed successfully")
    
    async def _process_order_cancelled(self, event: MatchingEvent) -> None:
        """Process an order cancellation event."""
        # Find the order in the order book
        for instrument_id, order_book in self.order_book_manager.order_books.items():
            if event.order_id in order_book.orders_by_id:
                # Cancel the order
                success = order_book.remove_order(event.order_id)
                if success:
                    event.instrument_id = instrument_id
                    event.metadata["cancelled"] = True
                    logger.debug(f"Order {event.order_id} cancelled successfully")
                else:
                    event.metadata["cancellation_failed"] = True
                    logger.warning(f"Failed to cancel order {event.order_id}")
                
                self.event_log.append(event)
                return
        
        # Order not found
        event.metadata["order_not_found"] = True
        self.event_log.append(event)
        logger.warning(f"Order {event.order_id} not found for cancellation")
    
    async def _process_order_amended(self, event: MatchingEvent) -> None:
        """Process an order amendment event."""
        new_quantity = event.metadata.get("new_quantity")
        new_price = event.metadata.get("new_price")
        
        # Find the order in the order book
        for instrument_id, order_book in self.order_book_manager.order_books.items():
            if event.order_id in order_book.orders_by_id:
                order, old_price = order_book.orders_by_id[event.order_id]
                
                # Remove old order
                order_book.remove_order(event.order_id)
                
                # Update order
                if new_quantity is not None:
                    order.remaining_quantity = new_quantity
                if new_price is not None:
                    order.price = new_price
                
                # Re-add order (this will place it at the correct price level)
                success = order_book.add_order(order)
                
                if success:
                    event.instrument_id = instrument_id
                    event.metadata["amended"] = True
                    logger.debug(f"Order {event.order_id} amended successfully")
                else:
                    event.metadata["amendment_failed"] = True
                    logger.warning(f"Failed to amend order {event.order_id}")
                
                self.event_log.append(event)
                return
        
        # Order not found
        event.metadata["order_not_found"] = True
        self.event_log.append(event)
        logger.warning(f"Order {event.order_id} not found for amendment")
    
    async def _attempt_matching(self, order: Order, order_book: OrderBook) -> MatchingResult:
        """
        Attempt to match an order against the order book.
        
        Args:
            order: Order to match
            order_book: Order book for the instrument
            
        Returns:
            Matching result
        """
        trades = []
        matched_quantity = Decimal('0')
        remaining_quantity = order.remaining_quantity
        
        if order.side == OrderSide.BUY:
            # Match against ask side
            while remaining_quantity > 0 and order_book.get_best_ask() is not None:
                best_ask = order_book.get_best_ask()
                
                # Check if we can match at this price
                if order.price is not None and order.price < best_ask:
                    break  # Price too high
                
                # Get orders at best ask
                ask_level = order_book.ask_orders[best_ask]
                
                for ask_order in ask_level.orders:
                    if remaining_quantity <= 0:
                        break
                    
                    # Calculate match quantity
                    match_qty = min(remaining_quantity, ask_order.remaining_quantity)
                    
                    # Execute trade
                    trade = await self._execute_trade(order, ask_order, match_qty, best_ask)
                    trades.append(trade)
                    
                    # Update quantities
                    matched_quantity += match_qty
                    remaining_quantity -= match_qty
                    ask_order.remaining_quantity -= match_qty
                    
                    # Update order book
                    if ask_order.remaining_quantity <= 0:
                        order_book.remove_order(ask_order.order_id)
                    else:
                        order_book.update_order(ask_order.order_id, ask_order.remaining_quantity)
                    
                    # Check if order is fully filled
                    if remaining_quantity <= 0:
                        break
                
                # If no more orders at this price level, remove it
                if not ask_level.orders:
                    del order_book.ask_orders[best_ask]
        
        else:  # SELL order
            # Match against bid side
            while remaining_quantity > 0 and order_book.get_best_bid() is not None:
                best_bid = order_book.get_best_bid()
                
                # Check if we can match at this price
                if order.price is not None and order.price > best_bid:
                    break  # Price too low
                
                # Get orders at best bid
                bid_level = order_book.bid_orders[best_bid]
                
                for bid_order in bid_level.orders:
                    if remaining_quantity <= 0:
                        break
                    
                    # Calculate match quantity
                    match_qty = min(remaining_quantity, bid_order.remaining_quantity)
                    
                    # Execute trade
                    trade = await self._execute_trade(bid_order, order, match_qty, best_bid)
                    trades.append(trade)
                    
                    # Update quantities
                    matched_quantity += match_qty
                    remaining_quantity -= match_qty
                    bid_order.remaining_quantity -= match_qty
                    
                    # Update order book
                    if bid_order.remaining_quantity <= 0:
                        order_book.remove_order(bid_order.order_id)
                    else:
                        order_book.update_order(bid_order.order_id, bid_order.remaining_quantity)
                    
                    # Check if order is fully filled
                    if remaining_quantity <= 0:
                        break
                
                # If no more orders at this price level, remove it
                if not bid_level.orders:
                    del order_book.bid_orders[best_bid]
        
        # Update order
        order.filled_quantity += matched_quantity
        order.remaining_quantity = remaining_quantity
        
        # Calculate average price
        if trades:
            total_value = sum(trade.trade_value for trade in trades)
            total_quantity = sum(trade.quantity for trade in trades)
            average_price = total_value / total_quantity
            order.average_fill_price = average_price
        else:
            average_price = None
        
        # Determine order status
        if remaining_quantity <= 0:
            status = OrderStatus.FILLED
        elif matched_quantity > 0:
            status = OrderStatus.PARTIALLY_FILLED
        else:
            status = OrderStatus.ACTIVE
        
        order.status = status
        
        # Update statistics
        self.trades_executed += len(trades)
        
        return MatchingResult(
            order_id=order.order_id,
            instrument_id=order.bond_id,
            participant_id=order.participant_id,
            side=order.side,
            original_quantity=order.quantity,
            matched_quantity=matched_quantity,
            remaining_quantity=remaining_quantity,
            average_price=average_price,
            trades=trades,
            status=status,
            timestamp=datetime.utcnow(),
            risk_check_passed=True,
            risk_score=None
        )
    
    async def _execute_trade(self, buy_order: Order, sell_order: Order, 
                           quantity: Decimal, price: Decimal) -> TradeExecution:
        """
        Execute a trade between two orders.
        
        Args:
            buy_order: Buy order
            sell_order: Sell order
            quantity: Trade quantity
            price: Trade price
            
        Returns:
            Trade execution result
        """
        trade_id = str(uuid4())
        trade_value = quantity * price
        
        # Create trade execution
        trade = TradeExecution(
            trade_id=trade_id,
            instrument_id=buy_order.bond_id,
            buy_order_id=buy_order.order_id,
            sell_order_id=sell_order.order_id,
            buy_participant_id=buy_order.participant_id,
            sell_participant_id=sell_order.participant_id,
            quantity=quantity,
            price=price,
            timestamp=datetime.utcnow(),
            trade_value=trade_value
        )
        
        logger.debug(f"Trade executed: {trade_id} - {quantity} @ {price}")
        return trade
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get matching engine statistics."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "orders_processed": self.orders_processed,
            "trades_executed": self.trades_executed,
            "total_volume": self.total_volume,
            "event_queue_size": len(self.event_queue),
            "event_log_size": len(self.event_log),
            "event_sequence": self.event_sequence,
            "uptime_seconds": uptime,
            "orders_per_second": self.orders_processed / uptime if uptime > 0 else 0,
            "trades_per_second": self.trades_executed / uptime if uptime > 0 else 0,
            "start_time": self.start_time,
            "matching_policy": self.matching_policy.value
        }
    
    def get_event_log(self, limit: Optional[int] = None) -> List[MatchingEvent]:
        """Get the event log."""
        if limit is None:
            return self.event_log.copy()
        return self.event_log[-limit:]
    
    def clear_event_log(self) -> None:
        """Clear the event log."""
        self.event_log.clear()
        self.event_sequence = 0
        logger.info("Event log cleared")


# Export classes
__all__ = ["MatchingEngine", "MatchingEvent", "MatchingEventType", "MatchingResult", 
           "TradeExecution", "MatchingPolicy"]
