"""
Real-time Order Book Management for BondX Trading Engine.

This module implements a high-performance, in-memory order book system
with price-time priority, stable sorting, and O(log n) operations.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import heapq
from uuid import uuid4

from ..core.logging import get_logger
from .trading_models import Order, OrderSide, OrderStatus, OrderType
from ..database.models import Bond

logger = get_logger(__name__)


class OrderBookSide(Enum):
    """Order book side enumeration."""
    BID = "BID"
    ASK = "ASK"


@dataclass
class OrderBookLevel:
    """Represents a price level in the order book."""
    price: Decimal
    total_quantity: Decimal = Decimal('0')
    order_count: int = 0
    orders: List[Order] = field(default_factory=list)
    
    def add_order(self, order: Order) -> None:
        """Add an order to this price level."""
        self.orders.append(order)
        self.total_quantity += order.remaining_quantity
        self.order_count += 1
    
    def remove_order(self, order: Order) -> None:
        """Remove an order from this price level."""
        if order in self.orders:
            self.orders.remove(order)
            self.total_quantity -= order.remaining_quantity
            self.order_count -= 1
    
    def update_order(self, order: Order, old_quantity: Decimal) -> None:
        """Update an order's quantity at this price level."""
        quantity_diff = order.remaining_quantity - old_quantity
        self.total_quantity += quantity_diff


@dataclass
class OrderBookSnapshot:
    """Snapshot of order book state."""
    timestamp: datetime
    instrument_id: str
    best_bid: Optional[Decimal]
    best_ask: Optional[Decimal]
    mid_price: Optional[Decimal]
    spread: Optional[Decimal]
    bid_levels: List[Tuple[Decimal, Decimal]]  # (price, quantity)
    ask_levels: List[Tuple[Decimal, Decimal]]  # (price, quantity)
    total_bid_volume: Decimal
    total_ask_volume: Decimal


@dataclass
class OrderBookUpdate:
    """Incremental update to order book."""
    timestamp: datetime
    instrument_id: str
    side: OrderBookSide
    price: Decimal
    quantity_change: Decimal
    order_count_change: int
    update_type: str  # "ADD", "REMOVE", "UPDATE"


class OrderBook:
    """
    High-performance in-memory order book for a single instrument.
    
    Features:
    - Price-time priority ordering
    - O(log n) insert/remove operations
    - Stable sorting for deterministic behavior
    - Real-time snapshots and incremental updates
    - Level 1 (top-of-book) and Level 2 (depth) views
    """
    
    def __init__(self, instrument_id: str, max_levels: int = 50):
        """Initialize the order book for an instrument."""
        self.instrument_id = instrument_id
        self.max_levels = max_levels
        
        # Bid side (buy orders) - max heap for best bid
        self.bid_orders: Dict[Decimal, OrderBookLevel] = {}
        self.bid_prices: List[Decimal] = []  # Max heap
        
        # Ask side (sell orders) - min heap for best ask
        self.ask_orders: Dict[Decimal, OrderBookLevel] = {}
        self.ask_prices: List[Decimal] = []  # Min heap
        
        # Order tracking
        self.orders_by_id: Dict[str, Tuple[Order, Decimal)] = {}  # order_id -> (order, price)
        
        # Statistics
        self.total_bid_volume = Decimal('0')
        self.total_ask_volume = Decimal('0')
        self.last_update = datetime.utcnow()
        
        # Event tracking
        self.update_sequence = 0
        self.pending_updates: List[OrderBookUpdate] = []
        
        logger.info(f"Order book initialized for instrument {instrument_id}")
    
    def add_order(self, order: Order) -> bool:
        """
        Add a new order to the order book.
        
        Args:
            order: Order to add
            
        Returns:
            True if order was added successfully, False otherwise
        """
        try:
            if order.side == OrderSide.BUY:
                return self._add_bid_order(order)
            else:
                return self._add_ask_order(order)
        except Exception as e:
            logger.error(f"Error adding order {order.order_id}: {e}")
            return False
    
    def _add_bid_order(self, order: Order) -> bool:
        """Add a buy order to the bid side."""
        price = order.price
        if price is None:
            logger.error(f"Bid order {order.order_id} has no price")
            return False
        
        # Create or update price level
        if price not in self.bid_orders:
            self.bid_orders[price] = OrderBookLevel(price)
            heapq.heappush(self.bid_prices, -price)  # Negative for max heap
        
        # Add order to level
        self.bid_orders[price].add_order(order)
        self.orders_by_id[order.order_id] = (order, price)
        self.total_bid_volume += order.remaining_quantity
        
        # Track update
        self._track_update(OrderBookSide.BID, price, order.remaining_quantity, 1, "ADD")
        
        logger.debug(f"Added bid order {order.order_id} at price {price}")
        return True
    
    def _add_ask_order(self, order: Order) -> bool:
        """Add a sell order to the ask side."""
        price = order.price
        if price is None:
            logger.error(f"Ask order {order.order_id} has no price")
            return False
        
        # Create or update price level
        if price not in self.ask_orders:
            self.ask_orders[price] = OrderBookLevel(price)
            heapq.heappush(self.ask_prices, price)  # Min heap
        
        # Add order to level
        self.ask_orders[price].add_order(order)
        self.orders_by_id[order.order_id] = (order, price)
        self.total_ask_volume += order.remaining_quantity
        
        # Track update
        self._track_update(OrderBookSide.ASK, price, order.remaining_quantity, 1, "ADD")
        
        logger.debug(f"Added ask order {order.order_id} at price {price}")
        return True
    
    def remove_order(self, order_id: str) -> bool:
        """
        Remove an order from the order book.
        
        Args:
            order_id: ID of order to remove
            
        Returns:
            True if order was removed successfully, False otherwise
        """
        if order_id not in self.orders_by_id:
            logger.warning(f"Order {order_id} not found in order book")
            return False
        
        order, price = self.orders_by_id[order_id]
        side = order.side
        
        try:
            if side == OrderSide.BUY:
                return self._remove_bid_order(order_id, price)
            else:
                return self._remove_ask_order(order_id, price)
        except Exception as e:
            logger.error(f"Error removing order {order_id}: {e}")
            return False
    
    def _remove_bid_order(self, order_id: str, price: Decimal) -> bool:
        """Remove a buy order from the bid side."""
        level = self.bid_orders[price]
        order = level.orders[0]  # Get the order
        
        # Remove from level
        level.remove_order(order)
        del self.orders_by_id[order_id]
        self.total_bid_volume -= order.remaining_quantity
        
        # Track update
        self._track_update(OrderBookSide.BID, price, -order.remaining_quantity, -1, "REMOVE")
        
        # Remove price level if empty
        if level.order_count == 0:
            del self.bid_orders[price]
            # Note: We don't remove from heap here for performance
        
        logger.debug(f"Removed bid order {order_id} at price {price}")
        return True
    
    def _remove_ask_order(self, order_id: str, price: Decimal) -> bool:
        """Remove a sell order from the ask side."""
        level = self.ask_orders[price]
        order = level.orders[0]  # Get the order
        
        # Remove from level
        level.remove_order(order)
        del self.orders_by_id[order_id]
        self.total_ask_volume -= order.remaining_quantity
        
        # Track update
        self._track_update(OrderBookSide.ASK, price, -order.remaining_quantity, -1, "REMOVE")
        
        # Remove price level if empty
        if level.order_count == 0:
            del self.ask_orders[price]
            # Note: We don't remove from price list here for performance
        
        logger.debug(f"Removed ask order {order_id} at price {price}")
        return True
    
    def update_order(self, order_id: str, new_quantity: Decimal) -> bool:
        """
        Update an order's quantity in the order book.
        
        Args:
            order_id: ID of order to update
            new_quantity: New quantity
            
        Returns:
            True if order was updated successfully, False otherwise
        """
        if order_id not in self.orders_by_id:
            logger.warning(f"Order {order_id} not found in order book")
            return False
        
        order, price = self.orders_by_id[order_id]
        old_quantity = order.remaining_quantity
        quantity_diff = new_quantity - old_quantity
        
        if quantity_diff == 0:
            return True  # No change
        
        # Update order
        order.remaining_quantity = new_quantity
        
        # Update level
        level = self.bid_orders[price] if order.side == OrderSide.BUY else self.ask_orders[price]
        level.update_order(order, old_quantity)
        
        # Update total volume
        if order.side == OrderSide.BUY:
            self.total_bid_volume += quantity_diff
        else:
            self.total_ask_volume += quantity_diff
        
        # Track update
        self._track_update(
            OrderBookSide.BID if order.side == OrderSide.BUY else OrderBookSide.ASK,
            price, quantity_diff, 0, "UPDATE"
        )
        
        logger.debug(f"Updated order {order_id} quantity from {old_quantity} to {new_quantity}")
        return True
    
    def get_best_bid(self) -> Optional[Decimal]:
        """Get the best bid price (highest buy price)."""
        if not self.bid_prices:
            return None
        
        # Clean up empty price levels
        while self.bid_prices and self.bid_prices[0] not in self.bid_orders:
            heapq.heappop(self.bid_prices)
        
        if not self.bid_prices:
            return None
        
        return -self.bid_prices[0]  # Convert back from negative
    
    def get_best_ask(self) -> Optional[Decimal]:
        """Get the best ask price (lowest sell price)."""
        if not self.ask_prices:
            return None
        
        # Clean up empty price levels
        while self.ask_prices and self.ask_prices[0] not in self.ask_orders:
            heapq.heappop(self.ask_prices)
        
        if not self.ask_prices:
            return None
        
        return self.ask_prices[0]
    
    def get_mid_price(self) -> Optional[Decimal]:
        """Get the mid price between best bid and ask."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return (best_bid + best_ask) / 2
    
    def get_spread(self) -> Optional[Decimal]:
        """Get the spread between best bid and ask."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return best_ask - best_bid
    
    def get_level_1_snapshot(self) -> Dict[str, Any]:
        """Get Level 1 (top-of-book) snapshot."""
        return {
            "instrument_id": self.instrument_id,
            "timestamp": datetime.utcnow(),
            "best_bid": self.get_best_bid(),
            "best_ask": self.get_best_ask(),
            "mid_price": self.get_mid_price(),
            "spread": self.get_spread(),
            "best_bid_quantity": self.bid_orders[self.get_best_bid()].total_quantity if self.get_best_bid() else None,
            "best_ask_quantity": self.ask_orders[self.get_best_ask()].total_quantity if self.get_best_ask() else None,
        }
    
    def get_level_2_snapshot(self, max_levels: Optional[int] = None) -> Dict[str, Any]:
        """Get Level 2 (depth) snapshot."""
        if max_levels is None:
            max_levels = self.max_levels
        
        # Get top bid levels
        bid_levels = []
        for price in sorted(self.bid_orders.keys(), reverse=True)[:max_levels]:
            level = self.bid_orders[price]
            bid_levels.append({
                "price": price,
                "quantity": level.total_quantity,
                "order_count": level.order_count
            })
        
        # Get top ask levels
        ask_levels = []
        for price in sorted(self.ask_orders.keys())[:max_levels]:
            level = self.ask_orders[price]
            ask_levels.append({
                "price": price,
                "quantity": level.total_quantity,
                "order_count": level.order_count
            })
        
        return {
            "instrument_id": self.instrument_id,
            "timestamp": datetime.utcnow(),
            "bid_levels": bid_levels,
            "ask_levels": ask_levels,
            "total_bid_volume": self.total_bid_volume,
            "total_ask_volume": self.total_ask_volume,
        }
    
    def get_full_snapshot(self) -> OrderBookSnapshot:
        """Get a complete snapshot of the order book."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        return OrderBookSnapshot(
            timestamp=datetime.utcnow(),
            instrument_id=self.instrument_id,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=self.get_mid_price(),
            spread=self.get_spread(),
            bid_levels=[(price, level.total_quantity) for price, level in 
                       sorted(self.bid_orders.items(), reverse=True)],
            ask_levels=[(price, level.total_quantity) for price, level in 
                       sorted(self.ask_orders.items())],
            total_bid_volume=self.total_bid_volume,
            total_ask_volume=self.total_ask_volume
        )
    
    def get_pending_updates(self) -> List[OrderBookUpdate]:
        """Get pending updates since last call."""
        updates = self.pending_updates.copy()
        self.pending_updates.clear()
        return updates
    
    def _track_update(self, side: OrderBookSide, price: Decimal, 
                     quantity_change: Decimal, order_count_change: int, 
                     update_type: str) -> None:
        """Track an update for WebSocket streaming."""
        self.update_sequence += 1
        self.last_update = datetime.utcnow()
        
        update = OrderBookUpdate(
            timestamp=self.last_update,
            instrument_id=self.instrument_id,
            side=side,
            price=price,
            quantity_change=quantity_change,
            order_count_change=order_count_change,
            update_type=update_type
        )
        
        self.pending_updates.append(update)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get order book statistics."""
        return {
            "instrument_id": self.instrument_id,
            "total_orders": len(self.orders_by_id),
            "bid_levels": len(self.bid_orders),
            "ask_levels": len(self.ask_orders),
            "total_bid_volume": self.total_bid_volume,
            "total_ask_volume": self.total_ask_volume,
            "last_update": self.last_update,
            "update_sequence": self.update_sequence
        }
    
    def clear(self) -> None:
        """Clear all orders from the order book."""
        self.bid_orders.clear()
        self.ask_orders.clear()
        self.orders_by_id.clear()
        self.bid_prices.clear()
        self.ask_prices.clear()
        self.total_bid_volume = Decimal('0')
        self.total_ask_volume = Decimal('0')
        self.pending_updates.clear()
        self.update_sequence = 0
        
        logger.info(f"Order book cleared for instrument {self.instrument_id}")


class OrderBookManager:
    """
    Manages multiple order books across different instruments.
    
    Provides:
    - Instrument-specific order books
    - Global order book statistics
    - Batch operations across instruments
    """
    
    def __init__(self):
        """Initialize the order book manager."""
        self.order_books: Dict[str, OrderBook] = {}
        self.logger = logger
    
    def get_order_book(self, instrument_id: str) -> OrderBook:
        """Get or create an order book for an instrument."""
        if instrument_id not in self.order_books:
            self.order_books[instrument_id] = OrderBook(instrument_id)
            self.logger.info(f"Created new order book for instrument {instrument_id}")
        
        return self.order_books[instrument_id]
    
    def remove_order_book(self, instrument_id: str) -> bool:
        """Remove an order book for an instrument."""
        if instrument_id in self.order_books:
            del self.order_books[instrument_id]
            self.logger.info(f"Removed order book for instrument {instrument_id}")
            return True
        return False
    
    def get_all_snapshots(self) -> Dict[str, OrderBookSnapshot]:
        """Get snapshots for all order books."""
        return {
            instrument_id: order_book.get_full_snapshot()
            for instrument_id, order_book in self.order_books.items()
        }
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global statistics across all order books."""
        total_orders = sum(len(ob.orders_by_id) for ob in self.order_books.values())
        total_bid_volume = sum(ob.total_bid_volume for ob in self.order_books.values())
        total_ask_volume = sum(ob.total_ask_volume for ob in self.order_books.values())
        
        return {
            "total_instruments": len(self.order_books),
            "total_orders": total_orders,
            "total_bid_volume": total_bid_volume,
            "total_ask_volume": total_ask_volume,
            "timestamp": datetime.utcnow()
        }
    
    def clear_all(self) -> None:
        """Clear all order books."""
        for order_book in self.order_books.values():
            order_book.clear()
        self.order_books.clear()
        self.logger.info("All order books cleared")


# Export classes
__all__ = ["OrderBook", "OrderBookManager", "OrderBookSnapshot", "OrderBookUpdate", "OrderBookLevel"]
