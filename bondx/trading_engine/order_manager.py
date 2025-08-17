"""
Order Management System for BondX Trading Engine.

This module provides comprehensive order management including:
- Order validation and risk checks
- Order lifecycle management
- Order book maintenance
- Order matching preparation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from ..core.logging import get_logger
from ..core.monitoring import MetricsCollector
from .trading_models import (
    Order, OrderType, OrderSide, OrderStatus, TimeInForce, OrderPriority,
    OrderBook, OrderBookEntry, MarketData
)
from ..ai_risk_engine.risk_scoring import RiskScoringEngine

logger = get_logger(__name__)


class OrderValidationResult(Enum):
    """Order validation results."""
    
    VALID = "VALID"
    INVALID_QUANTITY = "INVALID_QUANTITY"
    INVALID_PRICE = "INVALID_PRICE"
    INVALID_TIMING = "INVALID_TIMING"
    RISK_CHECK_FAILED = "RISK_CHECK_FAILED"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    POSITION_LIMIT_EXCEEDED = "POSITION_LIMIT_EXCEEDED"
    MARKET_CLOSED = "MARKET_CLOSED"
    INVALID_ORDER_TYPE = "INVALID_ORDER_TYPE"


@dataclass
class OrderValidation:
    """Order validation details."""
    
    is_valid: bool
    result: OrderValidationResult
    error_message: Optional[str] = None
    risk_score: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


class OrderManager:
    """
    Comprehensive order management system.
    
    This manager provides:
    - Order validation and risk checks
    - Order lifecycle management
    - Order book maintenance
    - Integration with risk management
    """
    
    def __init__(self, db_session: Session):
        """Initialize the order manager."""
        self.db_session = db_session
        self.logger = get_logger(__name__)
        self.metrics = MetricsCollector()
        
        # Order book management
        self.order_books: Dict[str, OrderBook] = {}
        self.active_orders: Dict[str, Order] = {}
        self.order_history: Dict[str, List[Order]] = {}
        
        # Risk management
        self.risk_engine = RiskScoringEngine()
        
        # Market hours (configurable)
        self.market_open_time = "09:00"
        self.market_close_time = "17:00"
        self.market_timezone = "Asia/Kolkata"
        
        # Order limits
        self.max_order_quantity = Decimal('1000000')  # 1M bonds
        self.min_order_quantity = Decimal('100')      # 100 bonds
        self.max_price_deviation = Decimal('0.10')    # 10% from market price
        
        logger.info("Order Manager initialized successfully")
    
    async def submit_order(self, order_data: Dict[str, Any]) -> Tuple[Order, OrderValidation]:
        """
        Submit a new order with comprehensive validation.
        
        Args:
            order_data: Order data dictionary
            
        Returns:
            Tuple of (Order, OrderValidation)
        """
        try:
            # Create order object
            order = Order(**order_data)
            
            # Validate order
            validation = await self._validate_order(order)
            
            if not validation.is_valid:
                order.status = OrderStatus.REJECTED
                self.logger.warning(f"Order {order.order_id} rejected: {validation.error_message}")
                return order, validation
            
            # Perform risk checks
            risk_check = await self._perform_risk_checks(order)
            if not risk_check:
                validation.is_valid = False
                validation.result = OrderValidationResult.RISK_CHECK_FAILED
                validation.error_message = "Order failed risk checks"
                order.status = OrderStatus.REJECTED
                return order, validation
            
            # Add to active orders
            self.active_orders[order.order_id] = order
            order.status = OrderStatus.VALIDATED
            
            # Update order book
            await self._update_order_book(order)
            
            # Log order submission
            self.logger.info(f"Order {order.order_id} submitted successfully")
            self.metrics.increment_counter("orders_submitted_total", {"status": "success"})
            
            return order, validation
            
        except Exception as e:
            self.logger.error(f"Error submitting order: {str(e)}")
            self.metrics.increment_counter("orders_submitted_total", {"status": "error"})
            raise
    
    async def cancel_order(self, order_id: str, participant_id: int) -> bool:
        """
        Cancel an active order.
        
        Args:
            order_id: Order ID to cancel
            participant_id: ID of participant requesting cancellation
            
        Returns:
            True if order was cancelled successfully
        """
        try:
            if order_id not in self.active_orders:
                self.logger.warning(f"Order {order_id} not found for cancellation")
                return False
            
            order = self.active_orders[order_id]
            
            # Check ownership
            if order.participant_id != participant_id:
                self.logger.warning(f"Participant {participant_id} not authorized to cancel order {order_id}")
                return False
            
            # Check if order can be cancelled
            if order.status not in [OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED]:
                self.logger.warning(f"Order {order_id} cannot be cancelled in status {order.status}")
                return False
            
            # Cancel order
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.utcnow()
            
            # Remove from active orders
            del self.active_orders[order_id]
            
            # Update order book
            await self._remove_from_order_book(order)
            
            # Move to history
            if order_id not in self.order_history:
                self.order_history[order_id] = []
            self.order_history[order_id].append(order)
            
            self.logger.info(f"Order {order_id} cancelled successfully")
            self.metrics.increment_counter("orders_cancelled_total", {"status": "success"})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {str(e)}")
            self.metrics.increment_counter("orders_cancelled_total", {"status": "error"})
            return False
    
    async def modify_order(self, order_id: str, participant_id: int, 
                          modifications: Dict[str, Any]) -> Tuple[Order, bool]:
        """
        Modify an active order.
        
        Args:
            order_id: Order ID to modify
            participant_id: ID of participant requesting modification
            modifications: Dictionary of modifications
            
        Returns:
            Tuple of (modified Order, success status)
        """
        try:
            if order_id not in self.active_orders:
                self.logger.warning(f"Order {order_id} not found for modification")
                return None, False
            
            order = self.active_orders[order_id]
            
            # Check ownership
            if order.participant_id != participant_id:
                self.logger.warning(f"Participant {participant_id} not authorized to modify order {order_id}")
                return None, False
            
            # Check if order can be modified
            if order.status not in [OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED]:
                self.logger.warning(f"Order {order_id} cannot be modified in status {order.status}")
                return None, False
            
            # Apply modifications
            old_order = Order(**order.__dict__)
            
            for field, value in modifications.items():
                if hasattr(order, field) and field not in ['order_id', 'created_at']:
                    setattr(order, field, value)
            
            # Re-validate modified order
            validation = await self._validate_order(order)
            if not validation.is_valid:
                # Revert changes
                for field, value in old_order.__dict__.items():
                    if hasattr(order, field):
                        setattr(order, field, value)
                
                self.logger.warning(f"Order {order_id} modification failed validation: {validation.error_message}")
                return order, False
            
            # Update order book
            await self._remove_from_order_book(old_order)
            await self._update_order_book(order)
            
            order.updated_at = datetime.utcnow()
            
            self.logger.info(f"Order {order_id} modified successfully")
            self.metrics.increment_counter("orders_modified_total", {"status": "success"})
            
            return order, True
            
        except Exception as e:
            self.logger.error(f"Error modifying order {order_id}: {str(e)}")
            self.metrics.increment_counter("orders_modified_total", {"status": "error"})
            return None, False
    
    async def get_order_book(self, bond_id: str) -> Optional[OrderBook]:
        """
        Get current order book for a bond.
        
        Args:
            bond_id: Bond identifier
            
        Returns:
            OrderBook object or None if not found
        """
        return self.order_books.get(bond_id)
    
    async def get_active_orders(self, participant_id: Optional[int] = None) -> List[Order]:
        """
        Get active orders, optionally filtered by participant.
        
        Args:
            participant_id: Optional participant ID filter
            
        Returns:
            List of active orders
        """
        if participant_id is None:
            return list(self.active_orders.values())
        
        return [order for order in self.active_orders.values() 
                if order.participant_id == participant_id]
    
    async def get_order_history(self, order_id: str) -> List[Order]:
        """
        Get order history for a specific order.
        
        Args:
            order_id: Order ID
            
        Returns:
            List of order history entries
        """
        return self.order_history.get(order_id, [])
    
    async def _validate_order(self, order: Order) -> OrderValidation:
        """
        Validate an order for basic requirements.
        
        Args:
            order: Order to validate
            
        Returns:
            OrderValidation object
        """
        warnings = []
        
        # Check quantity
        if order.quantity < self.min_order_quantity:
            return OrderValidation(
                is_valid=False,
                result=OrderValidationResult.INVALID_QUANTITY,
                error_message=f"Order quantity {order.quantity} below minimum {self.min_order_quantity}"
            )
        
        if order.quantity > self.max_order_quantity:
            return OrderValidation(
                is_valid=False,
                result=OrderValidationResult.INVALID_QUANTITY,
                error_message=f"Order quantity {order.quantity} above maximum {self.max_order_quantity}"
            )
        
        # Check price for limit orders
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.price is None or order.price <= 0:
                return OrderValidation(
                    is_valid=False,
                    result=OrderValidationResult.INVALID_PRICE,
                    error_message="Limit orders must have a positive price"
                )
        
        # Check stop price for stop orders
        if order.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LIMIT]:
            if order.stop_price is None or order.stop_price <= 0:
                return OrderValidation(
                    is_valid=False,
                    result=OrderValidationResult.INVALID_PRICE,
                    error_message="Stop orders must have a positive stop price"
                )
        
        # Check market hours
        if not self._is_market_open():
            return OrderValidation(
                is_valid=False,
                result=OrderValidationResult.MARKET_CLOSED,
                error_message="Market is currently closed"
            )
        
        # Check order type validity
        if not self._is_valid_order_type(order.order_type):
            return OrderValidation(
                is_valid=False,
                result=OrderValidationResult.INVALID_ORDER_TYPE,
                error_message=f"Order type {order.order_type} not supported"
            )
        
        # Check time in force
        if not self._is_valid_time_in_force(order.time_in_force):
            return OrderValidation(
                is_valid=False,
                result=OrderValidationResult.INVALID_TIMING,
                error_message=f"Time in force {order.time_in_force} not supported"
            )
        
        # Check expiration for GTC orders
        if order.time_in_force == TimeInForce.GOOD_TILL_CANCELLED:
            if order.expires_at is None:
                warnings.append("GTC orders should have an expiration date")
        
        return OrderValidation(
            is_valid=True,
            result=OrderValidationResult.VALID,
            warnings=warnings
        )
    
    async def _perform_risk_checks(self, order: Order) -> bool:
        """
        Perform risk checks on an order.
        
        Args:
            order: Order to check
            
        Returns:
            True if risk checks pass
        """
        try:
            # Basic risk checks
            risk_score = await self.risk_engine.calculate_order_risk(order)
            order.risk_score = risk_score
            
            # Check if risk score is acceptable
            if risk_score > 0.8:  # High risk threshold
                self.logger.warning(f"Order {order.order_id} has high risk score: {risk_score}")
                return False
            
            # Additional risk checks can be added here
            # - Position limits
            # - Concentration limits
            # - Credit limits
            # - Liquidity checks
            
            order.risk_check_passed = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error performing risk checks on order {order.order_id}: {str(e)}")
            return False
    
    async def _update_order_book(self, order: Order):
        """
        Update order book with a new order.
        
        Args:
            order: Order to add to order book
        """
        if order.bond_id not in self.order_books:
            self.order_books[order.bond_id] = OrderBook(bond_id=order.bond_id)
        
        order_book = self.order_books[order.bond_id]
        
        # Find or create price level
        price_level = None
        if order.side == OrderSide.BUY:
            entries = order_book.bid_entries
        else:
            entries = order_book.ask_entries
        
        for entry in entries:
            if entry.price == order.price:
                price_level = entry
                break
        
        if price_level is None:
            price_level = OrderBookEntry(
                price=order.price,
                total_quantity=Decimal('0'),
                order_count=0,
                side=order.side
            )
            entries.append(price_level)
            
            # Sort entries by price (bids descending, asks ascending)
            if order.side == OrderSide.BUY:
                entries.sort(key=lambda x: x.price, reverse=True)
            else:
                entries.sort(key=lambda x: x.price)
        
        # Add order to price level
        price_level.add_order(order)
        order_book.last_updated = datetime.utcnow()
    
    async def _remove_from_order_book(self, order: Order):
        """
        Remove an order from the order book.
        
        Args:
            order: Order to remove
        """
        if order.bond_id not in self.order_books:
            return
        
        order_book = self.order_books[order.bond_id]
        
        # Find price level
        if order.side == OrderSide.BUY:
            entries = order_book.bid_entries
        else:
            entries = order_book.ask_entries
        
        for entry in entries:
            if entry.price == order.price:
                entry.remove_order(order)
                
                # Remove empty price levels
                if entry.total_quantity == 0:
                    entries.remove(entry)
                
                break
        
        order_book.last_updated = datetime.utcnow()
    
    def _is_market_open(self) -> bool:
        """
        Check if market is currently open.
        
        Returns:
            True if market is open
        """
        # Simple implementation - can be enhanced with proper timezone handling
        now = datetime.utcnow()
        current_time = now.strftime("%H:%M")
        
        return self.market_open_time <= current_time <= self.market_close_time
    
    def _is_valid_order_type(self, order_type: OrderType) -> bool:
        """
        Check if order type is valid.
        
        Args:
            order_type: Order type to check
            
        Returns:
            True if order type is valid
        """
        return order_type in OrderType
    
    def _is_valid_time_in_force(self, time_in_force: TimeInForce) -> bool:
        """
        Check if time in force is valid.
        
        Args:
            time_in_force: Time in force to check
            
        Returns:
            True if time in force is valid
        """
        return time_in_force in TimeInForce
    
    async def cleanup_expired_orders(self):
        """Clean up expired orders."""
        try:
            current_time = datetime.utcnow()
            expired_orders = []
            
            for order_id, order in self.active_orders.items():
                if order.expires_at and order.expires_at <= current_time:
                    expired_orders.append(order_id)
            
            for order_id in expired_orders:
                order = self.active_orders[order_id]
                order.status = OrderStatus.EXPIRED
                order.updated_at = current_time
                
                # Remove from active orders
                del self.active_orders[order_id]
                
                # Update order book
                await self._remove_from_order_book(order)
                
                # Move to history
                if order_id not in self.order_history:
                    self.order_history[order_id] = []
                self.order_history[order_id].append(order)
                
                self.logger.info(f"Order {order_id} expired and cleaned up")
            
            if expired_orders:
                self.metrics.increment_counter("orders_expired_total", {"count": len(expired_orders)})
                
        except Exception as e:
            self.logger.error(f"Error cleaning up expired orders: {str(e)}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Dictionary of system statistics
        """
        return {
            "active_orders_count": len(self.active_orders),
            "order_books_count": len(self.order_books),
            "total_order_history": sum(len(history) for history in self.order_history.values()),
            "market_status": "OPEN" if self._is_market_open() else "CLOSED",
            "last_updated": datetime.utcnow().isoformat()
        }
