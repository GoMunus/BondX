"""
Execution Engine for BondX Trading Engine.

This module provides comprehensive trade execution including:
- Order matching algorithms
- Trade execution and settlement
- Order lifecycle management
- Real-time execution reporting
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
    OrderBook, OrderBookEntry, Trade, MarketData
)
from .order_manager import OrderManager

logger = get_logger(__name__)


class ExecutionResult(Enum):
    """Trade execution results."""
    
    EXECUTED = "EXECUTED"
    PARTIALLY_EXECUTED = "PARTIALLY_EXECUTED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


@dataclass
class ExecutionReport:
    """Trade execution report."""
    
    execution_id: str
    order_id: str
    execution_result: ExecutionResult
    executed_quantity: Decimal
    executed_price: Decimal
    execution_time: datetime
    trade_id: Optional[str] = None
    remaining_quantity: Decimal = Decimal('0')
    fees: Decimal = Decimal('0')
    taxes: Decimal = Decimal('0')
    market_impact: Optional[Decimal] = None
    execution_venue: str = "SECONDARY_MARKET"
    counterparty_id: Optional[int] = None
    metadata: Dict[str, Any] = None


class ExecutionEngine:
    """
    Comprehensive trade execution engine.
    
    This engine provides:
    - Price-time priority order matching
    - Multiple order type execution
    - Trade execution and reporting
    - Market impact analysis
    """
    
    def __init__(self, db_session: Session, order_manager: OrderManager):
        """Initialize the execution engine."""
        self.db_session = db_session
        self.order_manager = order_manager
        self.logger = get_logger(__name__)
        self.metrics = MetricsCollector()
        
        # Execution tracking
        self.execution_history: Dict[str, List[ExecutionReport]] = {}
        self.trade_history: Dict[str, Trade] = {}
        self.active_executions: Dict[str, ExecutionReport] = {}
        
        # Market data
        self.market_data: Dict[str, MarketData] = {}
        
        # Execution parameters
        self.min_tick_size = Decimal('0.01')  # Minimum price increment
        self.max_slippage = Decimal('0.05')   # Maximum allowed slippage (5%)
        self.execution_delay_ms = 1           # Simulated execution delay
        
        # Performance tracking
        self.total_trades_executed = 0
        self.total_volume_executed = Decimal('0')
        self.average_execution_time_ms = 0
        
        logger.info("Execution Engine initialized successfully")
    
    async def execute_orders(self) -> List[ExecutionReport]:
        """
        Execute all executable orders in the order book.
        
        Returns:
            List of execution reports
        """
        try:
            execution_reports = []
            
            # Get all order books
            for bond_id in self.order_manager.order_books:
                order_book = await self.order_manager.get_order_book(bond_id)
                if order_book:
                    # Execute orders for this bond
                    bond_executions = await self._execute_bond_orders(order_book)
                    execution_reports.extend(bond_executions)
            
            # Update metrics
            if execution_reports:
                self.total_trades_executed += len(executions)
                self.metrics.increment_counter("trades_executed_total", {"count": len(executions)})
            
            return execution_reports
            
        except Exception as e:
            self.logger.error(f"Error executing orders: {str(e)}")
            self.metrics.increment_counter("execution_errors_total", {"error": str(e)})
            return []
    
    async def execute_single_order(self, order: Order) -> List[ExecutionReport]:
        """
        Execute a single order against the order book.
        
        Args:
            order: Order to execute
            
        Returns:
            List of execution reports
        """
        try:
            if order.status != OrderStatus.VALIDATED:
                self.logger.warning(f"Order {order.order_id} not in valid status for execution: {order.status}")
                return []
            
            # Get order book for this bond
            order_book = await self.order_manager.get_order_book(order.bond_id)
            if not order_book:
                self.logger.warning(f"No order book found for bond {order.bond_id}")
                return []
            
            # Execute order
            execution_reports = await self._execute_order_against_book(order, order_book)
            
            # Update order status
            if execution_reports:
                total_executed = sum(report.executed_quantity for report in execution_reports)
                if total_executed >= order.quantity:
                    order.status = OrderStatus.FILLED
                elif total_executed > 0:
                    order.status = OrderStatus.PARTIALLY_FILLED
                
                order.filled_quantity = total_executed
                order.remaining_quantity = order.quantity - total_executed
                order.updated_at = datetime.utcnow()
                
                # Update order book
                await self._update_order_book_after_execution(order, execution_reports)
            
            return execution_reports
            
        except Exception as e:
            self.logger.error(f"Error executing single order {order.order_id}: {str(e)}")
            return []
    
    async def _execute_bond_orders(self, order_book: OrderBook) -> List[ExecutionReport]:
        """
        Execute all executable orders for a specific bond.
        
        Args:
            order_book: Order book for the bond
            
        Returns:
            List of execution reports
        """
        execution_reports = []
        
        try:
            # Get best bid and ask
            best_bid = order_book.get_best_bid()
            best_ask = order_book.get_best_ask()
            
            if not best_bid or not best_ask:
                return execution_reports
            
            # Check if orders can cross
            while best_bid and best_ask and best_bid.price >= best_ask.price:
                # Find orders to match
                buy_orders = [o for o in best_bid.orders if o.status == OrderStatus.VALIDATED]
                sell_orders = [o for o in best_ask.orders if o.status == OrderStatus.VALIDATED]
                
                if not buy_orders or not sell_orders:
                    break
                
                # Execute cross
                cross_executions = await self._execute_cross(buy_orders, sell_orders, best_bid.price, best_ask.price)
                execution_reports.extend(cross_executions)
                
                # Update order book
                await self._update_order_book_after_cross(order_book, cross_executions)
                
                # Get updated best bid and ask
                best_bid = order_book.get_best_bid()
                best_ask = order_book.get_best_ask()
            
            return execution_reports
            
        except Exception as e:
            self.logger.error(f"Error executing bond orders: {str(e)}")
            return execution_reports
    
    async def _execute_cross(self, buy_orders: List[Order], sell_orders: List[Order], 
                           bid_price: Decimal, ask_price: Decimal) -> List[ExecutionReport]:
        """
        Execute a cross between buy and sell orders.
        
        Args:
            buy_orders: List of buy orders
            sell_orders: List of sell orders
            bid_price: Best bid price
            ask_price: Best ask price
            
        Returns:
            List of execution reports
        """
        execution_reports = []
        
        try:
            # Sort orders by priority (time first, then other factors)
            buy_orders.sort(key=lambda x: (x.created_at, x.priority.value))
            sell_orders.sort(key=lambda x: (x.created_at, x.priority.value))
            
            # Calculate execution price (mid-price or weighted average)
            execution_price = (bid_price + ask_price) / 2
            
            # Execute orders
            buy_idx = 0
            sell_idx = 0
            
            while buy_idx < len(buy_orders) and sell_idx < len(sell_orders):
                buy_order = buy_orders[buy_idx]
                sell_order = sell_orders[sell_idx]
                
                # Calculate executable quantity
                executable_quantity = min(
                    buy_order.remaining_quantity,
                    sell_order.remaining_quantity
                )
                
                if executable_quantity <= 0:
                    break
                
                # Create execution reports
                buy_execution = ExecutionReport(
                    execution_id=f"exec_{buy_order.order_id}_{datetime.utcnow().timestamp()}",
                    order_id=buy_order.order_id,
                    execution_result=ExecutionResult.EXECUTED,
                    executed_quantity=executable_quantity,
                    executed_price=execution_price,
                    execution_time=datetime.utcnow(),
                    remaining_quantity=buy_order.remaining_quantity - executable_quantity,
                    execution_venue="SECONDARY_MARKET",
                    counterparty_id=sell_order.participant_id
                )
                
                sell_execution = ExecutionReport(
                    execution_id=f"exec_{sell_order.order_id}_{datetime.utcnow().timestamp()}",
                    order_id=sell_order.order_id,
                    execution_result=ExecutionResult.EXECUTED,
                    executed_quantity=executable_quantity,
                    executed_price=execution_price,
                    execution_time=datetime.utcnow(),
                    remaining_quantity=sell_order.remaining_quantity - executable_quantity,
                    execution_venue="SECONDARY_MARKET",
                    counterparty_id=buy_order.participant_id
                )
                
                execution_reports.extend([buy_execution, sell_execution])
                
                # Update order quantities
                buy_order.remaining_quantity -= executable_quantity
                sell_order.remaining_quantity -= executable_quantity
                
                # Check if orders are fully filled
                if buy_order.remaining_quantity <= 0:
                    buy_idx += 1
                if sell_order.remaining_quantity <= 0:
                    sell_idx += 1
            
            return execution_reports
            
        except Exception as e:
            self.logger.error(f"Error executing cross: {str(e)}")
            return execution_reports
    
    async def _execute_order_against_book(self, order: Order, order_book: OrderBook) -> List[ExecutionReport]:
        """
        Execute a single order against the order book.
        
        Args:
            order: Order to execute
            order_book: Order book to execute against
            
        Returns:
            List of execution reports
        """
        execution_reports = []
        
        try:
            if order.side == OrderSide.BUY:
                # Execute against ask orders
                entries = order_book.ask_entries
                price_comparison = lambda x: x.price <= order.price
            else:
                # Execute against bid orders
                entries = order_book.bid_entries
                price_comparison = lambda x: x.price >= order.price
            
            remaining_quantity = order.remaining_quantity
            
            for entry in entries:
                if not price_comparison(entry) or remaining_quantity <= 0:
                    break
                
                # Execute against this price level
                level_executions = await self._execute_against_price_level(
                    order, entry, remaining_quantity
                )
                
                execution_reports.extend(level_executions)
                
                # Update remaining quantity
                executed_quantity = sum(exec.executed_quantity for exec in level_executions)
                remaining_quantity -= executed_quantity
                
                if remaining_quantity <= 0:
                    break
            
            return execution_reports
            
        except Exception as e:
            self.logger.error(f"Error executing order against book: {str(e)}")
            return execution_reports
    
    async def _execute_against_price_level(self, order: Order, price_level: OrderBookEntry, 
                                        max_quantity: Decimal) -> List[ExecutionReport]:
        """
        Execute an order against a specific price level.
        
        Args:
            order: Order to execute
            price_level: Price level to execute against
            max_quantity: Maximum quantity to execute
            
        Returns:
            List of execution reports
        """
        execution_reports = []
        
        try:
            # Sort orders in price level by time priority
            level_orders = sorted(price_level.orders, key=lambda x: x.created_at)
            
            remaining_quantity = max_quantity
            
            for level_order in level_orders:
                if remaining_quantity <= 0:
                    break
                
                if level_order.status != OrderStatus.VALIDATED:
                    continue
                
                # Calculate executable quantity
                executable_quantity = min(
                    remaining_quantity,
                    level_order.remaining_quantity
                )
                
                if executable_quantity <= 0:
                    continue
                
                # Create execution report
                execution_report = ExecutionReport(
                    execution_id=f"exec_{order.order_id}_{level_order.order_id}_{datetime.utcnow().timestamp()}",
                    order_id=order.order_id,
                    execution_result=ExecutionResult.EXECUTED,
                    executed_quantity=executable_quantity,
                    executed_price=price_level.price,
                    execution_time=datetime.utcnow(),
                    remaining_quantity=remaining_quantity - executable_quantity,
                    execution_venue="SECONDARY_MARKET",
                    counterparty_id=level_order.participant_id
                )
                
                execution_reports.append(execution_report)
                
                # Update remaining quantity
                remaining_quantity -= executable_quantity
                
                # Update level order
                level_order.remaining_quantity -= executable_quantity
                
                # Check if level order is fully filled
                if level_order.remaining_quantity <= 0:
                    level_order.orders.remove(level_order)
            
            return execution_reports
            
        except Exception as e:
            self.logger.error(f"Error executing against price level: {str(e)}")
            return execution_reports
    
    async def _update_order_book_after_execution(self, order: Order, execution_reports: List[ExecutionReport]):
        """
        Update order book after order execution.
        
        Args:
            order: Executed order
            execution_reports: Execution reports
        """
        try:
            # Update order book entries
            if order.side == OrderSide.BUY:
                entries = self.order_manager.order_books[order.bond_id].ask_entries
            else:
                entries = self.order_manager.order_books[order.bond_id].bid_entries
            
            # Remove fully filled orders
            for entry in entries[:]:  # Copy list to avoid modification during iteration
                entry.orders = [o for o in entry.orders if o.remaining_quantity > 0]
                if not entry.orders:
                    entries.remove(entry)
                else:
                    # Update total quantity
                    entry.total_quantity = sum(o.remaining_quantity for o in entry.orders)
                    entry.order_count = len(entry.orders)
            
            # Update order book timestamp
            self.order_manager.order_books[order.bond_id].last_updated = datetime.utcnow()
            
        except Exception as e:
            self.logger.error(f"Error updating order book after execution: {str(e)}")
    
    async def _update_order_book_after_cross(self, order_book: OrderBook, execution_reports: List[ExecutionReport]):
        """
        Update order book after cross execution.
        
        Args:
            order_book: Order book to update
            execution_reports: Execution reports
        """
        try:
            # Update bid entries
            for entry in order_book.bid_entries[:]:
                entry.orders = [o for o in entry.orders if o.remaining_quantity > 0]
                if not entry.orders:
                    order_book.bid_entries.remove(entry)
                else:
                    entry.total_quantity = sum(o.remaining_quantity for o in entry.orders)
                    entry.order_count = len(entry.orders)
            
            # Update ask entries
            for entry in order_book.ask_entries[:]:
                entry.orders = [o for o in entry.orders if o.remaining_quantity > 0]
                if not entry.orders:
                    order_book.ask_entries.remove(entry)
                else:
                    entry.total_quantity = sum(o.remaining_quantity for o in entry.orders)
                    entry.order_count = len(entry.orders)
            
            # Update timestamp
            order_book.last_updated = datetime.utcnow()
            
        except Exception as e:
            self.logger.error(f"Error updating order book after cross: {str(e)}")
    
    async def create_trade_from_execution(self, execution_report: ExecutionReport) -> Trade:
        """
        Create a trade record from an execution report.
        
        Args:
            execution_report: Execution report
            
        Returns:
            Trade object
        """
        try:
            # Get the order
            order = self.order_manager.active_orders.get(execution_report.order_id)
            if not order:
                # Try to find in history
                order_history = await self.order_manager.get_order_history(execution_report.order_id)
                if order_history:
                    order = order_history[-1]
            
            if not order:
                raise ValueError(f"Order {execution_report.order_id} not found")
            
            # Create trade
            trade = Trade(
                trade_id=execution_report.execution_id,
                order_id=execution_report.order_id,
                participant_id=order.participant_id,
                bond_id=order.bond_id,
                side=order.side,
                quantity=execution_report.executed_quantity,
                price=execution_report.executed_price,
                trade_value=execution_report.executed_quantity * execution_report.executed_price,
                execution_time=execution_report.execution_time,
                trade_type="REGULAR",
                counterparty_id=execution_report.counterparty_id,
                venue=execution_report.execution_venue,
                fees=execution_report.fees,
                taxes=execution_report.taxes,
                net_value=execution_report.executed_quantity * execution_report.executed_price - execution_report.fees - execution_report.taxes
            )
            
            # Store trade
            self.trade_history[trade.trade_id] = trade
            
            # Update market data
            await self._update_market_data(trade)
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Error creating trade from execution: {str(e)}")
            raise
    
    async def _update_market_data(self, trade: Trade):
        """
        Update market data with trade information.
        
        Args:
            trade: Trade to update market data with
        """
        try:
            if trade.bond_id not in self.market_data:
                self.market_data[trade.bond_id] = MarketData(bond_id=trade.bond_id)
            
            market_data = self.market_data[trade.bond_id]
            market_data.update_from_trade(trade)
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {str(e)}")
    
    async def get_execution_history(self, order_id: Optional[str] = None) -> List[ExecutionReport]:
        """
        Get execution history, optionally filtered by order ID.
        
        Args:
            order_id: Optional order ID filter
            
        Returns:
            List of execution reports
        """
        if order_id is None:
            all_reports = []
            for reports in self.execution_history.values():
                all_reports.extend(reports)
            return all_reports
        
        return self.execution_history.get(order_id, [])
    
    async def get_trade_history(self, bond_id: Optional[str] = None) -> List[Trade]:
        """
        Get trade history, optionally filtered by bond ID.
        
        Args:
            bond_id: Optional bond ID filter
            
        Returns:
            List of trades
        """
        if bond_id is None:
            return list(self.trade_history.values())
        
        return [trade for trade in self.trade_history.values() if trade.bond_id == bond_id]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Dictionary of system statistics
        """
        return {
            "total_trades_executed": self.total_trades_executed,
            "total_volume_executed": float(self.total_volume_executed),
            "average_execution_time_ms": self.average_execution_time_ms,
            "active_executions_count": len(self.active_executions),
            "execution_history_count": sum(len(reports) for reports in self.execution_history.values()),
            "trade_history_count": len(self.trade_history),
            "market_data_count": len(self.market_data),
            "last_updated": datetime.utcnow().isoformat()
        }
