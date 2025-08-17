"""
Trading API endpoints for BondX.

This module provides comprehensive REST API endpoints for trading operations,
including order management, trade execution, and market data access.
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import json

from ...database.base import get_db
from ...trading_engine.order_manager import OrderManager
from ...trading_engine.execution_engine import ExecutionEngine
from ...trading_engine.trading_models import (
    Order, OrderType, OrderSide, OrderStatus, TimeInForce, OrderPriority,
    OrderBook, MarketData, Trade
)
from ...core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/trading", tags=["trading"])

# Global instances (in production, these would be properly managed)
order_manager: Optional[OrderManager] = None
execution_engine: Optional[ExecutionEngine] = None


def get_order_manager(db: Session = Depends(get_db)) -> OrderManager:
    """Get order manager instance."""
    global order_manager
    if order_manager is None:
        order_manager = OrderManager(db)
    return order_manager


def get_execution_engine(db: Session = Depends(get_db)) -> ExecutionEngine:
    """Get execution engine instance."""
    global execution_engine
    if execution_engine is None:
        if order_manager is None:
            order_manager = OrderManager(db)
        execution_engine = ExecutionEngine(db, order_manager)
    return execution_engine


# Order Management Endpoints

@router.post("/orders", response_model=Dict[str, Any])
async def submit_order(
    order_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    order_manager: OrderManager = Depends(get_order_manager)
):
    """
    Submit a new trading order.
    
    This endpoint accepts order submissions with comprehensive validation and risk checks.
    """
    try:
        logger.info(f"Processing order submission: {order_data.get('order_id', 'Unknown')}")
        
        # Validate required fields
        required_fields = ['participant_id', 'bond_id', 'order_type', 'side', 'quantity']
        for field in required_fields:
            if field not in order_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Submit order
        order, validation = await order_manager.submit_order(order_data)
        
        if not validation.is_valid:
            return {
                "success": False,
                "message": "Order submission failed",
                "validation": {
                    "is_valid": validation.is_valid,
                    "result": validation.result.value,
                    "error_message": validation.error_message,
                    "warnings": validation.warnings
                }
            }
        
        return {
            "success": True,
            "message": "Order submitted successfully",
            "order": {
                "order_id": order.order_id,
                "participant_id": order.participant_id,
                "bond_id": order.bond_id,
                "order_type": order.order_type.value,
                "side": order.side.value,
                "quantity": float(order.quantity),
                "price": float(order.price) if order.price else None,
                "status": order.status.value,
                "created_at": order.created_at.isoformat(),
                "risk_check_passed": order.risk_check_passed,
                "risk_score": order.risk_score
            },
            "validation": {
                "is_valid": validation.is_valid,
                "result": validation.result.value,
                "warnings": validation.warnings
            }
        }
        
    except ValueError as e:
        logger.error(f"Validation error submitting order: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error submitting order: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/orders", response_model=List[Dict[str, Any]])
async def get_orders(
    participant_id: Optional[int] = Query(None, description="Filter by participant ID"),
    bond_id: Optional[str] = Query(None, description="Filter by bond ID"),
    status: Optional[OrderStatus] = Query(None, description="Filter by order status"),
    order_type: Optional[OrderType] = Query(None, description="Filter by order type"),
    side: Optional[OrderSide] = Query(None, description="Filter by order side"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of orders to return"),
    offset: int = Query(0, ge=0, description="Number of orders to skip"),
    order_manager: OrderManager = Depends(get_order_manager)
):
    """
    Get list of orders with optional filtering.
    
    This endpoint returns a paginated list of orders with comprehensive filtering options.
    """
    try:
        # Get active orders
        active_orders = await order_manager.get_active_orders(participant_id)
        
        # Apply additional filters
        filtered_orders = []
        for order in active_orders:
            if bond_id and order.bond_id != bond_id:
                continue
            if status and order.status != status:
                continue
            if order_type and order.order_type != order_type:
                continue
            if side and order.side != side:
                continue
            filtered_orders.append(order)
        
        # Apply pagination
        total_count = len(filtered_orders)
        paginated_orders = filtered_orders[offset:offset + limit]
        
        # Format response
        order_list = []
        for order in paginated_orders:
            order_data = {
                "order_id": order.order_id,
                "participant_id": order.participant_id,
                "bond_id": order.bond_id,
                "order_type": order.order_type.value,
                "side": order.side.value,
                "quantity": float(order.quantity),
                "price": float(order.price) if order.price else None,
                "stop_price": float(order.stop_price) if order.stop_price else None,
                "time_in_force": order.time_in_force.value,
                "priority": order.priority.value,
                "status": order.status.value,
                "filled_quantity": float(order.filled_quantity),
                "average_fill_price": float(order.average_fill_price) if order.average_fill_price else None,
                "remaining_quantity": float(order.remaining_quantity),
                "created_at": order.created_at.isoformat(),
                "updated_at": order.updated_at.isoformat(),
                "expires_at": order.expires_at.isoformat() if order.expires_at else None,
                "risk_check_passed": order.risk_check_passed,
                "risk_score": order.risk_score
            }
            order_list.append(order_data)
        
        return {
            "orders": order_list,
            "pagination": {
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting orders: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/orders/{order_id}", response_model=Dict[str, Any])
async def get_order(
    order_id: str = Path(..., description="ID of the order"),
    order_manager: OrderManager = Depends(get_order_manager)
):
    """
    Get detailed information about a specific order.
    
    This endpoint returns comprehensive order information including execution details.
    """
    try:
        # Get active orders
        active_orders = await order_manager.get_active_orders()
        
        # Find the specific order
        order = None
        for active_order in active_orders:
            if active_order.order_id == order_id:
                order = active_order
                break
        
        if not order:
            # Try to find in history
            order_history = await order_manager.get_order_history(order_id)
            if order_history:
                order = order_history[-1]
        
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        
        # Format response
        order_data = {
            "order_id": order.order_id,
            "participant_id": order.participant_id,
            "bond_id": order.bond_id,
            "order_type": order.order_type.value,
            "side": order.side.value,
            "quantity": float(order.quantity),
            "price": float(order.price) if order.price else None,
            "stop_price": float(order.stop_price) if order.stop_price else None,
            "time_in_force": order.time_in_force.value,
            "priority": order.priority.value,
            "status": order.status.value,
            "filled_quantity": float(order.filled_quantity),
            "average_fill_price": float(order.average_fill_price) if order.average_fill_price else None,
            "remaining_quantity": float(order.remaining_quantity),
            "created_at": order.created_at.isoformat(),
            "updated_at": order.updated_at.isoformat(),
            "expires_at": order.expires_at.isoformat() if order.expires_at else None,
            "risk_check_passed": order.risk_check_passed,
            "risk_score": order.risk_score,
            "metadata": order.metadata
        }
        
        return order_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting order {order_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/orders/{order_id}", response_model=Dict[str, Any])
async def modify_order(
    order_id: str = Path(..., description="ID of the order to modify"),
    modifications: Dict[str, Any] = Body(...),
    participant_id: int = Body(..., description="ID of participant requesting modification"),
    order_manager: OrderManager = Depends(get_order_manager)
):
    """
    Modify an active order.
    
    This endpoint allows participants to modify their active orders.
    """
    try:
        logger.info(f"Modifying order {order_id} for participant {participant_id}")
        
        # Modify order
        modified_order, success = await order_manager.modify_order(
            order_id, participant_id, modifications
        )
        
        if not success:
            return {
                "success": False,
                "message": "Order modification failed",
                "order_id": order_id
            }
        
        return {
            "success": True,
            "message": "Order modified successfully",
            "order": {
                "order_id": modified_order.order_id,
                "participant_id": modified_order.participant_id,
                "bond_id": modified_order.bond_id,
                "order_type": modified_order.order_type.value,
                "side": modified_order.side.value,
                "quantity": float(modified_order.quantity),
                "price": float(modified_order.price) if modified_order.price else None,
                "status": modified_order.status.value,
                "updated_at": modified_order.updated_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error modifying order {order_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/orders/{order_id}", response_model=Dict[str, Any])
async def cancel_order(
    order_id: str = Path(..., description="ID of the order to cancel"),
    participant_id: int = Query(..., description="ID of participant requesting cancellation"),
    order_manager: OrderManager = Depends(get_order_manager)
):
    """
    Cancel an active order.
    
    This endpoint allows participants to cancel their active orders.
    """
    try:
        logger.info(f"Cancelling order {order_id} for participant {participant_id}")
        
        # Cancel order
        success = await order_manager.cancel_order(order_id, participant_id)
        
        if not success:
            return {
                "success": False,
                "message": "Order cancellation failed",
                "order_id": order_id
            }
        
        return {
            "success": True,
            "message": "Order cancelled successfully",
            "order_id": order_id
        }
        
    except Exception as e:
        logger.error(f"Error cancelling order {order_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Order Book Endpoints

@router.get("/orderbook/{bond_id}", response_model=Dict[str, Any])
async def get_order_book(
    bond_id: str = Path(..., description="ID of the bond"),
    order_manager: OrderManager = Depends(get_order_manager)
):
    """
    Get current order book for a bond.
    
    This endpoint returns the complete order book with bid and ask entries.
    """
    try:
        # Get order book
        order_book = await order_manager.get_order_book(bond_id)
        
        if not order_book:
            return {
                "bond_id": bond_id,
                "bid_entries": [],
                "ask_entries": [],
                "last_updated": datetime.utcnow().isoformat(),
                "best_bid": None,
                "best_ask": None,
                "spread": None,
                "mid_price": None
            }
        
        # Format bid entries
        bid_entries = []
        for entry in order_book.bid_entries:
            bid_entry = {
                "price": float(entry.price),
                "total_quantity": float(entry.total_quantity),
                "order_count": entry.order_count
            }
            bid_entries.append(bid_entry)
        
        # Format ask entries
        ask_entries = []
        for entry in order_book.ask_entries:
            ask_entry = {
                "price": float(entry.price),
                "total_quantity": float(entry.total_quantity),
                "order_count": entry.order_count
            }
            ask_entries.append(ask_entry)
        
        # Get best bid and ask
        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()
        
        return {
            "bond_id": bond_id,
            "bid_entries": bid_entries,
            "ask_entries": ask_entries,
            "last_updated": order_book.last_updated.isoformat(),
            "best_bid": {
                "price": float(best_bid.price),
                "quantity": float(best_bid.total_quantity)
            } if best_bid else None,
            "best_ask": {
                "price": float(best_ask.price),
                "quantity": float(best_ask.total_quantity)
            } if best_ask else None,
            "spread": float(order_book.get_spread()) if order_book.get_spread() else None,
            "mid_price": float(order_book.get_mid_price()) if order_book.get_mid_price() else None
        }
        
    except Exception as e:
        logger.error(f"Error getting order book for bond {bond_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Trade Execution Endpoints

@router.post("/orders/{order_id}/execute", response_model=Dict[str, Any])
async def execute_order(
    order_id: str = Path(..., description="ID of the order to execute"),
    execution_engine: ExecutionEngine = Depends(get_execution_engine)
):
    """
    Execute a single order.
    
    This endpoint executes a specific order against the order book.
    """
    try:
        logger.info(f"Executing order {order_id}")
        
        # Get order from order manager
        active_orders = await order_manager.get_active_orders()
        order = None
        for active_order in active_orders:
            if active_order.order_id == order_id:
                order = active_order
                break
        
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        
        # Execute order
        execution_reports = await execution_engine.execute_single_order(order)
        
        if not execution_reports:
            return {
                "success": False,
                "message": "No executions for order",
                "order_id": order_id
            }
        
        # Format execution reports
        executions = []
        total_executed = Decimal('0')
        total_value = Decimal('0')
        
        for report in execution_reports:
            execution = {
                "execution_id": report.execution_id,
                "executed_quantity": float(report.executed_quantity),
                "executed_price": float(report.executed_price),
                "execution_time": report.execution_time.isoformat(),
                "execution_result": report.execution_result.value,
                "remaining_quantity": float(report.remaining_quantity),
                "fees": float(report.fees),
                "taxes": float(report.taxes)
            }
            executions.append(execution)
            
            total_executed += report.executed_quantity
            total_value += report.executed_quantity * report.executed_price
        
        return {
            "success": True,
            "message": "Order executed successfully",
            "order_id": order_id,
            "executions": executions,
            "summary": {
                "total_executed": float(total_executed),
                "total_value": float(total_value),
                "average_price": float(total_value / total_executed) if total_executed > 0 else 0,
                "execution_count": len(executions)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing order {order_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/execute-all", response_model=Dict[str, Any])
async def execute_all_orders(
    execution_engine: ExecutionEngine = Depends(get_execution_engine)
):
    """
    Execute all executable orders in the order book.
    
    This endpoint processes all executable orders and returns execution results.
    """
    try:
        logger.info("Executing all executable orders")
        
        # Execute all orders
        execution_reports = await execution_engine.execute_orders()
        
        if not execution_reports:
            return {
                "success": True,
                "message": "No executable orders found",
                "executions": [],
                "summary": {
                    "total_executions": 0,
                    "total_volume": 0,
                    "total_value": 0
                }
            }
        
        # Format execution reports
        executions = []
        total_volume = Decimal('0')
        total_value = Decimal('0')
        
        for report in execution_reports:
            execution = {
                "execution_id": report.execution_id,
                "order_id": report.order_id,
                "executed_quantity": float(report.executed_quantity),
                "executed_price": float(report.executed_price),
                "execution_time": report.execution_time.isoformat(),
                "execution_result": report.execution_result.value
            }
            executions.append(execution)
            
            total_volume += report.executed_quantity
            total_value += report.executed_quantity * report.executed_price
        
        return {
            "success": True,
            "message": "Orders executed successfully",
            "executions": executions,
            "summary": {
                "total_executions": len(executions),
                "total_volume": float(total_volume),
                "total_value": float(total_value)
            }
        }
        
    except Exception as e:
        logger.error(f"Error executing all orders: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Trade History Endpoints

@router.get("/trades", response_model=List[Dict[str, Any]])
async def get_trades(
    bond_id: Optional[str] = Query(None, description="Filter by bond ID"),
    participant_id: Optional[int] = Query(None, description="Filter by participant ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of trades to return"),
    offset: int = Query(0, ge=0, description="Number of trades to skip"),
    execution_engine: ExecutionEngine = Depends(get_execution_engine)
):
    """
    Get trade history with optional filtering.
    
    This endpoint returns a paginated list of trades with filtering options.
    """
    try:
        # Get trade history
        trades = await execution_engine.get_trade_history(bond_id)
        
        # Apply participant filter if specified
        if participant_id:
            trades = [trade for trade in trades if trade.participant_id == participant_id]
        
        # Apply pagination
        total_count = len(trades)
        paginated_trades = trades[offset:offset + limit]
        
        # Format response
        trade_list = []
        for trade in paginated_trades:
            trade_data = {
                "trade_id": trade.trade_id,
                "order_id": trade.order_id,
                "participant_id": trade.participant_id,
                "bond_id": trade.bond_id,
                "side": trade.side.value,
                "quantity": float(trade.quantity),
                "price": float(trade.price),
                "trade_value": float(trade.trade_value),
                "execution_time": trade.execution_time.isoformat(),
                "trade_type": trade.trade_type,
                "counterparty_id": trade.counterparty_id,
                "venue": trade.venue,
                "fees": float(trade.fees),
                "taxes": float(trade.taxes),
                "net_value": float(trade.net_value)
            }
            trade_list.append(trade_data)
        
        return {
            "trades": trade_list,
            "pagination": {
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting trades: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Market Data Endpoints

@router.get("/market-data/{bond_id}", response_model=Dict[str, Any])
async def get_market_data(
    bond_id: str = Path(..., description="ID of the bond"),
    execution_engine: ExecutionEngine = Depends(get_execution_engine)
):
    """
    Get real-time market data for a bond.
    
    This endpoint returns comprehensive market data including prices, volumes, and statistics.
    """
    try:
        # Get market data
        market_data = execution_engine.market_data.get(bond_id)
        
        if not market_data:
            return {
                "bond_id": bond_id,
                "message": "No market data available",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return {
            "bond_id": bond_id,
            "last_price": float(market_data.last_price) if market_data.last_price else None,
            "last_quantity": float(market_data.last_quantity) if market_data.last_quantity else None,
            "last_trade_time": market_data.last_trade_time.isoformat() if market_data.last_trade_time else None,
            "bid_price": float(market_data.bid_price) if market_data.bid_price else None,
            "ask_price": float(market_data.ask_price) if market_data.ask_price else None,
            "bid_quantity": float(market_data.bid_quantity) if market_data.bid_quantity else None,
            "ask_quantity": float(market_data.ask_quantity) if market_data.ask_quantity else None,
            "spread": float(market_data.spread) if market_data.spread else None,
            "mid_price": float(market_data.mid_price) if market_data.mid_price else None,
            "volume_24h": float(market_data.volume_24h),
            "high_24h": float(market_data.high_24h) if market_data.high_24h else None,
            "low_24h": float(market_data.low_24h) if market_data.low_24h else None,
            "change_24h": float(market_data.change_24h) if market_data.change_24h else None,
            "change_percent_24h": market_data.change_percent_24h,
            "yield_to_maturity": market_data.yield_to_maturity,
            "modified_duration": market_data.modified_duration,
            "convexity": market_data.convexity,
            "timestamp": market_data.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting market data for bond {bond_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# System Status Endpoints

@router.get("/status", response_model=Dict[str, Any])
async def get_trading_status(
    order_manager: OrderManager = Depends(get_order_manager),
    execution_engine: ExecutionEngine = Depends(get_execution_engine)
):
    """
    Get trading system status.
    
    This endpoint returns comprehensive system statistics and status information.
    """
    try:
        # Get system stats
        order_stats = order_manager.get_system_stats()
        execution_stats = execution_engine.get_system_stats()
        
        return {
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "order_manager": order_stats,
            "execution_engine": execution_stats,
            "system_health": {
                "order_books": order_stats["order_books_count"],
                "active_orders": order_stats["active_orders_count"],
                "total_trades": execution_stats["total_trades_executed"],
                "market_data_instruments": execution_stats["market_data_count"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting trading status: {str(e)}")
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


# WebSocket Endpoint for Real-time Updates

@router.websocket("/ws/{participant_id}")
async def trading_websocket(
    websocket: WebSocket,
    participant_id: int = Path(..., description="ID of the participant"),
    order_manager: OrderManager = Depends(get_order_manager),
    execution_engine: ExecutionEngine = Depends(get_execution_engine)
):
    """
    WebSocket endpoint for real-time trading updates.
    
    This endpoint provides real-time updates for orders, trades, and market data.
    """
    try:
        await websocket.accept()
        logger.info(f"WebSocket connection established for participant {participant_id}")
        
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "participant_id": participant_id,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        # Main WebSocket loop
        while True:
            try:
                # Send periodic updates
                await asyncio.sleep(1)
                
                # Get participant-specific updates
                participant_orders = await order_manager.get_active_orders(participant_id)
                
                # Send order updates
                order_updates = []
                for order in participant_orders:
                    order_update = {
                        "type": "order_update",
                        "order_id": order.order_id,
                        "status": order.status.value,
                        "filled_quantity": float(order.filled_quantity),
                        "remaining_quantity": float(order.remaining_quantity),
                        "average_fill_price": float(order.average_fill_price) if order.average_fill_price else None,
                        "timestamp": order.updated_at.isoformat()
                    }
                    order_updates.append(order_update)
                
                if order_updates:
                    await websocket.send_text(json.dumps({
                        "type": "order_updates",
                        "updates": order_updates,
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                
                # Send market data updates (simplified)
                market_updates = {
                    "type": "market_data_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "active_instruments": len(execution_engine.market_data)
                }
                
                await websocket.send_text(json.dumps(market_updates))
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for participant {participant_id}")
                break
            except Exception as e:
                logger.error(f"WebSocket error for participant {participant_id}: {str(e)}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Internal server error",
                    "timestamp": datetime.utcnow().isoformat()
                }))
                break
                
    except Exception as e:
        logger.error(f"Error establishing WebSocket connection: {str(e)}")
        if websocket.client_state.value != 3:  # Not disconnected
            await websocket.close(code=1011, reason="Internal server error")
