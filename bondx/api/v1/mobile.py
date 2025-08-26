"""
Mobile-Optimized API Endpoints for BondX.

This module provides mobile-specific API endpoints with:
- Slim payloads and field selection
- Delta-updates and cursor-based pagination
- Push notification integration
- Offline sync capabilities
- Mobile security features
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
from fastapi import APIRouter, HTTPException, Depends, Query, Header, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import jwt
from sqlalchemy.orm import Session

from ...core.logging import get_logger
from ...database.base import get_db
from ...trading_engine.trading_models import Order, OrderSide, OrderType, OrderStatus
from ...websocket.websocket_manager import WebSocketManager
from ...risk_management.real_time_risk import RealTimeRiskEngine
from ...core.auth import get_current_user, User

logger = get_logger(__name__)

router = APIRouter(prefix="/mobile", tags=["Mobile API"])


# Mobile-specific models
class MobileOrderRequest(BaseModel):
    """Mobile-optimized order request."""
    bond_id: str = Field(..., description="Bond identifier")
    side: OrderSide = Field(..., description="Order side (BUY/SELL)")
    quantity: float = Field(..., description="Order quantity")
    price: Optional[float] = Field(None, description="Order price (for limit orders)")
    order_type: OrderType = Field(OrderType.LIMIT, description="Order type")
    time_in_force: str = Field("DAY", description="Time in force")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class MobileOrderResponse(BaseModel):
    """Mobile-optimized order response."""
    order_id: str
    status: str
    side: str
    quantity: float
    price: Optional[float]
    filled_quantity: float
    remaining_quantity: float
    average_price: Optional[float]
    created_at: str
    updated_at: str
    estimated_cost: Optional[float]
    estimated_fees: Optional[float]


class MobilePortfolioSummary(BaseModel):
    """Mobile-optimized portfolio summary."""
    total_value: float
    total_pnl: float
    daily_pnl: float
    position_count: int
    risk_metrics: Dict[str, Any]
    last_updated: str


class MobilePosition(BaseModel):
    """Mobile-optimized position."""
    instrument_id: str
    instrument_name: str
    quantity: float
    market_value: float
    unrealized_pnl: float
    duration: Optional[float]
    yield_to_maturity: Optional[float]
    last_price: float
    last_updated: str


class MobileMarketData(BaseModel):
    """Mobile-optimized market data."""
    instrument_id: str
    best_bid: Optional[float]
    best_ask: Optional[float]
    last_price: Optional[float]
    volume: Optional[float]
    change: Optional[float]
    change_percent: Optional[float]
    spread: Optional[float]
    last_updated: str


class MobileNotification(BaseModel):
    """Mobile notification."""
    notification_id: str
    type: str
    title: str
    message: str
    data: Optional[Dict[str, Any]]
    timestamp: str
    is_read: bool


class MobileSyncRequest(BaseModel):
    """Mobile sync request."""
    last_sync_timestamp: str
    sync_type: str  # "portfolio", "orders", "market_data"
    filters: Optional[Dict[str, Any]]


class MobileSyncResponse(BaseModel):
    """Mobile sync response."""
    sync_timestamp: str
    changes: List[Dict[str, Any]]
    deleted_items: List[str]
    has_more: bool
    next_cursor: Optional[str]


# Mobile API endpoints
@router.post("/orders", response_model=MobileOrderResponse)
async def create_mobile_order(
    order_request: MobileOrderRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new order via mobile API."""
    try:
        # Validate order
        if order_request.quantity <= 0:
            raise HTTPException(status_code=400, detail="Quantity must be positive")
        
        if order_request.order_type == OrderType.LIMIT and not order_request.price:
            raise HTTPException(status_code=400, detail="Price required for limit orders")
        
        # Create order (simplified - in practice would use trading engine)
        order = Order(
            order_id=f"MOB_{datetime.utcnow().timestamp()}",
            participant_id=current_user.id,
            bond_id=order_request.bond_id,
            order_type=order_request.order_type,
            side=order_request.side,
            quantity=order_request.quantity,
            price=order_request.price,
            time_in_force=order_request.time_in_force,
            status=OrderStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Calculate estimated costs
        estimated_cost = order_request.quantity * (order_request.price or 100.0)
        estimated_fees = estimated_cost * 0.0005  # 0.05% fee
        
        # Schedule push notification
        background_tasks.add_task(
            send_push_notification,
            current_user.id,
            "order_created",
            f"Order {order.order_id} created successfully",
            {"order_id": order.order_id, "status": "PENDING"}
        )
        
        return MobileOrderResponse(
            order_id=order.order_id,
            status=order.status.value,
            side=order.side.value,
            quantity=float(order.quantity),
            price=float(order.price) if order.price else None,
            filled_quantity=0.0,
            remaining_quantity=float(order.quantity),
            average_price=None,
            created_at=order.created_at.isoformat(),
            updated_at=order.updated_at.isoformat(),
            estimated_cost=estimated_cost,
            estimated_fees=estimated_fees
        )
        
    except Exception as e:
        logger.error(f"Error creating mobile order: {e}")
        raise HTTPException(status_code=500, detail="Failed to create order")


@router.get("/orders", response_model=List[MobileOrderResponse])
async def get_mobile_orders(
    status: Optional[str] = Query(None, description="Filter by order status"),
    side: Optional[str] = Query(None, description="Filter by order side"),
    limit: int = Query(50, le=100, description="Maximum number of orders"),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user orders with mobile optimization."""
    try:
        # In practice, this would query the database
        # For now, return mock data
        mock_orders = [
            {
                "order_id": f"MOB_{i}",
                "status": "ACTIVE",
                "side": "BUY",
                "quantity": 1000.0,
                "price": 100.5,
                "filled_quantity": 0.0,
                "remaining_quantity": 1000.0,
                "average_price": None,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "estimated_cost": 100500.0,
                "estimated_fees": 50.25
            }
            for i in range(min(limit, 10))
        ]
        
        return mock_orders
        
    except Exception as e:
        logger.error(f"Error fetching mobile orders: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch orders")


@router.get("/orders/{order_id}", response_model=MobileOrderResponse)
async def get_mobile_order(
    order_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get specific order details."""
    try:
        # In practice, this would query the database
        # For now, return mock data
        return MobileOrderResponse(
            order_id=order_id,
            status="ACTIVE",
            side="BUY",
            quantity=1000.0,
            price=100.5,
            filled_quantity=0.0,
            remaining_quantity=1000.0,
            average_price=None,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            estimated_cost=100500.0,
            estimated_fees=50.25
        )
        
    except Exception as e:
        logger.error(f"Error fetching mobile order {order_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch order")


@router.delete("/orders/{order_id}")
async def cancel_mobile_order(
    order_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Cancel an order via mobile API."""
    try:
        # In practice, this would cancel the order in the trading engine
        # For now, just return success
        
        # Schedule push notification
        background_tasks.add_task(
            send_push_notification,
            current_user.id,
            "order_cancelled",
            f"Order {order_id} cancelled successfully",
            {"order_id": order_id, "status": "CANCELLED"}
        )
        
        return {"message": "Order cancelled successfully", "order_id": order_id}
        
    except Exception as e:
        logger.error(f"Error cancelling mobile order {order_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel order")


@router.get("/portfolio/summary", response_model=MobilePortfolioSummary)
async def get_mobile_portfolio_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get mobile-optimized portfolio summary."""
    try:
        # In practice, this would calculate from real portfolio data
        # For now, return mock data
        return MobilePortfolioSummary(
            total_value=1000000.0,
            total_pnl=50000.0,
            daily_pnl=2500.0,
            position_count=15,
            risk_metrics={
                "var_95_1d": 15000.0,
                "duration": 4.5,
                "convexity": 25.0
            },
            last_updated=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error fetching mobile portfolio summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch portfolio summary")


@router.get("/portfolio/positions", response_model=List[MobilePosition])
async def get_mobile_positions(
    limit: int = Query(50, le=100, description="Maximum number of positions"),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get mobile-optimized portfolio positions."""
    try:
        # In practice, this would query the database
        # For now, return mock data
        mock_positions = [
            {
                "instrument_id": f"BOND_{i}",
                "instrument_name": f"Government Bond {i}",
                "quantity": 10000.0,
                "market_value": 100000.0,
                "unrealized_pnl": 5000.0,
                "duration": 5.0 + i * 0.5,
                "yield_to_maturity": 6.5 + i * 0.1,
                "last_price": 100.0 + i * 0.5,
                "last_updated": datetime.utcnow().isoformat()
            }
            for i in range(min(limit, 10))
        ]
        
        return mock_positions
        
    except Exception as e:
        logger.error(f"Error fetching mobile positions: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch positions")


@router.get("/market-data/{instrument_id}", response_model=MobileMarketData)
async def get_mobile_market_data(
    instrument_id: str,
    fields: Optional[str] = Query(None, description="Comma-separated fields to include"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get mobile-optimized market data for an instrument."""
    try:
        # In practice, this would fetch from real-time market data
        # For now, return mock data
        market_data = MobileMarketData(
            instrument_id=instrument_id,
            best_bid=99.5,
            best_ask=100.5,
            last_price=100.0,
            volume=1000000.0,
            change=0.5,
            change_percent=0.5,
            spread=1.0,
            last_updated=datetime.utcnow().isoformat()
        )
        
        # Apply field selection if specified
        if fields:
            selected_fields = [f.strip() for f in fields.split(",")]
            filtered_data = {field: getattr(market_data, field) for field in selected_fields if hasattr(market_data, field)}
            return filtered_data
        
        return market_data
        
    except Exception as e:
        logger.error(f"Error fetching mobile market data for {instrument_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch market data")


@router.get("/market-data", response_model=List[MobileMarketData])
async def get_mobile_market_data_batch(
    instrument_ids: str = Query(..., description="Comma-separated instrument IDs"),
    fields: Optional[str] = Query(None, description="Comma-separated fields to include"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get mobile-optimized market data for multiple instruments."""
    try:
        ids = [id.strip() for id in instrument_ids.split(",")]
        
        # In practice, this would fetch from real-time market data
        # For now, return mock data
        market_data_list = []
        for instrument_id in ids[:20]:  # Limit to 20 instruments
            market_data = MobileMarketData(
                instrument_id=instrument_id,
                best_bid=99.5,
                best_ask=100.5,
                last_price=100.0,
                volume=1000000.0,
                change=0.5,
                change_percent=0.5,
                spread=1.0,
                last_updated=datetime.utcnow().isoformat()
            )
            market_data_list.append(market_data)
        
        # Apply field selection if specified
        if fields:
            selected_fields = [f.strip() for f in fields.split(",")]
            filtered_list = []
            for data in market_data_list:
                filtered_data = {field: getattr(data, field) for field in selected_fields if hasattr(data, field)}
                filtered_list.append(filtered_data)
            return filtered_list
        
        return market_data_list
        
    except Exception as e:
        logger.error(f"Error fetching mobile market data batch: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch market data")


@router.get("/notifications", response_model=List[MobileNotification])
async def get_mobile_notifications(
    unread_only: bool = Query(False, description="Return only unread notifications"),
    limit: int = Query(50, le=100, description="Maximum number of notifications"),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get mobile notifications for the user."""
    try:
        # In practice, this would query the notifications database
        # For now, return mock data
        mock_notifications = [
            {
                "notification_id": f"NOTIF_{i}",
                "type": "order_update",
                "title": "Order Update",
                "message": f"Order {i} has been updated",
                "data": {"order_id": f"ORDER_{i}"},
                "timestamp": datetime.utcnow().isoformat(),
                "is_read": i % 2 == 0
            }
            for i in range(min(limit, 10))
        ]
        
        if unread_only:
            mock_notifications = [n for n in mock_notifications if not n["is_read"]]
        
        return mock_notifications
        
    except Exception as e:
        logger.error(f"Error fetching mobile notifications: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch notifications")


@router.post("/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark a notification as read."""
    try:
        # In practice, this would update the notification in the database
        # For now, just return success
        return {"message": "Notification marked as read", "notification_id": notification_id}
        
    except Exception as e:
        logger.error(f"Error marking notification {notification_id} as read: {e}")
        raise HTTPException(status_code=500, detail="Failed to mark notification as read")


@router.post("/sync", response_model=MobileSyncResponse)
async def mobile_sync(
    sync_request: MobileSyncRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mobile sync endpoint for offline capabilities."""
    try:
        last_sync = datetime.fromisoformat(sync_request.last_sync_timestamp)
        current_time = datetime.utcnow()
        
        # In practice, this would query for changes since last sync
        # For now, return mock sync data
        changes = []
        deleted_items = []
        
        if sync_request.sync_type == "portfolio":
            # Mock portfolio changes
            changes = [
                {
                    "type": "position_update",
                    "instrument_id": "BOND_001",
                    "data": {
                        "quantity": 10000.0,
                        "market_value": 100000.0,
                        "unrealized_pnl": 5000.0
                    }
                }
            ]
        elif sync_request.sync_type == "orders":
            # Mock order changes
            changes = [
                {
                    "type": "order_update",
                    "order_id": "ORDER_001",
                    "data": {
                        "status": "FILLED",
                        "filled_quantity": 1000.0
                    }
                }
            ]
        
        return MobileSyncResponse(
            sync_timestamp=current_time.isoformat(),
            changes=changes,
            deleted_items=deleted_items,
            has_more=False,
            next_cursor=None
        )
        
    except Exception as e:
        logger.error(f"Error in mobile sync: {e}")
        raise HTTPException(status_code=500, detail="Sync failed")


@router.post("/push-token")
async def register_push_token(
    token: str = Query(..., description="Push notification token"),
    platform: str = Query(..., description="Platform (ios/android)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Register push notification token for mobile device."""
    try:
        # In practice, this would store the token in the database
        # For now, just return success
        return {
            "message": "Push token registered successfully",
            "user_id": current_user.id,
            "platform": platform
        }
        
    except Exception as e:
        logger.error(f"Error registering push token: {e}")
        raise HTTPException(status_code=500, detail="Failed to register push token")


@router.delete("/push-token")
async def unregister_push_token(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Unregister push notification token."""
    try:
        # In practice, this would remove the token from the database
        # For now, just return success
        return {"message": "Push token unregistered successfully"}
        
    except Exception as e:
        logger.error(f"Error unregistering push token: {e}")
        raise HTTPException(status_code=500, detail="Failed to unregister push token")


@router.get("/health")
async def mobile_health_check(
    current_user: User = Depends(get_current_user)
):
    """Mobile-specific health check endpoint."""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": current_user.id,
            "mobile_api_version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Error in mobile health check: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


# Background task functions
async def send_push_notification(user_id: int, notification_type: str, message: str, data: Dict[str, Any]):
    """Send push notification to user (background task)."""
    try:
        # In practice, this would send via FCM/APNs
        logger.info(f"Push notification sent to user {user_id}: {message}")
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
    except Exception as e:
        logger.error(f"Error sending push notification to user {user_id}: {e}")


# Mobile-specific middleware and utilities
class MobileResponseMiddleware:
    """Middleware for mobile-optimized responses."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Add mobile-specific headers
            async def mobile_send(message):
                if message["type"] == "http.response.start":
                    # Add mobile optimization headers
                    headers = message.get("headers", [])
                    headers.extend([
                        (b"X-Mobile-Optimized", b"true"),
                        (b"X-Response-Compression", b"gzip"),
                        (b"X-Cache-Control", b"max-age=300")
                    ])
                    message["headers"] = headers
                
                await send(message)
            
            await self.app(scope, receive, mobile_send)
        else:
            await self.app(scope, receive, send)


# Mobile API configuration
MOBILE_API_CONFIG = {
    "rate_limit": {
        "requests_per_minute": 100,
        "burst_size": 20
    },
    "compression": {
        "enabled": True,
        "min_size": 1024
    },
    "caching": {
        "enabled": True,
        "ttl_seconds": 300
    },
    "security": {
        "require_https": True,
        "max_token_age": 3600,
        "rate_limit_by_ip": True
    }
}


# Export router and configuration
__all__ = ["router", "MobileResponseMiddleware", "MOBILE_API_CONFIG"]
