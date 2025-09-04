"""
Dashboard WebSocket endpoints for real-time updates.

This module provides WebSocket endpoints specifically for dashboard widgets,
delivering real-time updates for portfolio, market, risk, and trading data.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.orm import Session

from ...database.base import get_db
from ...core.logging import get_logger
from ...risk_management.portfolio_risk import PortfolioRiskManager
from ...trading_engine.order_manager import OrderManager
from ...trading_engine.execution_engine import ExecutionEngine
from .dashboard import (
    get_portfolio_summary_data,
    get_market_status_data,
    get_risk_metrics_data,
    get_trading_activity_data,
    get_system_health_data,
    get_platform_stats_data
)

logger = get_logger(__name__)

router = APIRouter(prefix="/ws/dashboard", tags=["dashboard-websocket"])

# Active connections tracking
active_connections: Dict[str, WebSocket] = {}
connection_subscriptions: Dict[str, List[str]] = {}

class ConnectionManager:
    """Manages WebSocket connections for dashboard updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, List[str]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = []
        logger.info(f"Dashboard WebSocket client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        logger.info(f"Dashboard WebSocket client {client_id} disconnected")
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to client {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict, subscription_type: str = None):
        """Broadcast a message to all subscribed clients."""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            # Check if client is subscribed to this type of update
            if subscription_type and subscription_type not in self.subscriptions.get(client_id, []):
                continue
            
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    def subscribe(self, client_id: str, subscription_types: List[str]):
        """Subscribe client to specific update types."""
        if client_id in self.subscriptions:
            for sub_type in subscription_types:
                if sub_type not in self.subscriptions[client_id]:
                    self.subscriptions[client_id].append(sub_type)
    
    def unsubscribe(self, client_id: str, subscription_types: List[str]):
        """Unsubscribe client from specific update types."""
        if client_id in self.subscriptions:
            for sub_type in subscription_types:
                if sub_type in self.subscriptions[client_id]:
                    self.subscriptions[client_id].remove(sub_type)

# Global connection manager
manager = ConnectionManager()

@router.websocket("/connect")
async def dashboard_websocket_endpoint(
    websocket: WebSocket,
    client_id: str = Query(..., description="Unique client identifier"),
    user_id: Optional[str] = Query(None, description="User ID for authentication"),
    portfolio_id: Optional[str] = Query(None, description="Portfolio ID for personalized updates")
):
    """
    Main dashboard WebSocket endpoint for real-time updates.
    
    Provides real-time updates for:
    - Portfolio summary
    - Market status
    - Risk metrics
    - Trading activity
    - System health
    """
    await manager.connect(websocket, client_id)
    
    try:
        # Send initial connection confirmation
        await manager.send_personal_message({
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Dashboard WebSocket connected successfully"
        }, client_id)
        
        # Handle incoming messages and subscriptions
        while True:
            try:
                # Wait for client messages (subscriptions, unsubscriptions, etc.)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                await handle_client_message(message, client_id, user_id, portfolio_id)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat()
                }, client_id)
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Internal server error",
                    "timestamp": datetime.utcnow().isoformat()
                }, client_id)
    
    except Exception as e:
        logger.error(f"Dashboard WebSocket error: {e}")
    finally:
        manager.disconnect(client_id)

async def handle_client_message(message: dict, client_id: str, user_id: Optional[str], portfolio_id: Optional[str]):
    """Handle incoming client messages."""
    message_type = message.get("type")
    
    if message_type == "subscribe":
        # Subscribe to specific update types
        subscription_types = message.get("subscriptions", [])
        manager.subscribe(client_id, subscription_types)
        
        await manager.send_personal_message({
            "type": "subscription_confirmed",
            "subscriptions": subscription_types,
            "timestamp": datetime.utcnow().isoformat()
        }, client_id)
        
        # Send initial data for subscribed types
        await send_initial_data(client_id, subscription_types, user_id, portfolio_id)
    
    elif message_type == "unsubscribe":
        # Unsubscribe from specific update types
        subscription_types = message.get("subscriptions", [])
        manager.unsubscribe(client_id, subscription_types)
        
        await manager.send_personal_message({
            "type": "unsubscription_confirmed",
            "subscriptions": subscription_types,
            "timestamp": datetime.utcnow().isoformat()
        }, client_id)
    
    elif message_type == "ping":
        # Respond to ping with pong
        await manager.send_personal_message({
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat()
        }, client_id)
    
    elif message_type == "get_snapshot":
        # Send current snapshot of specific data
        data_type = message.get("data_type")
        if data_type:
            await send_snapshot(client_id, data_type, user_id, portfolio_id)

async def send_initial_data(client_id: str, subscription_types: List[str], user_id: Optional[str], portfolio_id: Optional[str]):
    """Send initial data for newly subscribed types."""
    for sub_type in subscription_types:
        await send_snapshot(client_id, sub_type, user_id, portfolio_id)

async def send_snapshot(client_id: str, data_type: str, user_id: Optional[str], portfolio_id: Optional[str]):
    """Send snapshot of specific data type."""
    try:
        # Create mock database and engine instances for data fetching
        # In production, these would be proper dependency injection
        db = None  # Mock database session
        
        if data_type == "portfolio_summary":
            from ...risk_management.portfolio_risk import PortfolioRiskManager
            portfolio_risk_manager = PortfolioRiskManager(db) if db else None
            data = await get_portfolio_summary_data(portfolio_risk_manager, portfolio_id)
            
        elif data_type == "market_status":
            data = await get_market_status_data()
            
        elif data_type == "risk_metrics":
            from ...risk_management.portfolio_risk import PortfolioRiskManager
            portfolio_risk_manager = PortfolioRiskManager(db) if db else None
            data = await get_risk_metrics_data(portfolio_risk_manager, portfolio_id)
            
        elif data_type == "trading_activity":
            from ...trading_engine.execution_engine import ExecutionEngine
            from ...trading_engine.order_manager import OrderManager
            order_manager = OrderManager(db) if db else None
            execution_engine = ExecutionEngine(db, order_manager) if db else None
            data = await get_trading_activity_data(execution_engine)
            
        elif data_type == "system_health":
            from ...risk_management.portfolio_risk import PortfolioRiskManager
            from ...trading_engine.execution_engine import ExecutionEngine
            from ...trading_engine.order_manager import OrderManager
            portfolio_risk_manager = PortfolioRiskManager(db) if db else None
            order_manager = OrderManager(db) if db else None
            execution_engine = ExecutionEngine(db, order_manager) if db else None
            data = await get_system_health_data(portfolio_risk_manager, execution_engine)
            
        elif data_type == "platform_stats":
            data = await get_platform_stats_data()
            
        else:
            await manager.send_personal_message({
                "type": "error",
                "message": f"Unknown data type: {data_type}",
                "timestamp": datetime.utcnow().isoformat()
            }, client_id)
            return
        
        await manager.send_personal_message({
            "type": "data_update",
            "data_type": data_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }, client_id)
        
    except Exception as e:
        logger.error(f"Error sending snapshot for {data_type}: {e}")
        await manager.send_personal_message({
            "type": "error",
            "message": f"Failed to fetch {data_type}",
            "timestamp": datetime.utcnow().isoformat()
        }, client_id)

# Background task to send periodic updates
async def periodic_updates():
    """Send periodic updates to all connected clients."""
    while True:
        try:
            await asyncio.sleep(30)  # Update every 30 seconds
            
            current_time = datetime.utcnow()
            
            # Broadcast market status updates
            market_data = await get_market_status_data()
            await manager.broadcast({
                "type": "data_update",
                "data_type": "market_status", 
                "data": market_data,
                "timestamp": current_time.isoformat()
            }, "market_status")
            
            # Broadcast system health updates
            system_health = await get_system_health_data(None, None)
            await manager.broadcast({
                "type": "data_update",
                "data_type": "system_health",
                "data": system_health,
                "timestamp": current_time.isoformat()
            }, "system_health")
            
            # Broadcast platform stats updates
            platform_stats = await get_platform_stats_data()
            await manager.broadcast({
                "type": "data_update",
                "data_type": "platform_stats",
                "data": platform_stats,
                "timestamp": current_time.isoformat()
            }, "platform_stats")
            
        except Exception as e:
            logger.error(f"Error in periodic updates: {e}")
            await asyncio.sleep(30)  # Continue even if there's an error

# Start periodic updates task
@router.on_event("startup")
async def start_periodic_updates():
    """Start the periodic updates background task."""
    asyncio.create_task(periodic_updates())

# Trade stream endpoint for real-time trade updates
@router.websocket("/trades")
async def trades_websocket_endpoint(
    websocket: WebSocket,
    client_id: str = Query(..., description="Unique client identifier")
):
    """
    WebSocket endpoint for real-time trade updates.
    
    Provides streaming updates for:
    - New trades
    - Order executions
    - Market activity
    """
    await websocket.accept()
    
    try:
        logger.info(f"Trade stream WebSocket client {client_id} connected")
        
        # Send connection confirmation
        await websocket.send_text(json.dumps({
            "type": "trade_stream_connected",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        # Simulate real-time trade updates
        while True:
            await asyncio.sleep(5)  # Send updates every 5 seconds
            
            # Generate mock trade update
            mock_trade = {
                "trade_id": f"TRD_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}",
                "timestamp": datetime.utcnow().isoformat(),
                "bond_id": f"BOND_{(hash(datetime.utcnow()) % 100) + 1:03d}",
                "bond_name": f"Corporate Bond {(hash(datetime.utcnow()) % 10) + 1}",
                "side": "BUY" if datetime.utcnow().second % 2 == 0 else "SELL",
                "quantity": 100 + (datetime.utcnow().second * 10),
                "price": round(98.5 + (datetime.utcnow().second * 0.01), 2),
                "yield": round(7.2 + (datetime.utcnow().second * 0.01), 2),
                "venue": "NSE" if datetime.utcnow().second % 2 == 0 else "BSE",
                "counterparty": f"CP_{(datetime.utcnow().second % 5) + 1}"
            }
            mock_trade["trade_value"] = mock_trade["quantity"] * mock_trade["price"]
            
            await websocket.send_text(json.dumps({
                "type": "new_trade",
                "data": mock_trade,
                "timestamp": datetime.utcnow().isoformat()
            }))
    
    except WebSocketDisconnect:
        logger.info(f"Trade stream WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Trade stream WebSocket error: {e}")

# Risk alerts endpoint for real-time risk monitoring
@router.websocket("/risk-alerts")
async def risk_alerts_websocket_endpoint(
    websocket: WebSocket,
    client_id: str = Query(..., description="Unique client identifier"),
    portfolio_id: Optional[str] = Query(None, description="Portfolio ID for risk monitoring")
):
    """
    WebSocket endpoint for real-time risk alerts.
    
    Provides streaming updates for:
    - Risk limit breaches
    - VaR threshold alerts
    - Concentration risk warnings
    - Liquidity alerts
    """
    await websocket.accept()
    
    try:
        logger.info(f"Risk alerts WebSocket client {client_id} connected")
        
        # Send connection confirmation
        await websocket.send_text(json.dumps({
            "type": "risk_alerts_connected",
            "client_id": client_id,
            "portfolio_id": portfolio_id,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        # Simulate risk alerts (in production, these would be triggered by actual risk calculations)
        while True:
            await asyncio.sleep(60)  # Check every minute
            
            # Randomly generate risk alerts for demonstration
            import random
            if random.random() < 0.1:  # 10% chance of alert each minute
                alert_types = ["VAR_BREACH", "CONCENTRATION_WARNING", "LIQUIDITY_ALERT", "DURATION_DRIFT"]
                alert_type = random.choice(alert_types)
                
                risk_alert = {
                    "alert_id": f"ALERT_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    "timestamp": datetime.utcnow().isoformat(),
                    "portfolio_id": portfolio_id or "PORT_001",
                    "alert_type": alert_type,
                    "severity": random.choice(["LOW", "MEDIUM", "HIGH"]),
                    "message": f"{alert_type.replace('_', ' ').title()} detected",
                    "current_value": random.uniform(0.1, 10.0),
                    "threshold": random.uniform(5.0, 15.0),
                    "recommendation": "Review portfolio allocation and consider rebalancing"
                }
                
                await websocket.send_text(json.dumps({
                    "type": "risk_alert",
                    "data": risk_alert,
                    "timestamp": datetime.utcnow().isoformat()
                }))
    
    except WebSocketDisconnect:
        logger.info(f"Risk alerts WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Risk alerts WebSocket error: {e}")

# Export router
__all__ = ["router"]
