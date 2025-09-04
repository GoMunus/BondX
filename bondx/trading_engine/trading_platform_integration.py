"""
Trading Platform Integration for Phase E

This module implements real-time integration of BondX analytics, risk, and ML pipelines
with trading systems including:
- Trading API Gateway (REST + WebSocket)
- Algorithmic Trading Engine Integration
- EMS Connectors (smart order routing, trade slicing)
- Real-Time Monitoring & Alerts
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import time
import uuid
from enum import Enum
import threading

# FastAPI imports
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order statuses"""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"

@dataclass
class Order:
    """Trading order"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0

@dataclass
class Trade:
    """Trade execution"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    exchange: str

@dataclass
class RiskCheck:
    """Pre-trade risk check result"""
    order_id: str
    passed: bool
    risk_score: float
    var_impact: float
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

class TradingPlatformIntegration:
    """Trading Platform Integration System"""
    
    def __init__(self):
        self.app = FastAPI(title="BondX Trading Platform Integration", version="2.0.0")
        self.setup_middleware()
        self.setup_routes()
        
        # Trading state
        self.orders: Dict[str, Order] = {}
        self.trades: Dict[str, Trade] = {}
        self.risk_checks: Dict[str, RiskCheck] = {}
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        self.connection_lock = threading.Lock()
        
        # Performance monitoring
        self.latency_metrics = {
            'order_processing': [],
            'risk_validation': []
        }
        
        logger.info("Trading Platform Integration initialized")
    
    def setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup API routes"""
        self.app.post("/api/v1/orders")(self.create_order)
        self.app.get("/api/v1/orders")(self.get_orders)
        self.app.get("/api/v1/orders/{order_id}")(self.get_order)
        self.app.delete("/api/v1/orders/{order_id}")(self.cancel_order)
        self.app.websocket("/ws/trading")(self.websocket_endpoint)
        self.app.get("/health")(self.health_check)
    
    async def create_order(self, order_data: dict):
        """Create a new trading order"""
        start_time = time.perf_counter()
        
        try:
            # Validate order data
            order = self._validate_order_data(order_data)
            
            # Pre-trade risk check
            risk_check = await self._perform_risk_check(order)
            if not risk_check.passed:
                raise HTTPException(
                    status_code=400,
                    detail=f"Risk check failed: {risk_check.errors}"
                )
            
            # Store order
            self.orders[order.order_id] = order
            self.risk_checks[order.order_id] = risk_check
            
            # Broadcast order update
            await self._broadcast_order_update(order)
            
            # Record latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_metrics['order_processing'].append(latency_ms)
            
            return {
                "order_id": order.order_id,
                "status": order.status.value,
                "risk_check": {
                    "passed": risk_check.passed,
                    "risk_score": risk_check.risk_score,
                    "var_impact": risk_check.var_impact
                },
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            logger.error(f"Order creation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _validate_order_data(self, order_data: dict) -> Order:
        """Validate and create Order object from order data"""
        required_fields = ['symbol', 'side', 'order_type', 'quantity']
        for field in required_fields:
            if field not in order_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Generate order ID
        order_id = str(uuid.uuid4())
        
        # Create Order object
        order = Order(
            order_id=order_id,
            symbol=order_data['symbol'],
            side=OrderSide(order_data['side']),
            order_type=OrderType(order_data['order_type']),
            quantity=float(order_data['quantity']),
            price=float(order_data.get('price', 0)) if order_data.get('price') else None
        )
        
        return order
    
    async def _perform_risk_check(self, order: Order) -> RiskCheck:
        """Perform pre-trade risk check"""
        start_time = time.perf_counter()
        
        try:
            # Calculate risk metrics
            risk_score = self._calculate_risk_score(order)
            var_impact = self._calculate_var_impact(order)
            
            # Check risk limits
            warnings = []
            errors = []
            
            if risk_score > 0.8:
                errors.append("Risk score exceeds maximum threshold")
            
            if var_impact > 1000000:  # $1M VaR impact
                errors.append("VaR impact exceeds maximum threshold")
            
            # Determine if check passed
            passed = len(errors) == 0
            
            # Record latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_metrics['risk_validation'].append(latency_ms)
            
            return RiskCheck(
                order_id=order.order_id,
                passed=passed,
                risk_score=risk_score,
                var_impact=var_impact,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Risk check failed: {e}")
            return RiskCheck(
                order_id=order.order_id,
                passed=False,
                risk_score=1.0,
                var_impact=float('inf'),
                errors=[str(e)]
            )
    
    def _calculate_risk_score(self, order: Order) -> float:
        """Calculate risk score for order"""
        base_score = 0.1
        
        # Volume-based risk
        if order.quantity > 1000000:
            base_score += 0.3
        
        # Price-based risk
        if order.price and order.price > 1000:
            base_score += 0.2
        
        return min(base_score, 1.0)
    
    def _calculate_var_impact(self, order: Order) -> float:
        """Calculate VaR impact of order"""
        base_var = order.quantity * 0.02  # 2% VaR assumption
        
        if order.order_type == OrderType.MARKET:
            base_var *= 1.5
        
        return base_var
    
    async def _broadcast_order_update(self, order: Order):
        """Broadcast order update to WebSocket clients"""
        if not self.active_connections:
            return
        
        message = {
            "type": "order_update",
            "data": {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "status": order.status.value,
                "quantity": order.quantity,
                "filled_quantity": order.filled_quantity,
                "timestamp": order.created_at.isoformat()
            }
        }
        
        # Send to all connected clients
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.active_connections.remove(connection)
    
    async def get_orders(self, status: Optional[str] = None, symbol: Optional[str] = None):
        """Get orders with optional filtering"""
        try:
            orders = list(self.orders.values())
            
            # Apply filters
            if status:
                orders = [o for o in orders if o.status.value == status]
            
            if symbol:
                orders = [o for o in orders if o.symbol == symbol]
            
            # Convert to dict format
            result = []
            for order in orders:
                result.append({
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "order_type": order.order_type.value,
                    "quantity": order.quantity,
                    "price": order.price,
                    "status": order.status.value,
                    "filled_quantity": order.filled_quantity,
                    "created_at": order.created_at.isoformat()
                })
            
            return {"orders": result, "count": len(result)}
            
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_order(self, order_id: str):
        """Get specific order by ID"""
        try:
            if order_id not in self.orders:
                raise HTTPException(status_code=404, detail="Order not found")
            
            order = self.orders[order_id]
            return {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "order_type": order.order_type.value,
                "quantity": order.quantity,
                "price": order.price,
                "status": order.status.value,
                "filled_quantity": order.filled_quantity,
                "created_at": order.created_at.isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def cancel_order(self, order_id: str):
        """Cancel existing order"""
        try:
            if order_id not in self.orders:
                raise HTTPException(status_code=404, detail="Order not found")
            
            order = self.orders[order_id]
            
            # Check if order can be cancelled
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                raise HTTPException(status_code=400, detail="Order cannot be cancelled")
            
            # Update status
            order.status = OrderStatus.CANCELLED
            
            # Broadcast update
            await self._broadcast_order_update(order)
            
            return {"message": "Order cancelled successfully", "order_id": order_id}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def websocket_endpoint(self, websocket: WebSocket):
        """WebSocket endpoint for real-time updates"""
        await websocket.accept()
        
        try:
            # Add to active connections
            with self.connection_lock:
                self.active_connections.append(websocket)
            
            logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")
            
            # Keep connection alive
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    # Handle client message if needed
                    
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    await websocket.send_text(json.dumps({"type": "ping", "timestamp": datetime.utcnow().isoformat()}))
                
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            # Remove from active connections
            with self.connection_lock:
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
            
            logger.info(f"WebSocket client removed. Total connections: {len(self.active_connections)}")
    
    async def health_check(self):
        """Health check endpoint"""
        try:
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "websocket_connections": len(self.active_connections),
                "total_orders": len(self.orders),
                "total_trades": len(self.trades)
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the trading platform integration server"""
        uvicorn.run(self.app, host=host, port=port)
