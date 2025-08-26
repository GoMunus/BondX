"""
WebSocket Infrastructure for BondX Real-time Streaming.

This module provides:
- Real-time market data streaming
- Trading updates and order status
- Risk alerts and portfolio updates
- Load balancing and message queuing
- Authentication and authorization
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import websockets
from websockets.server import WebSocketServerProtocol
import redis
from collections import defaultdict, deque

from ..core.logging import get_logger
from ..trading_engine.order_book import OrderBookUpdate
from ..trading_engine.matching_engine import MatchingEvent
from ..risk_management.real_time_risk import RiskLimitBreach

logger = get_logger(__name__)


class WebSocketMessageType(Enum):
    """Types of WebSocket messages."""
    MARKET_DATA = "MARKET_DATA"
    ORDER_UPDATE = "ORDER_UPDATE"
    TRADE_EXECUTION = "TRADE_EXECUTION"
    RISK_ALERT = "RISK_ALERT"
    PORTFOLIO_UPDATE = "PORTFOLIO_UPDATE"
    AUCTION_UPDATE = "AUCTION_UPDATE"
    HEARTBEAT = "HEARTBEAT"
    ERROR = "ERROR"


class SubscriptionType(Enum):
    """Types of subscriptions."""
    INSTRUMENT = "INSTRUMENT"
    PORTFOLIO = "PORTFOLIO"
    RISK = "RISK"
    AUCTION = "AUCTION"
    SYSTEM = "SYSTEM"


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    message_type: WebSocketMessageType
    topic: str
    data: Any
    timestamp: datetime
    sequence_number: int
    correlation_id: Optional[str] = None


@dataclass
class Subscription:
    """Client subscription information."""
    client_id: str
    subscription_type: SubscriptionType
    topic: str
    filters: Dict[str, Any]
    created_at: datetime
    last_activity: datetime


@dataclass
class ClientConnection:
    """Client connection information."""
    client_id: str
    websocket: WebSocketServerProtocol
    subscriptions: Set[str]
    user_id: Optional[int]
    permissions: List[str]
    connected_at: datetime
    last_heartbeat: datetime
    is_authenticated: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class WebSocketManager:
    """
    WebSocket manager for real-time streaming.
    
    Features:
    - Multiple topic subscriptions
    - Authentication and authorization
    - Load balancing and message queuing
    - Heartbeat and connection management
    - Rate limiting and backpressure
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize the WebSocket manager."""
        self.redis_url = redis_url
        self.redis_client = None
        
        # Connection management
        self.clients: Dict[str, ClientConnection] = {}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)  # topic -> client_ids
        
        # Message queuing
        self.message_queue: deque = deque(maxlen=10000)
        self.sequence_counter = 0
        
        # Configuration
        self.max_clients = 10000
        self.max_subscriptions_per_client = 50
        self.heartbeat_interval = 30  # seconds
        self.connection_timeout = 300  # seconds
        
        # Performance tracking
        self.messages_sent = 0
        self.connections_established = 0
        self.connections_closed = 0
        self.start_time = datetime.utcnow()
        
        # Task management
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.message_processor_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info("WebSocket Manager initialized")
    
    async def start(self) -> None:
        """Start the WebSocket manager."""
        if self.is_running:
            logger.warning("WebSocket manager is already running")
            return
        
        try:
            # Connect to Redis
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis")
            
            # Start background tasks
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self.message_processor_task = asyncio.create_task(self._message_processor_loop())
            
            self.is_running = True
            logger.info("WebSocket Manager started successfully")
            
        except Exception as e:
            logger.error(f"Error starting WebSocket manager: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the WebSocket manager."""
        if not self.is_running:
            logger.warning("WebSocket manager is not running")
            return
        
        self.is_running = False
        
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        if self.message_processor_task:
            self.message_processor_task.cancel()
            try:
                await self.message_processor_task
            except asyncio.CancelledError:
                pass
        
        # Close all client connections
        await self._close_all_connections()
        
        # Close Redis connection
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("WebSocket Manager stopped")
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """Handle a new WebSocket connection."""
        try:
            client_id = str(id(websocket))
            
            # Create client connection
            client = ClientConnection(
                client_id=client_id,
                websocket=websocket,
                subscriptions=set(),
                user_id=None,
                permissions=[],
                connected_at=datetime.utcnow(),
                last_heartbeat=datetime.utcnow(),
                is_authenticated=False
            )
            
            # Check connection limits
            if len(self.clients) >= self.max_clients:
                await self._send_error(websocket, "Maximum connections reached")
                await websocket.close()
                return
            
            # Store client connection
            self.clients[client_id] = client
            self.connections_established += 1
            
            logger.info(f"New WebSocket connection: {client_id}")
            
            # Send welcome message
            await self._send_message(websocket, WebSocketMessageType.SYSTEM, "welcome", {
                "message": "Connected to BondX WebSocket",
                "client_id": client_id,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Handle client messages
            async for message in websocket:
                await self._handle_client_message(client, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {client_id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket connection: {e}")
        finally:
            await self._cleanup_connection(client_id)
    
    async def _handle_client_message(self, client: ClientConnection, message: str) -> None:
        """Handle incoming client message."""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "authenticate":
                await self._handle_authentication(client, data)
            elif message_type == "subscribe":
                await self._handle_subscription(client, data)
            elif message_type == "unsubscribe":
                await self._handle_unsubscription(client, data)
            elif message_type == "heartbeat":
                await self._handle_heartbeat(client)
            else:
                await self._send_error(client.websocket, f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            await self._send_error(client.websocket, "Invalid JSON format")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
            await self._send_error(client.websocket, "Internal server error")
    
    async def _handle_authentication(self, client: ClientConnection, data: Dict[str, Any]) -> None:
        """Handle client authentication."""
        try:
            # In practice, this would validate JWT tokens or other auth mechanisms
            token = data.get("token")
            if not token:
                await self._send_error(client.websocket, "Authentication token required")
                return
            
            # Mock authentication (replace with real implementation)
            if token == "valid_token":
                client.is_authenticated = True
                client.user_id = 123  # Mock user ID
                client.permissions = ["read", "write"]  # Mock permissions
                
                await self._send_message(client.websocket, WebSocketMessageType.SYSTEM, "auth_success", {
                    "message": "Authentication successful",
                    "user_id": client.user_id,
                    "permissions": client.permissions
                })
                
                logger.info(f"Client {client.client_id} authenticated successfully")
            else:
                await self._send_error(client.websocket, "Invalid authentication token")
                
        except Exception as e:
            logger.error(f"Error in authentication: {e}")
            await self._send_error(client.websocket, "Authentication failed")
    
    async def _handle_subscription(self, client: ClientConnection, data: Dict[str, Any]) -> None:
        """Handle client subscription request."""
        try:
            if not client.is_authenticated:
                await self._send_error(client.websocket, "Authentication required")
                return
            
            subscription_type = data.get("subscription_type")
            topic = data.get("topic")
            filters = data.get("filters", {})
            
            if not subscription_type or not topic:
                await self._send_error(client.websocket, "Subscription type and topic required")
                return
            
            # Check subscription limits
            if len(client.subscriptions) >= self.max_subscriptions_per_client:
                await self._send_error(client.websocket, "Maximum subscriptions reached")
                return
            
            # Create subscription
            subscription = Subscription(
                client_id=client.client_id,
                subscription_type=SubscriptionType(subscription_type),
                topic=topic,
                filters=filters,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow()
            )
            
            # Add to subscriptions
            client.subscriptions.add(topic)
            self.subscriptions[topic].add(client.client_id)
            
            # Send confirmation
            await self._send_message(client.websocket, WebSocketMessageType.SYSTEM, "subscription_confirmed", {
                "topic": topic,
                "subscription_type": subscription_type,
                "filters": filters
            })
            
            logger.info(f"Client {client.client_id} subscribed to {topic}")
            
        except Exception as e:
            logger.error(f"Error in subscription: {e}")
            await self._send_error(client.websocket, "Subscription failed")
    
    async def _handle_unsubscription(self, client: ClientConnection, data: Dict[str, Any]) -> None:
        """Handle client unsubscription request."""
        try:
            topic = data.get("topic")
            if not topic:
                await self._send_error(client.websocket, "Topic required")
                return
            
            if topic in client.subscriptions:
                client.subscriptions.remove(topic)
                self.subscriptions[topic].discard(client.client_id)
                
                # Remove empty topics
                if not self.subscriptions[topic]:
                    del self.subscriptions[topic]
                
                await self._send_message(client.websocket, WebSocketMessageType.SYSTEM, "unsubscription_confirmed", {
                    "topic": topic
                })
                
                logger.info(f"Client {client.client_id} unsubscribed from {topic}")
            else:
                await self._send_error(client.websocket, f"Not subscribed to {topic}")
                
        except Exception as e:
            logger.error(f"Error in unsubscription: {e}")
            await self._send_error(client.websocket, "Unsubscription failed")
    
    async def _handle_heartbeat(self, client: ClientConnection) -> None:
        """Handle client heartbeat."""
        try:
            client.last_heartbeat = datetime.utcnow()
            
            # Send heartbeat response
            await self._send_message(client.websocket, WebSocketMessageType.HEARTBEAT, "pong", {
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error handling heartbeat: {e}")
    
    async def broadcast_market_data(self, instrument_id: str, market_data: Dict[str, Any]) -> None:
        """Broadcast market data to subscribed clients."""
        try:
            topic = f"market_data.{instrument_id}"
            
            message = WebSocketMessage(
                message_type=WebSocketMessageType.MARKET_DATA,
                topic=topic,
                data=market_data,
                timestamp=datetime.utcnow(),
                sequence_number=self._get_next_sequence()
            )
            
            await self._broadcast_message(topic, message)
            
        except Exception as e:
            logger.error(f"Error broadcasting market data: {e}")
    
    async def broadcast_order_update(self, user_id: int, order_update: Dict[str, Any]) -> None:
        """Broadcast order update to specific user."""
        try:
            topic = f"orders.{user_id}"
            
            message = WebSocketMessage(
                message_type=WebSocketMessageType.ORDER_UPDATE,
                topic=topic,
                data=order_update,
                timestamp=datetime.utcnow(),
                sequence_number=self._get_next_sequence()
            )
            
            await self._broadcast_message(topic, message)
            
        except Exception as e:
            logger.error(f"Error broadcasting order update: {e}")
    
    async def broadcast_trade_execution(self, trade_data: Dict[str, Any]) -> None:
        """Broadcast trade execution to all clients."""
        try:
            topic = "trades"
            
            message = WebSocketMessage(
                message_type=WebSocketMessageType.TRADE_EXECUTION,
                topic=topic,
                data=trade_data,
                timestamp=datetime.utcnow(),
                sequence_number=self._get_next_sequence()
            )
            
            await self._broadcast_message(topic, message)
            
        except Exception as e:
            logger.error(f"Error broadcasting trade execution: {e}")
    
    async def broadcast_risk_alert(self, risk_alert: RiskLimitBreach) -> None:
        """Broadcast risk alert to relevant clients."""
        try:
            # Send to risk monitoring clients
            topic = "risk_alerts"
            
            message = WebSocketMessage(
                message_type=WebSocketMessageType.RISK_ALERT,
                topic=topic,
                data={
                    "breach_id": risk_alert.breach_id,
                    "limit_type": risk_alert.limit_type,
                    "portfolio_id": risk_alert.portfolio_id,
                    "severity": risk_alert.severity,
                    "breach_amount": str(risk_alert.breach_amount),
                    "timestamp": risk_alert.timestamp.isoformat()
                },
                timestamp=datetime.utcnow(),
                sequence_number=self._get_next_sequence()
            )
            
            await self._broadcast_message(topic, message)
            
            # Send to specific portfolio clients
            portfolio_topic = f"risk_alerts.{risk_alert.portfolio_id}"
            await self._broadcast_message(portfolio_topic, message)
            
        except Exception as e:
            logger.error(f"Error broadcasting risk alert: {e}")
    
    async def broadcast_portfolio_update(self, user_id: int, portfolio_data: Dict[str, Any]) -> None:
        """Broadcast portfolio update to specific user."""
        try:
            topic = f"portfolio.{user_id}"
            
            message = WebSocketMessage(
                message_type=WebSocketMessageType.PORTFOLIO_UPDATE,
                topic=topic,
                data=portfolio_data,
                timestamp=datetime.utcnow(),
                sequence_number=self._get_next_sequence()
            )
            
            await self._broadcast_message(topic, message)
            
        except Exception as e:
            logger.error(f"Error broadcasting portfolio update: {e}")
    
    async def broadcast_auction_update(self, auction_id: str, auction_data: Dict[str, Any]) -> None:
        """Broadcast auction update to subscribed clients."""
        try:
            topic = f"auction.{auction_id}"
            
            message = WebSocketMessage(
                message_type=WebSocketMessageType.AUCTION_UPDATE,
                topic=topic,
                data=auction_data,
                timestamp=datetime.utcnow(),
                sequence_number=self._get_next_sequence()
            )
            
            await self._broadcast_message(topic, message)
            
        except Exception as e:
            logger.error(f"Error broadcasting auction update: {e}")
    
    async def _broadcast_message(self, topic: str, message: WebSocketMessage) -> None:
        """Broadcast message to all subscribed clients."""
        try:
            if topic not in self.subscriptions:
                return
            
            subscribed_clients = self.subscriptions[topic]
            
            # Add to message queue for processing
            self.message_queue.append((topic, message, subscribed_clients))
            
        except Exception as e:
            logger.error(f"Error queuing broadcast message: {e}")
    
    async def _message_processor_loop(self) -> None:
        """Process messages from the queue."""
        while self.is_running:
            try:
                if self.message_queue:
                    topic, message, subscribed_clients = self.message_queue.popleft()
                    
                    # Send to subscribed clients
                    for client_id in subscribed_clients:
                        if client_id in self.clients:
                            client = self.clients[client_id]
                            try:
                                await self._send_message_to_client(client, message)
                                self.messages_sent += 1
                            except Exception as e:
                                logger.warning(f"Error sending message to client {client_id}: {e}")
                                # Mark client for cleanup
                                await self._mark_client_for_cleanup(client_id)
                
                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in message processor loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _send_message_to_client(self, client: ClientConnection, message: WebSocketMessage) -> None:
        """Send message to a specific client."""
        try:
            # Check if client is still connected
            if client.websocket.closed:
                return
            
            # Prepare message payload
            payload = {
                "type": message.message_type.value,
                "topic": message.topic,
                "data": message.data,
                "timestamp": message.timestamp.isoformat(),
                "sequence_number": message.sequence_number
            }
            
            if message.correlation_id:
                payload["correlation_id"] = message.correlation_id
            
            # Send message
            await client.websocket.send(json.dumps(payload))
            
        except Exception as e:
            logger.error(f"Error sending message to client {client.client_id}: {e}")
            raise
    
    async def _send_message(self, websocket: WebSocketServerProtocol, message_type: WebSocketMessageType, 
                           topic: str, data: Any) -> None:
        """Send a message to a specific WebSocket."""
        try:
            message = WebSocketMessage(
                message_type=message_type,
                topic=topic,
                data=data,
                timestamp=datetime.utcnow(),
                sequence_number=self._get_next_sequence()
            )
            
            await self._send_message_to_client(ClientConnection(
                client_id="temp",
                websocket=websocket,
                subscriptions=set(),
                user_id=None,
                permissions=[],
                connected_at=datetime.utcnow(),
                last_heartbeat=datetime.utcnow(),
                is_authenticated=False
            ), message)
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def _send_error(self, websocket: WebSocketServerProtocol, error_message: str) -> None:
        """Send error message to client."""
        try:
            await self._send_message(websocket, WebSocketMessageType.ERROR, "error", {
                "error": error_message,
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error(f"Error sending error message: {e}")
    
    async def _heartbeat_loop(self) -> None:
        """Send heartbeat messages and check client health."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                # Check client health
                clients_to_remove = []
                
                for client_id, client in self.clients.items():
                    time_since_heartbeat = (current_time - client.last_heartbeat).total_seconds()
                    
                    if time_since_heartbeat > self.connection_timeout:
                        logger.warning(f"Client {client_id} timed out")
                        clients_to_remove.append(client_id)
                    elif time_since_heartbeat > self.heartbeat_interval:
                        # Send heartbeat
                        try:
                            await self._send_message(client.websocket, WebSocketMessageType.HEARTBEAT, "ping", {
                                "timestamp": current_time.isoformat()
                            })
                        except Exception as e:
                            logger.warning(f"Error sending heartbeat to client {client_id}: {e}")
                            clients_to_remove.append(client_id)
                
                # Remove timed out clients
                for client_id in clients_to_remove:
                    await self._cleanup_connection(client_id)
                
                # Wait for next heartbeat cycle
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_connection(self, client_id: str) -> None:
        """Clean up a client connection."""
        try:
            if client_id in self.clients:
                client = self.clients[client_id]
                
                # Remove from all subscriptions
                for topic in client.subscriptions:
                    if topic in self.subscriptions:
                        self.subscriptions[topic].discard(client_id)
                        if not self.subscriptions[topic]:
                            del self.subscriptions[topic]
                
                # Close WebSocket connection
                if not client.websocket.closed:
                    await client.websocket.close()
                
                # Remove client
                del self.clients[client_id]
                self.connections_closed += 1
                
                logger.info(f"Cleaned up connection for client {client_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up connection for client {client_id}: {e}")
    
    async def _mark_client_for_cleanup(self, client_id: str) -> None:
        """Mark a client for cleanup."""
        try:
            if client_id in self.clients:
                client = self.clients[client_id]
                client.metadata["marked_for_cleanup"] = True
        except Exception as e:
            logger.error(f"Error marking client for cleanup: {e}")
    
    async def _close_all_connections(self) -> None:
        """Close all client connections."""
        try:
            client_ids = list(self.clients.keys())
            for client_id in client_ids:
                await self._cleanup_connection(client_id)
        except Exception as e:
            logger.error(f"Error closing all connections: {e}")
    
    def _get_next_sequence(self) -> int:
        """Get next sequence number."""
        self.sequence_counter += 1
        return self.sequence_counter
    
    def get_connection_statistics(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_clients": len(self.clients),
            "total_subscriptions": sum(len(subs) for subs in self.subscriptions.values()),
            "messages_sent": self.messages_sent,
            "connections_established": self.connections_established,
            "connections_closed": self.connections_closed,
            "message_queue_size": len(self.message_queue),
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
        }
    
    def get_subscription_statistics(self) -> Dict[str, Any]:
        """Get subscription statistics."""
        subscription_counts = {}
        for topic, clients in self.subscriptions.items():
            subscription_counts[topic] = len(clients)
        
        return {
            "total_topics": len(self.subscriptions),
            "topic_subscriptions": subscription_counts,
            "most_popular_topics": sorted(subscription_counts.items(), 
                                        key=lambda x: x[1], reverse=True)[:10]
        }


# Export classes
__all__ = ["WebSocketManager", "WebSocketMessageType", "SubscriptionType", "WebSocketMessage", 
           "Subscription", "ClientConnection"]
