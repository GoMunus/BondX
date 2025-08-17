"""
WebSocket Manager for BondX Auction System.

This module provides real-time communication infrastructure including WebSocket connections,
message broadcasting, targeted notifications, and connection management for auction participants.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed

from ..core.logging import get_logger

logger = get_logger(__name__)


class MessageType(Enum):
    """Types of WebSocket messages."""
    
    AUCTION_UPDATE = "AUCTION_UPDATE"
    BID_UPDATE = "BID_UPDATE"
    ALLOCATION_UPDATE = "ALLOCATION_UPDATE"
    SETTLEMENT_UPDATE = "SETTLEMENT_UPDATE"
    POSITION_UPDATE = "POSITION_UPDATE"
    MARKET_DATA = "MARKET_DATA"
    SYSTEM_ALERT = "SYSTEM_ALERT"
    HEARTBEAT = "HEARTBEAT"
    ERROR = "ERROR"


class ConnectionStatus(Enum):
    """WebSocket connection status."""
    
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    RECONNECTING = "RECONNECTING"


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    
    message_type: MessageType
    timestamp: datetime
    data: Dict[str, Any]
    message_id: Optional[str] = None
    source: Optional[str] = None
    target: Optional[str] = None


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection."""
    
    connection_id: str
    participant_id: Optional[int]
    participant_type: Optional[str]
    connection_time: datetime
    last_activity: datetime
    status: ConnectionStatus
    subscriptions: Set[str]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class WebSocketManager:
    """
    Sophisticated WebSocket manager for real-time communication.
    
    This manager provides:
    - Multiple concurrent WebSocket connections
    - Message broadcasting and targeted notifications
    - Connection health monitoring and reconnection handling
    - Message queuing for offline participants
    - Security and rate limiting
    - Comprehensive logging and analytics
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        """Initialize the WebSocket manager."""
        self.host = host
        self.port = port
        self.logger = get_logger(__name__)
        
        # Connection management
        self.connections: Dict[str, WebSocketServerProtocol] = {}
        self.connection_info: Dict[str, ConnectionInfo] = {}
        self.connection_counter = 0
        
        # Message handling
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        self.message_queue: Dict[str, List[WebSocketMessage]] = {}
        
        # Subscription management
        self.subscriptions: Dict[str, Set[str]] = {}  # topic -> set of connection_ids
        
        # Performance monitoring
        self.connection_count = 0
        self.message_count = 0
        self.error_count = 0
        
        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        
        # Server instance
        self.server = None
        self.is_running = False
        
        logger.info("WebSocket Manager initialized")
    
    async def start_server(self):
        """Start the WebSocket server."""
        try:
            self.server = await websockets.serve(
                self._handle_connection,
                self.host,
                self.port
            )
            self.is_running = True
            
            logger.info(f"WebSocket server started on {self.host}:{self.port}")
            
            # Start background tasks
            asyncio.create_task(self._heartbeat_task())
            asyncio.create_task(self._cleanup_task())
            asyncio.create_task(self._monitoring_task())
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {str(e)}")
            raise
    
    async def stop_server(self):
        """Stop the WebSocket server."""
        try:
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            
            self.is_running = False
            
            # Close all connections
            for connection_id in list(self.connections.keys()):
                await self._close_connection(connection_id)
            
            logger.info("WebSocket server stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop WebSocket server: {str(e)}")
    
    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection."""
        connection_id = f"conn_{self.connection_counter}"
        self.connection_counter += 1
        
        try:
            # Create connection info
            connection_info = ConnectionInfo(
                connection_id=connection_id,
                participant_id=None,
                participant_type=None,
                connection_time=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                status=ConnectionStatus.CONNECTING,
                subscriptions=set(),
                ip_address=websocket.remote_address[0] if websocket.remote_address else None
            )
            
            # Store connection
            self.connections[connection_id] = websocket
            self.connection_info[connection_id] = connection_info
            self.connection_count += 1
            
            # Update status
            connection_info.status = ConnectionStatus.CONNECTED
            
            logger.info(f"New WebSocket connection: {connection_id}")
            
            # Send welcome message
            welcome_message = WebSocketMessage(
                message_type=MessageType.SYSTEM_ALERT,
                timestamp=datetime.utcnow(),
                data={
                    "message": "Connected to BondX Auction System",
                    "connection_id": connection_id,
                    "server_time": datetime.utcnow().isoformat()
                }
            )
            
            await self._send_message_to_connection(connection_id, welcome_message)
            
            # Handle incoming messages
            async for message in websocket:
                await self._handle_incoming_message(connection_id, message)
                connection_info.last_activity = datetime.utcnow()
                
        except ConnectionClosed:
            logger.info(f"WebSocket connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket connection {connection_id}: {str(e)}")
            self.error_count += 1
        finally:
            await self._close_connection(connection_id)
    
    async def _handle_incoming_message(self, connection_id: str, message: str):
        """Handle incoming WebSocket message."""
        try:
            # Parse message
            message_data = json.loads(message)
            message_type = MessageType(message_data.get("type", "ERROR"))
            
            # Rate limiting check
            if not self._check_rate_limit(connection_id, message_type):
                await self._send_error(connection_id, "Rate limit exceeded")
                return
            
            # Handle different message types
            if message_type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(connection_id, message_data)
            elif message_type == MessageType.AUCTION_UPDATE:
                await self._handle_auction_update(connection_id, message_data)
            elif message_type == MessageType.BID_UPDATE:
                await self._handle_bid_update(connection_id, message_data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
            
            self.message_count += 1
            
        except json.JSONDecodeError:
            await self._send_error(connection_id, "Invalid JSON format")
        except Exception as e:
            logger.error(f"Error handling incoming message: {str(e)}")
            await self._send_error(connection_id, "Internal server error")
    
    async def _handle_heartbeat(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle heartbeat message."""
        try:
            # Update last activity
            if connection_id in self.connection_info:
                self.connection_info[connection_id].last_activity = datetime.utcnow()
            
            # Send heartbeat response
            response = WebSocketMessage(
                message_type=MessageType.HEARTBEAT,
                timestamp=datetime.utcnow(),
                data={"status": "ok", "server_time": datetime.utcnow().isoformat()}
            )
            
            await self._send_message_to_connection(connection_id, response)
            
        except Exception as e:
            logger.error(f"Error handling heartbeat: {str(e)}")
    
    async def _handle_auction_update(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle auction update message."""
        try:
            # Extract auction information
            auction_id = message_data.get("auction_id")
            if not auction_id:
                await self._send_error(connection_id, "Missing auction_id")
                return
            
            # Subscribe to auction updates
            topic = f"auction_{auction_id}"
            await self._subscribe_connection(connection_id, topic)
            
            # Send confirmation
            response = WebSocketMessage(
                message_type=MessageType.SYSTEM_ALERT,
                timestamp=datetime.utcnow(),
                data={
                    "message": f"Subscribed to auction {auction_id} updates",
                    "topic": topic
                }
            )
            
            await self._send_message_to_connection(connection_id, response)
            
        except Exception as e:
            logger.error(f"Error handling auction update: {str(e)}")
    
    async def _handle_bid_update(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle bid update message."""
        try:
            # Extract bid information
            bid_id = message_data.get("bid_id")
            if not bid_id:
                await self._send_error(connection_id, "Missing bid_id")
                return
            
            # Subscribe to bid updates
            topic = f"bid_{bid_id}"
            await self._subscribe_connection(connection_id, topic)
            
            # Send confirmation
            response = WebSocketMessage(
                message_type=MessageType.SYSTEM_ALERT,
                timestamp=datetime.utcnow(),
                data={
                    "message": f"Subscribed to bid {bid_id} updates",
                    "topic": topic
                }
            )
            
            await self._send_message_to_connection(connection_id, response)
            
        except Exception as e:
            logger.error(f"Error handling bid update: {str(e)}")
    
    async def broadcast_message(self, message: WebSocketMessage, topic: Optional[str] = None):
        """
        Broadcast message to all connections or specific topic subscribers.
        
        Args:
            message: Message to broadcast
            topic: Optional topic to filter recipients
        """
        try:
            if topic:
                # Send to topic subscribers
                if topic in self.subscriptions:
                    for connection_id in self.subscriptions[topic]:
                        await self._send_message_to_connection(connection_id, message)
            else:
                # Send to all connections
                for connection_id in list(self.connections.keys()):
                    await self._send_message_to_connection(connection_id, message)
            
            logger.info(f"Broadcasted {message.message_type.value} message to {topic or 'all'} connections")
            
        except Exception as e:
            logger.error(f"Error broadcasting message: {str(e)}")
    
    async def send_targeted_message(self, participant_id: int, message: WebSocketMessage):
        """
        Send message to a specific participant.
        
        Args:
            participant_id: ID of the participant
            message: Message to send
        """
        try:
            # Find connections for this participant
            target_connections = []
            for conn_id, conn_info in self.connection_info.items():
                if conn_info.participant_id == participant_id:
                    target_connections.append(conn_id)
            
            # Send message to all participant connections
            for connection_id in target_connections:
                await self._send_message_to_connection(connection_id, message)
            
            logger.info(f"Sent targeted message to participant {participant_id} ({len(target_connections)} connections)")
            
        except Exception as e:
            logger.error(f"Error sending targeted message: {str(e)}")
    
    async def _send_message_to_connection(self, connection_id: str, message: WebSocketMessage):
        """Send message to a specific connection."""
        try:
            if connection_id not in self.connections:
                logger.warning(f"Connection {connection_id} not found")
                return
            
            websocket = self.connections[connection_id]
            
            # Convert message to JSON
            message_json = json.dumps(asdict(message), default=str)
            
            # Send message
            await websocket.send(message_json)
            
        except ConnectionClosed:
            logger.info(f"Connection {connection_id} closed, removing from active connections")
            await self._close_connection(connection_id)
        except Exception as e:
            logger.error(f"Error sending message to connection {connection_id}: {str(e)}")
            await self._close_connection(connection_id)
    
    async def _send_error(self, connection_id: str, error_message: str):
        """Send error message to connection."""
        try:
            error_msg = WebSocketMessage(
                message_type=MessageType.ERROR,
                timestamp=datetime.utcnow(),
                data={"error": error_message}
            )
            
            await self._send_message_to_connection(connection_id, error_msg)
            
        except Exception as e:
            logger.error(f"Error sending error message: {str(e)}")
    
    async def _subscribe_connection(self, connection_id: str, topic: str):
        """Subscribe a connection to a topic."""
        try:
            if topic not in self.subscriptions:
                self.subscriptions[topic] = set()
            
            self.subscriptions[topic].add(connection_id)
            
            # Update connection info
            if connection_id in self.connection_info:
                self.connection_info[connection_id].subscriptions.add(topic)
            
            logger.info(f"Connection {connection_id} subscribed to topic {topic}")
            
        except Exception as e:
            logger.error(f"Error subscribing connection to topic: {str(e)}")
    
    async def _unsubscribe_connection(self, connection_id: str, topic: str):
        """Unsubscribe a connection from a topic."""
        try:
            if topic in self.subscriptions:
                self.subscriptions[topic].discard(connection_id)
                
                # Remove empty topics
                if not self.subscriptions[topic]:
                    del self.subscriptions[topic]
            
            # Update connection info
            if connection_id in self.connection_info:
                self.connection_info[connection_id].subscriptions.discard(topic)
            
            logger.info(f"Connection {connection_id} unsubscribed from topic {topic}")
            
        except Exception as e:
            logger.error(f"Error unsubscribing connection from topic: {str(e)}")
    
    async def _close_connection(self, connection_id: str):
        """Close a WebSocket connection."""
        try:
            # Remove from active connections
            if connection_id in self.connections:
                del self.connections[connection_id]
                self.connection_count -= 1
            
            # Remove from subscriptions
            if connection_id in self.connection_info:
                conn_info = self.connection_info[connection_id]
                for topic in list(conn_info.subscriptions):
                    await self._unsubscribe_connection(connection_id, topic)
                
                del self.connection_info[connection_id]
            
            logger.info(f"Connection {connection_id} closed and cleaned up")
            
        except Exception as e:
            logger.error(f"Error closing connection {connection_id}: {str(e)}")
    
    def _check_rate_limit(self, connection_id: str, message_type: MessageType) -> bool:
        """Check if connection is within rate limits."""
        try:
            current_time = datetime.utcnow()
            
            if connection_id not in self.rate_limits:
                self.rate_limits[connection_id] = {
                    'message_count': 0,
                    'last_reset': current_time,
                    'limits': {
                        MessageType.HEARTBEAT: {'max_per_minute': 60},
                        MessageType.AUCTION_UPDATE: {'max_per_minute': 10},
                        MessageType.BID_UPDATE: {'max_per_minute': 20},
                        MessageType.ALLOCATION_UPDATE: {'max_per_minute': 5},
                        MessageType.SETTLEMENT_UPDATE: {'max_per_minute': 5},
                        MessageType.POSITION_UPDATE: {'max_per_minute': 10},
                        MessageType.MARKET_DATA: {'max_per_minute': 30},
                        MessageType.SYSTEM_ALERT: {'max_per_minute': 5},
                        MessageType.ERROR: {'max_per_minute': 10}
                    }
                }
            
            rate_info = self.rate_limits[connection_id]
            
            # Reset counter if minute has passed
            if (current_time - rate_info['last_reset']).total_seconds() >= 60:
                rate_info['message_count'] = 0
                rate_info['last_reset'] = current_time
            
            # Check limits for message type
            limits = rate_info['limits'].get(message_type, {'max_per_minute': 10})
            max_per_minute = limits['max_per_minute']
            
            if rate_info['message_count'] >= max_per_minute:
                return False
            
            rate_info['message_count'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {str(e)}")
            return True  # Allow if rate limiting fails
    
    async def _heartbeat_task(self):
        """Background task to send heartbeat messages."""
        try:
            while self.is_running:
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                
                if self.connections:
                    heartbeat_message = WebSocketMessage(
                        message_type=MessageType.HEARTBEAT,
                        timestamp=datetime.utcnow(),
                        data={
                            "status": "heartbeat",
                            "server_time": datetime.utcnow().isoformat(),
                            "connection_count": self.connection_count
                        }
                    )
                    
                    await self.broadcast_message(heartbeat_message)
                    
        except Exception as e:
            logger.error(f"Error in heartbeat task: {str(e)}")
    
    async def _cleanup_task(self):
        """Background task to clean up inactive connections."""
        try:
            while self.is_running:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.utcnow()
                inactive_connections = []
                
                for conn_id, conn_info in self.connection_info.items():
                    # Close connections inactive for more than 5 minutes
                    if (current_time - conn_info.last_activity).total_seconds() > 300:
                        inactive_connections.append(conn_id)
                
                # Close inactive connections
                for conn_id in inactive_connections:
                    await self._close_connection(conn_id)
                
                if inactive_connections:
                    logger.info(f"Cleaned up {len(inactive_connections)} inactive connections")
                    
        except Exception as e:
            logger.error(f"Error in cleanup task: {str(e)}")
    
    async def _monitoring_task(self):
        """Background task to monitor system health."""
        try:
            while self.is_running:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Log system statistics
                logger.info(f"WebSocket System Stats - Connections: {self.connection_count}, "
                          f"Messages: {self.message_count}, Errors: {self.error_count}")
                
                # Reset counters
                self.message_count = 0
                self.error_count = 0
                
        except Exception as e:
            logger.error(f"Error in monitoring task: {str(e)}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            return {
                "connection_count": self.connection_count,
                "message_count": self.message_count,
                "error_count": self.error_count,
                "subscription_count": len(self.subscriptions),
                "is_running": self.is_running,
                "server_address": f"{self.host}:{self.port}",
                "uptime": datetime.utcnow().isoformat() if self.is_running else None
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {}
    
    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific connection."""
        try:
            if connection_id not in self.connection_info:
                return None
            
            conn_info = self.connection_info[connection_id]
            return {
                "connection_id": conn_info.connection_id,
                "participant_id": conn_info.participant_id,
                "participant_type": conn_info.participant_type,
                "connection_time": conn_info.connection_time.isoformat(),
                "last_activity": conn_info.last_activity.isoformat(),
                "status": conn_info.status.value,
                "subscriptions": list(conn_info.subscriptions),
                "ip_address": conn_info.ip_address,
                "user_agent": conn_info.user_agent
            }
        except Exception as e:
            logger.error(f"Error getting connection info: {str(e)}")
            return None
