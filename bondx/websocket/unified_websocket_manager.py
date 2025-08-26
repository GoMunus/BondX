"""
Unified WebSocket Manager for BondX Real-time Streaming.

This module provides a centralized WebSocket management system that handles:
- Multiple topic subscriptions with room semantics
- Snapshot + incremental update protocol
- Connection management and authentication
- Backpressure handling and rate limiting
- Heartbeat and ping/pong management
- Load balancing and horizontal scaling
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import weakref

from fastapi import WebSocket
import redis.asyncio as redis

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)

# Enums and constants
class MessageType(Enum):
    """Types of WebSocket messages."""
    SNAPSHOT = "snapshot"
    DELTA = "delta"
    ALERT = "alert"
    ACK = "ack"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"
    HEARTBEAT = "heartbeat"

class Topic(Enum):
    """Predefined topic patterns."""
    MARKET_DATA = "prices.{isin}"
    AUCTION = "auctions.{auction_id}"
    TRADING = "trading.{user_id}"
    PORTFOLIO = "portfolio.{user_id}"
    RISK = "risk.{user_id}"
    RISK_ALERTS = "risk.{user_id}.alerts"
    MOBILE = "mobile.{user_id}"

class UpdateFrequency(Enum):
    """Update frequency levels."""
    HIGH = "high"      # Real-time updates
    NORMAL = "normal"  # Standard updates
    LOW = "low"        # Reduced frequency
    BACKGROUND = "background"  # Minimal updates

# Data models
@dataclass
class WebSocketMessage:
    """Standardized WebSocket message envelope."""
    type: MessageType
    topic: str
    sequence_number: int
    timestamp: datetime
    payload: Any
    meta: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "topic": self.topic,
            "seq": self.sequence_number,
            "ts": self.timestamp.isoformat(),
            "payload": self.payload,
            "meta": self.meta or {},
            "correlation_id": self.correlation_id
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

@dataclass
class ConnectionMetadata:
    """Connection metadata and configuration."""
    user_id: Optional[str] = None
    device_type: Optional[str] = None
    client_version: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    rate_limit_buckets: Dict[str, Any] = field(default_factory=dict)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    subscription_count: int = 0
    compression_enabled: bool = True
    update_frequency: UpdateFrequency = UpdateFrequency.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WebSocketConnection:
    """WebSocket connection representation."""
    id: str
    websocket: WebSocket
    metadata: ConnectionMetadata
    subscriptions: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    send_queue: deque = field(default_factory=lambda: deque(maxlen=1000))
    sequence_numbers: Dict[str, int] = field(default_factory=dict)

@dataclass
class TopicSubscription:
    """Topic subscription information."""
    topic: str
    connections: Set[str] = field(default_factory=set)
    last_sequence: int = 0
    last_update: datetime = field(default_factory=datetime.utcnow)
    update_frequency: UpdateFrequency = UpdateFrequency.NORMAL
    snapshot_provider: Optional[Callable] = None
    message_history: deque = field(default_factory=lambda: deque(maxlen=100))

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    burst: int
    sustained: int
    window: int
    tokens: int
    last_refill: datetime

class UnifiedWebSocketManager:
    """
    Unified WebSocket manager for centralized real-time streaming.
    
    Features:
    - Multi-topic subscription management
    - Snapshot + incremental update protocol
    - Connection lifecycle management
    - Rate limiting and backpressure handling
    - Heartbeat and health monitoring
    - Horizontal scaling support via Redis pub/sub
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize the unified WebSocket manager."""
        self.redis_url = redis_url or settings.redis.url
        self.redis_client: Optional[redis.Redis] = None
        
        # Connection management
        self.connections: Dict[str, WebSocketConnection] = {}
        self.topic_subscriptions: Dict[str, TopicSubscription] = defaultdict(
            lambda: TopicSubscription(topic="", connections=set())
        )
        
        # Message management
        self.sequence_counters: Dict[str, int] = defaultdict(int)
        self.message_queue: deque = deque(maxlen=10000)
        self.broadcast_queue: deque = deque(maxlen=5000)
        
        # Configuration
        self.max_connections = settings.websocket.max_connections
        self.heartbeat_interval = settings.websocket.ping_interval
        self.connection_timeout = settings.websocket.ping_timeout * 3
        self.max_queue_size = 1000
        self.batch_size = 100
        
        # Performance tracking
        self.messages_sent = 0
        self.messages_dropped = 0
        self.connections_established = 0
        self.connections_closed = 0
        self.start_time = datetime.utcnow()
        
        # Task management
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.message_processor_task: Optional[asyncio.Task] = None
        self.broadcast_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Redis pub/sub for horizontal scaling
        self.redis_pubsub: Optional[redis.client.PubSub] = None
        self.redis_subscription_task: Optional[asyncio.Task] = None
        
        logger.info("Unified WebSocket Manager initialized")
    
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
            
            # Initialize Redis pub/sub
            await self._setup_redis_pubsub()
            
            # Start background tasks
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self.message_processor_task = asyncio.create_task(self._message_processor_loop())
            self.broadcast_task = asyncio.create_task(self._broadcast_loop())
            
            self.is_running = True
            logger.info("Unified WebSocket Manager started successfully")
            
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
        tasks = [
            self.heartbeat_task,
            self.message_processor_task,
            self.broadcast_task,
            self.redis_subscription_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close all connections
        await self._close_all_connections()
        
        # Close Redis connections
        if self.redis_pubsub:
            await self.redis_pubsub.close()
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Unified WebSocket Manager stopped")
    
    async def _setup_redis_pubsub(self) -> None:
        """Setup Redis pub/sub for horizontal scaling."""
        try:
            self.redis_pubsub = self.redis_client.pubsub()
            
            # Subscribe to internal channels
            await self.redis_pubsub.subscribe(
                "websocket.broadcast",
                "websocket.topic.*",
                "websocket.connection.*"
            )
            
            # Start Redis subscription handler
            self.redis_subscription_task = asyncio.create_task(self._redis_subscription_handler())
            
            logger.info("Redis pub/sub setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up Redis pub/sub: {e}")
            raise
    
    async def _redis_subscription_handler(self) -> None:
        """Handle Redis pub/sub messages for horizontal scaling."""
        try:
            async for message in self.redis_pubsub.listen():
                if message["type"] == "message":
                    await self._handle_redis_message(message)
        except Exception as e:
            logger.error(f"Error in Redis subscription handler: {e}")
    
    async def _handle_redis_message(self, message: Dict[str, Any]) -> None:
        """Handle incoming Redis pub/sub message."""
        try:
            channel = message["channel"]
            data = json.loads(message["data"])
            
            if channel == "websocket.broadcast":
                await self._handle_broadcast_message(data)
            elif channel.startswith("websocket.topic."):
                topic = channel.replace("websocket.topic.", "")
                await self._handle_topic_message(topic, data)
            elif channel.startswith("websocket.connection."):
                connection_id = channel.replace("websocket.connection.", "")
                await self._handle_connection_message(connection_id, data)
                
        except Exception as e:
            logger.error(f"Error handling Redis message: {e}")
    
    async def _handle_broadcast_message(self, data: Dict[str, Any]) -> None:
        """Handle broadcast message from Redis."""
        try:
            topic = data.get("topic")
            message_data = data.get("message")
            if topic and message_data:
                # Add to broadcast queue for local processing
                self.broadcast_queue.append((topic, message_data, data.get("qos", 1)))
        except Exception as e:
            logger.error(f"Error handling broadcast message: {e}")
    
    async def _handle_topic_message(self, topic: str, data: Dict[str, Any]) -> None:
        """Handle topic-specific message from Redis."""
        try:
            message_data = data.get("message")
            if message_data:
                # Add to broadcast queue for local processing
                self.broadcast_queue.append((topic, message_data, data.get("qos", 1)))
        except Exception as e:
            logger.error(f"Error handling topic message: {e}")
    
    async def _handle_connection_message(self, connection_id: str, data: Dict[str, Any]) -> None:
        """Handle connection-specific message from Redis."""
        try:
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                # Handle connection-specific operations
                pass
        except Exception as e:
            logger.error(f"Error handling connection message: {e}")
    
    async def register_connection(self, connection: WebSocketConnection) -> None:
        """Register a new WebSocket connection."""
        try:
            # Check connection limits
            if len(self.connections) >= self.max_connections:
                raise ValueError("Maximum connections reached")
            
            # Store connection
            self.connections[connection.id] = connection
            self.connections_established += 1
            
            # Initialize sequence numbers
            for topic in connection.subscriptions:
                connection.sequence_numbers[topic] = 0
            
            logger.info(f"Connection registered: {connection.id}")
            
        except Exception as e:
            logger.error(f"Error registering connection: {e}")
            raise
    
    async def unregister_connection(self, connection_id: str) -> None:
        """Unregister a WebSocket connection."""
        try:
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                
                # Remove from all topic subscriptions
                for topic in list(connection.subscriptions):
                    await self.unsubscribe_connection(connection_id, topic)
                
                # Close WebSocket if still open
                if not connection.websocket.client_state.disconnected:
                    await connection.websocket.close()
                
                # Remove connection
                del self.connections[connection_id]
                self.connections_closed += 1
                
                logger.info(f"Connection unregistered: {connection_id}")
                
        except Exception as e:
            logger.error(f"Error unregistering connection: {connection_id}: {e}")
    
    async def subscribe_connection(self, connection_id: str, topic: str) -> bool:
        """Subscribe a connection to a topic."""
        try:
            if connection_id not in self.connections:
                return False
            
            connection = self.connections[connection_id]
            
            # Add to connection subscriptions
            connection.subscriptions.add(topic)
            connection.metadata.subscription_count = len(connection.subscriptions)
            
            # Add to topic subscriptions
            if topic not in self.topic_subscriptions:
                self.topic_subscriptions[topic] = TopicSubscription(topic=topic)
            
            self.topic_subscriptions[topic].connections.add(connection_id)
            
            # Initialize sequence number for this topic
            if topic not in connection.sequence_numbers:
                connection.sequence_numbers[topic] = 0
            
            logger.info(f"Connection {connection_id} subscribed to {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing connection {connection_id} to {topic}: {e}")
            return False
    
    async def unsubscribe_connection(self, connection_id: str, topic: str) -> bool:
        """Unsubscribe a connection from a topic."""
        try:
            if connection_id not in self.connections:
                return False
            
            connection = self.connections[connection_id]
            
            # Remove from connection subscriptions
            connection.subscriptions.discard(topic)
            connection.metadata.subscription_count = len(connection.subscriptions)
            
            # Remove from topic subscriptions
            if topic in self.topic_subscriptions:
                self.topic_subscriptions[topic].connections.discard(connection_id)
                
                # Clean up empty topics
                if not self.topic_subscriptions[topic].connections:
                    del self.topic_subscriptions[topic]
            
            # Remove sequence number
            connection.sequence_numbers.pop(topic, None)
            
            logger.info(f"Connection {connection_id} unsubscribed from {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Error unsubscribing connection {connection_id} from {topic}: {e}")
            return False
    
    async def publish_message(self, topic: str, payload: Any, message_type: MessageType = MessageType.DELTA, 
                            qos: int = 1, correlation_id: Optional[str] = None) -> None:
        """Publish a message to a topic."""
        try:
            # Get next sequence number
            sequence = self._get_next_sequence(topic)
            
            # Create message
            message = WebSocketMessage(
                type=message_type,
                topic=topic,
                sequence_number=sequence,
                timestamp=datetime.utcnow(),
                payload=payload,
                correlation_id=correlation_id
            )
            
            # Add to broadcast queue
            self.broadcast_queue.append((topic, message, qos))
            
            # Store in topic history
            if topic in self.topic_subscriptions:
                self.topic_subscriptions[topic].message_history.append(message)
                self.topic_subscriptions[topic].last_sequence = sequence
                self.topic_subscriptions[topic].last_update = datetime.utcnow()
            
            # Publish to Redis for horizontal scaling
            if self.redis_client:
                await self.redis_client.publish(
                    f"websocket.topic.{topic}",
                    json.dumps({
                        "topic": topic,
                        "message": message.to_dict(),
                        "qos": qos
                    })
                )
            
        except Exception as e:
            logger.error(f"Error publishing message to {topic}: {e}")
    
    async def broadcast_snapshot(self, topic: str, payload: Any, correlation_id: Optional[str] = None) -> None:
        """Broadcast a snapshot message to a topic."""
        await self.publish_message(topic, payload, MessageType.SNAPSHOT, qos=2, correlation_id=correlation_id)
    
    async def get_snapshot(self, topic: str) -> Optional[WebSocketMessage]:
        """Get the latest snapshot for a topic."""
        try:
            if topic in self.topic_subscriptions:
                subscription = self.topic_subscriptions[topic]
                
                # Check if we have a snapshot provider
                if subscription.snapshot_provider:
                    try:
                        payload = await subscription.snapshot_provider()
                        return WebSocketMessage(
                            type=MessageType.SNAPSHOT,
                            topic=topic,
                            sequence_number=subscription.last_sequence,
                            timestamp=datetime.utcnow(),
                            payload=payload
                        )
                    except Exception as e:
                        logger.error(f"Error getting snapshot from provider for {topic}: {e}")
                
                # Return last message if available
                if subscription.message_history:
                    last_message = subscription.message_history[-1]
                    return WebSocketMessage(
                        type=MessageType.SNAPSHOT,
                        topic=topic,
                        sequence_number=last_message.sequence_number,
                        timestamp=datetime.utcnow(),
                        payload=last_message.payload
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting snapshot for {topic}: {e}")
            return None
    
    async def get_sequence_number(self, topic: str) -> int:
        """Get the current sequence number for a topic."""
        return self.sequence_counters.get(topic, 0)
    
    async def set_connection_update_frequency(self, connection_id: str, topic: str, frequency: Union[str, UpdateFrequency]) -> bool:
        """Set update frequency for a connection on a specific topic."""
        try:
            if connection_id not in self.connections:
                return False
            
            if isinstance(frequency, str):
                frequency = UpdateFrequency(frequency)
            
            connection = self.connections[connection_id]
            connection.metadata.update_frequency = frequency
            
            logger.info(f"Connection {connection_id} update frequency set to {frequency.value} for {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting update frequency: {e}")
            return False
    
    async def list_topics(self) -> List[Dict[str, Any]]:
        """List all available topics with subscription counts."""
        try:
            topics = []
            for topic, subscription in self.topic_subscriptions.items():
                topics.append({
                    "topic": topic,
                    "subscriber_count": len(subscription.connections),
                    "last_sequence": subscription.last_sequence,
                    "last_update": subscription.last_update.isoformat(),
                    "update_frequency": subscription.update_frequency.value
                })
            
            return sorted(topics, key=lambda x: x["subscriber_count"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing topics: {e}")
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics."""
        try:
            return {
                "connections": {
                    "total": len(self.connections),
                    "established": self.connections_established,
                    "closed": self.connections_closed,
                    "active": len([c for c in self.connections.values() if c.is_active])
                },
                "topics": {
                    "total": len(self.topic_subscriptions),
                    "subscriptions": sum(len(sub.connections) for sub in self.topic_subscriptions.values())
                },
                "messages": {
                    "sent": self.messages_sent,
                    "dropped": self.messages_dropped,
                    "queue_size": len(self.message_queue),
                    "broadcast_queue_size": len(self.broadcast_queue)
                },
                "performance": {
                    "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
                    "messages_per_second": self.messages_sent / max(1, (datetime.utcnow() - self.start_time).total_seconds())
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    async def _broadcast_loop(self) -> None:
        """Process broadcast queue and send messages to subscribers."""
        while self.is_running:
            try:
                if self.broadcast_queue:
                    topic, message, qos = self.broadcast_queue.popleft()
                    
                    if topic in self.topic_subscriptions:
                        subscription = self.topic_subscriptions[topic]
                        
                        # Send to all subscribed connections
                        for connection_id in list(subscription.connections):
                            if connection_id in self.connections:
                                connection = self.connections[connection_id]
                                
                                if connection.is_active:
                                    try:
                                        # Check rate limits and backpressure
                                        if await self._should_send_message(connection, topic, message):
                                            await self._send_message_to_connection(connection, message)
                                            self.messages_sent += 1
                                        else:
                                            self.messages_dropped += 1
                                            
                                    except Exception as e:
                                        logger.warning(f"Error sending message to {connection_id}: {e}")
                                        # Mark connection for cleanup
                                        connection.is_active = False
                                else:
                                    # Remove inactive connection
                                    subscription.connections.discard(connection_id)
                
                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in broadcast loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _should_send_message(self, connection: WebSocketConnection, topic: str, message: WebSocketMessage) -> bool:
        """Check if message should be sent to connection based on rate limits and backpressure."""
        try:
            # Check connection queue size
            if len(connection.send_queue) >= self.max_queue_size:
                return False
            
            # Check rate limits based on update frequency
            frequency = connection.metadata.update_frequency
            if frequency == UpdateFrequency.BACKGROUND:
                # Only send critical messages in background mode
                return message.type in [MessageType.ALERT, MessageType.ERROR]
            elif frequency == UpdateFrequency.LOW:
                # Reduce message frequency
                return connection.sequence_numbers.get(topic, 0) % 3 == 0
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking if should send message: {e}")
            return True
    
    async def _send_message_to_connection(self, connection: WebSocketConnection, message: WebSocketMessage) -> None:
        """Send message to a specific connection."""
        try:
            # Check if connection is still active
            if not connection.is_active or connection.websocket.client_state.disconnected:
                return
            
            # Update sequence number
            connection.sequence_numbers[message.topic] = message.sequence_number
            
            # Add to send queue
            connection.send_queue.append(message)
            
            # Send immediately if queue is small
            if len(connection.send_queue) <= 10:
                await self._flush_connection_queue(connection)
            
        except Exception as e:
            logger.error(f"Error sending message to connection {connection.id}: {e}")
            connection.is_active = False
    
    async def _flush_connection_queue(self, connection: WebSocketConnection) -> None:
        """Flush the send queue for a connection."""
        try:
            while connection.send_queue and connection.is_active:
                message = connection.send_queue.popleft()
                
                # Check if connection is still active
                if connection.websocket.client_state.disconnected:
                    break
                
                # Send message
                await connection.websocket.send_text(message.to_json())
                
        except Exception as e:
            logger.error(f"Error flushing connection queue for {connection.id}: {e}")
            connection.is_active = False
    
    async def _message_processor_loop(self) -> None:
        """Process messages from the queue."""
        while self.is_running:
            try:
                if self.message_queue:
                    # Process message queue
                    pass
                
                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in message processor loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _heartbeat_loop(self) -> None:
        """Send heartbeat messages and check connection health."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                # Check connection health
                connections_to_remove = []
                
                for connection_id, connection in self.connections.items():
                    time_since_heartbeat = (current_time - connection.last_heartbeat).total_seconds()
                    
                    if time_since_heartbeat > self.connection_timeout:
                        logger.warning(f"Connection {connection_id} timed out")
                        connections_to_remove.append(connection_id)
                    elif time_since_heartbeat > self.heartbeat_interval:
                        # Send heartbeat
                        try:
                            heartbeat_msg = WebSocketMessage(
                                type=MessageType.HEARTBEAT,
                                topic="system",
                                sequence_number=0,
                                timestamp=current_time,
                                payload={"timestamp": current_time.isoformat()}
                            )
                            await connection.websocket.send_text(heartbeat_msg.to_json())
                        except Exception as e:
                            logger.warning(f"Error sending heartbeat to {connection_id}: {e}")
                            connections_to_remove.append(connection_id)
                
                # Remove timed out connections
                for connection_id in connections_to_remove:
                    await self.unregister_connection(connection_id)
                
                # Wait for next heartbeat cycle
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)
    
    async def _close_all_connections(self) -> None:
        """Close all client connections."""
        try:
            connection_ids = list(self.connections.keys())
            for connection_id in connection_ids:
                await self.unregister_connection(connection_id)
        except Exception as e:
            logger.error(f"Error closing all connections: {e}")
    
    def _get_next_sequence(self, topic: str) -> int:
        """Get next sequence number for a topic."""
        self.sequence_counters[topic] += 1
        return self.sequence_counters[topic]

# Export classes
__all__ = [
    "UnifiedWebSocketManager",
    "WebSocketConnection", 
    "WebSocketMessage",
    "ConnectionMetadata",
    "TopicSubscription",
    "MessageType",
    "Topic",
    "UpdateFrequency"
]
