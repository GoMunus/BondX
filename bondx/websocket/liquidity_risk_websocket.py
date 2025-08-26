"""
Liquidity-Risk Translator WebSocket Manager for BondX

This module provides real-time WebSocket updates for liquidity-risk translations,
including risk category updates, liquidity changes, and exit pathway updates.
"""

import asyncio
import json
import logging
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
from ..ai_risk_engine.liquidity_risk_orchestrator import (
    LiquidityRiskOrchestrator, LiquidityRiskTranslation
)

logger = get_logger(__name__)

class LiquidityRiskEventType(Enum):
    """Types of liquidity-risk WebSocket events."""
    SNAPSHOT = "snapshot"           # Full liquidity-risk translation
    RISK_UPDATE = "risk_update"      # Risk category score change
    LIQUIDITY_UPDATE = "liquidity_update"  # Liquidity profile change
    EXIT_PATH_UPDATE = "exit_path_update"  # Exit pathway change
    ALERT = "alert"                  # Risk or liquidity alert
    HEARTBEAT = "heartbeat"          # Keep-alive message

class LiquidityRiskTopic(Enum):
    """Predefined topic patterns for liquidity-risk updates."""
    LIQUIDITY_RISK = "lr.{isin}"           # Main liquidity-risk topic
    RISK_ALERTS = "lr.{isin}.alerts"       # Risk-specific alerts
    LIQUIDITY_ALERTS = "lr.{isin}.liquidity"  # Liquidity-specific alerts
    EXIT_ALERTS = "lr.{isin}.exit"         # Exit pathway alerts

@dataclass
class LiquidityRiskWebSocketMessage:
    """Standardized WebSocket message for liquidity-risk updates."""
    type: LiquidityRiskEventType
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
class LiquidityRiskConnection:
    """Connection metadata for liquidity-risk WebSocket connections."""
    websocket: WebSocket
    user_id: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    subscriptions: Set[str] = field(default_factory=set)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    sequence_numbers: Dict[str, int] = field(default_factory=dict)

class LiquidityRiskWebSocketManager:
    """
    Manages WebSocket connections for real-time liquidity-risk updates.
    """
    
    def __init__(self):
        self.active_connections: Dict[str, LiquidityRiskConnection] = {}
        self.topic_subscribers: Dict[str, Set[str]] = defaultdict(set)
        self.sequence_counters: Dict[str, int] = defaultdict(int)
        self.orchestrator = LiquidityRiskOrchestrator()
        self.redis_client: Optional[redis.Redis] = None
        self._initialize_redis()
        
    async def _initialize_redis(self):
        """Initialize Redis connection for pub/sub."""
        try:
            self.redis_client = redis.Redis.from_url(
                settings.REDIS_URL,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established for liquidity-risk WebSocket")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None):
        """Accept new WebSocket connection."""
        try:
            await websocket.accept()
            
            # Create connection metadata
            connection_id = str(id(websocket))
            connection = LiquidityRiskConnection(
                websocket=websocket,
                user_id=user_id,
                permissions=self._get_user_permissions(user_id)
            )
            
            self.active_connections[connection_id] = connection
            
            # Send welcome message
            welcome_msg = LiquidityRiskWebSocketMessage(
                type=LiquidityRiskEventType.SNAPSHOT,
                topic="system.welcome",
                sequence_number=0,
                timestamp=datetime.utcnow(),
                payload={
                    "message": "Connected to Liquidity-Risk Translator WebSocket",
                    "connection_id": connection_id,
                    "user_id": user_id,
                    "available_topics": [topic.value for topic in LiquidityRiskTopic]
                }
            )
            
            await websocket.send_text(welcome_msg.to_json())
            logger.info(f"New WebSocket connection: {connection_id} (user: {user_id})")
            
        except Exception as e:
            logger.error(f"Error accepting WebSocket connection: {e}")
            if websocket.client_state.value < 3:  # Not closed
                await websocket.close(code=1011)
    
    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection."""
        try:
            if connection_id in self.active_connections:
                connection = self.active_connections[connection_id]
                
                # Remove from all topic subscriptions
                for topic in connection.subscriptions:
                    self.topic_subscribers[topic].discard(connection_id)
                
                # Close websocket
                if connection.websocket.client_state.value < 3:  # Not closed
                    await connection.websocket.close()
                
                # Remove connection
                del self.active_connections[connection_id]
                
                logger.info(f"WebSocket disconnected: {connection_id}")
                
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def subscribe(self, connection_id: str, topic: str):
        """Subscribe connection to a topic."""
        try:
            if connection_id not in self.active_connections:
                logger.warning(f"Connection {connection_id} not found for subscription")
                return False
            
            connection = self.active_connections[connection_id]
            
            # Validate topic format
            if not self._validate_topic(topic):
                logger.warning(f"Invalid topic format: {topic}")
                return False
            
            # Add subscription
            connection.subscriptions.add(topic)
            self.topic_subscribers[topic].add(connection_id)
            
            # Send subscription confirmation
            confirm_msg = LiquidityRiskWebSocketMessage(
                type=LiquidityRiskEventType.SNAPSHOT,
                topic=topic,
                sequence_number=self._get_next_sequence(topic),
                timestamp=datetime.utcnow(),
                payload={
                    "action": "subscribed",
                    "topic": topic,
                    "message": f"Successfully subscribed to {topic}"
                }
            )
            
            await self._send_to_connection(connection_id, confirm_msg)
            logger.info(f"Connection {connection_id} subscribed to {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing {connection_id} to {topic}: {e}")
            return False
    
    async def unsubscribe(self, connection_id: str, topic: str):
        """Unsubscribe connection from a topic."""
        try:
            if connection_id not in self.active_connections:
                return False
            
            connection = self.active_connections[connection_id]
            
            # Remove subscription
            connection.subscriptions.discard(topic)
            self.topic_subscribers[topic].discard(connection_id)
            
            # Send unsubscription confirmation
            confirm_msg = LiquidityRiskWebSocketMessage(
                type=LiquidityRiskEventType.SNAPSHOT,
                topic=topic,
                sequence_number=self._get_next_sequence(topic),
                timestamp=datetime.utcnow(),
                payload={
                    "action": "unsubscribed",
                    "topic": topic,
                    "message": f"Successfully unsubscribed from {topic}"
                }
            )
            
            await self._send_to_connection(connection_id, confirm_msg)
            logger.info(f"Connection {connection_id} unsubscribed from {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Error unsubscribing {connection_id} from {topic}: {e}")
            return False
    
    async def broadcast_liquidity_risk_update(self, 
                                           isin: str,
                                           translation: LiquidityRiskTranslation,
                                           event_type: LiquidityRiskEventType = LiquidityRiskEventType.SNAPSHOT):
        """Broadcast liquidity-risk update to all subscribers."""
        try:
            topic = f"lr.{isin}"
            
            if topic not in self.topic_subscribers:
                logger.debug(f"No subscribers for topic {topic}")
                return
            
            # Create message payload based on event type
            if event_type == LiquidityRiskEventType.SNAPSHOT:
                payload = self._create_snapshot_payload(translation)
            elif event_type == LiquidityRiskEventType.RISK_UPDATE:
                payload = self._create_risk_update_payload(translation)
            elif event_type == LiquidityRiskEventType.LIQUIDITY_UPDATE:
                payload = self._create_liquidity_update_payload(translation)
            elif event_type == LiquidityRiskEventType.EXIT_PATH_UPDATE:
                payload = self._create_exit_path_update_payload(translation)
            else:
                logger.warning(f"Unknown event type: {event_type}")
                return
            
            # Create message
            message = LiquidityRiskWebSocketMessage(
                type=event_type,
                topic=topic,
                sequence_number=self._get_next_sequence(topic),
                timestamp=datetime.utcnow(),
                payload=payload,
                meta={
                    "isin": isin,
                    "data_freshness": translation.data_freshness,
                    "confidence": translation.confidence_overall
                }
            )
            
            # Broadcast to all subscribers
            await self._broadcast_to_topic(topic, message)
            
            # Also publish to Redis for cross-instance communication
            if self.redis_client:
                await self.redis_client.publish(
                    f"liquidity_risk:{topic}",
                    message.to_json()
                )
            
            logger.debug(f"Broadcasted {event_type.value} for {isin} to {len(self.topic_subscribers[topic])} subscribers")
            
        except Exception as e:
            logger.error(f"Error broadcasting liquidity-risk update: {e}")
    
    def _create_snapshot_payload(self, translation: LiquidityRiskTranslation) -> Dict[str, Any]:
        """Create snapshot payload for full liquidity-risk translation."""
        return {
            "isin": translation.isin,
            "as_of": translation.as_of.isoformat(),
            "risk_summary": {
                "overall_score": translation.risk_summary.overall_score,
                "confidence": translation.risk_summary.confidence,
                "categories": [
                    {
                        "name": cat.name,
                        "score_0_100": cat.score_0_100,
                        "level": cat.level
                    }
                    for cat in translation.risk_summary.categories
                ]
            },
            "liquidity_profile": {
                "liquidity_index": translation.liquidity_profile.liquidity_index,
                "spread_bps": translation.liquidity_profile.spread_bps,
                "liquidity_level": translation.liquidity_profile.liquidity_level.value,
                "expected_time_to_exit_minutes": translation.liquidity_profile.expected_time_to_exit_minutes
            },
            "exit_recommendations": [
                {
                    "path": rec.path.value,
                    "priority": rec.priority.value,
                    "fill_probability": rec.fill_probability,
                    "expected_time_to_exit_minutes": rec.expected_time_to_exit_minutes
                }
                for rec in translation.exit_recommendations
            ],
            "retail_narrative": translation.retail_narrative,
            "confidence_overall": translation.confidence_overall,
            "data_freshness": translation.data_freshness
        }
    
    def _create_risk_update_payload(self, translation: LiquidityRiskTranslation) -> Dict[str, Any]:
        """Create risk update payload for risk score changes."""
        return {
            "isin": translation.isin,
            "risk_summary": {
                "overall_score": translation.risk_summary.overall_score,
                "confidence": translation.risk_summary.confidence,
                "categories": [
                    {
                        "name": cat.name,
                        "score_0_100": cat.score_0_100,
                        "level": cat.level,
                        "change": "updated"  # In real implementation, track changes
                    }
                    for cat in translation.risk_summary.categories
                ]
            },
            "risk_warnings": translation.risk_warnings
        }
    
    def _create_liquidity_update_payload(self, translation: LiquidityRiskTranslation) -> Dict[str, Any]:
        """Create liquidity update payload for liquidity profile changes."""
        return {
            "isin": translation.isin,
            "liquidity_profile": {
                "liquidity_index": translation.liquidity_profile.liquidity_index,
                "spread_bps": translation.liquidity_profile.spread_bps,
                "liquidity_level": translation.liquidity_profile.liquidity_level.value,
                "expected_time_to_exit_minutes": translation.liquidity_profile.expected_time_to_exit_minutes,
                "depth_score": translation.liquidity_profile.depth_score,
                "turnover_rank": translation.liquidity_profile.turnover_rank
            },
            "data_freshness": translation.liquidity_profile.data_freshness
        }
    
    def _create_exit_path_update_payload(self, translation: LiquidityRiskTranslation) -> Dict[str, Any]:
        """Create exit path update payload for exit pathway changes."""
        return {
            "isin": translation.isin,
            "exit_recommendations": [
                {
                    "path": rec.path.value,
                    "priority": rec.priority.value,
                    "fill_probability": rec.fill_probability,
                    "expected_time_to_exit_minutes": rec.expected_time_to_exit_minutes,
                    "expected_spread_bps": rec.expected_spread_bps,
                    "constraints": [c.value for c in rec.constraints]
                }
                for rec in translation.exit_recommendations
            ],
            "best_path": translation.exit_recommendations[0].path.value if translation.exit_recommendations else None
        }
    
    async def broadcast_alert(self, isin: str, alert_type: str, message: str, severity: str = "info"):
        """Broadcast alert message to subscribers."""
        try:
            topic = f"lr.{isin}.alerts"
            
            alert_msg = LiquidityRiskWebSocketMessage(
                type=LiquidityRiskEventType.ALERT,
                topic=topic,
                sequence_number=self._get_next_sequence(topic),
                timestamp=datetime.utcnow(),
                payload={
                    "alert_type": alert_type,
                    "message": message,
                    "severity": severity,
                    "isin": isin,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            await self._broadcast_to_topic(topic, alert_msg)
            logger.info(f"Broadcasted alert for {isin}: {alert_type} - {message}")
            
        except Exception as e:
            logger.error(f"Error broadcasting alert: {e}")
    
    async def _broadcast_to_topic(self, topic: str, message: LiquidityRiskWebSocketMessage):
        """Broadcast message to all subscribers of a topic."""
        if topic not in self.topic_subscribers:
            return
        
        subscribers = self.topic_subscribers[topic].copy()
        failed_connections = []
        
        for connection_id in subscribers:
            try:
                await self._send_to_connection(connection_id, message)
            except Exception as e:
                logger.warning(f"Failed to send to {connection_id}: {e}")
                failed_connections.append(connection_id)
        
        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)
    
    async def _send_to_connection(self, connection_id: str, message: LiquidityRiskWebSocketMessage):
        """Send message to a specific connection."""
        if connection_id not in self.active_connections:
            return
        
        connection = self.active_connections[connection_id]
        
        try:
            # Update last activity
            connection.last_activity = datetime.utcnow()
            
            # Send message
            await connection.websocket.send_text(message.to_json())
            
        except Exception as e:
            logger.error(f"Error sending to connection {connection_id}: {e}")
            raise
    
    def _get_next_sequence(self, topic: str) -> int:
        """Get next sequence number for a topic."""
        self.sequence_counters[topic] += 1
        return self.sequence_counters[topic]
    
    def _validate_topic(self, topic: str) -> bool:
        """Validate topic format."""
        # Basic validation for liquidity-risk topics
        if topic.startswith("lr."):
            return True
        return False
    
    def _get_user_permissions(self, user_id: Optional[str]) -> List[str]:
        """Get user permissions for WebSocket access."""
        if not user_id:
            return ["public"]
        
        # In a real implementation, this would check user roles and permissions
        # For now, return basic permissions
        return ["public", "authenticated"]
    
    async def start_heartbeat(self):
        """Start heartbeat to keep connections alive."""
        while True:
            try:
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                
                heartbeat_msg = LiquidityRiskWebSocketMessage(
                    type=LiquidityRiskEventType.HEARTBEAT,
                    topic="system.heartbeat",
                    sequence_number=0,
                    timestamp=datetime.utcnow(),
                    payload={
                        "timestamp": datetime.utcnow().isoformat(),
                        "active_connections": len(self.active_connections)
                    }
                )
                
                # Send heartbeat to all connections
                for connection_id in list(self.active_connections.keys()):
                    try:
                        await self._send_to_connection(connection_id, heartbeat_msg)
                    except Exception as e:
                        logger.debug(f"Heartbeat failed for {connection_id}: {e}")
                        # Connection will be cleaned up on next message attempt
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
    
    async def cleanup_stale_connections(self):
        """Clean up stale connections."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.utcnow()
                stale_connections = []
                
                for connection_id, connection in self.active_connections.items():
                    # Mark as stale if no activity for 5 minutes
                    if (current_time - connection.last_activity).total_seconds() > 300:
                        stale_connections.append(connection_id)
                
                # Disconnect stale connections
                for connection_id in stale_connections:
                    await self.disconnect(connection_id)
                
                if stale_connections:
                    logger.info(f"Cleaned up {len(stale_connections)} stale connections")
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "active_connections": len(self.active_connections),
            "topic_subscriptions": {
                topic: len(subscribers) 
                for topic, subscribers in self.topic_subscribers.items()
            },
            "total_subscriptions": sum(len(subscribers) for subscribers in self.topic_subscribers.values())
        }
