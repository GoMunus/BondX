"""
Liquidity Pulse WebSocket Manager

This module provides WebSocket functionality for real-time liquidity pulse updates,
including snapshot, delta, and forecast updates with coalescing and rate limiting.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import weakref

from fastapi import WebSocket
from ...core.logging import get_logger
from ...api.v1.schemas_liquidity import LiquidityPulse, PulseWebSocketMessage
from ...liquidity.pulse import LiquidityPulseEngine

logger = get_logger(__name__)

class PulseMessageType(str, Enum):
    """Types of pulse WebSocket messages."""
    SNAPSHOT = "snapshot"
    DELTA = "delta"
    FORECAST_UPDATE = "forecast_update"
    ALERT = "alert"
    HEARTBEAT = "heartbeat"

class PulseAlertLevel(str, Enum):
    """Alert levels for pulse deterioration."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class PulseSubscription:
    """Subscription information for a pulse topic."""
    websocket: WebSocket
    isin: str
    user_id: Optional[str] = None
    view_type: str = "professional"
    last_heartbeat: datetime = field(default_factory=datetime.now)
    message_count: int = 0
    is_active: bool = True

@dataclass
class PulseUpdate:
    """Pulse update with metadata."""
    isin: str
    timestamp: datetime
    message_type: PulseMessageType
    payload: Any
    sequence_number: int
    correlation_id: Optional[str] = None

class LiquidityPulseWebSocketManager:
    """Manages WebSocket connections for liquidity pulse streaming."""
    
    def __init__(self, pulse_engine: LiquidityPulseEngine, config: Dict[str, Any]):
        self.pulse_engine = pulse_engine
        self.config = config
        self.logger = get_logger(__name__)
        
        # WebSocket management
        self.active_connections: Dict[str, PulseSubscription] = {}
        self.isin_subscriptions: Dict[str, Set[str]] = {}  # isin -> set of connection_ids
        
        # Message handling
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.sequence_counters: Dict[str, int] = {}  # isin -> sequence number
        
        # Configuration
        self.heartbeat_interval = config.get("heartbeat_interval", 30)  # seconds
        self.coalescing_interval = config.get("coalescing_interval", 0.5)  # seconds
        self.max_message_rate = config.get("max_message_rate", 100)  # messages per minute
        self.alert_thresholds = config.get("alert_thresholds", {
            "liquidity_deterioration": 10.0,  # 10 point drop
            "uncertainty_increase": 0.2,       # 20% increase
            "freshness_degradation": 300       # 5 minutes
        })
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.is_running = False
        
        self.logger.info("Liquidity Pulse WebSocket Manager initialized")
    
    async def start(self):
        """Start the WebSocket manager."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        message_processor_task = asyncio.create_task(self._message_processor_loop())
        
        self.background_tasks.add(heartbeat_task)
        self.background_tasks.add(message_processor_task)
        
        self.logger.info("Liquidity Pulse WebSocket Manager started")
    
    async def stop(self):
        """Stop the WebSocket manager."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Close all connections
        for connection_id in list(self.active_connections.keys()):
            await self._close_connection(connection_id)
        
        self.logger.info("Liquidity Pulse WebSocket Manager stopped")
    
    async def connect(self, websocket: WebSocket, isin: str, user_id: Optional[str] = None, view_type: str = "professional"):
        """Handle new WebSocket connection."""
        try:
            await websocket.accept()
            
            # Generate connection ID
            connection_id = f"pulse_{isin}_{user_id}_{datetime.now().timestamp()}"
            
            # Create subscription
            subscription = PulseSubscription(
                websocket=websocket,
                isin=isin,
                user_id=user_id,
                view_type=view_type
            )
            
            # Store connection
            self.active_connections[connection_id] = subscription
            
            # Add to ISIN subscriptions
            if isin not in self.isin_subscriptions:
                self.isin_subscriptions[isin] = set()
            self.isin_subscriptions[isin].add(connection_id)
            
            # Send initial snapshot
            await self._send_snapshot(connection_id, isin)
            
            self.logger.info(f"New pulse WebSocket connection: {connection_id} for {isin}")
            
            return connection_id
            
        except Exception as e:
            self.logger.error(f"Error accepting WebSocket connection: {e}")
            if websocket.client_state.value < 3:  # Not closed
                await websocket.close(code=1011, reason="Internal error")
            raise
    
    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection."""
        try:
            if connection_id in self.active_connections:
                subscription = self.active_connections[connection_id]
                isin = subscription.isin
                
                # Remove from ISIN subscriptions
                if isin in self.isin_subscriptions:
                    self.isin_subscriptions[isin].discard(connection_id)
                    if not self.isin_subscriptions[isin]:
                        del self.isin_subscriptions[isin]
                
                # Remove connection
                del self.active_connections[connection_id]
                
                self.logger.info(f"Pulse WebSocket disconnected: {connection_id}")
                
        except Exception as e:
            self.logger.error(f"Error handling disconnection for {connection_id}: {e}")
    
    async def broadcast_pulse_update(self, isin: str, pulse: LiquidityPulse, message_type: PulseMessageType = PulseMessageType.DELTA):
        """Broadcast pulse update to all subscribers."""
        try:
            if isin not in self.isin_subscriptions:
                return
            
            # Increment sequence number
            if isin not in self.sequence_counters:
                self.sequence_counters[isin] = 0
            self.sequence_counters[isin] += 1
            
            # Create update
            update = PulseUpdate(
                isin=isin,
                timestamp=datetime.now(),
                message_type=message_type,
                payload=pulse,
                sequence_number=self.sequence_counters[isin]
            )
            
            # Add to message queue
            await self.message_queue.put(update)
            
        except Exception as e:
            self.logger.error(f"Error broadcasting pulse update for {isin}: {e}")
    
    async def send_alert(self, isin: str, alert_level: PulseAlertLevel, message: str, data: Optional[Dict[str, Any]] = None):
        """Send alert to subscribers for an ISIN."""
        try:
            if isin not in self.isin_subscriptions:
                return
            
            # Create alert payload
            alert_payload = {
                "level": alert_level.value,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "data": data or {}
            }
            
            # Create update
            update = PulseUpdate(
                isin=isin,
                timestamp=datetime.now(),
                message_type=PulseMessageType.ALERT,
                payload=alert_payload,
                sequence_number=self.sequence_counters.get(isin, 0) + 1
            )
            
            # Add to message queue
            await self.message_queue.put(update)
            
        except Exception as e:
            self.logger.error(f"Error sending alert for {isin}: {e}")
    
    async def _send_snapshot(self, connection_id: str, isin: str):
        """Send initial snapshot to a new connection."""
        try:
            if connection_id not in self.active_connections:
                return
            
            subscription = self.active_connections[connection_id]
            
            # Calculate current pulse
            result = await self.pulse_engine.calculate_pulse(
                isin=isin,
                mode="fast",
                include_forecast=True,
                include_drivers=True
            )
            
            if result.success:
                pulse = self.pulse_engine.convert_to_liquidity_pulse(result)
                
                # Apply role-based filtering
                if subscription.view_type == "retail":
                    pulse.drivers = [d for d in pulse.drivers if d.source not in ["microstructure", "auction_mm"]]
                    pulse.missing_signals = []
                    pulse.uncertainty = min(pulse.uncertainty, 0.5)
                
                # Create snapshot message
                snapshot_message = PulseWebSocketMessage(
                    type=PulseMessageType.SNAPSHOT.value,
                    isin=isin,
                    sequence_number=0,
                    timestamp=datetime.now(),
                    payload=pulse.dict(),
                    correlation_id=f"snapshot_{connection_id}"
                )
                
                # Send snapshot
                await subscription.websocket.send_text(snapshot_message.json())
                
                self.logger.debug(f"Sent snapshot to {connection_id}")
                
        except Exception as e:
            self.logger.error(f"Error sending snapshot to {connection_id}: {e}")
    
    async def _send_message(self, connection_id: str, message: PulseWebSocketMessage):
        """Send message to a specific connection."""
        try:
            if connection_id not in self.active_connections:
                return
            
            subscription = self.active_connections[connection_id]
            
            # Check message rate limit
            if subscription.message_count > self.max_message_rate:
                self.logger.warning(f"Message rate limit exceeded for {connection_id}")
                return
            
            # Send message
            await subscription.websocket.send_text(message.json())
            
            # Update counters
            subscription.message_count += 1
            subscription.last_heartbeat = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error sending message to {connection_id}: {e}")
            # Mark connection as inactive
            subscription.is_active = False
    
    async def _close_connection(self, connection_id: str):
        """Close a WebSocket connection."""
        try:
            if connection_id in self.active_connections:
                subscription = self.active_connections[connection_id]
                
                if subscription.websocket.client_state.value < 3:  # Not closed
                    await subscription.websocket.close(code=1000, reason="Normal closure")
                
        except Exception as e:
            self.logger.error(f"Error closing connection {connection_id}: {e}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to all connections."""
        while self.is_running:
            try:
                now = datetime.now()
                
                # Send heartbeats
                for connection_id, subscription in list(self.active_connections.items()):
                    try:
                        if not subscription.is_active:
                            continue
                        
                        # Check if connection is stale
                        if (now - subscription.last_heartbeat).total_seconds() > self.heartbeat_interval * 2:
                            self.logger.warning(f"Connection {connection_id} is stale, closing")
                            await self.disconnect(connection_id)
                            continue
                        
                        # Send heartbeat
                        heartbeat_message = PulseWebSocketMessage(
                            type=PulseMessageType.HEARTBEAT.value,
                            isin=subscription.isin,
                            sequence_number=0,
                            timestamp=now,
                            payload={"status": "alive"}
                        )
                        
                        await self._send_message(connection_id, heartbeat_message)
                        
                    except Exception as e:
                        self.logger.error(f"Error sending heartbeat to {connection_id}: {e}")
                        subscription.is_active = False
                
                # Reset message counters
                for subscription in self.active_connections.values():
                    subscription.message_count = 0
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)
    
    async def _message_processor_loop(self):
        """Process messages from the queue and send to subscribers."""
        while self.is_running:
            try:
                # Get message from queue
                try:
                    update = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process update
                await self._process_update(update)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in message processor loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_update(self, update: PulseUpdate):
        """Process a single update and send to relevant subscribers."""
        try:
            isin = update.isin
            
            if isin not in self.isin_subscriptions:
                return
            
            # Create WebSocket message
            ws_message = PulseWebSocketMessage(
                type=update.message_type.value,
                isin=isin,
                sequence_number=update.sequence_number,
                timestamp=update.timestamp,
                payload=update.payload,
                correlation_id=update.correlation_id
            )
            
            # Send to all subscribers
            for connection_id in list(self.isin_subscriptions[isin]):
                if connection_id in self.active_connections:
                    subscription = self.active_connections[connection_id]
                    
                    if subscription.is_active:
                        # Apply role-based filtering for retail users
                        if subscription.view_type == "retail" and update.message_type == PulseMessageType.DELTA:
                            # Filter sensitive data for retail users
                            filtered_payload = self._filter_payload_for_retail(update.payload)
                            ws_message.payload = filtered_payload
                        
                        await self._send_message(connection_id, ws_message)
                        
                        # Reset message counter for this connection
                        subscription.message_count = 0
                
        except Exception as e:
            self.logger.error(f"Error processing update for {update.isin}: {e}")
    
    def _filter_payload_for_retail(self, payload: Any) -> Any:
        """Filter payload to remove sensitive information for retail users."""
        try:
            if isinstance(payload, dict):
                filtered = payload.copy()
                
                # Remove sensitive fields
                sensitive_fields = ["missing_signals", "uncertainty", "inputs_hash", "model_versions"]
                for field in sensitive_fields:
                    if field in filtered:
                        del filtered[field]
                
                # Filter drivers to remove microstructure and auction/MM sources
                if "drivers" in filtered:
                    filtered["drivers"] = [
                        d for d in filtered["drivers"] 
                        if d.get("source") not in ["microstructure", "auction_mm"]
                    ]
                
                return filtered
            
            return payload
            
        except Exception as e:
            self.logger.debug(f"Error filtering payload for retail: {e}")
            return payload
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about active connections."""
        try:
            total_connections = len(self.active_connections)
            active_connections = sum(1 for s in self.active_connections.values() if s.is_active)
            
            isin_stats = {}
            for isin, connections in self.isin_subscriptions.items():
                isin_stats[isin] = {
                    "subscriber_count": len(connections),
                    "active_subscribers": sum(1 for c in connections if c in self.active_connections and self.active_connections[c].is_active)
                }
            
            return {
                "total_connections": total_connections,
                "active_connections": active_connections,
                "isin_subscriptions": isin_stats,
                "message_queue_size": self.message_queue.qsize(),
                "sequence_counters": self.sequence_counters
            }
            
        except Exception as e:
            self.logger.error(f"Error getting connection stats: {e}")
            return {"error": str(e)}
    
    async def check_pulse_alerts(self, isin: str, pulse: LiquidityPulse):
        """Check if pulse changes warrant alerts."""
        try:
            # Get previous pulse for comparison
            # This would typically come from a cache or database
            # For now, we'll use basic thresholds
            
            alerts = []
            
            # Check liquidity deterioration
            # This would compare with previous value
            # For demonstration, we'll use a simple threshold
            
            # Check uncertainty increase
            if pulse.uncertainty > 0.8:
                alerts.append({
                    "level": PulseAlertLevel.WARNING,
                    "message": f"High uncertainty detected for {isin}",
                    "data": {"uncertainty": pulse.uncertainty}
                })
            
            # Check data freshness
            if pulse.freshness.value == "stale":
                alerts.append({
                    "level": PulseAlertLevel.WARNING,
                    "message": f"Data freshness degraded for {isin}",
                    "data": {"freshness": pulse.freshness.value}
                })
            
            # Send alerts if any
            for alert in alerts:
                await self.send_alert(
                    isin=isin,
                    alert_level=alert["level"],
                    message=alert["message"],
                    data=alert["data"]
                )
                
        except Exception as e:
            self.logger.error(f"Error checking pulse alerts for {isin}: {e}")

# Factory function to create WebSocket manager
def create_pulse_websocket_manager(pulse_engine: LiquidityPulseEngine, config: Dict[str, Any]) -> LiquidityPulseWebSocketManager:
    """Create and configure a pulse WebSocket manager."""
    return LiquidityPulseWebSocketManager(pulse_engine, config)
