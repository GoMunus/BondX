"""
WebSocket Monitoring and Metrics Configuration.

This module provides monitoring, metrics, and alerting for the WebSocket system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, 
    start_http_server, generate_latest
)
import redis.asyncio as redis

from bondx.core.config import settings
from bondx.core.logging import get_logger
from bondx.websocket.unified_websocket_manager import UnifiedWebSocketManager

logger = get_logger(__name__)

# Prometheus metrics
class WebSocketMetrics:
    """WebSocket metrics collection."""
    
    def __init__(self):
        # Connection metrics
        self.connections_total = Gauge(
            'websocket_connections_total',
            'Total number of WebSocket connections',
            ['status']
        )
        
        self.connections_active = Gauge(
            'websocket_connections_active',
            'Number of active WebSocket connections'
        )
        
        self.connections_established = Counter(
            'websocket_connections_established_total',
            'Total number of WebSocket connections established'
        )
        
        self.connections_closed = Counter(
            'websocket_connections_closed_total',
            'Total number of WebSocket connections closed'
        )
        
        # Message metrics
        self.messages_sent = Counter(
            'websocket_messages_sent_total',
            'Total number of messages sent',
            ['topic', 'type']
        )
        
        self.messages_received = Counter(
            'websocket_messages_received_total',
            'Total number of messages received',
            ['topic', 'type']
        )
        
        self.messages_dropped = Counter(
            'websocket_messages_dropped_total',
            'Total number of messages dropped',
            ['reason']
        )
        
        # Performance metrics
        self.message_latency = Histogram(
            'websocket_message_latency_seconds',
            'Message latency in seconds',
            ['topic', 'type'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        
        self.queue_size = Gauge(
            'websocket_queue_size',
            'Current queue size',
            ['queue_type']
        )
        
        self.processing_time = Histogram(
            'websocket_processing_time_seconds',
            'Message processing time in seconds',
            ['operation'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        )
        
        # Error metrics
        self.errors_total = Counter(
            'websocket_errors_total',
            'Total number of errors',
            ['type', 'topic']
        )
        
        self.rate_limit_violations = Counter(
            'websocket_rate_limit_violations_total',
            'Total number of rate limit violations',
            ['topic', 'user_id']
        )
        
        # Subscription metrics
        self.subscriptions_total = Gauge(
            'websocket_subscriptions_total',
            'Total number of subscriptions',
            ['topic']
        )
        
        self.subscription_changes = Counter(
            'websocket_subscription_changes_total',
            'Total number of subscription changes',
            ['operation', 'topic']
        )

# Alert thresholds
@dataclass
class AlertThresholds:
    """Alert threshold configuration."""
    max_connection_failure_rate: float = 0.05  # 5%
    max_message_drop_rate: float = 0.01        # 1%
    max_latency_p95: float = 0.1               # 100ms
    max_queue_size: int = 1000
    max_error_rate: float = 0.02               # 2%
    min_connection_success_rate: float = 0.95  # 95%

# Alert types
class AlertType(Enum):
    """Types of alerts."""
    HIGH_CONNECTION_FAILURE = "high_connection_failure"
    HIGH_MESSAGE_DROP_RATE = "high_message_drop_rate"
    HIGH_LATENCY = "high_latency"
    QUEUE_OVERFLOW = "queue_overflow"
    HIGH_ERROR_RATE = "high_error_rate"
    LOW_CONNECTION_SUCCESS = "low_connection_success"
    REDIS_CONNECTION_ISSUE = "redis_connection_issue"

@dataclass
class Alert:
    """Alert information."""
    type: AlertType
    severity: str
    message: str
    timestamp: datetime
    details: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class WebSocketMonitor:
    """WebSocket system monitor."""
    
    def __init__(self, websocket_manager: UnifiedWebSocketManager):
        self.websocket_manager = websocket_manager
        self.metrics = WebSocketMetrics()
        self.thresholds = AlertThresholds()
        
        # Alert tracking
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Performance tracking
        self.performance_window = 300  # 5 minutes
        self.performance_data: Dict[str, List[float]] = {
            'latency': [],
            'throughput': [],
            'error_rate': [],
            'connection_rate': []
        }
        
        # Monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info("WebSocket monitor initialized")
    
    async def start(self) -> None:
        """Start the monitoring system."""
        if self.is_running:
            logger.warning("WebSocket monitor is already running")
            return
        
        try:
            # Start Prometheus metrics server
            start_http_server(8001)
            logger.info("Prometheus metrics server started on port 8001")
            
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.is_running = True
            logger.info("WebSocket monitor started successfully")
            
        except Exception as e:
            logger.error(f"Error starting WebSocket monitor: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the monitoring system."""
        if not self.is_running:
            logger.warning("WebSocket monitor is not running")
            return
        
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("WebSocket monitor stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Collect metrics
                await self._collect_metrics()
                
                # Check thresholds
                await self._check_thresholds()
                
                # Update performance data
                await self._update_performance_data()
                
                # Generate alerts
                await self._generate_alerts()
                
                # Wait for next cycle
                await asyncio.sleep(30)  # 30 second intervals
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _collect_metrics(self) -> None:
        """Collect metrics from WebSocket manager."""
        try:
            stats = await self.websocket_manager.get_statistics()
            
            # Update connection metrics
            self.metrics.connections_total.labels(status='total').set(stats['connections']['total'])
            self.metrics.connections_active.set(stats['connections']['active'])
            self.metrics.connections_total.labels(status='established').set(stats['connections']['established'])
            self.metrics.connections_total.labels(status='closed').set(stats['connections']['closed'])
            
            # Update message metrics
            self.metrics.messages_sent.labels(topic='all', type='all').inc(stats['messages']['sent'])
            self.metrics.messages_dropped.labels(reason='backpressure').inc(stats['messages']['dropped'])
            
            # Update queue metrics
            self.metrics.queue_size.labels(queue_type='message').set(stats['messages']['queue_size'])
            self.metrics.queue_size.labels(queue_type='broadcast').set(stats['messages']['broadcast_queue_size'])
            
            # Update subscription metrics
            topics = await self.websocket_manager.list_topics()
            for topic_info in topics:
                self.metrics.subscriptions_total.labels(topic=topic_info['topic']).set(
                    topic_info['subscriber_count']
                )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            self.metrics.errors_total.labels(type='collection_error', topic='system').inc()
    
    async def _check_thresholds(self) -> None:
        """Check metrics against thresholds."""
        try:
            stats = await self.websocket_manager.get_statistics()
            
            # Check connection success rate
            total_connections = stats['connections']['established'] + stats['connections']['closed']
            if total_connections > 0:
                success_rate = stats['connections']['established'] / total_connections
                if success_rate < self.thresholds.min_connection_success_rate:
                    await self._create_alert(
                        AlertType.LOW_CONNECTION_SUCCESS,
                        "high",
                        f"Connection success rate {success_rate:.2%} below threshold {self.thresholds.min_connection_success_rate:.2%}",
                        {"current_rate": success_rate, "threshold": self.thresholds.min_connection_success_rate}
                    )
            
            # Check message drop rate
            total_messages = stats['messages']['sent'] + stats['messages']['dropped']
            if total_messages > 0:
                drop_rate = stats['messages']['dropped'] / total_messages
                if drop_rate > self.thresholds.max_message_drop_rate:
                    await self._create_alert(
                        AlertType.HIGH_MESSAGE_DROP_RATE,
                        "high",
                        f"Message drop rate {drop_rate:.2%} above threshold {self.thresholds.max_message_drop_rate:.2%}",
                        {"current_rate": drop_rate, "threshold": self.thresholds.max_message_drop_rate}
                    )
            
            # Check queue sizes
            if stats['messages']['queue_size'] > self.thresholds.max_queue_size:
                await self._create_alert(
                    AlertType.QUEUE_OVERFLOW,
                    "medium",
                    f"Message queue size {stats['messages']['queue_size']} above threshold {self.thresholds.max_queue_size}",
                    {"current_size": stats['messages']['queue_size'], "threshold": self.thresholds.max_queue_size}
                )
            
            if stats['messages']['broadcast_queue_size'] > self.thresholds.max_queue_size:
                await self._create_alert(
                    AlertType.QUEUE_OVERFLOW,
                    "medium",
                    f"Broadcast queue size {stats['messages']['broadcast_queue_size']} above threshold {self.thresholds.max_queue_size}",
                    {"current_size": stats['messages']['broadcast_queue_size'], "threshold": self.thresholds.max_queue_size}
                )
            
        except Exception as e:
            logger.error(f"Error checking thresholds: {e}")
    
    async def _update_performance_data(self) -> None:
        """Update performance tracking data."""
        try:
            stats = await self.websocket_manager.get_statistics()
            
            # Calculate current performance metrics
            uptime = stats['performance']['uptime_seconds']
            if uptime > 0:
                throughput = stats['messages']['sent'] / uptime
                self.performance_data['throughput'].append(throughput)
                
                # Keep only recent data
                if len(self.performance_data['throughput']) > 60:  # 5 minutes at 5-second intervals
                    self.performance_data['throughput'].pop(0)
            
        except Exception as e:
            logger.error(f"Error updating performance data: {e}")
    
    async def _generate_alerts(self) -> None:
        """Generate and manage alerts."""
        try:
            current_time = datetime.utcnow()
            
            # Check for resolved alerts
            resolved_alerts = []
            for alert_id, alert in self.active_alerts.items():
                if alert.resolved:
                    resolved_alerts.append(alert_id)
                    alert.resolved_at = current_time
                    self.alert_history.append(alert)
            
            # Remove resolved alerts
            for alert_id in resolved_alerts:
                del self.active_alerts[alert_id]
            
            # Log active alerts
            if self.active_alerts:
                logger.warning(f"Active alerts: {len(self.active_alerts)}")
                for alert in self.active_alerts.values():
                    logger.warning(f"Alert: {alert.type.value} - {alert.message}")
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
    
    async def _create_alert(self, alert_type: AlertType, severity: str, message: str, details: Dict[str, Any]) -> None:
        """Create a new alert."""
        try:
            alert_id = f"{alert_type.value}_{datetime.utcnow().isoformat()}"
            
            # Check if similar alert already exists
            if alert_type.value in [a.type.value for a in self.active_alerts.values()]:
                return  # Don't create duplicate alerts
            
            alert = Alert(
                type=alert_type,
                severity=severity,
                message=message,
                timestamp=datetime.utcnow(),
                details=details
            )
            
            self.active_alerts[alert_id] = alert
            
            # Log alert
            logger.warning(f"Alert created: {alert_type.value} - {message}")
            
            # TODO: Send alert to external systems (Slack, email, etc.)
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        try:
            stats = await self.websocket_manager.get_statistics()
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "connections": {
                    "total": stats['connections']['total'],
                    "active": stats['connections']['active'],
                    "success_rate": stats['connections']['established'] / max(1, stats['connections']['established'] + stats['connections']['closed'])
                },
                "messages": {
                    "sent": stats['messages']['sent'],
                    "dropped": stats['messages']['dropped'],
                    "drop_rate": stats['messages']['dropped'] / max(1, stats['messages']['sent'] + stats['messages']['dropped'])
                },
                "performance": {
                    "uptime_seconds": stats['performance']['uptime_seconds'],
                    "messages_per_second": stats['performance']['messages_per_second']
                },
                "alerts": {
                    "active": len(self.active_alerts),
                    "total": len(self.alert_history)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {}
    
    async def resolve_alert(self, alert_type: AlertType) -> bool:
        """Mark an alert as resolved."""
        try:
            for alert in self.active_alerts.values():
                if alert.type == alert_type:
                    alert.resolved = True
                    logger.info(f"Alert resolved: {alert_type.value}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return False

# Health check functions
async def check_websocket_health(websocket_manager: UnifiedWebSocketManager) -> Dict[str, Any]:
    """Check WebSocket system health."""
    try:
        stats = await websocket_manager.get_statistics()
        
        # Calculate health indicators
        total_connections = stats['connections']['established'] + stats['connections']['closed']
        connection_success_rate = stats['connections']['established'] / max(1, total_connections)
        
        total_messages = stats['messages']['sent'] + stats['messages']['dropped']
        message_success_rate = stats['messages']['sent'] / max(1, total_messages)
        
        # Determine overall health
        if connection_success_rate >= 0.95 and message_success_rate >= 0.99:
            health_status = "healthy"
        elif connection_success_rate >= 0.90 and message_success_rate >= 0.95:
            health_status = "degraded"
        else:
            health_status = "unhealthy"
        
        return {
            "status": health_status,
            "timestamp": datetime.utcnow().isoformat(),
            "indicators": {
                "connection_success_rate": connection_success_rate,
                "message_success_rate": message_success_rate,
                "active_connections": stats['connections']['active'],
                "queue_health": {
                    "message_queue": stats['messages']['queue_size'] < 1000,
                    "broadcast_queue": stats['messages']['broadcast_queue_size'] < 1000
                }
            },
            "details": stats
        }
        
    except Exception as e:
        logger.error(f"Error checking WebSocket health: {e}")
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

# Export
__all__ = [
    "WebSocketMetrics",
    "WebSocketMonitor", 
    "AlertThresholds",
    "AlertType",
    "Alert",
    "check_websocket_health"
]
