"""
WebSocket Router for BondX Real-time Streaming API.

This module provides unified WebSocket endpoints for:
- Market data streaming (L1/L2, trades, aggregates)
- Auction updates and bid histograms
- Trading notifications and order status
- Risk management alerts and portfolio updates
- Mobile-optimized multiplexed streams
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, ValidationError
import redis.asyncio as redis

from ...core.config import settings
from ...core.logging import get_logger
from ...websocket.unified_websocket_manager import (
    UnifiedWebSocketManager, 
    WebSocketConnection, 
    WebSocketMessage,
    MessageType
)

logger = get_logger(__name__)

# Create WebSocket router
router = APIRouter(prefix="/ws", tags=["websocket"])

# Security
security = HTTPBearer(auto_error=False)

# Message models
class SubscriptionRequest(BaseModel):
    """Client subscription request model."""
    topic: str = Field(..., description="Topic to subscribe to")
    subtopics: Optional[Dict[str, Any]] = Field(None, description="Optional subtopic filters")
    resume_from_seq: Optional[int] = Field(None, description="Resume from sequence number")
    compression: Optional[bool] = Field(True, description="Enable compression")
    batch_size: Optional[int] = Field(100, description="Batch size for updates")

class SubscriptionResponse(BaseModel):
    """Subscription confirmation response."""
    success: bool
    topic: str
    sequence_number: int
    message: str
    compression_enabled: bool
    rate_limits: Dict[str, Any]

class WebSocketError(BaseModel):
    """WebSocket error response."""
    error: str
    code: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime

# Rate limiting configuration
RATE_LIMITS = {
    "market": {"burst": 1000, "sustained": 100, "window": 60},
    "auction": {"burst": 500, "sustained": 50, "window": 60},
    "trading": {"burst": 200, "sustained": 20, "window": 60},
    "risk": {"burst": 100, "sustained": 10, "window": 60},
    "mobile": {"burst": 300, "sustained": 30, "window": 60},
}

# Connection metadata
@dataclass
class ConnectionMetadata:
    """Connection metadata for tracking and management."""
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

# Global WebSocket manager instance
websocket_manager: Optional[UnifiedWebSocketManager] = None

async def get_websocket_manager() -> UnifiedWebSocketManager:
    """Get the global WebSocket manager instance."""
    global websocket_manager
    if websocket_manager is None:
        websocket_manager = UnifiedWebSocketManager()
        await websocket_manager.start()
    return websocket_manager

async def authenticate_websocket(websocket: WebSocket) -> Optional[Dict[str, Any]]:
    """Authenticate WebSocket connection."""
    try:
        # Extract token from query parameters or headers
        token = websocket.query_params.get("token")
        if not token:
            # Try to get from headers
            auth_header = websocket.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]
        
        if not token:
            return None
        
        # TODO: Implement proper JWT validation
        # For now, mock authentication
        if token == "valid_token":
            return {
                "user_id": "user_123",
                "permissions": ["read", "write"],
                "device_type": "web"
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None

async def check_rate_limit(
    metadata: ConnectionMetadata, 
    topic: str, 
    manager: UnifiedWebSocketManager
) -> bool:
    """Check rate limits for a topic."""
    try:
        current_time = datetime.utcnow()
        topic_type = topic.split(".")[0] if "." in topic else "default"
        
        if topic_type not in RATE_LIMITS:
            return True
        
        limits = RATE_LIMITS[topic_type]
        
        # Initialize rate limit bucket if not exists
        if topic_type not in metadata.rate_limit_buckets:
            metadata.rate_limit_buckets[topic_type] = {
                "tokens": limits["burst"],
                "last_refill": current_time,
                "window_size": limits["window"]
            }
        
        bucket = metadata.rate_limit_buckets[topic_type]
        
        # Refill tokens based on time passed
        time_passed = (current_time - bucket["last_refill"]).total_seconds()
        tokens_to_add = (time_passed / bucket["window_size"]) * limits["sustained"]
        bucket["tokens"] = min(limits["burst"], bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = current_time
        
        # Check if we have tokens
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Rate limit check error: {e}")
        return True  # Fail open in case of errors

async def send_error_message(websocket: WebSocket, error: str, code: str, details: Optional[Dict[str, Any]] = None):
    """Send error message to WebSocket client."""
    try:
        error_msg = WebSocketError(
            error=error,
            code=code,
            details=details,
            timestamp=datetime.utcnow()
        )
        await websocket.send_text(error_msg.model_dump_json())
    except Exception as e:
        logger.error(f"Error sending error message: {e}")

@router.websocket("/market/{isin}")
async def market_data_stream(
    websocket: WebSocket,
    isin: str,
    manager: UnifiedWebSocketManager = Depends(get_websocket_manager)
):
    """Market data streaming endpoint for specific ISIN."""
    await websocket.accept()
    
    try:
        # Authenticate connection
        auth_data = await authenticate_websocket(websocket)
        if not auth_data:
            await send_error_message(websocket, "Authentication required", "AUTH_REQUIRED")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        # Create connection metadata
        metadata = ConnectionMetadata(
            user_id=auth_data["user_id"],
            device_type=auth_data["device_type"],
            permissions=auth_data["permissions"],
            ip_address=websocket.client.host if websocket.client else None,
            user_agent=websocket.headers.get("user-agent")
        )
        
        # Create connection
        connection = WebSocketConnection(
            id=str(uuid.uuid4()),
            websocket=websocket,
            metadata=metadata
        )
        
        # Register connection
        await manager.register_connection(connection)
        
        # Subscribe to market data topic
        topic = f"prices.{isin}"
        await manager.subscribe_connection(connection.id, topic)
        
        # Send initial snapshot
        snapshot = await manager.get_snapshot(topic)
        if snapshot:
            await websocket.send_text(snapshot.model_dump_json())
        
        # Handle client messages
        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "subscribe":
                    # Handle additional subscriptions
                    sub_request = SubscriptionRequest(**data)
                    if await check_rate_limit(metadata, sub_request.topic, manager):
                        await manager.subscribe_connection(connection.id, sub_request.topic)
                        response = SubscriptionResponse(
                            success=True,
                            topic=sub_request.topic,
                            sequence_number=await manager.get_sequence_number(sub_request.topic),
                            message="Subscription successful",
                            compression_enabled=metadata.compression_enabled,
                            rate_limits=RATE_LIMITS.get(sub_request.topic.split(".")[0], {})
                        )
                        await websocket.send_text(response.model_dump_json())
                    else:
                        await send_error_message(websocket, "Rate limit exceeded", "RATE_LIMIT_EXCEEDED")
                
                elif data.get("type") == "ping":
                    # Handle ping/pong
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                
                metadata.last_activity = datetime.utcnow()
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection.id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await send_error_message(websocket, "Internal server error", "INTERNAL_ERROR")
    
    except Exception as e:
        logger.error(f"Error in market data stream: {e}")
        await send_error_message(websocket, "Connection error", "CONNECTION_ERROR")
    finally:
        # Cleanup connection
        if 'connection' in locals():
            await manager.unregister_connection(connection.id)

@router.websocket("/auction/{auction_id}")
async def auction_stream(
    websocket: WebSocket,
    auction_id: str,
    manager: UnifiedWebSocketManager = Depends(get_websocket_manager)
):
    """Auction updates streaming endpoint."""
    await websocket.accept()
    
    try:
        # Authenticate connection
        auth_data = await authenticate_websocket(websocket)
        if not auth_data:
            await send_error_message(websocket, "Authentication required", "AUTH_REQUIRED")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        # Create connection metadata
        metadata = ConnectionMetadata(
            user_id=auth_data["user_id"],
            device_type=auth_data["device_type"],
            permissions=auth_data["permissions"],
            ip_address=websocket.client.host if websocket.client else None,
            user_agent=websocket.headers.get("user-agent")
        )
        
        # Create connection
        connection = WebSocketConnection(
            id=str(uuid.uuid4()),
            websocket=websocket,
            metadata=metadata
        )
        
        # Register connection
        await manager.register_connection(connection)
        
        # Subscribe to auction topic
        topic = f"auctions.{auction_id}"
        await manager.subscribe_connection(connection.id, topic)
        
        # Send initial snapshot
        snapshot = await manager.get_snapshot(topic)
        if snapshot:
            await websocket.send_text(snapshot.model_dump_json())
        
        # Handle client messages
        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                
                metadata.last_activity = datetime.utcnow()
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection.id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await send_error_message(websocket, "Internal server error", "INTERNAL_ERROR")
    
    except Exception as e:
        logger.error(f"Error in auction stream: {e}")
        await send_error_message(websocket, "Connection error", "CONNECTION_ERROR")
    finally:
        # Cleanup connection
        if 'connection' in locals():
            await manager.unregister_connection(connection.id)

@router.websocket("/trading/{user_id}")
async def trading_stream(
    websocket: WebSocket,
    user_id: str,
    manager: UnifiedWebSocketManager = Depends(get_websocket_manager)
):
    """Trading notifications streaming endpoint."""
    await websocket.accept()
    
    try:
        # Authenticate connection
        auth_data = await authenticate_websocket(websocket)
        if not auth_data or auth_data["user_id"] != user_id:
            await send_error_message(websocket, "Authentication required", "AUTH_REQUIRED")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        # Create connection metadata
        metadata = ConnectionMetadata(
            user_id=auth_data["user_id"],
            device_type=auth_data["device_type"],
            permissions=auth_data["permissions"],
            ip_address=websocket.client.host if websocket.client else None,
            user_agent=websocket.headers.get("user-agent")
        )
        
        # Create connection
        connection = WebSocketConnection(
            id=str(uuid.uuid4()),
            websocket=websocket,
            metadata=metadata
        )
        
        # Register connection
        await manager.register_connection(connection)
        
        # Subscribe to trading topics
        trading_topic = f"trading.{user_id}"
        portfolio_topic = f"portfolio.{user_id}"
        
        await manager.subscribe_connection(connection.id, trading_topic)
        await manager.subscribe_connection(connection.id, portfolio_topic)
        
        # Send initial snapshots
        for topic in [trading_topic, portfolio_topic]:
            snapshot = await manager.get_snapshot(topic)
            if snapshot:
                await websocket.send_text(snapshot.model_dump_json())
        
        # Handle client messages
        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                
                metadata.last_activity = datetime.utcnow()
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection.id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await send_error_message(websocket, "Internal server error", "INTERNAL_ERROR")
    
    except Exception as e:
        logger.error(f"Error in trading stream: {e}")
        await send_error_message(websocket, "Connection error", "CONNECTION_ERROR")
    finally:
        # Cleanup connection
        if 'connection' in locals():
            await manager.unregister_connection(connection.id)

@router.websocket("/risk/{user_id}")
async def risk_stream(
    websocket: WebSocket,
    user_id: str,
    manager: UnifiedWebSocketManager = Depends(get_websocket_manager)
):
    """Risk management streaming endpoint."""
    await websocket.accept()
    
    try:
        # Authenticate connection
        auth_data = await authenticate_websocket(websocket)
        if not auth_data or auth_data["user_id"] != user_id:
            await send_error_message(websocket, "Authentication required", "AUTH_REQUIRED")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        # Create connection metadata
        metadata = ConnectionMetadata(
            user_id=auth_data["user_id"],
            device_type=auth_data["device_type"],
            permissions=auth_data["permissions"],
            ip_address=websocket.client.host if websocket.client else None,
            user_agent=websocket.headers.get("user-agent")
        )
        
        # Create connection
        connection = WebSocketConnection(
            id=str(uuid.uuid4()),
            websocket=websocket,
            metadata=metadata
        )
        
        # Register connection
        await manager.register_connection(connection)
        
        # Subscribe to risk topics
        risk_topic = f"risk.{user_id}"
        alerts_topic = f"risk.{user_id}.alerts"
        
        await manager.subscribe_connection(connection.id, risk_topic)
        await manager.subscribe_connection(connection.id, alerts_topic)
        
        # Send initial snapshots
        for topic in [risk_topic, alerts_topic]:
            snapshot = await manager.get_snapshot(topic)
            if snapshot:
                await websocket.send_text(snapshot.model_dump_json())
        
        # Handle client messages
        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                
                metadata.last_activity = datetime.utcnow()
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection.id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await send_error_message(websocket, "Internal server error", "INTERNAL_ERROR")
    
    except Exception as e:
        logger.error(f"Error in risk stream: {e}")
        await send_error_message(websocket, "Connection error", "CONNECTION_ERROR")
    finally:
        # Cleanup connection
        if 'connection' in locals():
            await manager.unregister_connection(connection.id)

@router.websocket("/mobile/{user_id}")
async def mobile_stream(
    websocket: WebSocket,
    user_id: str,
    manager: UnifiedWebSocketManager = Depends(get_websocket_manager)
):
    """Mobile-optimized multiplexed streaming endpoint."""
    await websocket.accept()
    
    try:
        # Authenticate connection
        auth_data = await authenticate_websocket(websocket)
        if not auth_data or auth_data["user_id"] != user_id:
            await send_error_message(websocket, "Authentication required", "AUTH_REQUIRED")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        # Extract mobile-specific headers
        device_type = websocket.headers.get("x-device-type", "mobile")
        client_version = websocket.headers.get("x-client-version", "1.0.0")
        background_mode = websocket.headers.get("x-background-mode", "false").lower() == "true"
        
        # Create connection metadata
        metadata = ConnectionMetadata(
            user_id=auth_data["user_id"],
            device_type=device_type,
            client_version=client_version,
            permissions=auth_data["permissions"],
            ip_address=websocket.client.host if websocket.client else None,
            user_agent=websocket.headers.get("user-agent"),
            compression_enabled=True  # Always enable compression for mobile
        )
        
        # Create connection
        connection = WebSocketConnection(
            id=str(uuid.uuid4()),
            websocket=websocket,
            metadata=metadata
        )
        
        # Register connection
        await manager.register_connection(connection)
        
        # Subscribe to mobile-optimized topics
        mobile_topic = f"mobile.{user_id}"
        await manager.subscribe_connection(connection.id, mobile_topic)
        
        # Send initial snapshot
        snapshot = await manager.get_snapshot(mobile_topic)
        if snapshot:
            await websocket.send_text(snapshot.model_dump_json())
        
        # Handle client messages
        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                
                elif data.get("type") == "background_mode":
                    # Update background mode status
                    background_mode = data.get("enabled", False)
                    metadata.metadata["background_mode"] = background_mode
                    
                    # Adjust update frequency based on background mode
                    if background_mode:
                        await manager.set_connection_update_frequency(connection.id, mobile_topic, "low")
                    else:
                        await manager.set_connection_update_frequency(connection.id, mobile_topic, "normal")
                
                metadata.last_activity = datetime.utcnow()
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection.id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await send_error_message(websocket, "Internal server error", "INTERNAL_ERROR")
    
    except Exception as e:
        logger.error(f"Error in mobile stream: {e}")
        await send_error_message(websocket, "Connection error", "CONNECTION_ERROR")
    finally:
        # Cleanup connection
        if 'connection' in locals():
            await manager.unregister_connection(connection.id)

@router.get("/health")
async def websocket_health(manager: UnifiedWebSocketManager = Depends(get_websocket_manager)):
    """WebSocket health check endpoint."""
    try:
        stats = await manager.get_statistics()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"WebSocket health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="WebSocket service unavailable"
        )

@router.get("/topics")
async def list_topics(manager: UnifiedWebSocketManager = Depends(get_websocket_manager)):
    """List available WebSocket topics."""
    try:
        topics = await manager.list_topics()
        return {
            "topics": topics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing topics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list topics"
        )

# Export router
__all__ = ["router"]
