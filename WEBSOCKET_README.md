# BondX WebSocket System

## Overview

The BondX WebSocket system provides real-time streaming capabilities for market data, trading notifications, risk management alerts, and auction updates. It's built on a unified architecture that centralizes WebSocket management while supporting horizontal scaling.

## Architecture

### Components

1. **UnifiedWebSocketManager** (`bondx/websocket/unified_websocket_manager.py`)
   - Centralized connection management
   - Topic-based subscription system
   - Snapshot + incremental update protocol
   - Backpressure handling and rate limiting
   - Redis pub/sub for horizontal scaling

2. **WebSocket Router** (`bondx/api/v1/websocket.py`)
   - FastAPI WebSocket endpoints
   - Authentication and authorization
   - Rate limiting per topic
   - Mobile-optimized endpoints

3. **Monitoring System** (`websocket_monitoring.py`)
   - Prometheus metrics collection
   - Health checks and alerting
   - Performance monitoring
   - Threshold-based alerts

## WebSocket Endpoints

### Base URL
```
ws://localhost:8000/api/v1/ws
```

### Endpoints

#### 1. Market Data Stream
```
GET /ws/market/{isin}
```
- **Purpose**: Real-time market data for specific ISIN
- **Topics**: `prices.{isin}`, L1/L2 levels, trades
- **Authentication**: Required (token in query params or headers)
- **Rate Limits**: 1000 burst, 100 sustained per minute

#### 2. Auction Stream
```
GET /ws/auction/{auction_id}
```
- **Purpose**: Real-time auction updates and bid histograms
- **Topics**: `auctions.{auction_id}`
- **Authentication**: Required
- **Rate Limits**: 500 burst, 50 sustained per minute

#### 3. Trading Stream
```
GET /ws/trading/{user_id}
```
- **Purpose**: Trading notifications and order status
- **Topics**: `trading.{user_id}`, `portfolio.{user_id}`
- **Authentication**: Required (user_id must match authenticated user)
- **Rate Limits**: 200 burst, 20 sustained per minute

#### 4. Risk Management Stream
```
GET /ws/risk/{user_id}
```
- **Purpose**: Risk alerts and portfolio updates
- **Topics**: `risk.{user_id}`, `risk.{user_id}.alerts`
- **Authentication**: Required (user_id must match authenticated user)
- **Rate Limits**: 100 burst, 10 sustained per minute

#### 5. Mobile Stream
```
GET /ws/mobile/{user_id}
```
- **Purpose**: Mobile-optimized multiplexed stream
- **Topics**: `mobile.{user_id}`
- **Features**: Compression, background mode, device optimization
- **Authentication**: Required (user_id must match authenticated user)
- **Rate Limits**: 300 burst, 30 sustained per minute

### Health and Monitoring
```
GET /ws/health          # WebSocket health check
GET /ws/topics          # List available topics
GET /metrics            # Prometheus metrics (port 8001)
```

## Message Format

### Standard Message Envelope
```json
{
  "type": "delta|snapshot|alert|ack|error|ping|pong|heartbeat",
  "topic": "prices.IN1234567890",
  "seq": 12345,
  "ts": "2024-01-01T12:00:00Z",
  "payload": {...},
  "meta": {...},
  "correlation_id": "uuid-123"
}
```

### Message Types

- **snapshot**: Initial state when subscribing to a topic
- **delta**: Incremental updates
- **alert**: Important notifications (risk breaches, system alerts)
- **ack**: Acknowledgment messages
- **error**: Error messages
- **ping/pong**: Heartbeat messages
- **heartbeat**: System heartbeat

## Authentication

### Token-based Authentication
```javascript
// Query parameter
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/market/IN1234567890?token=your_jwt_token');

// Or header-based
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/market/IN1234567890', {
  headers: {
    'Authorization': 'Bearer your_jwt_token'
  }
});
```

### Mobile Headers
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/mobile/user123', {
  headers: {
    'x-device-type': 'mobile',
    'x-client-version': '1.0.0',
    'x-background-mode': 'false'
  }
});
```

## Subscription Model

### Topic-based Subscriptions
Topics follow the pattern: `{domain}.{identifier}`

- `prices.{isin}` - Market data for specific ISIN
- `auctions.{auction_id}` - Auction updates
- `trading.{user_id}` - User trading notifications
- `risk.{user_id}` - User risk data
- `mobile.{user_id}` - Mobile-optimized stream

### Subscription Request
```json
{
  "type": "subscribe",
  "topic": "prices.IN1234567890",
  "subtopics": {
    "levels": "L1,L2",
    "trades": true
  },
  "resume_from_seq": 12345,
  "compression": true,
  "batch_size": 100
}
```

## Rate Limiting

### Per-Topic Limits
- **Market Data**: 1000 burst, 100 sustained per minute
- **Auctions**: 500 burst, 50 sustained per minute
- **Trading**: 200 burst, 20 sustained per minute
- **Risk**: 100 burst, 10 sustained per minute
- **Mobile**: 300 burst, 30 sustained per minute

### Rate Limit Response
```json
{
  "error": "Rate limit exceeded",
  "code": "RATE_LIMIT_EXCEEDED",
  "details": {
    "topic": "prices.IN1234567890",
    "limit": "100 per minute"
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Mobile Optimization

### Background Mode
```javascript
// Enable background mode
ws.send(JSON.stringify({
  "type": "background_mode",
  "enabled": true
}));

// Disable background mode
ws.send(JSON.stringify({
  "type": "background_mode",
  "enabled": false
}));
```

### Device Headers
- `x-device-type`: Device type (mobile, tablet, desktop)
- `x-client-version`: Client application version
- `x-background-mode`: Background mode status

## Load Testing

### K6 Test Script
```bash
# Run basic market data test
k6 run websocket_load_test.js

# Run with custom configuration
k6 run -e BASE_URL=ws://localhost:8000 -e TEST_DURATION=600 -e MIXED_WORKLOAD=true websocket_load_test.js
```

### Test Scenarios
1. **Market Data**: 100-200 concurrent connections to prices.{isin}
2. **Mixed Workload**: 50% market, 30% trading, 20% risk
3. **Reconnection**: Test connection resilience
4. **Mobile**: Test mobile-specific features

### Performance Targets
- **Connection Success Rate**: >95%
- **Message Latency P95**: <100ms
- **Message Drop Rate**: <1%
- **Queue Overflow**: <1000 messages

## Monitoring and Metrics

### Prometheus Metrics
- **Connection Metrics**: Total, active, established, closed
- **Message Metrics**: Sent, received, dropped, latency
- **Performance Metrics**: Queue sizes, processing times
- **Error Metrics**: Error counts, rate limit violations

### Health Checks
```bash
# WebSocket health
curl http://localhost:8000/api/v1/ws/health

# Prometheus metrics
curl http://localhost:8001/metrics
```

### Alerting
- High connection failure rate (>5%)
- High message drop rate (>1%)
- High latency (P95 >100ms)
- Queue overflow (>1000 messages)
- High error rate (>2%)

## Configuration

### Environment Variables
```bash
# WebSocket settings
WS_MAX_CONNECTIONS=1000
WS_HEARTBEAT_INTERVAL=30
WS_CONNECTION_TIMEOUT=90
WS_MAX_QUEUE_SIZE=1000

# Redis settings
REDIS_URL=redis://localhost:6379/0

# Monitoring settings
PROMETHEUS_PORT=8001
ALERT_THRESHOLDS_JSON={"max_connection_failure_rate": 0.05}
```

### Configuration File
```python
# bondx/core/config.py
class WebSocketSettings(BaseSettings):
    max_connections: int = 1000
    ping_interval: int = 30
    ping_timeout: int = 10
```

## Development

### Running Locally
```bash
# Start the application
python -m bondx.main

# WebSocket endpoints will be available at:
# ws://localhost:8000/api/v1/ws/*
```

### Testing
```bash
# Run WebSocket tests
python -m pytest test_websocket.py

# Run load tests
k6 run websocket_load_test.js
```

### Debugging
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Check WebSocket connections
curl http://localhost:8000/api/v1/ws/topics

# Monitor metrics
curl http://localhost:8001/metrics
```

## Production Deployment

### Scaling
- **Horizontal**: Multiple instances with Redis pub/sub
- **Load Balancing**: Use sticky sessions for WebSocket connections
- **Redis Cluster**: For high availability

### Security
- **TLS**: Enable WSS (WebSocket Secure)
- **Rate Limiting**: Enforce per-user and per-IP limits
- **Authentication**: JWT validation with short expiry
- **CORS**: Restrict origins in production

### Monitoring
- **Prometheus**: Collect metrics
- **Grafana**: Visualize performance
- **Alerting**: Configure PagerDuty/Slack alerts
- **Logging**: Centralized logging with correlation IDs

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Check authentication token
   - Verify rate limits
   - Check Redis connectivity

2. **High Latency**
   - Monitor queue sizes
   - Check Redis performance
   - Verify network connectivity

3. **Memory Issues**
   - Check connection cleanup
   - Monitor queue sizes
   - Verify Redis memory usage

### Debug Commands
```bash
# Check Redis connectivity
redis-cli ping

# Monitor WebSocket connections
curl http://localhost:8000/api/v1/ws/health

# Check system resources
top -p $(pgrep -f "bondx")
```

## API Reference

### WebSocket Events

#### Client to Server
- `subscribe`: Subscribe to topics
- `unsubscribe`: Unsubscribe from topics
- `ping`: Send heartbeat
- `background_mode`: Toggle mobile background mode

#### Server to Client
- `snapshot`: Initial state data
- `delta`: Incremental updates
- `alert`: Important notifications
- `pong`: Heartbeat response
- `error`: Error messages

### Error Codes
- `AUTH_REQUIRED`: Authentication required
- `RATE_LIMIT_EXCEEDED`: Rate limit exceeded
- `INTERNAL_ERROR`: Internal server error
- `CONNECTION_ERROR`: Connection error

## Contributing

### Code Style
- Follow PEP 8 for Python code
- Use type hints
- Add docstrings for all functions
- Include unit tests

### Testing
- Unit tests for all components
- Integration tests for WebSocket flows
- Load tests for performance validation
- Security tests for authentication

### Documentation
- Update this README for new features
- Document API changes
- Include usage examples
- Maintain troubleshooting guide
