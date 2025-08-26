# Liquidity-Risk Translator System for BondX

## Overview

The Liquidity-Risk Translator is an integrated system that bridges retail investors with market makers by providing plain-English risk translation and real-time liquidity profiling with actionable exit pathways. It delivers a single retail-facing narrative that reduces fear of illiquidity by quantifying both risk and feasible exits with expected fill time and price context.

## Key Features

- **Plain-English Risk Translation**: Probability-based category scores for liquidity, refinancing, leverage, governance, legal, and ESG risks
- **Real-time Liquidity Profiling**: Comprehensive liquidity assessment with actionable exit pathways
- **Exit Pathway Recommendations**: Market maker quotes, fractional auctions, RFQ batch windows, and tokenized P2P crossing
- **Integrated Narrative**: Single view combining risk summary, liquidity scorecard, and recommended exit paths
- **Real-time Updates**: WebSocket-based updates as market conditions change
- **Explainable & Auditable**: Every numeric is traceable with citations and model versions

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Liquidity-Risk Orchestrator                 │
│                    (Main Integration Layer)                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
    ┌───────────▼──────────┐    │    ┌──────────▼──────────┐
    │  Liquidity           │    │    │  Exit               │
    │  Intelligence        │    │    │  Recommender        │
    │  Service             │    │    │  Service            │
    └──────────────────────┘    │    └──────────────────────┘
                                │
                ┌───────────────▼───────────────┐
                │        Risk Scoring           │
                │        Engine                 │
                └───────────────────────────────┘
```

### Core Components

1. **Liquidity Intelligence Service** (`liquidity_intelligence_service.py`)
   - Computes liquidity scores (0-100)
   - Analyzes market microstructure
   - Estimates time-to-exit
   - Processes auction and market maker signals

2. **Exit Recommender Service** (`exit_recommender.py`)
   - Selects and ranks exit routes
   - Computes fill probability
   - Analyzes constraints and policy limits
   - Provides rationale for recommendations

3. **Liquidity-Risk Orchestrator** (`liquidity_risk_orchestrator.py`)
   - Merges risk and liquidity narratives
   - Ensures coherency and clarity
   - Generates retail and professional summaries
   - Manages data persistence and audit trails

4. **API Layer** (`liquidity_risk.py`)
   - REST endpoints for data retrieval
   - Background recomputation triggers
   - Audit trail access
   - Health monitoring

5. **WebSocket Manager** (`liquidity_risk_websocket.py`)
   - Real-time updates via WebSocket
   - Topic-based subscriptions
   - Connection management and heartbeat
   - Cross-instance communication via Redis

## Data Contracts

### Input Data

#### Risk Artifacts
```json
{
  "risk_summary": {
    "isin": "IN0012345678",
    "as_of": "2024-12-01T10:00:00Z",
    "categories": [
      {
        "name": "Liquidity Risk",
        "score_0_100": 35.0,
        "level": "low",
        "probability_note": "Based on market depth and trading activity",
        "citations": ["Market microstructure analysis"]
      }
    ],
    "overall_score": 42.5,
    "confidence": 0.85
  }
}
```

#### Market Microstructure
```json
{
  "timestamp": "2024-12-01T10:00:00Z",
  "isin": "IN0012345678",
  "bid": 100.50,
  "ask": 100.75,
  "bid_size": 1000000,
  "ask_size": 1000000,
  "l2_depth_qty": 5000000,
  "l2_levels": 5,
  "trades_count": 25,
  "vwap": 100.625,
  "volume_face": 10000000,
  "time_since_last_trade_s": 300
}
```

#### Auction Signals
```json
{
  "timestamp": "2024-12-01T10:00:00Z",
  "isin": "IN0012345678",
  "auction_id": "AUCTION_IN0012345678_20241201",
  "lots_offered": 10,
  "bids_count": 8,
  "demand_curve_points": [[100.0, 1000000], [100.25, 2000000]],
  "clearing_price_estimate": 100.25,
  "next_window": "2024-12-01T14:00:00Z"
}
```

#### Market Maker State
```json
{
  "timestamp": "2024-12-01T10:00:00Z",
  "isin": "IN0012345678",
  "mm_online": true,
  "mm_inventory_band": [500000, 1000000, 2000000],
  "mm_min_spread_bps": 10.0,
  "last_quote_spread_bps": 25.0,
  "quotes_last_24h": 45
}
```

### Output Data

#### Complete Translation
```json
{
  "isin": "IN0012345678",
  "as_of": "2024-12-01T10:00:00Z",
  "risk_summary": { /* Risk assessment */ },
  "liquidity_profile": { /* Liquidity metrics */ },
  "exit_recommendations": [ /* Exit pathways */ ],
  "retail_narrative": "This bond presents a moderate overall risk profile...",
  "professional_summary": "LIQUIDITY-RISK ANALYSIS SUMMARY...",
  "risk_warnings": [ /* Risk alerts */ ],
  "confidence_overall": 0.82,
  "data_freshness": "real_time",
  "inputs_hash": "a1b2c3d4e5f6g7h8",
  "model_versions": { /* Model versions */ },
  "caveats": [ /* Disclaimers */ ]
}
```

## API Endpoints

### Core Endpoints

#### Get Liquidity-Risk Translation
```http
GET /api/v1/liquidity-risk/{isin}?mode=fast|accurate&detail=summary|full&trade_size=100000
```

**Parameters:**
- `isin`: Bond ISIN identifier
- `mode`: `fast` for cached results, `accurate` for recompute
- `detail`: `summary` or `full` detail level
- `trade_size`: Size of position to exit (in currency units)

**Response:** Integrated liquidity-risk analysis with exit recommendations

#### Trigger Recomputation
```http
POST /api/v1/liquidity-risk/recompute
Content-Type: application/json

{
  "isins": ["IN0012345678", "IN0087654321"],
  "mode": "accurate"
}
```

#### Get Audit Trail
```http
GET /api/v1/liquidity-risk/audit/{isin}?as_of=2024-12-01T10:00:00Z
```

#### Health Check
```http
GET /api/v1/liquidity-risk/health
```

### Response Formats

#### Summary Response
```json
{
  "success": true,
  "message": "Success",
  "timestamp": "2024-12-01T10:00:00Z",
  "isin": "IN0012345678",
  "as_of": "2024-12-01T10:00:00Z",
  "risk_summary": {
    "overall_score": 42.5,
    "confidence": 0.85
  },
  "liquidity_profile": {
    "liquidity_index": 78.2,
    "spread_bps": 25.0,
    "liquidity_level": "good",
    "expected_time_to_exit_minutes": 30.0
  },
  "exit_recommendations": [
    {
      "path": "market_maker",
      "priority": "primary",
      "fill_probability": 0.82,
      "expected_time_to_exit_minutes": 15.0,
      "expected_spread_bps": 25.0
    }
  ],
  "retail_narrative": "This bond presents a moderate overall risk profile...",
  "confidence_overall": 0.82,
  "data_freshness": "real_time"
}
```

## WebSocket Events

### Topic Structure
- **Main Topic**: `lr.{isin}` - Complete liquidity-risk updates
- **Risk Alerts**: `lr.{isin}.alerts` - Risk-specific alerts
- **Liquidity Alerts**: `lr.{isin}.liquidity` - Liquidity-specific alerts
- **Exit Alerts**: `lr.{isin}.exit` - Exit pathway alerts

### Event Types

#### Snapshot Event
```json
{
  "type": "snapshot",
  "topic": "lr.IN0012345678",
  "seq": 1,
  "ts": "2024-12-01T10:00:00Z",
  "payload": {
    "isin": "IN0012345678",
    "risk_summary": { /* Risk data */ },
    "liquidity_profile": { /* Liquidity data */ },
    "exit_recommendations": [ /* Exit data */ ]
  },
  "meta": {
    "data_freshness": "real_time",
    "confidence": 0.82
  }
}
```

#### Risk Update Event
```json
{
  "type": "risk_update",
  "topic": "lr.IN0012345678",
  "seq": 2,
  "ts": "2024-12-01T10:01:00Z",
  "payload": {
    "isin": "IN0012345678",
    "risk_summary": { /* Updated risk data */ },
    "risk_warnings": [ /* Risk alerts */ ]
  }
}
```

#### Liquidity Update Event
```json
{
  "type": "liquidity_update",
  "topic": "lr.IN0012345678",
  "seq": 3,
  "ts": "2024-12-01T10:02:00Z",
  "payload": {
    "isin": "IN0012345678",
    "liquidity_profile": { /* Updated liquidity data */ }
  }
}
```

#### Exit Path Update Event
```json
{
  "type": "exit_path_update",
  "topic": "lr.IN0012345678",
  "seq": 4,
  "ts": "2024-12-01T10:03:00Z",
  "payload": {
    "isin": "IN0012345678",
    "exit_recommendations": [ /* Updated exit data */ ],
    "best_path": "market_maker"
  }
}
```

#### Alert Event
```json
{
  "type": "alert",
  "topic": "lr.IN0012345678.alerts",
  "seq": 5,
  "ts": "2024-12-01T10:04:00Z",
  "payload": {
    "alert_type": "liquidity_warning",
    "message": "Market maker offline - limited exit options",
    "severity": "warning",
    "isin": "IN0012345678",
    "timestamp": "2024-12-01T10:04:00Z"
  }
}
```

## Algorithms and Methods

### Liquidity Index Calculation (0-100)

The liquidity index combines multiple factors with calibrated weights:

1. **Spread Score** (30%): Normalized bid-ask spread
   - Formula: `max(0, 100 - (spread_bps * 2))`

2. **Depth Score** (25%): Available liquidity depth
   - Formula: `min(100, (l2_depth_qty / volume_face) * 1000)`

3. **Turnover Score** (25%): Recent trading activity
   - Formula: `max(0, 100 - (time_since_last_trade_s / 3600))`

4. **Volume Score** (20%): Issue size and trading volume
   - Formula: `min(100, (volume_face / 1000000) * 20)`

### Exit Path Analysis

#### Market Maker Path
- **Fill Probability**: Based on quote frequency, spread quality, and inventory
- **Time to Exit**: 15-60 minutes depending on trade size and market conditions
- **Constraints**: Inventory limits, rating caps, issuer class restrictions

#### Auction Path
- **Fill Probability**: Based on bid count and demand curve slope
- **Time to Exit**: Next auction window + 2 hours for settlement
- **Constraints**: Rating caps, auction cadence, time to next window

#### RFQ Batch Path
- **Fill Probability**: Based on rating and trade size
- **Time to Exit**: Next RFQ window + 1 hour for batch processing
- **Constraints**: Rating caps, RFQ window schedule

#### Tokenized P2P Path
- **Fill Probability**: Conservative estimate based on rating and size
- **Time to Exit**: 4 hours for P2P matching
- **Constraints**: Minimum lot size, tenor restrictions

### Risk Assessment

Risk scores are calculated across 8 categories:
1. **Liquidity Risk** (25%): Market depth and trading activity
2. **Credit Risk** (25%): Issuer rating and financial metrics
3. **Interest Rate Risk** (15%): Duration and yield curve sensitivity
4. **Refinancing Risk** (10%): Maturity profile and market access
5. **Leverage Risk** (10%): Debt ratios and coverage
6. **Governance Risk** (5%): Board composition and policies
7. **Legal Risk** (5%): Regulatory compliance and litigation
8. **ESG Risk** (5%): Environmental, social, and governance factors

## Usage Examples

### Python API Usage

```python
from bondx.ai_risk_engine.liquidity_risk_orchestrator import LiquidityRiskOrchestrator
from bondx.ai_risk_engine.liquidity_intelligence_service import MarketMicrostructure

# Initialize orchestrator
orchestrator = LiquidityRiskOrchestrator()

# Create market data
microstructure = MarketMicrostructure(
    timestamp=datetime.now(),
    isin="IN0012345678",
    bid=100.50,
    ask=100.75,
    # ... other fields
)

# Generate translation
translation = orchestrator.create_liquidity_risk_translation(
    isin="IN0012345678",
    microstructure=microstructure,
    risk_data=risk_data,
    trade_size=1000000
)

# Access results
print(f"Liquidity Index: {translation.liquidity_profile.liquidity_index}")
print(f"Risk Score: {translation.risk_summary.overall_score}")
print(f"Best Exit Path: {translation.exit_recommendations[0].path.value}")
print(f"Retail Narrative: {translation.retail_narrative}")
```

### WebSocket Client Usage

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/liquidity-risk');

// Subscribe to topic
ws.send(JSON.stringify({
    action: 'subscribe',
    topic: 'lr.IN0012345678'
}));

// Handle messages
ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    
    switch(message.type) {
        case 'snapshot':
            console.log('Full update:', message.payload);
            break;
        case 'risk_update':
            console.log('Risk update:', message.payload);
            break;
        case 'liquidity_update':
            console.log('Liquidity update:', message.payload);
            break;
        case 'exit_path_update':
            console.log('Exit path update:', message.payload);
            break;
        case 'alert':
            console.log('Alert:', message.payload);
            break;
    }
};
```

## Configuration

### Environment Variables

```bash
# Redis configuration
REDIS_URL=redis://localhost:6379

# Service configuration
LIQUIDITY_RISK_SERVICE_VERSION=1.0.0
LIQUIDITY_RISK_CACHE_TTL_HOURS=24
LIQUIDITY_RISK_WEBSOCKET_HEARTBEAT_SECONDS=30

# Model configuration
LIQUIDITY_INDEX_WEIGHTS_SPREAD=0.3
LIQUIDITY_INDEX_WEIGHTS_DEPTH=0.25
LIQUIDITY_INDEX_WEIGHTS_TURNOVER=0.25
LIQUIDITY_INDEX_WEIGHTS_VOLUME=0.2

# Exit policy configuration
EXIT_POLICY_MM_MIN_SPREAD_BPS=5.0
EXIT_POLICY_AUCTION_CADENCE_HOURS=4
EXIT_POLICY_RFQ_WINDOW_HOURS=2
EXIT_POLICY_TOKENIZED_MIN_LOT=25000
```

### Policy Configuration

```yaml
# exit_policy.yaml
rating_caps:
  AAA:
    mm_cap: 1000000
    auction_cap: 5000000
    rfq_cap: 2000000
  AA:
    mm_cap: 800000
    auction_cap: 4000000
    rfq_cap: 1500000
  # ... other ratings

tenor_restrictions:
  "0-1Y": ["market_maker", "auction", "rfq_batch"]
  "1-3Y": ["market_maker", "auction", "rfq_batch", "tokenized_p2p"]
  "3-5Y": ["market_maker", "auction", "rfq_batch", "tokenized_p2p"]
  "5-10Y": ["market_maker", "auction", "rfq_batch", "tokenized_p2p"]
  "10Y+": ["auction", "rfq_batch", "tokenized_p2p"]

issuer_class_limits:
  government:
    mm_cap: 5000000
    auction_cap: 10000000
    rfq_cap: 5000000
  # ... other issuer classes
```

## Testing

### Run Tests

```bash
# Run the comprehensive test
python test_liquidity_risk_translator.py

# Run specific component tests
python -m pytest bondx/ai_risk_engine/test_liquidity_intelligence_service.py
python -m pytest bondx/ai_risk_engine/test_exit_recommender.py
python -m pytest bondx/ai_risk_engine/test_liquidity_risk_orchestrator.py
```

### Test Scenarios

1. **Medium Credit Risk + High Liquidity**: MM path recommended
2. **Low Credit Risk + Thin Market**: Auction/RFQ recommended
3. **High Credit Risk + High Demand**: Exit path with risk warnings
4. **No MM + No Auction Window**: Tokenized P2P prioritized

## Performance and Scalability

### Latency Targets
- **Fast Mode API**: ≤300ms p95
- **WebSocket Updates**: ≤200ms publish-to-receive
- **Real-time Processing**: ≤100ms for market data updates

### Caching Strategy
- **Risk Data**: 24-hour TTL with background refresh
- **Liquidity Data**: 5-minute TTL for real-time accuracy
- **Exit Analysis**: 15-minute TTL with market condition triggers

### Horizontal Scaling
- **Stateless Services**: Multiple instances behind load balancer
- **Redis Pub/Sub**: Cross-instance WebSocket communication
- **Database Sharding**: By ISIN range for large portfolios

## Monitoring and Observability

### Key Metrics
- **Recommendation Adoption**: User acceptance of exit recommendations
- **Prediction Accuracy**: Predicted vs realized fill time (MAPE)
- **Liquidity Index Drift**: Changes in liquidity assessments
- **Data Freshness**: Rate of stale data
- **WebSocket Latency**: End-to-end message delivery time

### Dashboards
- **Liquidity-Risk Translator Panel**: Real-time KPIs and error budgets
- **Exit Path Performance**: Fill rates and execution quality
- **Risk Assessment Quality**: Confidence levels and data lineage
- **System Health**: Component status and performance metrics

### Alerts
- **High Risk Warnings**: Critical risk level alerts
- **Liquidity Deterioration**: Significant liquidity index drops
- **Exit Path Failures**: Unavailable exit pathways
- **Data Quality Issues**: Stale or missing data alerts

## Security and Compliance

### RBAC Implementation
- **Retail Users**: Summary data + basic exit recommendations
- **Professional Users**: Full technical details + advanced analytics
- **Risk Managers**: Internal metrics + model governance
- **Compliance Officers**: Audit trails + regulatory reporting

### Data Privacy
- **Sensitive Data Masking**: MM inventory details hidden from retail users
- **Audit Logging**: All data access and modifications logged
- **Input Validation**: Comprehensive schema validation for all inputs
- **Rate Limiting**: API and WebSocket rate limiting per user

### Regulatory Compliance
- **Disclaimers**: Clear statements that recommendations are informational
- **Model Governance**: Version tracking and validation
- **Data Lineage**: Complete audit trail of data sources
- **Risk Disclosures**: Comprehensive risk factor explanations

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "bondx.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: liquidity-risk-translator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: liquidity-risk-translator
  template:
    metadata:
      labels:
        app: liquidity-risk-translator
    spec:
      containers:
      - name: liquidity-risk-translator
        image: bondx/liquidity-risk-translator:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis:6379"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up Redis
docker run -d -p 6379:6379 redis:7-alpine

# Run the service
uvicorn bondx.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
python test_liquidity_risk_translator.py
```

## Future Enhancements

### Phase 2 Features
- **Machine Learning Models**: Enhanced prediction accuracy using ML
- **Real-time Market Data**: Integration with live market feeds
- **Portfolio Analytics**: Multi-bond portfolio risk assessment
- **Advanced Exit Strategies**: Dynamic exit path optimization

### Phase 3 Features
- **Predictive Analytics**: Forward-looking risk and liquidity forecasts
- **Scenario Analysis**: Stress testing and what-if analysis
- **Regulatory Reporting**: Automated compliance reporting
- **Mobile Applications**: Native mobile apps for retail users

## Support and Documentation

### API Documentation
- **Swagger UI**: Available at `/docs` when service is running
- **OpenAPI Spec**: Available at `/openapi.json`
- **Postman Collection**: Available in `/docs/postman/`

### Troubleshooting
- **Health Checks**: Monitor `/health` endpoint
- **Logs**: Check application logs for detailed error information
- **Metrics**: Monitor Prometheus metrics for performance issues
- **Redis Status**: Verify Redis connectivity for WebSocket functionality

### Contact
- **Development Team**: dev@bondx.com
- **Support**: support@bondx.com
- **Documentation**: docs.bondx.com/liquidity-risk-translator

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Acknowledgments

- BondX Development Team
- Financial Risk Management Community
- Open Source Contributors
- Regulatory Compliance Experts
