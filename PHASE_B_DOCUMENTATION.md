# BondX Phase B Implementation Documentation

## Overview

Phase B of the BondX system implements advanced financial modeling capabilities including Option-Adjusted Spread (OAS) calculations, comprehensive stress testing, and portfolio analytics with performance attribution. This document provides comprehensive coverage of the implementation, mathematical foundations, and usage examples.

## Table of Contents

1. [OAS Calculator (B1)](#oas-calculator-b1)
2. [Stress Testing Engine (B2)](#stress-testing-engine-b2)
3. [Portfolio Analytics & Attribution (B3)](#portfolio-analytics--attribution-b3)
4. [Cross-Cutting Contracts & Persistence (B4)](#cross-cutting-contracts--persistence-b4)
5. [API Endpoints](#api-endpoints)
6. [Performance Tuning Guide](#performance-tuning-guide)
7. [Testing & Validation](#testing--validation)
8. [Deployment & Configuration](#deployment--configuration)

---

## OAS Calculator (B1)

### Overview

The OAS Calculator provides production-grade option-adjusted spread calculations for bonds with embedded options using both lattice and Monte Carlo methods.

### Supported Option Types

- **Callable Bonds**: Issuer can call the bond at specified dates/prices
- **Putable Bonds**: Holder can put the bond back to issuer
- **Callable with Make-Whole**: Callable bonds with make-whole provisions
- **Prepayment-Capable**: Mortgage-backed securities with prepayment options

### Pricing Methods

#### 1. Lattice Method

**Short-Rate Models:**
- **Ho-Lee Model**: `dr = θ(t)dt + σdW`
  - Simple, fast implementation
  - Suitable for basic option pricing
  - Calibrated to match yield curve

- **Black-Derman-Toy Model**: `dln(r) = θ(t)dt + σ(t)dW`
  - Handles mean reversion
  - More sophisticated rate dynamics
  - Better for complex term structures

**Implementation Details:**
```python
# Lattice construction
lattice_steps = 500  # Configurable
dt = 1.0 / lattice_steps

# Backward induction with optimal exercise
for step in range(final_step - 1, -1, -1):
    for node in lattice[step]:
        continuation_value = calculate_continuation_value(lattice, step, node)
        exercise_value = calculate_exercise_value(step, node)
        
        # Choose optimal value based on option type
        if option_type == OptionType.CALLABLE:
            node.bond_value = min(continuation_value, exercise_value)  # Issuer optimal
        elif option_type == OptionType.PUTABLE:
            node.bond_value = max(continuation_value, exercise_value)  # Holder optimal
```

#### 2. Monte Carlo Method

**Path Generation:**
- Simulate short-rate paths using calibrated parameters
- Support for path-dependent optimal exercise decisions
- Configurable number of paths (default: 10,000)

**Longstaff-Schwartz Style Exercise:**
```python
# Evaluate cash flows under pathwise optimal exercise
for path in rate_paths:
    pv = 0.0
    discount_factor = 1.0
    
    for step, rate in enumerate(path):
        # Add cash flow if any
        cf = calculate_cash_flow_at_step(step)
        pv += cf * discount_factor
        
        # Update discount factor with OAS
        discount_factor *= exp(-(rate + oas_rate) * dt)
    
    present_values.append(pv)

# Return average present value
avg_pv = np.mean(present_values)
```

### Input Parameters

**Required:**
- Base yield curve (from YieldCurveEngine)
- Volatility surface (rate vol by tenor)
- Cash flows (option-aware schedule)
- Option type and schedules
- Market observed price

**Optional:**
- Prepayment function hook
- Day-count conventions
- Compounding frequency
- Settlement date

### Output Metrics

**Primary:**
- OAS value (basis points)
- Model present value at OAS
- Option value (embedded option premium)

**Risk Measures:**
- Option-adjusted duration
- Option-adjusted convexity
- Greeks (delta, gamma, theta) where applicable

**Diagnostics:**
- Convergence status
- Iteration count
- Execution time
- Lattice steps/Monte Carlo paths

### Performance Targets

- **Lattice Mode**: ≤30ms for 500-step tree (typical case)
- **Monte Carlo Mode**: Configurable paths with performance profiling
- **Fast Mode**: ≤10ms for option-free bonds
- **Accurate Mode**: ≤100ms for complex option structures

---

## Stress Testing Engine (B2)

### Overview

The Stress Testing Engine provides configurable scenario analysis for portfolio-level and instrument-level stress testing with both fast approximation and full reprice modes.

### Predefined Scenarios

#### 1. Rate Curve Shocks

**Parallel Shifts:**
- +50bps, +100bps, +200bps
- Applies uniform shift across all tenors
- Suitable for level risk assessment

**Curve Shape Changes:**
- **Steepening**: +50bps (long-term rates increase more than short-term)
- **Flattening**: +50bps (long-term rates decrease more than short-term)

**Implementation:**
```python
def apply_curve_steepening(curve, steepening_bps):
    steepening_factor = curve.tenors / np.max(curve.tenors)
    curve.rates += (steepening_bps / 10000.0) * steepening_factor

def apply_curve_flattening(curve, flattening_bps):
    flattening_factor = 1.0 - (curve.tenors / np.max(curve.tenors))
    curve.rates -= (flattening_bps / 10000.0) * flattening_factor
```

#### 2. Credit Spread Shocks

**Rating-Based Blowouts:**
```python
credit_spread_shocks = {
    RatingBucket.AAA: 50,   # +50bps
    RatingBucket.AA: 75,    # +75bps
    RatingBucket.A: 100,    # +100bps
    RatingBucket.BBB: 150,  # +150bps
    RatingBucket.BB: 200,   # +200bps
    RatingBucket.B: 300,    # +300bps
    RatingBucket.CCC: 400   # +400bps
}
```

#### 3. Liquidity Shocks

- **Additional Spread**: +50bps liquidity premium
- **Bid-Ask Widening**: +25bps spread widening
- **Market Impact**: Simulates reduced market liquidity

#### 4. Volatility Shocks

- **Volatility Multiplier**: 2x (doubling of volatility surface)
- **Affects**: OAS-sensitive instruments, option pricing

### Calculation Modes

#### 1. Fast Approximation

**Method:**
- Duration/convexity approximation
- Spread DV01 analysis
- **Performance**: ≤100ms for 10,000 positions

**Formula:**
```
P&L ≈ -Duration × Face_Value × Rate_Change + 
       0.5 × Convexity × Face_Value × (Rate_Change)² +
       -Spread_DV01 × Face_Value × Spread_Change
```

#### 2. Full Reprice

**Method:**
- Rebuild curves with shocks
- Full revaluation via CashFlowEngine + YieldCurveEngine
- OAS calculations for option-embedded instruments
- **Performance**: Configurable, typically 1-10 seconds for 10,000 positions

### Output Analysis

**Portfolio Impact:**
- Total P&L (absolute and basis points)
- Risk metric changes (ΔDV01, Δkey-rate duration, Δspread DV)

**Drilldown Results:**
- P&L by issuer, sector, rating, tenor
- Limit breach identification
- Scenario narrative and model settings

---

## Portfolio Analytics & Attribution (B3)

### Overview

The Portfolio Analytics engine provides comprehensive risk metrics, concentration analysis, and performance attribution suitable for risk and performance reporting.

### Risk Metrics

#### 1. Duration & Convexity

**Portfolio Duration:**
```python
portfolio_duration = sum(
    position.market_value * position.duration 
    for position in positions
) / total_portfolio_value
```

**Portfolio Convexity:**
```python
portfolio_convexity = sum(
    position.market_value * position.convexity 
    for position in positions
) / total_portfolio_value
```

#### 2. Key Rate Durations

**Tenor Buckets:**
- 0-1Y, 1-3Y, 3-5Y, 5-10Y, 10Y+
- Aggregated by position weights
- Normalized by total portfolio value

#### 3. Spread Risk

**Rating Buckets:**
- AAA, AA, A, BBB, BB, B, CCC, DEFAULT
- Spread DV01 by rating category
- Credit risk concentration analysis

#### 4. Liquidity Metrics

**Liquidity Score:**
- 0.0 (illiquid) to 1.0 (highly liquid)
- Weighted average across portfolio
- Illiquid exposure percentage

### Performance Attribution

#### 1. Factor Decomposition

**Curve Factors (PCA-based):**
```python
# Extract curve factors using PCA
curve_matrix = np.vstack([end_curve.rates - start_curve.rates 
                          for start_curve, end_curve in curve_pairs])
pca.fit(curve_matrix.T)

# Map to level, slope, curvature
curve_decomposition = {
    'level': pca.explained_variance_ratio_[0],
    'slope': pca.explained_variance_ratio_[1],
    'curvature': pca.explained_variance_ratio_[2]
}
```

**Attribution Factors:**
- **Carry/Roll-down**: Interest income and time decay
- **Curve Level**: Parallel yield curve shifts
- **Curve Slope**: Steepening/flattening effects
- **Curve Curvature**: Shape changes
- **Credit Spread**: Credit risk premium changes
- **Selection**: Security-specific outperformance
- **Trading**: Timing effects
- **Idiosyncratic**: Uncaptured factors

#### 2. Attribution Calculation

**Return Decomposition:**
```
Total_Return = Carry_Roll + Curve_Level + Curve_Slope + 
               Curve_Curvature + Credit_Spread + Selection + 
               Trading + Idiosyncratic + Residual
```

**Basis Point Conversion:**
```python
factor_contributions_bps = {
    factor: contribution * 10000 
    for factor, contribution in factor_contributions.items()
}
```

### Turnover Analysis

**Metrics:**
- **Gross Turnover**: Total position changes
- **Net Turnover**: Net position changes
- **Buy/Sell Turnover**: Directional changes

**Calculation:**
```python
gross_turnover = (value_added + value_removed + value_modified) / avg_portfolio_value
net_turnover = abs(value_added - value_removed) / avg_portfolio_value
```

### Concentration Analysis

**Limits:**
- **Issuer**: 5% warning, 10% limit
- **Sector**: 25% warning, 40% limit
- **Rating**: 30% warning, 50% limit (below investment grade)
- **Tenor**: 40% warning, 60% limit
- **Liquidity**: 20% warning, 30% limit (illiquid)

---

## Cross-Cutting Contracts & Persistence (B4)

### Model Contracts

#### 1. Model Types

```python
class ModelType(Enum):
    OAS_CALCULATION = "OAS_CALCULATION"
    STRESS_TEST = "STRESS_TEST"
    PORTFOLIO_ANALYTICS = "PORTFOLIO_ANALYTICS"
    YIELD_CURVE_CONSTRUCTION = "YIELD_CURVE_CONSTRUCTION"
    CASH_FLOW_PROJECTION = "CASH_FLOW_PROJECTION"
    RISK_METRICS = "RISK_METRICS"
```

#### 2. Validation Framework

**Severity Levels:**
- **INFO**: Informational messages
- **WARNING**: Non-critical issues
- **ERROR**: Validation failures
- **CRITICAL**: System errors

**Validation Rules:**
- Required field validation
- Type and range checking
- Business rule validation
- Cross-field consistency

#### 3. Model Result Storage

**Structure:**
```python
@dataclass
class ModelResult:
    model_type: ModelType
    inputs: ModelInputs
    outputs: ModelOutputs
    model_id: str
    execution_id: str
    created_date: datetime
    updated_date: datetime
    curve_id: Optional[str]
    vol_id: Optional[str]
    reproducibility_seed: Optional[int]
    cache_key: Optional[str]
    metadata: Dict[str, Any]
```

### Caching & Persistence

#### 1. In-Memory Caching

**Features:**
- LRU eviction policy
- Configurable cache size
- Hit/miss statistics
- Cache warming strategies

**Performance:**
- Cache hit rate target: >80%
- Eviction threshold: 1000 results
- Memory usage: ~100MB for 1000 results

#### 2. Persistence Layer

**Backends:**
- PostgreSQL (structured data)
- Redis (caching)
- MongoDB (document storage)
- File system (large results)

**Data Lifecycle:**
- Hot data: In-memory cache
- Warm data: Redis cache
- Cold data: Database storage
- Archive: Compressed files

---

## API Endpoints

### OAS Calculator API

#### 1. Single Calculation
```http
POST /api/v1/oas/calculate
Content-Type: application/json

{
  "curve_id": "INR_ZERO_20241201",
  "volatility_surface": {
    "tenors": [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
    "volatilities": [0.15, 0.18, 0.20, 0.22, 0.24, 0.25, 0.26, 0.27]
  },
  "cash_flows": [...],
  "option_type": "CALLABLE",
  "call_schedule": [
    {
      "call_date": "2025-06-15",
      "call_price": 100.0,
      "notice_period_days": 30
    }
  ],
  "market_price": 98.50,
  "pricing_method": "LATTICE",
  "lattice_model": "HO_LEE",
  "lattice_steps": 500
}
```

#### 2. Batch Processing
```http
POST /api/v1/oas/calculate/batch
Content-Type: application/json

{
  "calculations": [...],
  "enable_parallel": true,
  "max_workers": 4
}
```

#### 3. Progress Monitoring
```http
GET /api/v1/oas/progress/{calculation_id}
```

### Stress Testing API

#### 1. Single Portfolio Test
```http
POST /api/v1/stress-testing/run
Content-Type: application/json

{
  "portfolio": [...],
  "scenarios": [
    {
      "scenario_id": "PARALLEL_UP_100",
      "scenario_type": "PARALLEL_SHIFT",
      "parallel_shift_bps": 100
    }
  ],
  "calculation_mode": "FAST_APPROXIMATION"
}
```

#### 2. Batch Portfolio Testing
```http
POST /api/v1/stress-testing/run/batch
Content-Type: application/json

{
  "portfolios": [...],
  "scenarios": [...],
  "calculation_mode": "FAST_APPROXIMATION",
  "enable_parallel": true,
  "max_workers": 4
}
```

#### 3. Predefined Scenarios
```http
GET /api/v1/stress-testing/scenarios/predefined
```

### Portfolio Analytics API

#### 1. Risk Metrics
```http
POST /api/v1/portfolio-analytics/metrics
Content-Type: application/json

{
  "positions": [...],
  "include_risk_metrics": true,
  "yield_curves": {"INR": "INR_ZERO_20241201"}
}
```

#### 2. Performance Attribution
```http
POST /api/v1/portfolio-analytics/attribution
Content-Type: application/json

{
  "positions_start": [...],
  "positions_end": [...],
  "period_start": "2024-01-01",
  "period_end": "2024-01-31",
  "yield_curves_start": {"INR": "INR_ZERO_20240101"},
  "yield_curves_end": {"INR": "INR_ZERO_20240131"}
}
```

#### 3. Turnover Analysis
```http
POST /api/v1/portfolio-analytics/turnover
Content-Type: application/json

{
  "positions_start": [...],
  "positions_end": [...],
  "period_start": "2024-01-01",
  "period_end": "2024-01-31"
}
```

---

## Performance Tuning Guide

### Configuration Parameters

#### 1. OAS Calculator

**Lattice Mode:**
```python
# Performance vs Accuracy trade-off
lattice_steps = 500      # Default: 500, Range: 100-1000
convergence_tolerance = 1e-6  # Default: 1e-6, Range: 1e-8 to 1e-4
max_iterations = 100     # Default: 100, Range: 50-200
```

**Monte Carlo Mode:**
```python
monte_carlo_paths = 10000    # Default: 10000, Range: 1000-100000
random_seed = None            # For reproducibility
```

#### 2. Stress Testing Engine

**Calculation Mode:**
```python
# Fast mode for real-time monitoring
calculation_mode = CalculationMode.FAST_APPROXIMATION

# Full reprice for regulatory reporting
calculation_mode = CalculationMode.FULL_REPRICE
```

**Parallel Processing:**
```python
parallel_processing = True
max_workers = 4              # Range: 1-16
```

#### 3. Portfolio Analytics

**PCA Configuration:**
```python
enable_pca = True
curve_factors = 3             # Range: 2-5
```

**Caching:**
```python
enable_caching = True
max_cache_size = 1000         # Range: 100-10000
```

### Performance Benchmarks

#### 1. OAS Calculations

| Instrument Type | Lattice Steps | Expected Time | Memory Usage |
|----------------|---------------|---------------|--------------|
| Option-free    | 100           | ≤5ms         | ~10MB        |
| Callable       | 500           | ≤30ms        | ~50MB        |
| Complex MBS    | 1000          | ≤100ms       | ~100MB       |

#### 2. Stress Testing

| Portfolio Size | Scenarios | Fast Mode | Full Reprice |
|----------------|-----------|-----------|--------------|
| 1,000          | 10        | ≤10ms     | ≤1s          |
| 10,000         | 10        | ≤100ms    | ≤10s         |
| 100,000        | 10        | ≤1s       | ≤100s        |

#### 3. Portfolio Analytics

| Operation | Portfolio Size | Expected Time |
|-----------|----------------|---------------|
| Risk Metrics | 10,000 | ≤50ms |
| Attribution | 10,000 | ≤100ms |
| Turnover | 10,000 | ≤30ms |

### Optimization Strategies

#### 1. Caching

**Cache Keys:**
```python
# OAS cache key
cache_key = hash(
    base_curve_hash + vol_surface_hash + 
    cash_flows_hash + option_type + market_price
)

# Stress test cache key
cache_key = hash(
    portfolio_hash + scenarios_hash + calculation_mode
)
```

**Cache Warming:**
- Pre-calculate common scenarios
- Warm cache during system startup
- Background cache population

#### 2. Parallel Processing

**Task Distribution:**
```python
# OAS batch processing
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [
        executor.submit(calculate_oas, calc_request)
        for calc_request in calculations
    ]
    results = [future.result() for future in futures]
```

**Load Balancing:**
- Distribute by portfolio size
- Consider instrument complexity
- Balance memory usage

#### 3. Memory Management

**Streaming Processing:**
- Process large portfolios in chunks
- Use generators for memory efficiency
- Implement pagination for results

**Resource Limits:**
```python
# Memory limits per calculation
max_memory_mb = 512
max_cpu_percent = 80
```

---

## Testing & Validation

### Unit Tests

#### 1. OAS Calculator Tests

**Consistency Tests:**
```python
def test_oas_consistency():
    """Test that OAS ≈ Z-spread for option-free bonds."""
    # Option-free bond should have OAS close to Z-spread
    oas_result = oas_calculator.calculate_oas(option_free_inputs)
    z_spread = calculate_z_spread(option_free_inputs)
    
    assert abs(oas_result.oas_bps - z_spread) < 1.0  # Within 1bp
```

**Monotonicity Tests:**
```python
def test_volatility_monotonicity():
    """Test that volatility ↑ → callable value ↓ → OAS ↑."""
    base_vol = 0.20
    high_vol = 0.30
    
    oas_low = oas_calculator.calculate_oas(inputs_low_vol)
    oas_high = oas_calculator.calculate_oas(inputs_high_vol)
    
    assert oas_high.oas_bps > oas_low.oas_bps
```

**Convergence Tests:**
```python
def test_lattice_convergence():
    """Test convergence across different step counts."""
    step_counts = [100, 250, 500, 1000]
    results = []
    
    for steps in step_counts:
        oas_calculator.lattice_steps = steps
        result = oas_calculator.calculate_oas(inputs)
        results.append(result.oas_bps)
    
    # Check convergence
    for i in range(1, len(results)):
        assert abs(results[i] - results[i-1]) < 0.1  # Within 0.1bp
```

#### 2. Stress Testing Tests

**Golden Tests:**
```python
def test_parallel_shift_100bps():
    """Test known 100bp parallel shift result."""
    scenario = StressScenario(
        scenario_id="TEST_PARALLEL_100",
        parallel_shift_bps=100
    )
    
    result = stress_engine.run_stress_test(
        portfolio=test_portfolio,
        base_curves=test_curves,
        spread_surfaces=test_spreads,
        scenario=scenario
    )
    
    # Known result: 100bp shift should give ~100bp P&L for 1-year duration
    expected_pnl_bps = -100  # Negative for rate increase
    assert abs(result.total_pnl_bps - expected_pnl_bps) < 5.0
```

**Sanity Tests:**
```python
def test_larger_shock_larger_pnl():
    """Test that larger shock → larger |P&L|."""
    small_scenario = StressScenario(parallel_shift_bps=50)
    large_scenario = StressScenario(parallel_shift_bps=200)
    
    small_result = stress_engine.run_stress_test(..., small_scenario)
    large_result = stress_engine.run_stress_test(..., large_scenario)
    
    assert abs(large_result.total_pnl_bps) > abs(small_result.total_pnl_bps)
```

#### 3. Portfolio Analytics Tests

**Small Portfolio Tests:**
```python
def test_portfolio_metrics_small():
    """Test portfolio metrics on small, hand-calculated portfolio."""
    # Create 3-position portfolio with known characteristics
    positions = [
        Position(face_value=1000, duration=1.0, market_value=1000),
        Position(face_value=2000, duration=2.0, market_value=2000),
        Position(face_value=3000, duration=3.0, market_value=3000)
    ]
    
    metrics = portfolio_analytics.calculate_portfolio_metrics(positions)
    
    # Expected: weighted average duration = (1*1000 + 2*2000 + 3*3000) / 6000 = 2.33
    expected_duration = (1*1000 + 2*2000 + 3*3000) / 6000
    assert abs(metrics.portfolio_duration - expected_duration) < 0.01
```

**Attribution Tests:**
```python
def test_attribution_decomposition():
    """Test that attribution factors sum to total return."""
    attribution = portfolio_analytics.calculate_performance_attribution(...)
    
    explained_return = (
        attribution.curve_level_contribution +
        attribution.curve_slope_contribution +
        attribution.curve_curvature_contribution +
        attribution.credit_spread_contribution +
        attribution.selection_contribution +
        attribution.trading_contribution +
        attribution.idiosyncratic_contribution
    )
    
    # Residual should be small
    assert abs(attribution.residual) < 0.001  # Within 0.1%
```

### Integration Tests

#### 1. End-to-End Workflows

**OAS → Stress Test → Analytics:**
```python
def test_end_to_end_workflow():
    """Test complete workflow from OAS to stress testing to analytics."""
    # 1. Calculate OAS
    oas_result = oas_calculator.calculate_oas(oas_inputs)
    
    # 2. Run stress test
    stress_result = stress_engine.run_stress_test(
        portfolio=portfolio_with_oas,
        scenarios=test_scenarios
    )
    
    # 3. Calculate portfolio metrics
    metrics = portfolio_analytics.calculate_portfolio_metrics(
        positions=portfolio_with_oas
    )
    
    # Verify consistency
    assert metrics.positions_count == len(portfolio_with_oas)
    assert stress_result.positions_processed == len(portfolio_with_oas)
```

#### 2. API Integration Tests

**REST API Tests:**
```python
def test_oas_api_integration():
    """Test OAS calculation via REST API."""
    response = client.post(
        "/api/v1/oas/calculate",
        json=oas_request_data
    )
    
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert "oas_bps" in response.json()["data"]
```

**WebSocket Tests:**
```python
def test_websocket_progress_updates():
    """Test WebSocket progress updates for long-running calculations."""
    with client.websocket_connect("/ws/oas/progress") as websocket:
        # Start calculation
        client.post("/api/v1/oas/calculate", json=request_data)
        
        # Receive progress updates
        progress = websocket.receive_json()
        assert progress["message_type"] == "OAS_PROGRESS_UPDATE"
```

### Performance Tests

#### 1. Load Testing

**Concurrent Users:**
```python
def test_concurrent_oas_calculations():
    """Test system under concurrent OAS calculation load."""
    import asyncio
    import time
    
    async def calculate_oas_concurrent():
        start_time = time.time()
        await client.post("/api/v1/oas/calculate", json=request_data)
        return time.time() - start_time
    
    # Run 100 concurrent calculations
    start_time = time.time()
    tasks = [calculate_oas_concurrent() for _ in range(100)]
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    # Verify performance targets
    avg_time = sum(results) / len(results)
    assert avg_time < 0.1  # Average < 100ms
    assert total_time < 10  # Total < 10s
```

**Memory Usage Tests:**
```python
def test_memory_usage():
    """Test memory usage under load."""
    import psutil
    import gc
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run calculations
    for _ in range(100):
        oas_calculator.calculate_oas(test_inputs)
    
    # Force garbage collection
    gc.collect()
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable
    assert memory_increase < 100  # < 100MB increase
```

---

## Deployment & Configuration

### Environment Configuration

#### 1. Production Settings

**OAS Calculator:**
```yaml
# config/production.yaml
oas_calculator:
  default_pricing_method: "LATTICE"
  default_lattice_steps: 500
  default_monte_carlo_paths: 10000
  convergence_tolerance: 1e-6
  max_iterations: 100
  enable_caching: true
  cache_size: 1000
```

**Stress Testing Engine:**
```yaml
stress_testing:
  default_calculation_mode: "FAST_APPROXIMATION"
  parallel_processing: true
  max_workers: 8
  cache_results: true
  max_cache_size: 2000
```

**Portfolio Analytics:**
```yaml
portfolio_analytics:
  enable_pca: true
  curve_factors: 3
  attribution_method: "FACTOR_MODEL"
  enable_caching: true
  max_cache_size: 1500
```

#### 2. Staging Settings

**Development Configuration:**
```yaml
# config/staging.yaml
oas_calculator:
  default_lattice_steps: 250  # Faster for development
  default_monte_carlo_paths: 5000
  convergence_tolerance: 1e-4  # Less strict

stress_testing:
  max_workers: 4  # Smaller for staging
  cache_results: false  # Disable caching for testing

portfolio_analytics:
  curve_factors: 2  # Simpler for development
```

### Monitoring & Observability

#### 1. Metrics Collection

**Performance Metrics:**
```python
# Custom metrics
oas_calculation_duration = Histogram(
    'oas_calculation_duration_seconds',
    'OAS calculation duration in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
)

stress_test_duration = Histogram(
    'stress_test_duration_seconds',
    'Stress test duration in seconds',
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
)

portfolio_metrics_duration = Histogram(
    'portfolio_metrics_duration_seconds',
    'Portfolio metrics calculation duration in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)
```

**Business Metrics:**
```python
# Cache performance
cache_hit_rate = Gauge(
    'cache_hit_rate',
    'Cache hit rate percentage'
)

# Error rates
oas_calculation_errors = Counter(
    'oas_calculation_errors_total',
    'Total OAS calculation errors'
)

# Throughput
calculations_per_second = Counter(
    'calculations_per_second_total',
    'Total calculations per second'
)
```

#### 2. Health Checks

**System Health:**
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "components": {
            "oas_calculator": check_oas_calculator_health(),
            "stress_testing": check_stress_testing_health(),
            "portfolio_analytics": check_portfolio_analytics_health(),
            "cache": check_cache_health(),
            "database": check_database_health()
        }
    }
```

**Component Health:**
```python
def check_oas_calculator_health():
    try:
        # Test calculation
        test_inputs = create_test_oas_inputs()
        result = oas_calculator.calculate_oas(test_inputs)
        return {"status": "healthy", "response_time_ms": result.solve_time_ms}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

#### 3. Logging Configuration

**Structured Logging:**
```python
# config/logging.yaml
version: 1
formatters:
  structured:
    format: '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "extra": %(extra)s}'

handlers:
  console:
    class: logging.StreamHandler
    formatter: structured
    level: INFO
  
  file:
    class: logging.handlers.RotatingFileHandler
    filename: logs/bondx_phase_b.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    formatter: structured
    level: DEBUG

loggers:
  bondx.mathematics.option_adjusted_spread:
    level: DEBUG
    handlers: [console, file]
  
  bondx.risk_management.stress_testing:
    level: DEBUG
    handlers: [console, file]
  
  bondx.risk_management.portfolio_analytics:
    level: DEBUG
    handlers: [console, file]
```

### Scaling & Performance

#### 1. Horizontal Scaling

**Load Balancer Configuration:**
```nginx
# nginx.conf
upstream bondx_backend {
    server bondx-1:8000;
    server bondx-2:8000;
    server bondx-3:8000;
    server bondx-4:8000;
}

server {
    listen 80;
    server_name api.bondx.com;
    
    location /api/v1/ {
        proxy_pass http://bondx_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # Timeout settings for long-running calculations
        proxy_read_timeout 300s;
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
    }
}
```

**Kubernetes Deployment:**
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bondx-backend
spec:
  replicas: 4
  selector:
    matchLabels:
      app: bondx-backend
  template:
    metadata:
      labels:
        app: bondx-backend
    spec:
      containers:
      - name: bondx-backend
        image: bondx/backend:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        env:
        - name: MAX_WORKERS
          value: "2"  # 2 workers per pod, 4 pods = 8 total
        - name: CACHE_SIZE
          value: "250"  # 250 per pod, 4 pods = 1000 total
```

#### 2. Caching Strategy

**Redis Configuration:**
```yaml
# docker-compose.yml
services:
  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  redis-sentinel:
    image: redis:7-alpine
    command: redis-sentinel /usr/local/etc/redis/sentinel.conf
    ports:
      - "26379:26379"
    volumes:
      - ./redis/sentinel.conf:/usr/local/etc/redis/sentinel.conf
```

**Cache Distribution:**
```python
# Cache key distribution across Redis instances
def get_cache_instance(cache_key: str) -> str:
    """Distribute cache keys across Redis instances."""
    hash_value = hash(cache_key)
    instance_count = 3  # Number of Redis instances
    
    return f"redis-{hash_value % instance_count}"
```

---

## Conclusion

Phase B of the BondX system provides a comprehensive, production-ready implementation of advanced financial modeling capabilities. The system is designed for:

- **Performance**: Meeting strict latency requirements for real-time operations
- **Scalability**: Supporting large portfolios and high-throughput scenarios
- **Reliability**: Robust error handling and validation frameworks
- **Maintainability**: Clean architecture with comprehensive testing and documentation

The implementation follows industry best practices and provides a solid foundation for future enhancements including Phase C components (correlation matrices, advanced volatility models, and PDE-based pricing engines).

For questions or support, please refer to the API documentation or contact the development team.
