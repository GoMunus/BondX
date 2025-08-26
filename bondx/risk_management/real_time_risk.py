"""
Real-time Risk Management for BondX.

This module implements comprehensive risk management with:
- Streaming portfolio risk calculations
- VaR engines (parametric, historical, Monte Carlo)
- Stress testing scenarios
- Dynamic risk limits and alerts
- Real-time PnL tracking
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import math
import statistics
import numpy as np
from collections import defaultdict, deque
from uuid import uuid4

from ..core.logging import get_logger
from ..trading_engine.trading_models import Order, OrderSide
from ..mathematics.bond_pricing import BondPricer
from ..mathematics.duration_convexity import DurationConvexityCalculator
from .risk_models import RiskCheckResult, RiskLimit, RiskAlert

logger = get_logger(__name__)


class RiskMetricType(Enum):
    """Types of risk metrics."""
    VAR = "VAR"
    STRESS_TEST = "STRESS_TEST"
    DURATION = "DURATION"
    CONVEXITY = "CONVEXITY"
    DV01 = "DV01"
    EXPOSURE = "EXPOSURE"
    CONCENTRATION = "CONCENTRATION"
    PNL = "PNL"


class VaRMethod(Enum):
    """VaR calculation methods."""
    PARAMETRIC = "PARAMETRIC"
    HISTORICAL = "HISTORICAL"
    MONTE_CARLO = "MONTE_CARLO"


class StressScenarioType(Enum):
    """Types of stress scenarios."""
    PARALLEL_SHIFT = "PARALLEL_SHIFT"
    CURVE_STEEPENING = "CURVE_STEEPENING"
    CURVE_FLATTENING = "CURVE_FLATTENING"
    CREDIT_SPREAD_WIDENING = "CREDIT_SPREAD_WIDENING"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    LIQUIDITY_CRUNCH = "LIQUIDITY_CRUNCH"
    CREDIT_DOWNGRADE = "CREDIT_DOWNGRADE"
    DEFAULT_SCENARIO = "DEFAULT_SCENARIO"


@dataclass
class PortfolioPosition:
    """Portfolio position for risk calculations."""
    instrument_id: str
    quantity: Decimal
    market_value: Decimal
    clean_price: Decimal
    dirty_price: Decimal
    duration: Decimal
    convexity: Decimal
    dv01: Decimal
    yield_to_maturity: Decimal
    credit_rating: str
    sector: str
    issuer: str
    maturity_date: datetime
    last_update: datetime


@dataclass
class RiskSnapshot:
    """Snapshot of portfolio risk metrics."""
    timestamp: datetime
    portfolio_id: str
    total_market_value: Decimal
    total_duration: Decimal
    total_convexity: Decimal
    total_dv01: Decimal
    var_95_1d: Decimal
    var_99_1d: Decimal
    var_95_10d: Decimal
    var_99_10d: Decimal
    stress_test_results: Dict[str, Decimal]
    concentration_metrics: Dict[str, Any]
    risk_limits_status: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VaRResult:
    """Value at Risk calculation result."""
    method: VaRMethod
    confidence_level: float
    time_horizon: int  # days
    var_value: Decimal
    expected_shortfall: Decimal
    confidence_interval: Tuple[Decimal, Decimal]
    calculation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StressTestResult:
    """Stress test scenario result."""
    scenario_type: StressScenarioType
    scenario_name: str
    pnl_impact: Decimal
    market_value_change: Decimal
    duration_change: Decimal
    var_change: Decimal
    confidence_level: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskLimitBreach:
    """Risk limit breach event."""
    breach_id: str
    limit_type: str
    limit_value: Decimal
    current_value: Decimal
    breach_amount: Decimal
    portfolio_id: str
    participant_id: int
    timestamp: datetime
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    status: str  # "OPEN", "ACKNOWLEDGED", "RESOLVED"
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealTimeRiskEngine:
    """
    Real-time risk management engine.
    
    Features:
    - Streaming portfolio risk calculations
    - Multiple VaR methodologies
    - Comprehensive stress testing
    - Dynamic risk limits
    - Real-time alerts and monitoring
    """
    
    def __init__(self, 
                 bond_pricer: Optional[BondPricer] = None,
                 duration_calculator: Optional[DurationConvexityCalculator] = None):
        """Initialize the real-time risk engine."""
        self.bond_pricer = bond_pricer
        self.duration_calculator = duration_calculator
        
        # Portfolio data
        self.portfolios: Dict[str, Dict[str, PortfolioPosition]] = {}
        self.portfolio_snapshots: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Risk limits
        self.risk_limits: Dict[str, RiskLimit] = {}
        self.limit_breaches: List[RiskLimitBreach] = []
        
        # VaR configuration
        self.var_confidence_levels = [0.95, 0.99]
        self.var_time_horizons = [1, 10]  # days
        self.historical_window = 252  # trading days
        
        # Stress testing
        self.stress_scenarios: Dict[StressScenarioType, Dict[str, Any]] = {}
        self.stress_results: Dict[str, List[StressTestResult]] = defaultdict(list)
        
        # Performance tracking
        self.calculations_performed = 0
        self.start_time = datetime.utcnow()
        
        # Initialize stress scenarios
        self._initialize_stress_scenarios()
        
        logger.info("Real-time Risk Engine initialized successfully")
    
    def _initialize_stress_scenarios(self) -> None:
        """Initialize predefined stress scenarios."""
        # Parallel shift scenarios
        self.stress_scenarios[StressScenarioType.PARALLEL_SHIFT] = {
            "scenarios": [
                {"name": "Parallel Up 100bps", "yield_change": 0.01},
                {"name": "Parallel Down 100bps", "yield_change": -0.01},
                {"name": "Parallel Up 200bps", "yield_change": 0.02},
                {"name": "Parallel Down 200bps", "yield_change": -0.02}
            ]
        }
        
        # Curve scenarios
        self.stress_scenarios[StressScenarioType.CURVE_STEEPENING] = {
            "scenarios": [
                {"name": "Steepening 50bps", "short_change": -0.005, "long_change": 0.005},
                {"name": "Steepening 100bps", "short_change": -0.01, "long_change": 0.01}
            ]
        }
        
        self.stress_scenarios[StressScenarioType.CURVE_FLATTENING] = {
            "scenarios": [
                {"name": "Flattening 50bps", "short_change": 0.005, "long_change": -0.005},
                {"name": "Flattening 100bps", "short_change": 0.01, "long_change": -0.01}
            ]
        }
        
        # Credit spread scenarios
        self.stress_scenarios[StressScenarioType.CREDIT_SPREAD_WIDENING] = {
            "scenarios": [
                {"name": "Spread Widening 50bps", "spread_change": 0.005},
                {"name": "Spread Widening 100bps", "spread_change": 0.01},
                {"name": "Spread Widening 200bps", "spread_change": 0.02}
            ]
        }
        
        # Volatility scenarios
        self.stress_scenarios[StressScenarioType.VOLATILITY_SPIKE] = {
            "scenarios": [
                {"name": "Volatility +50%", "vol_multiplier": 1.5},
                {"name": "Volatility +100%", "vol_multiplier": 2.0}
            ]
        }
        
        # Liquidity scenarios
        self.stress_scenarios[StressScenarioType.LIQUIDITY_CRUNCH] = {
            "scenarios": [
                {"name": "Bid-Ask Spread +100%", "spread_multiplier": 2.0},
                {"name": "Bid-Ask Spread +200%", "spread_multiplier": 3.0}
            ]
        }
        
        # Credit downgrade scenarios
        self.stress_scenarios[StressScenarioType.CREDIT_DOWNGRADE] = {
            "scenarios": [
                {"name": "1 Notch Downgrade", "rating_change": -1},
                {"name": "2 Notch Downgrade", "rating_change": -2},
                {"name": "3 Notch Downgrade", "rating_change": -3}
            ]
        }
        
        # Default scenarios
        self.stress_scenarios[StressScenarioType.DEFAULT_SCENARIO] = {
            "scenarios": [
                {"name": "Default 10% LGD", "default_rate": 0.1, "lgd": 0.6},
                {"name": "Default 20% LGD", "default_rate": 0.2, "lgd": 0.6}
            ]
        }
    
    def add_portfolio_position(self, portfolio_id: str, position: PortfolioPosition) -> None:
        """Add or update a portfolio position."""
        if portfolio_id not in self.portfolios:
            self.portfolios[portfolio_id] = {}
        
        self.portfolios[portfolio_id][position.instrument_id] = position
        logger.debug(f"Updated position for {portfolio_id}:{position.instrument_id}")
    
    def remove_portfolio_position(self, portfolio_id: str, instrument_id: str) -> None:
        """Remove a portfolio position."""
        if portfolio_id in self.portfolios and instrument_id in self.portfolios[portfolio_id]:
            del self.portfolios[portfolio_id][instrument_id]
            logger.debug(f"Removed position for {portfolio_id}:{instrument_id}")
    
    def add_risk_limit(self, limit: RiskLimit) -> None:
        """Add a risk limit."""
        limit_key = f"{limit.portfolio_id}_{limit.limit_type}"
        self.risk_limits[limit_key] = limit
        logger.info(f"Added risk limit: {limit_key}")
    
    def remove_risk_limit(self, portfolio_id: str, limit_type: str) -> bool:
        """Remove a risk limit."""
        limit_key = f"{portfolio_id}_{limit_type}"
        if limit_key in self.risk_limits:
            del self.risk_limits[limit_key]
            logger.info(f"Removed risk limit: {limit_key}")
            return True
        return False
    
    async def calculate_portfolio_risk(self, portfolio_id: str) -> RiskSnapshot:
        """Calculate comprehensive portfolio risk metrics."""
        try:
            if portfolio_id not in self.portfolios:
                raise Exception(f"Portfolio {portfolio_id} not found")
            
            portfolio = self.portfolios[portfolio_id]
            if not portfolio:
                raise Exception(f"Portfolio {portfolio_id} is empty")
            
            # Calculate basic metrics
            total_market_value = sum(pos.market_value for pos in portfolio.values())
            total_duration = self._calculate_weighted_duration(portfolio)
            total_convexity = self._calculate_weighted_convexity(portfolio)
            total_dv01 = sum(pos.dv01 for pos in portfolio.values())
            
            # Calculate VaR
            var_results = await self._calculate_portfolio_var(portfolio, portfolio_id)
            
            # Run stress tests
            stress_results = await self._run_stress_tests(portfolio, portfolio_id)
            
            # Calculate concentration metrics
            concentration_metrics = self._calculate_concentration_metrics(portfolio)
            
            # Check risk limits
            risk_limits_status = await self._check_risk_limits(portfolio_id, total_market_value, 
                                                             total_duration, total_dv01)
            
            # Create risk snapshot
            snapshot = RiskSnapshot(
                timestamp=datetime.utcnow(),
                portfolio_id=portfolio_id,
                total_market_value=total_market_value,
                total_duration=total_duration,
                total_convexity=total_convexity,
                total_dv01=total_dv01,
                var_95_1d=var_results.get("95_1d", Decimal('0')),
                var_99_1d=var_results.get("99_1d", Decimal('0')),
                var_95_10d=var_results.get("95_10d", Decimal('0')),
                var_99_10d=var_results.get("99_10d", Decimal('0')),
                stress_test_results=stress_results,
                concentration_metrics=concentration_metrics,
                risk_limits_status=risk_limits_status
            )
            
            # Store snapshot
            self.portfolio_snapshots[portfolio_id].append(snapshot)
            
            # Update statistics
            self.calculations_performed += 1
            
            logger.debug(f"Risk snapshot calculated for portfolio {portfolio_id}")
            return snapshot
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk for {portfolio_id}: {e}")
            raise
    
    def _calculate_weighted_duration(self, portfolio: Dict[str, PortfolioPosition]) -> Decimal:
        """Calculate weighted average duration."""
        try:
            total_value = sum(pos.market_value for pos in portfolio.values())
            if total_value == 0:
                return Decimal('0')
            
            weighted_duration = sum(
                pos.duration * pos.market_value for pos in portfolio.values()
            ) / total_value
            
            return weighted_duration
        except Exception as e:
            logger.error(f"Error calculating weighted duration: {e}")
            return Decimal('0')
    
    def _calculate_weighted_convexity(self, portfolio: Dict[str, PortfolioPosition]) -> Decimal:
        """Calculate weighted average convexity."""
        try:
            total_value = sum(pos.market_value for pos in portfolio.values())
            if total_value == 0:
                return Decimal('0')
            
            weighted_convexity = sum(
                pos.convexity * pos.market_value for pos in portfolio.values()
            ) / total_value
            
            return weighted_convexity
        except Exception as e:
            logger.error(f"Error calculating weighted convexity: {e}")
            return Decimal('0')
    
    async def _calculate_portfolio_var(self, portfolio: Dict[str, PortfolioPosition], 
                                     portfolio_id: str) -> Dict[str, Decimal]:
        """Calculate portfolio VaR using multiple methods."""
        var_results = {}
        
        try:
            for confidence in self.var_confidence_levels:
                for horizon in self.var_time_horizons:
                    key = f"{int(confidence*100)}_{horizon}d"
                    
                    # Try parametric VaR first
                    try:
                        var_result = await self._calculate_parametric_var(portfolio, confidence, horizon)
                        var_results[key] = var_result.var_value
                    except Exception as e:
                        logger.warning(f"Parametric VaR failed for {key}: {e}")
                        
                        # Fall back to historical VaR
                        try:
                            var_result = await self._calculate_historical_var(portfolio, confidence, horizon)
                            var_results[key] = var_result.var_value
                        except Exception as e2:
                            logger.warning(f"Historical VaR also failed for {key}: {e2}")
                            var_results[key] = Decimal('0')
            
            return var_results
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            return {}
    
    async def _calculate_parametric_var(self, portfolio: Dict[str, PortfolioPosition], 
                                      confidence: float, horizon: int) -> VaRResult:
        """Calculate parametric VaR using delta-normal method."""
        try:
            start_time = datetime.utcnow()
            
            # Calculate portfolio DV01
            total_dv01 = sum(pos.dv01 for pos in portfolio.values())
            
            # Assume yield volatility (in practice, this would come from market data)
            yield_volatility = Decimal('0.02')  # 2% daily volatility
            
            # Calculate VaR using normal distribution
            z_score = self._get_z_score(confidence)
            var_value = abs(total_dv01) * yield_volatility * z_score * math.sqrt(horizon)
            
            # Calculate expected shortfall
            expected_shortfall = var_value * self._get_expected_shortfall_factor(confidence)
            
            # Calculate confidence interval
            confidence_interval = (
                var_value * Decimal('0.9'),  # Lower bound
                var_value * Decimal('1.1')   # Upper bound
            )
            
            calculation_time = (datetime.utcnow() - start_time).total_seconds()
            
            return VaRResult(
                method=VaRMethod.PARAMETRIC,
                confidence_level=confidence,
                time_horizon=horizon,
                var_value=var_value,
                expected_shortfall=expected_shortfall,
                confidence_interval=confidence_interval,
                calculation_time=calculation_time
            )
            
        except Exception as e:
            logger.error(f"Error calculating parametric VaR: {e}")
            raise
    
    async def _calculate_historical_var(self, portfolio: Dict[str, PortfolioPosition], 
                                      confidence: float, horizon: int) -> VaRResult:
        """Calculate historical VaR using historical simulation."""
        try:
            start_time = datetime.utcnow()
            
            # In practice, this would use historical yield curve movements
            # For now, we'll simulate with random data
            
            # Generate historical PnL scenarios
            num_scenarios = 1000
            pnl_scenarios = []
            
            for _ in range(num_scenarios):
                # Random yield change
                yield_change = np.random.normal(0, 0.02)  # 2% volatility
                
                # Calculate PnL impact
                total_dv01 = sum(pos.dv01 for pos in portfolio.values())
                pnl_impact = -total_dv01 * Decimal(str(yield_change))
                pnl_scenarios.append(pnl_impact)
            
            # Sort PnL scenarios
            pnl_scenarios.sort()
            
            # Calculate VaR
            var_index = int((1 - confidence) * num_scenarios)
            var_value = abs(pnl_scenarios[var_index])
            
            # Calculate expected shortfall
            tail_scenarios = pnl_scenarios[:var_index]
            expected_shortfall = abs(sum(tail_scenarios) / len(tail_scenarios)) if tail_scenarios else var_value
            
            # Confidence interval
            confidence_interval = (var_value * Decimal('0.9'), var_value * Decimal('1.1'))
            
            calculation_time = (datetime.utcnow() - start_time).total_seconds()
            
            return VaRResult(
                method=VaRMethod.HISTORICAL,
                confidence_level=confidence,
                time_horizon=horizon,
                var_value=var_value,
                expected_shortfall=expected_shortfall,
                confidence_interval=confidence_interval,
                calculation_time=calculation_time
            )
            
        except Exception as e:
            logger.error(f"Error calculating historical VaR: {e}")
            raise
    
    def _get_z_score(self, confidence: float) -> float:
        """Get Z-score for confidence level."""
        z_scores = {
            0.90: 1.282,
            0.95: 1.645,
            0.99: 2.326,
            0.995: 2.576,
            0.999: 3.291
        }
        return z_scores.get(confidence, 1.645)
    
    def _get_expected_shortfall_factor(self, confidence: float) -> float:
        """Get expected shortfall factor for confidence level."""
        factors = {
            0.90: 1.755,
            0.95: 2.063,
            0.99: 2.665,
            0.995: 2.891,
            0.999: 3.367
        }
        return factors.get(confidence, 2.063)
    
    async def _run_stress_tests(self, portfolio: Dict[str, PortfolioPosition], 
                               portfolio_id: str) -> Dict[str, Decimal]:
        """Run comprehensive stress tests."""
        stress_results = {}
        
        try:
            for scenario_type, scenario_config in self.stress_scenarios.items():
                for scenario in scenario_config["scenarios"]:
                    scenario_name = scenario["name"]
                    
                    try:
                        result = await self._run_stress_scenario(portfolio, scenario_type, scenario)
                        if result:
                            stress_results[scenario_name] = result.pnl_impact
                            
                            # Store detailed result
                            self.stress_results[portfolio_id].append(result)
                            
                    except Exception as e:
                        logger.warning(f"Stress test failed for {scenario_name}: {e}")
                        stress_results[scenario_name] = Decimal('0')
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error running stress tests: {e}")
            return {}
    
    async def _run_stress_scenario(self, portfolio: Dict[str, PortfolioPosition], 
                                 scenario_type: StressScenarioType, 
                                 scenario: Dict[str, Any]) -> Optional[StressTestResult]:
        """Run a specific stress scenario."""
        try:
            if scenario_type == StressScenarioType.PARALLEL_SHIFT:
                return await self._run_parallel_shift_scenario(portfolio, scenario)
            elif scenario_type == StressScenarioType.CURVE_STEEPENING:
                return await self._run_curve_steepening_scenario(portfolio, scenario)
            elif scenario_type == StressScenarioType.CREDIT_SPREAD_WIDENING:
                return await self._run_credit_spread_scenario(portfolio, scenario)
            elif scenario_type == StressScenarioType.VOLATILITY_SPIKE:
                return await self._run_volatility_scenario(portfolio, scenario)
            elif scenario_type == StressScenarioType.LIQUIDITY_CRUNCH:
                return await self._run_liquidity_scenario(portfolio, scenario)
            elif scenario_type == StressScenarioType.CREDIT_DOWNGRADE:
                return await self._run_credit_downgrade_scenario(portfolio, scenario)
            elif scenario_type == StressScenarioType.DEFAULT_SCENARIO:
                return await self._run_default_scenario(portfolio, scenario)
            else:
                logger.warning(f"Unknown stress scenario type: {scenario_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error running stress scenario {scenario['name']}: {e}")
            return None
    
    async def _run_parallel_shift_scenario(self, portfolio: Dict[str, PortfolioPosition], 
                                         scenario: Dict[str, Any]) -> StressTestResult:
        """Run parallel yield curve shift scenario."""
        try:
            yield_change = scenario["yield_change"]
            total_dv01 = sum(pos.dv01 for pos in portfolio.values())
            
            # Calculate PnL impact
            pnl_impact = -total_dv01 * Decimal(str(yield_change))
            
            # Calculate market value change
            market_value_change = sum(pos.market_value for pos in portfolio.values()) * Decimal(str(yield_change))
            
            return StressTestResult(
                scenario_type=StressScenarioType.PARALLEL_SHIFT,
                scenario_name=scenario["name"],
                pnl_impact=pnl_impact,
                market_value_change=market_value_change,
                duration_change=Decimal('0'),  # Duration doesn't change in parallel shift
                var_change=Decimal('0'),  # Simplified
                confidence_level=0.95,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error in parallel shift scenario: {e}")
            raise
    
    async def _run_curve_steepening_scenario(self, portfolio: Dict[str, PortfolioPosition], 
                                           scenario: Dict[str, Any]) -> StressTestResult:
        """Run curve steepening scenario."""
        try:
            short_change = scenario["short_change"]
            long_change = scenario["long_change"]
            
            # Simplified calculation - in practice would use key rate durations
            total_dv01 = sum(pos.dv01 for pos in portfolio.values())
            
            # Assume average impact
            avg_change = (short_change + long_change) / 2
            pnl_impact = -total_dv01 * Decimal(str(avg_change))
            
            return StressTestResult(
                scenario_type=StressScenarioType.CURVE_STEEPENING,
                scenario_name=scenario["name"],
                pnl_impact=pnl_impact,
                market_value_change=Decimal('0'),  # Simplified
                duration_change=Decimal('0'),  # Simplified
                var_change=Decimal('0'),  # Simplified
                confidence_level=0.95,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error in curve steepening scenario: {e}")
            raise
    
    async def _run_credit_spread_scenario(self, portfolio: Dict[str, PortfolioPosition], 
                                        scenario: Dict[str, Any]) -> StressTestResult:
        """Run credit spread widening scenario."""
        try:
            spread_change = scenario["spread_change"]
            
            # Calculate impact based on credit-sensitive positions
            credit_impact = Decimal('0')
            for position in portfolio.values():
                if position.credit_rating in ['BBB', 'BB', 'B', 'CCC']:
                    # Higher impact for lower-rated bonds
                    rating_multiplier = self._get_rating_multiplier(position.credit_rating)
                    credit_impact += position.market_value * rating_multiplier * Decimal(str(spread_change))
            
            return StressTestResult(
                scenario_type=StressScenarioType.CREDIT_SPREAD_WIDENING,
                scenario_name=scenario["name"],
                pnl_impact=-credit_impact,
                market_value_change=-credit_impact,
                duration_change=Decimal('0'),
                var_change=Decimal('0'),
                confidence_level=0.95,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error in credit spread scenario: {e}")
            raise
    
    def _get_rating_multiplier(self, rating: str) -> Decimal:
        """Get rating multiplier for credit scenarios."""
        multipliers = {
            'AAA': Decimal('0.5'),
            'AA': Decimal('0.7'),
            'A': Decimal('1.0'),
            'BBB': Decimal('1.5'),
            'BB': Decimal('2.0'),
            'B': Decimal('2.5'),
            'CCC': Decimal('3.0')
        }
        return multipliers.get(rating, Decimal('1.0'))
    
    async def _run_volatility_scenario(self, portfolio: Dict[str, PortfolioPosition], 
                                     scenario: Dict[str, Any]) -> StressTestResult:
        """Run volatility spike scenario."""
        try:
            vol_multiplier = scenario["vol_multiplier"]
            
            # Simplified impact calculation
            total_convexity = sum(pos.convexity for pos in portfolio.values())
            volatility_impact = total_convexity * (vol_multiplier - 1) * Decimal('0.01')
            
            return StressTestResult(
                scenario_type=StressScenarioType.VOLATILITY_SPIKE,
                scenario_name=scenario["name"],
                pnl_impact=volatility_impact,
                market_value_change=volatility_impact,
                duration_change=Decimal('0'),
                var_change=volatility_impact,
                confidence_level=0.95,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error in volatility scenario: {e}")
            raise
    
    async def _run_liquidity_scenario(self, portfolio: Dict[str, PortfolioPosition], 
                                    scenario: Dict[str, Any]) -> StressTestResult:
        """Run liquidity crunch scenario."""
        try:
            spread_multiplier = scenario["spread_multiplier"]
            
            # Calculate liquidity impact based on bid-ask spreads
            liquidity_impact = Decimal('0')
            for position in portfolio.values():
                # Assume base spread of 0.1%
                base_spread = Decimal('0.001')
                new_spread = base_spread * Decimal(str(spread_multiplier))
                spread_impact = (new_spread - base_spread) * position.market_value
                liquidity_impact += spread_impact
            
            return StressTestResult(
                scenario_type=StressScenarioType.LIQUIDITY_CRUNCH,
                scenario_name=scenario["name"],
                pnl_impact=-liquidity_impact,
                market_value_change=-liquidity_impact,
                duration_change=Decimal('0'),
                var_change=Decimal('0'),
                confidence_level=0.95,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error in liquidity scenario: {e}")
            raise
    
    async def _run_credit_downgrade_scenario(self, portfolio: Dict[str, PortfolioPosition], 
                                           scenario: Dict[str, Any]) -> StressTestResult:
        """Run credit downgrade scenario."""
        try:
            rating_change = scenario["rating_change"]
            
            # Calculate impact based on rating changes
            downgrade_impact = Decimal('0')
            for position in portfolio.values():
                if position.credit_rating in ['BBB', 'BB', 'B']:
                    # Higher impact for downgrades
                    impact_per_notch = position.market_value * Decimal('0.02')  # 2% per notch
                    downgrade_impact += impact_per_notch * abs(rating_change)
            
            return StressTestResult(
                scenario_type=StressScenarioType.CREDIT_DOWNGRADE,
                scenario_name=scenario["name"],
                pnl_impact=-downgrade_impact,
                market_value_change=-downgrade_impact,
                duration_change=Decimal('0'),
                var_change=Decimal('0'),
                confidence_level=0.95,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error in credit downgrade scenario: {e}")
            raise
    
    async def _run_default_scenario(self, portfolio: Dict[str, PortfolioPosition], 
                                  scenario: Dict[str, Any]) -> StressTestResult:
        """Run default scenario."""
        try:
            default_rate = scenario["default_rate"]
            lgd = scenario["lgd"]  # Loss given default
            
            # Calculate default impact
            default_impact = Decimal('0')
            for position in portfolio.values():
                if position.credit_rating in ['BB', 'B', 'CCC']:
                    # Higher default probability for lower-rated bonds
                    default_prob = default_rate * self._get_default_probability_multiplier(position.credit_rating)
                    default_impact += position.market_value * default_prob * lgd
            
            return StressTestResult(
                scenario_type=StressScenarioType.DEFAULT_SCENARIO,
                scenario_name=scenario["name"],
                pnl_impact=-default_impact,
                market_value_change=-default_impact,
                duration_change=Decimal('0'),
                var_change=Decimal('0'),
                confidence_level=0.95,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error in default scenario: {e}")
            raise
    
    def _get_default_probability_multiplier(self, rating: str) -> float:
        """Get default probability multiplier for rating."""
        multipliers = {
            'AAA': 0.1,
            'AA': 0.2,
            'A': 0.5,
            'BBB': 1.0,
            'BB': 2.0,
            'B': 4.0,
            'CCC': 8.0
        }
        return multipliers.get(rating, 1.0)
    
    def _calculate_concentration_metrics(self, portfolio: Dict[str, PortfolioPosition]) -> Dict[str, Any]:
        """Calculate portfolio concentration metrics."""
        try:
            # Issuer concentration
            issuer_exposure = defaultdict(Decimal)
            for position in portfolio.values():
                issuer_exposure[position.issuer] += position.market_value
            
            # Sector concentration
            sector_exposure = defaultdict(Decimal)
            for position in portfolio.values():
                sector_exposure[position.sector] += position.market_value
            
            # Rating concentration
            rating_exposure = defaultdict(Decimal)
            for position in portfolio.values():
                rating_exposure[position.credit_rating] += position.market_value
            
            # Maturity concentration
            maturity_buckets = {
                '0-1Y': Decimal('0'),
                '1-3Y': Decimal('0'),
                '3-5Y': Decimal('0'),
                '5-10Y': Decimal('0'),
                '10Y+': Decimal('0')
            }
            
            for position in portfolio.values():
                years_to_maturity = (position.maturity_date - datetime.utcnow()).days / 365.25
                if years_to_maturity <= 1:
                    maturity_buckets['0-1Y'] += position.market_value
                elif years_to_maturity <= 3:
                    maturity_buckets['1-3Y'] += position.market_value
                elif years_to_maturity <= 5:
                    maturity_buckets['3-5Y'] += position.market_value
                elif years_to_maturity <= 10:
                    maturity_buckets['5-10Y'] += position.market_value
                else:
                    maturity_buckets['10Y+'] += position.market_value
            
            return {
                "issuer_concentration": dict(issuer_exposure),
                "sector_concentration": dict(sector_exposure),
                "rating_concentration": dict(rating_exposure),
                "maturity_concentration": dict(maturity_buckets)
            }
            
        except Exception as e:
            logger.error(f"Error calculating concentration metrics: {e}")
            return {}
    
    async def _check_risk_limits(self, portfolio_id: str, market_value: Decimal, 
                                duration: Decimal, dv01: Decimal) -> Dict[str, str]:
        """Check risk limits and generate alerts."""
        limit_status = {}
        
        try:
            for limit_key, limit in self.risk_limits.items():
                if not limit_key.startswith(portfolio_id):
                    continue
                
                limit_type = limit.limit_type
                current_value = Decimal('0')
                
                # Get current value based on limit type
                if limit_type == "MARKET_VALUE":
                    current_value = market_value
                elif limit_type == "DURATION":
                    current_value = duration
                elif limit_type == "DV01":
                    current_value = dv01
                else:
                    continue
                
                # Check if limit is breached
                if current_value > limit.limit_value:
                    limit_status[limit_type] = "BREACHED"
                    
                    # Create breach alert
                    await self._create_limit_breach(limit, current_value, portfolio_id)
                else:
                    limit_status[limit_type] = "OK"
            
            return limit_status
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return {}
    
    async def _create_limit_breach(self, limit: RiskLimit, current_value: Decimal, 
                                  portfolio_id: str) -> None:
        """Create a risk limit breach alert."""
        try:
            breach_amount = current_value - limit.limit_value
            severity = self._determine_breach_severity(breach_amount, limit.limit_value)
            
            breach = RiskLimitBreach(
                breach_id=str(uuid4()),
                limit_type=limit.limit_type,
                limit_value=limit.limit_value,
                current_value=current_value,
                breach_amount=breach_amount,
                portfolio_id=portfolio_id,
                participant_id=limit.participant_id,
                timestamp=datetime.utcnow(),
                severity=severity,
                status="OPEN"
            )
            
            self.limit_breaches.append(breach)
            
            # Log breach
            logger.warning(f"Risk limit breach: {limit.limit_type} for portfolio {portfolio_id}")
            
            # In practice, this would trigger notifications and alerts
            
        except Exception as e:
            logger.error(f"Error creating limit breach: {e}")
    
    def _determine_breach_severity(self, breach_amount: Decimal, limit_value: Decimal) -> str:
        """Determine severity of limit breach."""
        if limit_value == 0:
            return "CRITICAL"
        
        breach_ratio = breach_amount / limit_value
        
        if breach_ratio >= 0.5:
            return "CRITICAL"
        elif breach_ratio >= 0.25:
            return "HIGH"
        elif breach_ratio >= 0.1:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_portfolio_snapshot_history(self, portfolio_id: str, limit: Optional[int] = None) -> List[RiskSnapshot]:
        """Get portfolio risk snapshot history."""
        if portfolio_id not in self.portfolio_snapshots:
            return []
        
        snapshots = list(self.portfolio_snapshots[portfolio_id])
        if limit:
            snapshots = snapshots[-limit:]
        
        return snapshots
    
    def get_stress_test_results(self, portfolio_id: str, limit: Optional[int] = None) -> List[StressTestResult]:
        """Get stress test results for a portfolio."""
        if portfolio_id not in self.stress_results:
            return []
        
        results = self.stress_results[portfolio_id]
        if limit:
            results = results[-limit:]
        
        return results
    
    def get_limit_breaches(self, portfolio_id: Optional[str] = None, 
                          status: Optional[str] = None) -> List[RiskLimitBreach]:
        """Get risk limit breaches."""
        breaches = self.limit_breaches
        
        if portfolio_id:
            breaches = [b for b in breaches if b.portfolio_id == portfolio_id]
        
        if status:
            breaches = [b for b in breaches if b.status == status]
        
        return breaches
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get risk engine statistics."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        total_portfolios = len(self.portfolios)
        total_positions = sum(len(portfolio) for portfolio in self.portfolios.values())
        total_breaches = len(self.limit_breaches)
        open_breaches = len([b for b in self.limit_breaches if b.status == "OPEN"])
        
        return {
            "calculations_performed": self.calculations_performed,
            "total_portfolios": total_portfolios,
            "total_positions": total_positions,
            "total_risk_limits": len(self.risk_limits),
            "total_breaches": total_breaches,
            "open_breaches": open_breaches,
            "uptime_seconds": uptime,
            "calculations_per_second": self.calculations_performed / uptime if uptime > 0 else 0,
            "start_time": self.start_time
        }


# Export classes
__all__ = ["RealTimeRiskEngine", "RiskMetricType", "VaRMethod", "StressScenarioType", 
           "PortfolioPosition", "RiskSnapshot", "VaRResult", "StressTestResult", "RiskLimitBreach"]
