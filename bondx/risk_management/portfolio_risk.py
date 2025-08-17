"""
Portfolio Risk Manager for BondX Risk Management System.

This module provides comprehensive portfolio risk analysis including:
- Value at Risk (VaR) calculations
- Stress testing and scenario analysis
- Risk decomposition and attribution
- Portfolio optimization recommendations
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from ..core.logging import get_logger
from ..core.monitoring import MetricsCollector
from .risk_models import (
    RiskMetrics, Portfolio, PortfolioPosition, RiskLevel,
    StressTestScenario, StressTestResult
)
from ..mathematics.bond_pricing import BondPricingEngine
from ..mathematics.duration_convexity import DurationConvexityCalculator

logger = get_logger(__name__)


class VaRMethod(Enum):
    """VaR calculation methods."""
    
    HISTORICAL_SIMULATION = "HISTORICAL_SIMULATION"
    PARAMETRIC = "PARAMETRIC"
    MONTE_CARLO = "MONTE_CARLO"
    BOOTSTRAP = "BOOTSTRAP"


class StressTestType(Enum):
    """Types of stress tests."""
    
    INTEREST_RATE = "INTEREST_RATE"
    CREDIT_SPREAD = "CREDIT_SPREAD"
    LIQUIDITY = "LIQUIDITY"
    MARKET_CRASH = "MARKET_CRASH"
    CORRELATION_BREAKDOWN = "CORRELATION_BREAKDOWN"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"


@dataclass
class RiskDecomposition:
    """Risk decomposition analysis."""
    
    portfolio_id: str
    calculation_time: datetime = field(default_factory=datetime.utcnow)
    
    # Position-level contributions
    position_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Factor contributions
    factor_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Sector contributions
    sector_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Duration contributions
    duration_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Credit quality contributions
    credit_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Total risk
    total_risk: float = 0.0
    systematic_risk: float = 0.0
    idiosyncratic_risk: float = 0.0


class PortfolioRiskManager:
    """
    Comprehensive portfolio risk management system.
    
    This manager provides:
    - VaR calculations using multiple methods
    - Stress testing and scenario analysis
    - Risk decomposition and attribution
    - Portfolio optimization recommendations
    """
    
    def __init__(self, db_session: Session):
        """Initialize the portfolio risk manager."""
        self.db_session = db_session
        self.logger = get_logger(__name__)
        self.metrics = MetricsCollector()
        
        # Risk calculation engines
        self.bond_pricing = BondPricingEngine()
        self.duration_calculator = DurationConvexityCalculator()
        
        # Risk parameters
        self.default_confidence_level = 0.95
        self.default_time_horizon = 1  # days
        self.historical_data_days = 252  # 1 year of trading days
        
        # Stress test scenarios
        self.default_scenarios = self._initialize_default_scenarios()
        
        # Performance tracking
        self.calculation_times: Dict[str, float] = {}
        self.last_calculations: Dict[str, datetime] = {}
        
        logger.info("Portfolio Risk Manager initialized successfully")
    
    async def calculate_portfolio_risk(self, portfolio_id: str, 
                                     method: VaRMethod = VaRMethod.HISTORICAL_SIMULATION,
                                     confidence_level: float = None,
                                     time_horizon: int = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            method: VaR calculation method
            confidence_level: Confidence level for VaR (0.95, 0.99, etc.)
            time_horizon: Time horizon in days
            
        Returns:
            RiskMetrics object with comprehensive risk analysis
        """
        try:
            start_time = datetime.utcnow()
            
            # Set default parameters
            confidence_level = confidence_level or self.default_confidence_level
            time_horizon = time_horizon or self.default_time_horizon
            
            # Get portfolio and positions
            portfolio = await self._get_portfolio(portfolio_id)
            if not portfolio:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            positions = await self._get_portfolio_positions(portfolio_id)
            if not positions:
                # Empty portfolio
                return self._create_empty_risk_metrics(portfolio_id, confidence_level, time_horizon)
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(
                portfolio, positions, method, confidence_level, time_horizon
            )
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.calculation_times[portfolio_id] = execution_time
            self.last_calculations[portfolio_id] = datetime.utcnow()
            
            # Update metrics
            self.metrics.increment_counter("portfolio_risk_calculations_total", {"method": method.value})
            self.metrics.observe_histogram("portfolio_risk_calculation_time_seconds", execution_time)
            
            logger.info(f"Portfolio risk calculated for {portfolio_id} in {execution_time:.2f}s")
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk for {portfolio_id}: {str(e)}")
            self.metrics.increment_counter("portfolio_risk_calculation_errors_total", {"error": str(e)})
            raise
    
    async def perform_stress_test(self, portfolio_id: str, 
                                scenario: StressTestScenario = None) -> StressTestResult:
        """
        Perform stress test on a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            scenario: Stress test scenario (uses default if None)
            
        Returns:
            StressTestResult with stress test analysis
        """
        try:
            start_time = datetime.utcnow()
            
            # Get or create scenario
            if scenario is None:
                scenario = self.default_scenarios[StressTestType.INTEREST_RATE]
            
            # Get portfolio and positions
            portfolio = await self._get_portfolio(portfolio_id)
            if not portfolio:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            positions = await self._get_portfolio_positions(portfolio_id)
            if not positions:
                return self._create_empty_stress_test_result(portfolio_id, scenario)
            
            # Calculate baseline metrics
            baseline_metrics = await self.calculate_portfolio_risk(portfolio_id)
            
            # Apply stress scenario
            stressed_metrics = await self._apply_stress_scenario(
                portfolio, positions, scenario, baseline_metrics
            )
            
            # Calculate impact
            result = await self._calculate_stress_test_result(
                portfolio_id, scenario, baseline_metrics, stressed_metrics
            )
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result.execution_duration_seconds = execution_time
            
            # Update metrics
            self.metrics.increment_counter("stress_tests_performed_total", {"scenario_type": scenario.scenario_type})
            self.metrics.observe_histogram("stress_test_execution_time_seconds", execution_time)
            
            logger.info(f"Stress test completed for {portfolio_id} in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error performing stress test for {portfolio_id}: {str(e)}")
            self.metrics.increment_counter("stress_test_errors_total", {"error": str(e)})
            raise
    
    async def calculate_risk_decomposition(self, portfolio_id: str) -> RiskDecomposition:
        """
        Calculate risk decomposition for a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            RiskDecomposition object with risk attribution
        """
        try:
            # Get portfolio and positions
            portfolio = await self._get_portfolio(portfolio_id)
            if not portfolio:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            positions = await self._get_portfolio_positions(portfolio_id)
            if not positions:
                return self._create_empty_risk_decomposition(portfolio_id)
            
            # Calculate risk decomposition
            decomposition = await self._decompose_risk(portfolio, positions)
            
            # Update metrics
            self.metrics.increment_counter("risk_decompositions_calculated_total")
            
            return decomposition
            
        except Exception as e:
            self.logger.error(f"Error calculating risk decomposition for {portfolio_id}: {str(e)}")
            raise
    
    async def get_portfolio_risk_summary(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Get comprehensive risk summary for a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Dictionary with risk summary
        """
        try:
            # Get latest risk metrics
            risk_metrics = await self.calculate_portfolio_risk(portfolio_id)
            
            # Get risk decomposition
            decomposition = await self.calculate_risk_decomposition(portfolio_id)
            
            # Get recent stress test results
            stress_results = await self._get_recent_stress_test_results(portfolio_id)
            
            # Compile summary
            summary = {
                "portfolio_id": portfolio_id,
                "calculation_time": risk_metrics.calculation_time.isoformat(),
                "risk_metrics": {
                    "var_95_1d": risk_metrics.var_95_1d,
                    "var_99_1d": risk_metrics.var_99_1d,
                    "portfolio_volatility": risk_metrics.portfolio_volatility,
                    "modified_duration": risk_metrics.modified_duration,
                    "convexity": risk_metrics.convexity,
                    "liquidity_score": risk_metrics.liquidity_score,
                    "concentration_risk": risk_metrics.concentration_risk
                },
                "risk_decomposition": {
                    "total_risk": decomposition.total_risk,
                    "systematic_risk": decomposition.systematic_risk,
                    "idiosyncratic_risk": decomposition.idiosyncratic_risk,
                    "top_position_contributors": dict(list(decomposition.position_contributions.items())[:5]),
                    "top_sector_contributors": dict(list(decomposition.sector_contributions.items())[:5])
                },
                "stress_test_summary": {
                    "total_tests": len(stress_results),
                    "passed_tests": len([r for r in stress_results if r.is_passed]),
                    "failed_tests": len([r for r in stress_results if not r.is_passed]),
                    "worst_case_impact": min([r.portfolio_value_change_percent for r in stress_results]) if stress_results else None
                },
                "performance_metrics": {
                    "last_calculation_time": self.last_calculations.get(portfolio_id),
                    "average_calculation_time": self.calculation_times.get(portfolio_id, 0)
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting risk summary for {portfolio_id}: {str(e)}")
            raise
    
    async def _calculate_risk_metrics(self, portfolio: Portfolio, positions: List[PortfolioPosition],
                                    method: VaRMethod, confidence_level: float, 
                                    time_horizon: int) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            portfolio: Portfolio object
            positions: List of portfolio positions
            method: VaR calculation method
            confidence_level: Confidence level
            time_horizon: Time horizon in days
            
        Returns:
            RiskMetrics object
        """
        # Initialize risk metrics
        risk_metrics = RiskMetrics(
            portfolio_id=portfolio.portfolio_id,
            confidence_level=confidence_level,
            time_horizon_days=time_horizon,
            calculation_method=method.value
        )
        
        # Calculate VaR
        if method == VaRMethod.HISTORICAL_SIMULATION:
            await self._calculate_historical_var(risk_metrics, positions, confidence_level, time_horizon)
        elif method == VaRMethod.PARAMETRIC:
            await self._calculate_parametric_var(risk_metrics, positions, confidence_level, time_horizon)
        elif method == VaRMethod.MONTE_CARLO:
            await self._calculate_monte_carlo_var(risk_metrics, positions, confidence_level, time_horizon)
        
        # Calculate fixed income specific metrics
        await self._calculate_fixed_income_metrics(risk_metrics, positions)
        
        # Calculate liquidity and concentration metrics
        await self._calculate_liquidity_metrics(risk_metrics, positions)
        
        # Calculate leverage and exposure metrics
        await self._calculate_exposure_metrics(risk_metrics, portfolio, positions)
        
        # Calculate historical performance metrics
        await self._calculate_historical_metrics(risk_metrics, portfolio, positions)
        
        return risk_metrics
    
    async def _calculate_historical_var(self, risk_metrics: RiskMetrics, 
                                      positions: List[PortfolioPosition],
                                      confidence_level: float, time_horizon: int):
        """Calculate VaR using historical simulation."""
        try:
            # Get historical price data for all positions
            historical_returns = await self._get_historical_returns(positions, self.historical_data_days)
            
            if not historical_returns:
                return
            
            # Calculate portfolio returns
            portfolio_returns = []
            for date in historical_returns[0].keys():
                daily_return = 0.0
                for position in positions:
                    if date in historical_returns[position.bond_id]:
                        daily_return += historical_returns[position.bond_id][date] * float(position.market_value)
                portfolio_returns.append(daily_return)
            
            # Calculate VaR
            portfolio_returns = np.array(portfolio_returns)
            var_percentile = (1 - confidence_level) * 100
            
            risk_metrics.var_95_1d = np.percentile(portfolio_returns, 5) if confidence_level == 0.95 else None
            risk_metrics.var_99_1d = np.percentile(portfolio_returns, 1) if confidence_level == 0.99 else None
            
            # Calculate CVaR
            if risk_metrics.var_95_1d is not None:
                risk_metrics.cvar_95_1d = portfolio_returns[portfolio_returns <= risk_metrics.var_95_1d].mean()
            if risk_metrics.var_99_1d is not None:
                risk_metrics.cvar_99_1d = portfolio_returns[portfolio_returns <= risk_metrics.var_99_1d].mean()
            
            # Calculate volatility
            risk_metrics.portfolio_volatility = float(np.std(portfolio_returns))
            
            # Scale for time horizon
            if time_horizon > 1:
                risk_metrics.var_95_10d = risk_metrics.var_95_1d * np.sqrt(time_horizon) if risk_metrics.var_95_1d else None
                risk_metrics.var_99_10d = risk_metrics.var_99_1d * np.sqrt(time_horizon) if risk_metrics.var_99_1d else None
            
            risk_metrics.data_points = len(portfolio_returns)
            
        except Exception as e:
            self.logger.error(f"Error calculating historical VaR: {str(e)}")
    
    async def _calculate_parametric_var(self, risk_metrics: RiskMetrics, 
                                      positions: List[PortfolioPosition],
                                      confidence_level: float, time_horizon: int):
        """Calculate VaR using parametric method."""
        try:
            # Calculate portfolio volatility
            portfolio_volatility = await self._calculate_portfolio_volatility(positions)
            risk_metrics.portfolio_volatility = portfolio_volatility
            
            # Calculate VaR using normal distribution
            z_score = self._get_z_score(confidence_level)
            
            risk_metrics.var_95_1d = -z_score * portfolio_volatility if confidence_level == 0.95 else None
            risk_metrics.var_99_1d = -z_score * portfolio_volatility if confidence_level == 0.99 else None
            
            # Scale for time horizon
            if time_horizon > 1:
                risk_metrics.var_95_10d = risk_metrics.var_95_1d * np.sqrt(time_horizon) if risk_metrics.var_95_1d else None
                risk_metrics.var_99_10d = risk_metrics.var_99_1d * np.sqrt(time_horizon) if risk_metrics.var_99_1d else None
            
        except Exception as e:
            self.logger.error(f"Error calculating parametric VaR: {str(e)}")
    
    async def _calculate_monte_carlo_var(self, risk_metrics: RiskMetrics, 
                                       positions: List[PortfolioPosition],
                                       confidence_level: float, time_horizon: int):
        """Calculate VaR using Monte Carlo simulation."""
        try:
            # Get correlation matrix and volatilities
            correlation_matrix = await self._get_correlation_matrix(positions)
            volatilities = await self._get_position_volatilities(positions)
            
            # Generate random scenarios
            num_scenarios = 10000
            random_returns = np.random.multivariate_normal(
                mean=np.zeros(len(positions)),
                cov=correlation_matrix * np.outer(volatilities, volatilities),
                size=num_scenarios
            )
            
            # Calculate portfolio returns
            position_values = [float(pos.market_value) for pos in positions]
            portfolio_returns = np.dot(random_returns, position_values)
            
            # Calculate VaR
            var_percentile = (1 - confidence_level) * 100
            
            risk_metrics.var_95_1d = np.percentile(portfolio_returns, 5) if confidence_level == 0.95 else None
            risk_metrics.var_99_1d = np.percentile(portfolio_returns, 1) if confidence_level == 0.99 else None
            
            # Calculate volatility
            risk_metrics.portfolio_volatility = float(np.std(portfolio_returns))
            
            risk_metrics.data_points = num_scenarios
            
        except Exception as e:
            self.logger.error(f"Error calculating Monte Carlo VaR: {str(e)}")
    
    async def _calculate_fixed_income_metrics(self, risk_metrics: RiskMetrics, 
                                            positions: List[PortfolioPosition]):
        """Calculate fixed income specific risk metrics."""
        try:
            total_value = sum(float(pos.market_value) for pos in positions)
            if total_value == 0:
                return
            
            # Calculate weighted duration and convexity
            weighted_duration = 0.0
            weighted_convexity = 0.0
            weighted_yield = 0.0
            
            for position in positions:
                weight = float(position.market_value) / total_value
                
                if position.modified_duration:
                    weighted_duration += position.modified_duration * weight
                if position.convexity:
                    weighted_convexity += position.convexity * weight
                if position.yield_to_maturity:
                    weighted_yield += position.yield_to_maturity * weight
            
            risk_metrics.modified_duration = weighted_duration
            risk_metrics.convexity = weighted_convexity
            risk_metrics.yield_to_maturity = weighted_yield
            
        except Exception as e:
            self.logger.error(f"Error calculating fixed income metrics: {str(e)}")
    
    async def _calculate_liquidity_metrics(self, risk_metrics: RiskMetrics, 
                                         positions: List[PortfolioPosition]):
        """Calculate liquidity and concentration risk metrics."""
        try:
            total_value = sum(float(pos.market_value) for pos in positions)
            if total_value == 0:
                return
            
            # Calculate concentration metrics
            position_weights = [float(pos.market_value) / total_value for pos in positions]
            herfindahl_index = sum(w * w for w in position_weights)
            
            risk_metrics.concentration_risk = herfindahl_index
            
            # Calculate sector concentration
            sector_values = {}
            for position in positions:
                sector = position.sector or "UNKNOWN"
                sector_values[sector] = sector_values.get(sector, 0) + float(position.market_value)
            
            sector_concentration = max(sector_values.values()) / total_value if sector_values else 0
            risk_metrics.sector_concentration = sector_concentration
            
            # Calculate issuer concentration
            issuer_values = {}
            for position in positions:
                issuer = position.issuer or "UNKNOWN"
                issuer_values[issuer] = issuer_values.get(issuer, 0) + float(position.market_value)
            
            issuer_concentration = max(issuer_values.values()) / total_value if issuer_values else 0
            risk_metrics.issuer_concentration = issuer_concentration
            
            # Simple liquidity score (can be enhanced with actual liquidity data)
            risk_metrics.liquidity_score = 1.0 - (herfindahl_index * 0.5)
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity metrics: {str(e)}")
    
    async def _calculate_exposure_metrics(self, risk_metrics: RiskMetrics, 
                                        portfolio: Portfolio, positions: List[PortfolioPosition]):
        """Calculate leverage and exposure metrics."""
        try:
            total_value = float(portfolio.total_value)
            if total_value == 0:
                return
            
            # Calculate gross and net exposure
            gross_exposure = sum(abs(float(pos.market_value)) for pos in positions)
            net_exposure = sum(float(pos.market_value) for pos in positions)
            
            risk_metrics.gross_exposure = gross_exposure
            risk_metrics.net_exposure = net_exposure
            
            # Calculate leverage ratio
            risk_metrics.leverage_ratio = gross_exposure / total_value if total_value > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating exposure metrics: {str(e)}")
    
    async def _calculate_historical_metrics(self, risk_metrics: RiskMetrics, 
                                          portfolio: Portfolio, positions: List[PortfolioPosition]):
        """Calculate historical performance metrics."""
        try:
            # This would typically use historical portfolio performance data
            # For now, we'll set placeholder values
            risk_metrics.max_drawdown = None
            risk_metrics.sharpe_ratio = None
            risk_metrics.sortino_ratio = None
            risk_metrics.calmar_ratio = None
            
        except Exception as e:
            self.logger.error(f"Error calculating historical metrics: {str(e)}")
    
    async def _apply_stress_scenario(self, portfolio: Portfolio, positions: List[PortfolioPosition],
                                   scenario: StressTestScenario, baseline_metrics: RiskMetrics) -> RiskMetrics:
        """Apply stress scenario to portfolio."""
        try:
            # Create stressed risk metrics
            stressed_metrics = RiskMetrics(
                portfolio_id=portfolio.portfolio_id,
                calculation_method=f"STRESS_TEST_{scenario.scenario_type}"
            )
            
            # Apply scenario-specific stress
            if scenario.scenario_type == "INTEREST_RATE":
                await self._apply_interest_rate_stress(stressed_metrics, positions, scenario)
            elif scenario.scenario_type == "CREDIT_SPREAD":
                await self._apply_credit_spread_stress(stressed_metrics, positions, scenario)
            elif scenario.scenario_type == "MARKET_CRASH":
                await self._apply_market_crash_stress(stressed_metrics, positions, scenario)
            
            return stressed_metrics
            
        except Exception as e:
            self.logger.error(f"Error applying stress scenario: {str(e)}")
            return baseline_metrics
    
    async def _apply_interest_rate_stress(self, stressed_metrics: RiskMetrics, 
                                        positions: List[PortfolioPosition], scenario: StressTestScenario):
        """Apply interest rate stress scenario."""
        try:
            # Get stress parameters
            rate_shock = scenario.parameters.get("rate_shock", 100)  # basis points
            
            # Calculate stressed portfolio value
            stressed_value = 0.0
            for position in positions:
                if position.modified_duration:
                    # Simple duration-based price change
                    price_change = -position.modified_duration * (rate_shock / 10000)
                    stressed_price = float(position.current_price or 100) * (1 + price_change)
                    stressed_value += float(position.quantity) * stressed_price
                else:
                    stressed_value += float(position.market_value)
            
            stressed_metrics.portfolio_volatility = scenario.parameters.get("volatility_multiplier", 2.0)
            
        except Exception as e:
            self.logger.error(f"Error applying interest rate stress: {str(e)}")
    
    async def _apply_credit_spread_stress(self, stressed_metrics: RiskMetrics, 
                                        positions: List[PortfolioPosition], scenario: StressTestScenario):
        """Apply credit spread stress scenario."""
        try:
            # Get stress parameters
            spread_widening = scenario.parameters.get("spread_widening", 50)  # basis points
            
            # Calculate stressed portfolio value
            stressed_value = 0.0
            for position in positions:
                if position.credit_spread_duration:
                    # Credit spread duration-based price change
                    price_change = -position.credit_spread_duration * (spread_widening / 10000)
                    stressed_price = float(position.current_price or 100) * (1 + price_change)
                    stressed_value += float(position.quantity) * stressed_price
                else:
                    stressed_value += float(position.market_value)
            
        except Exception as e:
            self.logger.error(f"Error applying credit spread stress: {str(e)}")
    
    async def _apply_market_crash_stress(self, stressed_metrics: RiskMetrics, 
                                       positions: List[PortfolioPosition], scenario: StressTestScenario):
        """Apply market crash stress scenario."""
        try:
            # Get stress parameters
            market_decline = scenario.parameters.get("market_decline", 0.20)  # 20% decline
            
            # Calculate stressed portfolio value
            stressed_value = 0.0
            for position in positions:
                # Apply market decline with some variation
                decline_factor = market_decline * (0.8 + 0.4 * np.random.random())
                stressed_price = float(position.current_price or 100) * (1 - decline_factor)
                stressed_value += float(position.quantity) * stressed_price
            
            stressed_metrics.portfolio_volatility = scenario.parameters.get("volatility_multiplier", 3.0)
            
        except Exception as e:
            self.logger.error(f"Error applying market crash stress: {str(e)}")
    
    async def _calculate_stress_test_result(self, portfolio_id: str, scenario: StressTestScenario,
                                          baseline_metrics: RiskMetrics, 
                                          stressed_metrics: RiskMetrics) -> StressTestResult:
        """Calculate stress test result."""
        try:
            # Calculate portfolio value change
            baseline_value = baseline_metrics.portfolio_volatility or 1000000  # Default value
            stressed_value = stressed_metrics.portfolio_volatility or baseline_value
            
            value_change = stressed_value - baseline_value
            value_change_percent = (value_change / baseline_value) * 100 if baseline_value > 0 else 0
            
            # Determine if test passed
            is_passed = abs(value_change_percent) <= scenario.parameters.get("max_acceptable_loss", 20)
            
            # Create result
            result = StressTestResult(
                scenario_id=scenario.scenario_id,
                portfolio_id=portfolio_id,
                portfolio_value_before=baseline_value,
                portfolio_value_after=stressed_value,
                portfolio_value_change=value_change,
                portfolio_value_change_percent=value_change_percent,
                is_passed=is_passed,
                failure_reason=None if is_passed else f"Loss exceeded {scenario.parameters.get('max_acceptable_loss', 20)}%"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating stress test result: {str(e)}")
            raise
    
    async def _decompose_risk(self, portfolio: Portfolio, positions: List[PortfolioPosition]) -> RiskDecomposition:
        """Decompose portfolio risk into components."""
        try:
            decomposition = RiskDecomposition(portfolio_id=portfolio.portfolio_id)
            
            total_value = sum(float(pos.market_value) for pos in positions)
            if total_value == 0:
                return decomposition
            
            # Position-level contributions
            for position in positions:
                weight = float(position.market_value) / total_value
                contribution = weight * weight  # Simplified contribution calculation
                decomposition.position_contributions[position.bond_id] = contribution
            
            # Sector contributions
            sector_values = {}
            for position in positions:
                sector = position.sector or "UNKNOWN"
                sector_values[sector] = sector_values.get(sector, 0) + float(position.market_value)
            
            for sector, value in sector_values.items():
                weight = value / total_value
                decomposition.sector_contributions[sector] = weight * weight
            
            # Duration contributions
            duration_buckets = {"SHORT": 0, "MEDIUM": 0, "LONG": 0}
            for position in positions:
                if position.modified_duration:
                    if position.modified_duration <= 3:
                        duration_buckets["SHORT"] += float(position.market_value)
                    elif position.modified_duration <= 7:
                        duration_buckets["MEDIUM"] += float(position.market_value)
                    else:
                        duration_buckets["LONG"] += float(position.market_value)
            
            for bucket, value in duration_buckets.items():
                if value > 0:
                    weight = value / total_value
                    decomposition.duration_contributions[bucket] = weight * weight
            
            # Calculate total risk
            decomposition.total_risk = sum(decomposition.position_contributions.values())
            decomposition.systematic_risk = decomposition.total_risk * 0.7  # Simplified
            decomposition.idiosyncratic_risk = decomposition.total_risk * 0.3  # Simplified
            
            return decomposition
            
        except Exception as e:
            self.logger.error(f"Error decomposing risk: {str(e)}")
            raise
    
    def _initialize_default_scenarios(self) -> Dict[StressTestType, StressTestScenario]:
        """Initialize default stress test scenarios."""
        scenarios = {}
        
        # Interest rate stress
        scenarios[StressTestType.INTEREST_RATE] = StressTestScenario(
            scenario_name="Interest Rate Shock",
            scenario_type="INTEREST_RATE",
            description="Parallel shift in yield curve",
            parameters={
                "rate_shock": 100,  # 100 basis points
                "volatility_multiplier": 2.0,
                "max_acceptable_loss": 15
            },
            stress_level="MODERATE"
        )
        
        # Credit spread stress
        scenarios[StressTestType.CREDIT_SPREAD] = StressTestScenario(
            scenario_name="Credit Spread Widening",
            scenario_type="CREDIT_SPREAD",
            description="Widening of credit spreads",
            parameters={
                "spread_widening": 50,  # 50 basis points
                "max_acceptable_loss": 10
            },
            stress_level="MODERATE"
        )
        
        # Market crash stress
        scenarios[StressTestType.MARKET_CRASH] = StressTestScenario(
            scenario_name="Market Crash",
            scenario_type="MARKET_CRASH",
            description="Severe market decline scenario",
            parameters={
                "market_decline": 0.20,  # 20% decline
                "volatility_multiplier": 3.0,
                "max_acceptable_loss": 25
            },
            stress_level="SEVERE"
        )
        
        return scenarios
    
    def _get_z_score(self, confidence_level: float) -> float:
        """Get Z-score for given confidence level."""
        z_scores = {
            0.90: 1.282,
            0.95: 1.645,
            0.99: 2.326,
            0.995: 2.576,
            0.999: 3.291
        }
        return z_scores.get(confidence_level, 1.645)
    
    async def _get_portfolio(self, portfolio_id: str) -> Optional[Portfolio]:
        """Get portfolio from database."""
        # This would typically query the database
        # For now, return a mock portfolio
        return Portfolio(
            portfolio_id=portfolio_id,
            portfolio_name=f"Portfolio {portfolio_id}",
            participant_id=1,
            total_value=Decimal('1000000')
        )
    
    async def _get_portfolio_positions(self, portfolio_id: str) -> List[PortfolioPosition]:
        """Get portfolio positions from database."""
        # This would typically query the database
        # For now, return mock positions
        return [
            PortfolioPosition(
                portfolio_id=portfolio_id,
                bond_id="BOND001",
                participant_id=1,
                quantity=Decimal('1000'),
                market_value=Decimal('500000'),
                modified_duration=5.2,
                convexity=25.0,
                yield_to_maturity=6.5
            ),
            PortfolioPosition(
                portfolio_id=portfolio_id,
                bond_id="BOND002",
                participant_id=1,
                quantity=Decimal('2000'),
                market_value=Decimal('500000'),
                modified_duration=3.8,
                convexity=18.0,
                yield_to_maturity=5.8
            )
        ]
    
    async def _get_historical_returns(self, positions: List[PortfolioPosition], days: int) -> Dict[str, Dict[str, float]]:
        """Get historical returns for positions."""
        # This would typically query historical data
        # For now, return mock data
        historical_returns = {}
        for position in positions:
            returns = {}
            for i in range(days):
                date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
                returns[date] = np.random.normal(0, 0.02)  # 2% daily volatility
            historical_returns[position.bond_id] = returns
        return historical_returns
    
    async def _get_correlation_matrix(self, positions: List[PortfolioPosition]) -> np.ndarray:
        """Get correlation matrix for positions."""
        # This would typically use historical data
        # For now, return identity matrix
        n = len(positions)
        return np.eye(n)
    
    async def _get_position_volatilities(self, positions: List[PortfolioPosition]) -> List[float]:
        """Get volatilities for positions."""
        # This would typically use historical data
        # For now, return default volatilities
        return [0.02] * len(positions)  # 2% daily volatility
    
    async def _calculate_portfolio_volatility(self, positions: List[PortfolioPosition]) -> float:
        """Calculate portfolio volatility."""
        # This would typically use historical data and correlations
        # For now, return a simple estimate
        total_value = sum(float(pos.market_value) for pos in positions)
        if total_value == 0:
            return 0.0
        
        # Weighted average volatility
        weighted_vol = 0.0
        for position in positions:
            weight = float(position.market_value) / total_value
            weighted_vol += 0.02 * weight  # 2% default volatility
        
        return weighted_vol
    
    async def _get_recent_stress_test_results(self, portfolio_id: str) -> List[StressTestResult]:
        """Get recent stress test results."""
        # This would typically query the database
        # For now, return empty list
        return []
    
    def _create_empty_risk_metrics(self, portfolio_id: str, confidence_level: float, 
                                  time_horizon: int) -> RiskMetrics:
        """Create empty risk metrics for empty portfolio."""
        return RiskMetrics(
            portfolio_id=portfolio_id,
            confidence_level=confidence_level,
            time_horizon_days=time_horizon,
            calculation_method="EMPTY_PORTFOLIO"
        )
    
    def _create_empty_stress_test_result(self, portfolio_id: str, 
                                       scenario: StressTestScenario) -> StressTestResult:
        """Create empty stress test result for empty portfolio."""
        return StressTestResult(
            scenario_id=scenario.scenario_id,
            portfolio_id=portfolio_id,
            portfolio_value_before=0,
            portfolio_value_after=0,
            portfolio_value_change=0,
            portfolio_value_change_percent=0.0,
            is_passed=True
        )
    
    def _create_empty_risk_decomposition(self, portfolio_id: str) -> RiskDecomposition:
        """Create empty risk decomposition for empty portfolio."""
        return RiskDecomposition(portfolio_id=portfolio_id)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "total_calculations": len(self.last_calculations),
            "average_calculation_time": np.mean(list(self.calculation_times.values())) if self.calculation_times else 0,
            "last_calculation": max(self.last_calculations.values()).isoformat() if self.last_calculations else None,
            "default_scenarios_count": len(self.default_scenarios),
            "last_updated": datetime.utcnow().isoformat()
        }
