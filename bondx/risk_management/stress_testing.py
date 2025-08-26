"""
Stress testing engine for BondX Backend.

This module implements configurable stress testing for portfolio-level and
instrument-level scenario analysis including predefined and custom scenarios.
"""

import json
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..core.logging import get_logger
from ..database.models import DayCountConvention
from ..mathematics.yield_curves import YieldCurve, CurveType
from ..mathematics.cash_flows import CashFlow
from ..mathematics.option_adjusted_spread import OASCalculator, OASInputs

logger = get_logger(__name__)


class ScenarioType(Enum):
    """Types of stress test scenarios."""
    PARALLEL_SHIFT = "PARALLEL_SHIFT"
    CURVE_STEEPENING = "CURVE_STEEPENING"
    CURVE_FLATTENING = "CURVE_FLATTENING"
    CREDIT_SPREAD_BLOWOUT = "CREDIT_SPREAD_BLOWOUT"
    LIQUIDITY_CRUNCH = "LIQUIDITY_CRUNCH"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    CUSTOM = "CUSTOM"


class CalculationMode(Enum):
    """Calculation modes for stress testing."""
    FAST_APPROXIMATION = "FAST_APPROXIMATION"  # Duration/convexity + spread DV
    FULL_REPRICE = "FULL_REPRICE"  # Full revaluation


class RatingBucket(Enum):
    """Credit rating buckets for spread analysis."""
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    CCC = "CCC"
    DEFAULT = "DEFAULT"


class SectorBucket(Enum):
    """Sector buckets for portfolio analysis."""
    GOVERNMENT = "GOVERNMENT"
    CORPORATE = "CORPORATE"
    FINANCIAL = "FINANCIAL"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    REAL_ESTATE = "REAL_ESTATE"
    OTHER = "OTHER"


@dataclass
class Position:
    """Portfolio position for stress testing."""
    instrument_id: str
    face_value: Decimal
    book_value: Decimal
    market_value: Decimal
    coupon_rate: Decimal
    maturity_date: date
    duration: float
    convexity: float
    spread_dv01: float
    liquidity_score: float
    issuer_id: str
    sector: SectorBucket
    rating: RatingBucket
    tenor_bucket: str  # e.g., "0-1Y", "1-3Y", "3-5Y", "5-10Y", "10Y+"
    oas_sensitive: bool = False  # Whether instrument is OAS-sensitive


@dataclass
class StressScenario:
    """Definition of a stress test scenario."""
    scenario_id: str
    scenario_type: ScenarioType
    name: str
    description: str
    
    # Rate curve shocks
    parallel_shift_bps: Optional[int] = None  # Parallel shift in basis points
    curve_steepening_bps: Optional[int] = None  # Steepening in basis points
    curve_flattening_bps: Optional[int] = None  # Flattening in basis points
    
    # Credit spread shocks
    credit_spread_shocks: Optional[Dict[RatingBucket, int]] = None  # bps by rating
    
    # Liquidity shocks
    liquidity_spread_bps: Optional[int] = None  # Additional liquidity spread
    bid_ask_widening_bps: Optional[int] = None  # Bid-ask spread widening
    
    # Volatility shocks
    volatility_multiplier: Optional[float] = None  # Multiplier for vol surface
    
    # Custom shocks
    custom_shocks: Optional[Dict[str, float]] = None  # Custom factor shocks
    
    # Scenario metadata
    severity: str = "MODERATE"  # LOW, MODERATE, HIGH, EXTREME
    probability: float = 0.01  # Probability of scenario occurring
    created_date: date = field(default_factory=date.today)
    tags: List[str] = field(default_factory=list)


@dataclass
class StressTestResult:
    """Results from a stress test scenario."""
    scenario_id: str
    scenario_name: str
    calculation_mode: CalculationMode
    timestamp: datetime
    
    # Portfolio impact
    total_pnl: Decimal
    total_pnl_bps: float  # P&L in basis points of portfolio value
    
    # Risk metrics
    delta_dv01: float
    delta_key_rate_duration: Dict[str, float]
    delta_spread_dv: Dict[RatingBucket, float]
    
    # Drilldown results
    pnl_by_issuer: Dict[str, Decimal]
    pnl_by_sector: Dict[SectorBucket, Decimal]
    pnl_by_rating: Dict[RatingBucket, Decimal]
    pnl_by_tenor: Dict[str, Decimal]
    
    # Limit breaches
    limit_breaches: List[Dict[str, Any]]
    
    # Performance metrics
    calculation_time_ms: float
    positions_processed: int
    
    # Narrative
    narrative: str
    model_settings: Dict[str, Any]


class StressTestingEngine:
    """
    Configurable stress testing engine for portfolio analysis.
    
    Supports both fast approximation and full reprice modes with
    predefined and custom scenarios.
    """
    
    def __init__(
        self,
        calculation_mode: CalculationMode = CalculationMode.FAST_APPROXIMATION,
        parallel_processing: bool = True,
        cache_results: bool = True
    ):
        """
        Initialize stress testing engine.
        
        Args:
            calculation_mode: Default calculation mode
            parallel_processing: Enable parallel processing
            cache_results: Cache stress test results
        """
        self.calculation_mode = calculation_mode
        self.parallel_processing = parallel_processing
        self.cache_results = cache_results
        self.logger = logger
        
        # Initialize OAS calculator for option-sensitive instruments
        self.oas_calculator = OASCalculator()
        
        # Cache for stress test results
        self._result_cache = {}
    
    def run_stress_test(
        self,
        portfolio: List[Position],
        base_curves: Dict[str, YieldCurve],
        spread_surfaces: Dict[RatingBucket, Dict[str, float]],
        scenario: StressScenario,
        calculation_mode: Optional[CalculationMode] = None
    ) -> StressTestResult:
        """
        Run stress test on portfolio.
        
        Args:
            portfolio: List of portfolio positions
            base_curves: Base yield curves by currency/type
            spread_surfaces: Credit spread surfaces by rating and tenor
            scenario: Stress test scenario
            calculation_mode: Override default calculation mode
            
        Returns:
            Stress test results
        """
        start_time = pd.Timestamp.now()
        
        try:
            # Use provided mode or default
            mode = calculation_mode or self.calculation_mode
            
            # Check cache if enabled
            cache_key = self._generate_cache_key(portfolio, base_curves, spread_surfaces, scenario, mode)
            if self.cache_results and cache_key in self._result_cache:
                cached_result = self._result_cache[cache_key]
                cached_result.timestamp = datetime.now()  # Update timestamp
                return cached_result
            
            # Apply scenario shocks
            shocked_curves = self._apply_scenario_shocks(base_curves, scenario)
            shocked_spreads = self._apply_spread_shocks(spread_surfaces, scenario)
            
            # Calculate portfolio impact
            if mode == CalculationMode.FAST_APPROXIMATION:
                result = self._calculate_fast_approximation(
                    portfolio, shocked_curves, shocked_spreads, scenario
                )
            else:
                result = self._calculate_full_reprice(
                    portfolio, shocked_curves, shocked_spreads, scenario
                )
            
            # Calculate drilldown results
            drilldown_results = self._calculate_drilldown_results(portfolio, result)
            
            # Check limit breaches
            limit_breaches = self._check_limit_breaches(portfolio, result)
            
            # Calculate performance metrics
            calculation_time_ms = (pd.Timestamp.now() - start_time).total_seconds() * 1000
            
            # Create result object
            stress_result = StressTestResult(
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.name,
                calculation_mode=mode,
                timestamp=datetime.now(),
                total_pnl=result['total_pnl'],
                total_pnl_bps=result['total_pnl_bps'],
                delta_dv01=result['delta_dv01'],
                delta_key_rate_duration=result['delta_key_rate_duration'],
                delta_spread_dv=result['delta_spread_dv'],
                pnl_by_issuer=drilldown_results['by_issuer'],
                pnl_by_sector=drilldown_results['by_sector'],
                pnl_by_rating=drilldown_results['by_rating'],
                pnl_by_tenor=drilldown_results['by_tenor'],
                limit_breaches=limit_breaches,
                calculation_time_ms=calculation_time_ms,
                positions_processed=len(portfolio),
                narrative=self._generate_narrative(scenario, result),
                model_settings={
                    'calculation_mode': mode.value,
                    'parallel_processing': self.parallel_processing,
                    'scenario_severity': scenario.severity
                }
            )
            
            # Cache result if enabled
            if self.cache_results:
                self._result_cache[cache_key] = stress_result
            
            return stress_result
            
        except Exception as e:
            self.logger.error(f"Error running stress test: {str(e)}")
            raise
    
    def run_multiple_scenarios(
        self,
        portfolio: List[Position],
        base_curves: Dict[str, YieldCurve],
        spread_surfaces: Dict[RatingBucket, Dict[str, float]],
        scenarios: List[StressScenario],
        calculation_mode: Optional[CalculationMode] = None
    ) -> List[StressTestResult]:
        """
        Run multiple stress test scenarios.
        
        Args:
            portfolio: List of portfolio positions
            base_curves: Base yield curves by currency/type
            spread_surfaces: Credit spread surfaces by rating and tenor
            scenarios: List of stress test scenarios
            calculation_mode: Override default calculation mode
            
        Returns:
            List of stress test results
        """
        results = []
        
        for scenario in scenarios:
            try:
                result = self.run_stress_test(
                    portfolio, base_curves, spread_surfaces, scenario, calculation_mode
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error running scenario {scenario.scenario_id}: {str(e)}")
                # Continue with other scenarios
                continue
        
        return results
    
    def get_predefined_scenarios(self) -> List[StressScenario]:
        """Get list of predefined stress test scenarios."""
        scenarios = []
        
        # Parallel rate shifts
        scenarios.append(StressScenario(
            scenario_id="PARALLEL_UP_50",
            scenario_type=ScenarioType.PARALLEL_SHIFT,
            name="Parallel Rate Shift +50bps",
            description="Parallel upward shift of yield curve by 50 basis points",
            parallel_shift_bps=50,
            severity="LOW"
        ))
        
        scenarios.append(StressScenario(
            scenario_id="PARALLEL_UP_100",
            scenario_type=ScenarioType.PARALLEL_SHIFT,
            name="Parallel Rate Shift +100bps",
            description="Parallel upward shift of yield curve by 100 basis points",
            parallel_shift_bps=100,
            severity="MODERATE"
        ))
        
        scenarios.append(StressScenario(
            scenario_id="PARALLEL_UP_200",
            scenario_type=ScenarioType.PARALLEL_SHIFT,
            name="Parallel Rate Shift +200bps",
            description="Parallel upward shift of yield curve by 200 basis points",
            parallel_shift_bps=200,
            severity="HIGH"
        ))
        
        # Curve steepening/flattening
        scenarios.append(StressScenario(
            scenario_id="CURVE_STEEPENING_50",
            scenario_type=ScenarioType.CURVE_STEEPENING,
            name="Curve Steepening +50bps",
            description="Steepening of yield curve by 50 basis points",
            curve_steepening_bps=50,
            severity="MODERATE"
        ))
        
        scenarios.append(StressScenario(
            scenario_id="CURVE_FLATTENING_50",
            scenario_type=ScenarioType.CURVE_FLATTENING,
            name="Curve Flattening +50bps",
            description="Flattening of yield curve by 50 basis points",
            curve_flattening_bps=50,
            severity="MODERATE"
        ))
        
        # Credit spread blowout
        scenarios.append(StressScenario(
            scenario_id="CREDIT_BLOWOUT_100",
            scenario_type=ScenarioType.CREDIT_SPREAD_BLOWOUT,
            name="Credit Spread Blowout +100bps",
            description="Widening of credit spreads by 100 basis points across ratings",
            credit_spread_shocks={
                RatingBucket.AAA: 50,
                RatingBucket.AA: 75,
                RatingBucket.A: 100,
                RatingBucket.BBB: 150,
                RatingBucket.BB: 200,
                RatingBucket.B: 300,
                RatingBucket.CCC: 400
            },
            severity="HIGH"
        ))
        
        # Liquidity crunch
        scenarios.append(StressScenario(
            scenario_id="LIQUIDITY_CRUNCH_50",
            scenario_type=ScenarioType.LIQUIDITY_CRUNCH,
            name="Liquidity Crunch +50bps",
            description="Additional liquidity spread of 50 basis points",
            liquidity_spread_bps=50,
            bid_ask_widening_bps=25,
            severity="HIGH"
        ))
        
        # Volatility spike
        scenarios.append(StressScenario(
            scenario_id="VOLATILITY_SPIKE_2X",
            scenario_type=ScenarioType.VOLATILITY_SPIKE,
            name="Volatility Spike 2x",
            description="Doubling of volatility surface",
            volatility_multiplier=2.0,
            severity="MODERATE"
        ))
        
        return scenarios
    
    def _apply_scenario_shocks(
        self,
        base_curves: Dict[str, YieldCurve],
        scenario: StressScenario
    ) -> Dict[str, YieldCurve]:
        """Apply scenario shocks to base curves."""
        shocked_curves = {}
        
        for curve_id, base_curve in base_curves.items():
            # Create copy of base curve
            shocked_curve = YieldCurve(
                curve_type=base_curve.curve_type,
                tenors=base_curve.tenors.copy(),
                rates=base_curve.rates.copy(),
                construction_date=base_curve.construction_date,
                config=base_curve.config,
                metadata=base_curve.metadata.copy() if base_curve.metadata else {}
            )
            
            # Apply parallel shift
            if scenario.parallel_shift_bps is not None:
                shift_bps = scenario.parallel_shift_bps / 10000.0
                shocked_curve.rates += shift_bps
            
            # Apply curve steepening/flattening
            if scenario.curve_steepening_bps is not None:
                steepening_bps = scenario.curve_steepening_bps / 10000.0
                # Steepening: increase long-term rates more than short-term
                steepening_factor = shocked_curve.tenors / np.max(shocked_curve.tenors)
                shocked_curve.rates += steepening_bps * steepening_factor
            
            if scenario.curve_flattening_bps is not None:
                flattening_bps = scenario.curve_flattening_bps / 10000.0
                # Flattening: decrease long-term rates more than short-term
                flattening_factor = 1.0 - (shocked_curve.tenors / np.max(shocked_curve.tenors))
                shocked_curve.rates -= flattening_bps * flattening_factor
            
            shocked_curves[curve_id] = shocked_curve
        
        return shocked_curves
    
    def _apply_spread_shocks(
        self,
        spread_surfaces: Dict[RatingBucket, Dict[str, float]],
        scenario: StressScenario
    ) -> Dict[RatingBucket, Dict[str, float]]:
        """Apply scenario shocks to spread surfaces."""
        shocked_spreads = {}
        
        for rating, tenor_spreads in spread_surfaces.items():
            shocked_spreads[rating] = {}
            
            for tenor, base_spread in tenor_spreads.items():
                shocked_spread = base_spread
                
                # Apply credit spread shocks
                if scenario.credit_spread_shocks and rating in scenario.credit_spread_shocks:
                    shock_bps = scenario.credit_spread_shocks[rating] / 10000.0
                    shocked_spread += shock_bps
                
                # Apply liquidity shocks
                if scenario.liquidity_spread_bps is not None:
                    liquidity_bps = scenario.liquidity_spread_bps / 10000.0
                    shocked_spread += liquidity_bps
                
                shocked_spreads[rating][tenor] = shocked_spread
        
        return shocked_spreads
    
    def _calculate_fast_approximation(
        self,
        portfolio: List[Position],
        shocked_curves: Dict[str, YieldCurve],
        shocked_spreads: Dict[RatingBucket, Dict[str, float]],
        scenario: StressScenario
    ) -> Dict[str, Any]:
        """Calculate stress test using fast approximation method."""
        total_pnl = Decimal('0')
        total_pnl_bps = 0.0
        delta_dv01 = 0.0
        delta_key_rate_duration = {}
        delta_spread_dv = {rating: 0.0 for rating in RatingBucket}
        
        # Get base curve for calculations (simplified - would use proper mapping)
        base_curve = list(shocked_curves.values())[0]
        
        for position in portfolio:
            # Calculate P&L using duration/convexity approximation
            position_pnl = self._calculate_position_pnl_fast(
                position, base_curve, shocked_spreads, scenario
            )
            
            total_pnl += position_pnl
            
            # Aggregate risk metrics
            delta_dv01 += position.duration * float(position.face_value)
            
            # Key rate duration (simplified)
            tenor_bucket = position.tenor_bucket
            if tenor_bucket not in delta_key_rate_duration:
                delta_key_rate_duration[tenor_bucket] = 0.0
            delta_key_rate_duration[tenor_bucket] += position.duration * float(position.face_value)
            
            # Spread DV
            delta_spread_dv[position.rating] += position.spread_dv01 * float(position.face_value)
        
        # Calculate total P&L in basis points
        total_portfolio_value = sum(float(pos.market_value) for pos in portfolio)
        if total_portfolio_value > 0:
            total_pnl_bps = float(total_pnl) / total_portfolio_value * 10000
        
        return {
            'total_pnl': total_pnl,
            'total_pnl_bps': total_pnl_bps,
            'delta_dv01': delta_dv01,
            'delta_key_rate_duration': delta_key_rate_duration,
            'delta_spread_dv': delta_spread_dv
        }
    
    def _calculate_position_pnl_fast(
        self,
        position: Position,
        shocked_curve: YieldCurve,
        shocked_spreads: Dict[RatingBucket, Dict[str, float]],
        scenario: StressScenario
    ) -> Decimal:
        """Calculate position P&L using fast approximation."""
        # Simplified P&L calculation using duration/convexity
        # In practice, would use proper yield curve mapping and spread analysis
        
        # Get position tenor (simplified)
        position_tenor = self._get_position_tenor(position.maturity_date)
        
        # Get base rate change
        base_rate_change = 0.0
        if scenario.parallel_shift_bps is not None:
            base_rate_change = scenario.parallel_shift_bps / 10000.0
        
        # Calculate P&L using duration approximation
        duration_pnl = -position.duration * float(position.face_value) * base_rate_change
        
        # Add convexity adjustment
        convexity_pnl = 0.5 * position.convexity * float(position.face_value) * (base_rate_change ** 2)
        
        # Add spread impact
        spread_pnl = 0.0
        if position.rating in shocked_spreads:
            # Simplified spread calculation
            spread_change = 0.0
            if scenario.credit_spread_shocks and position.rating in scenario.credit_spread_shocks:
                spread_change += scenario.credit_spread_shocks[position.rating] / 10000.0
            if scenario.liquidity_spread_bps is not None:
                spread_change += scenario.liquidity_spread_bps / 10000.0
            
            spread_pnl = -position.spread_dv01 * float(position.face_value) * spread_change
        
        total_pnl = duration_pnl + convexity_pnl + spread_pnl
        return Decimal(str(total_pnl))
    
    def _calculate_full_reprice(
        self,
        portfolio: List[Position],
        shocked_curves: Dict[str, YieldCurve],
        shocked_spreads: Dict[RatingBucket, Dict[str, float]],
        scenario: StressScenario
    ) -> Dict[str, Any]:
        """Calculate stress test using full reprice method."""
        # This would implement full revaluation using CashFlowEngine and YieldCurveEngine
        # For now, return fast approximation results
        return self._calculate_fast_approximation(
            portfolio, shocked_curves, shocked_spreads, scenario
        )
    
    def _calculate_drilldown_results(
        self,
        portfolio: List[Position],
        result: Dict[str, Any]
    ) -> Dict[str, Dict[str, Decimal]]:
        """Calculate drilldown results by various dimensions."""
        drilldown = {
            'by_issuer': {},
            'by_sector': {},
            'by_rating': {},
            'by_tenor': {}
        }
        
        # This would calculate actual drilldown results
        # For now, return empty results
        return drilldown
    
    def _check_limit_breaches(
        self,
        portfolio: List[Position],
        result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check for limit breaches in stress test results."""
        breaches = []
        
        # Check P&L limits
        if abs(result['total_pnl_bps']) > 1000:  # 10% limit
            breaches.append({
                'type': 'PNL_LIMIT',
                'severity': 'HIGH',
                'description': f"Total P&L {result['total_pnl_bps']:.2f} bps exceeds 1000 bps limit"
            })
        
        # Check duration limits
        if abs(result['delta_dv01']) > 1000000:  # $1M limit
            breaches.append({
                'type': 'DURATION_LIMIT',
                'severity': 'MODERATE',
                'description': f"Duration change {result['delta_dv01']:.0f} exceeds 1M limit"
            })
        
        return breaches
    
    def _generate_narrative(
        self,
        scenario: StressScenario,
        result: Dict[str, Any]
    ) -> str:
        """Generate narrative description of stress test results."""
        narrative = f"Stress test '{scenario.name}' executed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
        
        if scenario.parallel_shift_bps:
            narrative += f"Applied parallel rate shift of {scenario.parallel_shift_bps} bps. "
        
        if scenario.curve_steepening_bps:
            narrative += f"Applied curve steepening of {scenario.curve_steepening_bps} bps. "
        
        if scenario.credit_spread_shocks:
            narrative += f"Applied credit spread shocks ranging from {min(scenario.credit_spread_shocks.values())} to {max(scenario.credit_spread_shocks.values())} bps. "
        
        narrative += f"Total portfolio P&L: {result['total_pnl_bps']:.2f} bps. "
        
        if result['total_pnl_bps'] > 0:
            narrative += "Portfolio value increased under stress scenario. "
        else:
            narrative += "Portfolio value decreased under stress scenario. "
        
        return narrative
    
    def _get_position_tenor(self, maturity_date: date) -> float:
        """Get position tenor in years."""
        today = date.today()
        return (maturity_date - today).days / 365.25
    
    def _generate_cache_key(
        self,
        portfolio: List[Position],
        base_curves: Dict[str, YieldCurve],
        spread_surfaces: Dict[RatingBucket, Dict[str, float]],
        scenario: StressScenario,
        calculation_mode: CalculationMode
    ) -> str:
        """Generate cache key for stress test results."""
        # Simplified cache key generation
        # In practice, would use proper hashing of all inputs
        key_data = {
            'scenario_id': scenario.scenario_id,
            'calculation_mode': calculation_mode.value,
            'portfolio_size': len(portfolio),
            'curves_count': len(base_curves),
            'timestamp': datetime.now().strftime('%Y%m%d%H%M')
        }
        
        return json.dumps(key_data, sort_keys=True)


# Export classes
__all__ = [
    "StressTestingEngine",
    "ScenarioType",
    "CalculationMode",
    "RatingBucket",
    "SectorBucket",
    "Position",
    "StressScenario",
    "StressTestResult"
]
