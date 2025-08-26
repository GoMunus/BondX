"""
Portfolio analytics and attribution engine for BondX Backend.

This module implements portfolio risk metrics, concentration analysis,
and performance attribution suitable for risk and performance reporting.
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
from scipy import stats
from sklearn.decomposition import PCA

from ..core.logging import get_logger
from ..database.models import DayCountConvention
from ..mathematics.yield_curves import YieldCurve, CurveType
from .stress_testing import Position, RatingBucket, SectorBucket

logger = get_logger(__name__)


class AttributionFactor(Enum):
    """Performance attribution factors."""
    CARRY_ROLLDOWN = "CARRY_ROLLDOWN"
    CURVE_LEVEL = "CURVE_LEVEL"
    CURVE_SLOPE = "CURVE_SLOPE"
    CURVE_CURVATURE = "CURVE_CURVATURE"
    CREDIT_SPREAD = "CREDIT_SPREAD"
    SELECTION = "SELECTION"
    TRADING = "TRADING"
    IDIOSYNCRATIC = "IDIOSYNCRATIC"


class TenorBucket(Enum):
    """Tenor buckets for portfolio analysis."""
    ZERO_TO_ONE_YEAR = "0-1Y"
    ONE_TO_THREE_YEARS = "1-3Y"
    THREE_TO_FIVE_YEARS = "3-5Y"
    FIVE_TO_TEN_YEARS = "5-10Y"
    TEN_PLUS_YEARS = "10Y+"


@dataclass
class PortfolioMetrics:
    """Portfolio-level risk metrics."""
    total_market_value: Decimal
    total_face_value: Decimal
    total_book_value: Decimal
    
    # Duration metrics
    portfolio_duration: float
    portfolio_convexity: float
    key_rate_durations: Dict[str, float]
    
    # Spread metrics
    portfolio_spread_dv01: float
    spread_dv_by_rating: Dict[RatingBucket, float]
    spread_dv_by_tenor: Dict[str, float]
    
    # Concentration metrics
    issuer_concentration: Dict[str, float]  # Percentage by issuer
    sector_concentration: Dict[SectorBucket, float]  # Percentage by sector
    rating_concentration: Dict[RatingBucket, float]  # Percentage by rating
    tenor_concentration: Dict[str, float]  # Percentage by tenor
    
    # Liquidity metrics
    portfolio_liquidity_score: float
    illiquid_exposure: float  # Percentage of illiquid positions
    
    # Risk metrics
    portfolio_var_95: Optional[float] = None  # 95% VaR
    portfolio_var_99: Optional[float] = None  # 99% VaR
    expected_shortfall: Optional[float] = None
    
    # Metadata
    calculation_date: date = field(default_factory=date.today)
    positions_count: int = 0


@dataclass
class AttributionResult:
    """Performance attribution results."""
    period_start: date
    period_end: date
    total_return: float
    
    # Factor contributions
    factor_contributions: Dict[AttributionFactor, float]
    factor_contributions_bps: Dict[AttributionFactor, float]
    
    # Curve factor decomposition
    curve_level_contribution: float
    curve_slope_contribution: float
    curve_curvature_contribution: float
    
    # Credit and selection
    credit_spread_contribution: float
    selection_contribution: float
    trading_contribution: float
    idiosyncratic_contribution: float
    
    # Residual
    residual: float
    
    # Metadata
    calculation_method: str
    curve_factors_count: int
    attribution_date: datetime = field(default_factory=datetime.now)


@dataclass
class TurnoverMetrics:
    """Portfolio turnover metrics."""
    period_start: date
    period_end: date
    
    # Turnover measures
    gross_turnover: float  # Gross turnover as percentage
    net_turnover: float    # Net turnover as percentage
    buy_turnover: float    # Buy turnover as percentage
    sell_turnover: float   # Sell turnover as percentage
    
    # Position changes
    positions_added: int
    positions_removed: int
    positions_modified: int
    
    # Value changes
    value_added: Decimal
    value_removed: Decimal
    value_modified: Decimal
    
    # Metadata
    calculation_date: date = field(default_factory=date.today)


class PortfolioAnalytics:
    """
    Portfolio analytics and attribution engine.
    
    Provides comprehensive portfolio risk metrics, concentration analysis,
    and performance attribution suitable for risk and performance reporting.
    """
    
    def __init__(
        self,
        enable_pca: bool = True,
        curve_factors: int = 3,
        attribution_method: str = "FACTOR_MODEL"
    ):
        """
        Initialize portfolio analytics engine.
        
        Args:
            enable_pca: Enable PCA for curve factor decomposition
            curve_factors: Number of curve factors to extract
            attribution_method: Attribution method to use
        """
        self.enable_pca = enable_pca
        self.curve_factors = curve_factors
        self.attribution_method = attribution_method
        self.logger = logger
        
        # Initialize PCA for curve factor decomposition
        if self.enable_pca:
            self.pca = PCA(n_components=self.curve_factors)
        else:
            self.pca = None
    
    def calculate_portfolio_metrics(
        self,
        positions: List[Position],
        yield_curves: Optional[Dict[str, YieldCurve]] = None,
        spread_surfaces: Optional[Dict[RatingBucket, Dict[str, float]]] = None
    ) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Args:
            positions: List of portfolio positions
            yield_curves: Yield curves for calculations
            spread_surfaces: Credit spread surfaces
            
        Returns:
            Portfolio metrics
        """
        try:
            # Basic portfolio values
            total_market_value = sum(pos.market_value for pos in positions)
            total_face_value = sum(pos.face_value for pos in positions)
            total_book_value = sum(pos.book_value for pos in positions)
            
            # Duration metrics
            portfolio_duration = self._calculate_portfolio_duration(positions)
            portfolio_convexity = self._calculate_portfolio_convexity(positions)
            key_rate_durations = self._calculate_key_rate_durations(positions)
            
            # Spread metrics
            portfolio_spread_dv01 = self._calculate_portfolio_spread_dv01(positions)
            spread_dv_by_rating = self._calculate_spread_dv_by_rating(positions)
            spread_dv_by_tenor = self._calculate_spread_dv_by_tenor(positions)
            
            # Concentration metrics
            issuer_concentration = self._calculate_issuer_concentration(positions, total_market_value)
            sector_concentration = self._calculate_sector_concentration(positions, total_market_value)
            rating_concentration = self._calculate_rating_concentration(positions, total_market_value)
            tenor_concentration = self._calculate_tenor_concentration(positions, total_market_value)
            
            # Liquidity metrics
            portfolio_liquidity_score = self._calculate_portfolio_liquidity_score(positions)
            illiquid_exposure = self._calculate_illiquid_exposure(positions, total_market_value)
            
            # Risk metrics (if yield curves provided)
            portfolio_var_95 = None
            portfolio_var_99 = None
            expected_shortfall = None
            
            if yield_curves:
                risk_metrics = self._calculate_risk_metrics(positions, yield_curves)
                portfolio_var_95 = risk_metrics.get('var_95')
                portfolio_var_99 = risk_metrics.get('var_99')
                expected_shortfall = risk_metrics.get('expected_shortfall')
            
            return PortfolioMetrics(
                total_market_value=total_market_value,
                total_face_value=total_face_value,
                total_book_value=total_book_value,
                portfolio_duration=portfolio_duration,
                portfolio_convexity=portfolio_convexity,
                key_rate_durations=key_rate_durations,
                portfolio_spread_dv01=portfolio_spread_dv01,
                spread_dv_by_rating=spread_dv_by_rating,
                spread_dv_by_tenor=spread_dv_by_tenor,
                issuer_concentration=issuer_concentration,
                sector_concentration=sector_concentration,
                rating_concentration=rating_concentration,
                tenor_concentration=tenor_concentration,
                portfolio_liquidity_score=portfolio_liquidity_score,
                illiquid_exposure=illiquid_exposure,
                portfolio_var_95=portfolio_var_95,
                portfolio_var_99=portfolio_var_99,
                expected_shortfall=expected_shortfall,
                calculation_date=date.today(),
                positions_count=len(positions)
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {str(e)}")
            raise
    
    def calculate_performance_attribution(
        self,
        positions_start: List[Position],
        positions_end: List[Position],
        yield_curves_start: Dict[str, YieldCurve],
        yield_curves_end: Dict[str, YieldCurve],
        period_start: date,
        period_end: date,
        benchmark_returns: Optional[Dict[str, float]] = None
    ) -> AttributionResult:
        """
        Calculate performance attribution for a period.
        
        Args:
            positions_start: Portfolio positions at start of period
            positions_end: Portfolio positions at end of period
            yield_curves_start: Yield curves at start of period
            yield_curves_end: Yield curves at end of period
            period_start: Start date of attribution period
            period_end: End date of attribution period
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            Attribution results
        """
        try:
            # Calculate total return
            total_return = self._calculate_total_return(positions_start, positions_end)
            
            # Calculate factor contributions
            factor_contributions = self._calculate_factor_contributions(
                positions_start, positions_end, yield_curves_start, yield_curves_end
            )
            
            # Decompose curve factors
            curve_decomposition = self._decompose_curve_factors(
                yield_curves_start, yield_curves_end
            )
            
            # Calculate credit and selection effects
            credit_contribution = self._calculate_credit_contribution(
                positions_start, positions_end
            )
            
            selection_contribution = self._calculate_selection_contribution(
                positions_start, positions_end
            )
            
            trading_contribution = self._calculate_trading_contribution(
                positions_start, positions_end
            )
            
            # Calculate idiosyncratic effect
            idiosyncratic_contribution = self._calculate_idiosyncratic_contribution(
                total_return, factor_contributions, credit_contribution,
                selection_contribution, trading_contribution
            )
            
            # Calculate residual
            residual = self._calculate_residual(
                total_return, factor_contributions, credit_contribution,
                selection_contribution, trading_contribution, idiosyncratic_contribution
            )
            
            # Convert to basis points
            factor_contributions_bps = {
                factor: contribution * 10000 for factor, contribution in factor_contributions.items()
            }
            
            return AttributionResult(
                period_start=period_start,
                period_end=period_end,
                total_return=total_return,
                factor_contributions=factor_contributions,
                factor_contributions_bps=factor_contributions_bps,
                curve_level_contribution=curve_decomposition.get('level', 0.0),
                curve_slope_contribution=curve_decomposition.get('slope', 0.0),
                curve_curvature_contribution=curve_decomposition.get('curvature', 0.0),
                credit_spread_contribution=credit_contribution,
                selection_contribution=selection_contribution,
                trading_contribution=trading_contribution,
                idiosyncratic_contribution=idiosyncratic_contribution,
                residual=residual,
                calculation_method=self.attribution_method,
                curve_factors_count=self.curve_factors
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating performance attribution: {str(e)}")
            raise
    
    def calculate_turnover_metrics(
        self,
        positions_start: List[Position],
        positions_end: List[Position],
        period_start: date,
        period_end: date
    ) -> TurnoverMetrics:
        """
        Calculate portfolio turnover metrics.
        
        Args:
            positions_start: Portfolio positions at start of period
            positions_end: Portfolio positions at end of period
            period_start: Start date of turnover period
            period_end: End date of turnover period
            
        Returns:
            Turnover metrics
        """
        try:
            # Create position maps for comparison
            start_positions = {pos.instrument_id: pos for pos in positions_start}
            end_positions = {pos.instrument_id: pos for pos in positions_end}
            
            # Calculate position changes
            positions_added = len(set(end_positions.keys()) - set(start_positions.keys()))
            positions_removed = len(set(start_positions.keys()) - set(end_positions.keys()))
            positions_modified = len(set(start_positions.keys()) & set(end_positions.keys()))
            
            # Calculate value changes
            value_added = sum(
                end_positions[inst_id].market_value
                for inst_id in set(end_positions.keys()) - set(start_positions.keys())
            )
            
            value_removed = sum(
                start_positions[inst_id].market_value
                for inst_id in set(start_positions.keys()) - set(end_positions.keys())
            )
            
            value_modified = sum(
                abs(end_positions[inst_id].market_value - start_positions[inst_id].market_value)
                for inst_id in set(start_positions.keys()) & set(end_positions.keys())
            )
            
            # Calculate turnover percentages
            total_start_value = sum(pos.market_value for pos in positions_start)
            total_end_value = sum(pos.market_value for pos in positions_end)
            avg_portfolio_value = (total_start_value + total_end_value) / 2
            
            if avg_portfolio_value > 0:
                gross_turnover = float((value_added + value_removed + value_modified) / avg_portfolio_value)
                net_turnover = float(abs(value_added - value_removed) / avg_portfolio_value)
                buy_turnover = float(value_added / avg_portfolio_value)
                sell_turnover = float(value_removed / avg_portfolio_value)
            else:
                gross_turnover = net_turnover = buy_turnover = sell_turnover = 0.0
            
            return TurnoverMetrics(
                period_start=period_start,
                period_end=period_end,
                gross_turnover=gross_turnover,
                net_turnover=net_turnover,
                buy_turnover=buy_turnover,
                sell_turnover=sell_turnover,
                positions_added=positions_added,
                positions_removed=positions_removed,
                positions_modified=positions_modified,
                value_added=value_added,
                value_removed=value_removed,
                value_modified=value_modified,
                calculation_date=date.today()
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating turnover metrics: {str(e)}")
            raise
    
    def _calculate_portfolio_duration(self, positions: List[Position]) -> float:
        """Calculate portfolio duration."""
        total_value = sum(float(pos.market_value) for pos in positions)
        if total_value == 0:
            return 0.0
        
        weighted_duration = sum(
            float(pos.market_value) * pos.duration for pos in positions
        )
        
        return weighted_duration / total_value
    
    def _calculate_portfolio_convexity(self, positions: List[Position]) -> float:
        """Calculate portfolio convexity."""
        total_value = sum(float(pos.market_value) for pos in positions)
        if total_value == 0:
            return 0.0
        
        weighted_convexity = sum(
            float(pos.market_value) * pos.convexity for pos in positions
        )
        
        return weighted_convexity / total_value
    
    def _calculate_key_rate_durations(self, positions: List[Position]) -> Dict[str, float]:
        """Calculate key rate durations."""
        key_rate_durations = {}
        
        # Group positions by tenor bucket
        for position in positions:
            tenor_bucket = position.tenor_bucket
            if tenor_bucket not in key_rate_durations:
                key_rate_durations[tenor_bucket] = 0.0
            
            key_rate_durations[tenor_bucket] += (
                float(position.market_value) * position.duration
            )
        
        # Normalize by total portfolio value
        total_value = sum(float(pos.market_value) for pos in positions)
        if total_value > 0:
            for bucket in key_rate_durations:
                key_rate_durations[bucket] /= total_value
        
        return key_rate_durations
    
    def _calculate_portfolio_spread_dv01(self, positions: List[Position]) -> float:
        """Calculate portfolio spread DV01."""
        total_value = sum(float(pos.market_value) for pos in positions)
        if total_value == 0:
            return 0.0
        
        weighted_spread_dv = sum(
            float(pos.market_value) * pos.spread_dv01 for pos in positions
        )
        
        return weighted_spread_dv / total_value
    
    def _calculate_spread_dv_by_rating(self, positions: List[Position]) -> Dict[RatingBucket, float]:
        """Calculate spread DV01 by rating bucket."""
        spread_dv_by_rating = {rating: 0.0 for rating in RatingBucket}
        
        for position in positions:
            spread_dv_by_rating[position.rating] += (
                float(position.market_value) * position.spread_dv01
            )
        
        # Normalize by total portfolio value
        total_value = sum(float(pos.market_value) for pos in positions)
        if total_value > 0:
            for rating in spread_dv_by_rating:
                spread_dv_by_rating[rating] /= total_value
        
        return spread_dv_by_rating
    
    def _calculate_spread_dv_by_tenor(self, positions: List[Position]) -> Dict[str, float]:
        """Calculate spread DV01 by tenor bucket."""
        spread_dv_by_tenor = {}
        
        for position in positions:
            tenor_bucket = position.tenor_bucket
            if tenor_bucket not in spread_dv_by_tenor:
                spread_dv_by_tenor[tenor_bucket] = 0.0
            
            spread_dv_by_tenor[tenor_bucket] += (
                float(position.market_value) * position.spread_dv01
            )
        
        # Normalize by total portfolio value
        total_value = sum(float(pos.market_value) for pos in positions)
        if total_value > 0:
            for bucket in spread_dv_by_tenor:
                spread_dv_by_tenor[bucket] /= total_value
        
        return spread_dv_by_tenor
    
    def _calculate_issuer_concentration(
        self,
        positions: List[Position],
        total_market_value: Decimal
    ) -> Dict[str, float]:
        """Calculate issuer concentration."""
        issuer_values = {}
        
        for position in positions:
            if position.issuer_id not in issuer_values:
                issuer_values[position.issuer_id] = Decimal('0')
            issuer_values[position.issuer_id] += position.market_value
        
        # Convert to percentages
        issuer_concentration = {}
        for issuer_id, value in issuer_values.items():
            issuer_concentration[issuer_id] = float(value / total_market_value * 100)
        
        return issuer_concentration
    
    def _calculate_sector_concentration(
        self,
        positions: List[Position],
        total_market_value: Decimal
    ) -> Dict[SectorBucket, float]:
        """Calculate sector concentration."""
        sector_values = {sector: Decimal('0') for sector in SectorBucket}
        
        for position in positions:
            sector_values[position.sector] += position.market_value
        
        # Convert to percentages
        sector_concentration = {}
        for sector, value in sector_values.items():
            sector_concentration[sector] = float(value / total_market_value * 100)
        
        return sector_concentration
    
    def _calculate_rating_concentration(
        self,
        positions: List[Position],
        total_market_value: Decimal
    ) -> Dict[RatingBucket, float]:
        """Calculate rating concentration."""
        rating_values = {rating: Decimal('0') for rating in RatingBucket}
        
        for position in positions:
            rating_values[position.rating] += position.market_value
        
        # Convert to percentages
        rating_concentration = {}
        for rating, value in rating_values.items():
            rating_concentration[rating] = float(value / total_market_value * 100)
        
        return rating_concentration
    
    def _calculate_tenor_concentration(
        self,
        positions: List[Position],
        total_market_value: Decimal
    ) -> Dict[str, float]:
        """Calculate tenor concentration."""
        tenor_values = {}
        
        for position in positions:
            tenor_bucket = position.tenor_bucket
            if tenor_bucket not in tenor_values:
                tenor_values[tenor_bucket] = Decimal('0')
            tenor_values[tenor_bucket] += position.market_value
        
        # Convert to percentages
        tenor_concentration = {}
        for tenor, value in tenor_values.items():
            tenor_concentration[tenor] = float(value / total_market_value * 100)
        
        return tenor_concentration
    
    def _calculate_portfolio_liquidity_score(self, positions: List[Position]) -> float:
        """Calculate portfolio liquidity score."""
        total_value = sum(float(pos.market_value) for pos in positions)
        if total_value == 0:
            return 0.0
        
        weighted_liquidity = sum(
            float(pos.market_value) * pos.liquidity_score for pos in positions
        )
        
        return weighted_liquidity / total_value
    
    def _calculate_illiquid_exposure(
        self,
        positions: List[Position],
        total_market_value: Decimal
    ) -> float:
        """Calculate illiquid exposure percentage."""
        illiquid_threshold = 0.5  # Positions with liquidity score < 0.5 are illiquid
        
        illiquid_value = sum(
            pos.market_value for pos in positions if pos.liquidity_score < illiquid_threshold
        )
        
        return float(illiquid_value / total_market_value * 100)
    
    def _calculate_risk_metrics(
        self,
        positions: List[Position],
        yield_curves: Dict[str, YieldCurve]
    ) -> Dict[str, float]:
        """Calculate portfolio risk metrics."""
        # Simplified risk calculation
        # In practice, would use proper VaR models
        
        # Calculate portfolio volatility (simplified)
        position_returns = []
        for position in positions:
            # Simplified return calculation
            # In practice, would use actual price changes
            position_returns.append(position.duration * 0.01)  # 1% rate change
        
        if position_returns:
            portfolio_vol = np.std(position_returns)
            
            # Calculate VaR (simplified)
            var_95 = np.percentile(position_returns, 5)
            var_99 = np.percentile(position_returns, 1)
            
            # Calculate expected shortfall
            expected_shortfall = np.mean([r for r in position_returns if r <= var_95])
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'expected_shortfall': expected_shortfall
            }
        
        return {}
    
    def _calculate_total_return(
        self,
        positions_start: List[Position],
        positions_end: List[Position]
    ) -> float:
        """Calculate total portfolio return."""
        start_value = sum(float(pos.market_value) for pos in positions_start)
        end_value = sum(float(pos.market_value) for pos in positions_end)
        
        if start_value == 0:
            return 0.0
        
        return (end_value - start_value) / start_value
    
    def _calculate_factor_contributions(
        self,
        positions_start: List[Position],
        positions_end: List[Position],
        yield_curves_start: Dict[str, YieldCurve],
        yield_curves_end: Dict[str, YieldCurve]
    ) -> Dict[AttributionFactor, float]:
        """Calculate factor contributions."""
        # Simplified factor contribution calculation
        # In practice, would use proper factor models
        
        factor_contributions = {}
        
        # Carry/roll-down (simplified)
        factor_contributions[AttributionFactor.CARRY_ROLLDOWN] = 0.02  # 2% assumption
        
        # Curve factors (will be calculated separately)
        factor_contributions[AttributionFactor.CURVE_LEVEL] = 0.0
        factor_contributions[AttributionFactor.CURVE_SLOPE] = 0.0
        factor_contributions[AttributionFactor.CURVE_CURVATURE] = 0.0
        
        # Credit spread
        factor_contributions[AttributionFactor.CREDIT_SPREAD] = 0.01  # 1% assumption
        
        # Selection
        factor_contributions[AttributionFactor.SELECTION] = 0.005  # 0.5% assumption
        
        # Trading
        factor_contributions[AttributionFactor.TRADING] = 0.003  # 0.3% assumption
        
        # Idiosyncratic
        factor_contributions[AttributionFactor.IDIOSYNCRATIC] = 0.002  # 0.2% assumption
        
        return factor_contributions
    
    def _decompose_curve_factors(
        self,
        yield_curves_start: Dict[str, YieldCurve],
        yield_curves_end: Dict[str, YieldCurve]
    ) -> Dict[str, float]:
        """Decompose yield curve changes into factors."""
        if not self.enable_pca or not self.pca:
            return {'level': 0.0, 'slope': 0.0, 'curvature': 0.0}
        
        try:
            # Extract curve data for PCA
            curve_data = []
            for curve_id, start_curve in yield_curves_start.items():
                if curve_id in yield_curves_end:
                    end_curve = yield_curves_end[curve_id]
                    # Calculate rate changes
                    rate_changes = end_curve.rates - start_curve.rates
                    curve_data.append(rate_changes)
            
            if not curve_data:
                return {'level': 0.0, 'slope': 0.0, 'curvature': 0.0}
            
            # Stack curves and fit PCA
            curve_matrix = np.vstack(curve_data)
            self.pca.fit(curve_matrix.T)
            
            # Extract factors
            factors = self.pca.components_
            explained_variance = self.pca.explained_variance_ratio_
            
            # Map to level, slope, curvature
            curve_decomposition = {
                'level': explained_variance[0] if len(explained_variance) > 0 else 0.0,
                'slope': explained_variance[1] if len(explained_variance) > 1 else 0.0,
                'curvature': explained_variance[2] if len(explained_variance) > 2 else 0.0
            }
            
            return curve_decomposition
            
        except Exception as e:
            self.logger.warning(f"Could not decompose curve factors: {str(e)}")
            return {'level': 0.0, 'slope': 0.0, 'curvature': 0.0}
    
    def _calculate_credit_contribution(
        self,
        positions_start: List[Position],
        positions_end: List[Position]
    ) -> float:
        """Calculate credit spread contribution."""
        # Simplified credit contribution calculation
        # In practice, would use actual spread changes
        
        start_spread_dv = sum(
            float(pos.market_value) * pos.spread_dv01 for pos in positions_start
        )
        end_spread_dv = sum(
            float(pos.market_value) * pos.spread_dv01 for pos in positions_end
        )
        
        if start_spread_dv == 0:
            return 0.0
        
        # Assume 10bp spread change
        spread_change = 0.001
        return (end_spread_dv - start_spread_dv) * spread_change / start_spread_dv
    
    def _calculate_selection_contribution(
        self,
        positions_start: List[Position],
        positions_end: List[Position]
    ) -> float:
        """Calculate selection contribution."""
        # Simplified selection contribution
        # In practice, would compare to benchmark
        
        return 0.005  # 0.5% assumption
    
    def _calculate_trading_contribution(
        self,
        positions_start: List[Position],
        positions_end: List[Position]
    ) -> float:
        """Calculate trading contribution."""
        # Simplified trading contribution
        # In practice, would use actual trade data
        
        return 0.003  # 0.3% assumption
    
    def _calculate_idiosyncratic_contribution(
        self,
        total_return: float,
        factor_contributions: Dict[AttributionFactor, float],
        credit_contribution: float,
        selection_contribution: float,
        trading_contribution: float
    ) -> float:
        """Calculate idiosyncratic contribution."""
        # Idiosyncratic is the residual after accounting for all factors
        explained_return = (
            sum(factor_contributions.values()) +
            credit_contribution +
            selection_contribution +
            trading_contribution
        )
        
        return total_return - explained_return
    
    def _calculate_residual(
        self,
        total_return: float,
        factor_contributions: Dict[AttributionFactor, float],
        credit_contribution: float,
        selection_contribution: float,
        trading_contribution: float,
        idiosyncratic_contribution: float
    ) -> float:
        """Calculate residual."""
        # Residual should be zero if attribution is complete
        explained_return = (
            sum(factor_contributions.values()) +
            credit_contribution +
            selection_contribution +
            trading_contribution +
            idiosyncratic_contribution
        )
        
        return total_return - explained_return


# Export classes
__all__ = [
    "PortfolioAnalytics",
    "AttributionFactor",
    "TenorBucket",
    "PortfolioMetrics",
    "AttributionResult",
    "TurnoverMetrics"
]
