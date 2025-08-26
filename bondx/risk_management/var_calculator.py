"""
VaR Calculator for BondX Backend.

This module implements Value at Risk (VaR) calculations for bond portfolios
including parametric (delta-normal) and historical simulation methods.
"""

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import warnings

from ..core.logging import get_logger
from ..mathematics.yield_curves import YieldCurve
from ..mathematics.cash_flows import CashFlow

logger = get_logger(__name__)


class VaRMethod(Enum):
    """VaR calculation methods."""
    PARAMETRIC = "PARAMETRIC"  # Delta-normal VaR
    HISTORICAL = "HISTORICAL"  # Historical simulation VaR


class ConfidenceLevel(Enum):
    """Confidence levels for VaR calculations."""
    P95 = 0.95  # 95% confidence
    P99 = 0.99  # 99% confidence
    P99_9 = 0.999  # 99.9% confidence


@dataclass
class Position:
    """Represents a bond position in a portfolio."""
    instrument_id: str
    face_value: Decimal
    market_value: Decimal
    duration: float  # Modified duration
    convexity: float
    dv01: float  # Dollar value of 1bp change in yield
    credit_spread: float  # Credit spread in basis points
    issuer_rating: str
    sector: str
    maturity_date: date
    coupon_rate: float
    yield_to_maturity: float


@dataclass
class RiskFactor:
    """Represents a risk factor for VaR calculations."""
    factor_id: str
    factor_type: str  # "rate", "spread", "credit"
    tenor: Optional[float] = None  # For rate factors
    rating_bucket: Optional[str] = None  # For credit factors
    sector: Optional[str] = None  # For sector factors
    current_value: float = 0.0
    volatility: float = 0.0
    historical_returns: Optional[np.ndarray] = None


@dataclass
class VaRResult:
    """Result of VaR calculation."""
    var_value: float
    confidence_level: float
    time_horizon: float  # in days
    method: VaRMethod
    calculation_date: date
    portfolio_value: float
    var_contribution: Dict[str, float]  # By position
    risk_factors: List[str]
    metadata: Dict


class VaRCalculator:
    """
    Production-grade VaR calculator for bond portfolios.
    
    Supports both parametric (delta-normal) and historical simulation methods
    with configurable confidence levels and time horizons.
    """
    
    def __init__(self):
        """Initialize the VaR calculator."""
        self.logger = logger
        self._risk_factor_cache: Dict[str, RiskFactor] = {}
    
    def calculate_var(
        self,
        positions: List[Position],
        risk_factors: List[RiskFactor],
        method: VaRMethod = VaRMethod.PARAMETRIC,
        confidence_level: ConfidenceLevel = ConfidenceLevel.P95,
        time_horizon: float = 1.0,  # 1 day default
        use_full_repricing: bool = False,
        historical_data: Optional[pd.DataFrame] = None
    ) -> VaRResult:
        """
        Calculate VaR for a portfolio of bond positions.
        
        Args:
            positions: List of bond positions
            risk_factors: List of risk factors
            method: VaR calculation method
            confidence_level: Confidence level for VaR
            time_horizon: Time horizon in days
            use_full_repricing: Whether to use full repricing for historical VaR
            historical_data: Historical factor data for historical VaR
            
        Returns:
            VaR calculation result
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not positions:
            raise ValueError("At least one position is required")
        
        if not risk_factors:
            raise ValueError("At least one risk factor is required")
        
        # Validate inputs
        self._validate_inputs(positions, risk_factors, method, historical_data)
        
        # Calculate portfolio value
        portfolio_value = sum(float(pos.market_value) for pos in positions)
        
        if method == VaRMethod.PARAMETRIC:
            var_value, var_contribution = self._calculate_parametric_var(
                positions, risk_factors, confidence_level, time_horizon
            )
        else:  # Historical
            var_value, var_contribution = self._calculate_historical_var(
                positions, risk_factors, confidence_level, time_horizon,
                use_full_repricing, historical_data
            )
        
        # Create result object
        result = VaRResult(
            var_value=var_value,
            confidence_level=confidence_level.value,
            time_horizon=time_horizon,
            method=method,
            calculation_date=date.today(),
            portfolio_value=portfolio_value,
            var_contribution=var_contribution,
            risk_factors=[rf.factor_id for rf in risk_factors],
            metadata={
                "method": method.value,
                "confidence_level": confidence_level.value,
                "time_horizon": time_horizon,
                "num_positions": len(positions),
                "num_risk_factors": len(risk_factors)
            }
        )
        
        self.logger.info(
            f"VaR calculated: {var_value:.2f} at {confidence_level.value*100}% confidence",
            method=method.value,
            portfolio_value=portfolio_value,
            var_value=var_value
        )
        
        return result
    
    def _validate_inputs(
        self,
        positions: List[Position],
        risk_factors: List[RiskFactor],
        method: VaRMethod,
        historical_data: Optional[pd.DataFrame]
    ):
        """Validate input parameters."""
        # Check for duplicate instrument IDs
        instrument_ids = [pos.instrument_id for pos in positions]
        if len(instrument_ids) != len(set(instrument_ids)):
            raise ValueError("Duplicate instrument IDs found in positions")
        
        # Check for duplicate factor IDs
        factor_ids = [rf.factor_id for rf in risk_factors]
        if len(factor_ids) != len(set(factor_ids)):
            raise ValueError("Duplicate factor IDs found in risk factors")
        
        # Validate position data
        for pos in positions:
            if pos.face_value <= 0:
                raise ValueError(f"Invalid face value for position {pos.instrument_id}")
            if pos.market_value <= 0:
                raise ValueError(f"Invalid market value for position {pos.instrument_id}")
            if pos.duration < 0:
                raise ValueError(f"Invalid duration for position {pos.instrument_id}")
            if pos.dv01 < 0:
                raise ValueError(f"Invalid DV01 for position {pos.instrument_id}")
        
        # Validate risk factors
        for rf in risk_factors:
            if rf.volatility < 0:
                raise ValueError(f"Invalid volatility for risk factor {rf.factor_id}")
        
        # Check historical data for historical VaR
        if method == VaRMethod.HISTORICAL and historical_data is None:
            raise ValueError("Historical data required for historical VaR")
    
    def _calculate_parametric_var(
        self,
        positions: List[Position],
        risk_factors: List[RiskFactor],
        confidence_level: ConfidenceLevel,
        time_horizon: float
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate parametric (delta-normal) VaR."""
        # Build covariance matrix of risk factors
        cov_matrix = self._build_covariance_matrix(risk_factors)
        
        # Calculate portfolio sensitivities to risk factors
        portfolio_sensitivities = self._calculate_portfolio_sensitivities(positions, risk_factors)
        
        # Calculate portfolio variance
        portfolio_variance = portfolio_sensitivities.T @ cov_matrix @ portfolio_sensitivities
        
        # Apply time scaling
        portfolio_variance *= time_horizon
        
        # Calculate VaR using normal distribution assumption
        z_score = self._get_z_score(confidence_level)
        var_value = z_score * np.sqrt(portfolio_variance)
        
        # Calculate VaR contribution by position
        var_contribution = self._calculate_position_var_contribution(
            positions, risk_factors, cov_matrix, z_score, time_horizon
        )
        
        return float(var_value), var_contribution
    
    def _calculate_historical_var(
        self,
        positions: List[Position],
        risk_factors: List[RiskFactor],
        confidence_level: ConfidenceLevel,
        time_horizon: float,
        use_full_repricing: bool,
        historical_data: pd.DataFrame
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate historical simulation VaR."""
        # Get historical factor returns
        factor_returns = self._get_historical_factor_returns(risk_factors, historical_data)
        
        if factor_returns.empty:
            raise ValueError("No historical factor returns available")
        
        # Calculate historical P&L scenarios
        pnl_scenarios = self._calculate_historical_pnl_scenarios(
            positions, risk_factors, factor_returns, use_full_repricing
        )
        
        # Calculate VaR from P&L distribution
        var_percentile = (1 - confidence_level.value) * 100
        var_value = np.percentile(pnl_scenarios, var_percentile)
        
        # Calculate VaR contribution by position (simplified)
        var_contribution = self._calculate_historical_var_contribution(
            positions, pnl_scenarios, var_percentile
        )
        
        return float(var_value), var_contribution
    
    def _build_covariance_matrix(self, risk_factors: List[RiskFactor]) -> np.ndarray:
        """Build covariance matrix of risk factors."""
        n_factors = len(risk_factors)
        cov_matrix = np.zeros((n_factors, n_factors))
        
        for i, rf1 in enumerate(risk_factors):
            for j, rf2 in enumerate(risk_factors):
                if i == j:
                    # Diagonal: variance
                    cov_matrix[i, j] = rf1.volatility ** 2
                else:
                    # Off-diagonal: covariance (simplified correlation assumption)
                    correlation = self._estimate_correlation(rf1, rf2)
                    cov_matrix[i, j] = correlation * rf1.volatility * rf2.volatility
        
        return cov_matrix
    
    def _estimate_correlation(self, rf1: RiskFactor, rf2: RiskFactor) -> float:
        """Estimate correlation between two risk factors."""
        # Simplified correlation estimation based on factor types
        if rf1.factor_type == rf2.factor_type:
            if rf1.factor_type == "rate":
                # Rate factors: high correlation within same currency
                return 0.8
            elif rf1.factor_type == "spread":
                # Spread factors: moderate correlation
                return 0.6
            elif rf1.factor_type == "credit":
                # Credit factors: correlation based on rating proximity
                return self._estimate_credit_correlation(rf1, rf2)
        else:
            # Different factor types: lower correlation
            return 0.3
        
        return 0.0
    
    def _estimate_credit_correlation(self, rf1: RiskFactor, rf2: RiskFactor) -> float:
        """Estimate correlation between credit factors."""
        if not rf1.rating_bucket or not rf2.rating_bucket:
            return 0.5
        
        # Rating buckets: AAA, AA, A, BBB, BB, B, CCC
        rating_order = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
        
        try:
            idx1 = rating_order.index(rf1.rating_bucket)
            idx2 = rating_order.index(rf2.rating_bucket)
            distance = abs(idx1 - idx2)
            
            # Higher correlation for closer ratings
            if distance == 0:
                return 0.9
            elif distance == 1:
                return 0.7
            elif distance == 2:
                return 0.5
            else:
                return 0.3
        except ValueError:
            return 0.5
    
    def _calculate_portfolio_sensitivities(
        self,
        positions: List[Position],
        risk_factors: List[RiskFactor]
    ) -> np.ndarray:
        """Calculate portfolio sensitivities to risk factors."""
        n_factors = len(risk_factors)
        sensitivities = np.zeros(n_factors)
        
        for rf_idx, rf in enumerate(risk_factors):
            for pos in positions:
                if rf.factor_type == "rate":
                    # Rate sensitivity: DV01
                    sensitivities[rf_idx] += pos.dv01
                elif rf.factor_type == "spread":
                    # Spread sensitivity: credit DV01 (approximated)
                    sensitivities[rf_idx] += pos.dv01 * 0.1  # Simplified assumption
                elif rf.factor_type == "credit":
                    # Credit sensitivity: based on rating bucket
                    if rf.rating_bucket == pos.issuer_rating:
                        sensitivities[rf_idx] += pos.market_value * 0.01  # 1% sensitivity
        
        return sensitivities
    
    def _get_z_score(self, confidence_level: ConfidenceLevel) -> float:
        """Get z-score for given confidence level."""
        if confidence_level == ConfidenceLevel.P95:
            return 1.645
        elif confidence_level == ConfidenceLevel.P99:
            return 2.326
        elif confidence_level == ConfidenceLevel.P99_9:
            return 3.090
        else:
            return 1.645  # Default to 95%
    
    def _calculate_position_var_contribution(
        self,
        positions: List[Position],
        risk_factors: List[RiskFactor],
        cov_matrix: np.ndarray,
        z_score: float,
        time_horizon: float
    ) -> Dict[str, float]:
        """Calculate VaR contribution by position."""
        var_contribution = {}
        
        for pos in positions:
            # Calculate position sensitivity to risk factors
            position_sensitivities = np.zeros(len(risk_factors))
            
            for rf_idx, rf in enumerate(risk_factors):
                if rf.factor_type == "rate":
                    position_sensitivities[rf_idx] = pos.dv01
                elif rf.factor_type == "spread":
                    position_sensitivities[rf_idx] = pos.dv01 * 0.1
                elif rf.factor_type == "credit":
                    if rf.rating_bucket == pos.issuer_rating:
                        position_sensitivities[rf_idx] = pos.market_value * 0.01
            
            # Calculate position variance contribution
            position_variance = position_sensitivities.T @ cov_matrix @ position_sensitivities
            position_variance *= time_horizon
            
            # Position VaR contribution
            position_var = z_score * np.sqrt(position_variance)
            var_contribution[pos.instrument_id] = float(position_var)
        
        return var_contribution
    
    def _get_historical_factor_returns(
        self,
        risk_factors: List[RiskFactor],
        historical_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Get historical factor returns from data."""
        # This is a simplified implementation
        # In practice, you would need to properly align historical data with risk factors
        
        factor_returns = pd.DataFrame()
        
        for rf in risk_factors:
            if rf.factor_id in historical_data.columns:
                # Calculate returns
                factor_returns[rf.factor_id] = historical_data[rf.factor_id].pct_change().dropna()
        
        return factor_returns
    
    def _calculate_historical_pnl_scenarios(
        self,
        positions: List[Position],
        risk_factors: List[RiskFactor],
        factor_returns: pd.DataFrame,
        use_full_repricing: bool
    ) -> np.ndarray:
        """Calculate historical P&L scenarios."""
        if use_full_repricing:
            # Full repricing approach (simplified)
            pnl_scenarios = self._full_repricing_pnl(positions, risk_factors, factor_returns)
        else:
            # Fast approximation using sensitivities
            pnl_scenarios = self._approximate_pnl(positions, risk_factors, factor_returns)
        
        return pnl_scenarios
    
    def _approximate_pnl(
        self,
        positions: List[Position],
        risk_factors: List[RiskFactor],
        factor_returns: pd.DataFrame
    ) -> np.ndarray:
        """Calculate P&L using sensitivity approximation."""
        n_scenarios = len(factor_returns)
        pnl_scenarios = np.zeros(n_scenarios)
        
        for scenario_idx in range(n_scenarios):
            for pos in positions:
                scenario_pnl = 0.0
                
                for rf in risk_factors:
                    if rf.factor_id in factor_returns.columns:
                        factor_return = factor_returns.iloc[scenario_idx][rf.factor_id]
                        
                        if rf.factor_type == "rate":
                            # Rate change impact
                            scenario_pnl -= pos.dv01 * factor_return * 10000  # Convert to bps
                        elif rf.factor_type == "spread":
                            # Spread change impact
                            scenario_pnl -= pos.dv01 * factor_return * 10000 * 0.1
                        elif rf.factor_type == "credit":
                            # Credit change impact
                            if rf.rating_bucket == pos.issuer_rating:
                                scenario_pnl -= pos.market_value * factor_return
                
                pnl_scenarios[scenario_idx] += scenario_pnl
        
        return pnl_scenarios
    
    def _full_repricing_pnl(
        self,
        positions: List[Position],
        risk_factors: List[RiskFactor],
        factor_returns: pd.DataFrame
    ) -> np.ndarray:
        """Calculate P&L using full repricing (simplified)."""
        # This is a placeholder for full repricing implementation
        # In practice, you would reprice each position under each scenario
        
        n_scenarios = len(factor_returns)
        pnl_scenarios = np.zeros(n_scenarios)
        
        # Simplified implementation using duration approximation
        for scenario_idx in range(n_scenarios):
            for pos in positions:
                # Approximate price change using duration
                total_yield_change = 0.0
                
                for rf in risk_factors:
                    if rf.factor_id in factor_returns.columns:
                        factor_return = factor_returns.iloc[scenario_idx][rf.factor_id]
                        
                        if rf.factor_type == "rate":
                            total_yield_change += factor_return
                        elif rf.factor_type == "spread":
                            total_yield_change += factor_return * 0.1
                
                # Price change = -duration * yield_change + 0.5 * convexity * yield_change^2
                price_change = (-pos.duration * total_yield_change + 
                               0.5 * pos.convexity * total_yield_change ** 2)
                
                pnl_scenarios[scenario_idx] += float(pos.market_value) * price_change
        
        return pnl_scenarios
    
    def _calculate_historical_var_contribution(
        self,
        positions: List[Position],
        pnl_scenarios: np.ndarray,
        var_percentile: float
    ) -> Dict[str, float]:
        """Calculate VaR contribution by position for historical VaR."""
        # Simplified contribution calculation
        var_contribution = {}
        
        for pos in positions:
            # Approximate contribution based on position size
            position_weight = float(pos.market_value) / sum(float(p.market_value) for p in positions)
            var_contribution[pos.instrument_id] = float(np.percentile(pnl_scenarios, var_percentile) * position_weight)
        
        return var_contribution
    
    def calculate_cvar(self, pnl_scenarios: np.ndarray, var_percentile: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var_value = np.percentile(pnl_scenarios, var_percentile)
        tail_scenarios = pnl_scenarios[pnl_scenarios <= var_value]
        
        if len(tail_scenarios) == 0:
            return var_value
        
        return float(np.mean(tail_scenarios))
    
    def backtest_var(
        self,
        var_predictions: List[float],
        actual_pnl: List[float],
        confidence_level: ConfidenceLevel = ConfidenceLevel.P95
    ) -> Dict:
        """Perform VaR backtesting using Kupiec test."""
        if len(var_predictions) != len(actual_pnl):
            raise ValueError("VaR predictions and actual P&L must have same length")
        
        # Count VaR violations
        expected_violations = (1 - confidence_level.value) * len(var_predictions)
        actual_violations = sum(1 for pred, actual in zip(var_predictions, actual_pnl) 
                              if actual < -pred)
        
        # Kupiec test statistic
        if actual_violations == 0:
            test_statistic = 0
        else:
            violation_rate = actual_violations / len(var_predictions)
            expected_rate = 1 - confidence_level.value
            
            test_statistic = -2 * np.log(
                (violation_rate ** actual_violations) * 
                ((1 - violation_rate) ** (len(var_predictions) - actual_violations)) /
                (expected_rate ** actual_violations) * 
                ((1 - expected_rate) ** (len(var_predictions) - actual_violations))
            )
        
        # Critical value at 5% significance (chi-square with 1 degree of freedom)
        critical_value = 3.841
        
        return {
            "expected_violations": expected_violations,
            "actual_violations": actual_violations,
            "violation_rate": actual_violations / len(var_predictions),
            "expected_rate": 1 - confidence_level.value,
            "test_statistic": test_statistic,
            "critical_value": critical_value,
            "reject_null": test_statistic > critical_value,
            "p_value": 1 - self._chi2_cdf(test_statistic, 1)
        }
    
    def _chi2_cdf(self, x: float, df: int) -> float:
        """Calculate chi-square CDF (simplified approximation)."""
        # This is a simplified implementation
        # In practice, use scipy.stats.chi2.cdf
        if x <= 0:
            return 0.0
        elif x < df:
            return 0.5 * (x / df) ** (df / 2)
        else:
            return 1.0 - 0.5 * np.exp(-(x - df) / 2)


__all__ = [
    "VaRCalculator",
    "VaRResult",
    "Position",
    "RiskFactor",
    "VaRMethod",
    "ConfidenceLevel"
]
