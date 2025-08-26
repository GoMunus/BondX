"""
Volatility Models for BondX Risk Management System

This module provides volatility modeling utilities for interest rates and credit spreads.

Features:
- Historical windowed volatility
- EWMA with configurable λ
- Realized volatility aggregations
- Optional GARCH(1,1) scaffold
- Volatility term structures with tenor smoothing
- Performance: ≤40ms for 200 factors per update
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import warnings
from scipy import interpolate
from scipy.optimize import minimize
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class VolatilityMethod(Enum):
    """Available volatility estimation methods."""
    HISTORICAL = "historical"
    EWMA = "ewma"
    REALIZED = "realized"
    GARCH = "garch"

class AggregationMethod(Enum):
    """Volatility aggregation methods."""
    SIMPLE = "simple"
    SQUARE_ROOT_TIME = "square_root_time"
    WEIGHTED = "weighted"

@dataclass
class VolatilityConfig:
    """Configuration for volatility modeling."""
    method: VolatilityMethod = VolatilityMethod.EWMA
    window_size: int = 125  # Trading days
    ewma_lambda: float = 0.94  # RiskMetrics standard
    garch_p: int = 1
    garch_q: int = 1
    min_volatility: float = 1e-6  # Minimum volatility
    max_volatility: float = 10.0  # Maximum volatility (1000%)
    outlier_threshold: float = 5.0  # Standard deviations for outlier detection
    enable_smoothing: bool = True
    smoothing_method: Literal["monotone_spline", "exponential"] = "monotone_spline"
    business_day_calendar: Optional[List[date]] = None
    enable_caching: bool = True
    cache_ttl_hours: int = 24

@dataclass
class VolatilityResult:
    """Volatility estimation result."""
    volatility_series: pd.Series
    method: VolatilityMethod
    config: VolatilityConfig
    calculation_date: date
    diagnostics: Dict[str, Union[float, bool, str]]
    metadata: Dict[str, Union[str, float, int]]

@dataclass
class TermStructureVolatility:
    """Volatility term structure."""
    tenors: np.ndarray
    volatilities: np.ndarray
    method: str
    smoothing_applied: bool
    calculation_date: date
    diagnostics: Dict[str, Union[float, bool, str]]

class VolatilityModels:
    """
    Volatility modeling utilities for interest rates and credit spreads.
    
    Performance target: ≤40ms for 200 factors per update
    """
    
    def __init__(self, config: Optional[VolatilityConfig] = None):
        self.config = config or VolatilityConfig()
        self._cache: Dict[str, VolatilityResult] = {}
        self._last_calculation: Optional[datetime] = None
        
    def calculate_volatility(
        self,
        returns_data: pd.Series,
        method: Optional[VolatilityMethod] = None,
        config: Optional[VolatilityConfig] = None
    ) -> VolatilityResult:
        """
        Calculate volatility using the specified method.
        
        Args:
            returns_data: Time series of returns
            method: Volatility estimation method
            config: Configuration overrides
            
        Returns:
            VolatilityResult with volatility series and diagnostics
        """
        start_time = datetime.now()
        
        method = method or self.config.method
        config = config or self.config
        
        # Validate inputs
        if returns_data.empty:
            raise ValueError("Returns data cannot be empty")
        
        # Remove outliers if configured
        if config.outlier_threshold > 0:
            returns_data = self._remove_outliers(returns_data, config.outlier_threshold)
        
        # Calculate volatility based on method
        if method == VolatilityMethod.HISTORICAL:
            vol_series, diagnostics = self._historical_volatility(returns_data, config)
        elif method == VolatilityMethod.EWMA:
            vol_series, diagnostics = self._ewma_volatility(returns_data, config)
        elif method == VolatilityMethod.REALIZED:
            vol_series, diagnostics = self._realized_volatility(returns_data, config)
        elif method == VolatilityMethod.GARCH:
            vol_series, diagnostics = self._garch_volatility(returns_data, config)
        else:
            raise ValueError(f"Unknown volatility method: {method}")
        
        # Apply bounds
        vol_series = np.clip(vol_series, config.min_volatility, config.max_volatility)
        
        # Create result
        result = VolatilityResult(
            volatility_series=pd.Series(vol_series, index=returns_data.index),
            method=method,
            config=config,
            calculation_date=datetime.now().date(),
            diagnostics=diagnostics,
            metadata={
                "data_points": len(returns_data),
                "method": method.value,
                "window_size": config.window_size,
                "ewma_lambda": config.ewma_lambda if method == VolatilityMethod.EWMA else None
            }
        )
        
        # Cache result
        if config.enable_caching:
            cache_key = self._generate_cache_key(returns_data.name, method, config)
            self._cache[cache_key] = result
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Volatility calculation completed in {execution_time:.2f}ms")
        
        if execution_time > 40:
            logger.warning(f"Volatility calculation exceeded 40ms target: {execution_time:.2f}ms")
        
        return result
    
    def calculate_term_structure(
        self,
        tenor_volatilities: Dict[float, float],
        method: Optional[str] = None,
        enable_smoothing: Optional[bool] = None
    ) -> TermStructureVolatility:
        """
        Calculate volatility term structure with optional smoothing.
        
        Args:
            tenor_volatilities: Dictionary mapping tenors to volatilities
            method: Smoothing method
            enable_smoothing: Whether to apply smoothing
            
        Returns:
            TermStructureVolatility with smoothed term structure
        """
        if not tenor_volatilities:
            raise ValueError("Tenor volatilities cannot be empty")
        
        tenors = np.array(sorted(tenor_volatilities.keys()))
        volatilities = np.array([tenor_volatilities[t] for t in tenors])
        
        # Apply smoothing if enabled
        smoothing_applied = False
        if enable_smoothing or (enable_smoothing is None and self.config.enable_smoothing):
            method = method or self.config.smoothing_method
            if method == "monotone_spline":
                volatilities, smoothing_applied = self._monotone_spline_smoothing(tenors, volatilities)
            elif method == "exponential":
                volatilities, smoothing_applied = self._exponential_smoothing(tenors, volatilities)
        
        # Calculate diagnostics
        diagnostics = self._calculate_term_structure_diagnostics(tenors, volatilities)
        
        return TermStructureVolatility(
            tenors=tenors,
            volatilities=volatilities,
            method=method or "none",
            smoothing_applied=smoothing_applied,
            calculation_date=datetime.now().date(),
            diagnostics=diagnostics
        )
    
    def _historical_volatility(
        self,
        returns_data: pd.Series,
        config: VolatilityConfig
    ) -> Tuple[np.ndarray, Dict[str, Union[float, bool, str]]]:
        """Calculate historical volatility using rolling window."""
        window_size = min(config.window_size, len(returns_data))
        
        # Calculate rolling standard deviation
        vol_series = returns_data.rolling(window=window_size, min_periods=window_size//2).std()
        
        # Annualize (assuming daily data)
        vol_series = vol_series * np.sqrt(252)
        
        # Fill initial values with expanding window
        expanding_vol = returns_data.expanding(min_periods=2).std() * np.sqrt(252)
        vol_series = vol_series.fillna(expanding_vol)
        
        # Calculate diagnostics
        diagnostics = {
            "mean_volatility": float(vol_series.mean()),
            "volatility_of_volatility": float(vol_series.std()),
            "min_volatility": float(vol_series.min()),
            "max_volatility": float(vol_series.max()),
            "window_size": window_size,
            "method": "historical"
        }
        
        return vol_series.values, diagnostics
    
    def _ewma_volatility(
        self,
        returns_data: pd.Series,
        config: VolatilityConfig
    ) -> Tuple[np.ndarray, Dict[str, Union[float, bool, str]]]:
        """Calculate EWMA volatility."""
        lambda_param = config.ewma_lambda
        
        # Initialize volatility series
        vol_series = np.zeros(len(returns_data))
        
        # Set initial volatility (sample variance of first 30 observations)
        initial_window = min(30, len(returns_data))
        initial_vol = returns_data.iloc[:initial_window].std() * np.sqrt(252)
        vol_series[0] = initial_vol
        
        # Apply EWMA recursion
        for i in range(1, len(returns_data)):
            vol_series[i] = np.sqrt(
                lambda_param * vol_series[i-1]**2 + (1 - lambda_param) * returns_data.iloc[i-1]**2
            ) * np.sqrt(252)
        
        # Calculate diagnostics
        diagnostics = {
            "mean_volatility": float(np.mean(vol_series)),
            "volatility_of_volatility": float(np.std(vol_series)),
            "min_volatility": float(np.min(vol_series)),
            "max_volatility": float(np.max(vol_series)),
            "ewma_lambda": lambda_param,
            "method": "ewma"
        }
        
        return vol_series, diagnostics
    
    def _realized_volatility(
        self,
        returns_data: pd.Series,
        config: VolatilityConfig
    ) -> Tuple[np.ndarray, Dict[str, Union[float, bool, str]]]:
        """Calculate realized volatility using high-frequency aggregation."""
        window_size = min(config.window_size, len(returns_data))
        
        # Calculate rolling sum of squared returns
        squared_returns = returns_data ** 2
        rolling_sum = squared_returns.rolling(window=window_size, min_periods=window_size//2).sum()
        
        # Convert to volatility
        vol_series = np.sqrt(rolling_sum) * np.sqrt(252)
        
        # Fill initial values
        expanding_sum = squared_returns.expanding(min_periods=2).sum()
        expanding_vol = np.sqrt(expanding_sum) * np.sqrt(252)
        vol_series = vol_series.fillna(expanding_vol)
        
        # Calculate diagnostics
        diagnostics = {
            "mean_volatility": float(vol_series.mean()),
            "volatility_of_volatility": float(vol_series.std()),
            "min_volatility": float(vol_series.min()),
            "max_volatility": float(vol_series.max()),
            "window_size": window_size,
            "method": "realized"
        }
        
        return vol_series.values, diagnostics
    
    def _garch_volatility(
        self,
        returns_data: pd.Series,
        config: VolatilityConfig
    ) -> Tuple[np.ndarray, Dict[str, Union[float, bool, str]]]:
        """
        Calculate GARCH(1,1) volatility.
        
        Note: This is a simplified implementation. In production, use specialized
        GARCH libraries like arch or statsmodels.
        """
        try:
            # Simple GARCH(1,1) estimation
            vol_series, garch_params = self._fit_garch_11(returns_data)
            
            diagnostics = {
                "mean_volatility": float(np.mean(vol_series)),
                "volatility_of_volatility": float(np.std(vol_series)),
                "min_volatility": float(np.min(vol_series)),
                "max_volatility": float(np.max(vol_series)),
                "garch_alpha": float(garch_params.get("alpha", 0)),
                "garch_beta": float(garch_params.get("beta", 0)),
                "garch_omega": float(garch_params.get("omega", 0)),
                "method": "garch"
            }
            
            return vol_series, diagnostics
            
        except Exception as e:
            logger.warning(f"GARCH estimation failed, falling back to EWMA: {e}")
            # Fallback to EWMA
            return self._ewma_volatility(returns_data, config)
    
    def _fit_garch_11(self, returns_data: pd.Series) -> Tuple[np.ndarray, Dict[str, float]]:
        """Fit GARCH(1,1) model using maximum likelihood."""
        # Initialize parameters
        omega = np.var(returns_data) * 0.1
        alpha = 0.1
        beta = 0.8
        
        # Simple optimization (in production, use proper MLE)
        def objective(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            
            # Calculate conditional variances
            h = np.zeros(len(returns_data))
            h[0] = omega / (1 - alpha - beta)
            
            for i in range(1, len(returns_data)):
                h[i] = omega + alpha * returns_data.iloc[i-1]**2 + beta * h[i-1]
            
            # Log-likelihood (simplified)
            ll = -0.5 * np.sum(np.log(h) + returns_data**2 / h)
            return -ll
        
        # Optimize
        result = minimize(
            objective,
            [omega, alpha, beta],
            method='L-BFGS-B',
            bounds=[(1e-6, None), (0, 1), (0, 1)]
        )
        
        if result.success:
            omega, alpha, beta = result.x
        else:
            # Use default values if optimization fails
            omega, alpha, beta = np.var(returns_data) * 0.1, 0.1, 0.8
        
        # Calculate conditional volatilities
        h = np.zeros(len(returns_data))
        h[0] = omega / (1 - alpha - beta)
        
        for i in range(1, len(returns_data)):
            h[i] = omega + alpha * returns_data.iloc[i-1]**2 + beta * h[i-1]
        
        vol_series = np.sqrt(h) * np.sqrt(252)
        
        return vol_series, {"omega": omega, "alpha": alpha, "beta": beta}
    
    def _monotone_spline_smoothing(
        self,
        tenors: np.ndarray,
        volatilities: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """Apply monotone spline smoothing to term structure."""
        try:
            # Use PchipInterpolator for monotone interpolation
            spline = interpolate.PchipInterpolator(tenors, volatilities)
            smoothed_vols = spline(tenors)
            return smoothed_vols, True
        except Exception as e:
            logger.warning(f"Monotone spline smoothing failed: {e}")
            return volatilities, False
    
    def _exponential_smoothing(
        self,
        tenors: np.ndarray,
        volatilities: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """Apply exponential smoothing to term structure."""
        try:
            # Simple exponential smoothing
            alpha = 0.3
            smoothed_vols = np.zeros_like(volatilities)
            smoothed_vols[0] = volatilities[0]
            
            for i in range(1, len(volatilities)):
                smoothed_vols[i] = alpha * volatilities[i] + (1 - alpha) * smoothed_vols[i-1]
            
            return smoothed_vols, True
        except Exception as e:
            logger.warning(f"Exponential smoothing failed: {e}")
            return volatilities, False
    
    def _remove_outliers(
        self,
        data: pd.Series,
        threshold: float
    ) -> pd.Series:
        """Remove outliers using standard deviation threshold."""
        mean = data.mean()
        std = data.std()
        
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        
        # Replace outliers with bounds
        cleaned_data = data.copy()
        cleaned_data[cleaned_data < lower_bound] = lower_bound
        cleaned_data[cleaned_data > upper_bound] = upper_bound
        
        return cleaned_data
    
    def _calculate_term_structure_diagnostics(
        self,
        tenors: np.ndarray,
        volatilities: np.ndarray
    ) -> Dict[str, Union[float, bool, str]]:
        """Calculate diagnostics for term structure."""
        # Check monotonicity
        is_monotonic = np.all(np.diff(volatilities) >= 0) or np.all(np.diff(volatilities) <= 0)
        
        # Calculate slope
        if len(tenors) > 1:
            slope = np.polyfit(tenors, volatilities, 1)[0]
        else:
            slope = 0.0
        
        # Calculate curvature
        if len(tenors) > 2:
            curvature = np.polyfit(tenors, volatilities, 2)[0]
        else:
            curvature = 0.0
        
        return {
            "is_monotonic": bool(is_monotonic),
            "slope": float(slope),
            "curvature": float(curvature),
            "mean_volatility": float(np.mean(volatilities)),
            "volatility_range": float(np.max(volatilities) - np.min(volatilities)),
            "tenor_range": float(np.max(tenors) - np.min(tenors))
        }
    
    def _generate_cache_key(
        self,
        factor_name: str,
        method: VolatilityMethod,
        config: VolatilityConfig
    ) -> str:
        """Generate cache key for volatility results."""
        return f"{factor_name}_{method.value}_{config.window_size}_{config.ewma_lambda}"
    
    def get_volatility_by_date(
        self,
        factor_name: str,
        target_date: date,
        method: Optional[VolatilityMethod] = None
    ) -> Optional[VolatilityResult]:
        """Retrieve volatility for a specific date."""
        # Implementation would query cached results
        # For now, return None if not found
        return None
    
    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()
        logger.info("Volatility models cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Union[int, str]]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "last_calculation": self._last_calculation.isoformat() if self._last_calculation else None
        }
