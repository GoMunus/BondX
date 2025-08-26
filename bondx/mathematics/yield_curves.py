"""
Yield curve engine for BondX Backend.

This module implements yield curve construction and analysis including
par curves, zero curves, discount curves, and forward curves with
various interpolation and extrapolation methods.
"""

import json
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize_scalar

from ..core.logging import get_logger
from ..database.models import DayCountConvention
from .day_count import DayCountCalculator

logger = get_logger(__name__)


class CurveType(Enum):
    """Types of yield curves."""
    PAR_CURVE = "PAR_CURVE"  # Par yield curve
    ZERO_CURVE = "ZERO_CURVE"  # Zero (spot) rate curve
    DISCOUNT_CURVE = "DISCOUNT_CURVE"  # Discount factor curve
    FORWARD_CURVE = "FORWARD_CURVE"  # Forward rate curve


class InterpolationMethod(Enum):
    """Interpolation methods for yield curves."""
    LINEAR_ON_YIELD = "LINEAR_ON_YIELD"  # Linear interpolation on yields
    LINEAR_ON_ZERO = "LINEAR_ON_ZERO"  # Linear interpolation on zero rates
    CUBIC_SPLINE = "CUBIC_SPLINE"  # Cubic spline interpolation
    MONOTONE_HERMITE = "MONOTONE_HERMITE"  # Monotone Hermite spline


class ExtrapolationMethod(Enum):
    """Extrapolation methods for yield curves."""
    FLAT_FORWARD = "FLAT_FORWARD"  # Flat forward rates
    LINEAR = "LINEAR"  # Linear extrapolation
    FLAT_YIELD = "FLAT_YIELD"  # Flat yields


class CompoundingConvention(Enum):
    """Compounding conventions."""
    ANNUAL = "ANNUAL"
    SEMI_ANNUAL = "SEMI_ANNUAL"
    CONTINUOUS = "CONTINUOUS"


@dataclass
class MarketQuote:
    """Market quote for yield curve construction."""
    tenor: Union[float, date]  # Tenor in years or specific date
    quote_type: CurveType
    quote_value: Decimal
    day_count: DayCountConvention
    instrument_id: Optional[str] = None
    quote_date: Optional[date] = None
    currency: str = "INR"


@dataclass
class CurveConstructionConfig:
    """Configuration for yield curve construction."""
    interpolation_method: InterpolationMethod = InterpolationMethod.LINEAR_ON_ZERO
    extrapolation_method: ExtrapolationMethod = ExtrapolationMethod.FLAT_FORWARD
    compounding: CompoundingConvention = CompoundingConvention.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_365
    bootstrapping_tolerance: float = 1e-6
    max_iterations: int = 100
    smoothing_penalty: Optional[float] = None  # For cubic spline
    currency: str = "INR"


@dataclass
class YieldCurve:
    """Yield curve object with various evaluation methods."""
    curve_type: CurveType
    tenors: np.ndarray
    rates: np.ndarray
    construction_date: date
    config: CurveConstructionConfig
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        """Validate curve data."""
        if len(self.tenors) != len(self.rates):
            raise ValueError("Tenors and rates must have the same length")
        if len(self.tenors) == 0:
            raise ValueError("Curve must have at least one point")
        
        # Ensure tenors are sorted
        sort_idx = np.argsort(self.tenors)
        self.tenors = self.tenors[sort_idx]
        self.rates = self.rates[sort_idx]
    
    def zero_rate(self, t: Union[float, date]) -> float:
        """Get zero rate at time t."""
        if isinstance(t, date):
            t = self._date_to_tenor(t)
        
        if t <= 0:
            return float(self.rates[0]) if len(self.rates) > 0 else 0.0
        
        return self._interpolate_rate(t)
    
    def discount_factor(self, t: Union[float, date]) -> float:
        """Get discount factor at time t."""
        if isinstance(t, date):
            t = self._date_to_tenor(t)
        
        if t <= 0:
            return 1.0
        
        zero_rate = self.zero_rate(t)
        
        if self.config.compounding == CompoundingConvention.CONTINUOUS:
            return np.exp(-zero_rate * t)
        elif self.config.compounding == CompoundingConvention.SEMI_ANNUAL:
            return 1 / (1 + zero_rate / 2) ** (2 * t)
        else:  # Annual
            return 1 / (1 + zero_rate) ** t
    
    def forward_rate(self, t1: Union[float, date], t2: Union[float, date]) -> float:
        """Get forward rate from t1 to t2."""
        if isinstance(t1, date):
            t1 = self._date_to_tenor(t1)
        if isinstance(t2, date):
            t2 = self._date_to_tenor(t2)
        
        if t1 >= t2:
            raise ValueError("t1 must be less than t2")
        
        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)
        
        if self.config.compounding == CompoundingConvention.CONTINUOUS:
            return -np.log(df2 / df1) / (t2 - t1)
        elif self.config.compounding == CompoundingConvention.SEMI_ANNUAL:
            return 2 * ((df1 / df2) ** (1 / (2 * (t2 - t1))) - 1)
        else:  # Annual
            return (df1 / df2) ** (1 / (t2 - t1)) - 1
    
    def par_yield(self, t: Union[float, date]) -> float:
        """Get par yield at time t."""
        if isinstance(t, date):
            t = self._date_to_tenor(t)
        
        if t <= 0:
            return 0.0
        
        # For par yield, we need to solve: 1 = coupon * annuity + discount_factor
        # This is a simplified calculation - in practice, this would be more complex
        zero_rate = self.zero_rate(t)
        return zero_rate  # Simplified approximation
    
    def df(self, date: date) -> float:
        """Get discount factor for a specific date."""
        return self.discount_factor(date)
    
    def shift(self, shift_type: str, amount: float) -> 'YieldCurve':
        """Apply parallel, slope, or curvature shift to the curve."""
        if shift_type == "parallel":
            new_rates = self.rates + amount
        elif shift_type == "slope":
            # Apply slope shift (increase with tenor)
            new_rates = self.rates + amount * self.tenors
        elif shift_type == "curvature":
            # Apply curvature shift (quadratic in tenor)
            new_rates = self.rates + amount * self.tenors ** 2
        else:
            raise ValueError(f"Unknown shift type: {shift_type}")
        
        return YieldCurve(
            curve_type=self.curve_type,
            tenors=self.tenors.copy(),
            rates=new_rates,
            construction_date=self.construction_date,
            config=self.config,
            metadata=self.metadata
        )
    
    def roll(self, roll_date: date) -> 'YieldCurve':
        """Roll the curve to a new date."""
        # This is a simplified roll - in practice, this would involve
        # more complex logic for handling different market conventions
        days_diff = (roll_date - self.construction_date).days / 365.0
        
        if days_diff > 0:
            # Roll forward: shift tenors left
            new_tenors = np.maximum(0, self.tenors - days_diff)
            valid_idx = new_tenors >= 0
            new_tenors = new_tenors[valid_idx]
            new_rates = self.rates[valid_idx]
        else:
            # Roll backward: shift tenors right
            new_tenors = self.tenors - days_diff
            new_rates = self.rates
        
        return YieldCurve(
            curve_type=self.curve_type,
            tenors=new_tenors,
            rates=new_rates,
            construction_date=roll_date,
            config=self.config,
            metadata=self.metadata
        )
    
    def _interpolate_rate(self, t: float) -> float:
        """Interpolate rate at time t."""
        if t <= self.tenors[0]:
            return float(self.rates[0])
        elif t >= self.tenors[-1]:
            return float(self.rates[-1])
        
        # Find interpolation interval
        idx = np.searchsorted(self.tenors, t)
        if idx == 0:
            return float(self.rates[0])
        elif idx == len(self.tenors):
            return float(self.rates[-1])
        
        # Linear interpolation
        t1, t2 = self.tenors[idx-1], self.tenors[idx]
        r1, r2 = self.rates[idx-1], self.rates[idx]
        
        return float(r1 + (r2 - r1) * (t - t1) / (t2 - t1))
    
    def _date_to_tenor(self, date: date) -> float:
        """Convert date to tenor relative to construction date."""
        return (date - self.construction_date).days / 365.0
    
    def to_dict(self) -> Dict:
        """Convert curve to dictionary for serialization."""
        return {
            "curve_type": self.curve_type.value,
            "tenors": self.tenors.tolist(),
            "rates": [float(r) for r in self.rates],
            "construction_date": self.construction_date.isoformat(),
            "config": {
                "interpolation_method": self.config.interpolation_method.value,
                "extrapolation_method": self.config.extrapolation_method.value,
                "compounding": self.config.compounding.value,
                "day_count": self.config.day_count.value,
                "currency": self.config.currency
            },
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'YieldCurve':
        """Create curve from dictionary."""
        config = CurveConstructionConfig(
            interpolation_method=InterpolationMethod(data["config"]["interpolation_method"]),
            extrapolation_method=ExtrapolationMethod(data["config"]["extrapolation_method"]),
            compounding=CompoundingConvention(data["config"]["compounding"]),
            day_count=DayCountConvention(data["config"]["day_count"]),
            currency=data["config"]["currency"]
        )
        
        return cls(
            curve_type=CurveType(data["curve_type"]),
            tenors=np.array(data["tenors"]),
            rates=np.array(data["rates"]),
            construction_date=datetime.fromisoformat(data["construction_date"]).date(),
            config=config,
            metadata=data.get("metadata")
        )


class YieldCurveEngine:
    """
    Production-grade yield curve engine for bond pricing and risk management.
    
    Supports construction of various curve types with multiple interpolation
    methods and bootstrapping from market instruments.
    """
    
    def __init__(self):
        """Initialize the yield curve engine."""
        self.logger = logger
        self.day_count_calculator = DayCountCalculator()
        self._curve_cache: Dict[str, YieldCurve] = {}
    
    def construct_curve(
        self,
        quotes: List[MarketQuote],
        config: CurveConstructionConfig,
        curve_id: Optional[str] = None
    ) -> YieldCurve:
        """
        Construct a yield curve from market quotes.
        
        Args:
            quotes: List of market quotes
            config: Construction configuration
            curve_id: Optional curve identifier for caching
            
        Returns:
            Constructed yield curve
            
        Raises:
            ValueError: If quotes are invalid or insufficient
        """
        if not quotes:
            raise ValueError("At least one market quote is required")
        
        # Validate and sort quotes
        validated_quotes = self._validate_quotes(quotes)
        sorted_quotes = sorted(validated_quotes, key=lambda q: self._get_tenor_value(q.tenor))
        
        # Convert tenors to years if they are dates
        tenors = []
        for quote in sorted_quotes:
            if isinstance(quote.tenor, date):
                # Use a reference date (e.g., today) for date-based tenors
                reference_date = date.today()
                tenor_years = (quote.tenor - reference_date).days / 365.0
                tenors.append(tenor_years)
            else:
                tenors.append(float(quote.tenor))
        
        tenors = np.array(tenors)
        
        # Construct curve based on quote types
        if all(q.quote_type == CurveType.PAR_CURVE for q in sorted_quotes):
            rates = self._bootstrap_from_par_yields(sorted_quotes, tenors, config)
            curve_type = CurveType.ZERO_CURVE
        elif all(q.quote_type == CurveType.ZERO_CURVE for q in sorted_quotes):
            rates = np.array([float(q.quote_value) for q in sorted_quotes])
            curve_type = CurveType.ZERO_CURVE
        elif all(q.quote_type == CurveType.DISCOUNT_CURVE for q in sorted_quotes):
            # Convert discount factors to zero rates
            dfs = np.array([float(q.quote_value) for q in sorted_quotes])
            rates = self._discount_factors_to_zero_rates(dfs, tenors, config)
            curve_type = CurveType.ZERO_CURVE
        else:
            raise ValueError("Mixed quote types not supported for curve construction")
        
        # Create curve object
        curve = YieldCurve(
            curve_type=curve_type,
            tenors=tenors,
            rates=rates,
            construction_date=date.today(),
            config=config,
            metadata={
                "source_quotes": [q.instrument_id for q in sorted_quotes if q.instrument_id],
                "construction_method": "bootstrapping"
            }
        )
        
        # Cache curve if ID provided
        if curve_id:
            self._curve_cache[curve_id] = curve
        
        return curve
    
    def _validate_quotes(self, quotes: List[MarketQuote]) -> List[MarketQuote]:
        """Validate market quotes."""
        validated = []
        
        for quote in quotes:
            if quote.quote_value <= 0:
                self.logger.warning(f"Skipping quote with non-positive value: {quote.quote_value}")
                continue
            
            if isinstance(quote.tenor, (int, float)) and quote.tenor < 0:
                self.logger.warning(f"Skipping quote with negative tenor: {quote.tenor}")
                continue
            
            validated.append(quote)
        
        if len(validated) < 2:
            raise ValueError("At least two valid quotes are required for curve construction")
        
        return validated
    
    def _get_tenor_value(self, tenor: Union[float, date]) -> float:
        """Get numeric value of tenor for sorting."""
        if isinstance(tenor, date):
            return (tenor - date.today()).days / 365.0
        return float(tenor)
    
    def _bootstrap_from_par_yields(
        self,
        quotes: List[MarketQuote],
        tenors: np.ndarray,
        config: CurveConstructionConfig
    ) -> np.ndarray:
        """Bootstrap zero rates from par yields."""
        zero_rates = []
        
        for i, quote in enumerate(quotes):
            tenor = tenors[i]
            par_yield = float(quote.quote_value)
            
            if i == 0:
                # First point: assume zero rate equals par yield
                zero_rates.append(par_yield)
            else:
                # Bootstrap using previous zero rates
                zero_rate = self._solve_zero_rate_from_par(
                    par_yield, tenor, zero_rates, tenors[:i], config
                )
                zero_rates.append(zero_rate)
        
        return np.array(zero_rates)
    
    def _solve_zero_rate_from_par(
        self,
        par_yield: float,
        tenor: float,
        previous_zero_rates: List[float],
        previous_tenors: np.ndarray,
        config: CurveConstructionConfig
    ) -> float:
        """Solve for zero rate given par yield and previous zero rates."""
        def objective(zero_rate):
            # Calculate present value of par bond
            pv = 0.0
            for i, prev_tenor in enumerate(previous_tenors):
                # Coupon payment
                coupon = par_yield / config.compounding.value
                df = self._calculate_discount_factor(prev_tenor, previous_zero_rates[i], config)
                pv += coupon * df
            
            # Final coupon and principal
            final_df = self._calculate_discount_factor(tenor, zero_rate, config)
            pv += (1 + par_yield / config.compounding.value) * final_df
            
            # Should equal par value (1.0)
            return abs(pv - 1.0)
        
        # Use optimization to find zero rate
        result = minimize_scalar(
            objective,
            bounds=(0, 1),  # Reasonable bounds for zero rates
            method='bounded'
        )
        
        if result.success:
            return result.x
        else:
            # Fallback: use par yield as approximation
            self.logger.warning(f"Failed to bootstrap zero rate for tenor {tenor}, using par yield")
            return par_yield
    
    def _calculate_discount_factor(self, tenor: float, zero_rate: float, config: CurveConstructionConfig) -> float:
        """Calculate discount factor for given tenor and zero rate."""
        if config.compounding == CompoundingConvention.CONTINUOUS:
            return np.exp(-zero_rate * tenor)
        elif config.compounding == CompoundingConvention.SEMI_ANNUAL:
            return 1 / (1 + zero_rate / 2) ** (2 * tenor)
        else:  # Annual
            return 1 / (1 + zero_rate) ** tenor
    
    def _discount_factors_to_zero_rates(
        self,
        dfs: np.ndarray,
        tenors: np.ndarray,
        config: CurveConstructionConfig
    ) -> np.ndarray:
        """Convert discount factors to zero rates."""
        zero_rates = []
        
        for df, tenor in zip(dfs, tenors):
            if df <= 0 or tenor <= 0:
                zero_rates.append(0.0)
                continue
            
            if config.compounding == CompoundingConvention.CONTINUOUS:
                zero_rate = -np.log(df) / tenor
            elif config.compounding == CompoundingConvention.SEMI_ANNUAL:
                zero_rate = 2 * (df ** (-1 / (2 * tenor)) - 1)
            else:  # Annual
                zero_rate = df ** (-1 / tenor) - 1
            
            zero_rates.append(zero_rate)
        
        return np.array(zero_rates)
    
    def interpolate_curve(
        self,
        curve: YieldCurve,
        new_tenors: np.ndarray,
        method: Optional[InterpolationMethod] = None
    ) -> YieldCurve:
        """Interpolate curve to new tenors."""
        if method is None:
            method = curve.config.interpolation_method
        
        if method == InterpolationMethod.LINEAR_ON_ZERO:
            interpolated_rates = np.interp(new_tenors, curve.tenors, curve.rates)
        elif method == InterpolationMethod.CUBIC_SPLINE:
            # Use scipy cubic spline
            cs = CubicSpline(curve.tenors, curve.rates, bc_type='natural')
            interpolated_rates = cs(new_tenors)
        else:
            # Default to linear interpolation
            interpolated_rates = np.interp(new_tenors, curve.tenors, curve.rates)
        
        return YieldCurve(
            curve_type=curve.curve_type,
            tenors=new_tenors,
            rates=interpolated_rates,
            construction_date=curve.construction_date,
            config=curve.config,
            metadata=curve.metadata
        )
    
    def detect_arbitrage(self, curve: YieldCurve) -> List[str]:
        """Detect potential arbitrage opportunities in the curve."""
        warnings = []
        
        # Check for negative discount factors
        for i, (tenor, rate) in enumerate(zip(curve.tenors, curve.rates)):
            df = curve.discount_factor(tenor)
            if df <= 0:
                warnings.append(f"Negative discount factor at tenor {tenor}: {df}")
        
        # Check for negative forward rates
        for i in range(len(curve.tenors) - 1):
            t1, t2 = curve.tenors[i], curve.tenors[i + 1]
            forward_rate = curve.forward_rate(t1, t2)
            if forward_rate < 0:
                warnings.append(f"Negative forward rate from {t1} to {t2}: {forward_rate}")
        
        # Check for non-monotonic discount factors
        dfs = [curve.discount_factor(t) for t in curve.tenors]
        if not all(dfs[i] >= dfs[i+1] for i in range(len(dfs)-1)):
            warnings.append("Non-monotonic discount factors detected")
        
        return warnings
    
    def get_cached_curve(self, curve_id: str) -> Optional[YieldCurve]:
        """Get cached curve by ID."""
        return self._curve_cache.get(curve_id)
    
    def clear_cache(self):
        """Clear curve cache."""
        self._curve_cache.clear()
    
    def export_curve(self, curve: YieldCurve, filepath: str):
        """Export curve to file."""
        curve_data = curve.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(curve_data, f, indent=2)
        
        self.logger.info(f"Curve exported to {filepath}")
    
    def import_curve(self, filepath: str) -> YieldCurve:
        """Import curve from file."""
        with open(filepath, 'r') as f:
            curve_data = json.load(f)
        
        curve = YieldCurve.from_dict(curve_data)
        self.logger.info(f"Curve imported from {filepath}")
        
        return curve


__all__ = [
    "YieldCurveEngine",
    "YieldCurve",
    "MarketQuote",
    "CurveConstructionConfig",
    "CurveType",
    "InterpolationMethod",
    "ExtrapolationMethod",
    "CompoundingConvention"
]
