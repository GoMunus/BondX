"""
Duration and convexity calculations for BondX Backend.

This module implements comprehensive duration and convexity calculations
for bond risk analysis and portfolio management.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Dict, Any, Optional, Union

from ..core.logging import get_logger
from ..database.models import DayCountConvention
from .day_count import DayCountCalculator

logger = get_logger(__name__)


class DurationCalculator:
    """
    Advanced duration and convexity calculator.
    
    Implements multiple duration measures:
    - Macaulay duration
    - Modified duration
    - Effective duration (for bonds with embedded options)
    - Key rate duration
    """
    
    def __init__(self):
        """Initialize the duration calculator."""
        self.logger = logger
        self.day_count_calc = DayCountCalculator()
    
    def calculate_duration_metrics(
        self,
        face_value: Decimal,
        coupon_rate: Decimal,
        coupon_frequency: int,
        maturity_date: Union[date, datetime],
        settlement_date: Union[date, datetime],
        yield_rate: Decimal,
        day_count_convention: DayCountConvention = DayCountConvention.THIRTY_360,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive duration and convexity metrics.
        
        Args:
            face_value: Face value of the bond
            coupon_rate: Annual coupon rate (as decimal)
            coupon_frequency: Number of coupon payments per year
            maturity_date: Bond maturity date
            settlement_date: Settlement date
            yield_rate: Current yield rate
            day_count_convention: Day count convention to use
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing duration and convexity metrics
        """
        try:
            # Calculate time to maturity
            time_to_maturity = self._calculate_time_to_maturity(
                settlement_date, maturity_date, day_count_convention
            )
            
            # Calculate Macaulay duration
            macaulay_duration = self._calculate_macaulay_duration(
                face_value, coupon_rate, coupon_frequency, time_to_maturity, yield_rate
            )
            
            # Calculate modified duration
            modified_duration = self._calculate_modified_duration(macaulay_duration, yield_rate)
            
            # Calculate convexity
            convexity = self._calculate_convexity(
                face_value, coupon_rate, coupon_frequency, time_to_maturity, yield_rate
            )
            
            # Calculate price value of basis point (PVBP)
            pvbp = self._calculate_price_value_of_basis_point(
                face_value, modified_duration
            )
            
            return {
                "macaulay_duration": macaulay_duration,
                "modified_duration": modified_duration,
                "convexity": convexity,
                "price_value_of_basis_point": pvbp,
                "time_to_maturity": time_to_maturity,
                "yield_rate": yield_rate,
                "calculation_date": datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.error(
                "Duration calculation failed",
                face_value=str(face_value),
                coupon_rate=str(coupon_rate),
                yield_rate=str(yield_rate),
                error=str(e)
            )
            raise
    
    def _calculate_time_to_maturity(
        self,
        settlement_date: Union[date, datetime],
        maturity_date: Union[date, datetime],
        day_count_convention: DayCountConvention
    ) -> float:
        """Calculate time to maturity in years."""
        days_between, days_in_year = self.day_count_calc.calculate_days(
            settlement_date, maturity_date, day_count_convention
        )
        return days_between / days_in_year
    
    def _calculate_macaulay_duration(
        self,
        face_value: Decimal,
        coupon_rate: Decimal,
        coupon_frequency: int,
        time_to_maturity: float,
        yield_rate: Decimal
    ) -> Decimal:
        """Calculate Macaulay duration."""
        if time_to_maturity <= 0:
            return Decimal("0")
        
        # Calculate coupon payment
        coupon_payment = (coupon_rate * face_value) / coupon_frequency
        
        # Calculate weighted average time to cash flows
        total_weighted_time = Decimal("0")
        total_present_value = Decimal("0")
        
        # Coupon payments
        for i in range(1, int(time_to_maturity * coupon_frequency) + 1):
            payment_time = i / coupon_frequency
            if payment_time <= time_to_maturity:
                discount_factor = Decimal(str(1 / (1 + yield_rate) ** payment_time))
                present_value = coupon_payment * discount_factor
                total_weighted_time += present_value * Decimal(str(payment_time))
                total_present_value += present_value
        
        # Face value payment
        face_value_discount = Decimal(str(1 / (1 + yield_rate) ** time_to_maturity))
        face_value_pv = face_value * face_value_discount
        total_weighted_time += face_value_pv * Decimal(str(time_to_maturity))
        total_present_value += face_value_pv
        
        if total_present_value > 0:
            macaulay_duration = total_weighted_time / total_present_value
        else:
            macaulay_duration = Decimal("0")
        
        return macaulay_duration
    
    def _calculate_modified_duration(
        self,
        macaulay_duration: Decimal,
        yield_rate: Decimal
    ) -> Decimal:
        """Calculate modified duration from Macaulay duration."""
        if yield_rate == 0:
            return Decimal("0")
        
        return macaulay_duration / (1 + yield_rate)
    
    def _calculate_convexity(
        self,
        face_value: Decimal,
        coupon_rate: Decimal,
        coupon_frequency: int,
        time_to_maturity: float,
        yield_rate: Decimal
    ) -> Decimal:
        """Calculate convexity."""
        if time_to_maturity <= 0:
            return Decimal("0")
        
        # Calculate coupon payment
        coupon_payment = (coupon_rate * face_value) / coupon_frequency
        
        # Calculate weighted average of squared time to cash flows
        total_weighted_squared_time = Decimal("0")
        total_present_value = Decimal("0")
        
        # Coupon payments
        for i in range(1, int(time_to_maturity * coupon_frequency) + 1):
            payment_time = i / coupon_frequency
            if payment_time <= time_to_maturity:
                discount_factor = Decimal(str(1 / (1 + yield_rate) ** payment_time))
                present_value = coupon_payment * discount_factor
                squared_time = Decimal(str(payment_time * payment_time))
                total_weighted_squared_time += present_value * squared_time
                total_present_value += present_value
        
        # Face value payment
        face_value_discount = Decimal(str(1 / (1 + yield_rate) ** time_to_maturity))
        face_value_pv = face_value * face_value_discount
        squared_maturity = Decimal(str(time_to_maturity * time_to_maturity))
        total_weighted_squared_time += face_value_pv * squared_maturity
        total_present_value += face_value_pv
        
        if total_present_value > 0:
            convexity = total_weighted_squared_time / total_present_value
        else:
            convexity = Decimal("0")
        
        return convexity
    
    def _calculate_price_value_of_basis_point(
        self,
        face_value: Decimal,
        modified_duration: Decimal
    ) -> Decimal:
        """Calculate price value of basis point (PVBP)."""
        # PVBP = -Modified Duration * Face Value * 0.0001
        return -modified_duration * face_value * Decimal("0.0001")
    
    def calculate_effective_duration(
        self,
        price_up: Decimal,
        price_down: Decimal,
        price_current: Decimal,
        yield_change: Decimal
    ) -> Decimal:
        """
        Calculate effective duration for bonds with embedded options.
        
        Args:
            price_up: Bond price when yield increases
            price_down: Bond price when yield decreases
            price_current: Current bond price
            yield_change: Yield change used in calculation
            
        Returns:
            Effective duration
        """
        if price_current <= 0 or yield_change <= 0:
            return Decimal("0")
        
        # Effective duration = -(Price_up - Price_down) / (2 * Price_current * Yield_change)
        effective_duration = -(price_up - price_down) / (2 * price_current * yield_change)
        return effective_duration
    
    def calculate_key_rate_duration(
        self,
        key_rates: list,
        price_changes: list,
        price_current: Decimal,
        yield_changes: list
    ) -> Dict[str, Decimal]:
        """
        Calculate key rate durations for yield curve risk analysis.
        
        Args:
            key_rates: List of key rate tenors
            price_changes: List of price changes for each key rate
            price_current: Current bond price
            yield_changes: List of yield changes for each key rate
            
        Returns:
            Dictionary mapping key rates to durations
        """
        if len(key_rates) != len(price_changes) or len(key_rates) != len(yield_changes):
            raise ValueError("All input lists must have the same length")
        
        if price_current <= 0:
            return {str(rate): Decimal("0") for rate in key_rates}
        
        key_rate_durations = {}
        
        for i, key_rate in enumerate(key_rates):
            if yield_changes[i] > 0:
                duration = -price_changes[i] / (price_current * yield_changes[i])
            else:
                duration = Decimal("0")
            
            key_rate_durations[str(key_rate)] = duration
        
        return key_rate_durations


# Export the calculator
__all__ = ["DurationCalculator"]
