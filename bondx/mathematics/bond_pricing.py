"""
Bond pricing engine for BondX Backend.

This module implements comprehensive bond pricing calculations including:
- Clean and dirty pricing
- Accrued interest calculations
- Price sensitivity to yield changes
- Support for various bond types and structures
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Dict, Any, Optional, Union

from ..core.logging import get_logger
from ..database.models import DayCountConvention, CouponType
from .day_count import DayCountCalculator

logger = get_logger(__name__)


class BondPricingEngine:
    """
    Comprehensive bond pricing engine.
    
    Handles pricing calculations for various bond types including:
    - Fixed rate bonds
    - Floating rate bonds
    - Zero coupon bonds
    - Inflation-indexed bonds
    - Bonds with embedded options
    """
    
    def __init__(self):
        """Initialize the bond pricing engine."""
        self.logger = logger
        self.day_count_calc = DayCountCalculator()
    
    def calculate_bond_price(
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
        Calculate comprehensive bond pricing.
        
        Args:
            face_value: Face value of the bond
            coupon_rate: Annual coupon rate (as decimal)
            coupon_frequency: Number of coupon payments per year
            maturity_date: Bond maturity date
            settlement_date: Settlement date
            yield_rate: Yield rate for pricing
            day_count_convention: Day count convention to use
            **kwargs: Additional parameters for specific bond types
            
        Returns:
            Dictionary containing pricing information
        """
        try:
            # Calculate time to maturity
            time_to_maturity = self._calculate_time_to_maturity(
                settlement_date, maturity_date, day_count_convention
            )
            
            # Calculate coupon payment
            coupon_payment = (coupon_rate * face_value) / coupon_frequency
            
            # Calculate present value of coupons
            coupon_pv = self._calculate_coupon_present_value(
                coupon_payment, coupon_frequency, time_to_maturity, yield_rate
            )
            
            # Calculate present value of face value
            face_value_pv = self._calculate_face_value_present_value(
                face_value, time_to_maturity, yield_rate
            )
            
            # Calculate clean price
            clean_price = coupon_pv + face_value_pv
            
            # Calculate accrued interest
            accrued_interest = self._calculate_accrued_interest(
                face_value, coupon_rate, coupon_frequency,
                settlement_date, maturity_date, day_count_convention
            )
            
            # Calculate dirty price
            dirty_price = clean_price + accrued_interest
            
            # Calculate price sensitivity metrics
            modified_duration = self._calculate_modified_duration(
                clean_price, face_value, coupon_rate, coupon_frequency,
                time_to_maturity, yield_rate
            )
            
            convexity = self._calculate_convexity(
                clean_price, face_value, coupon_rate, coupon_frequency,
                time_to_maturity, yield_rate
            )
            
            return {
                "clean_price": clean_price,
                "dirty_price": dirty_price,
                "accrued_interest": accrued_interest,
                "yield_rate": yield_rate,
                "time_to_maturity": time_to_maturity,
                "modified_duration": modified_duration,
                "convexity": convexity,
                "day_count_convention": day_count_convention.value,
                "calculation_date": datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.error(
                "Bond pricing calculation failed",
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
    
    def _calculate_coupon_present_value(
        self,
        coupon_payment: Decimal,
        coupon_frequency: int,
        time_to_maturity: float,
        yield_rate: Decimal
    ) -> Decimal:
        """Calculate present value of coupon payments."""
        if time_to_maturity <= 0:
            return Decimal("0")
        
        total_pv = Decimal("0")
        for i in range(1, int(time_to_maturity * coupon_frequency) + 1):
            payment_time = i / coupon_frequency
            if payment_time <= time_to_maturity:
                discount_factor = Decimal(str(1 / (1 + yield_rate) ** payment_time))
                total_pv += coupon_payment * discount_factor
        
        return total_pv
    
    def _calculate_face_value_present_value(
        self,
        face_value: Decimal,
        time_to_maturity: float,
        yield_rate: Decimal
    ) -> Decimal:
        """Calculate present value of face value."""
        if time_to_maturity <= 0:
            return face_value
        
        discount_factor = Decimal(str(1 / (1 + yield_rate) ** time_to_maturity))
        return face_value * discount_factor
    
    def _calculate_accrued_interest(
        self,
        face_value: Decimal,
        coupon_rate: Decimal,
        coupon_frequency: int,
        settlement_date: Union[date, datetime],
        maturity_date: Union[date, datetime],
        day_count_convention: DayCountConvention
    ) -> Decimal:
        """Calculate accrued interest."""
        # This is a simplified calculation
        # In practice, you'd need the actual coupon schedule
        days_since_last_coupon, days_in_coupon_period = self.day_count_calc.calculate_accrued_interest_days(
            settlement_date, maturity_date, day_count_convention
        )
        
        if days_in_coupon_period > 0:
            accrued_fraction = Decimal(str(days_since_last_coupon / days_in_coupon_period))
            annual_coupon = coupon_rate * face_value
            return (annual_coupon / coupon_frequency) * accrued_fraction
        
        return Decimal("0")
    
    def _calculate_modified_duration(
        self,
        clean_price: Decimal,
        face_value: Decimal,
        coupon_rate: Decimal,
        coupon_frequency: int,
        time_to_maturity: float,
        yield_rate: Decimal
    ) -> Decimal:
        """Calculate modified duration."""
        if time_to_maturity <= 0:
            return Decimal("0")
        
        # Simplified duration calculation
        # In practice, you'd use the actual cash flow schedule
        duration = time_to_maturity / (1 + yield_rate)
        return Decimal(str(duration))
    
    def _calculate_convexity(
        self,
        clean_price: Decimal,
        face_value: Decimal,
        coupon_rate: Decimal,
        coupon_frequency: int,
        time_to_maturity: float,
        yield_rate: Decimal
    ) -> Decimal:
        """Calculate convexity."""
        if time_to_maturity <= 0:
            return Decimal("0")
        
        # Simplified convexity calculation
        # In practice, you'd use the actual cash flow schedule
        convexity = (time_to_maturity * (time_to_maturity + 1)) / ((1 + yield_rate) ** 2)
        return Decimal(str(convexity))
    
    def calculate_floating_rate_price(
        self,
        face_value: Decimal,
        reference_rate: Decimal,
        spread: Decimal,
        coupon_frequency: int,
        maturity_date: Union[date, datetime],
        settlement_date: Union[date, datetime],
        day_count_convention: DayCountConvention = DayCountConvention.THIRTY_360,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate price for floating rate bonds.
        
        Args:
            face_value: Face value of the bond
            reference_rate: Current reference rate (e.g., MIBOR)
            spread: Spread over reference rate
            coupon_frequency: Number of coupon payments per year
            maturity_date: Bond maturity date
            settlement_date: Settlement date
            day_count_convention: Day count convention to use
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing pricing information
        """
        # For floating rate bonds, price is typically close to par
        # This is a simplified implementation
        current_coupon_rate = reference_rate + spread
        
        return self.calculate_bond_price(
            face_value=face_value,
            coupon_rate=current_coupon_rate,
            coupon_frequency=coupon_frequency,
            maturity_date=maturity_date,
            settlement_date=settlement_date,
            yield_rate=current_coupon_rate,
            day_count_convention=day_count_convention,
            **kwargs
        )
    
    def calculate_zero_coupon_price(
        self,
        face_value: Decimal,
        maturity_date: Union[date, datetime],
        settlement_date: Union[date, datetime],
        yield_rate: Decimal,
        day_count_convention: DayCountConvention = DayCountConvention.THIRTY_360,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate price for zero coupon bonds.
        
        Args:
            face_value: Face value of the bond
            maturity_date: Bond maturity date
            settlement_date: Settlement date
            yield_rate: Yield rate for pricing
            day_count_convention: Day count convention to use
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing pricing information
        """
        # Zero coupon bonds have no coupon payments
        time_to_maturity = self._calculate_time_to_maturity(
            settlement_date, maturity_date, day_count_convention
        )
        
        if time_to_maturity <= 0:
            clean_price = face_value
        else:
            discount_factor = Decimal(str(1 / (1 + yield_rate) ** time_to_maturity))
            clean_price = face_value * discount_factor
        
        return {
            "clean_price": clean_price,
            "dirty_price": clean_price,  # No accrued interest for zero coupon bonds
            "accrued_interest": Decimal("0"),
            "yield_rate": yield_rate,
            "time_to_maturity": time_to_maturity,
            "modified_duration": Decimal(str(time_to_maturity / (1 + yield_rate))),
            "convexity": Decimal(str((time_to_maturity * (time_to_maturity + 1)) / ((1 + yield_rate) ** 2))),
            "day_count_convention": day_count_convention.value,
            "calculation_date": datetime.utcnow()
        }


# Export the pricing engine
__all__ = ["BondPricingEngine"]
