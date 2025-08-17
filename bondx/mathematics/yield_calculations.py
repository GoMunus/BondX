"""
Yield calculations for BondX Backend.

This module implements sophisticated yield-to-maturity calculations
using Newton-Raphson iteration with intelligent initial guesses.
"""

import math
from datetime import date, datetime
from decimal import Decimal
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import newton

from ..core.logging import get_logger
from ..database.models import CouponType, DayCountConvention
from .day_count import DayCountCalculator

logger = get_logger(__name__)


class YieldCalculator:
    """
    Advanced yield calculator for bond pricing.
    
    Implements sophisticated yield-to-maturity calculations using:
    - Newton-Raphson iteration for convergence
    - Intelligent initial guess algorithms
    - Multiple fallback methods for edge cases
    - Comprehensive error handling
    """
    
    def __init__(self):
        """Initialize the yield calculator."""
        self.logger = logger
        self.day_count_calc = DayCountCalculator()
        
        # Convergence parameters
        self.max_iterations = 100
        self.tolerance = 1e-8
        self.max_yield = 100.0  # 100% maximum yield
        self.min_yield = -50.0  # -50% minimum yield
    
    def calculate_yield_to_maturity(
        self,
        clean_price: Decimal,
        face_value: Decimal,
        coupon_rate: Decimal,
        coupon_frequency: int,
        maturity_date: Union[date, datetime],
        settlement_date: Union[date, datetime],
        issue_date: Optional[Union[date, datetime]] = None,
        day_count_convention: DayCountConvention = DayCountConvention.THIRTY_360,
        **kwargs
    ) -> Tuple[Decimal, dict]:
        """
        Calculate yield-to-maturity using Newton-Raphson iteration.
        
        Args:
            clean_price: Clean price of the bond
            face_value: Face value of the bond
            coupon_rate: Annual coupon rate (as decimal, e.g., 0.05 for 5%)
            coupon_frequency: Number of coupon payments per year
            maturity_date: Bond maturity date
            settlement_date: Settlement date
            issue_date: Bond issue date (optional)
            day_count_convention: Day count convention to use
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (yield_to_maturity, calculation_details)
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If calculation fails to converge
        """
        try:
            # Validate inputs
            self._validate_yield_inputs(
                clean_price, face_value, coupon_rate, coupon_frequency,
                maturity_date, settlement_date
            )
            
            # Convert to float for calculations
            clean_price_float = float(clean_price)
            face_value_float = float(face_value)
            coupon_rate_float = float(coupon_rate)
            
            # Calculate initial guess
            initial_guess = self._calculate_initial_guess(
                clean_price_float, face_value_float, coupon_rate_float,
                coupon_frequency, maturity_date, settlement_date,
                day_count_convention
            )
            
            # Calculate yield using Newton-Raphson
            ytm = self._newton_raphson_yield(
                clean_price_float, face_value_float, coupon_rate_float,
                coupon_frequency, maturity_date, settlement_date,
                day_count_convention, initial_guess, **kwargs
            )
            
            # Validate result
            if not self._is_valid_yield(ytm):
                # Try fallback methods
                ytm = self._fallback_yield_calculation(
                    clean_price_float, face_value_float, coupon_rate_float,
                    coupon_frequency, maturity_date, settlement_date,
                    day_count_convention, **kwargs
                )
            
            # Calculate additional metrics
            calculation_details = self._calculate_additional_metrics(
                ytm, clean_price_float, face_value_float, coupon_rate_float,
                coupon_frequency, maturity_date, settlement_date,
                day_count_convention
            )
            
            self.logger.info(
                "Yield calculation completed successfully",
                clean_price=clean_price_float,
                face_value=face_value_float,
                coupon_rate=coupon_rate_float,
                ytm=ytm,
                method="newton_raphson"
            )
            
            return Decimal(str(ytm)), calculation_details
            
        except Exception as e:
            self.logger.error(
                "Yield calculation failed",
                clean_price=str(clean_price),
                face_value=str(face_value),
                coupon_rate=str(coupon_rate),
                error=str(e)
            )
            raise
    
    def _validate_yield_inputs(
        self,
        clean_price: Decimal,
        face_value: Decimal,
        coupon_rate: Decimal,
        coupon_frequency: int,
        maturity_date: Union[date, datetime],
        settlement_date: Union[date, datetime]
    ) -> None:
        """Validate input parameters for yield calculation."""
        if clean_price <= 0:
            raise ValueError("Clean price must be positive")
        
        if face_value <= 0:
            raise ValueError("Face value must be positive")
        
        if coupon_rate < 0:
            raise ValueError("Coupon rate cannot be negative")
        
        if coupon_frequency <= 0:
            raise ValueError("Coupon frequency must be positive")
        
        if maturity_date <= settlement_date:
            raise ValueError("Maturity date must be after settlement date")
        
        if clean_price > face_value * 2:
            self.logger.warning(
                "Clean price is more than twice face value - may indicate data error",
                clean_price=str(clean_price),
                face_value=str(face_value)
            )
    
    def _calculate_initial_guess(
        self,
        clean_price: float,
        face_value: float,
        coupon_rate: float,
        coupon_frequency: int,
        maturity_date: Union[date, datetime],
        settlement_date: Union[date, datetime],
        day_count_convention: DayCountConvention
    ) -> float:
        """
        Calculate intelligent initial guess for yield-to-maturity.
        
        Uses multiple methods and combines them for robustness.
        """
        # Method 1: Current yield approximation
        current_yield = (coupon_rate * face_value) / clean_price
        
        # Method 2: Simple yield approximation
        time_to_maturity = self._calculate_time_to_maturity(
            settlement_date, maturity_date, day_count_convention
        )
        
        if time_to_maturity > 0:
            # Approximate yield using price-yield relationship
            price_ratio = clean_price / face_value
            if price_ratio > 1.0:
                # Premium bond - yield below coupon rate
                simple_yield = coupon_rate - (price_ratio - 1.0) / time_to_maturity
            else:
                # Discount bond - yield above coupon rate
                simple_yield = coupon_rate + (1.0 - price_ratio) / time_to_maturity
        else:
            simple_yield = current_yield
        
        # Method 3: Duration-adjusted guess
        duration_adjusted_yield = self._duration_adjusted_guess(
            clean_price, face_value, coupon_rate, coupon_frequency,
            time_to_maturity
        )
        
        # Combine methods with weights
        initial_guess = (
            0.4 * current_yield +
            0.3 * simple_yield +
            0.3 * duration_adjusted_yield
        )
        
        # Ensure reasonable bounds
        initial_guess = max(self.min_yield, min(self.max_yield, initial_guess))
        
        self.logger.debug(
            "Initial yield guess calculated",
            current_yield=current_yield,
            simple_yield=simple_yield,
            duration_adjusted_yield=duration_adjusted_yield,
            final_guess=initial_guess
        )
        
        return initial_guess
    
    def _duration_adjusted_guess(
        self,
        clean_price: float,
        face_value: float,
        coupon_rate: float,
        coupon_frequency: int,
        time_to_maturity: float
    ) -> float:
        """Calculate duration-adjusted initial yield guess."""
        if time_to_maturity <= 0:
            return coupon_rate
        
        # Approximate modified duration
        if clean_price != face_value:
            # Price sensitivity to yield changes
            price_change = abs(clean_price - face_value)
            duration_approx = time_to_maturity / (1 + coupon_rate)
            
            # Adjust yield based on duration and price deviation
            yield_adjustment = price_change / (clean_price * duration_approx)
            
            if clean_price > face_value:
                # Premium bond - yield below coupon
                return max(0, coupon_rate - yield_adjustment)
            else:
                # Discount bond - yield above coupon
                return coupon_rate + yield_adjustment
        else:
            return coupon_rate
    
    def _newton_raphson_yield(
        self,
        clean_price: float,
        face_value: float,
        coupon_rate: float,
        coupon_frequency: int,
        maturity_date: Union[date, datetime],
        settlement_date: Union[date, datetime],
        day_count_convention: DayCountConvention,
        initial_guess: float,
        **kwargs
    ) -> float:
        """
        Calculate yield using Newton-Raphson iteration.
        
        This is the primary method for yield calculation.
        """
        try:
            # Define the price function and its derivative
            def price_function(yield_rate: float) -> float:
                """Calculate bond price given yield rate."""
                return self._calculate_bond_price(
                    yield_rate, face_value, coupon_rate, coupon_frequency,
                    maturity_date, settlement_date, day_count_convention
                ) - clean_price
            
            def price_derivative(yield_rate: float) -> float:
                """Calculate derivative of price function with respect to yield."""
                return self._calculate_price_derivative(
                    yield_rate, face_value, coupon_rate, coupon_frequency,
                    maturity_date, settlement_date, day_count_convention
                )
            
            # Use scipy's newton function for robust convergence
            ytm = newton(
                price_function,
                initial_guess,
                fprime=price_derivative,
                maxiter=self.max_iterations,
                tol=self.tolerance
            )
            
            # Validate result
            if not self._is_valid_yield(ytm):
                raise ValueError(f"Invalid yield calculated: {ytm}")
            
            return ytm
            
        except Exception as e:
            self.logger.warning(
                "Newton-Raphson failed, trying fallback methods",
                error=str(e),
                initial_guess=initial_guess
            )
            raise
    
    def _calculate_bond_price(
        self,
        yield_rate: float,
        face_value: float,
        coupon_rate: float,
        coupon_frequency: int,
        maturity_date: Union[date, datetime],
        settlement_date: Union[date, datetime],
        day_count_convention: DayCountConvention
    ) -> float:
        """
        Calculate bond price given yield rate.
        
        This is the core pricing function used in Newton-Raphson iteration.
        """
        # Calculate time to maturity
        time_to_maturity = self._calculate_time_to_maturity(
            settlement_date, maturity_date, day_count_convention
        )
        
        if time_to_maturity <= 0:
            return face_value
        
        # Calculate coupon payment
        coupon_payment = (coupon_rate * face_value) / coupon_frequency
        
        # Calculate present value of coupons
        coupon_pv = 0.0
        for i in range(1, int(time_to_maturity * coupon_frequency) + 1):
            payment_time = i / coupon_frequency
            if payment_time <= time_to_maturity:
                discount_factor = math.exp(-yield_rate * payment_time)
                coupon_pv += coupon_payment * discount_factor
        
        # Calculate present value of face value
        face_value_pv = face_value * math.exp(-yield_rate * time_to_maturity)
        
        # Total price
        total_price = coupon_pv + face_value_pv
        
        return total_price
    
    def _calculate_price_derivative(
        self,
        yield_rate: float,
        face_value: float,
        coupon_rate: float,
        coupon_frequency: int,
        maturity_date: Union[date, datetime],
        settlement_date: Union[date, datetime],
        day_count_convention: DayCountConvention
    ) -> float:
        """
        Calculate derivative of price function with respect to yield.
        
        This is used in Newton-Raphson iteration for faster convergence.
        """
        # Calculate time to maturity
        time_to_maturity = self._calculate_time_to_maturity(
            settlement_date, maturity_date, day_count_convention
        )
        
        if time_to_maturity <= 0:
            return 0.0
        
        # Calculate coupon payment
        coupon_payment = (coupon_rate * face_value) / coupon_frequency
        
        # Calculate derivative of coupon present values
        coupon_derivative = 0.0
        for i in range(1, int(time_to_maturity * coupon_frequency) + 1):
            payment_time = i / coupon_frequency
            if payment_time <= time_to_maturity:
                discount_factor = math.exp(-yield_rate * payment_time)
                coupon_derivative += -payment_time * coupon_payment * discount_factor
        
        # Calculate derivative of face value present value
        face_value_derivative = -time_to_maturity * face_value * math.exp(-yield_rate * time_to_maturity)
        
        # Total derivative
        total_derivative = coupon_derivative + face_value_derivative
        
        return total_derivative
    
    def _calculate_time_to_maturity(
        self,
        settlement_date: Union[date, datetime],
        maturity_date: Union[date, datetime],
        day_count_convention: DayCountConvention
    ) -> float:
        """Calculate time to maturity in years using day count convention."""
        days_between, days_in_year = self.day_count_calc.calculate_days(
            settlement_date, maturity_date, day_count_convention
        )
        return days_between / days_in_year
    
    def _is_valid_yield(self, yield_rate: float) -> bool:
        """Check if calculated yield is within valid bounds."""
        return (
            isinstance(yield_rate, (int, float)) and
            not math.isnan(yield_rate) and
            not math.isinf(yield_rate) and
            self.min_yield <= yield_rate <= self.max_yield
        )
    
    def _fallback_yield_calculation(
        self,
        clean_price: float,
        face_value: float,
        coupon_rate: float,
        coupon_frequency: int,
        maturity_date: Union[date, datetime],
        settlement_date: Union[date, datetime],
        day_count_convention: DayCountConvention,
        **kwargs
    ) -> float:
        """
        Fallback yield calculation methods when Newton-Raphson fails.
        
        Uses simpler methods that are more robust but less accurate.
        """
        self.logger.info("Using fallback yield calculation methods")
        
        # Method 1: Current yield
        current_yield = (coupon_rate * face_value) / clean_price
        
        # Method 2: Simple approximation
        time_to_maturity = self._calculate_time_to_maturity(
            settlement_date, maturity_date, day_count_convention
        )
        
        if time_to_maturity > 0:
            # Simple yield approximation
            price_ratio = clean_price / face_value
            if price_ratio > 1.0:
                # Premium bond
                simple_yield = coupon_rate - (price_ratio - 1.0) / time_to_maturity
            else:
                # Discount bond
                simple_yield = coupon_rate + (1.0 - price_ratio) / time_to_maturity
        else:
            simple_yield = current_yield
        
        # Method 3: Linear interpolation
        linear_yield = self._linear_interpolation_yield(
            clean_price, face_value, coupon_rate, time_to_maturity
        )
        
        # Use the most reasonable result
        candidates = [current_yield, simple_yield, linear_yield]
        valid_candidates = [y for y in candidates if self._is_valid_yield(y)]
        
        if valid_candidates:
            # Choose the candidate closest to coupon rate
            best_yield = min(valid_candidates, key=lambda y: abs(y - coupon_rate))
            self.logger.info(
                "Fallback yield calculation successful",
                method="fallback",
                yield_rate=best_yield
            )
            return best_yield
        else:
            # Last resort - return coupon rate
            self.logger.warning("All fallback methods failed, using coupon rate")
            return coupon_rate
    
    def _linear_interpolation_yield(
        self,
        clean_price: float,
        face_value: float,
        coupon_rate: float,
        time_to_maturity: float
    ) -> float:
        """Calculate yield using linear interpolation."""
        if time_to_maturity <= 0:
            return coupon_rate
        
        # Simple linear relationship
        price_ratio = clean_price / face_value
        yield_rate = coupon_rate + (1.0 - price_ratio) / time_to_maturity
        
        return yield_rate
    
    def _calculate_additional_metrics(
        self,
        ytm: float,
        clean_price: float,
        face_value: float,
        coupon_rate: float,
        coupon_frequency: int,
        maturity_date: Union[date, datetime],
        settlement_date: Union[date, datetime],
        day_count_convention: DayCountConvention
    ) -> dict:
        """Calculate additional metrics for the yield calculation."""
        time_to_maturity = self._calculate_time_to_maturity(
            settlement_date, maturity_date, day_count_convention
        )
        
        # Current yield
        current_yield = (coupon_rate * face_value) / clean_price
        
        # Yield spread over risk-free rate (simplified)
        # In practice, you'd compare with appropriate benchmark
        yield_spread = ytm - coupon_rate
        
        # Price sensitivity
        price_sensitivity = -time_to_maturity / (1 + ytm)
        
        return {
            "current_yield": current_yield,
            "yield_spread": yield_spread,
            "time_to_maturity": time_to_maturity,
            "price_sensitivity": price_sensitivity,
            "calculation_method": "newton_raphson",
            "convergence_achieved": True
        }
    
    def calculate_yield_to_call(
        self,
        clean_price: Decimal,
        face_value: Decimal,
        coupon_rate: Decimal,
        coupon_frequency: int,
        call_date: Union[date, datetime],
        call_price: Decimal,
        settlement_date: Union[date, datetime],
        day_count_convention: DayCountConvention = DayCountConvention.THIRTY_360,
        **kwargs
    ) -> Tuple[Decimal, dict]:
        """
        Calculate yield-to-call for callable bonds.
        
        Similar to YTM but uses call date and call price instead of maturity.
        """
        # Implementation would be similar to YTM but with call parameters
        # This is a placeholder for the full implementation
        raise NotImplementedError("Yield-to-call calculation not yet implemented")
    
    def calculate_yield_to_put(
        self,
        clean_price: Decimal,
        face_value: Decimal,
        coupon_rate: Decimal,
        coupon_frequency: int,
        put_date: Union[date, datetime],
        put_price: Decimal,
        settlement_date: Union[date, datetime],
        day_count_convention: DayCountConvention = DayCountConvention.THIRTY_360,
        **kwargs
    ) -> Tuple[Decimal, dict]:
        """
        Calculate yield-to-put for putable bonds.
        
        Similar to YTM but uses put date and put price instead of maturity.
        """
        # Implementation would be similar to YTM but with put parameters
        # This is a placeholder for the full implementation
        raise NotImplementedError("Yield-to-put calculation not yet implemented")


# Export the calculator class
__all__ = ["YieldCalculator"]
