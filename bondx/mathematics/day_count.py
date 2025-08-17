"""
Day count convention calculations for BondX Backend.

This module implements various day count conventions used in bond calculations
including ACT/ACT, ACT/365, ACT/360, 30/360, and 30/365 methods.
"""

import calendar
from datetime import date, datetime
from typing import Tuple, Union

from ..core.logging import get_logger
from ..database.models import DayCountConvention

logger = get_logger(__name__)


class DayCountCalculator:
    """
    Calculator for various day count conventions used in bond markets.
    
    Supports multiple day count conventions:
    - ACT/ACT: Actual/Actual (used for government securities)
    - ACT/365: Actual/365
    - ACT/360: Actual/360
    - 30/360: 30/360 (used for most corporate bonds)
    - 30/365: 30/365
    """
    
    def __init__(self):
        """Initialize the day count calculator."""
        self.logger = logger
    
    def calculate_days(
        self,
        start_date: Union[date, datetime],
        end_date: Union[date, datetime],
        convention: DayCountConvention,
        **kwargs
    ) -> Tuple[int, int]:
        """
        Calculate the number of days between two dates using the specified convention.
        
        Args:
            start_date: Start date
            end_date: End date
            convention: Day count convention to use
            **kwargs: Additional parameters for specific conventions
            
        Returns:
            Tuple of (days_between_dates, days_in_year)
            
        Raises:
            ValueError: If dates are invalid or convention is not supported
        """
        if start_date > end_date:
            raise ValueError("Start date must be before or equal to end date")
        
        # Convert datetime to date if necessary
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()
        
        try:
            if convention == DayCountConvention.ACT_ACT:
                return self._act_act(start_date, end_date, **kwargs)
            elif convention == DayCountConvention.ACT_365:
                return self._act_365(start_date, end_date, **kwargs)
            elif convention == DayCountConvention.ACT_360:
                return self._act_360(start_date, end_date, **kwargs)
            elif convention == DayCountConvention.THIRTY_360:
                return self._thirty_360(start_date, end_date, **kwargs)
            elif convention == DayCountConvention.THIRTY_365:
                return self._thirty_365(start_date, end_date, **kwargs)
            else:
                raise ValueError(f"Unsupported day count convention: {convention}")
                
        except Exception as e:
            self.logger.error(
                "Error calculating days",
                start_date=str(start_date),
                end_date=str(end_date),
                convention=convention.value,
                error=str(e)
            )
            raise
    
    def _act_act(self, start_date: date, end_date: date, **kwargs) -> Tuple[int, int]:
        """
        Calculate days using ACT/ACT convention.
        
        This is the most accurate method and is used for government securities.
        It counts actual days and uses actual days in the year.
        
        Args:
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (days_between_dates, days_in_year)
        """
        days_between = (end_date - start_date).days
        
        # For ACT/ACT, we need to handle leap years properly
        # If the period spans a leap year, we need to calculate weighted average
        start_year = start_date.year
        end_year = end_date.year
        
        if start_year == end_year:
            # Same year - use actual days in that year
            days_in_year = 366 if calendar.isleap(start_year) else 365
        else:
            # Different years - calculate weighted average
            total_days = 0
            total_weight = 0
            
            # Handle start year
            start_year_end = date(start_year, 12, 31)
            days_in_start_year = (start_year_end - start_date).days + 1
            days_in_start_year_total = 366 if calendar.isleap(start_year) else 365
            total_days += days_in_start_year
            total_weight += days_in_start_year_total
            
            # Handle middle years
            for year in range(start_year + 1, end_year):
                days_in_year_total = 366 if calendar.isleap(year) else 365
                total_days += days_in_year_total
                total_weight += days_in_year_total
            
            # Handle end year
            end_year_start = date(end_year, 1, 1)
            days_in_end_year = (end_date - end_year_start).days + 1
            days_in_end_year_total = 366 if calendar.isleap(end_year) else 365
            total_days += days_in_end_year
            total_weight += days_in_end_year_total
            
            # Calculate weighted average
            days_in_year = total_weight
        
        return days_between, days_in_year
    
    def _act_365(self, start_date: date, end_date: date, **kwargs) -> Tuple[int, int]:
        """
        Calculate days using ACT/365 convention.
        
        Counts actual days between dates but always uses 365 days in the year.
        
        Args:
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (days_between_dates, 365)
        """
        days_between = (end_date - start_date).days
        return days_between, 365
    
    def _act_360(self, start_date: date, end_date: date, **kwargs) -> Tuple[int, int]:
        """
        Calculate days using ACT/360 convention.
        
        Counts actual days between dates but always uses 360 days in the year.
        
        Args:
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (days_between_dates, 360)
        """
        days_between = (end_date - start_date).days
        return days_between, 360
    
    def _thirty_360(self, start_date: date, end_date: date, **kwargs) -> Tuple[int, int]:
        """
        Calculate days using 30/360 convention.
        
        This is the most common convention for corporate bonds.
        Assumes 30 days per month and 360 days per year.
        
        Args:
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (days_between_dates, 360)
        """
        # Extract components
        start_day = start_date.day
        start_month = start_date.month
        start_year = start_date.year
        
        end_day = end_date.day
        end_month = end_date.month
        end_year = end_date.year
        
        # Apply 30/360 rules
        # If start date is 31st, treat as 30th
        if start_day == 31:
            start_day = 30
        
        # If end date is 31st and start date is 30th or 31st, treat as 30th
        if end_day == 31 and start_day >= 30:
            end_day = 30
        
        # Calculate days
        days_between = (
            (end_year - start_year) * 360 +
            (end_month - start_month) * 30 +
            (end_day - start_day)
        )
        
        return days_between, 360
    
    def _thirty_365(self, start_date: date, end_date: date, **kwargs) -> Tuple[int, int]:
        """
        Calculate days using 30/365 convention.
        
        Similar to 30/360 but uses 365 days in the year.
        
        Args:
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (days_between_dates, 365)
        """
        # Use 30/360 calculation but return 365 as denominator
        days_between, _ = self._thirty_360(start_date, end_date, **kwargs)
        return days_between, 365
    
    def calculate_fraction(
        self,
        start_date: Union[date, datetime],
        end_date: Union[date, datetime],
        convention: DayCountConvention,
        **kwargs
    ) -> float:
        """
        Calculate the fraction of a year between two dates.
        
        Args:
            start_date: Start date
            end_date: End date
            convention: Day count convention to use
            **kwargs: Additional parameters
            
        Returns:
            Fraction of a year as a float
        """
        days_between, days_in_year = self.calculate_days(
            start_date, end_date, convention, **kwargs
        )
        return days_between / days_in_year
    
    def calculate_accrued_interest_days(
        self,
        last_coupon_date: Union[date, datetime],
        settlement_date: Union[date, datetime],
        convention: DayCountConvention,
        **kwargs
    ) -> Tuple[int, int]:
        """
        Calculate days for accrued interest calculation.
        
        Args:
            last_coupon_date: Date of last coupon payment
            settlement_date: Settlement date
            convention: Day count convention to use
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (days_since_last_coupon, days_in_coupon_period)
        """
        days_since_last = self.calculate_days(
            last_coupon_date, settlement_date, convention, **kwargs
        )[0]
        
        # For coupon period, we need to estimate the next coupon date
        # This is a simplified approach - in practice, you'd need the actual schedule
        if convention in [DayCountConvention.THIRTY_360, DayCountConvention.THIRTY_365]:
            days_in_period = 360 if convention == DayCountConvention.THIRTY_360 else 365
        else:
            # For ACT conventions, estimate based on typical coupon frequency
            # This is a simplification - actual implementation would need coupon schedule
            days_in_period = 365
        
        return days_since_last, days_in_period
    
    def get_convention_description(self, convention: DayCountConvention) -> str:
        """
        Get a human-readable description of the day count convention.
        
        Args:
            convention: Day count convention
            
        Returns:
            Description string
        """
        descriptions = {
            DayCountConvention.ACT_ACT: "Actual/Actual - Most accurate, used for government securities",
            DayCountConvention.ACT_365: "Actual/365 - Actual days, 365 days per year",
            DayCountConvention.ACT_360: "Actual/360 - Actual days, 360 days per year",
            DayCountConvention.THIRTY_360: "30/360 - 30 days per month, 360 days per year (corporate bonds)",
            DayCountConvention.THIRTY_365: "30/365 - 30 days per month, 365 days per year",
        }
        return descriptions.get(convention, "Unknown convention")
    
    def validate_convention_for_bond_type(
        self,
        convention: DayCountConvention,
        bond_type: str
    ) -> bool:
        """
        Validate if a day count convention is appropriate for a bond type.
        
        Args:
            convention: Day count convention
            bond_type: Type of bond
            
        Returns:
            True if convention is appropriate, False otherwise
        """
        # Government securities typically use ACT/ACT
        if bond_type.upper() in ["GOVERNMENT_SECURITY", "G-SEC", "GOVT"]:
            return convention == DayCountConvention.ACT_ACT
        
        # Corporate bonds typically use 30/360
        if bond_type.upper() in ["CORPORATE_BOND", "CORP", "BANK_BOND", "PSU_BOND"]:
            return convention in [DayCountConvention.THIRTY_360, DayCountConvention.THIRTY_365]
        
        # Municipal bonds can use either
        if bond_type.upper() in ["MUNICIPAL_BOND", "MUNI"]:
            return convention in [
                DayCountConvention.THIRTY_360,
                DayCountConvention.ACT_ACT,
                DayCountConvention.ACT_365
            ]
        
        # Default to True for other types
        return True
    
    def get_recommended_convention(self, bond_type: str) -> DayCountConvention:
        """
        Get the recommended day count convention for a bond type.
        
        Args:
            bond_type: Type of bond
            
        Returns:
            Recommended day count convention
        """
        recommendations = {
            "GOVERNMENT_SECURITY": DayCountConvention.ACT_ACT,
            "G-SEC": DayCountConvention.ACT_ACT,
            "GOVT": DayCountConvention.ACT_ACT,
            "STATE_DEVELOPMENT_LOAN": DayCountConvention.ACT_ACT,
            "SDL": DayCountConvention.ACT_ACT,
            "CORPORATE_BOND": DayCountConvention.THIRTY_360,
            "CORP": DayCountConvention.THIRTY_360,
            "BANK_BOND": DayCountConvention.THIRTY_360,
            "PSU_BOND": DayCountConvention.THIRTY_360,
            "MUNICIPAL_BOND": DayCountConvention.THIRTY_360,
            "MUNI": DayCountConvention.THIRTY_360,
        }
        
        return recommendations.get(bond_type.upper(), DayCountConvention.THIRTY_360)


# Export the calculator class
__all__ = ["DayCountCalculator"]
