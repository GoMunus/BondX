"""
Cash flow engine for BondX Backend.

This module implements cash flow projections and analysis for various bond types
including fixed-rate, floating-rate, zero-coupon, and amortizing bonds.
"""

import calendar
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.logging import get_logger
from ..database.models import DayCountConvention, CouponType
from .day_count import DayCountCalculator

logger = get_logger(__name__)


class CouponFrequency(Enum):
    """Coupon frequency enumeration."""
    MONTHLY = 12
    QUARTERLY = 4
    SEMI_ANNUAL = 2
    ANNUAL = 1


class RollConvention(Enum):
    """Roll convention for payment dates."""
    FOLLOWING = "FOLLOWING"  # Roll to next business day
    PRECEDING = "PRECEDING"  # Roll to previous business day
    MODIFIED_FOLLOWING = "MODIFIED_FOLLOWING"  # Roll to next business day, but if next month, roll to previous
    MODIFIED_PRECEDING = "MODIFIED_PRECEDING"  # Roll to previous business day, but if previous month, roll to next


class AmortizationType(Enum):
    """Amortization type for bonds."""
    EQUAL_PRINCIPAL = "EQUAL_PRINCIPAL"  # Equal principal payments
    ANNUITY = "ANNUITY"  # Equal total payments (principal + interest)
    BULLET = "BULLET"  # No amortization until maturity
    CUSTOM = "CUSTOM"  # Custom amortization schedule


@dataclass
class CashFlow:
    """Represents a single cash flow period."""
    period_index: int
    accrual_start: date
    accrual_end: date
    payment_date: date
    coupon_rate_effective: Decimal
    coupon_amount: Decimal
    principal_repayment: Decimal
    outstanding_principal_before: Decimal
    outstanding_principal_after: Decimal
    accrued_days: int
    accrual_factor: Decimal
    is_stub: bool
    notes: str = ""


@dataclass
class BondCashFlowConfig:
    """Configuration for bond cash flow generation."""
    coupon_rate: Decimal
    coupon_frequency: CouponFrequency
    day_count_convention: DayCountConvention
    issue_date: date
    maturity_date: date
    face_value: Decimal
    coupon_type: CouponType = CouponType.FIXED
    floating_index: Optional[str] = None  # e.g., "MIBOR", "LIBOR"
    spread_bps: Optional[int] = None  # Spread in basis points
    rate_reset_dates: Optional[List[date]] = None
    rate_cap: Optional[Decimal] = None
    rate_floor: Optional[Decimal] = None
    amortization_type: AmortizationType = AmortizationType.BULLET
    amortization_schedule: Optional[Dict[date, Decimal]] = None
    sinking_fund: Optional[Dict[date, Decimal]] = None
    stub_periods: Optional[Dict[str, date]] = None  # "front" or "back" stub dates
    holiday_calendar: Optional[List[date]] = None
    roll_convention: RollConvention = RollConvention.FOLLOWING
    first_coupon_date: Optional[date] = None
    last_coupon_date: Optional[date] = None


class HolidayCalendar:
    """Holiday calendar for business day adjustments."""
    
    def __init__(self, holidays: Optional[List[date]] = None):
        """Initialize holiday calendar."""
        self.holidays = set(holidays) if holidays else set()
    
    def is_business_day(self, check_date: date) -> bool:
        """Check if date is a business day."""
        return (
            check_date.weekday() < 5 and  # Monday = 0, Friday = 4
            check_date not in self.holidays
        )
    
    def adjust_date(self, target_date: date, convention: RollConvention) -> date:
        """Adjust date according to roll convention."""
        if self.is_business_day(target_date):
            return target_date
        
        if convention == RollConvention.FOLLOWING:
            return self._next_business_day(target_date)
        elif convention == RollConvention.PRECEDING:
            return self._previous_business_day(target_date)
        elif convention == RollConvention.MODIFIED_FOLLOWING:
            return self._modified_following(target_date)
        elif convention == RollConvention.MODIFIED_PRECEDING:
            return self._modified_preceding(target_date)
        else:
            return target_date
    
    def _next_business_day(self, target_date: date) -> date:
        """Find next business day."""
        current = target_date
        while not self.is_business_day(current):
            current += timedelta(days=1)
        return current
    
    def _previous_business_day(self, target_date: date) -> date:
        """Find previous business day."""
        current = target_date
        while not self.is_business_day(current):
            current -= timedelta(days=1)
        return current
    
    def _modified_following(self, target_date: date) -> date:
        """Modified following convention."""
        next_business = self._next_business_day(target_date)
        if next_business.month != target_date.month:
            return self._previous_business_day(target_date)
        return next_business
    
    def _modified_preceding(self, target_date: date) -> date:
        """Modified preceding convention."""
        prev_business = self._previous_business_day(target_date)
        if prev_business.month != target_date.month:
            return self._next_business_day(target_date)
        return prev_business


class CashFlowEngine:
    """
    Production-grade cash flow engine for bond instruments.
    
    Supports fixed-rate, floating-rate, zero-coupon, and amortizing bonds
    with comprehensive business day handling and validation.
    """
    
    def __init__(self):
        """Initialize the cash flow engine."""
        self.logger = logger
        self.day_count_calculator = DayCountCalculator()
    
    def generate_cash_flows(
        self,
        config: BondCashFlowConfig,
        valuation_date: Optional[date] = None,
        index_rates: Optional[Dict[date, Decimal]] = None
    ) -> List[CashFlow]:
        """
        Generate cash flows for a bond instrument.
        
        Args:
            config: Bond configuration
            valuation_date: Valuation date for floating rate bonds
            index_rates: Historical index rates for floating rate bonds
            
        Returns:
            List of cash flow objects
            
        Raises:
            ValueError: If configuration is invalid
        """
        self._validate_config(config)
        
        # Initialize holiday calendar
        holiday_calendar = HolidayCalendar(config.holiday_calendar)
        
        # Generate schedule dates
        schedule_dates = self._generate_schedule_dates(config, holiday_calendar)
        
        # Generate cash flows based on bond type
        if config.coupon_type == CouponType.ZERO_COUPON:
            return self._generate_zero_coupon_flows(config, schedule_dates, holiday_calendar)
        elif config.coupon_type == CouponType.FLOATING:
            return self._generate_floating_flows(config, schedule_dates, holiday_calendar, index_rates)
        else:
            return self._generate_fixed_flows(config, schedule_dates, holiday_calendar)
    
    def _validate_config(self, config: BondCashFlowConfig):
        """Validate bond configuration."""
        if config.issue_date >= config.maturity_date:
            raise ValueError("Issue date must be before maturity date")
        
        if config.face_value <= 0:
            raise ValueError("Face value must be positive")
        
        if config.coupon_rate < 0:
            raise ValueError("Coupon rate cannot be negative")
        
        if config.coupon_type == CouponType.FLOATING:
            if not config.floating_index:
                raise ValueError("Floating index required for floating rate bonds")
            if config.spread_bps is None:
                raise ValueError("Spread required for floating rate bonds")
        
        if config.amortization_type == AmortizationType.CUSTOM and not config.amortization_schedule:
            raise ValueError("Custom amortization schedule required for custom amortization type")
    
    def _generate_schedule_dates(
        self,
        config: BondCashFlowConfig,
        holiday_calendar: HolidayCalendar
    ) -> List[date]:
        """Generate schedule dates for the bond."""
        dates = []
        
        # Handle stub periods
        if config.stub_periods:
            if "front" in config.stub_periods:
                dates.append(config.stub_periods["front"])
            
            # Generate regular schedule
            current_date = config.issue_date
            while current_date < config.maturity_date:
                next_date = self._add_months(current_date, 12 // config.coupon_frequency.value)
                if next_date <= config.maturity_date:
                    dates.append(next_date)
                current_date = next_date
            
            if "back" in config.stub_periods:
                dates.append(config.stub_periods["back"])
        else:
            # Standard schedule
            current_date = config.issue_date
            while current_date < config.maturity_date:
                next_date = self._add_months(current_date, 12 // config.coupon_frequency.value)
                if next_date <= config.maturity_date:
                    dates.append(next_date)
                current_date = next_date
        
        # Add maturity date if not already included
        if config.maturity_date not in dates:
            dates.append(config.maturity_date)
        
        # Sort and remove duplicates
        dates = sorted(list(set(dates)))
        
        # Adjust for business days
        adjusted_dates = []
        for date in dates:
            adjusted_date = holiday_calendar.adjust_date(date, config.roll_convention)
            adjusted_dates.append(adjusted_date)
        
        return adjusted_dates
    
    def _add_months(self, start_date: date, months: int) -> date:
        """Add months to a date, handling year boundaries."""
        year = start_date.year + (start_date.month + months - 1) // 12
        month = (start_date.month + months - 1) % 12 + 1
        day = min(start_date.day, calendar.monthrange(year, month)[1])
        return date(year, month, day)
    
    def _generate_zero_coupon_flows(
        self,
        config: BondCashFlowConfig,
        schedule_dates: List[date],
        holiday_calendar: HolidayCalendar
    ) -> List[CashFlow]:
        """Generate cash flows for zero-coupon bonds."""
        flows = []
        outstanding_principal = config.face_value
        
        for i, payment_date in enumerate(schedule_dates):
            if i == 0:  # Issue date
                flows.append(CashFlow(
                    period_index=i,
                    accrual_start=config.issue_date,
                    accrual_end=payment_date,
                    payment_date=payment_date,
                    coupon_rate_effective=Decimal('0'),
                    coupon_amount=Decimal('0'),
                    principal_repayment=Decimal('0'),
                    outstanding_principal_before=outstanding_principal,
                    outstanding_principal_after=outstanding_principal,
                    accrued_days=0,
                    accrual_factor=Decimal('0'),
                    is_stub=False,
                    notes="Issue period"
                ))
            elif i == len(schedule_dates) - 1:  # Maturity
                flows.append(CashFlow(
                    period_index=i,
                    accrual_start=schedule_dates[i-1],
                    accrual_end=payment_date,
                    payment_date=payment_date,
                    coupon_rate_effective=Decimal('0'),
                    coupon_amount=Decimal('0'),
                    principal_repayment=outstanding_principal,
                    outstanding_principal_before=outstanding_principal,
                    outstanding_principal_after=Decimal('0'),
                    accrued_days=self._calculate_accrued_days(schedule_dates[i-1], payment_date, config.day_count_convention),
                    accrual_factor=self._calculate_accrual_factor(schedule_dates[i-1], payment_date, config.day_count_convention),
                    is_stub=False,
                    notes="Maturity payment"
                ))
        
        return flows
    
    def _generate_fixed_flows(
        self,
        config: BondCashFlowConfig,
        schedule_dates: List[date],
        holiday_calendar: HolidayCalendar
    ) -> List[CashFlow]:
        """Generate cash flows for fixed-rate bonds."""
        flows = []
        outstanding_principal = config.face_value
        
        # Calculate coupon per period
        coupon_per_period = (config.face_value * config.coupon_rate / 
                           config.coupon_frequency.value).quantize(Decimal('0.01'), ROUND_HALF_UP)
        
        for i, payment_date in enumerate(schedule_dates):
            if i == 0:  # Issue date
                flows.append(CashFlow(
                    period_index=i,
                    accrual_start=config.issue_date,
                    accrual_end=payment_date,
                    payment_date=payment_date,
                    coupon_rate_effective=config.coupon_rate,
                    coupon_amount=Decimal('0'),
                    principal_repayment=Decimal('0'),
                    outstanding_principal_before=outstanding_principal,
                    outstanding_principal_after=outstanding_principal,
                    accrued_days=0,
                    accrual_factor=Decimal('0'),
                    is_stub=False,
                    notes="Issue period"
                ))
            else:
                # Calculate principal repayment for amortizing bonds
                principal_repayment = self._calculate_principal_repayment(
                    config, outstanding_principal, i, len(schedule_dates)
                )
                
                # Apply sinking fund if applicable
                if config.sinking_fund and payment_date in config.sinking_fund:
                    principal_repayment += config.sinking_fund[payment_date]
                
                # Ensure we don't overpay
                principal_repayment = min(principal_repayment, outstanding_principal)
                
                # Calculate coupon amount
                accrual_start = schedule_dates[i-1]
                accrued_days = self._calculate_accrued_days(accrual_start, payment_date, config.day_count_convention)
                accrual_factor = self._calculate_accrual_factor(accrual_start, payment_date, config.day_count_convention)
                
                coupon_amount = (outstanding_principal * config.coupon_rate * accrual_factor).quantize(
                    Decimal('0.01'), ROUND_HALF_UP
                )
                
                # Update outstanding principal
                outstanding_principal_after = outstanding_principal - principal_repayment
                
                flows.append(CashFlow(
                    period_index=i,
                    accrual_start=accrual_start,
                    accrual_end=payment_date,
                    payment_date=payment_date,
                    coupon_rate_effective=config.coupon_rate,
                    coupon_amount=coupon_amount,
                    principal_repayment=principal_repayment,
                    outstanding_principal_before=outstanding_principal,
                    outstanding_principal_after=outstanding_principal_after,
                    accrued_days=accrued_days,
                    accrual_factor=accrual_factor,
                    is_stub=self._is_stub_period(accrual_start, payment_date, config),
                    notes=self._generate_flow_notes(config, i, payment_date)
                ))
                
                outstanding_principal = outstanding_principal_after
        
        return flows
    
    def _generate_floating_flows(
        self,
        config: BondCashFlowConfig,
        schedule_dates: List[date],
        holiday_calendar: HolidayCalendar,
        index_rates: Optional[Dict[date, Decimal]]
    ) -> List[CashFlow]:
        """Generate cash flows for floating-rate bonds."""
        flows = []
        outstanding_principal = config.face_value
        
        for i, payment_date in enumerate(schedule_dates):
            if i == 0:  # Issue date
                flows.append(CashFlow(
                    period_index=i,
                    accrual_start=config.issue_date,
                    accrual_end=payment_date,
                    payment_date=payment_date,
                    coupon_rate_effective=Decimal('0'),
                    coupon_amount=Decimal('0'),
                    principal_repayment=Decimal('0'),
                    outstanding_principal_before=outstanding_principal,
                    outstanding_principal_after=outstanding_principal,
                    accrued_days=0,
                    accrual_factor=Decimal('0'),
                    is_stub=False,
                    notes="Issue period"
                ))
            else:
                # Calculate floating rate
                accrual_start = schedule_dates[i-1]
                reset_date = self._get_reset_date(config, accrual_start)
                
                if index_rates and reset_date in index_rates:
                    base_rate = index_rates[reset_date]
                else:
                    # Use last known rate or default
                    base_rate = Decimal('0.05')  # 5% default
                
                # Apply spread
                effective_rate = base_rate + (Decimal(config.spread_bps) / Decimal('10000'))
                
                # Apply caps and floors
                if config.rate_cap:
                    effective_rate = min(effective_rate, config.rate_cap)
                if config.rate_floor:
                    effective_rate = max(effective_rate, config.rate_floor)
                
                # Calculate coupon amount
                accrued_days = self._calculate_accrued_days(accrual_start, payment_date, config.day_count_convention)
                accrual_factor = self._calculate_accrual_factor(accrual_start, payment_date, config.day_count_convention)
                
                coupon_amount = (outstanding_principal * effective_rate * accrual_factor).quantize(
                    Decimal('0.01'), ROUND_HALF_UP
                )
                
                # Principal repayment (same logic as fixed)
                principal_repayment = self._calculate_principal_repayment(
                    config, outstanding_principal, i, len(schedule_dates)
                )
                
                if config.sinking_fund and payment_date in config.sinking_fund:
                    principal_repayment += config.sinking_fund[payment_date]
                
                principal_repayment = min(principal_repayment, outstanding_principal)
                outstanding_principal_after = outstanding_principal - principal_repayment
                
                flows.append(CashFlow(
                    period_index=i,
                    accrual_start=accrual_start,
                    accrual_end=payment_date,
                    payment_date=payment_date,
                    coupon_rate_effective=effective_rate,
                    coupon_amount=coupon_amount,
                    principal_repayment=principal_repayment,
                    outstanding_principal_before=outstanding_principal,
                    outstanding_principal_after=outstanding_principal_after,
                    accrued_days=accrued_days,
                    accrual_factor=accrual_factor,
                    is_stub=self._is_stub_period(accrual_start, payment_date, config),
                    notes=f"Floating rate: {effective_rate:.4%}, Reset: {reset_date}"
                ))
                
                outstanding_principal = outstanding_principal_after
        
        return flows
    
    def _calculate_principal_repayment(
        self,
        config: BondCashFlowConfig,
        outstanding_principal: Decimal,
        period_index: int,
        total_periods: int
    ) -> Decimal:
        """Calculate principal repayment for the period."""
        if config.amortization_type == AmortizationType.BULLET:
            return Decimal('0') if period_index < total_periods - 1 else outstanding_principal
        
        elif config.amortization_type == AmortizationType.EQUAL_PRINCIPAL:
            return outstanding_principal / (total_periods - period_index)
        
        elif config.amortization_type == AmortizationType.ANNUITY:
            # Equal total payment (principal + interest)
            # This is a simplified calculation - in practice, this would be more complex
            if period_index == total_periods - 1:
                return outstanding_principal
            else:
                return Decimal('0')  # Simplified for now
        
        elif config.amortization_type == AmortizationType.CUSTOM:
            # Use custom schedule
            payment_date = config.amortization_schedule.get(payment_date, Decimal('0'))
            return payment_date
        
        return Decimal('0')
    
    def _get_reset_date(self, config: BondCashFlowConfig, accrual_start: date) -> date:
        """Get the rate reset date for a floating rate bond."""
        if config.rate_reset_dates:
            # Find the most recent reset date before or on accrual start
            reset_dates = sorted(config.rate_reset_dates)
            for reset_date in reversed(reset_dates):
                if reset_date <= accrual_start:
                    return reset_date
        
        # Default to accrual start date
        return accrual_start
    
    def _calculate_accrued_days(self, start_date: date, end_date: date, convention: DayCountConvention) -> int:
        """Calculate accrued days using day count convention."""
        days, _ = self.day_count_calculator.calculate_days(start_date, end_date, convention)
        return days
    
    def _calculate_accrual_factor(self, start_date: date, end_date: date, convention: DayCountConvention) -> Decimal:
        """Calculate accrual factor using day count convention."""
        days, days_in_year = self.day_count_calculator.calculate_days(start_date, end_date, convention)
        return Decimal(days) / Decimal(days_in_year)
    
    def _is_stub_period(self, start_date: date, end_date: date, config: BondCashFlowConfig) -> bool:
        """Check if period is a stub period."""
        if not config.stub_periods:
            return False
        
        expected_months = 12 // config.coupon_frequency.value
        actual_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        
        return actual_months != expected_months
    
    def _generate_flow_notes(self, config: BondCashFlowConfig, period_index: int, payment_date: date) -> str:
        """Generate notes for cash flow."""
        notes = []
        
        if config.sinking_fund and payment_date in config.sinking_fund:
            notes.append(f"Sinking fund: {config.sinking_fund[payment_date]}")
        
        if period_index == 0:
            notes.append("Issue period")
        elif payment_date == config.maturity_date:
            notes.append("Maturity payment")
        
        return "; ".join(notes) if notes else ""
    
    def get_next_payment_date(self, flows: List[CashFlow], from_date: date) -> Optional[CashFlow]:
        """Get the next payment date after a given date."""
        for flow in flows:
            if flow.payment_date > from_date:
                return flow
        return None
    
    def get_last_coupon_date(self, flows: List[CashFlow]) -> Optional[date]:
        """Get the last coupon payment date."""
        coupon_flows = [f for f in flows if f.coupon_amount > 0]
        if not coupon_flows:
            return None
        return max(f.payment_date for f in coupon_flows)
    
    def get_total_principal_repayment(self, flows: List[CashFlow]) -> Decimal:
        """Get total principal repayment."""
        return sum(f.principal_repayment for f in flows)
    
    def get_total_coupon_payments(self, flows: List[CashFlow]) -> Decimal:
        """Get total coupon payments."""
        return sum(f.coupon_amount for f in flows)
    
    def validate_cash_flows(self, flows: List[CashFlow], face_value: Decimal) -> bool:
        """Validate cash flows for consistency."""
        if not flows:
            return False
        
        # Check total principal repayment equals face value
        total_principal = self.get_total_principal_repayment(flows)
        if abs(total_principal - face_value) > Decimal('0.01'):
            return False
        
        # Check dates are monotonic
        dates = [f.payment_date for f in flows]
        if dates != sorted(dates):
            return False
        
        # Check no overlapping accrual periods
        for i in range(len(flows) - 1):
            if flows[i].accrual_end > flows[i + 1].accrual_start:
                return False
        
        return True


__all__ = [
    "CashFlowEngine",
    "CashFlow",
    "BondCashFlowConfig",
    "CouponFrequency",
    "RollConvention",
    "AmortizationType",
    "HolidayCalendar"
]
