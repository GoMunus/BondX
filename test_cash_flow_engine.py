#!/usr/bin/env python3
"""
Comprehensive test suite for CashFlowEngine.

Tests all bond types, edge cases, validation logic, and performance requirements.
"""

import unittest
from datetime import date, timedelta
from decimal import Decimal
import time

from bondx.mathematics.cash_flows import (
    CashFlowEngine, CashFlow, BondCashFlowConfig, CouponFrequency,
    RollConvention, AmortizationType, HolidayCalendar
)
from bondx.database.models import DayCountConvention, CouponType


class TestCashFlowEngine(unittest.TestCase):
    """Test cases for CashFlowEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = CashFlowEngine()
        self.today = date.today()
        self.issue_date = self.today
        self.maturity_date = self.today + timedelta(days=365*5)  # 5 years
        
        # Sample holiday calendar
        self.holidays = [
            date(2024, 1, 1),   # New Year
            date(2024, 1, 26),  # Republic Day
            date(2024, 8, 15),  # Independence Day
            date(2024, 10, 2),  # Gandhi Jayanti
        ]
    
    def test_fixed_rate_bond_basic(self):
        """Test basic fixed-rate bond cash flow generation."""
        config = BondCashFlowConfig(
            coupon_rate=Decimal('0.05'),  # 5%
            coupon_frequency=CouponFrequency.SEMI_ANNUAL,
            day_count_convention=DayCountConvention.ACT_365,
            issue_date=self.issue_date,
            maturity_date=self.maturity_date,
            face_value=Decimal('1000000'),  # 1M
            coupon_type=CouponType.FIXED
        )
        
        flows = self.engine.generate_cash_flows(config)
        
        # Basic validation
        self.assertGreater(len(flows), 0)
        self.assertEqual(flows[0].period_index, 0)
        self.assertEqual(flows[-1].payment_date, self.maturity_date)
        
        # Check total principal repayment equals face value
        total_principal = self.engine.get_total_principal_repayment(flows)
        self.assertAlmostEqual(float(total_principal), 1000000.0, places=2)
        
        # Check coupon amounts are positive
        for flow in flows[1:]:  # Skip issue period
            self.assertGreater(flow.coupon_amount, 0)
    
    def test_zero_coupon_bond(self):
        """Test zero-coupon bond cash flow generation."""
        config = BondCashFlowConfig(
            coupon_rate=Decimal('0'),
            coupon_frequency=CouponFrequency.ANNUAL,
            day_count_convention=DayCountConvention.ACT_365,
            issue_date=self.issue_date,
            maturity_date=self.maturity_date,
            face_value=Decimal('1000000'),
            coupon_type=CouponType.ZERO_COUPON
        )
        
        flows = self.engine.generate_cash_flows(config)
        
        # Zero-coupon should have only issue and maturity flows
        self.assertEqual(len(flows), 2)
        
        # No coupon payments
        for flow in flows:
            self.assertEqual(flow.coupon_amount, 0)
        
        # Principal only at maturity
        self.assertEqual(flows[-1].principal_repayment, Decimal('1000000'))
        self.assertEqual(flows[-1].outstanding_principal_after, 0)
    
    def test_floating_rate_bond(self):
        """Test floating-rate bond cash flow generation."""
        config = BondCashFlowConfig(
            coupon_rate=Decimal('0.05'),  # Base rate
            coupon_frequency=CouponFrequency.QUARTERLY,
            day_count_convention=DayCountConvention.ACT_360,
            issue_date=self.issue_date,
            maturity_date=self.maturity_date,
            face_value=Decimal('1000000'),
            coupon_type=CouponType.FLOATING,
            floating_index="MIBOR",
            spread_bps=50,  # 50 bps spread
            rate_cap=Decimal('0.08'),  # 8% cap
            rate_floor=Decimal('0.02')  # 2% floor
        )
        
        # Mock index rates
        index_rates = {
            self.issue_date: Decimal('0.05'),  # 5%
            self.issue_date + timedelta(days=90): Decimal('0.06'),  # 6%
            self.issue_date + timedelta(days=180): Decimal('0.04'),  # 4%
        }
        
        flows = self.engine.generate_cash_flows(config, index_rates=index_rates)
        
        # Check floating rate characteristics
        for flow in flows[1:]:  # Skip issue period
            if flow.coupon_amount > 0:
                # Rate should be within cap/floor bounds
                self.assertGreaterEqual(flow.coupon_rate_effective, Decimal('0.02'))
                self.assertLessEqual(flow.coupon_rate_effective, Decimal('0.08'))
    
    def test_amortizing_bond_equal_principal(self):
        """Test amortizing bond with equal principal payments."""
        config = BondCashFlowConfig(
            coupon_rate=Decimal('0.06'),  # 6%
            coupon_frequency=CouponFrequency.SEMI_ANNUAL,
            day_count_convention=DayCountConvention.THIRTY_360,
            issue_date=self.issue_date,
            maturity_date=self.maturity_date,
            face_value=Decimal('1000000'),
            coupon_type=CouponType.FIXED,
            amortization_type=AmortizationType.EQUAL_PRINCIPAL
        )
        
        flows = self.engine.generate_cash_flows(config)
        
        # Check amortization
        total_periods = len(flows) - 1  # Exclude issue period
        expected_principal_per_period = Decimal('1000000') / total_periods
        
        for flow in flows[1:]:  # Skip issue period
            if flow.period_index < len(flows) - 1:  # Not maturity
                self.assertAlmostEqual(
                    float(flow.principal_repayment),
                    float(expected_principal_per_period),
                    places=2
                )
    
    def test_sinking_fund_bond(self):
        """Test bond with sinking fund payments."""
        sinking_schedule = {
            self.issue_date + timedelta(days=365*2): Decimal('200000'),  # 2 years
            self.issue_date + timedelta(days=365*3): Decimal('300000'),  # 3 years
        }
        
        config = BondCashFlowConfig(
            coupon_rate=Decimal('0.05'),
            coupon_frequency=CouponFrequency.ANNUAL,
            day_count_convention=DayCountConvention.ACT_365,
            issue_date=self.issue_date,
            maturity_date=self.maturity_date,
            face_value=Decimal('1000000'),
            coupon_type=CouponType.FIXED,
            sinking_fund=sinking_schedule
        )
        
        flows = self.engine.generate_cash_flows(config)
        
        # Check sinking fund payments
        for flow in flows:
            if flow.payment_date in sinking_schedule:
                expected_sinking = sinking_schedule[flow.payment_date]
                self.assertGreaterEqual(flow.principal_repayment, expected_sinking)
    
    def test_stub_periods(self):
        """Test bonds with stub periods."""
        front_stub = self.issue_date + timedelta(days=90)
        back_stub = self.maturity_date - timedelta(days=60)
        
        config = BondCashFlowConfig(
            coupon_rate=Decimal('0.05'),
            coupon_frequency=CouponFrequency.SEMI_ANNUAL,
            day_count_convention=DayCountConvention.ACT_365,
            issue_date=self.issue_date,
            maturity_date=self.maturity_date,
            face_value=Decimal('1000000'),
            coupon_type=CouponType.FIXED,
            stub_periods={
                "front": front_stub,
                "back": back_stub
            }
        )
        
        flows = self.engine.generate_cash_flows(config)
        
        # Check stub periods are marked
        stub_flows = [f for f in flows if f.is_stub]
        self.assertGreater(len(stub_flows), 0)
    
    def test_business_day_adjustment(self):
        """Test business day adjustment with holiday calendar."""
        # Create a date that falls on a weekend
        weekend_date = self.issue_date
        while weekend_date.weekday() < 5:  # Find next weekend
            weekend_date += timedelta(days=1)
        
        config = BondCashFlowConfig(
            coupon_rate=Decimal('0.05'),
            coupon_frequency=CouponFrequency.MONTHLY,
            day_count_convention=DayCountConvention.ACT_365,
            issue_date=weekend_date,
            maturity_date=weekend_date + timedelta(days=365),
            face_value=Decimal('100000'),
            coupon_type=CouponType.FIXED,
            holiday_calendar=self.holidays,
            roll_convention=RollConvention.FOLLOWING
        )
        
        flows = self.engine.generate_cash_flows(config)
        
        # Check that payment dates are business days
        holiday_calendar = HolidayCalendar(self.holidays)
        for flow in flows:
            self.assertTrue(holiday_calendar.is_business_day(flow.payment_date))
    
    def test_validation_errors(self):
        """Test input validation and error handling."""
        # Invalid issue date
        with self.assertRaises(ValueError):
            config = BondCashFlowConfig(
                coupon_rate=Decimal('0.05'),
                coupon_frequency=CouponFrequency.ANNUAL,
                day_count_convention=DayCountConvention.ACT_365,
                issue_date=self.maturity_date,  # After maturity
                maturity_date=self.issue_date,
                face_value=Decimal('1000000'),
                coupon_type=CouponType.FIXED
            )
            self.engine.generate_cash_flows(config)
        
        # Invalid face value
        with self.assertRaises(ValueError):
            config = BondCashFlowConfig(
                coupon_rate=Decimal('0.05'),
                coupon_frequency=CouponFrequency.ANNUAL,
                day_count_convention=DayCountConvention.ACT_365,
                issue_date=self.issue_date,
                maturity_date=self.maturity_date,
                face_value=Decimal('0'),  # Zero face value
                coupon_type=CouponType.FIXED
            )
            self.engine.generate_cash_flows(config)
        
        # Invalid coupon rate
        with self.assertRaises(ValueError):
            config = BondCashFlowConfig(
                coupon_rate=Decimal('-0.05'),  # Negative rate
                coupon_frequency=CouponFrequency.ANNUAL,
                day_count_convention=DayCountConvention.ACT_365,
                issue_date=self.issue_date,
                maturity_date=self.maturity_date,
                face_value=Decimal('1000000'),
                coupon_type=CouponType.FIXED
            )
            self.engine.generate_cash_flows(config)
    
    def test_floating_rate_validation(self):
        """Test floating rate bond validation."""
        # Missing floating index
        with self.assertRaises(ValueError):
            config = BondCashFlowConfig(
                coupon_rate=Decimal('0.05'),
                coupon_frequency=CouponFrequency.QUARTERLY,
                day_count_convention=DayCountConvention.ACT_360,
                issue_date=self.issue_date,
                maturity_date=self.maturity_date,
                face_value=Decimal('1000000'),
                coupon_type=CouponType.FLOATING,
                # Missing floating_index
                spread_bps=50
            )
            self.engine.generate_cash_flows(config)
        
        # Missing spread
        with self.assertRaises(ValueError):
            config = BondCashFlowConfig(
                coupon_rate=Decimal('0.05'),
                coupon_frequency=CouponFrequency.QUARTERLY,
                day_count_convention=DayCountConvention.ACT_360,
                issue_date=self.issue_date,
                maturity_date=self.maturity_date,
                face_value=Decimal('1000000'),
                coupon_type=CouponType.FLOATING,
                floating_index="MIBOR"
                # Missing spread_bps
            )
            self.engine.generate_cash_flows(config)
    
    def test_cash_flow_validation(self):
        """Test cash flow validation logic."""
        config = BondCashFlowConfig(
            coupon_rate=Decimal('0.05'),
            coupon_frequency=CouponFrequency.SEMI_ANNUAL,
            day_count_convention=DayCountConvention.ACT_365,
            issue_date=self.issue_date,
            maturity_date=self.maturity_date,
            face_value=Decimal('1000000'),
            coupon_type=CouponType.FIXED
        )
        
        flows = self.engine.generate_cash_flows(config)
        
        # Validate generated flows
        self.assertTrue(self.engine.validate_cash_flows(flows, Decimal('1000000')))
        
        # Test with invalid flows
        invalid_flows = flows.copy()
        invalid_flows[0].principal_repayment = Decimal('2000000')  # Exceeds face value
        
        self.assertFalse(self.engine.validate_cash_flows(invalid_flows, Decimal('1000000')))
    
    def test_utility_functions(self):
        """Test utility functions."""
        config = BondCashFlowConfig(
            coupon_rate=Decimal('0.05'),
            coupon_frequency=CouponFrequency.SEMI_ANNUAL,
            day_count_convention=DayCountConvention.ACT_365,
            issue_date=self.issue_date,
            maturity_date=self.maturity_date,
            face_value=Decimal('1000000'),
            coupon_type=CouponType.FIXED
        )
        
        flows = self.engine.generate_cash_flows(config)
        
        # Test next payment finder
        next_payment = self.engine.get_next_payment_date(flows, self.issue_date)
        self.assertIsNotNone(next_payment)
        
        # Test last coupon date
        last_coupon = self.engine.get_last_coupon_date(flows)
        self.assertIsNotNone(last_coupon)
        
        # Test totals
        total_coupons = self.engine.get_total_coupon_payments(flows)
        self.assertGreater(total_coupons, 0)
    
    def test_performance_requirements(self):
        """Test performance requirements (10,000 instruments under 100ms)."""
        start_time = time.time()
        
        # Generate 10,000 simple instruments
        for i in range(10000):
            config = BondCashFlowConfig(
                coupon_rate=Decimal('0.05'),
                coupon_frequency=CouponFrequency.SEMI_ANNUAL,
                day_count_convention=DayCountConvention.ACT_365,
                issue_date=self.issue_date,
                maturity_date=self.maturity_date + timedelta(days=i),
                face_value=Decimal('1000000'),
                coupon_type=CouponType.FIXED
            )
            
            flows = self.engine.generate_cash_flows(config)
            # Basic validation
            self.assertGreater(len(flows), 0)
        
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Should complete under 100ms
        self.assertLess(execution_time, 100, f"Performance test failed: {execution_time:.2f}ms")
    
    def test_day_count_integration(self):
        """Test integration with DayCountCalculator."""
        config = BondCashFlowConfig(
            coupon_rate=Decimal('0.05'),
            coupon_frequency=CouponFrequency.QUARTERLY,
            day_count_convention=DayCountConvention.ACT_ACT,  # Different convention
            issue_date=self.issue_date,
            maturity_date=self.maturity_date,
            face_value=Decimal('1000000'),
            coupon_type=CouponType.FIXED
        )
        
        flows = self.engine.generate_cash_flows(config)
        
        # Check that accrual factors are calculated correctly
        for flow in flows[1:]:  # Skip issue period
            self.assertGreater(flow.accrual_factor, 0)
            self.assertLessEqual(flow.accrual_factor, 1)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very short maturity
        short_maturity = self.issue_date + timedelta(days=30)
        config = BondCashFlowConfig(
            coupon_rate=Decimal('0.05'),
            coupon_frequency=CouponFrequency.ANNUAL,
            day_count_convention=DayCountConvention.ACT_365,
            issue_date=self.issue_date,
            maturity_date=short_maturity,
            face_value=Decimal('1000000'),
            coupon_type=CouponType.FIXED
        )
        
        flows = self.engine.generate_cash_flows(config)
        self.assertGreater(len(flows), 0)
        
        # Very long maturity
        long_maturity = self.issue_date + timedelta(days=365*50)  # 50 years
        config = BondCashFlowConfig(
            coupon_rate=Decimal('0.05'),
            coupon_frequency=CouponFrequency.ANNUAL,
            day_count_convention=DayCountConvention.ACT_365,
            issue_date=self.issue_date,
            maturity_date=long_maturity,
            face_value=Decimal('1000000'),
            coupon_type=CouponType.FIXED
        )
        
        flows = self.engine.generate_cash_flows(config)
        self.assertGreater(len(flows), 0)
        
        # Zero coupon rate
        config = BondCashFlowConfig(
            coupon_rate=Decimal('0'),
            coupon_frequency=CouponFrequency.ANNUAL,
            day_count_convention=DayCountConvention.ACT_365,
            issue_date=self.issue_date,
            maturity_date=self.maturity_date,
            face_value=Decimal('1000000'),
            coupon_type=CouponType.FIXED
        )
        
        flows = self.engine.generate_cash_flows(config)
        for flow in flows:
            self.assertEqual(flow.coupon_amount, 0)


class TestHolidayCalendar(unittest.TestCase):
    """Test cases for HolidayCalendar."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.holidays = [
            date(2024, 1, 1),   # New Year
            date(2024, 1, 26),  # Republic Day
        ]
        self.calendar = HolidayCalendar(self.holidays)
    
    def test_business_day_check(self):
        """Test business day checking."""
        # Monday should be business day
        monday = date(2024, 1, 8)  # Monday
        self.assertTrue(self.calendar.is_business_day(monday))
        
        # Saturday should not be business day
        saturday = date(2024, 1, 6)  # Saturday
        self.assertFalse(self.calendar.is_business_day(saturday))
        
        # Holiday should not be business day
        self.assertFalse(self.calendar.is_business_day(self.holidays[0]))
    
    def test_date_adjustment(self):
        """Test date adjustment with different roll conventions."""
        # Test following convention
        weekend_date = date(2024, 1, 6)  # Saturday
        adjusted = self.calendar.adjust_date(weekend_date, RollConvention.FOLLOWING)
        self.assertEqual(adjusted, date(2024, 1, 8))  # Monday
        
        # Test preceding convention
        adjusted = self.calendar.adjust_date(weekend_date, RollConvention.PRECEDING)
        self.assertEqual(adjusted, date(2024, 1, 5))  # Friday


if __name__ == '__main__':
    unittest.main()
