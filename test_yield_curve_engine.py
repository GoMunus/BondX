#!/usr/bin/env python3
"""
Comprehensive test suite for YieldCurveEngine.

Tests curve construction, interpolation, validation, and edge cases.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from decimal import Decimal
import tempfile
import os

from bondx.mathematics.yield_curves import (
    YieldCurveEngine, YieldCurve, MarketQuote, CurveConstructionConfig,
    CurveType, InterpolationMethod, ExtrapolationMethod, CompoundingConvention
)
from bondx.database.models import DayCountConvention


class TestYieldCurveEngine(unittest.TestCase):
    """Test cases for YieldCurveEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = YieldCurveEngine()
        self.today = date.today()
        
        # Sample market quotes for testing
        self.par_quotes = [
            MarketQuote(
                tenor=0.25,  # 3 months
                quote_type=CurveType.PAR_CURVE,
                quote_value=Decimal('0.045'),  # 4.5%
                day_count=DayCountConvention.ACT_365,
                instrument_id="TBILL_3M"
            ),
            MarketQuote(
                tenor=0.5,   # 6 months
                quote_type=CurveType.PAR_CURVE,
                quote_value=Decimal('0.048'),  # 4.8%
                day_count=DayCountConvention.ACT_365,
                instrument_id="TBILL_6M"
            ),
            MarketQuote(
                tenor=1.0,   # 1 year
                quote_type=CurveType.PAR_CURVE,
                quote_value=Decimal('0.050'),  # 5.0%
                day_count=DayCountConvention.ACT_365,
                instrument_id="G_SEC_1Y"
            ),
            MarketQuote(
                tenor=2.0,   # 2 years
                quote_type=CurveType.PAR_CURVE,
                quote_value=Decimal('0.052'),  # 5.2%
                day_count=DayCountConvention.ACT_365,
                instrument_id="G_SEC_2Y"
            ),
            MarketQuote(
                tenor=5.0,   # 5 years
                quote_type=CurveType.PAR_CURVE,
                quote_value=Decimal('0.055'),  # 5.5%
                day_count=DayCountConvention.ACT_365,
                instrument_id="G_SEC_5Y"
            ),
            MarketQuote(
                tenor=10.0,  # 10 years
                quote_type=CurveType.PAR_CURVE,
                quote_value=Decimal('0.058'),  # 5.8%
                day_count=DayCountConvention.ACT_365,
                instrument_id="G_SEC_10Y"
            )
        ]
        
        self.zero_quotes = [
            MarketQuote(
                tenor=0.25,
                quote_type=CurveType.ZERO_CURVE,
                quote_value=Decimal('0.045'),
                day_count=DayCountConvention.ACT_365,
                instrument_id="ZERO_3M"
            ),
            MarketQuote(
                tenor=1.0,
                quote_type=CurveType.ZERO_CURVE,
                quote_value=Decimal('0.050'),
                day_count=DayCountConvention.ACT_365,
                instrument_id="ZERO_1Y"
            ),
            MarketQuote(
                tenor=5.0,
                quote_type=CurveType.ZERO_CURVE,
                quote_value=Decimal('0.055'),
                day_count=DayCountConvention.ACT_365,
                instrument_id="ZERO_5Y"
            )
        ]
        
        self.discount_quotes = [
            MarketQuote(
                tenor=0.25,
                quote_type=CurveType.DISCOUNT_CURVE,
                quote_value=Decimal('0.989'),  # DF = 0.989
                day_count=DayCountConvention.ACT_365,
                instrument_id="DF_3M"
            ),
            MarketQuote(
                tenor=1.0,
                quote_type=CurveType.DISCOUNT_CURVE,
                quote_value=Decimal('0.952'),  # DF = 0.952
                day_count=DayCountConvention.ACT_365,
                instrument_id="DF_1Y"
            ),
            MarketQuote(
                tenor=5.0,
                quote_type=CurveType.DISCOUNT_CURVE,
                quote_value=Decimal('0.779'),  # DF = 0.779
                day_count=DayCountConvention.ACT_365,
                instrument_id="DF_5Y"
            )
        ]
    
    def test_curve_construction_from_par_yields(self):
        """Test curve construction from par yields."""
        config = CurveConstructionConfig(
            interpolation_method=InterpolationMethod.LINEAR_ON_ZERO,
            extrapolation_method=ExtrapolationMethod.FLAT_FORWARD,
            compounding=CompoundingConvention.SEMI_ANNUAL,
            day_count=DayCountConvention.ACT_365
        )
        
        curve = self.engine.construct_curve(self.par_quotes, config)
        
        # Basic validation
        self.assertEqual(curve.curve_type, CurveType.ZERO_CURVE)
        self.assertEqual(len(curve.tenors), len(self.par_quotes))
        self.assertEqual(len(curve.rates), len(self.par_quotes))
        
        # Check that tenors are sorted
        self.assertTrue(np.all(np.diff(curve.tenors) >= 0))
        
        # Check that rates are reasonable
        for rate in curve.rates:
            self.assertGreater(rate, 0)
            self.assertLess(rate, 1)  # Should be less than 100%
    
    def test_curve_construction_from_zero_rates(self):
        """Test curve construction from zero rates."""
        config = CurveConstructionConfig(
            interpolation_method=InterpolationMethod.LINEAR_ON_ZERO,
            extrapolation_method=ExtrapolationMethod.FLAT_FORWARD,
            compounding=CompoundingConvention.SEMI_ANNUAL,
            day_count=DayCountConvention.ACT_365
        )
        
        curve = self.engine.construct_curve(self.zero_quotes, config)
        
        # Should preserve input rates
        for i, quote in enumerate(self.zero_quotes):
            self.assertAlmostEqual(curve.rates[i], float(quote.quote_value), places=6)
    
    def test_curve_construction_from_discount_factors(self):
        """Test curve construction from discount factors."""
        config = CurveConstructionConfig(
            interpolation_method=InterpolationMethod.LINEAR_ON_ZERO,
            extrapolation_method=ExtrapolationMethod.FLAT_FORWARD,
            compounding=CompoundingConvention.SEMI_ANNUAL,
            day_count=DayCountConvention.ACT_365
        )
        
        curve = self.engine.construct_curve(self.discount_quotes, config)
        
        # Should convert DFs to zero rates
        self.assertEqual(curve.curve_type, CurveType.ZERO_CURVE)
        
        # Check that rates are positive
        for rate in curve.rates:
            self.assertGreater(rate, 0)
    
    def test_mixed_quote_types_validation(self):
        """Test validation for mixed quote types."""
        mixed_quotes = [
            MarketQuote(
                tenor=1.0,
                quote_type=CurveType.PAR_CURVE,
                quote_value=Decimal('0.050'),
                day_count=DayCountConvention.ACT_365,
                instrument_id="MIXED_1"
            ),
            MarketQuote(
                tenor=2.0,
                quote_type=CurveType.ZERO_CURVE,
                quote_value=Decimal('0.052'),
                day_count=DayCountConvention.ACT_365,
                instrument_id="MIXED_2"
            )
        ]
        
        config = CurveConstructionConfig()
        
        with self.assertRaises(ValueError):
            self.engine.construct_curve(mixed_quotes, config)
    
    def test_insufficient_quotes_validation(self):
        """Test validation for insufficient quotes."""
        single_quote = [self.par_quotes[0]]
        config = CurveConstructionConfig()
        
        with self.assertRaises(ValueError):
            self.engine.construct_curve(single_quote, config)
    
    def test_invalid_quote_values_validation(self):
        """Test validation for invalid quote values."""
        invalid_quotes = [
            MarketQuote(
                tenor=1.0,
                quote_type=CurveType.PAR_CURVE,
                quote_value=Decimal('0'),  # Zero rate
                day_count=DayCountConvention.ACT_365,
                instrument_id="INVALID_1"
            ),
            MarketQuote(
                tenor=2.0,
                quote_type=CurveType.PAR_CURVE,
                quote_value=Decimal('-0.05'),  # Negative rate
                day_count=DayCountConvention.ACT_365,
                instrument_id="INVALID_2"
            )
        ]
        
        config = CurveConstructionConfig()
        
        # Should filter out invalid quotes
        curve = self.engine.construct_curve(invalid_quotes, config)
        self.assertEqual(len(curve.tenors), 0)
    
    def test_curve_evaluation_methods(self):
        """Test curve evaluation methods."""
        config = CurveConstructionConfig(
            compounding=CompoundingConvention.SEMI_ANNUAL
        )
        
        curve = self.engine.construct_curve(self.zero_quotes, config)
        
        # Test zero rate interpolation
        test_tenor = 0.75  # Between 0.5 and 1.0
        zero_rate = curve.zero_rate(test_tenor)
        self.assertGreater(zero_rate, 0)
        
        # Test discount factor calculation
        df = curve.discount_factor(test_tenor)
        self.assertGreater(df, 0)
        self.assertLess(df, 1)
        
        # Test forward rate calculation
        forward_rate = curve.forward_rate(0.5, 1.0)
        self.assertGreater(forward_rate, 0)
        
        # Test par yield (simplified)
        par_yield = curve.par_yield(test_tenor)
        self.assertGreater(par_yield, 0)
    
    def test_curve_shift_methods(self):
        """Test curve shift methods."""
        config = CurveConstructionConfig()
        curve = self.engine.construct_curve(self.zero_quotes, config)
        
        # Test parallel shift
        shifted_curve = curve.shift("parallel", 0.01)  # +100 bps
        for i, rate in enumerate(shifted_curve.rates):
            self.assertAlmostEqual(rate, curve.rates[i] + 0.01, places=6)
        
        # Test slope shift
        slope_shifted = curve.shift("slope", 0.01)  # +100 bps per year
        for i, rate in enumerate(slope_shifted.rates):
            expected_rate = curve.rates[i] + 0.01 * curve.tenors[i]
            self.assertAlmostEqual(rate, expected_rate, places=6)
        
        # Test curvature shift
        curvature_shifted = curve.shift("curvature", 0.01)  # +100 bps per year^2
        for i, rate in enumerate(curvature_shifted.rates):
            expected_rate = curve.rates[i] + 0.01 * (curve.tenors[i] ** 2)
            self.assertAlmostEqual(rate, expected_rate, places=6)
        
        # Test invalid shift type
        with self.assertRaises(ValueError):
            curve.shift("invalid", 0.01)
    
    def test_curve_roll_method(self):
        """Test curve roll method."""
        config = CurveConstructionConfig()
        curve = self.engine.construct_curve(self.zero_quotes, config)
        
        # Roll forward by 30 days
        roll_date = self.today + timedelta(days=30)
        rolled_curve = curve.roll(roll_date)
        
        # Check construction date updated
        self.assertEqual(rolled_curve.construction_date, roll_date)
        
        # Check tenors adjusted
        days_diff = 30 / 365.0
        for i, tenor in enumerate(rolled_curve.tenors):
            expected_tenor = curve.tenors[i] - days_diff
            self.assertAlmostEqual(tenor, expected_tenor, places=6)
    
    def test_curve_interpolation(self):
        """Test curve interpolation to new tenors."""
        config = CurveConstructionConfig()
        curve = self.engine.construct_curve(self.zero_quotes, config)
        
        # Interpolate to new tenors
        new_tenors = np.array([0.125, 0.375, 0.625, 0.875])
        interpolated_curve = self.engine.interpolate_curve(curve, new_tenors)
        
        # Check new tenors
        np.testing.assert_array_equal(interpolated_curve.tenors, new_tenors)
        
        # Check rates interpolated
        for i, tenor in enumerate(new_tenors):
            rate = interpolated_curve.rates[i]
            self.assertGreater(rate, 0)
            self.assertLess(rate, 1)
    
    def test_arbitrage_detection(self):
        """Test arbitrage detection."""
        config = CurveConstructionConfig()
        curve = self.engine.construct_curve(self.zero_quotes, config)
        
        # Check for arbitrage warnings
        warnings = self.engine.detect_arbitrage(curve)
        
        # Should not have negative discount factors
        for warning in warnings:
            self.assertNotIn("Negative discount factor", warning)
        
        # Should not have negative forward rates
        for warning in warnings:
            self.assertNotIn("Negative forward rate", warning)
    
    def test_curve_serialization(self):
        """Test curve serialization and deserialization."""
        config = CurveConstructionConfig()
        curve = self.engine.construct_curve(self.zero_quotes, config)
        
        # Convert to dictionary
        curve_dict = curve.to_dict()
        
        # Check required fields
        required_fields = ["curve_type", "tenors", "rates", "construction_date", "config"]
        for field in required_fields:
            self.assertIn(field, curve_dict)
        
        # Reconstruct from dictionary
        reconstructed_curve = YieldCurve.from_dict(curve_dict)
        
        # Check equality
        np.testing.assert_array_equal(curve.tenors, reconstructed_curve.tenors)
        np.testing.assert_array_equal(curve.rates, reconstructed_curve.rates)
        self.assertEqual(curve.curve_type, reconstructed_curve.curve_type)
        self.assertEqual(curve.construction_date, reconstructed_curve.construction_date)
    
    def test_curve_export_import(self):
        """Test curve export and import functionality."""
        config = CurveConstructionConfig()
        curve = self.engine.construct_curve(self.zero_quotes, config)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Export curve
            self.engine.export_curve(curve, temp_file)
            
            # Check file exists
            self.assertTrue(os.path.exists(temp_file))
            
            # Import curve
            imported_curve = self.engine.import_curve(temp_file)
            
            # Check equality
            np.testing.assert_array_equal(curve.tenors, imported_curve.tenors)
            np.testing.assert_array_equal(curve.rates, imported_curve.rates)
            self.assertEqual(curve.curve_type, imported_curve.curve_type)
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_curve_caching(self):
        """Test curve caching functionality."""
        config = CurveConstructionConfig()
        curve_id = "test_curve_001"
        
        # Construct and cache curve
        curve = self.engine.construct_curve(self.zero_quotes, config, curve_id)
        
        # Retrieve from cache
        cached_curve = self.engine.get_cached_curve(curve_id)
        self.assertIsNotNone(cached_curve)
        
        # Check equality
        np.testing.assert_array_equal(curve.tenors, cached_curve.tenors)
        np.testing.assert_array_equal(curve.rates, cached_curve.rates)
        
        # Test non-existent curve
        non_existent = self.engine.get_cached_curve("non_existent")
        self.assertIsNone(non_existent)
        
        # Clear cache
        self.engine.clear_cache()
        cleared_curve = self.engine.get_cached_curve(curve_id)
        self.assertIsNone(cleared_curve)
    
    def test_date_based_tenors(self):
        """Test curves with date-based tenors."""
        date_quotes = [
            MarketQuote(
                tenor=self.today + timedelta(days=90),   # 3 months
                quote_type=CurveType.PAR_CURVE,
                quote_value=Decimal('0.045'),
                day_count=DayCountConvention.ACT_365,
                instrument_id="DATE_3M"
            ),
            MarketQuote(
                tenor=self.today + timedelta(days=365),  # 1 year
                quote_type=CurveType.PAR_CURVE,
                quote_value=Decimal('0.050'),
                day_count=DayCountConvention.ACT_365,
                instrument_id="DATE_1Y"
            )
        ]
        
        config = CurveConstructionConfig()
        curve = self.engine.construct_curve(date_quotes, config)
        
        # Should convert dates to tenors
        self.assertEqual(len(curve.tenors), len(date_quotes))
        
        # Check tenor conversion
        for i, quote in enumerate(date_quotes):
            expected_tenor = (quote.tenor - self.today).days / 365.0
            self.assertAlmostEqual(curve.tenors[i], expected_tenor, places=6)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Single point curve
        single_quote = [self.zero_quotes[0]]
        config = CurveConstructionConfig()
        
        # Should handle single point
        curve = self.engine.construct_curve(single_quote, config)
        self.assertEqual(len(curve.tenors), 1)
        
        # Test evaluation at boundaries
        zero_rate = curve.zero_rate(0.25)  # Exact tenor
        self.assertAlmostEqual(zero_rate, float(single_quote[0].quote_value), places=6)
        
        # Test evaluation outside boundaries
        zero_rate_outside = curve.zero_rate(0.5)  # Beyond tenor
        self.assertAlmostEqual(zero_rate_outside, float(single_quote[0].quote_value), places=6)
    
    def test_compounding_conventions(self):
        """Test different compounding conventions."""
        # Test annual compounding
        config_annual = CurveConstructionConfig(
            compounding=CompoundingConvention.ANNUAL
        )
        curve_annual = self.engine.construct_curve(self.zero_quotes, config_annual)
        
        # Test continuous compounding
        config_continuous = CurveConstructionConfig(
            compounding=CompoundingConvention.CONTINUOUS
        )
        curve_continuous = self.engine.construct_curve(self.zero_quotes, config_continuous)
        
        # Test discount factor calculation
        test_tenor = 1.0
        df_annual = curve_annual.discount_factor(test_tenor)
        df_continuous = curve_continuous.discount_factor(test_tenor)
        
        # Should be different due to different compounding
        self.assertNotEqual(df_annual, df_continuous)
        
        # Both should be valid discount factors
        self.assertGreater(df_annual, 0)
        self.assertLess(df_annual, 1)
        self.assertGreater(df_continuous, 0)
        self.assertLess(df_continuous, 1)
    
    def test_curve_validation(self):
        """Test curve validation logic."""
        # Create invalid curve with overlapping tenors
        invalid_tenors = np.array([1.0, 0.5, 2.0])  # Not sorted
        invalid_rates = np.array([0.05, 0.04, 0.06])
        
        config = CurveConstructionConfig()
        
        # Should raise error during construction
        with self.assertRaises(ValueError):
            YieldCurve(
                curve_type=CurveType.ZERO_CURVE,
                tenors=invalid_tenors,
                rates=invalid_rates,
                construction_date=self.today,
                config=config
            )
    
    def test_performance_requirements(self):
        """Test performance requirements."""
        # Create larger set of quotes
        large_quotes = []
        for i in range(100):
            tenor = 0.1 + i * 0.1  # 0.1 to 10 years
            rate = 0.04 + i * 0.001  # 4% to 5%
            large_quotes.append(MarketQuote(
                tenor=tenor,
                quote_type=CurveType.PAR_CURVE,
                quote_value=Decimal(str(rate)),
                day_count=DayCountConvention.ACT_365,
                instrument_id=f"QUOTE_{i}"
            ))
        
        config = CurveConstructionConfig()
        
        # Should construct large curve efficiently
        import time
        start_time = time.time()
        
        curve = self.engine.construct_curve(large_quotes, config)
        
        end_time = time.time()
        construction_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Should complete under 100ms
        self.assertLess(construction_time, 100, f"Large curve construction took {construction_time:.2f}ms")
        
        # Should have correct number of points
        self.assertEqual(len(curve.tenors), len(large_quotes))


if __name__ == '__main__':
    unittest.main()
